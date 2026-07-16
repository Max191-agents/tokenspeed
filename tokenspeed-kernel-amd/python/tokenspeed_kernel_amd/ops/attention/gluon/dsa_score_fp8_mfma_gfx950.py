# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Experimental FP8 MFMA scorers for the fixed GLM DSA indexer on GFX950.

The production prefill candidate accepts the public BF16 query contract and
the packed page/workspace cache.  Short workloads use a uniform fast path when
all query heads fit E4M3 decomposition range and normalize only uncommon
high-range inputs; long workloads amortize a bounded preprocessing launch that
writes two contiguous E4M3 terms plus scaled head weights.  Both paths gather
arbitrary workspace slots directly into an FP8 MFMA operand.  The decomposition
is an intentional numerical approximation of the BF16 contract and must pass
scorer and TopK accuracy gates.

The matched-core entry point intentionally accepts contiguous FP8 queries and
keys.  It excludes BF16 conversion and packed-cache address translation so a
benchmark can compare the same computational core with another FP8 scorer.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = [
    "_dsa_matched_core_fp8_mfma_kernel",
    "_dsa_preprocess_prefill_query_fp8_kernel",
    "_dsa_prefill_logits_fp8_mfma_kernel",
    "_dsa_prefill_logits_fp8_tiled_fused_range_safe_kernel",
    "_dsa_prefill_logits_fp8_tiled_kernel",
    "_dsa_prefill_logits_fp8_tiled_multi_component_kernel",
    "launch_dsa_matched_core_fp8_gfx950",
    "launch_dsa_prefill_logits_fp8_mfma_gfx950",
    "_validate_prefill_query_scratch",
]

_NUM_HEADS = 32
_HEAD_DIM = 128
_PAGE_SIZE = 64
_ROW_BYTES = 132
_BLOCK_N = 32
_NUM_WARPS = 1
_WAVES_PER_EU = 3
_SHORT_PREFILL_WAVES_PER_EU = 3
_BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

_G_NUM_HEADS = gl.constexpr(_NUM_HEADS)
_G_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_G_PAGE_SIZE = gl.constexpr(_PAGE_SIZE)
_G_ROW_BYTES = gl.constexpr(_ROW_BYTES)
_G_BLOCK_N = gl.constexpr(_BLOCK_N)


@gluon.jit
def _load_query_and_weights(
    q,
    weights,
    token,
    q_is_fp8: gl.constexpr,
    load_layout: gl.constexpr,
    mfma_layout: gl.constexpr,
):
    heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, load_layout))
    dims = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, load_layout))
    q_offsets = (
        token * (_G_NUM_HEADS * _G_HEAD_DIM)
        + heads[:, None] * _G_HEAD_DIM
        + dims[None, :]
    ).to(gl.int32)
    q_block = gl.amd.cdna4.buffer_load(ptr=q, offsets=q_offsets)
    if not q_is_fp8:
        q_block = q_block.to(gl.float8e4nv)

    weight_heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))
    weight_block = gl.amd.cdna4.buffer_load(
        ptr=weights,
        offsets=(token * _G_NUM_HEADS + weight_heads).to(gl.int32),
    ).to(gl.float32)
    return q_block, weight_block


@gluon.jit
def _load_packed_k_gather(
    index_k_fp8,
    kv_workspace_slots,
    block_start,
    row_start,
    row_end,
    seq_len_sum,
    load_layout: gl.constexpr,
):
    candidates = block_start + gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(0, load_layout)
    )
    dims = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(1, load_layout))
    valid = (
        (candidates < seq_len_sum) & (candidates >= row_start) & (candidates < row_end)
    )
    slot_u = gl.amd.cdna4.buffer_load(
        ptr=kv_workspace_slots,
        offsets=candidates.to(gl.int32),
        mask=valid,
        other=0,
    ).to(gl.uint64)
    pages = slot_u >> 6
    row_offsets = gl.multiple_of(
        (slot_u << 7) + (pages << 8), _G_HEAD_DIM
    )
    byte_offsets = row_offsets[None, :] + dims[:, None].to(gl.int64)
    byte_offsets = gl.max_contiguous(byte_offsets, [16, 1])
    pointers = gl.max_contiguous(index_k_fp8 + byte_offsets, [16, 1])
    return gl.load(pointers, mask=valid[None, :], other=0.0)


@gluon.jit
def _load_contiguous_k(
    k_fp8,
    block_start,
    seq_len,
    load_layout: gl.constexpr,
):
    candidates = block_start + gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(0, load_layout)
    )
    dims = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(1, load_layout))
    valid = candidates < seq_len
    offsets = candidates[None, :].to(gl.int64) * _G_HEAD_DIM + dims[:, None]
    pointers = gl.max_contiguous(k_fp8 + offsets, [16, 1])
    return gl.load(pointers, mask=valid[None, :], other=0.0)


@gluon.jit
def _load_packed_k_scale(
    index_k_scale,
    kv_workspace_slots,
    candidates,
    valid,
):
    slot_u = gl.amd.cdna4.buffer_load(
        ptr=kv_workspace_slots,
        offsets=candidates.to(gl.int32),
        mask=valid,
        other=0,
    ).to(gl.uint64)
    pages = slot_u >> 6
    scale_offsets = slot_u + ((pages + 1) << 11)
    return gl.load(index_k_scale + scale_offsets, mask=valid, other=0.0).to(gl.float32)


@gluon.jit
def _folded_weighted_head_sum(
    head_scores,
    head_weights,
    mfma_layout: gl.constexpr,
):
    """Reduce 32 signed weighted heads as four short register-local chains."""
    weights = head_weights[:, None].broadcast_to([_G_NUM_HEADS, _G_BLOCK_N])

    # For a 32x32 non-transposed CDNA4 MFMA result, head bits 0, 1, 3,
    # and 4 are register-local while bit 2 spans lanes.  Keep bits 0 and 1
    # as four independent chains, fold bits 3 and 4 with FMAs, then perform
    # only the remaining chain and cross-lane reductions.
    bit_shape: gl.constexpr = (2, 2, 2, 2, 2, _G_BLOCK_N)
    folded_shape: gl.constexpr = (2, _G_BLOCK_N, 4, 2, 2)
    axis_order: gl.constexpr = (2, 5, 0, 1, 3, 4)
    scores = head_scores.reshape(bit_shape).permute(axis_order).reshape(folded_shape)
    weights = weights.reshape(bit_shape).permute(axis_order).reshape(folded_shape)

    scores_low, scores_high = scores.split()
    weights_low, weights_high = weights.split()
    scores_00, scores_01 = scores_low.split()
    scores_10, scores_11 = scores_high.split()
    weights_00, weights_01 = weights_low.split()
    weights_10, weights_11 = weights_high.split()

    folded = scores_00 * weights_00
    folded = gl.fma(scores_01, weights_01, folded)
    folded = gl.fma(scores_10, weights_10, folded)
    folded = gl.fma(scores_11, weights_11, folded)
    folded = gl.sum(folded, axis=2)
    folded = gl.sum(folded, axis=0)
    return gl.convert_layout(folded, gl.SliceLayout(0, mfma_layout))


@gluon.jit
def _score_fp8_tile(
    mfma_q,
    head_weights,
    mfma_k,
    k_scale,
    valid,
    softmax_scale: gl.constexpr,
    mfma_layout: gl.constexpr,
):
    head_scores = gl.zeros(
        [_G_NUM_HEADS, _G_BLOCK_N], dtype=gl.float32, layout=mfma_layout
    )
    head_scores = gl.amd.cdna4.mfma_scaled(
        a=mfma_q,
        a_scale=None,
        a_format="e4m3",
        b=mfma_k,
        b_scale=None,
        b_format="e4m3",
        acc=head_scores,
    )
    head_scores = gl.maximum(head_scores, 0.0)
    scores = _folded_weighted_head_sum(head_scores, head_weights, mfma_layout)
    scores *= k_scale * softmax_scale
    return gl.where(valid, scores, -float("inf"))


@gluon.jit
def _score_fp8_multi_component_tile(
    mfma_q_hi,
    mfma_q_mid,
    head_weights,
    mfma_k,
    k_scale,
    valid,
    softmax_scale: gl.constexpr,
    mfma_layout: gl.constexpr,
):
    head_scores = gl.zeros(
        [_G_NUM_HEADS, _G_BLOCK_N], dtype=gl.float32, layout=mfma_layout
    )
    head_scores = gl.amd.cdna4.mfma_scaled(
        a=mfma_q_hi,
        a_scale=None,
        a_format="e4m3",
        b=mfma_k,
        b_scale=None,
        b_format="e4m3",
        acc=head_scores,
    )
    residual_scores = gl.zeros(
        [_G_NUM_HEADS, _G_BLOCK_N], dtype=gl.float32, layout=mfma_layout
    )
    residual_scores = gl.amd.cdna4.mfma_scaled(
        a=mfma_q_mid,
        a_scale=None,
        a_format="e4m3",
        b=mfma_k,
        b_scale=None,
        b_format="e4m3",
        acc=residual_scores,
    )
    head_scores += residual_scores * 0.03125
    head_scores = gl.maximum(head_scores, 0.0)
    scores = _folded_weighted_head_sum(head_scores, head_weights, mfma_layout)
    scores *= k_scale * softmax_scale
    return gl.where(valid, scores, -float("inf"))


@gluon.jit
def _score_and_store_packed_tile(
    mfma_q,
    mfma_k,
    head_weights,
    index_k_scale,
    kv_workspace_slots,
    logits,
    token,
    block_start,
    row_start,
    row_end,
    seq_len_sum,
    logits_stride: gl.constexpr,
    softmax_scale: gl.constexpr,
    mfma_layout: gl.constexpr,
):
    candidates = block_start + gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    in_bounds = candidates < seq_len_sum
    valid = in_bounds & (candidates >= row_start) & (candidates < row_end)
    k_scale = _load_packed_k_scale(index_k_scale, kv_workspace_slots, candidates, valid)
    scores = _score_fp8_tile(
        mfma_q,
        head_weights,
        mfma_k,
        k_scale,
        valid,
        softmax_scale,
        mfma_layout,
    )
    gl.amd.cdna4.buffer_store(
        scores,
        ptr=logits,
        offsets=(token * logits_stride + candidates).to(gl.int32),
        mask=in_bounds,
    )


@gluon.jit
def _score_and_store_packed_multi_component_tile(
    mfma_q_hi,
    mfma_q_mid,
    mfma_k,
    head_weights,
    index_k_scale,
    kv_workspace_slots,
    logits,
    token,
    block_start,
    row_start,
    row_end,
    seq_len_sum,
    logits_stride: gl.constexpr,
    softmax_scale: gl.constexpr,
    mfma_layout: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    candidates = block_start + gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    in_bounds = candidates < seq_len_sum
    valid = in_bounds & (candidates >= row_start) & (candidates < row_end)
    k_scale = _load_packed_k_scale(index_k_scale, kv_workspace_slots, candidates, valid)
    scores = _score_fp8_multi_component_tile(
        mfma_q_hi,
        mfma_q_mid,
        head_weights,
        mfma_k,
        k_scale,
        valid,
        softmax_scale,
        mfma_layout,
    )
    output_offsets = token.to(gl.int64) * logits_stride + candidates
    if USE_BUFFER_STORE:
        gl.amd.cdna4.buffer_store(
            scores,
            ptr=logits,
            offsets=output_offsets.to(gl.int32),
            mask=in_bounds,
        )
    else:
        gl.store(logits + output_offsets, scores, mask=in_bounds)


@gluon.jit
def _score_and_store_contiguous_tile(
    mfma_q,
    mfma_k,
    head_weights,
    k_scale,
    logits,
    token,
    block_start,
    capacity,
    seq_len,
    logits_stride: gl.constexpr,
    softmax_scale: gl.constexpr,
    mfma_layout: gl.constexpr,
):
    candidates = block_start + gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    in_bounds = candidates < capacity
    valid = candidates < seq_len
    tile_scale = gl.load(k_scale + candidates.to(gl.int64), mask=valid, other=0.0).to(
        gl.float32
    )
    scores = _score_fp8_tile(
        mfma_q,
        head_weights,
        mfma_k,
        tile_scale,
        valid,
        softmax_scale,
        mfma_layout,
    )
    gl.amd.cdna4.buffer_store(
        scores,
        ptr=logits,
        offsets=(token * logits_stride + candidates).to(gl.int32),
        mask=in_bounds,
    )


@gluon.jit
def _dsa_prefill_logits_fp8_mfma_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: gl.constexpr,
    seq_len_sum: gl.int32,
    softmax_scale: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    k_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    q_block, head_weights = _load_query_and_weights(
        q,
        weights,
        token,
        q_is_fp8=False,
        load_layout=q_load_layout,
        mfma_layout=mfma_layout,
    )
    mfma_q = gl.convert_layout(q_block, dot_q_layout)
    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)

    # The current global-to-shared lowering cannot legalize per-row arbitrary
    # workspace pointers as a 128-bit async copy.  Use 64-bit global pointers
    # here so large packed caches remain addressable.
    for block_start in range(0, seq_len_sum, _G_BLOCK_N):
        k_block = _load_packed_k_gather(
            index_k_fp8,
            kv_workspace_slots,
            block_start,
            row_start,
            row_end,
            seq_len_sum,
            k_load_layout,
        )
        mfma_k = gl.convert_layout(k_block, dot_k_layout)
        _score_and_store_packed_tile(
            mfma_q,
            mfma_k,
            head_weights,
            index_k_scale,
            kv_workspace_slots,
            logits,
            token,
            block_start,
            row_start,
            row_end,
            seq_len_sum,
            logits_stride,
            softmax_scale,
            mfma_layout,
        )


@gluon.jit
def _dsa_prefill_logits_fp8_tiled_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: gl.constexpr,
    seq_len_sum: gl.int32,
    softmax_scale: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    block_start = gl.program_id(1) * _G_BLOCK_N
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    k_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    q_block, head_weights = _load_query_and_weights(
        q,
        weights,
        token,
        q_is_fp8=False,
        load_layout=q_load_layout,
        mfma_layout=mfma_layout,
    )
    mfma_q = gl.convert_layout(q_block, dot_q_layout)
    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    k_block = _load_packed_k_gather(
        index_k_fp8,
        kv_workspace_slots,
        block_start,
        row_start,
        row_end,
        seq_len_sum,
        k_load_layout,
    )
    mfma_k = gl.convert_layout(k_block, dot_k_layout)
    _score_and_store_packed_tile(
        mfma_q,
        mfma_k,
        head_weights,
        index_k_scale,
        kv_workspace_slots,
        logits,
        token,
        block_start,
        row_start,
        row_end,
        seq_len_sum,
        logits_stride,
        softmax_scale,
        mfma_layout,
    )


@gluon.jit
def _dsa_prefill_logits_fp8_tiled_fused_range_safe_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: gl.constexpr,
    seq_len_sum: gl.int32,
    softmax_scale: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    TILES_PER_PROGRAM: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    token = gl.program_id(0)
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    k_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, q_load_layout))
    dims = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    q_offsets = (
        token * (_G_NUM_HEADS * _G_HEAD_DIM)
        + heads[:, None] * _G_HEAD_DIM
        + dims[None, :]
    ).to(gl.int32)
    q_bf16 = gl.amd.cdna4.buffer_load(ptr=q, offsets=q_offsets)
    q_mag_bits = q_bf16.to(gl.uint16, bitcast=True) & 0x7FFF
    q_head_mag_bits = gl.max(q_mag_bits, axis=1, keep_dims=True)
    needs_scaling = (
        gl.max(q_head_mag_bits.reshape([_G_NUM_HEADS]), axis=0) > 0x4380
    )
    weight_heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))
    head_weights = gl.amd.cdna4.buffer_load(
        ptr=weights,
        offsets=(token * _G_NUM_HEADS + weight_heads).to(gl.int32),
    ).to(gl.float32)
    if needs_scaling:
        q_fp32 = q_bf16.to(gl.float32)
        q_amax = (
            q_head_mag_bits.to(gl.uint16).to(gl.bfloat16, bitcast=True).to(gl.float32)
        )
        q_scale = gl.maximum(q_amax * (1.0 / 256.0), 1.0)
        q_normalized = q_fp32 / q_scale
        q_hi = q_normalized.to(gl.float8e4nv)
        q_scaled_residual = q_normalized - q_hi.to(gl.float32)
        q_mid = (q_scaled_residual * 32.0).to(gl.float8e4nv)
        head_scale = gl.convert_layout(
            q_scale.reshape([_G_NUM_HEADS]), gl.SliceLayout(1, mfma_layout)
        )
        head_weights *= head_scale
    else:
        q_fp32 = q_bf16.to(gl.float32)
        q_hi = q_fp32.to(gl.float8e4nv)
        q_residual = q_fp32 - q_hi.to(gl.float32)
        q_mid = (q_residual * 32.0).to(gl.float8e4nv)
    mfma_q_hi = gl.convert_layout(q_hi, dot_q_layout)
    mfma_q_mid = gl.convert_layout(q_mid, dot_q_layout)

    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    first_block = gl.program_id(1) * (TILES_PER_PROGRAM * _G_BLOCK_N)
    for tile_index in gl.static_range(0, TILES_PER_PROGRAM):
        block_start = first_block + tile_index * _G_BLOCK_N
        k_block = _load_packed_k_gather(
            index_k_fp8,
            kv_workspace_slots,
            block_start,
            row_start,
            row_end,
            seq_len_sum,
            k_load_layout,
        )
        mfma_k = gl.convert_layout(k_block, dot_k_layout)
        _score_and_store_packed_multi_component_tile(
            mfma_q_hi,
            mfma_q_mid,
            mfma_k,
            head_weights,
            index_k_scale,
            kv_workspace_slots,
            logits,
            token,
            block_start,
            row_start,
            row_end,
            seq_len_sum,
            logits_stride,
            softmax_scale,
            mfma_layout,
            USE_BUFFER_STORE,
        )


@gluon.jit
def _dsa_preprocess_prefill_query_fp8_kernel(
    q,
    weights,
    query_fp8_scratch,
    scaled_weights_scratch,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    q_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, q_layout))
    dims = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, q_layout))
    q_offsets = (
        token * (_G_NUM_HEADS * _G_HEAD_DIM)
        + heads[:, None] * _G_HEAD_DIM
        + dims[None, :]
    ).to(gl.int32)
    q_fp32 = gl.amd.cdna4.buffer_load(ptr=q, offsets=q_offsets).to(gl.float32)
    q_amax = gl.max(gl.abs(q_fp32), axis=1, keep_dims=True)
    q_scale = gl.maximum(q_amax * (1.0 / 256.0), 1.0)
    q_normalized = q_fp32 / q_scale
    q_hi = q_normalized.to(gl.float8e4nv)
    q_residual = q_normalized - q_hi.to(gl.float32)
    q_mid = (q_residual * 32.0).to(gl.float8e4nv)

    scratch_row_offset: gl.constexpr = 2 * _G_NUM_HEADS * _G_HEAD_DIM
    scratch_mid_offset: gl.constexpr = _G_NUM_HEADS * _G_HEAD_DIM
    scratch_offsets = (
        token * scratch_row_offset + heads[:, None] * _G_HEAD_DIM + dims[None, :]
    ).to(gl.int32)
    gl.amd.cdna4.buffer_store(q_hi, ptr=query_fp8_scratch, offsets=scratch_offsets)
    gl.amd.cdna4.buffer_store(
        q_mid,
        ptr=query_fp8_scratch,
        offsets=scratch_offsets + scratch_mid_offset,
    )

    scale_offsets = (token * _G_NUM_HEADS + heads).to(gl.int32)
    head_weights = gl.amd.cdna4.buffer_load(ptr=weights, offsets=scale_offsets).to(
        gl.float32
    )
    head_scale = gl.convert_layout(
        q_scale.reshape([_G_NUM_HEADS]), gl.SliceLayout(1, q_layout)
    )
    scaled_weights = head_weights * head_scale
    gl.amd.cdna4.buffer_store(
        scaled_weights,
        ptr=scaled_weights_scratch,
        offsets=scale_offsets,
    )


@gluon.jit
def _dsa_prefill_logits_fp8_tiled_multi_component_kernel(
    query_fp8_scratch,
    scaled_weights_scratch,
    index_k_fp8,
    index_k_scale,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: gl.constexpr,
    seq_len_sum: gl.int32,
    softmax_scale: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    TILES_PER_PROGRAM: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    token = gl.program_id(0)
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    k_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, q_load_layout))
    dims = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    scratch_row_offset: gl.constexpr = 2 * _G_NUM_HEADS * _G_HEAD_DIM
    scratch_mid_offset: gl.constexpr = _G_NUM_HEADS * _G_HEAD_DIM
    q_offsets = (
        token * scratch_row_offset + heads[:, None] * _G_HEAD_DIM + dims[None, :]
    ).to(gl.int32)
    q_hi = gl.amd.cdna4.buffer_load(ptr=query_fp8_scratch, offsets=q_offsets)
    q_mid = gl.amd.cdna4.buffer_load(
        ptr=query_fp8_scratch, offsets=q_offsets + scratch_mid_offset
    )
    mfma_q_hi = gl.convert_layout(q_hi, dot_q_layout)
    mfma_q_mid = gl.convert_layout(q_mid, dot_q_layout)

    weight_heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))
    head_weights = gl.amd.cdna4.buffer_load(
        ptr=scaled_weights_scratch,
        offsets=(token * _G_NUM_HEADS + weight_heads).to(gl.int32),
    ).to(gl.float32)
    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    first_block = gl.program_id(1) * (TILES_PER_PROGRAM * _G_BLOCK_N)
    for tile_index in gl.static_range(0, TILES_PER_PROGRAM):
        block_start = first_block + tile_index * _G_BLOCK_N
        k_block = _load_packed_k_gather(
            index_k_fp8,
            kv_workspace_slots,
            block_start,
            row_start,
            row_end,
            seq_len_sum,
            k_load_layout,
        )
        mfma_k = gl.convert_layout(k_block, dot_k_layout)
        _score_and_store_packed_multi_component_tile(
            mfma_q_hi,
            mfma_q_mid,
            mfma_k,
            head_weights,
            index_k_scale,
            kv_workspace_slots,
            logits,
            token,
            block_start,
            row_start,
            row_end,
            seq_len_sum,
            logits_stride,
            softmax_scale,
            mfma_layout,
            USE_BUFFER_STORE,
        )


@gluon.jit
def _dsa_matched_core_fp8_mfma_kernel(
    q_fp8,
    k_fp8,
    k_scale,
    weights,
    logits,
    logits_stride: gl.constexpr,
    capacity: gl.int32,
    seq_len: gl.int32,
    softmax_scale: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    k_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    q_block, head_weights = _load_query_and_weights(
        q_fp8,
        weights,
        token,
        q_is_fp8=True,
        load_layout=q_load_layout,
        mfma_layout=mfma_layout,
    )
    mfma_q = gl.convert_layout(q_block, dot_q_layout)
    for block_start in range(0, capacity, _G_BLOCK_N):
        k_block = _load_contiguous_k(k_fp8, block_start, seq_len, k_load_layout)
        mfma_k = gl.convert_layout(k_block, dot_k_layout)
        _score_and_store_contiguous_tile(
            mfma_q,
            mfma_k,
            head_weights,
            k_scale,
            logits,
            token,
            block_start,
            capacity,
            seq_len,
            logits_stride,
            softmax_scale,
            mfma_layout,
        )


def _validate_fp8_core_inputs(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    if q_fp8.dtype != torch.float8_e4m3fn or q_fp8.dim() != 3:
        raise ValueError("q_fp8 must be contiguous E4M3 [tokens, 32, 128]")
    if tuple(q_fp8.shape[1:]) != (_NUM_HEADS, _HEAD_DIM):
        raise ValueError(f"q_fp8 must be [tokens, 32, 128], got {tuple(q_fp8.shape)}")
    if (
        k_fp8.dtype != torch.float8_e4m3fn
        or k_fp8.dim() != 2
        or k_fp8.shape[1] != _HEAD_DIM
    ):
        raise ValueError("k_fp8 must be contiguous E4M3 [capacity, 128]")
    if k_scale.dtype != torch.float32 or k_scale.shape != k_fp8.shape[:1]:
        raise ValueError(f"k_scale must be FP32 {tuple(k_fp8.shape[:1])}")
    if weights.dtype != torch.float32 or weights.shape != q_fp8.shape[:2]:
        raise ValueError(f"weights must be FP32 {tuple(q_fp8.shape[:2])}")
    if logits.dtype != torch.float32 or logits.shape != (
        q_fp8.shape[0],
        k_fp8.shape[0],
    ):
        raise ValueError(
            "logits must be preallocated FP32 " f"{(q_fp8.shape[0], k_fp8.shape[0])}"
        )
    tensors = (q_fp8, k_fp8, k_scale, weights, logits)
    if any(t.device != q_fp8.device for t in tensors):
        raise ValueError("matched-core tensors must share a device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("matched-core tensors must be contiguous")


def launch_dsa_matched_core_fp8_gfx950(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
    *,
    seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Score contiguous FP8 Q/K data without production address translation."""
    _validate_fp8_core_inputs(q_fp8, k_fp8, k_scale, weights, logits)
    capacity = int(k_fp8.shape[0])
    seq_len = int(seq_len)
    if seq_len < 0 or seq_len > capacity:
        raise ValueError(f"seq_len must be in [0, {capacity}], got {seq_len}")
    if q_fp8.shape[0] == 0 or capacity == 0:
        return logits

    _dsa_matched_core_fp8_mfma_kernel[(q_fp8.shape[0],)](
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        logits,
        logits.stride(0),
        capacity,
        seq_len,
        softmax_scale=float(softmax_scale),
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVES_PER_EU,
    )
    return logits


def _validate_prefill_query_scratch(
    q: torch.Tensor,
    query_fp8_scratch: torch.Tensor,
    scaled_weights_scratch: torch.Tensor,
) -> None:
    expected_query_shape = (q.shape[0], 2, _NUM_HEADS, _HEAD_DIM)
    if (
        query_fp8_scratch.dtype != torch.float8_e4m3fn
        or query_fp8_scratch.shape != expected_query_shape
    ):
        raise ValueError(
            "query_fp8_scratch must be preallocated E4M3 " f"{expected_query_shape}"
        )
    # Scratch buffer operations use 32-bit offsets. GLM prefill chunks are
    # several orders of magnitude below this architectural bound.
    if query_fp8_scratch.numel() >= _BUFFER_LIMIT_BYTES:
        raise ValueError("query_fp8_scratch must be smaller than 2 GiB")
    expected_weight_shape = (q.shape[0], _NUM_HEADS)
    if (
        scaled_weights_scratch.dtype != torch.float32
        or scaled_weights_scratch.shape != expected_weight_shape
    ):
        raise ValueError(
            "scaled_weights_scratch must be preallocated FP32 "
            f"{expected_weight_shape}"
        )
    scratch = (query_fp8_scratch, scaled_weights_scratch)
    if any(t.device != q.device for t in scratch):
        raise ValueError("prefill query scratch must share the query device")
    if any(not t.is_contiguous() for t in scratch):
        raise ValueError("prefill query scratch must be contiguous")


def launch_dsa_prefill_logits_fp8_mfma_gfx950(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    logits: torch.Tensor,
    query_fp8_scratch: torch.Tensor,
    scaled_weights_scratch: torch.Tensor,
    *,
    softmax_scale: float,
    tiles_per_program: int = 1,
) -> torch.Tensor:
    """Score packed GLM prefill inputs using caller-owned query scratch."""
    if q.dtype != torch.bfloat16 or q.dim() != 3:
        raise ValueError("q must be contiguous BF16 [tokens, 32, 128]")
    if tuple(q.shape[1:]) != (_NUM_HEADS, _HEAD_DIM):
        raise ValueError(f"q must be [tokens, 32, 128], got {tuple(q.shape)}")
    if weights.dtype != torch.float32 or weights.shape != q.shape[:2]:
        raise ValueError(f"weights must be FP32 {tuple(q.shape[:2])}")
    _validate_prefill_query_scratch(q, query_fp8_scratch, scaled_weights_scratch)
    if (
        index_k_cache.dtype != torch.uint8
        or index_k_cache.dim() != 2
        or index_k_cache.shape[1] != _ROW_BYTES
        or index_k_cache.shape[0] % _PAGE_SIZE != 0
    ):
        raise ValueError("index_k_cache must be packed uint8 [page-aligned slots, 132]")
    if kv_workspace_slots.dtype != torch.int64 or kv_workspace_slots.dim() != 1:
        raise ValueError("kv_workspace_slots must be contiguous int64 [candidates]")
    expected_rows = (q.shape[0],)
    if (
        row_starts.dtype != torch.int32
        or row_ends.dtype != torch.int32
        or row_starts.shape != expected_rows
        or row_ends.shape != expected_rows
    ):
        raise ValueError("row_starts and row_ends must be int32 [tokens]")
    seq_len_sum = int(kv_workspace_slots.numel())
    if logits.dtype != torch.float32 or logits.shape != (q.shape[0], seq_len_sum):
        raise ValueError(
            f"logits must be preallocated FP32 {(q.shape[0], seq_len_sum)}"
        )
    tensors = (
        q,
        index_k_cache,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        logits,
    )
    if any(t.device != q.device for t in tensors):
        raise ValueError("prefill tensors must share a device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("prefill tensors must be contiguous")
    if q.shape[0] == 0 or seq_len_sum == 0:
        return logits

    tiles_per_program = int(tiles_per_program)
    if tiles_per_program not in (1, 2, 4):
        raise ValueError("tiles_per_program must be 1, 2, or 4")
    grid = (
        q.shape[0],
        (seq_len_sum + tiles_per_program * _BLOCK_N - 1)
        // (tiles_per_program * _BLOCK_N),
    )
    use_buffer_store = logits.numel() * logits.element_size() < _BUFFER_LIMIT_BYTES
    if seq_len_sum <= 2048:
        _dsa_prefill_logits_fp8_tiled_fused_range_safe_kernel[grid](
            q,
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights,
            kv_workspace_slots,
            row_starts,
            row_ends,
            logits,
            logits.stride(0),
            seq_len_sum,
            softmax_scale=float(softmax_scale),
            NUM_WARPS=_NUM_WARPS,
            TILES_PER_PROGRAM=tiles_per_program,
            USE_BUFFER_STORE=use_buffer_store,
            num_warps=_NUM_WARPS,
            waves_per_eu=_SHORT_PREFILL_WAVES_PER_EU,
        )
        return logits

    _dsa_preprocess_prefill_query_fp8_kernel[(q.shape[0],)](
        q,
        weights,
        query_fp8_scratch,
        scaled_weights_scratch,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
        waves_per_eu=_SHORT_PREFILL_WAVES_PER_EU,
    )
    launch_args = (
        query_fp8_scratch,
        scaled_weights_scratch,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        kv_workspace_slots,
        row_starts,
        row_ends,
        logits,
        logits.stride(0),
        seq_len_sum,
    )
    _dsa_prefill_logits_fp8_tiled_multi_component_kernel[grid](
        *launch_args,
        softmax_scale=float(softmax_scale),
        NUM_WARPS=_NUM_WARPS,
        TILES_PER_PROGRAM=tiles_per_program,
        USE_BUFFER_STORE=use_buffer_store,
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVES_PER_EU,
    )
    return logits
