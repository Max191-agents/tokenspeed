# Copyright (c) 2026 LightSeek Foundation
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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

"""GFX950 split FP8 KPool decode scorer with runtime live lengths."""

from tokenspeed_kernel_amd._triton import gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.standard_cache_logits import (
    _load_key_scales,
    _load_key_tile_to_shared,
    _load_query_tile,
    _make_dot_layout,
    _make_key_blocked_layout,
    _make_key_shared_layout,
    _make_mfma_layout,
    _relu_f32,
)

__all__ = ["_dsa_kpool_decode_logits_fp8_folded_kernel"]


@gluon.constexpr_function
def _make_head_reduction_plan(
    linear_layout,
    num_heads: gl.constexpr,
    block_n: gl.constexpr,
    num_chains: gl.constexpr,
):
    """Split the MFMA head axis into parallel register FMA chains."""
    assert (
        num_chains >= 1 and (num_chains & (num_chains - 1)) == 0
    ), "head-reduction chains must be a positive power of two"
    head_bits: gl.constexpr = num_heads.bit_length() - 1
    chain_bits: gl.constexpr = num_chains.bit_length() - 1
    reg_bases: gl.constexpr = [tuple(basis) for basis in linear_layout.reg_bases]
    summed_head_bits: gl.constexpr = []
    folded_head_bits: gl.constexpr = []
    for bit in range(head_bits):
        stride: gl.constexpr = 1 << (head_bits - 1 - bit)
        if (stride, 0) in reg_bases:
            folded_head_bits.append(bit)
        else:
            summed_head_bits.append(bit)
    assert chain_bits <= len(
        folded_head_bits
    ), "head-reduction chain count exceeds register-resident head bits"
    chain_axis_bits: gl.constexpr = folded_head_bits[:chain_bits]
    chain_fold_bits: gl.constexpr = folded_head_bits[chain_bits:]
    fold_depth: gl.constexpr = len(chain_fold_bits)
    head_bit_shape: gl.constexpr = tuple([2] * head_bits + [block_n])
    head_bit_order: gl.constexpr = tuple(
        summed_head_bits + [head_bits] + chain_axis_bits + chain_fold_bits
    )
    folded_shape: gl.constexpr = tuple(
        [1 << len(summed_head_bits), block_n, num_chains] + [2] * fold_depth
    )
    return (
        head_bit_shape,
        head_bit_order,
        folded_shape,
        fold_depth,
        1 << fold_depth,
    )


@gluon.jit
def _split_reduction_leaf(
    value,
    leaf_index: gl.constexpr,
    depth: gl.constexpr,
):
    for bit in gl.static_range(0, depth):
        low, high = value.split()
        if (leaf_index // (2**bit)) % 2 == 0:
            value = low
        else:
            value = high
    return value


@gluon.jit
def _weighted_fma_fold_serial(
    scores,
    weights,
    num_leaves: gl.constexpr,
    depth: gl.constexpr,
):
    score_leaf = _split_reduction_leaf(scores, 0, depth)
    accumulator = score_leaf * _split_reduction_leaf(weights, 0, depth)
    for leaf_index in gl.static_range(1, num_leaves):
        score_leaf = _split_reduction_leaf(scores, leaf_index, depth)
        accumulator = gl.fma(
            score_leaf,
            _split_reduction_leaf(weights, leaf_index, depth),
            accumulator,
        )
    return accumulator


@gluon.jit
def _weighted_head_sum_four_chains(
    head_scores,
    head_weights,
    mfma_layout: gl.constexpr,
    block_n: gl.constexpr,
):
    """Match AITER's four-chain register fold for the 32-head FP8 scorer."""
    num_heads: gl.constexpr = 32
    num_chains: gl.constexpr = 4
    linear_layout: gl.constexpr = gl.to_linear_layout(
        mfma_layout,
        [num_heads, block_n],
    )
    plan: gl.constexpr = _make_head_reduction_plan(
        linear_layout,
        num_heads,
        block_n,
        num_chains,
    )
    head_bit_shape: gl.constexpr = plan[0]
    head_bit_order: gl.constexpr = plan[1]
    folded_shape: gl.constexpr = plan[2]
    fold_depth: gl.constexpr = plan[3]
    folded_count: gl.constexpr = plan[4]

    weights = head_weights[:, None].broadcast_to([num_heads, block_n])
    scores = (
        head_scores.reshape(head_bit_shape)
        .permute(head_bit_order)
        .reshape(folded_shape)
    )
    weights = (
        weights.reshape(head_bit_shape).permute(head_bit_order).reshape(folded_shape)
    )
    scores = _weighted_fma_fold_serial(
        scores,
        weights,
        folded_count,
        fold_depth,
    )
    scores = gl.sum(scores, axis=2)
    scores = gl.sum(scores, axis=0)
    return gl.convert_layout(scores, gl.SliceLayout(0, mfma_layout))


@gluon.jit
def _score_decode_folded_key_tile(
    query,
    weights,
    raw_key,
    key_scales,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    gl.static_assert(NUM_HEADS == 32)
    key = gl.convert_layout(raw_key, dot_b_layout)
    accumulator = gl.zeros([32, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    head_scores = gl.amd.cdna4.mfma_scaled(
        a=query,
        a_scale=None,
        a_format="e4m3",
        b=key,
        b_scale=None,
        b_format="e4m3",
        acc=accumulator,
    )
    scores = _weighted_head_sum_four_chains(
        _relu_f32(head_scores),
        weights,
        mfma_layout,
        BLOCK_N,
    )
    return scores * key_scales


@gluon.jit
def _dsa_kpool_decode_logits_fp8_folded_body(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    causal_lens,
    req_ids,
    block_table,
    logits,
    row_starts,
    row_ends,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    block_table_stride,
    logits_stride,
    max_candidates,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    POOL_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    """Reuse the standard-cache scorer pipeline over one balanced live split."""
    row_id = gl.program_id(0)
    split_id = gl.program_id(1)
    request_id = gl.load(req_ids + row_id).to(gl.int32)
    num_pools = gl.maximum(gl.load(causal_lens + row_id).to(gl.int32), 0) // POOL_SIZE
    live_end = gl.minimum(num_pools, gl.maximum(max_candidates, 0))
    if split_id == 0:
        gl.store(row_starts + row_id, 0)
        gl.store(row_ends + row_id, live_end)

    # Match AITER's per-row split-K balance. The grid supplies SplitKV while
    # live_end supplies each row's current context, so neither becomes a
    # specialization key and shrinking rows do not strand work at low splits.
    num_splits = gl.num_programs(1)
    candidate_blocks = tl.cdiv(live_end, BLOCK_N)
    blocks_per_split = candidate_blocks // num_splits
    residual_blocks = candidate_blocks - blocks_per_split * num_splits
    split_block_start = split_id * blocks_per_split + gl.minimum(
        split_id, residual_blocks
    )
    split_block_count = blocks_per_split + (split_id < residual_blocks).to(gl.int32)
    candidate_start = split_block_start * BLOCK_N
    candidate_end = gl.minimum(
        live_end,
        candidate_start + split_block_count * BLOCK_N,
    )
    if candidate_end <= candidate_start:
        return

    blocked: gl.constexpr = _make_key_blocked_layout(HEAD_DIM, BLOCK_N, NUM_WARPS)
    shared_layout: gl.constexpr = _make_key_shared_layout()
    mfma_layout: gl.constexpr = _make_mfma_layout(BLOCK_N, NUM_WARPS, True)
    dot_a_layout: gl.constexpr = _make_dot_layout(BLOCK_N, NUM_WARPS, True, 0)
    dot_b_layout: gl.constexpr = _make_dot_layout(BLOCK_N, NUM_WARPS, True, 1)
    key_shared = gl.allocate_shared_memory(
        index_k_fp8.type.element_ty,
        [2, HEAD_DIM, BLOCK_N],
        layout=shared_layout,
    )
    q_groups: gl.constexpr = HEAD_DIM // 16
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[64 // q_groups, q_groups],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    query, head_weights = _load_query_tile(
        q,
        weights,
        weights,
        row_id,
        0,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        stride_w_row,
        stride_w_head,
        layout_q,
        mfma_layout,
        dot_a_layout,
        HEAD_DIM,
        False,
    )

    tile_count = tl.cdiv(candidate_end - candidate_start, BLOCK_N)
    _load_key_tile_to_shared(
        index_k_fp8,
        key_shared,
        0,
        candidate_start,
        candidate_end,
        block_table,
        block_table,
        request_id,
        block_table_stride,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        HEAD_DIM,
        BLOCK_N,
        NUM_WARPS,
        False,
        USE_BUFFER_LOAD,
    )
    _load_key_tile_to_shared(
        index_k_fp8,
        key_shared,
        1,
        candidate_start + BLOCK_N,
        candidate_end,
        block_table,
        block_table,
        request_id,
        block_table_stride,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        HEAD_DIM,
        BLOCK_N,
        NUM_WARPS,
        False,
        USE_BUFFER_LOAD,
    )

    output_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    current_buffer: gl.int32 = 0
    for tile_id in tl.range(0, tile_count):
        tile_start = candidate_start + tile_id * BLOCK_N
        key_scales, valid = _load_key_scales(
            index_k_scale,
            tile_start,
            candidate_end,
            block_table,
            block_table,
            request_id,
            block_table_stride,
            output_layout,
            PAGE_SIZE,
            PAGE_STRIDE_BYTES,
            HEAD_DIM,
            BLOCK_N,
            False,
            USE_BUFFER_LOAD,
        )
        if tile_id + 1 < tile_count:
            gl.amd.cdna4.async_copy.wait_group(1)
        else:
            gl.amd.cdna4.async_copy.wait_group(0)
        raw_key = key_shared.index(current_buffer).load(layout=blocked)
        if tile_id + 2 < tile_count:
            _load_key_tile_to_shared(
                index_k_fp8,
                key_shared,
                current_buffer,
                tile_start + 2 * BLOCK_N,
                candidate_end,
                block_table,
                block_table,
                request_id,
                block_table_stride,
                PAGE_SIZE,
                PAGE_STRIDE_BYTES,
                HEAD_DIM,
                BLOCK_N,
                NUM_WARPS,
                False,
                USE_BUFFER_LOAD,
            )
        scores = _score_decode_folded_key_tile(
            query,
            head_weights,
            raw_key,
            key_scales,
            mfma_layout,
            dot_b_layout,
            NUM_HEADS,
            BLOCK_N,
        )
        output_offsets = (
            row_id.to(gl.int64) * logits_stride
            + tile_start.to(gl.int64)
            + output_columns.to(gl.int64)
        )
        if USE_BUFFER_STORE:
            gl.amd.cdna4.buffer_store(
                scores,
                ptr=logits,
                offsets=output_offsets.to(gl.int32),
                mask=valid,
            )
        else:
            gl.store(logits + output_offsets, scores, mask=valid)
        current_buffer = 1 - current_buffer
    gl.amd.cdna4.async_copy.wait_group(0)


@gluon.jit(
    do_not_specialize=[
        "block_table_stride",
        "logits_stride",
        "max_candidates",
    ],
    do_not_specialize_on_alignment=["max_candidates"],
)
def _dsa_kpool_decode_logits_fp8_folded_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    causal_lens,
    req_ids,
    block_table,
    logits,
    row_starts,
    row_ends,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_w_row,
    stride_w_head,
    block_table_stride,
    logits_stride,
    max_candidates,
    PAGE_SIZE: gl.constexpr,
    PAGE_STRIDE_BYTES: gl.constexpr,
    POOL_SIZE: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    """Score graph-static KPool decode rows with runtime split bounds."""
    gl.static_assert(PAGE_SIZE == 16, "decode scorer requires 16-row pages")
    gl.static_assert(POOL_SIZE == 4, "decode scorer requires four-token pools")
    gl.static_assert(NUM_HEADS == 32, "decode scorer requires 32 query heads")
    gl.static_assert(HEAD_DIM == 128, "decode scorer requires head dimension 128")
    gl.static_assert(BLOCK_N == 64, "decode scorer requires 64-key-row tiles")
    gl.static_assert(NUM_WARPS == 2, "decode scorer requires two warps")

    _dsa_kpool_decode_logits_fp8_folded_body(
        q,
        index_k_fp8,
        index_k_scale,
        weights,
        causal_lens,
        req_ids,
        block_table,
        logits,
        row_starts,
        row_ends,
        stride_q_row,
        stride_q_head,
        stride_q_dim,
        stride_w_row,
        stride_w_head,
        block_table_stride,
        logits_stride,
        max_candidates,
        PAGE_SIZE,
        PAGE_STRIDE_BYTES,
        POOL_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_N,
        NUM_WARPS,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )
