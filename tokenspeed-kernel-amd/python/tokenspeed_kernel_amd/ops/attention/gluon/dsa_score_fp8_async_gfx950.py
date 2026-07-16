# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

"""Async-pipelined contiguous FP8 DSA scoring core for AMD GFX950.

This is an isolated matched-core experiment. It consumes the same contiguous
FP8 query/key bytes, FP32 row scales, signed FP32 head weights, and caller-owned
logits as ``launch_dsa_matched_core_fp8_gfx950``. Production packed-cache
translation and BF16-to-FP8 conversion deliberately remain outside this module.

The padded async loader, double-buffered schedule, and folded weighted-head
reduction are adapted from AITER's MIT-licensed GFX950 ``fp8_mqa_logits.py``.
The kernel adds in-kernel rectangular masking for the caller's full logits
capacity instead of relying on a separate fill operation.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, tl

__all__ = [
    "_dsa_matched_core_fp8_async_kernel",
    "launch_dsa_matched_core_fp8_async_gfx950",
]

_NUM_HEADS = 32
_HEAD_DIM = 128
_BLOCK_KV = 32
_NUM_WARPS = 1
_NUM_BUFFERS = 2
_NUM_CHAINS = 4
_WAVES_PER_EU = 3
_BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

_G_NUM_HEADS = gl.constexpr(_NUM_HEADS)
_G_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_G_BLOCK_KV = gl.constexpr(_BLOCK_KV)
_G_NUM_BUFFERS = gl.constexpr(_NUM_BUFFERS)
_G_NUM_CHAINS = gl.constexpr(_NUM_CHAINS)


@gluon.jit
def _relu_f32(value):
    return gl.maximum(value, 0.0, propagate_nan=tl.PropagateNan.ALL)


@gluon.constexpr_function
def _offset_bases_to_blocked(offset_bases, contiguity, num_warps, warp_size, shape):
    """Build the distributed layout expected by coalesced async-copy."""
    rank = len(shape)
    log2_contiguity = contiguity.bit_length() - 1
    log2_num_warps = num_warps.bit_length() - 1
    log2_warp_size = warp_size.bit_length() - 1

    index = 0
    reg_bases = offset_bases[index : index + log2_contiguity]
    index += log2_contiguity
    lane_bases = offset_bases[index : index + log2_warp_size]
    index += log2_warp_size
    warp_bases = offset_bases[index : index + log2_num_warps]
    index += log2_num_warps
    warp_bases += [[0] * rank] * (log2_num_warps - len(warp_bases))
    reg_bases += offset_bases[index:]

    return gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=lane_bases,
        warp_bases=warp_bases,
        block_bases=[],
        shape=shape,
    )


@gluon.constexpr_function
def _make_k_load_layouts(head_dim, block_kv, num_warps, warp_size):
    """Return matching global-register and padded shared-memory K layouts."""
    contiguity = 16
    log2_head_dim = head_dim.bit_length() - 1
    log2_block_kv = block_kv.bit_length() - 1
    log2_contiguity = contiguity.bit_length() - 1
    head_lane = log2_head_dim - log2_contiguity

    offset_bases = [[1 << bit, 0] for bit in range(log2_head_dim)] + [
        [0, 1 << ((bit + head_lane) % log2_block_kv)] for bit in range(log2_block_kv)
    ]
    shared = gl.PaddedSharedLayout(
        interval_padding_pairs=[[1024, 16]],
        offset_bases=offset_bases,
        cga_layout=[],
        shape=[head_dim, block_kv],
    )
    blocked = _offset_bases_to_blocked(
        offset_bases,
        contiguity,
        num_warps,
        warp_size,
        [head_dim, block_kv],
    )
    return blocked, shared


@aggregate
class _AsyncKLoaderConfig:
    block_kv: gl.constexpr
    head_dim: gl.constexpr
    num_buffers: gl.constexpr
    blocked: gl.constexpr
    shared: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        block_kv,
        head_dim,
        num_warps,
        warp_size,
        num_buffers,
    ):
        blocked, shared = _make_k_load_layouts(head_dim, block_kv, num_warps, warp_size)
        self.block_kv = gl.constexpr(block_kv)
        self.head_dim = gl.constexpr(head_dim)
        self.num_buffers = gl.constexpr(num_buffers)
        self.blocked = gl.constexpr(blocked)
        self.shared = gl.constexpr(shared)


@aggregate
class _AsyncKLoader:
    """Double-buffered contiguous K loader with shared shape [D, N]."""

    config: _AsyncKLoaderConfig
    k_ptr: gl.tensor
    shared: gl.shared_memory_descriptor
    base_offset: gl.tensor
    seq_len: gl.tensor

    @gluon.constexpr_function
    def __init__(self, config, k_ptr, shared, base_offset, seq_len):
        self.config = config
        self.k_ptr = k_ptr
        self.shared = shared
        self.base_offset = base_offset
        self.seq_len = seq_len

    @gluon.jit
    def initialize(
        k_ptr,
        seq_len,
        block_kv: gl.constexpr,
        head_dim: gl.constexpr,
        num_warps: gl.constexpr,
        warp_size: gl.constexpr,
        num_buffers: gl.constexpr,
    ):
        config = _AsyncKLoaderConfig(
            block_kv,
            head_dim,
            num_warps,
            warp_size,
            num_buffers,
        )
        shared = gl.allocate_shared_memory(
            k_ptr.type.element_ty,
            [config.num_buffers, config.head_dim, config.block_kv],
            layout=config.shared,
        )
        dims = gl.arange(0, config.head_dim, layout=gl.SliceLayout(1, config.blocked))[
            :, None
        ]
        candidates = gl.arange(
            0, config.block_kv, layout=gl.SliceLayout(0, config.blocked)
        )[None, :]
        base_offset = dims + candidates * config.head_dim
        return _AsyncKLoader(config, k_ptr, shared, base_offset, seq_len)

    @gluon.jit
    def load_to_shared(
        self,
        row_offset,
        buffer_id,
        use_buffer_load: gl.constexpr,
        masked: gl.constexpr,
    ):
        if masked:
            candidates = gl.arange(
                0,
                self.config.block_kv,
                layout=gl.SliceLayout(0, self.config.blocked),
            )[None, :]
            mask = candidates < (self.seq_len - row_offset)
        else:
            mask = None

        row_byte_offset = row_offset.to(gl.int64) * self.config.head_dim
        if use_buffer_load:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                self.shared.index(buffer_id),
                self.k_ptr + row_byte_offset,
                self.base_offset,
                mask=mask,
            )
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                self.shared.index(buffer_id),
                self.k_ptr + row_byte_offset + self.base_offset,
                mask=mask,
            )
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def load_from_shared(self, wait_count, target_layout, buffer_id):
        gl.amd.cdna4.async_copy.wait_group(wait_count)
        return self.shared.index(buffer_id).load(layout=target_layout)

    @gluon.jit
    def wait(self, wait_count):
        gl.amd.cdna4.async_copy.wait_group(wait_count)


@gluon.constexpr_function
def _make_head_reduction_plan(linear_layout, num_heads, block_kv, num_chains):
    """Split register-local head bits between FMA folds and parallel chains."""
    assert num_chains >= 1 and (num_chains & (num_chains - 1)) == 0
    head_bits = num_heads.bit_length() - 1
    chain_bits = num_chains.bit_length() - 1
    reg_bases = [tuple(base) for base in linear_layout.reg_bases]
    summed_head_bits = []
    folded_head_bits = []
    for bit in range(head_bits):
        stride = 1 << (head_bits - 1 - bit)
        if (stride, 0) in reg_bases:
            folded_head_bits.append(bit)
        else:
            summed_head_bits.append(bit)

    assert chain_bits <= len(folded_head_bits)
    chain_axis_bits = folded_head_bits[:chain_bits]
    chain_fold_bits = folded_head_bits[chain_bits:]
    fold_depth = len(chain_fold_bits)
    head_bit_shape = tuple([2] * head_bits + [block_kv])
    head_bit_order = tuple(
        summed_head_bits + [head_bits] + chain_axis_bits + chain_fold_bits
    )
    folded_shape = tuple(
        [1 << len(summed_head_bits), block_kv, num_chains] + [2] * fold_depth
    )
    return (
        head_bit_shape,
        head_bit_order,
        folded_shape,
        fold_depth,
        1 << fold_depth,
    )


@gluon.jit
def _split_leaf(value, index: gl.constexpr, depth: gl.constexpr):
    for bit in gl.static_range(0, depth):
        low, high = value.split()
        if (index // (2**bit)) % 2 == 0:
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
    score_leaf = _split_leaf(scores, 0, depth)
    accumulator = score_leaf * _split_leaf(weights, 0, depth)
    for index in gl.static_range(1, num_leaves):
        score_leaf = _split_leaf(scores, index, depth)
        accumulator = gl.fma(
            score_leaf,
            _split_leaf(weights, index, depth),
            accumulator,
        )
    return accumulator


@gluon.jit
def _weighted_head_sum(
    scores,
    head_weights,
    num_heads: gl.constexpr,
    block_kv: gl.constexpr,
    mfma_layout: gl.constexpr,
    num_chains: gl.constexpr,
):
    linear_layout: gl.constexpr = gl.to_linear_layout(
        mfma_layout, [num_heads, block_kv]
    )
    plan: gl.constexpr = _make_head_reduction_plan(
        linear_layout, num_heads, block_kv, num_chains
    )
    head_bit_shape: gl.constexpr = plan[0]
    head_bit_order: gl.constexpr = plan[1]
    folded_shape: gl.constexpr = plan[2]
    fold_depth: gl.constexpr = plan[3]
    folded_count: gl.constexpr = plan[4]

    weights = head_weights.broadcast_to([num_heads, block_kv])
    scores = (
        scores.reshape(head_bit_shape).permute(head_bit_order).reshape(folded_shape)
    )
    weights = (
        weights.reshape(head_bit_shape).permute(head_bit_order).reshape(folded_shape)
    )
    scores = _weighted_fma_fold_serial(scores, weights, folded_count, fold_depth)
    scores = gl.sum(scores, axis=2)
    scores = gl.sum(scores, axis=0)
    return gl.convert_layout(scores, gl.SliceLayout(0, mfma_layout))


@gluon.jit
def _load_scales(
    scales,
    offset,
    relative_end,
    block_kv: gl.constexpr,
    mfma_layout: gl.constexpr,
    use_buffer_load: gl.constexpr,
    masked: gl.constexpr,
):
    offsets = gl.arange(0, block_kv, layout=gl.SliceLayout(0, mfma_layout))
    if masked:
        mask = offsets < (relative_end - offset)
    else:
        mask = None
    if use_buffer_load:
        if masked:
            return gl.amd.cdna4.buffer_load(
                ptr=scales + offset,
                offsets=offsets,
                mask=mask,
                other=0.0,
            ).to(gl.float32)
        return gl.amd.cdna4.buffer_load(
            ptr=scales + offset,
            offsets=offsets,
        ).to(gl.float32)
    if masked:
        return gl.load(
            scales + offset.to(gl.int64) + offsets,
            mask=mask,
            other=0.0,
        ).to(gl.float32)
    return gl.load(scales + offset.to(gl.int64) + offsets).to(gl.float32)


@gluon.jit
def _store_logits(
    logits,
    offsets,
    scores,
    use_buffer_store: gl.constexpr,
    mask=None,
):
    if use_buffer_store:
        gl.amd.cdna4.buffer_store(
            scores,
            ptr=logits,
            offsets=offsets,
            mask=mask,
        )
    else:
        gl.store(logits + offsets.to(gl.int64), scores, mask=mask)


@gluon.jit
def _score_tile(
    mfma_q,
    mfma_k,
    head_weights,
    scales,
    softmax_scale: gl.constexpr,
    num_heads: gl.constexpr,
    block_kv: gl.constexpr,
    num_chains: gl.constexpr,
    mfma_layout: gl.constexpr,
):
    accumulator = gl.zeros([num_heads, block_kv], dtype=gl.float32, layout=mfma_layout)
    scores = gl.amd.cdna4.mfma_scaled(
        a=mfma_q,
        a_scale=None,
        a_format="e4m3",
        b=mfma_k,
        b_scale=None,
        b_format="e4m3",
        acc=accumulator,
    )
    scores = _relu_f32(scores)
    scores = _weighted_head_sum(
        scores,
        head_weights,
        num_heads,
        block_kv,
        mfma_layout,
        num_chains,
    )
    return scores * scales * softmax_scale


@gluon.jit
def _score_contiguous_double_buffered(
    loader,
    mfma_q,
    head_weights,
    scales,
    logits,
    seq_len,
    num_full_tiles,
    softmax_scale: gl.constexpr,
    num_heads: gl.constexpr,
    block_kv: gl.constexpr,
    num_chains: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_k_layout: gl.constexpr,
    use_buffer_load: gl.constexpr,
    use_buffer_store: gl.constexpr,
):
    store_offsets = gl.arange(0, block_kv, layout=gl.SliceLayout(0, mfma_layout))
    scale_offset: gl.int32 = 0
    logits_offset: gl.int32 = 0
    current_buffer: gl.int32 = 0

    first_row = seq_len - seq_len
    loader.load_to_shared(
        first_row,
        buffer_id=0,
        use_buffer_load=use_buffer_load,
        masked=True,
    )
    loader.load_to_shared(
        first_row + block_kv,
        buffer_id=1,
        use_buffer_load=use_buffer_load,
        masked=True,
    )

    for tile_index in tl.range(0, num_full_tiles - 2):
        tile_scales = _load_scales(
            scales,
            scale_offset,
            seq_len,
            block_kv,
            mfma_layout,
            use_buffer_load,
            masked=False,
        )
        mfma_k = loader.load_from_shared(
            wait_count=1,
            target_layout=dot_k_layout,
            buffer_id=current_buffer,
        )
        loader.load_to_shared(
            (tile_index + 2) * block_kv,
            buffer_id=current_buffer,
            use_buffer_load=use_buffer_load,
            masked=False,
        )
        tile_scores = _score_tile(
            mfma_q,
            mfma_k,
            head_weights,
            tile_scales,
            softmax_scale,
            num_heads,
            block_kv,
            num_chains,
            mfma_layout,
        )
        _store_logits(
            logits,
            logits_offset + store_offsets,
            tile_scores,
            use_buffer_store,
        )
        scale_offset += block_kv
        logits_offset += block_kv
        current_buffer = 1 - current_buffer

    if num_full_tiles > 1:
        tile_scales = _load_scales(
            scales,
            scale_offset,
            seq_len,
            block_kv,
            mfma_layout,
            use_buffer_load,
            masked=False,
        )
        mfma_k = loader.load_from_shared(
            wait_count=1,
            target_layout=dot_k_layout,
            buffer_id=current_buffer,
        )
        loader.load_to_shared(
            num_full_tiles * block_kv,
            buffer_id=current_buffer,
            use_buffer_load=use_buffer_load,
            masked=True,
        )
        tile_scores = _score_tile(
            mfma_q,
            mfma_k,
            head_weights,
            tile_scales,
            softmax_scale,
            num_heads,
            block_kv,
            num_chains,
            mfma_layout,
        )
        _store_logits(
            logits,
            logits_offset + store_offsets,
            tile_scores,
            use_buffer_store,
        )
        scale_offset += block_kv
        logits_offset += block_kv
        current_buffer = 1 - current_buffer

    tile_scales = _load_scales(
        scales,
        scale_offset,
        seq_len,
        block_kv,
        mfma_layout,
        use_buffer_load,
        masked=True,
    )
    mfma_k = loader.load_from_shared(
        wait_count=1,
        target_layout=dot_k_layout,
        buffer_id=current_buffer,
    )
    tile_scores = _score_tile(
        mfma_q,
        mfma_k,
        head_weights,
        tile_scales,
        softmax_scale,
        num_heads,
        block_kv,
        num_chains,
        mfma_layout,
    )
    valid = store_offsets < (seq_len - logits_offset)
    _store_logits(
        logits,
        logits_offset + store_offsets,
        tile_scores,
        use_buffer_store,
        mask=valid,
    )
    scale_offset += block_kv
    logits_offset += block_kv
    current_buffer = 1 - current_buffer

    tile_scales = _load_scales(
        scales,
        scale_offset,
        seq_len,
        block_kv,
        mfma_layout,
        use_buffer_load,
        masked=True,
    )
    mfma_k = loader.load_from_shared(
        wait_count=0,
        target_layout=dot_k_layout,
        buffer_id=current_buffer,
    )
    tile_scores = _score_tile(
        mfma_q,
        mfma_k,
        head_weights,
        tile_scales,
        softmax_scale,
        num_heads,
        block_kv,
        num_chains,
        mfma_layout,
    )
    valid = store_offsets < (seq_len - logits_offset)
    _store_logits(
        logits,
        logits_offset + store_offsets,
        tile_scores,
        use_buffer_store,
        mask=valid,
    )


@gluon.jit
def _fill_rectangular_tail(
    logits,
    seq_len,
    capacity,
    block_kv: gl.constexpr,
    mfma_layout: gl.constexpr,
    use_buffer_store: gl.constexpr,
):
    offsets = gl.arange(0, block_kv, layout=gl.SliceLayout(0, mfma_layout))
    values = gl.full(
        [block_kv],
        -float("inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(0, mfma_layout),
    )
    for block_start in tl.range(seq_len, capacity, block_kv):
        candidates = block_start + offsets
        _store_logits(
            logits,
            candidates,
            values,
            use_buffer_store,
            mask=candidates < capacity,
        )


@gluon.jit
def _dsa_matched_core_fp8_async_kernel(
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
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
):
    gl.static_assert(NUM_WARPS == 1)
    warp_size: gl.constexpr = 64
    row = gl.num_programs(0) - gl.program_id(0) - 1

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[_G_NUM_HEADS, _G_BLOCK_KV, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_layout,
        k_width=16,
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_layout,
        k_width=16,
    )
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, q_load_layout))
    dims = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    q_offsets = (
        row * (_G_NUM_HEADS * _G_HEAD_DIM)
        + heads[:, None] * _G_HEAD_DIM
        + dims[None, :]
    ).to(gl.int32)
    query = gl.amd.cdna4.buffer_load(ptr=q_fp8, offsets=q_offsets, cache=".cg")
    mfma_q = gl.convert_layout(query, dot_q_layout)

    weight_heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))[
        :, None
    ]
    head_weights = gl.amd.cdna4.buffer_load(
        ptr=weights,
        offsets=(row * _G_NUM_HEADS + weight_heads).to(gl.int32),
        cache=".cg",
    ).to(gl.float32)

    loader = _AsyncKLoader.initialize(
        k_fp8,
        seq_len,
        _G_BLOCK_KV,
        _G_HEAD_DIM,
        NUM_WARPS,
        warp_size,
        _G_NUM_BUFFERS,
    )
    row_logits = logits + row.to(gl.int64) * logits_stride
    num_full_tiles = seq_len // _G_BLOCK_KV
    _score_contiguous_double_buffered(
        loader,
        mfma_q,
        head_weights,
        k_scale,
        row_logits,
        seq_len,
        num_full_tiles,
        softmax_scale,
        _G_NUM_HEADS,
        _G_BLOCK_KV,
        _G_NUM_CHAINS,
        mfma_layout,
        dot_k_layout,
        USE_BUFFER_LOAD,
        USE_BUFFER_STORE,
    )
    _fill_rectangular_tail(
        row_logits,
        seq_len,
        capacity,
        _G_BLOCK_KV,
        mfma_layout,
        USE_BUFFER_STORE,
    )


def _validate_matched_core_inputs(
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
    expected_logits_shape = (q_fp8.shape[0], k_fp8.shape[0])
    if logits.dtype != torch.float32 or logits.shape != expected_logits_shape:
        raise ValueError(f"logits must be preallocated FP32 {expected_logits_shape}")

    tensors = (q_fp8, k_fp8, k_scale, weights, logits)
    if any(tensor.device != q_fp8.device for tensor in tensors):
        raise ValueError("matched-core tensors must share a device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("matched-core tensors must be contiguous")


def launch_dsa_matched_core_fp8_async_gfx950(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
    *,
    seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Score contiguous FP8 Q/K with async K staging and exact tail masking."""
    _validate_matched_core_inputs(q_fp8, k_fp8, k_scale, weights, logits)
    capacity = int(k_fp8.shape[0])
    seq_len = int(seq_len)
    if seq_len < 0 or seq_len > capacity:
        raise ValueError(f"seq_len must be in [0, {capacity}], got {seq_len}")
    if q_fp8.shape[0] == 0 or capacity == 0:
        return logits

    use_buffer_load = k_fp8.numel() * k_fp8.element_size() < _BUFFER_LIMIT_BYTES
    use_buffer_store = logits.numel() * logits.element_size() < _BUFFER_LIMIT_BYTES
    _dsa_matched_core_fp8_async_kernel[(q_fp8.shape[0],)](
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
        USE_BUFFER_LOAD=use_buffer_load,
        USE_BUFFER_STORE=use_buffer_store,
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVES_PER_EU,
    )
    return logits
