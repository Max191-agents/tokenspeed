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

"""Experimental MFMA scorer for the fixed GLM DSA indexer on AMD GFX950.

The production cache packs each 64-row page as 64x128 FP8 bytes followed by
64 FP32 row scales.  The kernels below preserve BF16 queries, widen the FP8
mantissas to BF16 (an exact conversion), perform BF16 MFMA with FP32
accumulation, and apply the row scale before the model's per-head ReLU.

Each program owns one query row and scans all candidate tiles.  This avoids
reloading the 32x128 query for every tile and matches AITER's MQA-logits data
flow.  It is intentionally kept separate from production dispatch until its
accuracy and occupancy have been measured on gfx950.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = [
    "_dsa_decode_logits_mfma_kernel",
    "_dsa_prefill_logits_mfma_kernel",
    "launch_dsa_decode_logits_mfma_gfx950",
    "launch_dsa_prefill_logits_mfma_gfx950",
]

_NUM_HEADS = 32
_HEAD_DIM = 128
_PAGE_SIZE = 64
_ROW_BYTES = 132
_BLOCK_N = 32
_NUM_WARPS = 1

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
    load_layout: gl.constexpr,
    mfma_layout: gl.constexpr,
):
    head_offsets = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, load_layout))
    dim_offsets = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, load_layout))
    q_offsets = (
        token * (_G_NUM_HEADS * _G_HEAD_DIM)
        + head_offsets[:, None] * _G_HEAD_DIM
        + dim_offsets[None, :]
    ).to(gl.int32)
    q_block = gl.amd.cdna4.buffer_load(ptr=q, offsets=q_offsets)

    weight_heads = gl.arange(0, _G_NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))
    weight_block = gl.amd.cdna4.buffer_load(
        ptr=weights,
        offsets=(token * _G_NUM_HEADS + weight_heads).to(gl.int32),
    )
    return q_block, weight_block


@gluon.jit
def _score_tile(
    mfma_q,
    weight_block,
    k_block,
    k_scale,
    valid,
    softmax_scale: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_k_layout: gl.constexpr,
):
    # Every finite E4M3 value is exactly representable in BF16.  Applying the
    # per-row scale after the dot therefore avoids quantizing the BF16 query.
    k_block = k_block.to(gl.bfloat16)
    mfma_k = gl.convert_layout(k_block.T, dot_k_layout)
    head_scores = gl.zeros(
        [_G_NUM_HEADS, _G_BLOCK_N], dtype=gl.float32, layout=mfma_layout
    )
    head_scores = gl.amd.cdna4.mfma(mfma_q, mfma_k, head_scores)
    head_scores *= k_scale[None, :]
    head_scores = gl.maximum(head_scores, 0.0)
    scores = gl.sum(head_scores * weight_block[:, None], axis=0)
    scores *= softmax_scale
    return gl.where(valid, scores, -float("inf"))


@gluon.jit
def _dsa_decode_logits_mfma_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    block_table_stride: gl.constexpr,
    logits_stride: gl.constexpr,
    max_seq_len: gl.int32,
    softmax_scale: gl.constexpr,
    q_len_per_req: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    mfma_layout: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_layout,
        k_width=8,
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_layout,
        k_width=8,
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    q_block, weight_block = _load_query_and_weights(
        q, weights, token, load_layout, mfma_layout
    )
    mfma_q = gl.convert_layout(q_block, dot_q_layout)
    req = token // q_len_per_req
    q_offset = token - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset

    candidate_load_offsets = gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(1, load_layout)
    )
    dim_offsets = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, load_layout))
    candidate_score_offsets = gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    page_bytes: gl.constexpr = _G_PAGE_SIZE * _G_ROW_BYTES
    scale_page_offset: gl.constexpr = (_G_PAGE_SIZE * _G_HEAD_DIM) // 4

    for block_start in range(0, max_seq_len, _G_BLOCK_N):
        # BLOCK_N divides PAGE_SIZE, so every tile is wholly inside one page.
        logical_page = block_start // _G_PAGE_SIZE
        row_in_page = block_start - logical_page * _G_PAGE_SIZE
        page = gl.load(
            block_table + req * block_table_stride + logical_page,
            mask=block_start < seq_len,
            other=0,
        ).to(gl.int64)

        load_candidates = block_start + candidate_load_offsets
        load_valid = load_candidates < seq_len
        # Buffer offsets are 32-bit, so move the scalar page base with 64-bit
        # pointer arithmetic and keep only the bounded in-page offset here.
        k_page = index_k_fp8 + page * page_bytes
        k_page_offsets = (
            (row_in_page + candidate_load_offsets[:, None]) * _G_HEAD_DIM
            + dim_offsets[None, :]
        ).to(gl.int32)
        k_block = gl.amd.cdna4.buffer_load(
            ptr=k_page,
            offsets=k_page_offsets,
            mask=load_valid[:, None],
            other=0.0,
        )

        score_candidates = block_start + candidate_score_offsets
        score_valid = score_candidates < seq_len
        scale_page = index_k_scale + page * (page_bytes // 4)
        scale_offsets = (scale_page_offset + row_in_page + candidate_score_offsets).to(
            gl.int32
        )
        k_scale = gl.amd.cdna4.buffer_load(
            ptr=scale_page,
            offsets=scale_offsets,
            mask=score_valid,
            other=0.0,
        ).to(gl.float32)

        scores = _score_tile(
            mfma_q,
            weight_block,
            k_block,
            k_scale,
            score_valid,
            softmax_scale,
            mfma_layout,
            dot_k_layout,
        )
        gl.amd.cdna4.buffer_store(
            scores,
            ptr=logits,
            offsets=(token * logits_stride + score_candidates).to(gl.int32),
            mask=score_candidates < max_seq_len,
        )


@gluon.jit
def _dsa_prefill_logits_mfma_kernel(
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
    mfma_layout: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1],
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_layout,
        k_width=8,
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_layout,
        k_width=8,
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    q_block, weight_block = _load_query_and_weights(
        q, weights, token, load_layout, mfma_layout
    )
    mfma_q = gl.convert_layout(q_block, dot_q_layout)
    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    candidate_load_offsets = gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(1, load_layout)
    )
    dim_offsets = gl.arange(0, _G_HEAD_DIM, layout=gl.SliceLayout(0, load_layout))
    candidate_score_offsets = gl.arange(
        0, _G_BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    page_bytes: gl.constexpr = _G_PAGE_SIZE * _G_ROW_BYTES
    scale_page_offset: gl.constexpr = (_G_PAGE_SIZE * _G_HEAD_DIM) // 4

    for block_start in range(0, seq_len_sum, _G_BLOCK_N):
        load_candidates = block_start + candidate_load_offsets
        load_in_bounds = load_candidates < seq_len_sum
        load_valid = (
            load_in_bounds
            & (load_candidates >= row_start)
            & (load_candidates < row_end)
        )
        load_slots = gl.amd.cdna4.buffer_load(
            ptr=kv_workspace_slots,
            offsets=load_candidates.to(gl.int32),
            mask=load_in_bounds,
            other=0,
        ).to(gl.int64)
        load_pages = load_slots // _G_PAGE_SIZE
        load_rows = load_slots - load_pages * _G_PAGE_SIZE
        # Workspace lanes may name different pages.  Form each full physical
        # address in 64 bits instead of narrowing it to a buffer offset.
        k_offsets = (
            load_pages[:, None] * page_bytes
            + load_rows[:, None] * _G_HEAD_DIM
            + dim_offsets[None, :].to(gl.int64)
        )
        k_block = gl.load(
            index_k_fp8 + k_offsets,
            mask=load_valid[:, None],
            other=0.0,
        )

        score_candidates = block_start + candidate_score_offsets
        score_in_bounds = score_candidates < seq_len_sum
        score_valid = (
            score_in_bounds
            & (score_candidates >= row_start)
            & (score_candidates < row_end)
        )
        score_slots = gl.amd.cdna4.buffer_load(
            ptr=kv_workspace_slots,
            offsets=score_candidates.to(gl.int32),
            mask=score_in_bounds,
            other=0,
        ).to(gl.int64)
        score_pages = score_slots // _G_PAGE_SIZE
        score_rows = score_slots - score_pages * _G_PAGE_SIZE
        scale_offsets = score_pages * (page_bytes // 4) + scale_page_offset + score_rows
        k_scale = gl.load(
            index_k_scale + scale_offsets,
            mask=score_valid,
            other=0.0,
        ).to(gl.float32)

        scores = _score_tile(
            mfma_q,
            weight_block,
            k_block,
            k_scale,
            score_valid,
            softmax_scale,
            mfma_layout,
            dot_k_layout,
        )
        gl.amd.cdna4.buffer_store(
            scores,
            ptr=logits,
            offsets=(token * logits_stride + score_candidates).to(gl.int32),
            mask=score_in_bounds,
        )


def _validate_common(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    if q.dtype != torch.bfloat16 or q.dim() != 3:
        raise ValueError("q must be contiguous BF16 [tokens, 32, 128]")
    if tuple(q.shape[1:]) != (_NUM_HEADS, _HEAD_DIM):
        raise ValueError(f"q must be [tokens, 32, 128], got {tuple(q.shape)}")
    if weights.dtype != torch.float32 or weights.shape != q.shape[:2]:
        raise ValueError(f"weights must be FP32 {tuple(q.shape[:2])}")
    if (
        index_k_cache.dtype != torch.uint8
        or index_k_cache.dim() != 2
        or index_k_cache.shape[1] != _ROW_BYTES
        or index_k_cache.shape[0] % _PAGE_SIZE != 0
    ):
        raise ValueError("index_k_cache must be packed uint8 [page-aligned slots, 132]")
    if logits.dtype != torch.float32 or logits.dim() != 2:
        raise ValueError("logits must be preallocated FP32 [tokens, candidates]")
    if logits.shape[0] != q.shape[0]:
        raise ValueError("logits and q must have the same token count")
    tensors = (q, index_k_cache, weights, logits)
    if any(t.device != q.device for t in tensors):
        raise ValueError("q, index_k_cache, weights, and logits must share a device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("MFMA scoring inputs and logits must be contiguous")


def launch_dsa_decode_logits_mfma_gfx950(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    logits: torch.Tensor,
    *,
    softmax_scale: float,
    q_len_per_req: int = 1,
) -> torch.Tensor:
    """Launch the row-scanning decode scorer into preallocated ``logits``."""
    _validate_common(q, index_k_cache, weights, logits)
    if seq_lens.dtype != torch.int32 or seq_lens.dim() != 1:
        raise ValueError("seq_lens must be contiguous int32 [requests]")
    if block_table.dtype != torch.int32 or block_table.dim() != 2:
        raise ValueError("block_table must be contiguous int32 [requests, pages]")
    if seq_lens.device != q.device or block_table.device != q.device:
        raise ValueError("decode metadata must be on the q device")
    if not seq_lens.is_contiguous() or not block_table.is_contiguous():
        raise ValueError("decode metadata must be contiguous")
    q_len_per_req = int(q_len_per_req)
    if q_len_per_req < 1 or q_len_per_req > 6:
        raise ValueError("q_len_per_req must be in [1, 6]")
    if q.shape[0] != seq_lens.numel() * q_len_per_req:
        raise ValueError("q token count must equal requests * q_len_per_req")
    if block_table.shape[0] != seq_lens.numel():
        raise ValueError("block_table and seq_lens request counts must match")
    max_seq_len = int(block_table.shape[1]) * _PAGE_SIZE
    if logits.shape[1] != max_seq_len:
        raise ValueError(
            f"decode logits width must be {max_seq_len}, got {logits.shape[1]}"
        )
    if q.shape[0] == 0 or max_seq_len == 0:
        return logits

    _dsa_decode_logits_mfma_kernel[(q.shape[0],)](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        block_table.stride(0),
        logits.stride(0),
        max_seq_len,
        softmax_scale=float(softmax_scale),
        q_len_per_req=q_len_per_req,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
    )
    return logits


def launch_dsa_prefill_logits_mfma_gfx950(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    logits: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Launch the row-scanning prefill scorer into preallocated ``logits``."""
    _validate_common(q, index_k_cache, weights, logits)
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
    metadata = (kv_workspace_slots, row_starts, row_ends)
    if any(t.device != q.device for t in metadata):
        raise ValueError("prefill metadata must be on the q device")
    if any(not t.is_contiguous() for t in metadata):
        raise ValueError("prefill metadata must be contiguous")
    seq_len_sum = int(kv_workspace_slots.numel())
    if logits.shape[1] != seq_len_sum:
        raise ValueError(
            f"prefill logits width must be {seq_len_sum}, got {logits.shape[1]}"
        )
    if q.shape[0] == 0 or seq_len_sum == 0:
        return logits

    _dsa_prefill_logits_mfma_kernel[(q.shape[0],)](
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
        num_warps=_NUM_WARPS,
    )
    return logits
