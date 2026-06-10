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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


_CDNA4_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(9, 5),
    max_arch_version=ArchVersion(9, 5),
    vendors=frozenset({"amd"}),
)
_RMSNORM_SIGNATURES = format_signatures("x", "dense", {torch.float16, torch.bfloat16})


def _contiguous_elements(tensor: torch.Tensor) -> int:
    return max(1, 16 // tensor.element_size())


def _row_alignment_elements(hidden_size: int, tensor: torch.Tensor) -> int:
    alignment = _contiguous_elements(tensor)
    while alignment > 1 and hidden_size % alignment != 0:
        alignment //= 2
    return alignment


def _select_rows_per_cta(n_rows: int, target_workgroups: int = 512) -> int:
    if n_rows <= target_workgroups:
        return 1
    rows_per_cta = triton.next_power_of_2(triton.cdiv(n_rows, target_workgroups))
    return min(rows_per_cta, 16)


def _select_initial_pipeline_groups(rows_per_cta: int, num_buffers: int) -> int:
    return min(rows_per_cta, num_buffers)


def _validate_pipelined_row_mapping(row_mapping: str) -> None:
    if row_mapping not in {"contiguous", "interleaved"}:
        raise ValueError(
            "row_mapping must be either 'contiguous' or 'interleaved', "
            f"got {row_mapping!r}"
        )


@gluon.jit
def _rmsnorm_block_full_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: gl.constexpr,
    eps: gl.constexpr,
    BLOCK: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    ROW_CONTIGUITY: gl.constexpr,
    WEIGHT_CONTIGUITY: gl.constexpr,
    ROW_ALIGNMENT: gl.constexpr,
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    row_cols = gl.max_contiguous(offsets, ROW_CONTIGUITY)
    weight_offsets = gl.max_contiguous(offsets + 0, WEIGHT_CONTIGUITY)
    mask = offsets < n_cols
    row_start = row * n_cols
    row_start = gl.multiple_of(row_start, ROW_ALIGNMENT)
    row_offsets = row_start + row_cols
    row_offsets = gl.max_contiguous(row_offsets, ROW_CONTIGUITY)

    x = gl.amd.cdna4.buffer_load(
        ptr=x_ptr, offsets=row_offsets, mask=mask, other=0.0
    ).to(gl.float32)
    if HAS_RESIDUAL:
        residual = gl.amd.cdna4.buffer_load(
            ptr=residual_ptr, offsets=row_offsets, mask=mask, other=0.0
        ).to(gl.float32)
        x += residual
        gl.amd.cdna4.buffer_store(
            stored_value=x.to(residual_out_ptr.dtype.element_ty),
            ptr=residual_out_ptr,
            offsets=row_offsets,
            mask=mask,
        )

    variance = gl.sum(x * x, axis=0) / n_cols
    x *= gl.rsqrt(variance + eps)
    weight = gl.amd.cdna4.buffer_load(
        ptr=weight_ptr, offsets=weight_offsets, mask=mask, other=0.0
    ).to(gl.float32)
    out = x * weight
    gl.amd.cdna4.buffer_store(
        stored_value=out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=row_offsets,
        mask=mask,
    )


@gluon.jit
def _rmsnorm_triton_like_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: gl.constexpr,
    eps: gl.constexpr,
    BLOCK: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    x = gl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(gl.float32)
    if HAS_RESIDUAL:
        residual = gl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            gl.float32
        )
        x += residual
        gl.store(
            residual_out_ptr + row_offsets,
            x.to(residual_out_ptr.dtype.element_ty),
            mask=mask,
        )

    variance = gl.sum(x * x, axis=0) / n_cols
    x *= gl.rsqrt(variance + eps)
    weight = gl.load(weight_ptr + offsets, mask=mask, other=0.0).to(gl.float32)
    gl.store(
        out_ptr + row_offsets, (x * weight).to(out_ptr.dtype.element_ty), mask=mask
    )


@gluon.jit
def _rmsnorm_wave_row_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: gl.constexpr,
    eps: gl.constexpr,
    BLOCK: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    ROW_CONTIGUITY: gl.constexpr,
    WEIGHT_CONTIGUITY: gl.constexpr,
    ROW_ALIGNMENT: gl.constexpr,
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    row_cols = gl.max_contiguous(offsets, ROW_CONTIGUITY)
    weight_offsets = gl.max_contiguous(offsets + 0, WEIGHT_CONTIGUITY)
    mask = offsets < n_cols
    row_start = row * n_cols
    row_start = gl.multiple_of(row_start, ROW_ALIGNMENT)
    row_offsets = row_start + row_cols
    row_offsets = gl.max_contiguous(row_offsets, ROW_CONTIGUITY)

    x = gl.amd.cdna4.buffer_load(
        ptr=x_ptr, offsets=row_offsets, mask=mask, other=0.0
    ).to(gl.float32)
    if HAS_RESIDUAL:
        residual = gl.amd.cdna4.buffer_load(
            ptr=residual_ptr, offsets=row_offsets, mask=mask, other=0.0
        ).to(gl.float32)
        x += residual
        gl.amd.cdna4.buffer_store(
            stored_value=x.to(residual_out_ptr.dtype.element_ty),
            ptr=residual_out_ptr,
            offsets=row_offsets,
            mask=mask,
        )

    variance = gl.sum(x * x, axis=0) / n_cols
    x *= gl.rsqrt(variance + eps)
    weight = gl.amd.cdna4.buffer_load(
        ptr=weight_ptr, offsets=weight_offsets, mask=mask, other=0.0
    ).to(gl.float32)
    out = x * weight
    gl.amd.cdna4.buffer_store(
        stored_value=out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=row_offsets,
        mask=mask,
    )


@gluon.jit
def _rmsnorm_block_full_aiter_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: gl.constexpr,
    eps: gl.constexpr,
    BLOCK: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    ROW_CONTIGUITY: gl.constexpr,
    WEIGHT_CONTIGUITY: gl.constexpr,
    ROW_ALIGNMENT: gl.constexpr,
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    row_cols = gl.max_contiguous(offsets, ROW_CONTIGUITY)
    weight_offsets = gl.max_contiguous(offsets + 0, WEIGHT_CONTIGUITY)
    mask = offsets < n_cols
    row_start = row * n_cols
    row_start = gl.multiple_of(row_start, ROW_ALIGNMENT)
    row_ptr = x_ptr + row_start
    row_desc = gl.amd.cdna4.make_buffer_descriptor(row_ptr, (n_cols,), (1,))
    weight_desc = gl.amd.cdna4.make_buffer_descriptor(weight_ptr, (n_cols,), (1,))
    row_offsets = row_start + row_cols
    row_offsets = gl.max_contiguous(row_offsets, ROW_CONTIGUITY)

    x = gl.amd.cdna4.buffer_load(
        ptr=row_desc, offsets=row_cols, mask=mask, other=0.0, cache=".cs"
    ).to(gl.float32)
    if HAS_RESIDUAL:
        residual_row_ptr = residual_ptr + row_start
        residual_desc = gl.amd.cdna4.make_buffer_descriptor(
            residual_row_ptr, (n_cols,), (1,)
        )
        residual = gl.amd.cdna4.buffer_load(
            ptr=residual_desc, offsets=row_cols, mask=mask, other=0.0, cache=".cs"
        ).to(gl.float32)
        x += residual
        gl.amd.cdna4.buffer_store(
            stored_value=x.to(residual_out_ptr.dtype.element_ty),
            ptr=residual_out_ptr,
            offsets=row_offsets,
            mask=mask,
            cache=".cs",
        )

    variance = gl.sum(x * x, axis=0) / n_cols
    rstd = gl.rsqrt(variance + eps)
    weight = gl.amd.cdna4.buffer_load(
        ptr=weight_desc, offsets=weight_offsets, mask=mask, other=0.0
    ).to(gl.float32)
    out = x * rstd * weight
    gl.amd.cdna4.buffer_store(
        stored_value=out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=row_offsets,
        mask=mask,
    )


@gluon.jit
def _rmsnorm_block_full_aiter_aligned_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: gl.constexpr,
    eps: gl.constexpr,
    BLOCK: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    ROW_CONTIGUITY: gl.constexpr,
    WEIGHT_CONTIGUITY: gl.constexpr,
    ROW_ALIGNMENT: gl.constexpr,
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    row_cols = gl.max_contiguous(offsets, ROW_CONTIGUITY)
    weight_offsets = gl.max_contiguous(offsets + 0, WEIGHT_CONTIGUITY)
    store_mask = offsets < n_cols
    row_start = row * n_cols
    row_start = gl.multiple_of(row_start, ROW_ALIGNMENT)
    row_ptr = x_ptr + row_start
    row_desc = gl.amd.cdna4.make_buffer_descriptor(row_ptr, (n_cols,), (1,))
    weight_desc = gl.amd.cdna4.make_buffer_descriptor(weight_ptr, (n_cols,), (1,))
    row_offsets = row_start + row_cols
    row_offsets = gl.max_contiguous(row_offsets, ROW_CONTIGUITY)

    x = gl.amd.cdna4.buffer_load(ptr=row_desc, offsets=row_cols, cache=".cs").to(
        gl.float32
    )
    if HAS_RESIDUAL:
        residual_row_ptr = residual_ptr + row_start
        residual_desc = gl.amd.cdna4.make_buffer_descriptor(
            residual_row_ptr, (n_cols,), (1,)
        )
        residual = gl.amd.cdna4.buffer_load(
            ptr=residual_desc, offsets=row_cols, cache=".cs"
        ).to(gl.float32)
        x += residual
        gl.amd.cdna4.buffer_store(
            stored_value=x.to(residual_out_ptr.dtype.element_ty),
            ptr=residual_out_ptr,
            offsets=row_offsets,
            mask=store_mask,
            cache=".cs",
        )

    variance = gl.sum(x * x, axis=0) / n_cols
    rstd = gl.rsqrt(variance + eps)
    weight = gl.amd.cdna4.buffer_load(ptr=weight_desc, offsets=weight_offsets).to(
        gl.float32
    )
    out = x * rstd * weight
    gl.amd.cdna4.buffer_store(
        stored_value=out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=row_offsets,
        mask=store_mask,
    )


@gluon.jit
def _rmsnorm_block_full_aiter_pipelined_aligned_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: gl.constexpr,
    eps: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_TILES: gl.constexpr,
    ROWS_PER_CTA: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    INITIAL_GROUPS: gl.constexpr,
    INTERLEAVED_ROWS: gl.constexpr,
    ROW_CONTIGUITY: gl.constexpr,
    WEIGHT_CONTIGUITY: gl.constexpr,
    ROW_ALIGNMENT: gl.constexpr,
    layout: gl.constexpr,
):
    tile = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    row_cols = gl.max_contiguous(offsets, ROW_CONTIGUITY)
    weight_offsets = gl.max_contiguous(offsets + 0, WEIGHT_CONTIGUITY)
    store_mask = offsets < n_cols

    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    smem = gl.allocate_shared_memory(
        x_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK], shared_layout
    )

    base_row = tile * ROWS_PER_CTA

    for preload_offset in gl.static_range(0, NUM_BUFFERS):
        if preload_offset < ROWS_PER_CTA:
            if INTERLEAVED_ROWS:
                preload_row = tile + preload_offset * NUM_TILES
            else:
                preload_row = base_row + preload_offset
            preload_row_start = preload_row * n_cols
            preload_row_start = gl.multiple_of(preload_row_start, ROW_ALIGNMENT)
            preload_row_ptr = x_ptr + preload_row_start
            preload_row_desc = gl.amd.cdna4.make_buffer_descriptor(
                preload_row_ptr, (n_cols,), (1,)
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem.index(preload_offset),
                preload_row_desc,
                row_cols,
                cache_modifier=".cs",
            )
            gl.amd.cdna4.async_copy.commit_group()

    weight_desc = gl.amd.cdna4.make_buffer_descriptor(weight_ptr, (n_cols,), (1,))
    weight = gl.amd.cdna4.buffer_load(ptr=weight_desc, offsets=weight_offsets).to(
        gl.float32
    )

    gl.amd.cdna4.async_copy.wait_group(INITIAL_GROUPS - 1)

    x = gl.amd.cdna4.async_copy.load_shared_relaxed(smem.index(0), layout).to(
        gl.float32
    )

    for row_offset in gl.static_range(0, ROWS_PER_CTA):
        if INTERLEAVED_ROWS:
            row = tile + row_offset * NUM_TILES
            prefetch_row = tile + (row_offset + NUM_BUFFERS) * NUM_TILES
        else:
            row = base_row + row_offset
            prefetch_row = row + NUM_BUFFERS
        if row_offset + NUM_BUFFERS < ROWS_PER_CTA:
            prefetch_row_start = prefetch_row * n_cols
            prefetch_row_start = gl.multiple_of(prefetch_row_start, ROW_ALIGNMENT)
            prefetch_row_ptr = x_ptr + prefetch_row_start
            prefetch_row_desc = gl.amd.cdna4.make_buffer_descriptor(
                prefetch_row_ptr, (n_cols,), (1,)
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem.index(row_offset % NUM_BUFFERS),
                prefetch_row_desc,
                row_cols,
                cache_modifier=".cs",
            )
            gl.amd.cdna4.async_copy.commit_group()

        variance = gl.sum(x * x, axis=0) / n_cols
        rstd = gl.rsqrt(variance + eps)
        out = x * rstd * weight

        out_row_start = row * n_cols
        out_row_start = gl.multiple_of(out_row_start, ROW_ALIGNMENT)
        out_offsets = out_row_start + row_cols
        out_offsets = gl.max_contiguous(out_offsets, ROW_CONTIGUITY)
        gl.amd.cdna4.buffer_store(
            stored_value=out.to(out_ptr.dtype.element_ty),
            ptr=out_ptr,
            offsets=out_offsets,
            mask=store_mask,
        )

        if row_offset + 1 < ROWS_PER_CTA:
            if row_offset + NUM_BUFFERS < ROWS_PER_CTA:
                gl.amd.cdna4.async_copy.wait_group(NUM_BUFFERS - 1)
            else:
                gl.amd.cdna4.async_copy.wait_group(ROWS_PER_CTA - row_offset - 2)
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(
                smem.index((row_offset + 1) % NUM_BUFFERS), layout
            ).to(gl.float32)


@gluon.jit
def _rmsnorm_streaming_block_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: gl.constexpr,
    eps: gl.constexpr,
    COL_BLOCK: gl.constexpr,
    NUM_CHUNKS: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    ROW_CONTIGUITY: gl.constexpr,
    WEIGHT_CONTIGUITY: gl.constexpr,
    ROW_ALIGNMENT: gl.constexpr,
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, COL_BLOCK, layout=layout)
    row_start = row * n_cols
    row_start = gl.multiple_of(row_start, ROW_ALIGNMENT)
    sum_squares = gl.full((), value=0.0, dtype=gl.float32)

    for chunk in gl.static_range(0, NUM_CHUNKS):
        cols = chunk * COL_BLOCK + offsets
        row_cols = gl.max_contiguous(cols, ROW_CONTIGUITY)
        mask = cols < n_cols
        row_offsets = row_start + row_cols
        row_offsets = gl.max_contiguous(row_offsets, ROW_CONTIGUITY)
        x = gl.amd.cdna4.buffer_load(
            ptr=x_ptr, offsets=row_offsets, mask=mask, other=0.0
        ).to(gl.float32)
        if HAS_RESIDUAL:
            residual = gl.amd.cdna4.buffer_load(
                ptr=residual_ptr, offsets=row_offsets, mask=mask, other=0.0
            ).to(gl.float32)
            x += residual
            gl.amd.cdna4.buffer_store(
                stored_value=x.to(residual_out_ptr.dtype.element_ty),
                ptr=residual_out_ptr,
                offsets=row_offsets,
                mask=mask,
            )
        sum_squares += gl.sum(x * x, axis=0)

    rstd = gl.rsqrt(sum_squares / n_cols + eps)
    for chunk in gl.static_range(0, NUM_CHUNKS):
        cols = chunk * COL_BLOCK + offsets
        row_cols = gl.max_contiguous(cols, ROW_CONTIGUITY)
        weight_offsets = gl.max_contiguous(cols + 0, WEIGHT_CONTIGUITY)
        mask = cols < n_cols
        row_offsets = row_start + row_cols
        row_offsets = gl.max_contiguous(row_offsets, ROW_CONTIGUITY)
        if HAS_RESIDUAL:
            x = gl.amd.cdna4.buffer_load(
                ptr=residual_out_ptr, offsets=row_offsets, mask=mask, other=0.0
            ).to(gl.float32)
        else:
            x = gl.amd.cdna4.buffer_load(
                ptr=x_ptr, offsets=row_offsets, mask=mask, other=0.0
            ).to(gl.float32)
        weight = gl.amd.cdna4.buffer_load(
            ptr=weight_ptr, offsets=weight_offsets, mask=mask, other=0.0
        ).to(gl.float32)
        out = x * rstd * weight
        gl.amd.cdna4.buffer_store(
            stored_value=out.to(out_ptr.dtype.element_ty),
            ptr=out_ptr,
            offsets=row_offsets,
            mask=mask,
        )


def _rmsnorm_block_full(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    num_warps: int = 4,
    size_per_thread: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    residual_arg = residual if residual is not None else x
    residual_out_arg = residual_out if residual_out is not None else out
    row_contiguity = _contiguous_elements(x)
    weight_contiguity = _contiguous_elements(weight)
    row_alignment = _row_alignment_elements(hidden_size, x)

    block = triton.next_power_of_2(hidden_size)
    layout = gl.BlockedLayout([size_per_thread], [64], [num_warps], [0])
    _rmsnorm_block_full_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_arg,
        weight,
        out_2d,
        residual_out_arg,
        hidden_size,
        eps,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
        ROW_CONTIGUITY=row_contiguity,
        WEIGHT_CONTIGUITY=weight_contiguity,
        ROW_ALIGNMENT=row_alignment,
        layout=layout,
        num_warps=num_warps,
    )
    if residual is None:
        return out
    return out, residual_out


def _rmsnorm_wave_row(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    size_per_thread: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    residual_arg = residual if residual is not None else x
    residual_out_arg = residual_out if residual_out is not None else out
    row_contiguity = _contiguous_elements(x)
    weight_contiguity = _contiguous_elements(weight)
    row_alignment = _row_alignment_elements(hidden_size, x)

    block = triton.next_power_of_2(hidden_size)
    num_warps = 1
    layout = gl.BlockedLayout([size_per_thread], [64], [num_warps], [0])
    _rmsnorm_wave_row_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_arg,
        weight,
        out_2d,
        residual_out_arg,
        hidden_size,
        eps,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
        ROW_CONTIGUITY=row_contiguity,
        WEIGHT_CONTIGUITY=weight_contiguity,
        ROW_ALIGNMENT=row_alignment,
        layout=layout,
        num_warps=num_warps,
    )
    if residual is None:
        return out
    return out, residual_out


def _rmsnorm_triton_like(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    num_warps: int = 4,
    size_per_thread: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    residual_arg = residual if residual is not None else x
    residual_out_arg = residual_out if residual_out is not None else out

    block = triton.next_power_of_2(hidden_size)
    layout = gl.BlockedLayout([size_per_thread], [64], [num_warps], [0])
    _rmsnorm_triton_like_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_arg,
        weight,
        out_2d,
        residual_out_arg,
        hidden_size,
        eps,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
        layout=layout,
        num_warps=num_warps,
    )
    if residual is None:
        return out
    return out, residual_out


def _rmsnorm_block_full_aiter(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    num_warps: int = 4,
    size_per_thread: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    residual_arg = residual if residual is not None else x
    residual_out_arg = residual_out if residual_out is not None else out
    row_contiguity = _contiguous_elements(x)
    weight_contiguity = _contiguous_elements(weight)
    row_alignment = _row_alignment_elements(hidden_size, x)

    block = triton.next_power_of_2(hidden_size)
    layout = gl.BlockedLayout([size_per_thread], [64], [num_warps], [0])
    kernel = (
        _rmsnorm_block_full_aiter_aligned_kernel
        if hidden_size % row_contiguity == 0 and hidden_size % weight_contiguity == 0
        else _rmsnorm_block_full_aiter_kernel
    )
    kernel[(x_2d.shape[0],)](
        x_2d,
        residual_arg,
        weight,
        out_2d,
        residual_out_arg,
        hidden_size,
        eps,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
        ROW_CONTIGUITY=row_contiguity,
        WEIGHT_CONTIGUITY=weight_contiguity,
        ROW_ALIGNMENT=row_alignment,
        layout=layout,
        num_warps=num_warps,
    )
    if residual is None:
        return out
    return out, residual_out


def _rmsnorm_block_full_aiter_pipelined(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    num_warps: int = 4,
    size_per_thread: int = 8,
    target_workgroups: int = 512,
    num_buffers: int = 2,
    row_mapping: str = "contiguous",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if num_buffers <= 0:
        raise ValueError(f"num_buffers must be positive, got {num_buffers}")
    _validate_pipelined_row_mapping(row_mapping)
    if residual is not None:
        return _rmsnorm_block_full_aiter(
            x,
            weight,
            eps,
            residual=residual,
            out=out,
            num_warps=num_warps,
            size_per_thread=size_per_thread,
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    row_contiguity = _contiguous_elements(x)
    weight_contiguity = _contiguous_elements(weight)
    row_alignment = _row_alignment_elements(hidden_size, x)
    rows_per_cta = _select_rows_per_cta(x_2d.shape[0], target_workgroups)
    initial_groups = _select_initial_pipeline_groups(rows_per_cta, num_buffers)

    if (
        rows_per_cta == 1
        or x_2d.shape[0] % rows_per_cta != 0
        or hidden_size % row_contiguity != 0
        or hidden_size % weight_contiguity != 0
    ):
        return _rmsnorm_block_full_aiter(
            x,
            weight,
            eps,
            residual=None,
            out=out,
            num_warps=num_warps,
            size_per_thread=size_per_thread,
        )

    block = triton.next_power_of_2(hidden_size)
    layout = gl.BlockedLayout([size_per_thread], [64], [num_warps], [0])
    num_tiles = triton.cdiv(x_2d.shape[0], rows_per_cta)
    _rmsnorm_block_full_aiter_pipelined_aligned_kernel[(num_tiles,)](
        x_2d,
        x_2d,
        weight,
        out_2d,
        out_2d,
        hidden_size,
        eps,
        BLOCK=block,
        NUM_TILES=num_tiles,
        ROWS_PER_CTA=rows_per_cta,
        NUM_BUFFERS=num_buffers,
        INITIAL_GROUPS=initial_groups,
        INTERLEAVED_ROWS=row_mapping == "interleaved",
        ROW_CONTIGUITY=row_contiguity,
        WEIGHT_CONTIGUITY=weight_contiguity,
        ROW_ALIGNMENT=row_alignment,
        layout=layout,
        num_warps=num_warps,
    )
    return out


def _rmsnorm_streaming_block(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    col_block: int = 1024,
    num_warps: int = 4,
    size_per_thread: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )
    if col_block <= 0:
        raise ValueError(f"col_block must be positive, got {col_block}")

    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    residual_arg = residual if residual is not None else x
    residual_out_arg = residual_out if residual_out is not None else out
    row_contiguity = _contiguous_elements(x)
    weight_contiguity = _contiguous_elements(weight)
    row_alignment = _row_alignment_elements(hidden_size, x)

    num_chunks = triton.cdiv(hidden_size, col_block)
    layout = gl.BlockedLayout([size_per_thread], [64], [num_warps], [0])
    _rmsnorm_streaming_block_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_arg,
        weight,
        out_2d,
        residual_out_arg,
        hidden_size,
        eps,
        COL_BLOCK=col_block,
        NUM_CHUNKS=num_chunks,
        HAS_RESIDUAL=residual is not None,
        ROW_CONTIGUITY=row_contiguity,
        WEIGHT_CONTIGUITY=weight_contiguity,
        ROW_ALIGNMENT=row_alignment,
        layout=layout,
        num_warps=num_warps,
    )
    if residual is None:
        return out
    return out, residual_out


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency"},
)
def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_block_full(x, weight, eps, residual=residual, out=out)


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm_block_full",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency", "block_full"},
)
def rmsnorm_block_full(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_block_full(x, weight, eps, residual=residual, out=out)


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm_triton_like",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency", "block_full", "triton_like"},
)
def rmsnorm_triton_like(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_triton_like(x, weight, eps, residual=residual, out=out)


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm_block_full_aiter",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency", "block_full", "aiter"},
)
def rmsnorm_block_full_aiter(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_block_full_aiter(x, weight, eps, residual=residual, out=out)


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm_block_full_aiter_pipelined",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency", "block_full", "aiter", "pipelined"},
)
def rmsnorm_block_full_aiter_pipelined(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_block_full_aiter_pipelined(
        x, weight, eps, residual=residual, out=out
    )


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm_block_full_aiter_pipelined_interleaved",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency", "block_full", "aiter", "pipelined", "interleaved"},
)
def rmsnorm_block_full_aiter_pipelined_interleaved(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_block_full_aiter_pipelined(
        x,
        weight,
        eps,
        residual=residual,
        out=out,
        row_mapping="interleaved",
    )


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm_wave_row",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency", "wave_row"},
)
def rmsnorm_wave_row(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_wave_row(x, weight, eps, residual=residual, out=out)


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm_streaming_block",
    solution="gluon",
    capability=_CDNA4_CAPABILITY,
    signatures=_RMSNORM_SIGNATURES,
    traits={},
    priority=Priority.SPECIALIZED,
    tags={"latency", "streaming_block"},
)
def rmsnorm_streaming_block(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_streaming_block(x, weight, eps, residual=residual, out=out)
