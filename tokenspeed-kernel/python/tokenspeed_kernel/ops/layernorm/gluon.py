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
_RMSNORM_SIGNATURES = format_signatures(
    "x", "dense", {torch.float16, torch.bfloat16}
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
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

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
        ptr=weight_ptr, offsets=offsets, mask=mask, other=0.0
    ).to(gl.float32)
    out = x * weight
    gl.amd.cdna4.buffer_store(
        stored_value=out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=row_offsets,
        mask=mask,
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
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, BLOCK, layout=layout)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

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
        ptr=weight_ptr, offsets=offsets, mask=mask, other=0.0
    ).to(gl.float32)
    out = x * weight
    gl.amd.cdna4.buffer_store(
        stored_value=out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr,
        offsets=row_offsets,
        mask=mask,
    )


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
    layout: gl.constexpr,
):
    row = gl.program_id(0)
    offsets = gl.arange(0, COL_BLOCK, layout=layout)
    sum_squares = gl.full((), value=0.0, dtype=gl.float32)

    for chunk in gl.static_range(0, NUM_CHUNKS):
        cols = chunk * COL_BLOCK + offsets
        mask = cols < n_cols
        row_offsets = row * n_cols + cols
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
        mask = cols < n_cols
        row_offsets = row * n_cols + cols
        if HAS_RESIDUAL:
            x = gl.amd.cdna4.buffer_load(
                ptr=residual_out_ptr, offsets=row_offsets, mask=mask, other=0.0
            ).to(gl.float32)
        else:
            x = gl.amd.cdna4.buffer_load(
                ptr=x_ptr, offsets=row_offsets, mask=mask, other=0.0
            ).to(gl.float32)
        weight = gl.amd.cdna4.buffer_load(
            ptr=weight_ptr, offsets=cols, mask=mask, other=0.0
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
        layout=layout,
        num_warps=num_warps,
    )
    if residual is None:
        return out
    return out, residual_out


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
