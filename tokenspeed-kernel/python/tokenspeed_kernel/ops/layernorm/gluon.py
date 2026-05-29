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


@gluon.jit
def _rmsnorm_kernel(
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
        gl.store(residual_out_ptr + row_offsets, x, mask=mask)

    variance = gl.sum(x * x, axis=0) / n_cols
    x *= gl.rsqrt(variance + eps)
    weight = gl.load(weight_ptr + offsets, mask=mask, other=0.0).to(gl.float32)
    gl.store(out_ptr + row_offsets, x * weight, mask=mask)


@register_kernel(
    "norm",
    "rmsnorm",
    name="gluon_rmsnorm",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
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
    num_warps = 4
    layout = gl.BlockedLayout([1], [64], [num_warps], [0])
    _rmsnorm_kernel[(x_2d.shape[0],)](
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
