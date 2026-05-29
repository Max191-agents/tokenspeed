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
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "norm",
    "rmsnorm",
    name="torch_rmsnorm",
    solution="reference",
    signatures=format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
    traits={},
    priority=Priority.REFERENCE,
    tags={"portability"},
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

    x_float = x.to(torch.float32)
    residual_out = None
    if residual is not None:
        x_float = x_float + residual.to(torch.float32)
        residual_out = x_float.to(x.dtype)

    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    result = (x_float * torch.rsqrt(variance + eps) * weight.to(torch.float32)).to(
        x.dtype
    )
    if out is not None:
        out.copy_(result)
        result = out
    if residual is None:
        return result
    return result, residual_out
