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

"""Reference MoE kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures
from tokenspeed_numerics_input_generators import moe_reference


@register_kernel(
    "moe",
    "apply",
    name="torch_moe_apply",
    solution="reference",
    signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Reference routed MoE apply using the operation-level generated values."""

    del plan
    del x
    del router_logits
    del topk_weights
    del topk_ids
    del num_tokens_global
    del max_num_tokens_per_gpu
    del do_finalize
    del enable_pdl
    values = getattr(w, "_tokenspeed_numerics_values", None)
    if values is None:
        raise ValueError("MoE reference requires generated values on weight module")
    return moe_reference(values, output_dtype=values.hidden_states.dtype)


@register_kernel(
    "moe",
    "process_weights",
    name="torch_moe_process_weights",
    solution="reference",
    signatures=frozenset({format_signature()}),
    traits={"weight_dtype": frozenset({"mxfp4"})},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_moe_process_weights(
    plan: dict,
    w: torch.nn.Module,
) -> torch.Tensor:
    """Reference MoE output for the weights before backend preprocessing."""

    del plan
    values = getattr(w, "_tokenspeed_numerics_values", None)
    if values is None:
        raise ValueError("MoE reference requires generated values on weight module")
    return moe_reference(values, output_dtype=values.hidden_states.dtype)
