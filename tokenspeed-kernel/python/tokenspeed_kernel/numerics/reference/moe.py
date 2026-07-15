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
from tokenspeed_numerics_input_generators import GemmInputValues, MoeInputValues
from tokenspeed_numerics_input_generators.gemm import (
    _check_gemm_layout,
    _logical_operand,
)


def _moe_weight_operand(
    gemm_values: GemmInputValues,
    *,
    name: str,
    layout: str,
) -> torch.Tensor:
    if gemm_values.B is None:
        raise ValueError(f"{name}.B is required for moe_reference")
    layout = _check_gemm_layout(f"{name}_b_layout", layout)
    is_mxint4 = (
        gemm_values.B.dtype == torch.uint8
        and gemm_values.B_scales is not None
        and gemm_values.B_scales.dtype == torch.bfloat16
    )
    is_mxfp4 = (
        gemm_values.B.dtype == torch.uint8
        and gemm_values.B_scales is not None
        and not is_mxint4
    )
    return _logical_operand(
        gemm_values.B,
        gemm_values.B_scales,
        layout=layout,
        is_mxfp4=is_mxfp4,
        is_mxint4=is_mxint4,
    )


def _validate_topk_values(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> None:
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have matching shapes; got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if topk_ids.shape[0] != num_tokens:
        raise ValueError(
            "topk_ids first dimension must match hidden_states tokens; got "
            f"{topk_ids.shape[0]} and {num_tokens}"
        )
    if topk_ids.shape[1] <= 0:
        raise ValueError("topk_ids must select at least one expert per token")
    if topk_ids.dtype not in {torch.int16, torch.int32, torch.int64}:
        raise ValueError("topk_ids must use an integer dtype")
    if not torch.is_floating_point(topk_weights):
        raise ValueError("topk_weights must use a floating dtype")
    if topk_ids.numel() == 0:
        return
    if topk_ids.min().item() < 0:
        raise ValueError("topk_ids must be non-negative")
    if topk_ids.max().item() >= num_experts:
        raise ValueError("topk_ids must be less than the number of experts")
    sorted_ids = topk_ids.to(torch.long).sort(dim=-1).values
    if sorted_ids.shape[-1] > 1 and torch.any(
        sorted_ids[..., 1:] == sorted_ids[..., :-1]
    ):
        raise ValueError("topk_ids must not contain duplicate experts per token")
    weights = topk_weights.float()
    if not torch.isfinite(weights).all():
        raise ValueError("topk_weights must be finite")
    if torch.any(weights < 0):
        raise ValueError("topk_weights must be non-negative")
    row_sums = weights.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1.0e-3,
        atol=1.0e-3,
    ):
        raise ValueError("topk_weights rows must sum to 1")


def _moe_gate_up_activation(
    gate_up: torch.Tensor,
    *,
    activation: str,
    swiglu_alpha: float | None,
    swiglu_limit: float | None,
) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    if swiglu_limit is not None and swiglu_limit > 0.0:
        gate = gate.clamp(max=float(swiglu_limit))
        up = up.clamp(min=-float(swiglu_limit), max=float(swiglu_limit))
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    if activation == "swiglu":
        alpha = 1.0 if swiglu_alpha is None else float(swiglu_alpha)
        s = gate / (1.0 + torch.exp(-alpha * gate))
        return s * (up + 1.0)
    raise ValueError(f"unsupported MoE activation={activation!r}")


def moe_reference(
    values: MoeInputValues,
    *,
    activation: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    w13_b_layout: str = "NK",
    w2_b_layout: str = "NK",
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the semantic routed MoE layer output for generated values."""

    routing = values.routing
    if routing.hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must be rank-2, got " f"{tuple(routing.hidden_states.shape)}"
        )
    num_tokens, hidden_size = routing.hidden_states.shape
    if routing.router_logits.ndim != 2:
        raise ValueError(
            "router_logits must be rank-2, got " f"{tuple(routing.router_logits.shape)}"
        )
    if routing.router_logits.shape[0] != num_tokens:
        raise ValueError(
            "router_logits first dimension must match hidden_states tokens"
        )
    w13 = _moe_weight_operand(values.w13, name="w13", layout=w13_b_layout)
    w2 = _moe_weight_operand(values.w2, name="w2", layout=w2_b_layout)
    if w13.ndim != 3:
        raise ValueError(
            f"w13.B must be rank-3 after layout normalization, got {w13.ndim}D"
        )
    if w2.ndim != 3:
        raise ValueError(
            f"w2.B must be rank-3 after layout normalization, got {w2.ndim}D"
        )
    num_experts, two_intermediate_size, w13_hidden_size = w13.shape
    w2_num_experts, w2_hidden_size, intermediate_size = w2.shape
    if two_intermediate_size % 2 != 0:
        raise ValueError("w13 output dimension must be even for gate/up split")
    if two_intermediate_size != 2 * intermediate_size:
        raise ValueError(
            "w13 gate/up dimension must be twice the w2 intermediate dimension"
        )
    if w13_hidden_size != hidden_size or w2_hidden_size != hidden_size:
        raise ValueError(
            "MoE hidden dimensions must agree across hidden_states, w13, and w2"
        )
    if w2_num_experts != num_experts:
        raise ValueError("w13 and w2 must have matching expert counts")
    if routing.router_logits.shape[1] != num_experts:
        raise ValueError("router_logits expert dimension must match weight experts")
    _validate_topk_values(
        topk_ids=routing.topk_ids,
        topk_weights=routing.topk_weights,
        num_tokens=num_tokens,
        num_experts=num_experts,
    )
    if values.w13_bias is not None and values.w13_bias.shape != (
        num_experts,
        two_intermediate_size,
    ):
        raise ValueError(
            "w13_bias must have shape [num_experts, 2 * intermediate_size]"
        )
    if values.w2_bias is not None and values.w2_bias.shape != (
        num_experts,
        hidden_size,
    ):
        raise ValueError("w2_bias must have shape [num_experts, hidden_size]")
    if output_dtype is None:
        output_dtype = values.w2.C.dtype
    if not isinstance(output_dtype, torch.dtype):
        raise TypeError("output_dtype must be a torch.dtype")

    hidden = routing.hidden_states.float()
    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=hidden.device,
    )
    for slot in range(routing.topk_ids.shape[1]):
        expert_ids = routing.topk_ids[:, slot].to(torch.long)
        route_weights = routing.topk_weights[:, slot].float().reshape(num_tokens, 1)
        w13_selected = w13[expert_ids].to(device=hidden.device)
        gate_up = torch.bmm(w13_selected.float(), hidden.unsqueeze(-1)).squeeze(-1)
        if values.w13_bias is not None:
            gate_up = gate_up + values.w13_bias[expert_ids].float()
        activated = _moe_gate_up_activation(
            gate_up,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )
        w2_selected = w2[expert_ids].to(device=hidden.device)
        expert_output = torch.bmm(
            w2_selected.float(),
            activated.unsqueeze(-1),
        ).squeeze(-1)
        if values.w2_bias is not None:
            expert_output = expert_output + values.w2_bias[expert_ids].float()
        output = output + expert_output * route_weights
    return output.to(output_dtype)


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
    return moe_reference(values, output_dtype=values.routing.hidden_states.dtype)


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
    return moe_reference(values, output_dtype=values.routing.hidden_states.dtype)
