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

"""Numerics framework hooks for MoE family ops.

The ``align_block_size`` mode covers the trtllm ``moe_align_block_size``
op (the per-block expert dispatch helper). Its raw output is three int32
tensors plus an undefined per-expert order within each block (CUDA
atomics make ``sorted_ids`` non-deterministic). We canonicalize the
output to a single int32 tensor so ``compare_outputs`` works as-is.
"""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.outputs import set_output_extractor
from tokenspeed_kernel.numerics.reference.moe import (
    MoeRoutingResult,
    moe_routing_reference,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_numerics_input_generators import (
    CustomDType,
    MoeDispatchInputConfig,
    MoeDispatchInputs,
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
    MoeRoutingInputConfig,
)


def tolerance(dtype: torch.dtype, *, mode: str | None = None, **_: Any) -> Tolerance:
    if mode in {"apply", "process_weights"}:
        return Tolerance(atol=0.1, rtol=5.0e-2)
    # int32 outputs — both implementations must match exactly.
    return Tolerance(atol=0.0, rtol=0.0)


set_family_tolerance("moe", tolerance)


class MoeAlignBlockSizeInputGenerator(InputGenerator):
    """Adapter for standalone MoE align-block-size input generation.

    Shape kwargs:
        total_tokens: number of input tokens
        top_k:        number of experts each token routes to
        num_experts:  expert pool size
        block_size:   tile width the dispatch packs into
    """

    def generate(
        self,
        *,
        total_tokens: int,
        top_k: int,
        num_experts: int,
        block_size: int,
    ) -> dict[str, Any]:
        values = MoeDispatchInputs(
            MoeDispatchInputConfig(
                num_tokens=total_tokens,
                hidden_size=1,
                top_k=top_k,
                num_experts=num_experts,
                hidden_dtype=torch.float32,
                topk_ids_dtype=self.dtype,
                device=self.device,
            )
        ).generate(seed=self.seed, device=self.device)
        return {
            "topk_ids": values.topk_ids,
            "block_size": block_size,
            "num_experts": num_experts,
        }


set_input_generator("moe", "align_block_size", MoeAlignBlockSizeInputGenerator)


def _single_trait(
    traits: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    values = traits.get(name)
    if values is None:
        return default
    if isinstance(values, frozenset | set):
        if len(values) == 1:
            return next(iter(values))
        return default
    return values


def _make_moe_weight_module(
    values: MoeInputValues,
    *,
    top_k: int,
    weight_dtype: str,
    activation: str,
) -> torch.nn.Module:
    if values.w13.B is None or values.w2.B is None:
        raise ValueError("MoE values must include W13 and W2 weights")

    num_experts = int(values.w13.B.shape[0])
    layer = torch.nn.Module()
    layer.num_experts = num_experts
    layer.num_local_experts = num_experts
    layer.ep_size = 1
    layer.ep_rank = 0
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.top_k = top_k
    layer.activation = activation
    layer.swiglu_arg = None
    layer._tokenspeed_numerics_values = values

    layer.w13_weight = torch.nn.Parameter(values.w13.B.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(values.w2.B.clone(), requires_grad=False)
    if values.w13_bias is not None:
        layer.w13_weight_bias = torch.nn.Parameter(
            values.w13_bias.clone(),
            requires_grad=False,
        )
    if values.w2_bias is not None:
        layer.w2_weight_bias = torch.nn.Parameter(
            values.w2_bias.clone(),
            requires_grad=False,
        )

    if weight_dtype == "mxfp4":
        if values.w13.B_scales is None or values.w2.B_scales is None:
            raise ValueError("MXFP4 MoE values must include weight scales")
        layer.w13_weight_scale = torch.nn.Parameter(
            values.w13.B_scales.clone(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            values.w2.B_scales.clone(),
            requires_grad=False,
        )
        if values.w13_activation_scale is not None:
            layer.w13_input_scale = torch.nn.Parameter(
                values.w13_activation_scale.clone(),
                requires_grad=False,
            )
        if values.w2_activation_scale is not None:
            layer.w2_input_scale = torch.nn.Parameter(
                values.w2_activation_scale.clone(),
                requires_grad=False,
            )
    return layer


class MoeApplyInputGenerator(InputGenerator):
    """Adapter from operation-level ``MoeInputs`` to TokenSpeed MoE kwargs."""

    def _weight_dtype(self, override: str | None) -> str:
        if override is not None:
            return override
        return str(_single_trait(self.traits, "weight_dtype", "mxfp4"))

    def _activation(self, override: str | None) -> str:
        if override is not None:
            return override
        return str(_single_trait(self.traits, "activation", "silu"))

    def _needs_bias(self, weight_dtype: str) -> bool:
        solution = getattr(self.kernel_spec, "solution", None)
        if solution == "gluon" and weight_dtype == "mxfp4":
            return True
        return False

    def _needs_activation_scales(self, weight_dtype: str) -> bool:
        solution = getattr(self.kernel_spec, "solution", None)
        return bool(solution == "gluon" and weight_dtype == "mxfp4")

    def _process_weights_if_needed(
        self,
        plan: dict[str, Any],
        layer: torch.nn.Module,
        *,
        weight_dtype: str,
    ) -> None:
        if self.kernel_spec is None:
            return
        solution = getattr(self.kernel_spec, "solution", None)
        if solution in {None, "reference"}:
            return
        process_kernel = select_kernel(
            "moe",
            "process_weights",
            format_signature(),
            traits={"weight_dtype": weight_dtype},
            solution=solution,
        )
        plan["process_weights_kernel_name"] = process_kernel.name
        process_kernel(plan=plan, w=layer)

    def _select_apply_kernel_name(
        self,
        *,
        weight_dtype: str,
        activation: str,
        internal_activation_dtype: str | None,
    ) -> str:
        solution = getattr(self.kernel_spec, "solution", None)
        if solution in {None, "reference"}:
            return "reference"
        traits: dict[str, Any] = {
            "weight_dtype": weight_dtype,
            "activation": activation,
            "internal_activation_dtype": internal_activation_dtype or "input",
        }
        if weight_dtype == "mxfp4":
            traits["routing_mode"] = "precomputed_topk"
        apply_kernel = select_kernel(
            "moe",
            "apply",
            format_signature(x=dense_tensor_format(self.dtype)),
            traits=traits,
            solution=solution,
        )
        return apply_kernel.name

    def _build_moe_components(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        weight_dtype: str | None,
        activation: str | None,
        internal_activation_dtype: str | None,
        apply_kernel_name: str | None,
    ) -> tuple[MoeInputValues, torch.nn.Module, dict[str, Any], str, str]:
        resolved_weight_dtype = self._weight_dtype(weight_dtype)
        resolved_activation = self._activation(activation)
        if resolved_weight_dtype == "unquant":
            generated_weight_dtype: Any = self.dtype
            weight_scale_dtype = None
        elif resolved_weight_dtype == "mxfp4":
            generated_weight_dtype = CustomDType.MXFP4
            weight_scale_dtype = None
        else:
            raise ValueError(
                "MoE numerics generator currently supports "
                f"weight_dtype='unquant' or 'mxfp4', got {resolved_weight_dtype!r}"
            )

        values = MoeInputs(
            MoeInputConfig(
                routing=MoeRoutingInputConfig(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    hidden_dtype=self.dtype,
                    router_dtype=self.dtype,
                ),
                intermediate_size=intermediate_size,
                weight_dtype=generated_weight_dtype,
                weight_scale_dtype=weight_scale_dtype,
                bias_dtype=(
                    torch.float32 if self._needs_bias(resolved_weight_dtype) else None
                ),
                activation_scale_dtype=(
                    torch.float32
                    if self._needs_activation_scales(resolved_weight_dtype)
                    else None
                ),
            )
        ).generate(seed=self.seed, device=self.device)
        routing = moe_routing_reference(
            values.routing,
            top_k=top_k,
            topk_weights_dtype=self.dtype,
        )

        layer = _make_moe_weight_module(
            values,
            top_k=top_k,
            weight_dtype=resolved_weight_dtype,
            activation=resolved_activation,
        )
        layer._tokenspeed_numerics_routing = routing
        plan = {
            "weight_dtype": resolved_weight_dtype,
            "apply_kernel_name": (
                apply_kernel_name
                if apply_kernel_name is not None
                else getattr(self.kernel_spec, "name", "reference")
            ),
            "process_weights_kernel_name": None,
            "a2a_backend": None,
            "deepep_group": None,
            "support_routing": False,
            "supports_deferred_finalize": False,
            "solution": getattr(self.kernel_spec, "solution", "reference"),
            "internal_activation_dtype": internal_activation_dtype,
        }
        return values, layer, plan, resolved_weight_dtype, resolved_activation

    def generate(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        weight_dtype: str | None = None,
        activation: str | None = None,
        internal_activation_dtype: str | None = None,
    ) -> dict[str, Any]:
        values, layer, plan, resolved_weight_dtype, _ = self._build_moe_components(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            weight_dtype=weight_dtype,
            activation=activation,
            internal_activation_dtype=internal_activation_dtype,
            apply_kernel_name=None,
        )
        self._process_weights_if_needed(
            plan,
            layer,
            weight_dtype=resolved_weight_dtype,
        )
        routing = layer._tokenspeed_numerics_routing
        return {
            "plan": plan,
            "x": values.routing.hidden_states,
            "w": layer,
            "router_logits": routing.router_logits,
            "topk_weights": routing.topk_weights,
            "topk_ids": routing.topk_ids,
            "num_tokens_global": values.routing.hidden_states.shape[0],
            "max_num_tokens_per_gpu": values.routing.hidden_states.shape[0],
            "do_finalize": True,
            "enable_pdl": False,
        }


set_input_generator("moe", "apply", MoeApplyInputGenerator)


class MoeProcessWeightsInputGenerator(MoeApplyInputGenerator):
    """Adapter for observing MoE weight preprocessing through MoE apply.

    ``process_weights`` mutates backend-specific module attributes and returns
    ``None``. The operation-level invariant is that applying the processed
    weights still computes the same routed MoE layer represented by the
    generated values.
    """

    def generate(
        self,
        *,
        num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        weight_dtype: str | None = None,
        activation: str | None = None,
        internal_activation_dtype: str | None = None,
    ) -> dict[str, Any]:
        resolved_weight_dtype = self._weight_dtype(weight_dtype)
        resolved_activation = self._activation(activation)
        apply_kernel_name = self._select_apply_kernel_name(
            weight_dtype=resolved_weight_dtype,
            activation=resolved_activation,
            internal_activation_dtype=internal_activation_dtype,
        )
        _, layer, plan, _, _ = self._build_moe_components(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            weight_dtype=resolved_weight_dtype,
            activation=resolved_activation,
            internal_activation_dtype=internal_activation_dtype,
            apply_kernel_name=apply_kernel_name,
        )
        plan["process_weights_kernel_name"] = getattr(
            self.kernel_spec,
            "name",
            None,
        )
        return {"plan": plan, "w": layer}


set_input_generator("moe", "process_weights", MoeProcessWeightsInputGenerator)


def _process_weights_output(
    inputs: dict[str, Any],
    result: Any,
) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if result is not None:
        raise TypeError(
            "MoE process_weights output extractor expects None or tensor result, "
            f"got {type(result)!r}"
        )

    plan = inputs["plan"]
    w = inputs["w"]
    values = getattr(w, "_tokenspeed_numerics_values", None)
    if values is None:
        raise ValueError("processed MoE module must carry generated MoE values")
    routing = getattr(w, "_tokenspeed_numerics_routing", None)
    if not isinstance(routing, MoeRoutingResult):
        raise ValueError("processed MoE module must carry executed routing results")

    apply_kernel_name = plan.get("apply_kernel_name")
    apply_kernel = KernelRegistry.get().get_impl(apply_kernel_name)
    if apply_kernel is None:
        raise ValueError(f"MoE apply kernel {apply_kernel_name!r} is not registered")
    return apply_kernel(
        plan=plan,
        x=values.routing.hidden_states,
        w=w,
        router_logits=routing.router_logits,
        topk_weights=routing.topk_weights,
        topk_ids=routing.topk_ids,
        num_tokens_global=values.routing.hidden_states.shape[0],
        max_num_tokens_per_gpu=values.routing.hidden_states.shape[0],
        do_finalize=True,
        enable_pdl=False,
    )


set_output_extractor("moe", "process_weights", _process_weights_output)


_MOE_ALIGN_STANDARD_SHAPES: list[dict[str, int]] = [
    # DSv3 routed MoE: 256 experts, top-8.
    {"total_tokens": 16, "top_k": 8, "num_experts": 256, "block_size": 64},
    {"total_tokens": 128, "top_k": 8, "num_experts": 256, "block_size": 64},
    # MiniMax / Kimi 8-expert routing.
    {"total_tokens": 16, "top_k": 8, "num_experts": 64, "block_size": 64},
    {"total_tokens": 128, "top_k": 8, "num_experts": 64, "block_size": 128},
    # Decode-shape (single token, multiple experts).
    {"total_tokens": 1, "top_k": 8, "num_experts": 64, "block_size": 64},
]

set_standard_shapes("moe", "align_block_size", _MOE_ALIGN_STANDARD_SHAPES)
set_benchmark_shapes("moe", "align_block_size", _MOE_ALIGN_STANDARD_SHAPES)

_MOE_APPLY_STANDARD_SHAPES: list[dict[str, int | str]] = [
    {
        "num_tokens": 1,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_experts": 4,
        "top_k": 2,
        "weight_dtype": "mxfp4",
        "activation": "silu",
        "internal_activation_dtype": "input",
    },
    {
        "num_tokens": 4,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_experts": 4,
        "top_k": 2,
        "weight_dtype": "mxfp4",
        "activation": "silu",
        "internal_activation_dtype": "input",
    },
]

set_standard_shapes("moe", "apply", _MOE_APPLY_STANDARD_SHAPES)
set_benchmark_shapes("moe", "apply", _MOE_APPLY_STANDARD_SHAPES)
set_standard_shapes("moe", "process_weights", _MOE_APPLY_STANDARD_SHAPES)
set_benchmark_shapes("moe", "process_weights", _MOE_APPLY_STANDARD_SHAPES)


def compute_align_block_size_buffer_dims(
    pad_id: int, num_experts: int, block_size: int
) -> tuple[int, int]:
    """Output buffer dims for moe_align_block_size, block-aligned.

    Returns ``(num_blocks, sorted_ids_size)`` where
    ``sorted_ids_size == num_blocks * block_size`` so the canonical reshape
    can use ``view(num_blocks, block_size)`` without padding.
    """
    max_num_tokens_padded = pad_id + (num_experts + 1) * (block_size - 1)
    num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    return num_blocks, num_blocks * block_size


def canonicalize_align_block_size(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Pack the three moe_align_block_size outputs into a single int32 tensor.

    Within each block the ``sorted_ids`` slot order is non-deterministic
    (CUDA atomics in the trtllm impl), so we sort each block before
    concatenating. The set of token IDs assigned to each block is what
    matters for downstream MoE GEMM correctness.

    Caller must size ``sorted_ids`` to ``expert_ids.numel() * block_size``.
    """
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sorted_ids.numel() != expert_ids.numel() * block_size:
        raise ValueError("sorted_ids size must equal expert_ids.numel() * block_size")
    blocks = sorted_ids.reshape(expert_ids.numel(), block_size)
    blocks_sorted, _ = blocks.sort(dim=-1)
    return torch.cat(
        (
            num_tokens_post_pad.flatten().to(torch.int32),
            expert_ids.flatten().to(torch.int32),
            blocks_sorted.flatten().to(torch.int32),
        )
    )
