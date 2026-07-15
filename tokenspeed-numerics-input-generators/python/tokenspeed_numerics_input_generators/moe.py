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

"""Input generators for the fundamental stages of routed MoE computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from tokenspeed_numerics_input_generators.core import (
    CustomDType,
    DeviceLike,
    InputDType,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _resolve_device,
)
from tokenspeed_numerics_input_generators.gemm import (
    GemmInputConfig,
    GemmInputs,
    GemmInputValues,
    gemm_scale_shape,
)

__all__ = [
    "MoeDispatchInputConfig",
    "MoeDispatchInputs",
    "MoeDispatchInputValues",
    "MoeDownCombineInputConfig",
    "MoeDownCombineInputs",
    "MoeDownCombineInputValues",
    "MoeGateUpInputConfig",
    "MoeGateUpInputs",
    "MoeGateUpInputValues",
    "MoeInputConfig",
    "MoeInputs",
    "MoeInputValues",
    "MoeRoutingInputConfig",
    "MoeRoutingInputs",
    "MoeRoutingInputValues",
]

MoeRoutingScoreFunction = Literal["softmax", "sigmoid", "softplus_sqrt"]

_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_INTEGER_DTYPES = {torch.int16, torch.int32, torch.int64}
_ROUTING_SCORE_FUNCTIONS = {"softmax", "sigmoid", "softplus_sqrt"}
_DEFAULT_BLOCK_SIZE = {
    CustomDType.MXFP4: 32,
    CustomDType.MXINT4: 32,
    CustomDType.NVFP4: 16,
}


def _check_nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_float_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{name} must be a regular floating torch dtype, got {dtype}")
    return dtype


def _check_optional_float_dtype(
    name: str,
    dtype: torch.dtype | None,
) -> torch.dtype | None:
    if dtype is None:
        return None
    return _check_float_dtype(name, dtype)


def _check_integer_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must be torch.int16, torch.int32, or torch.int64")
    return dtype


def _check_positive_finite_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _check_top_k(*, top_k: int, num_experts: int) -> tuple[int, int]:
    top_k = _check_positive("top_k", top_k)
    num_experts = _check_positive("num_experts", num_experts)
    if top_k > num_experts:
        raise ValueError("top_k must be <= num_experts")
    return top_k, num_experts


def _resolved_weight_dtype(
    weight_dtype: InputDType,
    activation_dtype: torch.dtype,
) -> InputDType:
    return activation_dtype if weight_dtype is None else weight_dtype


def _weight_block_size(
    weight_dtype: InputDType,
    weight_scale_dtype: InputDType,
    configured_block_size: int,
) -> int | None:
    if weight_dtype in _DEFAULT_BLOCK_SIZE:
        block_size = _DEFAULT_BLOCK_SIZE[weight_dtype]
    elif weight_dtype == torch.float4_e2m1fn_x2:
        block_size = 16
    elif weight_scale_dtype is not None:
        block_size = _check_positive("weight_scale_block_size", configured_block_size)
    else:
        return None
    return block_size


def _make_expert_weight_generator(
    *,
    num_rows: int,
    output_size: int,
    input_size: int,
    num_experts: int,
    activation_dtype: torch.dtype,
    weight_dtype: InputDType,
    weight_scale_dtype: InputDType,
    weight_scale_block_size: int,
    device: DeviceLike,
) -> GemmInputs:
    resolved_dtype = _resolved_weight_dtype(weight_dtype, activation_dtype)
    block_size = _weight_block_size(
        resolved_dtype,
        weight_scale_dtype,
        weight_scale_block_size,
    )
    if block_size is not None and input_size % block_size != 0:
        raise ValueError(
            "expert projection input size must be divisible by the weight scale "
            f"block size; got input_size={input_size}, block_size={block_size}"
        )
    scale_shape = (
        None
        if block_size is None
        else gemm_scale_shape(
            "block",
            "b",
            M=num_rows,
            N=output_size,
            K=input_size,
            batch_shape=(num_experts,),
            block_shape=(block_size,),
        )
    )
    return GemmInputs(
        GemmInputConfig(
            M=num_rows,
            N=output_size,
            K=input_size,
            a_dtype=None,
            b_dtype=resolved_dtype,
            c_dtype=activation_dtype,
            batch_shape=(num_experts,),
            b_scale_shape=scale_shape,
            b_scale_dtype=weight_scale_dtype,
            b_device=device,
            b_scale_device=device,
            c_device=device,
        )
    )


def _generate_routes(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    ids_dtype: torch.dtype,
    weights_dtype: torch.dtype,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    random_scores = torch.rand(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_ids = random_scores.topk(top_k, dim=-1, sorted=False).indices.to(ids_dtype)
    topk_weights = torch.rand(
        (num_tokens, top_k),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    if num_tokens:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.contiguous(), topk_weights.to(weights_dtype).contiguous()


def _routing_scores(
    logits: torch.Tensor,
    score_function: MoeRoutingScoreFunction,
) -> torch.Tensor:
    logits = logits.float()
    if score_function == "softmax":
        return torch.softmax(logits, dim=-1)
    if score_function == "sigmoid":
        return torch.sigmoid(logits)
    if score_function == "softplus_sqrt":
        return torch.sqrt(torch.nn.functional.softplus(logits))
    raise ValueError(f"unsupported score_function={score_function!r}")


def _select_routes(
    *,
    logits: torch.Tensor,
    correction_bias: torch.Tensor | None,
    score_function: MoeRoutingScoreFunction,
    top_k: int,
    num_expert_groups: int,
    top_k_groups: int,
    renormalize: bool,
    routed_scaling_factor: float,
    ids_dtype: torch.dtype,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _routing_scores(logits, score_function)
    selection_scores = scores
    if correction_bias is not None:
        selection_scores = selection_scores + correction_bias.float().reshape(1, -1)

    if num_expert_groups > 1:
        num_tokens, num_experts = selection_scores.shape
        experts_per_group = num_experts // num_expert_groups
        grouped = selection_scores.reshape(
            num_tokens,
            num_expert_groups,
            experts_per_group,
        )
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        selected_groups = group_scores.topk(
            top_k_groups,
            dim=-1,
            sorted=False,
        ).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, num_expert_groups, experts_per_group)
            .reshape(num_tokens, num_experts)
        )
        selection_scores = selection_scores.masked_fill(~expert_mask, float("-inf"))

    topk_ids = selection_scores.topk(top_k, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize and topk_weights.shape[0]:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
    topk_weights = topk_weights * routed_scaling_factor
    return (
        topk_ids.to(ids_dtype).contiguous(),
        topk_weights.to(weights_dtype).contiguous(),
    )


def _make_activation_scale(
    *,
    num_experts: int,
    dtype: torch.dtype | None,
    value: float,
    device: torch.device,
) -> torch.Tensor | None:
    if dtype is None:
        return None
    return torch.full((num_experts,), value, dtype=dtype, device=device)


def _validate_topk_values(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    expected_weight_sum: float | None,
) -> None:
    expected_shape = (num_tokens, top_k)
    if topk_ids.shape != expected_shape:
        raise ValueError(
            f"topk_ids must have shape {expected_shape}, got {tuple(topk_ids.shape)}"
        )
    if topk_weights.shape != expected_shape:
        raise ValueError(
            "topk_weights must have shape "
            f"{expected_shape}, got {tuple(topk_weights.shape)}"
        )
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("topk_ids must use an integer dtype")
    if topk_weights.dtype not in _FLOAT_DTYPES:
        raise ValueError("topk_weights must use a regular floating dtype")
    if topk_ids.numel():
        if topk_ids.min().item() < 0 or topk_ids.max().item() >= num_experts:
            raise ValueError("topk_ids must be in [0, num_experts)")
        sorted_ids = topk_ids.to(torch.int64).sort(dim=-1).values
        if top_k > 1 and torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("topk_ids must be unique within each token row")
    if not torch.isfinite(topk_weights).all():
        raise ValueError("topk_weights must be finite")
    if torch.any(topk_weights < 0):
        raise ValueError("topk_weights must be non-negative")
    if expected_weight_sum is not None and num_tokens:
        expected = torch.full(
            (num_tokens,),
            expected_weight_sum,
            dtype=torch.float32,
            device=topk_weights.device,
        )
        if not torch.allclose(
            topk_weights.float().sum(dim=-1),
            expected,
            rtol=1.0e-3,
            atol=1.0e-3,
        ):
            raise ValueError(f"topk_weights rows must sum to {expected_weight_sum}")


def _validate_expert_weight_values(
    *,
    name: str,
    values: GemmInputValues,
    generator: GemmInputs,
) -> None:
    if values.A is not None:
        raise ValueError(f"{name}.A must be omitted for expert weight generation")
    if values.B is None:
        raise ValueError(f"{name}.B expert weights are required")
    expected_shape = generator._value_shape("b")
    if values.B.shape != expected_shape:
        raise ValueError(
            f"{name}.B must have shape {expected_shape}, got {tuple(values.B.shape)}"
        )
    if values.A_scales is not None:
        raise ValueError(f"{name}.A_scales must be omitted")
    expected_scale_shape = generator.config.b_scale_shape
    if expected_scale_shape is None:
        if values.B_scales is not None:
            raise ValueError(f"{name}.B_scales must be omitted for unscaled weights")
    elif values.B_scales is None or values.B_scales.shape != expected_scale_shape:
        actual_shape = None if values.B_scales is None else tuple(values.B_scales.shape)
        raise ValueError(
            f"{name}.B_scales must have shape {expected_scale_shape}, got {actual_shape}"
        )


@dataclass
class MoeRoutingInputValues:
    """Generated values spanning router projection and top-k selection."""

    hidden_states: torch.Tensor
    router_weight: torch.Tensor
    router_bias: torch.Tensor | None
    router_logits: torch.Tensor
    correction_bias: torch.Tensor | None
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class MoeRoutingInputConfig:
    """Initialization parameters for router projection and top-k selection."""

    # Required: number of input token rows.
    num_tokens: int
    # Required: hidden-state width consumed by the router projection.
    hidden_size: int
    # Required: number of routed experts and router output columns.
    num_experts: int
    # Required: number of experts selected per token.
    top_k: int
    # Required: generated hidden-state dtype.
    hidden_dtype: torch.dtype

    # Optional: router weight, bias, and logits dtype.
    router_dtype: torch.dtype = torch.float32
    # Optional: generate a router projection bias with this dtype.
    router_bias_dtype: torch.dtype | None = None
    # Optional: transformation applied to router logits before selection.
    score_function: MoeRoutingScoreFunction = "softmax"
    # Optional: generate a correction bias used only for expert selection.
    correction_bias_dtype: torch.dtype | None = None
    # Optional: number of equal-size expert groups used for candidate filtering.
    num_expert_groups: int = 1
    # Optional: number of groups retained before expert-level top-k selection.
    top_k_groups: int = 1
    # Optional: normalize selected scores before applying the routing scale.
    renormalize: bool = True
    # Optional: positive scale applied to selected routing weights.
    routed_scaling_factor: float = 1.0
    # Optional: generated selected-expert ID dtype.
    topk_ids_dtype: torch.dtype = torch.int32
    # Optional: generated selected routing-weight dtype.
    topk_weights_dtype: torch.dtype = torch.float32
    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MoeRoutingInputs(NumericsInputGenerator):
    """Generate consistent router operands, logits, and selected routes."""

    config: MoeRoutingInputConfig
    hidden_states_input: TensorInput | None
    router_weight_input: TensorInput | None
    router_bias_input: TensorInput | None
    correction_bias_input: TensorInput | None

    def __init__(self, config: MoeRoutingInputConfig) -> None:
        self.config = config
        self.hidden_states_input = None
        self.router_weight_input = None
        self.router_bias_input = None
        self.correction_bias_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_size = _check_positive(
            "hidden_size", self.config.hidden_size
        )
        self.config.top_k, self.config.num_experts = _check_top_k(
            top_k=self.config.top_k,
            num_experts=self.config.num_experts,
        )
        self.config.hidden_dtype = _check_float_dtype(
            "hidden_dtype", self.config.hidden_dtype
        )
        self.config.router_dtype = _check_float_dtype(
            "router_dtype", self.config.router_dtype
        )
        self.config.router_bias_dtype = _check_optional_float_dtype(
            "router_bias_dtype", self.config.router_bias_dtype
        )
        self.config.correction_bias_dtype = _check_optional_float_dtype(
            "correction_bias_dtype", self.config.correction_bias_dtype
        )
        if self.config.score_function not in _ROUTING_SCORE_FUNCTIONS:
            raise ValueError(
                "score_function must be 'softmax', 'sigmoid', or 'softplus_sqrt'"
            )
        self.config.num_expert_groups = _check_positive(
            "num_expert_groups", self.config.num_expert_groups
        )
        self.config.top_k_groups = _check_positive(
            "top_k_groups", self.config.top_k_groups
        )
        if self.config.num_experts % self.config.num_expert_groups != 0:
            raise ValueError("num_experts must be divisible by num_expert_groups")
        if self.config.top_k_groups > self.config.num_expert_groups:
            raise ValueError("top_k_groups must be <= num_expert_groups")
        experts_per_group = self.config.num_experts // self.config.num_expert_groups
        if self.config.num_expert_groups > 1 and experts_per_group < 2:
            raise ValueError("grouped routing requires at least two experts per group")
        if self.config.top_k > self.config.top_k_groups * experts_per_group:
            raise ValueError("selected expert groups do not contain top_k experts")
        self.config.routed_scaling_factor = _check_positive_finite_float(
            "routed_scaling_factor", self.config.routed_scaling_factor
        )
        self.config.topk_ids_dtype = _check_integer_dtype(
            "topk_ids_dtype", self.config.topk_ids_dtype
        )
        self.config.topk_weights_dtype = _check_float_dtype(
            "topk_weights_dtype", self.config.topk_weights_dtype
        )

        self.hidden_states_input = self.hidden_states_input or TensorInput(
            (self.config.num_tokens, self.config.hidden_size),
            self.config.hidden_dtype,
            device=self.config.device,
        )
        self.router_weight_input = self.router_weight_input or TensorInput(
            (self.config.num_experts, self.config.hidden_size),
            self.config.router_dtype,
            device=self.config.device,
        )
        if self.config.router_bias_dtype is None:
            self.router_bias_input = None
        else:
            self.router_bias_input = self.router_bias_input or TensorInput(
                (self.config.num_experts,),
                self.config.router_bias_dtype,
                device=self.config.device,
            )
        if self.config.correction_bias_dtype is None:
            self.correction_bias_input = None
        else:
            self.correction_bias_input = self.correction_bias_input or TensorInput(
                (self.config.num_experts,),
                self.config.correction_bias_dtype,
                device=self.config.device,
            )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoeRoutingInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.hidden_states_input is None or self.router_weight_input is None:
            raise ValueError("routing tensor generators must be initialized")
        hidden_states = self.hidden_states_input.generate(
            seed=_child_seed(seed, 1), device=target_device
        ).values
        router_weight = self.router_weight_input.generate(
            seed=_child_seed(seed, 2), device=target_device
        ).values
        if hidden_states is None or router_weight is None:
            raise ValueError("routing operands cannot skip tensor generation")
        router_weight = (router_weight.float() / math.sqrt(self.config.hidden_size)).to(
            self.config.router_dtype
        )
        router_bias = (
            None
            if self.router_bias_input is None
            else self.router_bias_input.generate(
                seed=_child_seed(seed, 3), device=target_device
            ).values
        )
        correction_bias = (
            None
            if self.correction_bias_input is None
            else self.correction_bias_input.generate(
                seed=_child_seed(seed, 4), device=target_device
            ).values
        )
        if router_bias is not None:
            router_bias = (router_bias.float() * 0.25).to(router_bias.dtype)
        if correction_bias is not None:
            correction_bias = (correction_bias.float() * 0.25).to(correction_bias.dtype)

        router_logits = hidden_states.float() @ router_weight.float().transpose(0, 1)
        if router_bias is not None:
            router_logits += router_bias.float()
        router_logits = router_logits.to(self.config.router_dtype).contiguous()
        topk_ids, topk_weights = _select_routes(
            logits=router_logits,
            correction_bias=correction_bias,
            score_function=self.config.score_function,
            top_k=self.config.top_k,
            num_expert_groups=self.config.num_expert_groups,
            top_k_groups=self.config.top_k_groups,
            renormalize=self.config.renormalize,
            routed_scaling_factor=self.config.routed_scaling_factor,
            ids_dtype=self.config.topk_ids_dtype,
            weights_dtype=self.config.topk_weights_dtype,
        )
        values = MoeRoutingInputValues(
            hidden_states=hidden_states.contiguous(),
            router_weight=router_weight.contiguous(),
            router_bias=None if router_bias is None else router_bias.contiguous(),
            router_logits=router_logits,
            correction_bias=(
                None if correction_bias is None else correction_bias.contiguous()
            ),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        _validate_routing_values(values, self.config)
        return values


def _validate_routing_values(
    values: MoeRoutingInputValues,
    config: MoeRoutingInputConfig,
) -> None:
    if values.hidden_states.shape != (config.num_tokens, config.hidden_size):
        raise ValueError("hidden_states shape does not match routing configuration")
    if values.router_weight.shape != (config.num_experts, config.hidden_size):
        raise ValueError("router_weight shape does not match routing configuration")
    if values.router_logits.shape != (config.num_tokens, config.num_experts):
        raise ValueError("router_logits shape does not match routing configuration")
    if not torch.isfinite(values.router_logits).all():
        raise ValueError("router_logits must be finite")
    if values.router_bias is not None and values.router_bias.shape != (
        config.num_experts,
    ):
        raise ValueError("router_bias shape does not match routing configuration")
    if values.correction_bias is not None and values.correction_bias.shape != (
        config.num_experts,
    ):
        raise ValueError("correction_bias shape does not match routing configuration")
    _validate_topk_values(
        topk_ids=values.topk_ids,
        topk_weights=values.topk_weights,
        num_tokens=config.num_tokens,
        num_experts=config.num_experts,
        top_k=config.top_k,
        expected_weight_sum=(
            config.routed_scaling_factor if config.renormalize else None
        ),
    )


@dataclass
class MoeDispatchInputValues:
    """Generated values entering token-to-expert dispatch."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class MoeDispatchInputConfig:
    """Initialization parameters for token-to-expert dispatch inputs."""

    # Required: number of input token rows.
    num_tokens: int
    # Required: hidden-state width.
    hidden_size: int
    # Required: total number of routed experts.
    num_experts: int
    # Required: number of experts selected per token.
    top_k: int
    # Required: generated hidden-state dtype.
    hidden_dtype: torch.dtype

    # Optional: selected-expert ID dtype.
    topk_ids_dtype: torch.dtype = torch.int32
    # Optional: selected routing-weight dtype.
    topk_weights_dtype: torch.dtype = torch.float32
    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MoeDispatchInputs(NumericsInputGenerator):
    """Generate hidden states and consistent precomputed routing metadata."""

    config: MoeDispatchInputConfig
    hidden_states_input: TensorInput | None

    def __init__(self, config: MoeDispatchInputConfig) -> None:
        self.config = config
        self.hidden_states_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_size = _check_positive(
            "hidden_size", self.config.hidden_size
        )
        self.config.top_k, self.config.num_experts = _check_top_k(
            top_k=self.config.top_k,
            num_experts=self.config.num_experts,
        )
        self.config.hidden_dtype = _check_float_dtype(
            "hidden_dtype", self.config.hidden_dtype
        )
        self.config.topk_ids_dtype = _check_integer_dtype(
            "topk_ids_dtype", self.config.topk_ids_dtype
        )
        self.config.topk_weights_dtype = _check_float_dtype(
            "topk_weights_dtype", self.config.topk_weights_dtype
        )
        self.hidden_states_input = self.hidden_states_input or TensorInput(
            (self.config.num_tokens, self.config.hidden_size),
            self.config.hidden_dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoeDispatchInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.hidden_states_input is None:
            raise ValueError("hidden_states_input must be initialized")
        hidden_states = self.hidden_states_input.generate(
            seed=_child_seed(seed, 1), device=target_device
        ).values
        if hidden_states is None:
            raise ValueError("dispatch cannot skip hidden-state generation")
        topk_ids, topk_weights = _generate_routes(
            num_tokens=self.config.num_tokens,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            ids_dtype=self.config.topk_ids_dtype,
            weights_dtype=self.config.topk_weights_dtype,
            seed=_child_seed(seed, 2),
            device=target_device,
        )
        values = MoeDispatchInputValues(
            hidden_states=hidden_states.contiguous(),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        _validate_dispatch_values(values, self.config)
        return values


def _validate_dispatch_values(
    values: MoeDispatchInputValues,
    config: MoeDispatchInputConfig,
) -> None:
    if values.hidden_states.shape != (config.num_tokens, config.hidden_size):
        raise ValueError("hidden_states shape does not match dispatch configuration")
    if values.hidden_states.dtype not in _FLOAT_DTYPES:
        raise ValueError("hidden_states must use a regular floating dtype")
    if not torch.isfinite(values.hidden_states).all():
        raise ValueError("hidden_states must be finite")
    _validate_topk_values(
        topk_ids=values.topk_ids,
        topk_weights=values.topk_weights,
        num_tokens=config.num_tokens,
        num_experts=config.num_experts,
        top_k=config.top_k,
        expected_weight_sum=1.0,
    )


@dataclass
class MoeGateUpInputValues:
    """Generated operands entering the expert gate/up projection."""

    routed_hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    w13: GemmInputValues
    w13_bias: torch.Tensor | None
    activation_scale: torch.Tensor | None


@dataclass
class MoeGateUpInputConfig:
    """Initialization parameters for expert gate/up projection inputs."""

    # Required: number of routed token-expert rows.
    num_routed_tokens: int
    # Required: input hidden-state width.
    hidden_size: int
    # Required: width of each gate and up projection.
    intermediate_size: int
    # Required: total number of routed experts.
    num_experts: int
    # Required: routed activation dtype.
    activation_dtype: torch.dtype

    # Optional: expert weight dtype. None uses activation_dtype.
    weight_dtype: InputDType = None
    # Optional: expert weight scale dtype; custom dtypes infer their default.
    weight_scale_dtype: InputDType = None
    # Optional: scale block size for scaled regular torch weights.
    weight_scale_block_size: int = 32
    # Optional: generate one gate/up bias row per expert.
    bias_dtype: torch.dtype | None = None
    # Optional: generate one input scale per expert.
    activation_scale_dtype: torch.dtype | None = None
    # Optional: positive value used for each generated input scale.
    activation_scale: float = 0.125
    # Optional: routed expert ID dtype.
    expert_ids_dtype: torch.dtype = torch.int32
    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MoeGateUpInputs(NumericsInputGenerator):
    """Generate routed activations and expert W13 operands."""

    config: MoeGateUpInputConfig
    routed_hidden_states_input: TensorInput | None
    w13: GemmInputs | None
    w13_bias_input: TensorInput | None

    def __init__(self, config: MoeGateUpInputConfig) -> None:
        self.config = config
        self.routed_hidden_states_input = None
        self.w13 = None
        self.w13_bias_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_routed_tokens = _check_nonnegative(
            "num_routed_tokens", self.config.num_routed_tokens
        )
        self.config.hidden_size = _check_positive(
            "hidden_size", self.config.hidden_size
        )
        self.config.intermediate_size = _check_positive(
            "intermediate_size", self.config.intermediate_size
        )
        self.config.num_experts = _check_positive(
            "num_experts", self.config.num_experts
        )
        self.config.activation_dtype = _check_float_dtype(
            "activation_dtype", self.config.activation_dtype
        )
        self.config.bias_dtype = _check_optional_float_dtype(
            "bias_dtype", self.config.bias_dtype
        )
        self.config.activation_scale_dtype = _check_optional_float_dtype(
            "activation_scale_dtype", self.config.activation_scale_dtype
        )
        self.config.activation_scale = _check_positive_finite_float(
            "activation_scale", self.config.activation_scale
        )
        self.config.expert_ids_dtype = _check_integer_dtype(
            "expert_ids_dtype", self.config.expert_ids_dtype
        )
        self.routed_hidden_states_input = (
            self.routed_hidden_states_input
            or TensorInput(
                (self.config.num_routed_tokens, self.config.hidden_size),
                self.config.activation_dtype,
                device=self.config.device,
            )
        )
        self.w13 = self.w13 or _make_expert_weight_generator(
            num_rows=self.config.num_routed_tokens,
            output_size=2 * self.config.intermediate_size,
            input_size=self.config.hidden_size,
            num_experts=self.config.num_experts,
            activation_dtype=self.config.activation_dtype,
            weight_dtype=self.config.weight_dtype,
            weight_scale_dtype=self.config.weight_scale_dtype,
            weight_scale_block_size=self.config.weight_scale_block_size,
            device=self.config.device,
        )
        if self.config.bias_dtype is None:
            self.w13_bias_input = None
        else:
            self.w13_bias_input = self.w13_bias_input or TensorInput(
                (self.config.num_experts, 2 * self.config.intermediate_size),
                self.config.bias_dtype,
                device=self.config.device,
            )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoeGateUpInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.routed_hidden_states_input is None or self.w13 is None:
            raise ValueError("gate/up child generators must be initialized")
        routed_hidden_states = self.routed_hidden_states_input.generate(
            seed=_child_seed(seed, 1), device=target_device
        ).values
        if routed_hidden_states is None:
            raise ValueError("gate/up cannot skip routed activation generation")
        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(_child_seed(seed, 2))
        expert_ids = torch.randint(
            0,
            self.config.num_experts,
            (self.config.num_routed_tokens,),
            dtype=self.config.expert_ids_dtype,
            device=target_device,
            generator=generator,
        )
        w13_bias = (
            None
            if self.w13_bias_input is None
            else self.w13_bias_input.generate(
                seed=_child_seed(seed, 4), device=target_device
            ).values
        )
        values = MoeGateUpInputValues(
            routed_hidden_states=routed_hidden_states.contiguous(),
            expert_ids=expert_ids.contiguous(),
            w13=self.w13.generate(seed=_child_seed(seed, 3), device=target_device),
            w13_bias=None if w13_bias is None else w13_bias.contiguous(),
            activation_scale=_make_activation_scale(
                num_experts=self.config.num_experts,
                dtype=self.config.activation_scale_dtype,
                value=self.config.activation_scale,
                device=target_device,
            ),
        )
        _validate_gate_up_values(values, self)
        return values


def _validate_gate_up_values(
    values: MoeGateUpInputValues,
    generator: MoeGateUpInputs,
) -> None:
    config = generator.config
    if values.routed_hidden_states.shape != (
        config.num_routed_tokens,
        config.hidden_size,
    ):
        raise ValueError(
            "routed_hidden_states shape does not match gate/up configuration"
        )
    if values.routed_hidden_states.dtype not in _FLOAT_DTYPES:
        raise ValueError("routed_hidden_states must use a regular floating dtype")
    if not torch.isfinite(values.routed_hidden_states).all():
        raise ValueError("routed_hidden_states must be finite")
    if values.expert_ids.shape != (config.num_routed_tokens,):
        raise ValueError("expert_ids shape does not match gate/up configuration")
    if values.expert_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("expert_ids must use an integer dtype")
    if values.expert_ids.numel() and (
        values.expert_ids.min().item() < 0
        or values.expert_ids.max().item() >= config.num_experts
    ):
        raise ValueError("expert_ids must be in [0, num_experts)")
    if generator.w13 is None:
        raise ValueError("w13 generator must be initialized")
    _validate_expert_weight_values(
        name="w13",
        values=values.w13,
        generator=generator.w13,
    )
    if values.w13_bias is not None and values.w13_bias.shape != (
        config.num_experts,
        2 * config.intermediate_size,
    ):
        raise ValueError("w13_bias shape does not match gate/up configuration")
    if values.activation_scale is not None and values.activation_scale.shape != (
        config.num_experts,
    ):
        raise ValueError("activation_scale shape does not match expert count")


@dataclass
class MoeDownCombineInputValues:
    """Generated operands entering expert down projection and route reduction."""

    activations: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w2: GemmInputValues
    w2_bias: torch.Tensor | None
    activation_scale: torch.Tensor | None
    shared_output: torch.Tensor | None


@dataclass
class MoeDownCombineInputConfig:
    """Initialization parameters for down projection and route combination."""

    # Required: number of original token rows.
    num_tokens: int
    # Required: final hidden-state width.
    hidden_size: int
    # Required: activated expert intermediate width.
    intermediate_size: int
    # Required: total number of routed experts.
    num_experts: int
    # Required: selected experts per original token.
    top_k: int
    # Required: post-activation and output dtype.
    activation_dtype: torch.dtype

    # Optional: expert weight dtype. None uses activation_dtype.
    weight_dtype: InputDType = None
    # Optional: expert weight scale dtype; custom dtypes infer their default.
    weight_scale_dtype: InputDType = None
    # Optional: scale block size for scaled regular torch weights.
    weight_scale_block_size: int = 32
    # Optional: generate one down-projection bias row per expert.
    bias_dtype: torch.dtype | None = None
    # Optional: generate one down-projection input scale per expert.
    activation_scale_dtype: torch.dtype | None = None
    # Optional: positive value used for each generated input scale.
    activation_scale: float = 0.125
    # Optional: selected expert ID dtype.
    topk_ids_dtype: torch.dtype = torch.int32
    # Optional: selected route-weight dtype.
    topk_weights_dtype: torch.dtype = torch.float32
    # Optional: generate a dense shared-expert contribution for combination.
    include_shared_output: bool = False
    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MoeDownCombineInputs(NumericsInputGenerator):
    """Generate activated expert rows, W2 operands, and combination metadata."""

    config: MoeDownCombineInputConfig
    activations_input: TensorInput | None
    w2: GemmInputs | None
    w2_bias_input: TensorInput | None
    shared_output_input: TensorInput | None

    def __init__(self, config: MoeDownCombineInputConfig) -> None:
        self.config = config
        self.activations_input = None
        self.w2 = None
        self.w2_bias_input = None
        self.shared_output_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_size = _check_positive(
            "hidden_size", self.config.hidden_size
        )
        self.config.intermediate_size = _check_positive(
            "intermediate_size", self.config.intermediate_size
        )
        self.config.top_k, self.config.num_experts = _check_top_k(
            top_k=self.config.top_k,
            num_experts=self.config.num_experts,
        )
        self.config.activation_dtype = _check_float_dtype(
            "activation_dtype", self.config.activation_dtype
        )
        self.config.bias_dtype = _check_optional_float_dtype(
            "bias_dtype", self.config.bias_dtype
        )
        self.config.activation_scale_dtype = _check_optional_float_dtype(
            "activation_scale_dtype", self.config.activation_scale_dtype
        )
        self.config.activation_scale = _check_positive_finite_float(
            "activation_scale", self.config.activation_scale
        )
        self.config.topk_ids_dtype = _check_integer_dtype(
            "topk_ids_dtype", self.config.topk_ids_dtype
        )
        self.config.topk_weights_dtype = _check_float_dtype(
            "topk_weights_dtype", self.config.topk_weights_dtype
        )
        num_routes = self.config.num_tokens * self.config.top_k
        self.activations_input = self.activations_input or TensorInput(
            (num_routes, self.config.intermediate_size),
            self.config.activation_dtype,
            device=self.config.device,
        )
        self.w2 = self.w2 or _make_expert_weight_generator(
            num_rows=num_routes,
            output_size=self.config.hidden_size,
            input_size=self.config.intermediate_size,
            num_experts=self.config.num_experts,
            activation_dtype=self.config.activation_dtype,
            weight_dtype=self.config.weight_dtype,
            weight_scale_dtype=self.config.weight_scale_dtype,
            weight_scale_block_size=self.config.weight_scale_block_size,
            device=self.config.device,
        )
        if self.config.bias_dtype is None:
            self.w2_bias_input = None
        else:
            self.w2_bias_input = self.w2_bias_input or TensorInput(
                (self.config.num_experts, self.config.hidden_size),
                self.config.bias_dtype,
                device=self.config.device,
            )
        if self.config.include_shared_output:
            self.shared_output_input = self.shared_output_input or TensorInput(
                (self.config.num_tokens, self.config.hidden_size),
                self.config.activation_dtype,
                device=self.config.device,
            )
        else:
            self.shared_output_input = None

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoeDownCombineInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.activations_input is None or self.w2 is None:
            raise ValueError("down/combine child generators must be initialized")
        activations = self.activations_input.generate(
            seed=_child_seed(seed, 1), device=target_device
        ).values
        if activations is None:
            raise ValueError("down/combine cannot skip activation generation")
        topk_ids, topk_weights = _generate_routes(
            num_tokens=self.config.num_tokens,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            ids_dtype=self.config.topk_ids_dtype,
            weights_dtype=self.config.topk_weights_dtype,
            seed=_child_seed(seed, 2),
            device=target_device,
        )
        w2_bias = (
            None
            if self.w2_bias_input is None
            else self.w2_bias_input.generate(
                seed=_child_seed(seed, 4), device=target_device
            ).values
        )
        shared_output = (
            None
            if self.shared_output_input is None
            else self.shared_output_input.generate(
                seed=_child_seed(seed, 5), device=target_device
            ).values
        )
        values = MoeDownCombineInputValues(
            activations=activations.contiguous(),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w2=self.w2.generate(seed=_child_seed(seed, 3), device=target_device),
            w2_bias=None if w2_bias is None else w2_bias.contiguous(),
            activation_scale=_make_activation_scale(
                num_experts=self.config.num_experts,
                dtype=self.config.activation_scale_dtype,
                value=self.config.activation_scale,
                device=target_device,
            ),
            shared_output=(
                None if shared_output is None else shared_output.contiguous()
            ),
        )
        _validate_down_combine_values(values, self)
        return values


def _validate_down_combine_values(
    values: MoeDownCombineInputValues,
    generator: MoeDownCombineInputs,
) -> None:
    config = generator.config
    num_routes = config.num_tokens * config.top_k
    if values.activations.shape != (num_routes, config.intermediate_size):
        raise ValueError("activations shape does not match down/combine configuration")
    if values.activations.dtype not in _FLOAT_DTYPES:
        raise ValueError("activations must use a regular floating dtype")
    if not torch.isfinite(values.activations).all():
        raise ValueError("activations must be finite")
    _validate_topk_values(
        topk_ids=values.topk_ids,
        topk_weights=values.topk_weights,
        num_tokens=config.num_tokens,
        num_experts=config.num_experts,
        top_k=config.top_k,
        expected_weight_sum=1.0,
    )
    if generator.w2 is None:
        raise ValueError("w2 generator must be initialized")
    _validate_expert_weight_values(
        name="w2",
        values=values.w2,
        generator=generator.w2,
    )
    if values.w2_bias is not None and values.w2_bias.shape != (
        config.num_experts,
        config.hidden_size,
    ):
        raise ValueError("w2_bias shape does not match down/combine configuration")
    if values.activation_scale is not None and values.activation_scale.shape != (
        config.num_experts,
    ):
        raise ValueError("activation_scale shape does not match expert count")
    if values.shared_output is not None and values.shared_output.shape != (
        config.num_tokens,
        config.hidden_size,
    ):
        raise ValueError(
            "shared_output shape does not match down/combine configuration"
        )


@dataclass
class MoeInputValues:
    """Generated external operands for a full fused routed MoE computation."""

    routing: MoeRoutingInputValues
    w13: GemmInputValues
    w2: GemmInputValues
    w13_bias: torch.Tensor | None
    w2_bias: torch.Tensor | None
    w13_activation_scale: torch.Tensor | None
    w2_activation_scale: torch.Tensor | None


@dataclass
class MoeInputConfig:
    """Initialization parameters for full fused routed MoE inputs."""

    # Required: router projection and top-k configuration. This object is
    # shared directly with the nested MoeRoutingInputs generator.
    routing: MoeRoutingInputConfig
    # Required: expert intermediate width after SwiGLU activation.
    intermediate_size: int

    # Optional: W13 and W2 weight dtype. None uses routing.hidden_dtype.
    weight_dtype: InputDType = None
    # Optional: W13 and W2 scale dtype; custom dtypes infer their default.
    weight_scale_dtype: InputDType = None
    # Optional: scale block size for scaled regular torch weights.
    weight_scale_block_size: int = 32
    # Optional: generate W13 and W2 expert biases with this dtype.
    bias_dtype: torch.dtype | None = None
    # Optional: generate one input scale per expert for both projections.
    activation_scale_dtype: torch.dtype | None = None
    # Optional: positive value used for each W13 input scale.
    w13_activation_scale: float = 0.125
    # Optional: positive value used for each W2 input scale.
    w2_activation_scale: float = 0.125
    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MoeInputs(NumericsInputGenerator):
    """Generate all external operands needed by a full fused routed MoE."""

    config: MoeInputConfig
    routing: MoeRoutingInputs | None
    w13: GemmInputs | None
    w2: GemmInputs | None
    w13_bias_input: TensorInput | None
    w2_bias_input: TensorInput | None

    def __init__(self, config: MoeInputConfig) -> None:
        self.config = config
        self.routing = None
        self.w13 = None
        self.w2 = None
        self.w13_bias_input = None
        self.w2_bias_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.config.routing, MoeRoutingInputConfig):
            raise TypeError("routing must be a MoeRoutingInputConfig")
        self.config.intermediate_size = _check_positive(
            "intermediate_size", self.config.intermediate_size
        )
        self.config.bias_dtype = _check_optional_float_dtype(
            "bias_dtype", self.config.bias_dtype
        )
        self.config.activation_scale_dtype = _check_optional_float_dtype(
            "activation_scale_dtype", self.config.activation_scale_dtype
        )
        self.config.w13_activation_scale = _check_positive_finite_float(
            "w13_activation_scale", self.config.w13_activation_scale
        )
        self.config.w2_activation_scale = _check_positive_finite_float(
            "w2_activation_scale", self.config.w2_activation_scale
        )
        if self.routing is None:
            self.routing = MoeRoutingInputs(self.config.routing)
        else:
            self.routing.config = self.config.routing
            self.routing.__post_init__()

        self.w13 = self.w13 or _make_expert_weight_generator(
            num_rows=self.config.routing.num_tokens,
            output_size=2 * self.config.intermediate_size,
            input_size=self.config.routing.hidden_size,
            num_experts=self.config.routing.num_experts,
            activation_dtype=self.config.routing.hidden_dtype,
            weight_dtype=self.config.weight_dtype,
            weight_scale_dtype=self.config.weight_scale_dtype,
            weight_scale_block_size=self.config.weight_scale_block_size,
            device=self.config.device,
        )
        self.w2 = self.w2 or _make_expert_weight_generator(
            num_rows=self.config.routing.num_tokens,
            output_size=self.config.routing.hidden_size,
            input_size=self.config.intermediate_size,
            num_experts=self.config.routing.num_experts,
            activation_dtype=self.config.routing.hidden_dtype,
            weight_dtype=self.config.weight_dtype,
            weight_scale_dtype=self.config.weight_scale_dtype,
            weight_scale_block_size=self.config.weight_scale_block_size,
            device=self.config.device,
        )
        if self.config.bias_dtype is None:
            self.w13_bias_input = None
            self.w2_bias_input = None
        else:
            self.w13_bias_input = self.w13_bias_input or TensorInput(
                (
                    self.config.routing.num_experts,
                    2 * self.config.intermediate_size,
                ),
                self.config.bias_dtype,
                device=self.config.device,
            )
            self.w2_bias_input = self.w2_bias_input or TensorInput(
                (
                    self.config.routing.num_experts,
                    self.config.routing.hidden_size,
                ),
                self.config.bias_dtype,
                device=self.config.device,
            )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoeInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.routing is None or self.w13 is None or self.w2 is None:
            raise ValueError("full MoE child generators must be initialized")
        w13_bias = (
            None
            if self.w13_bias_input is None
            else self.w13_bias_input.generate(
                seed=_child_seed(seed, 4), device=target_device
            ).values
        )
        w2_bias = (
            None
            if self.w2_bias_input is None
            else self.w2_bias_input.generate(
                seed=_child_seed(seed, 5), device=target_device
            ).values
        )
        values = MoeInputValues(
            routing=self.routing.generate(
                seed=_child_seed(seed, 1), device=target_device
            ),
            w13=self.w13.generate(seed=_child_seed(seed, 2), device=target_device),
            w2=self.w2.generate(seed=_child_seed(seed, 3), device=target_device),
            w13_bias=None if w13_bias is None else w13_bias.contiguous(),
            w2_bias=None if w2_bias is None else w2_bias.contiguous(),
            w13_activation_scale=_make_activation_scale(
                num_experts=self.config.routing.num_experts,
                dtype=self.config.activation_scale_dtype,
                value=self.config.w13_activation_scale,
                device=target_device,
            ),
            w2_activation_scale=_make_activation_scale(
                num_experts=self.config.routing.num_experts,
                dtype=self.config.activation_scale_dtype,
                value=self.config.w2_activation_scale,
                device=target_device,
            ),
        )
        _validate_moe_values(values, self)
        return values


def _validate_moe_values(
    values: MoeInputValues,
    generator: MoeInputs,
) -> None:
    config = generator.config
    if generator.routing is None or generator.w13 is None or generator.w2 is None:
        raise ValueError("full MoE child generators must be initialized")
    _validate_routing_values(values.routing, generator.routing.config)
    _validate_expert_weight_values(
        name="w13",
        values=values.w13,
        generator=generator.w13,
    )
    _validate_expert_weight_values(
        name="w2",
        values=values.w2,
        generator=generator.w2,
    )
    if values.w13_bias is not None and values.w13_bias.shape != (
        config.routing.num_experts,
        2 * config.intermediate_size,
    ):
        raise ValueError("w13_bias shape does not match full MoE configuration")
    if values.w2_bias is not None and values.w2_bias.shape != (
        config.routing.num_experts,
        config.routing.hidden_size,
    ):
        raise ValueError("w2_bias shape does not match full MoE configuration")
    for name, scale in (
        ("w13_activation_scale", values.w13_activation_scale),
        ("w2_activation_scale", values.w2_activation_scale),
    ):
        if scale is not None and scale.shape != (config.routing.num_experts,):
            raise ValueError(f"{name} shape does not match expert count")
