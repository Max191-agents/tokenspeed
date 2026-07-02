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

"""MoE-family input generators for numerical correctness tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    _check_gemm_layout,
    _logical_operand,
    mxfp4_gemm_input_config,
    mxint4_gemm_input_config,
)

__all__ = [
    "MoeAlignBlockSizeInputConfig",
    "MoeAlignBlockSizeInputs",
    "MoeAlignBlockSizeInputValues",
    "MoeAlignBlockSizeReferenceValues",
    "MoESoftmaxTopKRoutingInputConfig",
    "MoESoftmaxTopKRoutingInputs",
    "MoESoftmaxTopKRoutingInputValues",
    "MoESoftmaxTopKRoutingReferenceValues",
    "MoEBiasedGroupedTopKInputConfig",
    "MoEBiasedGroupedTopKInputs",
    "MoEBiasedGroupedTopKInputValues",
    "MoEBiasedGroupedTopKReferenceValues",
    "MoEDeepSeekV4MegaMoEStagingInputConfig",
    "MoEDeepSeekV4MegaMoEStagingInputs",
    "MoEDeepSeekV4MegaMoEStagingInputValues",
    "MoEDeepSeekV4MegaMoEStagingReferenceValues",
    "MoEFinalizeFuseSharedInputConfig",
    "MoEFinalizeFuseSharedInputs",
    "MoEFinalizeFuseSharedInputValues",
    "MoESoftplusSqrtTopKRoutingInputConfig",
    "MoESoftplusSqrtTopKRoutingInputs",
    "MoESoftplusSqrtTopKRoutingInputValues",
    "MoESoftplusSqrtTopKRoutingReferenceValues",
    "MoeInputConfig",
    "MoeInputValues",
    "MoeInputs",
    "canonicalize_moe_align_block_size",
    "moe_align_block_size_buffer_dims",
    "moe_align_block_size_reference",
    "moe_biased_grouped_topk_reference",
    "moe_deepseek_v4_mega_moe_staging_reference",
    "moe_finalize_fuse_shared_reference",
    "moe_reference",
    "moe_softplus_sqrt_topk_routing_reference",
    "moe_softmax_topk_routing_reference",
]


_TOPK_ID_DTYPES = {torch.int16, torch.int32, torch.int64}
_REGULAR_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE = 128
_DEEPSEEK_V4_MEGAMOE_FP8_GROUP_SIZE = 32
_DEEPSEEK_V4_MEGAMOE_FP8_MAX = 448.0
_MOE_FINALIZE_MAX_TOPK = 64


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _check_moe_float_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _REGULAR_FLOAT_DTYPES:
        raise ValueError(f"{name} must be a regular floating torch dtype, got {dtype}")
    return dtype


def _check_positive_finite_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _check_nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass
class MoeAlignBlockSizeInputValues:
    """Generated values for ``MoeAlignBlockSizeInputs``."""

    topk_ids: torch.Tensor
    block_size: int
    num_experts: int


@dataclass
class MoeAlignBlockSizeReferenceValues:
    """Reference output values for MoE align-block-size metadata."""

    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_pad: torch.Tensor


@dataclass
class MoESoftmaxTopKRoutingInputValues:
    """Generated values for softmax/bias top-k MoE routing."""

    logits: torch.Tensor
    correction_bias: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    num_experts_real: int
    scaling_factor: float
    renormalize: bool


@dataclass
class MoESoftmaxTopKRoutingReferenceValues:
    """Reference outputs for softmax/bias top-k MoE routing."""

    topk_indices: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class MoEBiasedGroupedTopKInputValues:
    """Generated values for sigmoid/bias grouped top-k MoE routing."""

    hidden_states: torch.Tensor
    gating_output: torch.Tensor
    correction_bias: torch.Tensor
    top_k: int
    renormalize: bool
    num_expert_groups: int
    top_k_groups: int
    routed_scaling_factor: float
    logical_to_physical_map: torch.Tensor | None
    num_token_non_padded: torch.Tensor | None


@dataclass
class MoEBiasedGroupedTopKReferenceValues:
    """Reference outputs for sigmoid/bias grouped top-k MoE routing."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


@dataclass
class MoEDeepSeekV4MegaMoEStagingInputValues:
    """Generated values for DeepSeek V4 MegaMoE input staging."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    x_fp8: torch.Tensor
    x_sf: torch.Tensor
    topk_idx_out: torch.Tensor
    topk_weights_out: torch.Tensor
    num_experts: int


@dataclass
class MoEDeepSeekV4MegaMoEStagingReferenceValues:
    """Reference outputs for DeepSeek V4 MegaMoE input staging."""

    x_fp8: torch.Tensor
    x_sf: torch.Tensor
    topk_idx_out: torch.Tensor
    topk_weights_out: torch.Tensor


@dataclass
class MoEFinalizeFuseSharedInputValues:
    """Generated values for MoE routed-output finalization.

    ``gemm2_out`` contains permuted expert down-projection outputs.
    ``expanded_idx_to_permuted_idx`` maps each flattened token/top-k slot to a
    row in ``gemm2_out``; ``-1`` drops the slot. ``expert_weights`` contains
    route weights for each token/top-k slot. ``shared_output`` is an optional
    per-token residual that is added after routed expert accumulation.
    """

    gemm2_out: torch.Tensor
    expanded_idx_to_permuted_idx: torch.Tensor
    expert_weights: torch.Tensor
    shared_output: torch.Tensor | None


@dataclass
class MoEFinalizeFuseSharedInputConfig:
    """Initialization parameters for MoE finalize + shared residual inputs.

    The represented operation computes, for each token ``t``:

    ``sum_k expert_weights[t, k] * gemm2_out[permuted_idx(t, k)]``

    and optionally adds ``shared_output[t]``. Dropped top-k slots use permute
    index ``-1`` and do not contribute to the sum.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows to finalize. Zero is valid for idle ranks.
    num_tokens: int

    # Required: logical output hidden width.
    hidden_size: int

    # Required: number of routed expert outputs combined per token.
    top_k: int

    # ------------------------------------------------------------------
    # Optional shape and routing metadata configuration.
    # ------------------------------------------------------------------

    # Optional: physical width of rows in gemm2_out. This may be larger than
    # hidden_size when the expert GEMM pads the hidden dimension.
    hidden_size_padded: int | None = None

    # Optional: physical number of rows in gemm2_out. When omitted, the
    # generator uses the number of non-dropped token/top-k slots.
    total_num_padded_tokens: int | None = None

    # Optional: number of flattened token/top-k slots marked as dropped with
    # permute index -1.
    num_dropped_slots: int = 0

    # Optional: generate shared_output and make the logical output width equal
    # hidden_size. When false, the output width is hidden_size_padded.
    include_shared_output: bool = True

    # Optional: route-weight dtype. TokenSpeed's CUDA helper supports FP32 and
    # BF16 expert weights.
    expert_weights_dtype: torch.dtype = torch.float32

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class MoESoftplusSqrtTopKRoutingInputValues:
    """Generated values for softplus-sqrt top-k MoE routing."""

    logits: torch.Tensor
    correction_bias: torch.Tensor | None
    input_ids: torch.Tensor | None
    hash_indices_table: torch.Tensor | None
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    renormalize: bool
    routed_scaling_factor: float


@dataclass
class MoESoftplusSqrtTopKRoutingReferenceValues:
    """Reference outputs for softplus-sqrt top-k MoE routing."""

    topk_indices: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class MoeAlignBlockSizeInputConfig:
    """Initialization parameters for ``MoeAlignBlockSizeInputs``.

    The represented operation takes top-k routing ids for a routed MoE layer,
    groups flattened token-expert slots by selected expert, and pads each
    expert's group to ``block_size`` slots so block GEMM launches can process
    expert-local token groups.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of input tokens before top-k routing expansion.
    total_tokens: int

    # Required: number of selected experts per token.
    top_k: int

    # Required: total number of routed experts.
    num_experts: int

    # Required: GEMM block size used to pad each expert's token slots.
    block_size: int

    # ------------------------------------------------------------------
    # Optional generation configuration.
    # ------------------------------------------------------------------

    # Optional: dtype for generated top-k ids. Integer dtypes are supported.
    topk_ids_dtype: torch.dtype = torch.int32

    # Optional: generation device for top-k ids.
    device: DeviceLike = None


@dataclass
class MoESoftmaxTopKRoutingInputConfig:
    """Initialization parameters for softmax/bias top-k MoE routing.

    The represented operation computes probabilities from router logits,
    selects experts by adding a correction bias to those probabilities, and
    emits selected expert ids plus route weights from the original softmax
    probabilities. Expert ids greater than or equal to ``num_experts_real`` are
    masked to ``-1`` to model padded zero experts.
    """

    # Required: number of token rows to route.
    num_tokens: int

    # Required: total experts including padded/zero experts.
    num_experts: int

    # Required: number of real experts before the padded zero-expert range.
    num_experts_real: int

    # Required: number of selected experts per token.
    top_k: int

    # Optional: generated output index dtype.
    topk_indices_dtype: torch.dtype = torch.int32

    # Optional: multiply selected route weights by this positive scalar.
    scaling_factor: float = 1.0

    # Optional: renormalize selected probabilities before scaling.
    renormalize: bool = False

    # Optional: bias at least one padded expert into the selected top-k set.
    include_zero_expert_selection: bool = True

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class MoEBiasedGroupedTopKInputConfig:
    """Initialization parameters for sigmoid/bias grouped top-k MoE routing.

    The represented operation computes sigmoid router scores, selects candidate
    expert groups using correction-biased scores, then selects top-k experts
    from those groups. Output weights come from the original sigmoid scores.
    """

    # Required: number of token rows to route.
    num_tokens: int

    # Required: hidden-state width associated with the routed token rows.
    hidden_size: int

    # Required: total number of routed experts.
    num_experts: int

    # Required: number of selected experts per token.
    top_k: int

    # Optional: number of equal-size expert groups used for candidate filtering.
    num_expert_groups: int = 1

    # Optional: number of expert groups selected before expert-level top-k.
    top_k_groups: int = 1

    # Optional: generated hidden-state dtype. Hidden values identify token rows
    # for implementation APIs but do not affect the routing scores.
    hidden_dtype: torch.dtype = torch.float32

    # Optional: generated router-logit dtype.
    router_dtype: torch.dtype = torch.float32

    # Optional: renormalize selected sigmoid weights before applying the scale.
    renormalize: bool = False

    # Optional: positive scale applied only when ``renormalize`` is enabled.
    routed_scaling_factor: float = 1.0

    # Optional: generate a logical-to-physical expert id permutation.
    use_logical_to_physical_map: bool = False

    # Optional: if set, rows at or after this token count get output ids ``-1``.
    num_token_non_padded: int | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class MoEDeepSeekV4MegaMoEStagingInputConfig:
    """Initialization parameters for DeepSeek V4 MegaMoE input staging.

    The represented operation prepares routed MoE inputs by quantizing
    hidden-state rows to FP8 in 128-wide blocks, packing four 32-wide scale
    exponent groups per block, and copying precomputed top-k routing tensors
    into staging outputs.
    """

    # Required: number of token rows to stage.
    num_tokens: int

    # Required: hidden-state width. Must be a multiple of 128.
    hidden_size: int

    # Required: total routed experts used to generate valid top-k ids.
    num_experts: int

    # Required: number of routed experts per token.
    top_k: int

    # Optional: generated hidden-state dtype.
    hidden_dtype: torch.dtype = torch.bfloat16

    # Optional: dtype for input and staged top-k ids.
    topk_ids_dtype: torch.dtype = torch.int32

    # Optional: dtype for staged FP8 hidden states.
    fp8_dtype: torch.dtype = torch.float8_e4m3fn

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class MoESoftplusSqrtTopKRoutingInputConfig:
    """Initialization parameters for softplus-sqrt top-k MoE routing.

    The represented operation transforms router logits with
    ``sqrt(softplus(x))`` and produces normalized top-k routing weights. In
    non-hash mode, top-k experts are selected from transformed scores plus a
    correction bias. In hash mode, selected experts come from an input-id keyed
    hash table.
    """

    # Required: number of token rows to route.
    num_tokens: int

    # Required: number of experts in the router logit row.
    num_experts: int

    # Optional: number of selected experts per token. TokenSpeed's CUDA helper
    # is specialized for six, but the operation-level reference is generic.
    top_k: int = 6

    # Optional: use token-id keyed hash routing instead of correction-bias top-k.
    use_hash_table: bool = False

    # Optional: hash-table row count when ``use_hash_table`` is enabled.
    hash_table_size: int | None = None

    # Optional: generated input-id dtype for hash routing.
    input_ids_dtype: torch.dtype = torch.int64

    # Optional: this operation always normalizes selected weights.
    renormalize: bool = True

    # Optional: positive scale applied after normalizing selected weights.
    routed_scaling_factor: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MoeAlignBlockSizeInputs(NumericsInputGenerator):
    """Input generator for MoE expert block-alignment metadata."""

    config: MoeAlignBlockSizeInputConfig

    def __init__(self, config: MoeAlignBlockSizeInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.total_tokens = int(self.config.total_tokens)
        self.config.top_k = int(self.config.top_k)
        self.config.num_experts = int(self.config.num_experts)
        self.config.block_size = int(self.config.block_size)
        if self.config.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.config.top_k > self.config.num_experts:
            raise ValueError("top_k must be <= num_experts")
        if self.config.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.config.topk_ids_dtype not in _TOPK_ID_DTYPES:
            raise ValueError("topk_ids_dtype must be an integer dtype")

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoeAlignBlockSizeInputValues:
        target_device = _resolve_device(self.config.device, device)
        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        topk_ids = torch.randint(
            0,
            self.config.num_experts,
            (self.config.total_tokens, self.config.top_k),
            dtype=self.config.topk_ids_dtype,
            device=target_device,
            generator=generator,
        )
        return MoeAlignBlockSizeInputValues(
            topk_ids=topk_ids,
            block_size=self.config.block_size,
            num_experts=self.config.num_experts,
        )


@dataclass(init=False)
class MoESoftmaxTopKRoutingInputs(NumericsInputGenerator):
    """Input generator for softmax/bias top-k MoE routing."""

    config: MoESoftmaxTopKRoutingInputConfig

    def __init__(self, config: MoESoftmaxTopKRoutingInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_experts = int(self.config.num_experts)
        self.config.num_experts_real = int(self.config.num_experts_real)
        self.config.top_k = int(self.config.top_k)
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.config.num_experts_real <= 0:
            raise ValueError("num_experts_real must be positive")
        if self.config.num_experts_real >= self.config.num_experts:
            raise ValueError("num_experts_real must be < num_experts")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.top_k > self.config.num_experts:
            raise ValueError("top_k must be <= num_experts")
        if self.config.topk_indices_dtype not in (torch.int32, torch.int64):
            raise ValueError("topk_indices_dtype must be torch.int32 or torch.int64")
        self.config.scaling_factor = _check_positive_finite_float(
            "scaling_factor", self.config.scaling_factor
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoESoftmaxTopKRoutingInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        logits = (
            torch.randn(
                (self.config.num_tokens, self.config.num_experts),
                dtype=torch.float32,
                device=target_device,
                generator=generator,
            )
            * 0.25
        )
        correction_bias = torch.zeros(
            (self.config.num_experts,),
            dtype=torch.float32,
            device=target_device,
        )
        preferred = self._preferred_experts()
        for rank, expert in enumerate(preferred):
            correction_bias[expert] = float(len(preferred) - rank)
        topk_indices = torch.full(
            (self.config.num_tokens, self.config.top_k),
            -1,
            dtype=self.config.topk_indices_dtype,
            device=target_device,
        )
        topk_weights = torch.zeros(
            (self.config.num_tokens, self.config.top_k),
            dtype=torch.float32,
            device=target_device,
        )
        values = MoESoftmaxTopKRoutingInputValues(
            logits=logits.contiguous(),
            correction_bias=correction_bias.contiguous(),
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            num_experts_real=self.config.num_experts_real,
            scaling_factor=self.config.scaling_factor,
            renormalize=self.config.renormalize,
        )
        _validate_moe_softmax_topk_routing_values(values)
        return values

    def _preferred_experts(self) -> list[int]:
        candidates: list[int] = []
        if self.config.include_zero_expert_selection:
            candidates.append(self.config.num_experts_real)
        candidates.extend(range(self.config.num_experts_real))
        candidates.extend(
            range(self.config.num_experts_real + 1, self.config.num_experts)
        )
        return candidates[: self.config.top_k]


@dataclass(init=False)
class MoEBiasedGroupedTopKInputs(NumericsInputGenerator):
    """Input generator for sigmoid/bias grouped top-k MoE routing."""

    config: MoEBiasedGroupedTopKInputConfig
    hidden_states_input: TensorInput | None
    gating_output_input: TensorInput | None

    def __init__(self, config: MoEBiasedGroupedTopKInputConfig) -> None:
        self.config = config
        self.hidden_states_input = None
        self.gating_output_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_size = int(self.config.hidden_size)
        self.config.num_experts = int(self.config.num_experts)
        self.config.top_k = int(self.config.top_k)
        self.config.num_expert_groups = int(self.config.num_expert_groups)
        self.config.top_k_groups = int(self.config.top_k_groups)
        if self.config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.num_expert_groups <= 0:
            raise ValueError("num_expert_groups must be positive")
        if self.config.num_experts % self.config.num_expert_groups != 0:
            raise ValueError("num_experts must be divisible by num_expert_groups")
        experts_per_group = self.config.num_experts // self.config.num_expert_groups
        if experts_per_group < 2:
            raise ValueError("each expert group must contain at least two experts")
        if self.config.top_k_groups <= 0:
            raise ValueError("top_k_groups must be positive")
        if self.config.top_k_groups > self.config.num_expert_groups:
            raise ValueError("top_k_groups must be <= num_expert_groups")
        if self.config.top_k > self.config.top_k_groups * experts_per_group:
            raise ValueError("top_k must fit within the experts in the selected groups")
        self.config.hidden_dtype = _check_moe_float_dtype(
            "hidden_dtype", self.config.hidden_dtype
        )
        self.config.router_dtype = _check_moe_float_dtype(
            "router_dtype", self.config.router_dtype
        )
        self.config.routed_scaling_factor = _check_positive_finite_float(
            "routed_scaling_factor", self.config.routed_scaling_factor
        )
        if self.config.num_token_non_padded is not None:
            self.config.num_token_non_padded = int(self.config.num_token_non_padded)
            if not 0 <= self.config.num_token_non_padded <= self.config.num_tokens:
                raise ValueError(
                    "num_token_non_padded must be in [0, num_tokens] when set"
                )
        if self.hidden_states_input is None:
            self.hidden_states_input = TensorInput(
                (self.config.num_tokens, self.config.hidden_size),
                self.config.hidden_dtype,
                device=self.config.device,
            )
        if self.gating_output_input is None:
            self.gating_output_input = TensorInput(
                (self.config.num_tokens, self.config.num_experts),
                self.config.router_dtype,
                device=self.config.device,
            )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoEBiasedGroupedTopKInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.hidden_states_input is None or self.gating_output_input is None:
            raise ValueError("MoEBiasedGroupedTopKInputs child generators are missing")

        hidden_states = self.hidden_states_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        ).values
        gating_output = self.gating_output_input.generate(
            seed=_child_seed(seed, 2),
            device=target_device,
        ).values
        if hidden_states is None:
            raise ValueError("hidden_states generation was skipped")
        if gating_output is None:
            raise ValueError("gating_output generation was skipped")

        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(_child_seed(seed, 3))
        correction_bias = (
            torch.randn(
                (self.config.num_experts,),
                dtype=torch.float32,
                device=target_device,
                generator=generator,
            )
            * 0.25
        )
        logical_to_physical_map = None
        if self.config.use_logical_to_physical_map:
            logical_to_physical_map = torch.randperm(
                self.config.num_experts,
                dtype=torch.int32,
                device=target_device,
                generator=generator,
            )
        num_token_non_padded = None
        if self.config.num_token_non_padded is not None:
            num_token_non_padded = torch.tensor(
                self.config.num_token_non_padded,
                dtype=torch.int32,
                device=target_device,
            )
        values = MoEBiasedGroupedTopKInputValues(
            hidden_states=hidden_states.contiguous(),
            gating_output=gating_output.contiguous(),
            correction_bias=correction_bias.contiguous(),
            top_k=self.config.top_k,
            renormalize=self.config.renormalize,
            num_expert_groups=self.config.num_expert_groups,
            top_k_groups=self.config.top_k_groups,
            routed_scaling_factor=self.config.routed_scaling_factor,
            logical_to_physical_map=logical_to_physical_map,
            num_token_non_padded=num_token_non_padded,
        )
        _validate_moe_biased_grouped_topk_values(values)
        return values


@dataclass(init=False)
class MoEDeepSeekV4MegaMoEStagingInputs(NumericsInputGenerator):
    """Input generator for DeepSeek V4 MegaMoE input staging."""

    config: MoEDeepSeekV4MegaMoEStagingInputConfig
    hidden_states_input: TensorInput | None

    def __init__(self, config: MoEDeepSeekV4MegaMoEStagingInputConfig) -> None:
        self.config = config
        self.hidden_states_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_size = int(self.config.hidden_size)
        self.config.num_experts = int(self.config.num_experts)
        self.config.top_k = int(self.config.top_k)
        if self.config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.config.hidden_size % _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE != 0:
            raise ValueError("hidden_size must be a multiple of 128")
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.top_k > self.config.num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.config.hidden_dtype = _check_moe_float_dtype(
            "hidden_dtype", self.config.hidden_dtype
        )
        if self.config.topk_ids_dtype not in _TOPK_ID_DTYPES:
            raise ValueError("topk_ids_dtype must be an integer dtype")
        if self.config.fp8_dtype != torch.float8_e4m3fn:
            raise ValueError("fp8_dtype must be torch.float8_e4m3fn")
        if self.hidden_states_input is None:
            self.hidden_states_input = TensorInput(
                (self.config.num_tokens, self.config.hidden_size),
                self.config.hidden_dtype,
                device=self.config.device,
            )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoEDeepSeekV4MegaMoEStagingInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.hidden_states_input is None:
            raise ValueError("hidden_states_input must be initialized")
        hidden_states = self.hidden_states_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        ).values
        if hidden_states is None:
            raise ValueError("hidden_states generation was skipped")
        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(_child_seed(seed, 2))
        raw_ids = torch.rand(
            (self.config.num_tokens, self.config.num_experts),
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        ).argsort(dim=-1)[:, : self.config.top_k]
        topk_ids = raw_ids.to(self.config.topk_ids_dtype)
        raw_weights = torch.rand(
            (self.config.num_tokens, self.config.top_k),
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        )
        topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        block_count = self.config.hidden_size // _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE
        values = MoEDeepSeekV4MegaMoEStagingInputValues(
            hidden_states=hidden_states.contiguous(),
            topk_ids=topk_ids.contiguous(),
            topk_weights=topk_weights.contiguous(),
            x_fp8=torch.empty(
                (self.config.num_tokens, self.config.hidden_size),
                dtype=self.config.fp8_dtype,
                device=target_device,
            ),
            x_sf=torch.empty(
                (self.config.num_tokens, block_count),
                dtype=torch.int32,
                device=target_device,
            ),
            topk_idx_out=torch.empty_like(topk_ids),
            topk_weights_out=torch.empty_like(topk_weights),
            num_experts=self.config.num_experts,
        )
        _validate_moe_deepseek_v4_mega_moe_staging_values(values)
        return values


@dataclass(init=False)
class MoEFinalizeFuseSharedInputs(NumericsInputGenerator):
    """Input generator for MoE routed-output finalization."""

    config: MoEFinalizeFuseSharedInputConfig
    gemm2_out_input: TensorInput | None
    shared_output_input: TensorInput | None

    def __init__(self, config: MoEFinalizeFuseSharedInputConfig) -> None:
        self.config = config
        self.gemm2_out_input = None
        self.shared_output_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_size = int(self.config.hidden_size)
        self.config.top_k = int(self.config.top_k)
        self.config.num_dropped_slots = int(self.config.num_dropped_slots)
        if self.config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.top_k > _MOE_FINALIZE_MAX_TOPK:
            raise ValueError(f"top_k must be <= {_MOE_FINALIZE_MAX_TOPK}")
        total_slots = self.config.num_tokens * self.config.top_k
        if not 0 <= self.config.num_dropped_slots <= total_slots:
            raise ValueError("num_dropped_slots must be in [0, num_tokens * top_k]")
        if self.config.hidden_size_padded is not None:
            self.config.hidden_size_padded = int(self.config.hidden_size_padded)
            if self.config.hidden_size_padded < self.config.hidden_size:
                raise ValueError("hidden_size_padded must be >= hidden_size")
        if self.config.total_num_padded_tokens is not None:
            self.config.total_num_padded_tokens = int(
                self.config.total_num_padded_tokens
            )
            if self.config.total_num_padded_tokens < self._active_slot_count():
                raise ValueError(
                    "total_num_padded_tokens must be at least the number of "
                    "non-dropped token/top-k slots"
                )
        if self.config.expert_weights_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "expert_weights_dtype must be torch.float32 or torch.bfloat16"
            )
        self.gemm2_out_input = self.gemm2_out_input or TensorInput(
            (self._total_num_padded_tokens(), self._hidden_size_padded()),
            torch.bfloat16,
            device=self.config.device,
        )
        self.gemm2_out_input.shape = (
            self._total_num_padded_tokens(),
            self._hidden_size_padded(),
        )
        self.gemm2_out_input.dtype = torch.bfloat16
        self.gemm2_out_input.device = self.config.device
        if self.config.include_shared_output:
            self.shared_output_input = self.shared_output_input or TensorInput(
                (self.config.num_tokens, self.config.hidden_size),
                torch.bfloat16,
                device=self.config.device,
            )
            self.shared_output_input.shape = (
                self.config.num_tokens,
                self.config.hidden_size,
            )
            self.shared_output_input.dtype = torch.bfloat16
            self.shared_output_input.device = self.config.device
        else:
            self.shared_output_input = None

    def _hidden_size_padded(self) -> int:
        return self.config.hidden_size_padded or self.config.hidden_size

    def _active_slot_count(self) -> int:
        return (
            self.config.num_tokens * self.config.top_k - self.config.num_dropped_slots
        )

    def _total_num_padded_tokens(self) -> int:
        if self.config.total_num_padded_tokens is not None:
            return self.config.total_num_padded_tokens
        return self._active_slot_count()

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoEFinalizeFuseSharedInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.gemm2_out_input is None:
            raise ValueError("gemm2_out_input must be initialized")
        gemm2_out = self.gemm2_out_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        ).values
        if gemm2_out is None:
            raise ValueError("gemm2_out generation was skipped")

        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(_child_seed(seed, 2))
        total_slots = self.config.num_tokens * self.config.top_k
        expanded_idx_to_permuted_idx = torch.full(
            (total_slots,),
            -1,
            dtype=torch.int32,
            device=target_device,
        )
        if total_slots > 0 and self._active_slot_count() > 0:
            slot_order = torch.randperm(
                total_slots,
                dtype=torch.int64,
                device=target_device,
                generator=generator,
            )
            active_slots = slot_order[self.config.num_dropped_slots :]
            permuted_rows = torch.randperm(
                self._total_num_padded_tokens(),
                dtype=torch.int32,
                device=target_device,
                generator=generator,
            )[: self._active_slot_count()]
            expanded_idx_to_permuted_idx[active_slots] = permuted_rows

        raw_weights = torch.rand(
            (self.config.num_tokens, self.config.top_k),
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        )
        expert_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        expert_weights = expert_weights.to(self.config.expert_weights_dtype)

        shared_output = None
        if self.shared_output_input is not None:
            shared_output = self.shared_output_input.generate(
                seed=_child_seed(seed, 3),
                device=target_device,
            ).values
            if shared_output is None:
                raise ValueError("shared_output generation was skipped")

        values = MoEFinalizeFuseSharedInputValues(
            gemm2_out=gemm2_out.contiguous(),
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx.contiguous(),
            expert_weights=expert_weights.contiguous(),
            shared_output=None if shared_output is None else shared_output.contiguous(),
        )
        _validate_moe_finalize_fuse_shared_values(values)
        return values


@dataclass(init=False)
class MoESoftplusSqrtTopKRoutingInputs(NumericsInputGenerator):
    """Input generator for softplus-sqrt top-k MoE routing."""

    config: MoESoftplusSqrtTopKRoutingInputConfig

    def __init__(self, config: MoESoftplusSqrtTopKRoutingInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_experts = int(self.config.num_experts)
        self.config.top_k = int(self.config.top_k)
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.top_k > self.config.num_experts:
            raise ValueError("top_k must be <= num_experts")
        if not self.config.renormalize:
            raise ValueError("softplus-sqrt routing requires renormalize=True")
        self.config.routed_scaling_factor = _check_positive_finite_float(
            "routed_scaling_factor", self.config.routed_scaling_factor
        )
        if self.config.input_ids_dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids_dtype must be torch.int32 or torch.int64")
        if self.config.hash_table_size is not None:
            self.config.hash_table_size = int(self.config.hash_table_size)
            if self.config.hash_table_size <= 0:
                raise ValueError("hash_table_size must be positive when set")

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MoESoftplusSqrtTopKRoutingInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        logits = (
            torch.randn(
                (self.config.num_tokens, self.config.num_experts),
                dtype=torch.float32,
                device=target_device,
                generator=generator,
            )
            * 0.5
        )
        topk_indices = torch.empty(
            (self.config.num_tokens, self.config.top_k),
            dtype=torch.int32,
            device=target_device,
        )
        topk_weights = torch.empty(
            (self.config.num_tokens, self.config.top_k),
            dtype=torch.float32,
            device=target_device,
        )
        correction_bias = None
        input_ids = None
        hash_indices_table = None
        if self.config.use_hash_table:
            table_size = self.config.hash_table_size or max(self.config.num_tokens, 1)
            input_ids = torch.randint(
                0,
                table_size,
                (self.config.num_tokens,),
                dtype=self.config.input_ids_dtype,
                device=target_device,
                generator=generator,
            )
            table_rows = [
                torch.randperm(
                    self.config.num_experts,
                    dtype=torch.int32,
                    device=target_device,
                    generator=generator,
                )[: self.config.top_k]
                for _ in range(table_size)
            ]
            hash_indices_table = (
                torch.stack(table_rows)
                if table_rows
                else torch.empty(
                    (0, self.config.top_k),
                    dtype=torch.int32,
                    device=target_device,
                )
            )
        else:
            correction_bias = (
                torch.randn(
                    (self.config.num_experts,),
                    dtype=torch.float32,
                    device=target_device,
                    generator=generator,
                )
                * 0.25
            )
        values = MoESoftplusSqrtTopKRoutingInputValues(
            logits=logits.contiguous(),
            correction_bias=(
                None if correction_bias is None else correction_bias.contiguous()
            ),
            input_ids=None if input_ids is None else input_ids.contiguous(),
            hash_indices_table=(
                None if hash_indices_table is None else hash_indices_table.contiguous()
            ),
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            renormalize=self.config.renormalize,
            routed_scaling_factor=self.config.routed_scaling_factor,
        )
        _validate_moe_softplus_sqrt_topk_routing_values(values)
        return values


def _validate_moe_align_block_size_values(
    values: MoeAlignBlockSizeInputValues,
) -> None:
    if values.topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(values.topk_ids.shape)}")
    if values.topk_ids.dtype not in _TOPK_ID_DTYPES:
        raise ValueError("topk_ids must use an integer dtype")
    if values.block_size <= 0:
        raise ValueError("block_size must be positive")
    if values.num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if values.topk_ids.numel() == 0:
        return
    if values.topk_ids.min().item() < 0:
        raise ValueError("topk_ids must be non-negative")
    if values.topk_ids.max().item() >= values.num_experts:
        raise ValueError("topk_ids must be less than num_experts")


def _validate_moe_softmax_topk_routing_values(
    values: MoESoftmaxTopKRoutingInputValues,
) -> None:
    if values.logits.ndim != 2:
        raise ValueError("logits must be rank-2")
    if values.correction_bias.ndim != 1:
        raise ValueError("correction_bias must be rank-1")
    if values.logits.dtype != torch.float32:
        raise TypeError(f"logits must use torch.float32, got {values.logits.dtype}")
    if values.correction_bias.dtype != torch.float32:
        raise TypeError(
            f"correction_bias must use torch.float32, got {values.correction_bias.dtype}"
        )
    num_tokens, num_experts = values.logits.shape
    if values.correction_bias.shape != (num_experts,):
        raise ValueError("correction_bias must have one value per expert")
    if values.topk_indices.ndim != 2:
        raise ValueError("topk_indices must be rank-2")
    if values.topk_weights.ndim != 2:
        raise ValueError("topk_weights must be rank-2")
    if values.topk_indices.shape != values.topk_weights.shape:
        raise ValueError("topk_indices and topk_weights must have matching shapes")
    if values.topk_indices.shape[0] != num_tokens:
        raise ValueError("topk output rows must match logits rows")
    top_k = values.topk_indices.shape[1]
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > num_experts:
        raise ValueError("top_k must be <= num_experts")
    if values.topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_indices must use torch.int32 or torch.int64")
    if values.topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must use torch.float32")
    if values.logits.device != values.correction_bias.device:
        raise ValueError("logits and correction_bias must share device")
    if values.topk_indices.device != values.logits.device:
        raise ValueError("topk_indices must share device with logits")
    if values.topk_weights.device != values.logits.device:
        raise ValueError("topk_weights must share device with logits")
    if values.num_experts_real <= 0 or values.num_experts_real >= num_experts:
        raise ValueError("num_experts_real must be in [1, num_experts)")
    _check_positive_finite_float("scaling_factor", values.scaling_factor)
    if not torch.isfinite(values.logits).all():
        raise ValueError("logits must be finite")
    if not torch.isfinite(values.correction_bias).all():
        raise ValueError("correction_bias must be finite")


def _validate_moe_softplus_sqrt_topk_routing_values(
    values: MoESoftplusSqrtTopKRoutingInputValues,
) -> None:
    if values.logits.ndim != 2:
        raise ValueError("logits must be rank-2")
    if values.logits.dtype != torch.float32:
        raise TypeError("logits must use torch.float32")
    num_tokens, num_experts = values.logits.shape
    if values.topk_indices.ndim != 2:
        raise ValueError("topk_indices must be rank-2")
    if values.topk_weights.ndim != 2:
        raise ValueError("topk_weights must be rank-2")
    if values.topk_indices.shape != values.topk_weights.shape:
        raise ValueError("topk_indices and topk_weights must have matching shapes")
    if values.topk_indices.shape[0] != num_tokens:
        raise ValueError("topk output rows must match logits rows")
    top_k = values.topk_indices.shape[1]
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > num_experts:
        raise ValueError("top_k must be <= num_experts")
    if values.topk_indices.dtype != torch.int32:
        raise TypeError("topk_indices must use torch.int32")
    if values.topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must use torch.float32")
    if values.topk_indices.device != values.logits.device:
        raise ValueError("topk_indices must share device with logits")
    if values.topk_weights.device != values.logits.device:
        raise ValueError("topk_weights must share device with logits")
    if not values.renormalize:
        raise ValueError("softplus-sqrt routing requires renormalize=True")
    _check_positive_finite_float("routed_scaling_factor", values.routed_scaling_factor)

    has_hash_inputs = (
        values.input_ids is not None or values.hash_indices_table is not None
    )
    if has_hash_inputs:
        if values.correction_bias is not None:
            raise ValueError("hash routing must not include correction_bias")
        if values.input_ids is None or values.hash_indices_table is None:
            raise ValueError("hash routing requires input_ids and hash_indices_table")
        if values.input_ids.ndim != 1:
            raise ValueError("input_ids must be rank-1")
        if values.input_ids.shape[0] != num_tokens:
            raise ValueError("input_ids length must match logits rows")
        if values.input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use torch.int32 or torch.int64")
        if values.input_ids.device != values.logits.device:
            raise ValueError("input_ids must share device with logits")
        table = values.hash_indices_table
        if table.ndim != 2:
            raise ValueError("hash_indices_table must be rank-2")
        if table.shape[1] != top_k:
            raise ValueError("hash_indices_table width must match top_k")
        if table.dtype != torch.int32:
            raise TypeError("hash_indices_table must use torch.int32")
        if table.device != values.logits.device:
            raise ValueError("hash_indices_table must share device with logits")
        if table.shape[0] <= 0:
            raise ValueError("hash_indices_table must contain at least one row")
        if num_tokens > 0:
            if values.input_ids.min().item() < 0:
                raise ValueError("input_ids must be non-negative")
            if values.input_ids.max().item() >= table.shape[0]:
                raise ValueError("input_ids must index hash_indices_table rows")
        if table.numel() > 0:
            if table.min().item() < 0 or table.max().item() >= num_experts:
                raise ValueError("hash_indices_table expert ids must be in range")
            sorted_rows = table.to(torch.long).sort(dim=-1).values
            if top_k > 1 and torch.any(sorted_rows[:, 1:] == sorted_rows[:, :-1]):
                raise ValueError("hash_indices_table rows must not repeat experts")
    else:
        if values.correction_bias is None:
            raise ValueError("non-hash routing requires correction_bias")
        if values.correction_bias.ndim != 1:
            raise ValueError("correction_bias must be rank-1")
        if values.correction_bias.shape != (num_experts,):
            raise ValueError("correction_bias must have one value per expert")
        if values.correction_bias.dtype != torch.float32:
            raise TypeError("correction_bias must use torch.float32")
        if values.correction_bias.device != values.logits.device:
            raise ValueError("correction_bias must share device with logits")
        if not torch.isfinite(values.correction_bias).all():
            raise ValueError("correction_bias must be finite")

    if not torch.isfinite(values.logits).all():
        raise ValueError("logits must be finite")


def _validate_moe_deepseek_v4_mega_moe_staging_values(
    values: MoEDeepSeekV4MegaMoEStagingInputValues,
) -> None:
    if values.hidden_states.ndim != 2:
        raise ValueError("hidden_states must be rank-2")
    if not torch.is_floating_point(values.hidden_states):
        raise TypeError("hidden_states must use a floating dtype")
    num_tokens, hidden_size = values.hidden_states.shape
    if hidden_size <= 0 or hidden_size % _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE != 0:
        raise ValueError("hidden_size must be a positive multiple of 128")
    if values.num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if values.topk_ids.ndim != 2:
        raise ValueError("topk_ids must be rank-2")
    if values.topk_weights.ndim != 2:
        raise ValueError("topk_weights must be rank-2")
    if values.topk_ids.shape != values.topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have matching shapes")
    if values.topk_ids.shape[0] != num_tokens:
        raise ValueError("topk rows must match hidden_states rows")
    if values.topk_ids.shape[1] <= 0:
        raise ValueError("top_k must be positive")
    if values.topk_ids.shape[1] > values.num_experts:
        raise ValueError("top_k must be <= num_experts")
    if values.topk_ids.dtype not in _TOPK_ID_DTYPES:
        raise TypeError("topk_ids must use an integer dtype")
    if values.topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must use torch.float32")
    if values.x_fp8.shape != values.hidden_states.shape:
        raise ValueError("x_fp8 shape must match hidden_states")
    if values.x_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("x_fp8 must use torch.float8_e4m3fn")
    if values.x_sf.shape != (
        num_tokens,
        hidden_size // _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE,
    ):
        raise ValueError("x_sf shape must be [num_tokens, hidden_size // 128]")
    if values.x_sf.dtype != torch.int32:
        raise TypeError("x_sf must use torch.int32")
    if values.topk_idx_out.shape != values.topk_ids.shape:
        raise ValueError("topk_idx_out shape must match topk_ids")
    if values.topk_idx_out.dtype != values.topk_ids.dtype:
        raise TypeError("topk_idx_out dtype must match topk_ids")
    if values.topk_weights_out.shape != values.topk_weights.shape:
        raise ValueError("topk_weights_out shape must match topk_weights")
    if values.topk_weights_out.dtype != values.topk_weights.dtype:
        raise TypeError("topk_weights_out dtype must match topk_weights")
    tensors = (
        values.topk_ids,
        values.topk_weights,
        values.x_fp8,
        values.x_sf,
        values.topk_idx_out,
        values.topk_weights_out,
    )
    if any(tensor.device != values.hidden_states.device for tensor in tensors):
        raise ValueError("all staging tensors must share device")
    if values.topk_ids.numel() > 0:
        if values.topk_ids.min().item() < 0:
            raise ValueError("topk_ids must be non-negative")
        if values.topk_ids.max().item() >= values.num_experts:
            raise ValueError("topk_ids must be less than num_experts")
    if not torch.isfinite(values.hidden_states).all():
        raise ValueError("hidden_states must be finite")
    if not torch.isfinite(values.topk_weights).all():
        raise ValueError("topk_weights must be finite")
    if torch.any(values.topk_weights < 0.0):
        raise ValueError("topk_weights must be non-negative")


def _validate_moe_finalize_fuse_shared_values(
    values: MoEFinalizeFuseSharedInputValues,
) -> None:
    if values.gemm2_out.ndim != 2:
        raise ValueError("gemm2_out must be rank-2")
    if values.gemm2_out.dtype != torch.bfloat16:
        raise TypeError("gemm2_out must use torch.bfloat16")
    total_num_padded_tokens, hidden_size_padded = values.gemm2_out.shape
    if hidden_size_padded <= 0:
        raise ValueError("gemm2_out hidden dimension must be positive")
    if values.expert_weights.ndim != 2:
        raise ValueError("expert_weights must be rank-2")
    if values.expert_weights.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("expert_weights must use torch.float32 or torch.bfloat16")
    num_tokens, top_k = values.expert_weights.shape
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > _MOE_FINALIZE_MAX_TOPK:
        raise ValueError(f"top_k must be <= {_MOE_FINALIZE_MAX_TOPK}")
    if values.expanded_idx_to_permuted_idx.ndim != 1:
        raise ValueError("expanded_idx_to_permuted_idx must be rank-1")
    if values.expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise TypeError("expanded_idx_to_permuted_idx must use torch.int32")
    if values.expanded_idx_to_permuted_idx.shape[0] != num_tokens * top_k:
        raise ValueError(
            "expanded_idx_to_permuted_idx length must equal num_tokens * top_k"
        )
    if values.expert_weights.device != values.gemm2_out.device:
        raise ValueError("expert_weights must share device with gemm2_out")
    if values.expanded_idx_to_permuted_idx.device != values.gemm2_out.device:
        raise ValueError(
            "expanded_idx_to_permuted_idx must share device with gemm2_out"
        )
    if values.shared_output is not None:
        if values.shared_output.ndim != 2:
            raise ValueError("shared_output must be rank-2 when present")
        if values.shared_output.dtype != torch.bfloat16:
            raise TypeError("shared_output must use torch.bfloat16")
        if values.shared_output.shape[0] != num_tokens:
            raise ValueError("shared_output rows must match expert_weights rows")
        if values.shared_output.shape[1] <= 0:
            raise ValueError("shared_output hidden dimension must be positive")
        if values.shared_output.shape[1] > hidden_size_padded:
            raise ValueError("shared_output hidden dimension cannot exceed gemm2_out")
        if values.shared_output.device != values.gemm2_out.device:
            raise ValueError("shared_output must share device with gemm2_out")
        if not torch.isfinite(values.shared_output.float()).all():
            raise ValueError("shared_output must be finite")
    indices = values.expanded_idx_to_permuted_idx
    if indices.numel() > 0:
        if indices.min().item() < -1:
            raise ValueError("expanded indices must be -1 or non-negative")
        if total_num_padded_tokens == 0:
            if torch.any(indices >= 0):
                raise ValueError("non-dropped indices require gemm2_out rows")
        elif indices.max().item() >= total_num_padded_tokens:
            raise ValueError("expanded indices must be less than gemm2_out rows")
        active = indices[indices >= 0].detach().cpu().to(torch.long)
        if active.numel() > 1:
            if torch.unique(active).numel() != active.numel():
                raise ValueError("non-dropped expanded indices must be unique")
    if not torch.isfinite(values.gemm2_out.float()).all():
        raise ValueError("gemm2_out must be finite")
    if not torch.isfinite(values.expert_weights.float()).all():
        raise ValueError("expert_weights must be finite")


def _validate_moe_biased_grouped_topk_values(
    values: MoEBiasedGroupedTopKInputValues,
) -> None:
    if values.hidden_states.ndim != 2:
        raise ValueError("hidden_states must be rank-2")
    if values.gating_output.ndim != 2:
        raise ValueError("gating_output must be rank-2")
    if values.correction_bias.ndim != 1:
        raise ValueError("correction_bias must be rank-1")
    if not torch.is_floating_point(values.hidden_states):
        raise TypeError("hidden_states must use a floating dtype")
    if not torch.is_floating_point(values.gating_output):
        raise TypeError("gating_output must use a floating dtype")
    if values.correction_bias.dtype != torch.float32:
        raise TypeError("correction_bias must use torch.float32")
    num_tokens, num_experts = values.gating_output.shape
    if values.hidden_states.shape[0] != num_tokens:
        raise ValueError("hidden_states rows must match gating_output rows")
    if values.correction_bias.shape != (num_experts,):
        raise ValueError("correction_bias must have one value per expert")
    top_k = int(values.top_k)
    num_expert_groups = int(values.num_expert_groups)
    top_k_groups = int(values.top_k_groups)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if num_expert_groups <= 0:
        raise ValueError("num_expert_groups must be positive")
    if num_experts % num_expert_groups != 0:
        raise ValueError("num_experts must be divisible by num_expert_groups")
    experts_per_group = num_experts // num_expert_groups
    if experts_per_group < 2:
        raise ValueError("each expert group must contain at least two experts")
    if top_k_groups <= 0 or top_k_groups > num_expert_groups:
        raise ValueError("top_k_groups must be in [1, num_expert_groups]")
    if top_k > top_k_groups * experts_per_group:
        raise ValueError("top_k must fit within the experts in the selected groups")
    if values.hidden_states.device != values.gating_output.device:
        raise ValueError("hidden_states and gating_output must share device")
    if values.correction_bias.device != values.gating_output.device:
        raise ValueError("correction_bias must share device with gating_output")
    if values.logical_to_physical_map is not None:
        mapping = values.logical_to_physical_map
        if mapping.shape != (num_experts,):
            raise ValueError("logical_to_physical_map must have one entry per expert")
        if mapping.dtype != torch.int32:
            raise TypeError("logical_to_physical_map must use torch.int32")
        if mapping.device != values.gating_output.device:
            raise ValueError("logical_to_physical_map must share device")
        sorted_mapping = mapping.detach().cpu().to(torch.long).sort().values
        expected = torch.arange(num_experts, dtype=torch.long)
        if not torch.equal(sorted_mapping, expected):
            raise ValueError("logical_to_physical_map must be a permutation")
    if values.num_token_non_padded is not None:
        if values.num_token_non_padded.shape != ():
            raise ValueError("num_token_non_padded must be a scalar tensor")
        if values.num_token_non_padded.dtype != torch.int32:
            raise TypeError("num_token_non_padded must use torch.int32")
        if values.num_token_non_padded.device != values.gating_output.device:
            raise ValueError("num_token_non_padded must share device")
        value = int(values.num_token_non_padded.detach().cpu().item())
        if not 0 <= value <= num_tokens:
            raise ValueError("num_token_non_padded must be in [0, num_tokens]")
    _check_positive_finite_float("routed_scaling_factor", values.routed_scaling_factor)
    if not torch.isfinite(values.hidden_states).all():
        raise ValueError("hidden_states must be finite")
    if not torch.isfinite(values.gating_output).all():
        raise ValueError("gating_output must be finite")
    if not torch.isfinite(values.correction_bias).all():
        raise ValueError("correction_bias must be finite")


def moe_align_block_size_buffer_dims(
    values: MoeAlignBlockSizeInputValues,
) -> tuple[int, int]:
    """Return exact reference output sizes for MoE align-block-size metadata."""

    _validate_moe_align_block_size_values(values)
    flat_count = values.topk_ids.numel()
    if flat_count == 0:
        return 0, 0
    expert_counts = torch.bincount(
        values.topk_ids.reshape(-1).to(torch.long).cpu(),
        minlength=values.num_experts,
    )
    padded_slots = sum(
        _ceil_div(int(count), values.block_size) * values.block_size
        for count in expert_counts
    )
    num_blocks = padded_slots // values.block_size
    return num_blocks, padded_slots


def moe_align_block_size_reference(
    values: MoeAlignBlockSizeInputValues,
) -> MoeAlignBlockSizeReferenceValues:
    """Reference implementation for MoE top-k id block alignment."""

    _validate_moe_align_block_size_values(values)
    device = values.topk_ids.device
    pad_id = values.topk_ids.numel()
    sorted_chunks: list[torch.Tensor] = []
    expert_blocks: list[int] = []
    flat_ids = values.topk_ids.reshape(-1)
    flat_positions = torch.arange(pad_id, dtype=torch.int32, device=device)

    for expert in range(values.num_experts):
        selected = flat_positions[flat_ids == expert]
        count = int(selected.numel())
        padded_count = _ceil_div(count, values.block_size) * values.block_size
        if padded_count == 0:
            continue
        if padded_count > count:
            padding = torch.full(
                (padded_count - count,),
                pad_id,
                dtype=torch.int32,
                device=device,
            )
            selected = torch.cat((selected.to(torch.int32), padding))
        else:
            selected = selected.to(torch.int32)
        sorted_chunks.append(selected)
        expert_blocks.extend([expert] * (padded_count // values.block_size))

    if sorted_chunks:
        sorted_token_ids = torch.cat(sorted_chunks)
    else:
        sorted_token_ids = torch.empty((0,), dtype=torch.int32, device=device)
    expert_ids = torch.tensor(expert_blocks, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.tensor(
        [sorted_token_ids.numel()],
        dtype=torch.int32,
        device=device,
    )
    return MoeAlignBlockSizeReferenceValues(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_pad=num_tokens_post_pad,
    )


def canonicalize_moe_align_block_size(
    values: MoeAlignBlockSizeReferenceValues,
    *,
    block_size: int,
) -> torch.Tensor:
    """Pack MoE align-block-size outputs for deterministic comparison.

    ``sorted_token_ids`` order within a block can vary across parallel
    implementations. Sorting each block keeps the semantic assignment while
    ignoring the non-deterministic intra-block order.
    """

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if values.sorted_token_ids.numel() != values.expert_ids.numel() * block_size:
        raise ValueError(
            "sorted_token_ids size must equal expert_ids.numel() * block_size"
        )
    blocks = values.sorted_token_ids.reshape(values.expert_ids.numel(), block_size)
    blocks_sorted, _ = blocks.sort(dim=-1)
    return torch.cat(
        (
            values.num_tokens_post_pad.flatten().to(torch.int32),
            values.expert_ids.flatten().to(torch.int32),
            blocks_sorted.flatten().to(torch.int32),
        )
    )


@dataclass
class MoeInputValues:
    """Generated values for ``MoeInputs``."""

    hidden_states: torch.Tensor | None
    router_logits: torch.Tensor | None
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w13: GemmInputValues
    w2: GemmInputValues
    w13_bias: torch.Tensor | None
    w2_bias: torch.Tensor | None
    w13_activation_scale: torch.Tensor | None
    w2_activation_scale: torch.Tensor | None


@dataclass
class MoeInputConfig:
    """Initialization parameters for ``MoeInputs``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: total routed tokens.
    num_tokens: int

    # Required: hidden-state width.
    hidden_size: int

    # Required: expert intermediate width.
    intermediate_size: int

    # Required: number of routed experts.
    num_experts: int

    # Required: number of selected experts per token.
    top_k: int

    # Required: generated hidden-state dtype.
    hidden_dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional dtype, format, and layer configuration.
    # ------------------------------------------------------------------

    # Optional: generated router-logit dtype.
    router_dtype: torch.dtype = torch.float32

    # Optional: generated expert-weight dtype. Defaults to hidden_dtype for
    # dense weights, MXFP4 for mxfp4 weights, and MXINT4 for mxint4 weights.
    weight_dtype: InputDType = None

    # Optional: expert-weight storage/scale format.
    weight_format: str = "dense"

    # Optional: expert-weight scale dtype for scaled formats. MXFP4 accepts
    # ``None`` to infer raw UE8M0 scale storage; MXINT4 accepts ``None`` to
    # infer BF16 group scales.
    weight_scale_dtype: InputDType = None

    # Optional: output dtype metadata reserved for layer-level tests.
    output_dtype: torch.dtype | None = None

    # Optional: generated bias dtype. ``None`` skips biases.
    bias_dtype: torch.dtype | None = None

    # Optional: dtype for generated per-expert activation scales used by
    # quantized expert projection paths. ``None`` skips these side inputs.
    activation_scale_dtype: torch.dtype | None = None

    # Optional: positive scalar used to fill the W13 activation scale tensor.
    w13_activation_scale: float = 0.125

    # Optional: positive scalar used to fill the W2 activation scale tensor.
    w2_activation_scale: float = 0.125

    # Optional: default generation device.
    device: DeviceLike = None

    # Optional: SwiGLU alpha metadata reserved for layer-level tests.
    swiglu_alpha: float | None = None

    # Optional: SwiGLU limit metadata reserved for layer-level tests.
    swiglu_limit: float | None = None

    # ------------------------------------------------------------------
    # Optional child generator configuration.
    # ------------------------------------------------------------------

    # Optional: nested config for gate/up projection expert weights.
    w13: GemmInputConfig | None = None

    # Optional: nested config for down projection expert weights.
    w2: GemmInputConfig | None = None


@dataclass(init=False)
class MoeInputs(NumericsInputGenerator):
    """Typed input generator for routed MoE layer tests.

    Defaults model a SwiGLU MoE layer with generated hidden states, router
    logits, random top-k expert ids, and expert weight inputs for the gate/up
    and down projections. Dense and quantized/scaled weights use ``GemmInputs``
    and skip activation operands by default, because activations normally come
    from the layer execution itself.

    Leaf tensor generators and child GEMM generators are initialized at
    construction time and reused by ``generate`` so callers can mutate child
    generator fields before generating through this parent.
    """

    config: MoeInputConfig
    hidden_states_input: TensorInput | None
    router_logits_input: TensorInput | None
    w13: GemmInputs | None
    w2: GemmInputs | None
    w13_bias_input: TensorInput | None
    w2_bias_input: TensorInput | None

    def __init__(
        self,
        config: MoeInputConfig,
    ) -> None:
        self.config = config
        self.hidden_states_input = None
        self.router_logits_input = None
        self.w13 = None
        self.w2 = None
        self.w13_bias_input = None
        self.w2_bias_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = int(self.config.num_tokens)
        self.config.hidden_size = int(self.config.hidden_size)
        self.config.intermediate_size = int(self.config.intermediate_size)
        self.config.num_experts = int(self.config.num_experts)
        self.config.top_k = int(self.config.top_k)
        if self.config.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self.config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.config.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.top_k > self.config.num_experts:
            raise ValueError("top_k must be <= num_experts")
        if not isinstance(self.config.hidden_dtype, torch.dtype):
            raise TypeError("hidden_dtype must be a torch.dtype")
        if not isinstance(self.config.router_dtype, torch.dtype):
            raise TypeError("router_dtype must be a torch.dtype")
        if self.config.activation_scale_dtype is not None:
            self.config.activation_scale_dtype = _check_moe_float_dtype(
                "activation_scale_dtype", self.config.activation_scale_dtype
            )
        self.config.w13_activation_scale = _check_positive_finite_float(
            "w13_activation_scale", self.config.w13_activation_scale
        )
        self.config.w2_activation_scale = _check_positive_finite_float(
            "w2_activation_scale", self.config.w2_activation_scale
        )
        if self.config.weight_dtype is None:
            self.config.weight_dtype = (
                CustomDType.MXFP4
                if self.config.weight_format == "mxfp4"
                else (
                    CustomDType.MXINT4
                    if self.config.weight_format == "mxint4"
                    else self.config.hidden_dtype
                )
            )
        if self.config.weight_format == "mxint4":
            if self.config.weight_dtype != CustomDType.MXINT4:
                raise ValueError("mxint4 MoE weights require CustomDType.MXINT4")
            if self.config.weight_scale_dtype not in (None, torch.bfloat16):
                raise ValueError(
                    "mxint4 MoE weight_scale_dtype must be torch.bfloat16 or None"
                )
        if (
            self.config.weight_format not in {"dense", "mxfp4", "mxint4"}
            and self.config.weight_scale_dtype is None
        ):
            self.config.weight_scale_dtype = torch.float8_e4m3fn
        if self.hidden_states_input is None:
            self.hidden_states_input = TensorInput(
                (self.config.num_tokens, self.config.hidden_size),
                self.config.hidden_dtype,
                device=self.config.device,
            )
        if self.router_logits_input is None:
            self.router_logits_input = TensorInput(
                (self.config.num_tokens, self.config.num_experts),
                self.config.router_dtype,
                device=self.config.device,
            )
        if self.w13 is None:
            self.w13 = self._make_weight_gemm(
                self.config.w13
                or self._make_weight_gemm_config(
                    N=2 * self.config.intermediate_size,
                    K=self.config.hidden_size,
                )
            )
        if self.w2 is None:
            self.w2 = self._make_weight_gemm(
                self.config.w2
                or self._make_weight_gemm_config(
                    N=self.config.hidden_size,
                    K=self.config.intermediate_size,
                )
            )
        if self.w13_bias_input is None:
            self.w13_bias_input = TensorInput(
                (self.config.num_experts, 2 * self.config.intermediate_size),
                self.config.bias_dtype,
                device=self.config.device,
            )
        if self.w2_bias_input is None:
            self.w2_bias_input = TensorInput(
                (self.config.num_experts, self.config.hidden_size),
                self.config.bias_dtype,
                device=self.config.device,
            )
        self.config.w13 = self.w13.config
        self.config.w2 = self.w2.config
        self._verify_weight_gemm_config(
            "w13",
            self.config.w13,
            N=2 * self.config.intermediate_size,
            K=self.config.hidden_size,
        )
        self._verify_weight_gemm_config(
            "w2",
            self.config.w2,
            N=self.config.hidden_size,
            K=self.config.intermediate_size,
        )

    def _make_weight_gemm(
        self,
        config: GemmInputConfig,
    ) -> GemmInputs:
        return GemmInputs(config)

    def _verify_weight_gemm_config(
        self,
        name: str,
        config: GemmInputConfig,
        *,
        N: int,
        K: int,
    ) -> None:
        if config.M != self.config.num_tokens or config.N != N or config.K != K:
            raise ValueError(
                f"{name} GEMM config must use M/N/K "
                f"{(self.config.num_tokens, N, K)}, got "
                f"{(config.M, config.N, config.K)}"
            )
        if config.batch_shape != (self.config.num_experts,):
            raise ValueError(
                f"{name} GEMM config batch_shape must be "
                f"{(self.config.num_experts,)}, got {config.batch_shape}"
            )
        if config.a_dtype is not None:
            raise ValueError(f"{name} GEMM config must use a_dtype=None")

    def _make_weight_gemm_config(
        self,
        *,
        N: int,
        K: int,
    ) -> GemmInputConfig:
        c_dtype = self.config.output_dtype or self.config.hidden_dtype
        if self.config.weight_format == "dense":
            if not isinstance(self.config.weight_dtype, torch.dtype):
                raise ValueError("dense MoE weights require a torch dtype")
            return GemmInputConfig(
                M=self.config.num_tokens,
                N=N,
                K=K,
                a_dtype=None,
                b_dtype=self.config.weight_dtype,
                c_dtype=c_dtype,
                batch_shape=(self.config.num_experts,),
            )

        if self.config.weight_format == "mxfp4":
            return mxfp4_gemm_input_config(
                M=self.config.num_tokens,
                N=N,
                K=K,
                a_dtype=None,
                b_dtype=self.config.weight_dtype,
                scale_dtype=self.config.weight_scale_dtype,
                c_dtype=c_dtype,
                batch_shape=(self.config.num_experts,),
            )

        if self.config.weight_format == "mxint4":
            return mxint4_gemm_input_config(
                M=self.config.num_tokens,
                N=N,
                K=K,
                a_dtype=None,
                b_dtype=self.config.weight_dtype,
                scale_dtype=self.config.weight_scale_dtype,
                c_dtype=c_dtype,
                batch_shape=(self.config.num_experts,),
            )

        return GemmInputConfig(
            M=self.config.num_tokens,
            N=N,
            K=K,
            a_dtype=None,
            b_dtype=self.config.weight_dtype,
            c_dtype=c_dtype,
            batch_shape=(self.config.num_experts,),
            b_scale_shape=(1,),
            b_scale_dtype=self.config.weight_scale_dtype or torch.float32,
        )

    def generate(self, *, seed: int, device: DeviceLike = None) -> MoeInputValues:
        default_device = _resolve_device(self.config.device, device)
        if (
            self.hidden_states_input is None
            or self.router_logits_input is None
            or self.w13 is None
            or self.w2 is None
            or self.w13_bias_input is None
            or self.w2_bias_input is None
        ):
            raise ValueError("MoeInputs child generators must be initialized")

        hidden_states = self.hidden_states_input.generate(
            seed=_child_seed(seed, 1),
            device=default_device,
        ).values
        router_logits = self.router_logits_input.generate(
            seed=_child_seed(seed, 2),
            device=default_device,
        ).values
        if hidden_states is None:
            raise ValueError("hidden_states generation was skipped")
        if router_logits is None:
            raise ValueError("router_logits generation was skipped")
        topk_weights, topk_ids = _moe_topk_from_router_logits(
            router_logits,
            self.config.top_k,
            dtype=hidden_states.dtype,
        )

        w13 = self.w13.generate(seed=_child_seed(seed, 4), device=default_device)
        w2 = self.w2.generate(seed=_child_seed(seed, 5), device=default_device)

        w13_bias = self.w13_bias_input.generate(
            seed=_child_seed(seed, 6),
            device=default_device,
        ).values
        w2_bias = self.w2_bias_input.generate(
            seed=_child_seed(seed, 7),
            device=default_device,
        ).values
        w13_activation_scale = self._make_activation_scale(
            self.config.w13_activation_scale,
            device=default_device,
        )
        w2_activation_scale = self._make_activation_scale(
            self.config.w2_activation_scale,
            device=default_device,
        )
        return MoeInputValues(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13=w13,
            w2=w2,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            w13_activation_scale=w13_activation_scale,
            w2_activation_scale=w2_activation_scale,
        )

    def _make_activation_scale(
        self,
        scale: float,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.config.activation_scale_dtype is None:
            return None
        return torch.full(
            (self.config.num_experts,),
            scale,
            dtype=self.config.activation_scale_dtype,
            device=device,
        )


def _moe_topk_from_router_logits(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if router_logits.ndim != 2:
        raise ValueError(
            f"router_logits must be rank-2, got {tuple(router_logits.shape)}"
        )
    top_k = int(top_k)
    num_experts = router_logits.shape[-1]
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > num_experts:
        raise ValueError("top_k must be <= router_logits.shape[-1]")
    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(dtype), topk_ids.to(torch.int32)


def moe_softmax_topk_routing_reference(
    values: MoESoftmaxTopKRoutingInputValues,
) -> MoESoftmaxTopKRoutingReferenceValues:
    """Reference implementation for softmax/bias top-k MoE routing."""

    _validate_moe_softmax_topk_routing_values(values)
    probs = torch.softmax(values.logits.float(), dim=-1)
    selection_scores = probs + values.correction_bias.reshape(1, -1)
    top_k = values.topk_indices.shape[1]
    out_indices = torch.empty_like(values.topk_indices)
    out_weights = torch.empty_like(values.topk_weights)
    scores_cpu = selection_scores.detach().cpu()
    probs_cpu = probs.detach().cpu()
    for row_idx in range(scores_cpu.shape[0]):
        ordered = sorted(
            range(scores_cpu.shape[1]),
            key=lambda expert: (-float(scores_cpu[row_idx, expert]), expert),
        )[:top_k]
        selected_probs = torch.tensor(
            [float(probs_cpu[row_idx, expert]) for expert in ordered],
            dtype=torch.float32,
            device=values.logits.device,
        )
        if values.renormalize:
            selected_probs = selected_probs / (selected_probs.sum() + 1.0e-10)
        indices = [
            -1 if expert >= values.num_experts_real else expert for expert in ordered
        ]
        out_indices[row_idx] = torch.tensor(
            indices,
            dtype=values.topk_indices.dtype,
            device=values.logits.device,
        )
        out_weights[row_idx] = selected_probs * float(values.scaling_factor)
    return MoESoftmaxTopKRoutingReferenceValues(
        topk_indices=out_indices,
        topk_weights=out_weights,
    )


def moe_biased_grouped_topk_reference(
    values: MoEBiasedGroupedTopKInputValues,
) -> MoEBiasedGroupedTopKReferenceValues:
    """Reference implementation for sigmoid/bias grouped top-k MoE routing."""

    _validate_moe_biased_grouped_topk_values(values)
    scores = values.gating_output.float().sigmoid()
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // values.num_expert_groups
    selection_scores = scores + values.correction_bias.reshape(1, -1)
    grouped_scores = selection_scores.reshape(
        num_tokens,
        values.num_expert_groups,
        experts_per_group,
    )
    group_scores = grouped_scores.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = torch.topk(
        group_scores,
        k=values.top_k_groups,
        dim=-1,
        sorted=False,
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, values.num_expert_groups, experts_per_group)
        .reshape(num_tokens, num_experts)
    )
    candidate_scores = selection_scores.masked_fill(~expert_mask, float("-inf"))
    topk_ids = torch.topk(
        candidate_scores,
        k=values.top_k,
        dim=-1,
        sorted=False,
    ).indices.to(torch.int32)
    topk_weights = scores.gather(1, topk_ids.to(torch.long)).to(torch.float32)

    if values.renormalize:
        weight_sum = topk_weights.sum(dim=-1, keepdim=True)
        denom = torch.where(weight_sum != 0.0, weight_sum, torch.ones_like(weight_sum))
        topk_weights = topk_weights / denom
        topk_weights = topk_weights * float(values.routed_scaling_factor)

    if values.logical_to_physical_map is not None:
        topk_ids = values.logical_to_physical_map[topk_ids.to(torch.long)].to(
            torch.int32
        )
    if values.num_token_non_padded is not None:
        valid_tokens = int(values.num_token_non_padded.detach().cpu().item())
        if valid_tokens < num_tokens:
            topk_ids = topk_ids.clone()
            topk_ids[valid_tokens:, :] = -1

    return MoEBiasedGroupedTopKReferenceValues(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def moe_softplus_sqrt_topk_routing_reference(
    values: MoESoftplusSqrtTopKRoutingInputValues,
) -> MoESoftplusSqrtTopKRoutingReferenceValues:
    """Reference implementation for softplus-sqrt top-k MoE routing."""

    _validate_moe_softplus_sqrt_topk_routing_values(values)
    transformed = torch.sqrt(torch.nn.functional.softplus(values.logits.float()))
    has_hash_inputs = values.input_ids is not None
    if has_hash_inputs:
        if values.hash_indices_table is None:
            raise ValueError("hash routing requires hash_indices_table")
        topk_indices = values.hash_indices_table[values.input_ids.to(torch.long)].to(
            torch.int32
        )
    else:
        if values.correction_bias is None:
            raise ValueError("non-hash routing requires correction_bias")
        selection_scores = transformed + values.correction_bias.reshape(1, -1)
        scores_cpu = selection_scores.detach().cpu()
        selected: list[torch.Tensor] = []
        for row_idx in range(scores_cpu.shape[0]):
            ordered = sorted(
                range(scores_cpu.shape[1]),
                key=lambda expert: (-float(scores_cpu[row_idx, expert]), expert),
            )[: values.topk_indices.shape[1]]
            selected.append(
                torch.tensor(ordered, dtype=torch.int32, device=values.logits.device)
            )
        topk_indices = (
            torch.stack(selected) if selected else torch.empty_like(values.topk_indices)
        )

    topk_weights = transformed.gather(1, topk_indices.to(torch.long))
    denom = topk_weights.sum(dim=-1, keepdim=True)
    denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
    topk_weights = topk_weights / denom
    topk_weights = topk_weights * float(values.routed_scaling_factor)
    return MoESoftplusSqrtTopKRoutingReferenceValues(
        topk_indices=topk_indices.to(torch.int32),
        topk_weights=topk_weights.to(torch.float32),
    )


def moe_deepseek_v4_mega_moe_staging_reference(
    values: MoEDeepSeekV4MegaMoEStagingInputValues,
) -> MoEDeepSeekV4MegaMoEStagingReferenceValues:
    """Reference implementation for DeepSeek V4 MegaMoE input staging."""

    _validate_moe_deepseek_v4_mega_moe_staging_values(values)
    hidden = values.hidden_states.float()
    num_tokens, hidden_size = hidden.shape
    block_size = _DEEPSEEK_V4_MEGAMOE_FP8_BLOCK_SIZE
    group_size = _DEEPSEEK_V4_MEGAMOE_FP8_GROUP_SIZE
    block_count = hidden_size // block_size
    grouped = hidden.reshape(num_tokens, block_count, block_size)
    grouped_abs = grouped.abs().reshape(num_tokens, block_count, 4, group_size)
    amax = grouped_abs.amax(dim=-1).clamp_min(1.0e-4)
    scale = amax / _DEEPSEEK_V4_MEGAMOE_FP8_MAX
    scale_bits = scale.contiguous().view(torch.int32)
    mantissa_nonzero = (scale_bits & 0x7FFFFF) != 0
    scale_exp = ((scale_bits >> 23) & 0xFF) + mantissa_nonzero.to(torch.int32)
    scale_exp = scale_exp.clamp(1, 254)
    rounded_scale = torch.pow(
        torch.full_like(scale, 2.0),
        scale_exp.float() - 127.0,
    )
    scaled = grouped.reshape(num_tokens, block_count, 4, group_size) / (
        rounded_scale.unsqueeze(-1)
    )
    x_fp8 = scaled.reshape_as(hidden).to(torch.float8_e4m3fn)
    shifts = torch.arange(
        0,
        32,
        8,
        dtype=torch.int32,
        device=values.hidden_states.device,
    )
    x_sf = ((scale_exp.to(torch.int32) << shifts).sum(dim=-1)).to(torch.int32)
    return MoEDeepSeekV4MegaMoEStagingReferenceValues(
        x_fp8=x_fp8,
        x_sf=x_sf,
        topk_idx_out=values.topk_ids.clone(),
        topk_weights_out=values.topk_weights.clone(),
    )


def moe_finalize_fuse_shared_reference(
    values: MoEFinalizeFuseSharedInputValues,
) -> torch.Tensor:
    """Reference implementation for MoE routed-output finalization."""

    _validate_moe_finalize_fuse_shared_values(values)
    num_tokens, top_k = values.expert_weights.shape
    hidden_size = (
        values.shared_output.shape[1]
        if values.shared_output is not None
        else values.gemm2_out.shape[1]
    )
    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=values.gemm2_out.device,
    )
    indices = values.expanded_idx_to_permuted_idx.reshape(num_tokens, top_k)
    for token_idx in range(num_tokens):
        for topk_idx in range(top_k):
            permuted_idx = int(indices[token_idx, topk_idx].item())
            if permuted_idx == -1:
                continue
            output[token_idx] += (
                values.expert_weights[token_idx, topk_idx].float()
                * values.gemm2_out[permuted_idx, :hidden_size].float()
            )
    if values.shared_output is not None:
        output += values.shared_output.float()
    return output.to(torch.bfloat16)


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
    if topk_ids.dtype not in _TOPK_ID_DTYPES:
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
    """Return the semantic routed MoE layer output for generated values.

    The reference uses explicit ``topk_ids`` and ``topk_weights`` as the routing
    inputs. ``MoeInputs`` derives those tensors from ``router_logits`` by
    default, but callers may still use this reference for precomputed routing as
    long as the selected experts and weights satisfy the layer contract.
    """

    if values.hidden_states is None:
        raise ValueError("hidden_states are required for moe_reference")
    if values.hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must be rank-2, got " f"{tuple(values.hidden_states.shape)}"
        )
    num_tokens, hidden_size = values.hidden_states.shape
    if values.router_logits is not None:
        if values.router_logits.ndim != 2:
            raise ValueError(
                "router_logits must be rank-2, got "
                f"{tuple(values.router_logits.shape)}"
            )
        if values.router_logits.shape[0] != num_tokens:
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
    if (
        values.router_logits is not None
        and values.router_logits.shape[1] != num_experts
    ):
        raise ValueError("router_logits expert dimension must match weight experts")
    _validate_topk_values(
        topk_ids=values.topk_ids,
        topk_weights=values.topk_weights,
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

    hidden = values.hidden_states.float()
    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=hidden.device,
    )
    for slot in range(values.topk_ids.shape[1]):
        expert_ids = values.topk_ids[:, slot].to(torch.long)
        route_weights = values.topk_weights[:, slot].float().reshape(num_tokens, 1)
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
