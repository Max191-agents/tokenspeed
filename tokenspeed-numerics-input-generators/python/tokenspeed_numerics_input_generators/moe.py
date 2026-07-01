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
    "MoeInputConfig",
    "MoeInputValues",
    "MoeInputs",
    "canonicalize_moe_align_block_size",
    "moe_align_block_size_buffer_dims",
    "moe_align_block_size_reference",
    "moe_reference",
]


_TOPK_ID_DTYPES = {torch.int16, torch.int32, torch.int64}
_REGULAR_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


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
