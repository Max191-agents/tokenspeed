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
    GemmInputValues,
    GemmInputs,
    mxfp4_gemm_input_config,
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
]


_TOPK_ID_DTYPES = {torch.int16, torch.int32, torch.int64}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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
    w13: GemmInputValues
    w2: GemmInputValues
    w13_bias: torch.Tensor | None
    w2_bias: torch.Tensor | None


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
    # dense weights and MXFP4 for mxfp4 weights.
    weight_dtype: InputDType = None

    # Optional: expert-weight storage/scale format.
    weight_format: str = "dense"

    # Optional: expert-weight scale dtype for scaled formats. MXFP4 accepts
    # ``None`` to infer raw UE8M0 scale storage.
    weight_scale_dtype: InputDType = None

    # Optional: output dtype metadata reserved for layer-level tests.
    output_dtype: torch.dtype | None = None

    # Optional: generated bias dtype. ``None`` skips biases.
    bias_dtype: torch.dtype | None = None

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
        if (
            min(
                self.config.num_tokens,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.top_k,
            )
            < 0
        ):
            raise ValueError("MoE shape dimensions must be non-negative")
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.config.weight_dtype is None:
            self.config.weight_dtype = (
                CustomDType.MXFP4
                if self.config.weight_format == "mxfp4"
                else self.config.hidden_dtype
            )
        if (
            self.config.weight_format not in {"dense", "mxfp4"}
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

    def _make_weight_gemm(
        self,
        config: GemmInputConfig,
    ) -> GemmInputs:
        return GemmInputs(config)

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
        rng_device = "cuda" if default_device.type == "cuda" else "cpu"
        topk_generator = torch.Generator(device=rng_device).manual_seed(
            _child_seed(seed, 3)
        )
        topk_ids = torch.randint(
            0,
            self.config.num_experts,
            (self.config.num_tokens, self.config.top_k),
            device=default_device,
            dtype=torch.int32,
            generator=topk_generator,
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
        return MoeInputValues(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_ids=topk_ids,
            w13=w13,
            w2=w2,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )
