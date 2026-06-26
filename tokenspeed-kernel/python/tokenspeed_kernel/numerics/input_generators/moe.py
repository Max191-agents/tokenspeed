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
from tokenspeed_kernel.numerics.input_generators.core import (
    CustomDType,
    DeviceLike,
    InputDType,
    NumericsInputGenerator,
    TensorInputConfig,
    TensorInput,
    _child_seed,
    _resolve_device,
)
from tokenspeed_kernel.numerics.input_generators.gemm import (
    GemmInputConfig,
    GemmInputValues,
    GemmInputs,
    ScaledGemmInputConfig,
    ScaledGemmInputValues,
    ScaledGemmInputs,
)

__all__ = ["MoeInputConfig", "MoeInputValues", "MoeInputs"]


@dataclass
class MoeInputValues:
    """Generated values for ``MoeInputs``."""

    hidden_states: torch.Tensor | None
    router_logits: torch.Tensor | None
    topk_ids: torch.Tensor
    w13: GemmInputValues | ScaledGemmInputValues
    w2: GemmInputValues | ScaledGemmInputValues
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

    # Optional: expert-weight scale dtype for scaled formats.
    weight_scale_dtype: torch.dtype | None = None

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

    # Optional: nested config for generated hidden states.
    hidden_states_input: TensorInputConfig | None = None

    # Optional: nested config for generated router logits.
    router_logits_input: TensorInputConfig | None = None

    # Optional: nested config for gate/up projection expert weights.
    w13: GemmInputConfig | ScaledGemmInputConfig | None = None

    # Optional: nested config for down projection expert weights.
    w2: GemmInputConfig | ScaledGemmInputConfig | None = None

    # Optional: nested config for gate/up projection bias.
    w13_bias_input: TensorInputConfig | None = None

    # Optional: nested config for down projection bias.
    w2_bias_input: TensorInputConfig | None = None


@dataclass(init=False)
class MoeInputs(NumericsInputGenerator):
    """Typed input generator for routed MoE layer tests.

    Defaults model a SwiGLU MoE layer with generated hidden states, router
    logits, random top-k expert ids, and expert weight inputs for the gate/up
    and down projections. Dense weights use ``GemmInputs``. Quantized/scaled
    weights use ``ScaledGemmInputs`` and skip activation operands by default,
    because activations normally come from the layer execution itself.

    Leaf tensor generators and child GEMM generators are initialized at
    construction time and reused by ``generate`` so callers can mutate child
    configuration before generating through this parent.
    """

    config: MoeInputConfig
    hidden_states_input: TensorInput | None
    router_logits_input: TensorInput | None
    w13: GemmInputs | ScaledGemmInputs | None
    w2: GemmInputs | ScaledGemmInputs | None
    w13_bias_input: TensorInput | None
    w2_bias_input: TensorInput | None

    def __init__(
        self,
        config: MoeInputConfig | None = None,
        **kwargs: object,
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword parameters, not both")

        hidden_states_input = kwargs.get("hidden_states_input")
        router_logits_input = kwargs.get("router_logits_input")
        w13 = kwargs.get("w13")
        w2 = kwargs.get("w2")
        w13_bias_input = kwargs.get("w13_bias_input")
        w2_bias_input = kwargs.get("w2_bias_input")
        if isinstance(hidden_states_input, TensorInput):
            kwargs["hidden_states_input"] = hidden_states_input.config
        if isinstance(router_logits_input, TensorInput):
            kwargs["router_logits_input"] = router_logits_input.config
        if isinstance(w13, (GemmInputs, ScaledGemmInputs)):
            kwargs["w13"] = w13.config
        if isinstance(w2, (GemmInputs, ScaledGemmInputs)):
            kwargs["w2"] = w2.config
        if isinstance(w13_bias_input, TensorInput):
            kwargs["w13_bias_input"] = w13_bias_input.config
        if isinstance(w2_bias_input, TensorInput):
            kwargs["w2_bias_input"] = w2_bias_input.config

        self.config = config or MoeInputConfig(**kwargs)  # type: ignore[arg-type]
        self.hidden_states_input = (
            hidden_states_input
            if isinstance(hidden_states_input, TensorInput)
            else None
        )
        self.router_logits_input = (
            router_logits_input
            if isinstance(router_logits_input, TensorInput)
            else None
        )
        self.w13 = w13 if isinstance(w13, (GemmInputs, ScaledGemmInputs)) else None
        self.w2 = w2 if isinstance(w2, (GemmInputs, ScaledGemmInputs)) else None
        self.w13_bias_input = (
            w13_bias_input if isinstance(w13_bias_input, TensorInput) else None
        )
        self.w2_bias_input = (
            w2_bias_input if isinstance(w2_bias_input, TensorInput) else None
        )
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
            self.config.weight_format != "dense"
            and self.config.weight_scale_dtype is None
        ):
            self.config.weight_scale_dtype = torch.float8_e4m3fn
        if self.hidden_states_input is None:
            self.hidden_states_input = TensorInput(
                self.config.hidden_states_input
                or TensorInputConfig(
                    (self.config.num_tokens, self.config.hidden_size),
                    self.config.hidden_dtype,
                    device=self.config.device,
                )
            )
        if self.router_logits_input is None:
            self.router_logits_input = TensorInput(
                self.config.router_logits_input
                or TensorInputConfig(
                    (self.config.num_tokens, self.config.num_experts),
                    self.config.router_dtype,
                    device=self.config.device,
                )
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
                self.config.w13_bias_input
                or TensorInputConfig(
                    (self.config.num_experts, 2 * self.config.intermediate_size),
                    self.config.bias_dtype,
                    device=self.config.device,
                )
            )
        if self.w2_bias_input is None:
            self.w2_bias_input = TensorInput(
                self.config.w2_bias_input
                or TensorInputConfig(
                    (self.config.num_experts, self.config.hidden_size),
                    self.config.bias_dtype,
                    device=self.config.device,
                )
            )
        self.config.hidden_states_input = self.hidden_states_input.config
        self.config.router_logits_input = self.router_logits_input.config
        self.config.w13 = self.w13.config
        self.config.w2 = self.w2.config
        self.config.w13_bias_input = self.w13_bias_input.config
        self.config.w2_bias_input = self.w2_bias_input.config

    def _make_weight_gemm(
        self,
        config: GemmInputConfig | ScaledGemmInputConfig,
    ) -> GemmInputs | ScaledGemmInputs:
        if isinstance(config, GemmInputConfig):
            return GemmInputs(config)
        return ScaledGemmInputs(config)

    def _make_weight_gemm_config(
        self,
        *,
        N: int,
        K: int,
    ) -> GemmInputConfig | ScaledGemmInputConfig:
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
            return ScaledGemmInputs.mxfp4(
                M=self.config.num_tokens,
                N=N,
                K=K,
                a_dtype=None,
                b_dtype=self.config.weight_dtype,
                scale_dtype=self.config.weight_scale_dtype or torch.float8_e4m3fn,
                c_dtype=c_dtype,
                batch_shape=(self.config.num_experts,),
            ).config

        return ScaledGemmInputConfig(
            M=self.config.num_tokens,
            N=N,
            K=K,
            a_dtype=None,
            b_dtype=self.config.weight_dtype,
            a_scale_dtype=None,
            b_scale_dtype=self.config.weight_scale_dtype or torch.float32,
            c_dtype=c_dtype,
            a_scale_shape=None,
            b_scale_shape=(1,),
            batch_shape=(self.config.num_experts,),
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
        )
        router_logits = self.router_logits_input.generate(
            seed=_child_seed(seed, 2),
            device=default_device,
        )
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
        )
        w2_bias = self.w2_bias_input.generate(
            seed=_child_seed(seed, 7),
            device=default_device,
        )
        return MoeInputValues(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_ids=topk_ids,
            w13=w13,
            w2=w2,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )
