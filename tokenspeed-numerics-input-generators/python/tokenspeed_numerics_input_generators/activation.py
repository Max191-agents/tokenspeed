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

"""Activation-family input generators for numerical correctness tests.

The activation family represented here is defined by pointwise gated
transformations used in feed-forward and attention output paths:

* ``sigmoid_mul`` computes ``x * sigmoid(gate)``. The gate may be a dense
  matrix or a strided view into packed QKV-style storage, but both layouts
  represent the same elementwise gate values after flattening.
* split gated activations interpret a ``[..., 2 * hidden_dim]`` tensor as
  ``[gate, up]`` and compute ``activation(gate) * up``.
* fused gate-sigmoid-mul-add computes one scalar gate per token from a
  hidden-state dot product, then adds the gated shared output to the final
  hidden states.

The generators choose bounded normally distributed values and scale wide
dot-product weights where needed so references exercise the operations without
being dominated by sigmoid saturation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _is_floating_storage_dtype,
    _resolve_device,
)

__all__ = [
    "FusedGateSigmoidMulAddInputConfig",
    "FusedGateSigmoidMulAddInputs",
    "FusedGateSigmoidMulAddInputValues",
    "FusedSwiGLUFP8UE8M0InputConfig",
    "FusedSwiGLUFP8UE8M0Inputs",
    "FusedSwiGLUFP8UE8M0InputValues",
    "fused_swiglu_fp8_ue8m0_reference",
    "fused_gate_sigmoid_mul_add_reference",
    "GatedActivationInputConfig",
    "GatedActivationInputs",
    "GatedActivationInputValues",
    "gated_activation_reference",
    "SigmoidMulInputConfig",
    "SigmoidMulInputs",
    "SigmoidMulInputValues",
    "sigmoid_mul_reference",
]

GatedActivationKind = Literal["silu", "gelu", "gelu_tanh"]
SigmoidGateLayout = Literal["dense", "qkv_split"]


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


def _check_torch_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if not _is_floating_storage_dtype(dtype):
        raise ValueError(f"{name} must be a floating torch dtype, got {dtype}")
    return dtype


def _require_tensor(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} generation unexpectedly returned None")
    return values


@dataclass
class SigmoidMulInputValues:
    """Generated values for ``SigmoidMulInputs``."""

    x: torch.Tensor
    gate: torch.Tensor
    gate_storage: torch.Tensor | None = None


def sigmoid_mul_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return a float32 reference for ``x * sigmoid(gate)``.

    ``gate`` may be the dense ``[num_tokens, hidden_dim]`` form or the strided
    ``[num_tokens, num_heads, head_dim]`` form. In the strided case the last
    two dimensions must flatten to the same hidden width as ``x``.
    """

    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.ndim}D")
    if gate.ndim == 2:
        if gate.shape != x.shape:
            raise ValueError(f"gate shape must match x, got {gate.shape} and {x.shape}")
        gate_values = gate
    elif gate.ndim == 3:
        gate_tokens, num_heads, head_dim = gate.shape
        if gate_tokens != x.shape[0]:
            raise ValueError(
                f"gate token count must match x, got {gate_tokens} and {x.shape[0]}"
            )
        if num_heads * head_dim != x.shape[1]:
            raise ValueError(
                "flattened gate hidden width must match x, got "
                f"{num_heads} * {head_dim} and {x.shape[1]}"
            )
        gate_values = gate.reshape_as(x)
    else:
        raise ValueError(f"gate must be 2D or 3D, got {gate.ndim}D")
    return x.float() * gate_values.float().sigmoid()


@dataclass
class SigmoidMulInputConfig:
    """Initialization parameters for ``SigmoidMulInputs``.

    The represented operation is ``x * sigmoid(gate)``. ``x`` is always
    generated as a contiguous ``[num_tokens, hidden_dim]`` tensor. ``gate`` is
    either generated with the same dense shape or as the strided
    ``[num_tokens, num_heads, head_dim]`` view produced from packed QKV-style
    storage.
    """

    # Required: number of token rows. Zero is valid for empty-input tests.
    num_tokens: int

    # Required: flattened activation width.
    hidden_dim: int

    # Required: dtype for both x and gate.
    dtype: torch.dtype

    # Optional: generate dense gate or a strided QKV-split gate view.
    gate_layout: SigmoidGateLayout = "dense"

    # Optional: required for qkv_split. Must satisfy num_heads * head_dim ==
    # hidden_dim.
    num_heads: int | None = None

    # Optional: required for qkv_split. Last dimension of each attention head.
    head_dim: int | None = None

    # Optional: qkv_split metadata used only to size backing QKV storage. It
    # controls the row stride of the generated gate view.
    num_kv_heads: int | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class SigmoidMulInputs(NumericsInputGenerator):
    """Generator for ``x * sigmoid(gate)`` activation inputs."""

    config: SigmoidMulInputConfig
    x_input: TensorInput | None
    gate_input: TensorInput | None
    qkv_input: TensorInput | None

    def __init__(self, config: SigmoidMulInputConfig) -> None:
        self.config = config
        self.x_input = None
        self.gate_input = None
        self.qkv_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_dim = _check_positive("hidden_dim", self.config.hidden_dim)
        self.config.dtype = _check_torch_dtype("dtype", self.config.dtype)
        if self.config.gate_layout not in ("dense", "qkv_split"):
            raise ValueError(f"unsupported gate_layout={self.config.gate_layout!r}")

        self.x_input = self.x_input or TensorInput(
            (self.config.num_tokens, self.config.hidden_dim),
            self.config.dtype,
            device=self.config.device,
        )
        if self.config.gate_layout == "dense":
            self.gate_input = self.gate_input or TensorInput(
                (self.config.num_tokens, self.config.hidden_dim),
                self.config.dtype,
                device=self.config.device,
            )
            return

        if self.config.num_heads is None:
            raise ValueError("qkv_split gate layout requires num_heads")
        if self.config.head_dim is None:
            raise ValueError("qkv_split gate layout requires head_dim")
        if self.config.num_kv_heads is None:
            raise ValueError("qkv_split gate layout requires num_kv_heads")
        self.config.num_heads = _check_positive("num_heads", self.config.num_heads)
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        if self.config.num_heads * self.config.head_dim != self.config.hidden_dim:
            raise ValueError(
                "qkv_split requires num_heads * head_dim == hidden_dim; got "
                f"{self.config.num_heads} * {self.config.head_dim} != "
                f"{self.config.hidden_dim}"
            )
        q_size = self.config.hidden_dim
        kv_size = self.config.num_kv_heads * self.config.head_dim
        self.qkv_input = self.qkv_input or TensorInput(
            (self.config.num_tokens, 2 * q_size + 2 * kv_size),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> SigmoidMulInputValues:
        if self.x_input is None:
            raise ValueError("SigmoidMulInputs child generators must be initialized")
        x = _require_tensor(
            self.x_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "x",
        )
        if self.config.gate_layout == "dense":
            if self.gate_input is None:
                raise ValueError("dense gate generator must be initialized")
            gate = _require_tensor(
                self.gate_input.generate(
                    seed=_child_seed(seed, 2), device=device
                ).values,
                "gate",
            )
            return SigmoidMulInputValues(x=x.contiguous(), gate=gate.contiguous())

        if self.qkv_input is None:
            raise ValueError("qkv_split backing generator must be initialized")
        qkv = _require_tensor(
            self.qkv_input.generate(seed=_child_seed(seed, 2), device=device).values,
            "qkv",
        )
        q_size = self.config.hidden_dim
        kv_size = int(self.config.num_kv_heads) * int(self.config.head_dim)
        q_gate, _k, _v = qkv.split([2 * q_size, kv_size, kv_size], dim=-1)
        q_gate = q_gate.view(
            self.config.num_tokens,
            int(self.config.num_heads),
            2 * int(self.config.head_dim),
        )
        _q, gate = torch.chunk(q_gate, 2, dim=-1)
        return SigmoidMulInputValues(x=x.contiguous(), gate=gate, gate_storage=qkv)


@dataclass
class GatedActivationInputValues:
    """Generated values for ``GatedActivationInputs``."""

    x: torch.Tensor


def gated_activation_reference(
    x: torch.Tensor,
    *,
    activation: GatedActivationKind = "silu",
) -> torch.Tensor:
    """Return a float32 reference for split gated activation inputs."""

    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dimension must be even, got {x.shape[-1]}")
    gate, up = x.float().chunk(2, dim=-1)
    if activation == "silu":
        activated = torch.nn.functional.silu(gate)
    elif activation == "gelu":
        activated = torch.nn.functional.gelu(gate)
    elif activation == "gelu_tanh":
        activated = torch.nn.functional.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"unsupported activation={activation!r}")
    return activated * up


@dataclass
class GatedActivationInputConfig:
    """Initialization parameters for split gated activation inputs.

    The represented operation splits ``x[..., :hidden_dim]`` as the gate and
    ``x[..., hidden_dim:]`` as the up/value operand, then computes one of:
    ``silu(gate) * up``, ``gelu(gate) * up``, or tanh-approx GELU times up.
    """

    # Required: number of rows in the generated 2D activation tensor.
    num_tokens: int

    # Required: output width. Generated input width is 2 * hidden_dim.
    hidden_dim: int

    # Required: dtype for the generated split activation tensor.
    dtype: torch.dtype

    # Optional: activation function applied to the gate half.
    activation: GatedActivationKind = "silu"

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class GatedActivationInputs(NumericsInputGenerator):
    """Generator for split gated activations such as ``silu_and_mul``."""

    config: GatedActivationInputConfig
    x_input: TensorInput | None

    def __init__(self, config: GatedActivationInputConfig) -> None:
        self.config = config
        self.x_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_dim = _check_positive("hidden_dim", self.config.hidden_dim)
        self.config.dtype = _check_torch_dtype("dtype", self.config.dtype)
        if self.config.activation not in ("silu", "gelu", "gelu_tanh"):
            raise ValueError(f"unsupported activation={self.config.activation!r}")
        self.x_input = self.x_input or TensorInput(
            (self.config.num_tokens, 2 * self.config.hidden_dim),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> GatedActivationInputValues:
        if self.x_input is None:
            raise ValueError(
                "GatedActivationInputs child generator must be initialized"
            )
        x = _require_tensor(
            self.x_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "x",
        )
        return GatedActivationInputValues(x=x)


@dataclass
class FusedGateSigmoidMulAddInputValues:
    """Generated values for ``FusedGateSigmoidMulAddInputs``."""

    hidden_states: torch.Tensor
    gate_weight: torch.Tensor
    shared_output: torch.Tensor
    final_hidden_states: torch.Tensor


def fused_gate_sigmoid_mul_add_reference(
    values: FusedGateSigmoidMulAddInputValues,
) -> torch.Tensor:
    """Return a float32 reference for fused gate-sigmoid-mul-add inputs."""

    if values.hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be 2D, got {values.hidden_states.ndim}D")
    num_tokens, hidden_dim = values.hidden_states.shape
    if values.gate_weight.shape != (hidden_dim,):
        raise ValueError(
            f"gate_weight must have shape {(hidden_dim,)}, "
            f"got {tuple(values.gate_weight.shape)}"
        )
    expected_matrix_shape = (num_tokens, hidden_dim)
    if tuple(values.shared_output.shape) != expected_matrix_shape:
        raise ValueError(
            f"shared_output must have shape {expected_matrix_shape}, "
            f"got {tuple(values.shared_output.shape)}"
        )
    if tuple(values.final_hidden_states.shape) != expected_matrix_shape:
        raise ValueError(
            f"final_hidden_states must have shape {expected_matrix_shape}, "
            f"got {tuple(values.final_hidden_states.shape)}"
        )
    gate = (
        values.hidden_states.float() @ values.gate_weight.float().unsqueeze(1)
    ).sigmoid()
    return values.final_hidden_states.float() + gate * values.shared_output.float()


@dataclass
class FusedGateSigmoidMulAddInputConfig:
    """Initialization parameters for fused gate-sigmoid-mul-add inputs.

    The represented operation is
    ``final_hidden_states + sigmoid(hidden_states @ gate_weight) * shared_output``.
    ``gate_weight`` is generated with variance scaled by ``1 / sqrt(hidden_dim)``
    so the sigmoid gate is not usually saturated by large hidden widths.
    """

    # Required: number of token rows.
    num_tokens: int

    # Required: hidden-state width.
    hidden_dim: int

    # Required: dtype for all generated tensors.
    dtype: torch.dtype

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class FusedGateSigmoidMulAddInputs(NumericsInputGenerator):
    """Generator for fused gate-sigmoid-mul-add inputs."""

    config: FusedGateSigmoidMulAddInputConfig
    hidden_states_input: TensorInput | None
    gate_weight_input: TensorInput | None
    shared_output_input: TensorInput | None
    final_hidden_states_input: TensorInput | None

    def __init__(self, config: FusedGateSigmoidMulAddInputConfig) -> None:
        self.config = config
        self.hidden_states_input = None
        self.gate_weight_input = None
        self.shared_output_input = None
        self.final_hidden_states_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_dim = _check_positive("hidden_dim", self.config.hidden_dim)
        self.config.dtype = _check_torch_dtype("dtype", self.config.dtype)
        matrix_shape = (self.config.num_tokens, self.config.hidden_dim)
        self.hidden_states_input = self.hidden_states_input or TensorInput(
            matrix_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.gate_weight_input = self.gate_weight_input or TensorInput(
            (self.config.hidden_dim,),
            self.config.dtype,
            device=self.config.device,
        )
        self.shared_output_input = self.shared_output_input or TensorInput(
            matrix_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.final_hidden_states_input = self.final_hidden_states_input or TensorInput(
            matrix_shape, self.config.dtype, device=self.config.device
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> FusedGateSigmoidMulAddInputValues:
        if (
            self.hidden_states_input is None
            or self.gate_weight_input is None
            or self.shared_output_input is None
            or self.final_hidden_states_input is None
        ):
            raise ValueError(
                "FusedGateSigmoidMulAddInputs child generators must be initialized"
            )
        hidden_states = _require_tensor(
            self.hidden_states_input.generate(
                seed=_child_seed(seed, 1),
                device=device,
            ).values,
            "hidden_states",
        )
        gate_weight = _require_tensor(
            self.gate_weight_input.generate(
                seed=_child_seed(seed, 2),
                device=device,
            ).values,
            "gate_weight",
        )
        gate_weight = (gate_weight.float() / math.sqrt(self.config.hidden_dim)).to(
            self.config.dtype
        )
        shared_output = _require_tensor(
            self.shared_output_input.generate(
                seed=_child_seed(seed, 3),
                device=device,
            ).values,
            "shared_output",
        )
        final_hidden_states = _require_tensor(
            self.final_hidden_states_input.generate(
                seed=_child_seed(seed, 4),
                device=device,
            ).values,
            "final_hidden_states",
        )
        return FusedGateSigmoidMulAddInputValues(
            hidden_states=hidden_states.contiguous(),
            gate_weight=gate_weight.contiguous(),
            shared_output=shared_output.contiguous(),
            final_hidden_states=final_hidden_states.contiguous(),
        )


@dataclass
class FusedSwiGLUFP8UE8M0InputValues:
    """Generated values for ``FusedSwiGLUFP8UE8M0Inputs``."""

    gate_up: torch.Tensor
    swiglu_limit: float
    group_size: int


@dataclass
class FusedSwiGLUFP8UE8M0InputConfig:
    """Initialization parameters for fused SwiGLU + FP8/UE8M0 quant inputs.

    The represented operation computes ``silu(gate) * up`` from a
    ``[num_tokens, 2 * hidden_dim]`` split tensor and then quantizes the result
    by per-token groups. ``hidden_dim`` must be divisible by ``group_size``.
    """

    # Required: number of token rows.
    num_tokens: int

    # Required: output width after splitting gate/up. Must be divisible by
    # group_size.
    hidden_dim: int

    # Required: dtype for generated gate/up values.
    dtype: torch.dtype

    # Optional: group width used for block quantization.
    group_size: int = 128

    # Optional: clamp bound. Non-positive values disable clamping.
    swiglu_limit: float = 0.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class FusedSwiGLUFP8UE8M0Inputs(NumericsInputGenerator):
    """Generator for fused SwiGLU + FP8/UE8M0 quantization inputs."""

    config: FusedSwiGLUFP8UE8M0InputConfig
    gate_up_input: TensorInput | None

    def __init__(self, config: FusedSwiGLUFP8UE8M0InputConfig) -> None:
        self.config = config
        self.gate_up_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_dim = _check_positive("hidden_dim", self.config.hidden_dim)
        self.config.group_size = _check_positive("group_size", self.config.group_size)
        self.config.dtype = _check_torch_dtype("dtype", self.config.dtype)
        if self.config.hidden_dim % self.config.group_size != 0:
            raise ValueError(
                "hidden_dim must be divisible by group_size for FP8/UE8M0 "
                f"quantization, got {self.config.hidden_dim} and "
                f"{self.config.group_size}"
            )
        self.config.swiglu_limit = float(self.config.swiglu_limit)
        self.gate_up_input = self.gate_up_input or TensorInput(
            (self.config.num_tokens, 2 * self.config.hidden_dim),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> FusedSwiGLUFP8UE8M0InputValues:
        if self.gate_up_input is None:
            raise ValueError(
                "FusedSwiGLUFP8UE8M0Inputs child generator must be initialized"
            )
        target_device = _resolve_device(self.config.device, device)
        gate_up = _require_tensor(
            self.gate_up_input.generate(
                seed=_child_seed(seed, 1),
                device=target_device,
            ).values,
            "gate_up",
        )
        return FusedSwiGLUFP8UE8M0InputValues(
            gate_up=gate_up,
            swiglu_limit=self.config.swiglu_limit,
            group_size=self.config.group_size,
        )


def fused_swiglu_fp8_ue8m0_reference(
    values: FusedSwiGLUFP8UE8M0InputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 SwiGLU values and packed UE8M0 group scales."""

    if values.gate_up.ndim != 2:
        raise ValueError(f"gate_up must be 2D, got {values.gate_up.ndim}D")
    if values.gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"gate_up last dimension must be even, got {values.gate_up.shape[-1]}"
        )
    group_size = _check_positive("group_size", values.group_size)
    num_tokens, two_hidden_dim = values.gate_up.shape
    hidden_dim = two_hidden_dim // 2
    if hidden_dim % group_size != 0:
        raise ValueError(
            "hidden_dim must be divisible by group_size for FP8/UE8M0 "
            f"quantization, got {hidden_dim} and {group_size}"
        )

    gate, up = values.gate_up.float().chunk(2, dim=-1)
    if values.swiglu_limit > 0.0:
        gate = gate.clamp(max=values.swiglu_limit)
        up = up.clamp(min=-values.swiglu_limit, max=values.swiglu_limit)
    activated = torch.nn.functional.silu(gate) * up

    groups_per_row = hidden_dim // group_size
    grouped = activated.reshape(num_tokens, groups_per_row, group_size)
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    scale_raw = grouped.abs().amax(dim=-1) / fp8_info.max
    exponent = torch.ceil(torch.log2(scale_raw.clamp(min=1.0e-10)))
    scale = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=values.gate_up.device),
        exponent,
    )
    quantized = torch.clamp(
        grouped / scale.unsqueeze(-1),
        min=fp8_info.min,
        max=fp8_info.max,
    ).to(fp8_dtype)

    packed_scale_cols = (groups_per_row + 3) // 4
    packed_scales = torch.zeros(
        (num_tokens, packed_scale_cols),
        dtype=torch.int32,
        device=values.gate_up.device,
    )
    biased_exponent = (exponent + 127.0).clamp(min=0.0, max=255.0).to(torch.int32)
    for group_idx in range(groups_per_row):
        packed_col = group_idx // 4
        packed_pos = group_idx % 4
        packed_scales[:, packed_col] |= biased_exponent[:, group_idx] << (
            packed_pos * 8
        )

    return quantized.reshape(num_tokens, hidden_dim).contiguous(), packed_scales
