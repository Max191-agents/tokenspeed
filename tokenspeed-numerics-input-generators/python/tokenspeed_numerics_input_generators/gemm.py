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

"""GEMM-family input generators for numerical correctness tests."""

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
    TensorValues,
    _child_seed,
    _normalize_shape,
    _packed_mxfp4_shape,
    _packed_mxint4_shape,
)
from tokenspeed_numerics_input_generators.quantization import (
    nvfp4_dequantization_reference,
    nvfp4_quantization_reference,
)

__all__ = [
    "GemmInputs",
    "GemmInputConfig",
    "GemmInputValues",
    "LMHeadProjectionInputConfig",
    "LMHeadProjectionInputs",
    "LMHeadProjectionInputValues",
    "NVFP4GemmSwiGLUNVFP4QuantInputConfig",
    "NVFP4GemmSwiGLUNVFP4QuantInputs",
    "NVFP4GemmSwiGLUNVFP4QuantInputValues",
    "RouterProjectionInputConfig",
    "RouterProjectionInputs",
    "RouterProjectionInputValues",
    "gemm_reference",
    "gemm_scale_shape",
    "lm_head_projection_reference",
    "mxfp4_gemm_input_config",
    "mxfp8_gemm_input_config",
    "mxint4_gemm_input_config",
    "nvfp4_gemm_swiglu_nvfp4_quant_reference",
    "router_projection_reference",
]

GemmLayout = Literal["MK", "KM", "NK", "KN"]
ScaleGranularity = Literal["tensor", "channel", "block"]

_DEFAULT_MXFP4_BLOCK_SIZE = 32
_DEFAULT_MXINT4_BLOCK_SIZE = 32
_DEFAULT_NVFP4_BLOCK_SIZE = 16
_GEMM_LAYOUTS = frozenset({"MK", "KM", "NK", "KN"})
_PROJECTION_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


def _gemm_value_shape(
    *,
    role: Literal["a", "b", "c"],
    M: int,
    N: int,
    K: int,
    batch_shape: tuple[int, ...],
    layout: GemmLayout,
    dtype: InputDType = None,
) -> tuple[int, ...]:
    if role == "a":
        shape = (K, M) if layout == "KM" else (M, K)
        packed_dim = 0 if layout == "KM" else 1
    elif role == "b":
        shape = (K, N) if layout == "KN" else (N, K)
        packed_dim = 0 if layout == "KN" else 1
    else:
        return batch_shape + (M, N)

    if dtype == CustomDType.MXFP4:
        shape = _packed_mxfp4_shape(shape, packed_dim=packed_dim)
    if dtype == CustomDType.MXINT4:
        shape = _packed_mxint4_shape(shape, packed_dim=packed_dim)
    return batch_shape + shape


def gemm_scale_shape(
    granularity: ScaleGranularity | str | None,
    role: Literal["a", "b"],
    *,
    M: int,
    N: int,
    K: int,
    batch_shape: tuple[int, ...] = (),
    block_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...] | None:
    """Return the physical scale shape for a GEMM operand role."""

    if granularity is None:
        return None

    if granularity == "tensor":
        return (1,)

    if granularity == "channel":
        return batch_shape + ((M,) if role == "a" else (N,))

    if granularity != "block":
        raise ValueError(f"unsupported GEMM scale granularity {granularity!r}")
    if block_shape is None:
        raise ValueError("block scale format requires concrete block_shape")

    block_shape = tuple(block_shape)
    if not block_shape or any(dim <= 0 for dim in block_shape):
        raise ValueError("block_shape must contain positive dimensions")

    if len(block_shape) == 1:
        block_k = block_shape[0]
        return batch_shape + (
            (M, math.ceil(K / block_k)) if role == "a" else (N, math.ceil(K / block_k))
        )

    if len(block_shape) == 2:
        block_n, block_k = block_shape
        if role == "a":
            return batch_shape + (M, math.ceil(K / block_k))
        return batch_shape + (math.ceil(N / block_n), math.ceil(K / block_k))

    raise ValueError(
        f"GEMM scale format supports 1D or 2D block shapes, got {block_shape}"
    )


def _generate_tensor(
    *,
    shape: tuple[int, ...],
    dtype: InputDType,
    seed: int,
    configured_device: DeviceLike,
    device: DeviceLike,
) -> torch.Tensor | None:
    if isinstance(dtype, CustomDType):
        return (
            TensorInput(shape, dtype, device=configured_device)
            .generate(
                seed=seed,
                device=device,
            )
            .values
        )
    if dtype is None or isinstance(dtype, torch.dtype):
        return (
            TensorInput(
                shape,
                dtype,
                device=configured_device,
            )
            .generate(seed=seed, device=device)
            .values
        )
    raise TypeError(f"unsupported GEMM dtype={dtype!r}")


def _check_gemm_layout(name: str, layout: GemmLayout) -> GemmLayout:
    if layout not in _GEMM_LAYOUTS:
        raise ValueError(
            f"{name} must be one of {sorted(_GEMM_LAYOUTS)}, got {layout!r}"
        )
    return layout


def _mxfp4_e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    magnitude_bits = nibbles & 0x7
    exponent = (magnitude_bits >> 1).to(torch.float32)
    mantissa = (magnitude_bits & 0x1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent - 1.0)
    subnormal = 0.5 * mantissa
    magnitude = torch.where(exponent == 0, subnormal, normal)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
    return magnitude * sign


def _dequantize_mxfp4_linear(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 values and UE8M0 scales must use torch.uint8 storage")
    if packed.ndim < 2:
        raise ValueError("MXFP4 GEMM operands must be at least 2D")
    if scales.shape[:-1] != packed.shape[:-1]:
        raise ValueError(
            "MXFP4 scale rows must match packed value rows; "
            f"packed={tuple(packed.shape)}, scales={tuple(scales.shape)}"
        )

    out = packed.new_empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.float32,
    )
    out[..., 0::2] = _mxfp4_e2m1_values(packed & 0xF)
    out[..., 1::2] = _mxfp4_e2m1_values(packed >> 4)

    expected_groups = math.ceil(out.shape[-1] / _DEFAULT_MXFP4_BLOCK_SIZE)
    if scales.shape[-1] != expected_groups:
        raise ValueError(
            "MXFP4 scale group count must match K dimension; "
            f"got scales={tuple(scales.shape)}, values={tuple(out.shape)}"
        )
    scale_values = torch.pow(2.0, scales.to(torch.int32) - 127).to(torch.float32)
    return (
        out
        * scale_values.repeat_interleave(
            _DEFAULT_MXFP4_BLOCK_SIZE,
            dim=-1,
        )[..., : out.shape[-1]]
    )


def _signed_int4_values_from_packed(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0xF
    high = packed >> 4
    nibbles = packed.new_empty((*packed.shape[:-1], packed.shape[-1] * 2))
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high
    signed = nibbles.to(torch.int8)
    return torch.where(signed >= 8, signed - 16, signed).to(torch.float32)


def _dequantize_mxint4_linear(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if packed.dtype != torch.uint8 or scales.dtype != torch.bfloat16:
        raise ValueError("MXINT4 values must be uint8 and scales must be bfloat16")
    if packed.ndim < 2:
        raise ValueError("MXINT4 GEMM operands must be at least 2D")
    if scales.shape[:-1] != packed.shape[:-1]:
        raise ValueError(
            "MXINT4 scale rows must match packed value rows; "
            f"packed={tuple(packed.shape)}, scales={tuple(scales.shape)}"
        )

    values = _signed_int4_values_from_packed(packed)
    expected_groups = math.ceil(values.shape[-1] / _DEFAULT_MXINT4_BLOCK_SIZE)
    if scales.shape[-1] != expected_groups:
        raise ValueError(
            "MXINT4 scale group count must match K dimension; "
            f"got scales={tuple(scales.shape)}, values={tuple(values.shape)}"
        )
    return (
        values
        * scales.float().repeat_interleave(_DEFAULT_MXINT4_BLOCK_SIZE, dim=-1)[
            ..., : values.shape[-1]
        ]
    )


def _apply_regular_scales(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale_values = scales.float()
    if scale_values.numel() == 1:
        return values * scale_values.reshape((1,) * values.ndim)
    if scale_values.shape == values.shape[:-1]:
        return values * scale_values.unsqueeze(-1)
    if scale_values.ndim == 1 and scale_values.shape[0] == values.shape[-2]:
        return values * scale_values.reshape(*values.shape[:-2], values.shape[-2], 1)
    if scale_values.ndim == values.ndim:
        row_groups, k_groups = scale_values.shape[-2:]
        if row_groups <= 0 or k_groups <= 0:
            raise ValueError("scale group counts must be positive")
        if row_groups > values.shape[-2] or k_groups > values.shape[-1]:
            raise ValueError(
                "scale groups cannot exceed value shape; "
                f"values={tuple(values.shape)}, scales={tuple(scales.shape)}"
            )
        if values.shape[-2] % row_groups != 0 or values.shape[-1] % k_groups != 0:
            raise ValueError(
                "regular GEMM scale grids must evenly partition value rows and K; "
                f"values={tuple(values.shape)}, scales={tuple(scales.shape)}"
            )
        row_repeat = values.shape[-2] // row_groups
        k_repeat = values.shape[-1] // k_groups
        expanded = scale_values.repeat_interleave(
            row_repeat,
            dim=-2,
        ).repeat_interleave(k_repeat, dim=-1)
        return values * expanded
    raise ValueError(
        "unsupported GEMM scale shape for reference; "
        f"values={tuple(values.shape)}, scales={tuple(scales.shape)}"
    )


def _check_nvfp4_source_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"NVFP4 fused GEMM input dtype must be bf16 or fp16, got {dtype}"
        )
    return dtype


def _check_projection_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _PROJECTION_DTYPES:
        raise ValueError(
            f"{name} must be one of {sorted(str(d) for d in _PROJECTION_DTYPES)}, "
            f"got {dtype}"
        )
    return dtype


def _check_positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nvfp4_global_scale_for(tensor: torch.Tensor) -> torch.Tensor:
    scale = (tensor.detach().abs().amax().to(torch.float32) / (448.0 * 6.0)).clamp(
        min=1.0e-8
    )
    return scale.reshape(1).to(device=tensor.device, dtype=torch.float32)


def _silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _logical_operand(
    values: torch.Tensor,
    scales: torch.Tensor | None,
    *,
    layout: GemmLayout,
    is_mxfp4: bool,
    is_mxint4: bool = False,
) -> torch.Tensor:
    if is_mxfp4 and is_mxint4:
        raise ValueError("GEMM operand cannot be both MXFP4 and MXINT4")
    if is_mxfp4:
        if layout not in ("MK", "NK"):
            raise ValueError(
                "MXFP4 GEMM reference currently supports row-major MK/NK operands"
            )
        if scales is None:
            raise ValueError("MXFP4 GEMM reference requires scales")
        logical = _dequantize_mxfp4_linear(values, scales)
    elif is_mxint4:
        if layout not in ("MK", "NK"):
            raise ValueError(
                "MXINT4 GEMM reference currently supports row-major MK/NK operands"
            )
        if scales is None:
            raise ValueError("MXINT4 GEMM reference requires scales")
        logical = _dequantize_mxint4_linear(values, scales)
    else:
        logical = values.float()
        if scales is not None:
            logical = _apply_regular_scales(logical, scales)

    if layout in ("KM", "KN"):
        logical = logical.transpose(-1, -2)
    return logical.contiguous()


@dataclass
class GemmInputValues:
    """Generated values for ``GemmInputs``."""

    A: torch.Tensor | None
    B: torch.Tensor | None
    C: torch.Tensor
    A_scales: torch.Tensor | None = None
    B_scales: torch.Tensor | None = None


@dataclass
class GemmInputConfig:
    """Initialization parameters for ``GemmInputs``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: logical M dimension for A @ B.T style GEMM generation.
    M: int

    # Required: logical N dimension for A @ B.T style GEMM generation.
    N: int

    # Required: logical K reduction dimension.
    K: int

    # Required: dtype for generated A. ``None`` skips A generation. Custom
    # dtypes generate their required storage representation directly.
    a_dtype: InputDType

    # Required: dtype for generated B. ``None`` skips B generation. Custom
    # dtypes generate their required storage representation directly.
    b_dtype: InputDType

    # Required: dtype for generated C/output accumulator.
    c_dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional operand and layout configuration.
    # ------------------------------------------------------------------

    # Optional: physical A layout.
    a_layout: GemmLayout = "MK"

    # Optional: physical B layout.
    b_layout: GemmLayout = "NK"

    # Optional: batch prefix for generated operands, e.g. expert dimension.
    batch_shape: tuple[int, ...] = ()

    # Optional: physical A scale shape. ``None`` skips A scale generation unless
    # A uses a custom dtype that requires scales.
    a_scale_shape: tuple[int, ...] | None = None

    # Optional: physical B scale shape. ``None`` skips B scale generation unless
    # B uses a custom dtype that requires scales.
    b_scale_shape: tuple[int, ...] | None = None

    # Optional: A scale dtype. ``None`` skips A scales unless A uses a custom
    # dtype with an inferred scale dtype, such as MXFP4.
    a_scale_dtype: InputDType = None

    # Optional: B scale dtype. ``None`` skips B scales unless B uses a custom
    # dtype with an inferred scale dtype, such as MXFP4.
    b_scale_dtype: InputDType = None

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: generated A device override.
    a_device: DeviceLike = None

    # Optional: generated B device override.
    b_device: DeviceLike = None

    # Optional: generated A scale device override.
    a_scale_device: DeviceLike = None

    # Optional: generated B scale device override.
    b_scale_device: DeviceLike = None

    # Optional: generated C device override.
    c_device: DeviceLike = None


@dataclass(init=False)
class GemmInputs(NumericsInputGenerator):
    """Typed input generator for dense GEMM-like calls.

    ``M``, ``N``, and ``K`` are the logical matrix dimensions for ``A @ B.T``
    style kernels used by TokenSpeed. ``generate`` returns generated ``A``,
    ``B``, ``C``, and optional scale tensors. ``A`` and ``B`` may use dtype
    ``None`` to skip generation when a layer generator only needs one side of
    the GEMM, but ``C`` is always generated.

    Mutate ``generator.config`` to adjust dtype, scale dtype, shape, layout, or
    device before calling ``generate``.
    """

    config: GemmInputConfig

    def __init__(
        self,
        config: GemmInputConfig,
    ) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.M = int(self.config.M)
        self.config.N = int(self.config.N)
        self.config.K = int(self.config.K)
        if min(self.config.M, self.config.N, self.config.K) < 0:
            raise ValueError("M, N, and K must be non-negative")
        self.config.a_layout = _check_gemm_layout("a_layout", self.config.a_layout)
        self.config.b_layout = _check_gemm_layout("b_layout", self.config.b_layout)
        if self.config.a_dtype == CustomDType.MXFP4 and self.config.a_layout != "MK":
            raise ValueError("MXFP4 A operands currently require a_layout='MK'")
        if self.config.b_dtype == CustomDType.MXFP4 and self.config.b_layout != "NK":
            raise ValueError("MXFP4 B operands currently require b_layout='NK'")
        if self.config.a_dtype == CustomDType.MXINT4 and self.config.a_layout != "MK":
            raise ValueError("MXINT4 A operands currently require a_layout='MK'")
        if self.config.b_dtype == CustomDType.MXINT4 and self.config.b_layout != "NK":
            raise ValueError("MXINT4 B operands currently require b_layout='NK'")
        if self.config.c_dtype is None:
            raise ValueError("c_dtype is required")
        if not isinstance(self.config.c_dtype, torch.dtype):
            raise TypeError("c_dtype must be a torch.dtype")
        self.config.batch_shape = _normalize_shape(self.config.batch_shape)
        if self.config.a_scale_shape is not None:
            self.config.a_scale_shape = _normalize_shape(self.config.a_scale_shape)
        if self.config.b_scale_shape is not None:
            self.config.b_scale_shape = _normalize_shape(self.config.b_scale_shape)
        if self.config.a_dtype is None and (
            self.config.a_scale_shape is not None
            or self.config.a_scale_dtype is not None
        ):
            raise ValueError("a_dtype=None skips A values and A scales")
        if self.config.b_dtype is None and (
            self.config.b_scale_shape is not None
            or self.config.b_scale_dtype is not None
        ):
            raise ValueError("b_dtype=None skips B values and B scales")

    def _value_shape(self, role: Literal["a", "b", "c"]) -> tuple[int, ...]:
        if role == "a":
            return _gemm_value_shape(
                role="a",
                M=self.config.M,
                N=self.config.N,
                K=self.config.K,
                batch_shape=self.config.batch_shape,
                layout=self.config.a_layout,
                dtype=self.config.a_dtype,
            )
        if role == "b":
            return _gemm_value_shape(
                role="b",
                M=self.config.M,
                N=self.config.N,
                K=self.config.K,
                batch_shape=self.config.batch_shape,
                layout=self.config.b_layout,
                dtype=self.config.b_dtype,
            )
        return _gemm_value_shape(
            role="c",
            M=self.config.M,
            N=self.config.N,
            K=self.config.K,
            batch_shape=self.config.batch_shape,
            layout="MK",
        )

    def _generate_operand(
        self,
        role: Literal["a", "b"],
        *,
        seed: int,
        device: DeviceLike,
    ) -> TensorValues:
        is_a = role == "a"
        return TensorInput(
            self._value_shape(role),
            self.config.a_dtype if is_a else self.config.b_dtype,
            scale_shape=(
                self.config.a_scale_shape if is_a else self.config.b_scale_shape
            ),
            scale_dtype=(
                self.config.a_scale_dtype if is_a else self.config.b_scale_dtype
            ),
            device=self.config.a_device if is_a else self.config.b_device,
            scale_device=(
                self.config.a_scale_device if is_a else self.config.b_scale_device
            ),
        ).generate(seed=seed, device=device)

    def generate(self, *, seed: int, device: DeviceLike = None) -> GemmInputValues:
        a_values = self._generate_operand("a", seed=_child_seed(seed, 1), device=device)
        b_values = self._generate_operand("b", seed=_child_seed(seed, 2), device=device)
        C = _generate_tensor(
            shape=self._value_shape("c"),
            dtype=self.config.c_dtype,
            seed=_child_seed(seed, 3),
            configured_device=self.config.c_device,
            device=device,
        )
        if C is None:
            raise ValueError("c_dtype is required")
        return GemmInputValues(
            A=a_values.values,
            B=b_values.values,
            C=C,
            A_scales=a_values.scales,
            B_scales=b_values.scales,
        )


def gemm_reference(
    values: GemmInputValues,
    *,
    a_layout: GemmLayout = "MK",
    b_layout: GemmLayout = "NK",
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | float | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the semantic ``A @ B.T`` GEMM result for generated values.

    Dense and scaled torch tensors are converted to fp32 and multiplied by
    their scale sidecars before the matmul. MXFP4 operands are unpacked from
    E2M1x2 bytes and dequantized with UE8M0 scales. MXINT4 operands are
    unpacked from signed INT4 nibbles and dequantized with BF16 group scales.
    The returned dtype defaults to the generated ``C`` dtype, matching
    TokenSpeed's ``out_dtype`` role.
    """

    a_layout = _check_gemm_layout("a_layout", a_layout)
    b_layout = _check_gemm_layout("b_layout", b_layout)
    if values.A is None or values.B is None:
        raise ValueError("gemm_reference requires generated A and B operands")
    out_dtype = values.C.dtype if out_dtype is None else out_dtype
    if not isinstance(out_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")

    a_is_mxfp4 = values.A.dtype == torch.uint8 and values.A_scales is not None
    b_is_mxfp4 = values.B.dtype == torch.uint8 and values.B_scales is not None
    a_is_mxint4 = (
        values.A.dtype == torch.uint8
        and values.A_scales is not None
        and values.A_scales.dtype == torch.bfloat16
    )
    b_is_mxint4 = (
        values.B.dtype == torch.uint8
        and values.B_scales is not None
        and values.B_scales.dtype == torch.bfloat16
    )
    a_is_mxfp4 = a_is_mxfp4 and not a_is_mxint4
    b_is_mxfp4 = b_is_mxfp4 and not b_is_mxint4
    A = _logical_operand(
        values.A,
        values.A_scales,
        layout=a_layout,
        is_mxfp4=a_is_mxfp4,
        is_mxint4=a_is_mxint4,
    )
    B = _logical_operand(
        values.B,
        values.B_scales,
        layout=b_layout,
        is_mxfp4=b_is_mxfp4,
        is_mxint4=b_is_mxint4,
    )
    if A.shape[-1] != B.shape[-1]:
        raise ValueError(
            "GEMM K dimensions must match after layout normalization; "
            f"A={tuple(A.shape)}, B={tuple(B.shape)}"
        )

    output = A.float() @ B.float().transpose(-1, -2)
    if alpha is not None:
        alpha_tensor = (
            alpha
            if isinstance(alpha, torch.Tensor)
            else torch.tensor(alpha, dtype=torch.float32, device=output.device)
        )
        output = output * alpha_tensor.to(device=output.device, dtype=output.dtype)
    if bias is not None:
        if bias.shape[-1] != output.shape[-1]:
            raise ValueError(
                "bias last dimension must match GEMM N dimension; "
                f"bias={tuple(bias.shape)}, output={tuple(output.shape)}"
            )
        output = output + bias.to(device=output.device, dtype=output.dtype)
    return output.to(out_dtype)


@dataclass
class RouterProjectionInputValues:
    """Generated values for router projection.

    ``hidden_states`` contains per-token hidden activations. ``router_weights``
    contains one projection row per expert. The represented operation computes
    fp32 router logits as ``hidden_states @ router_weights.T``.
    """

    hidden_states: torch.Tensor
    router_weights: torch.Tensor


@dataclass
class RouterProjectionInputConfig:
    """Initialization parameters for ``RouterProjectionInputs``.

    The generator models the unquantized router projection used before MoE
    routing. Generated weights are scaled by ``1 / sqrt(hidden_dim)`` by default
    so generated logits have roughly unit variance instead of growing with the
    reduction dimension.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows to route.
    num_tokens: int

    # Required: hidden-state width and router projection reduction dimension.
    hidden_dim: int

    # Required: number of expert logits produced for each token.
    num_experts: int

    # ------------------------------------------------------------------
    # Optional dtype and value-distribution configuration.
    # ------------------------------------------------------------------

    # Optional: dtype for generated hidden-state activations.
    hidden_dtype: torch.dtype = torch.bfloat16

    # Optional: dtype for generated router projection weights.
    router_weight_dtype: torch.dtype = torch.float32

    # Optional: multiplicative scale applied to generated hidden states.
    hidden_scale: float = 1.0

    # Optional: multiplicative scale applied to generated router weights. When
    # omitted, generation uses 1 / sqrt(hidden_dim).
    router_weight_scale: float | None = None

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class RouterProjectionInputs(NumericsInputGenerator):
    """Generator for MoE router projection inputs.

    The represented operation is:

    ``router_logits = hidden_states @ router_weights.T``

    This is a GEMM-family operation, but the named generator captures the
    layer-level meaning of each operand and chooses a default value range that
    keeps downstream routing tests numerically useful.
    """

    config: RouterProjectionInputConfig
    hidden_states_input: TensorInput | None
    router_weights_input: TensorInput | None

    def __init__(self, config: RouterProjectionInputConfig) -> None:
        self.config = config
        self.hidden_states_input = None
        self.router_weights_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = int(self.config.num_tokens)
        self.config.hidden_dim = int(self.config.hidden_dim)
        self.config.num_experts = int(self.config.num_experts)
        if self.config.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.config.hidden_dtype = _check_projection_dtype(
            "hidden_dtype",
            self.config.hidden_dtype,
        )
        self.config.router_weight_dtype = _check_projection_dtype(
            "router_weight_dtype",
            self.config.router_weight_dtype,
        )
        self.config.hidden_scale = _check_positive_float(
            "hidden_scale",
            self.config.hidden_scale,
        )
        if self.config.router_weight_scale is not None:
            self.config.router_weight_scale = _check_positive_float(
                "router_weight_scale",
                self.config.router_weight_scale,
            )
        self.hidden_states_input = self.hidden_states_input or TensorInput(
            (self.config.num_tokens, self.config.hidden_dim),
            self.config.hidden_dtype,
            device=self.config.device,
        )
        self.router_weights_input = self.router_weights_input or TensorInput(
            (self.config.num_experts, self.config.hidden_dim),
            self.config.router_weight_dtype,
            device=self.config.device,
        )
        self.hidden_states_input.shape = (
            self.config.num_tokens,
            self.config.hidden_dim,
        )
        self.hidden_states_input.dtype = self.config.hidden_dtype
        self.hidden_states_input.device = self.config.device
        self.router_weights_input.shape = (
            self.config.num_experts,
            self.config.hidden_dim,
        )
        self.router_weights_input.dtype = self.config.router_weight_dtype
        self.router_weights_input.device = self.config.device

    def _router_weight_scale(self) -> float:
        if self.config.router_weight_scale is not None:
            return self.config.router_weight_scale
        return 1.0 / math.sqrt(self.config.hidden_dim)

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> RouterProjectionInputValues:
        self.__post_init__()
        if self.hidden_states_input is None or self.router_weights_input is None:
            raise ValueError("router projection child generators must be initialized")

        hidden_states = self.hidden_states_input.generate(
            seed=_child_seed(seed, 1),
            device=device,
        ).values
        router_weights = self.router_weights_input.generate(
            seed=_child_seed(seed, 2),
            device=device,
        ).values
        if hidden_states is None or router_weights is None:
            raise ValueError("router projection tensors must not be skipped")

        hidden_states = (
            hidden_states.float()
            .mul(self.config.hidden_scale)
            .to(self.config.hidden_dtype)
            .contiguous()
        )
        router_weights = (
            router_weights.float()
            .mul(self._router_weight_scale())
            .to(self.config.router_weight_dtype)
            .contiguous()
        )
        values = RouterProjectionInputValues(
            hidden_states=hidden_states,
            router_weights=router_weights,
        )
        _check_router_projection_values(values)
        return values


def _check_router_projection_values(values: RouterProjectionInputValues) -> None:
    if values.hidden_states.ndim != 2:
        raise ValueError("hidden_states must be rank-2 [num_tokens, hidden_dim]")
    if values.router_weights.ndim != 2:
        raise ValueError("router_weights must be rank-2 [num_experts, hidden_dim]")
    if values.hidden_states.shape[0] <= 0:
        raise ValueError("hidden_states must contain at least one token row")
    if values.hidden_states.shape[1] <= 0:
        raise ValueError("hidden_dim must be positive")
    if values.router_weights.shape[0] <= 0:
        raise ValueError("router_weights must contain at least one expert row")
    if values.router_weights.shape[1] != values.hidden_states.shape[1]:
        raise ValueError(
            "router projection hidden dimensions must match; "
            f"hidden_states={tuple(values.hidden_states.shape)}, "
            f"router_weights={tuple(values.router_weights.shape)}"
        )
    _check_projection_dtype("hidden_states dtype", values.hidden_states.dtype)
    _check_projection_dtype(
        "router_weights dtype",
        values.router_weights.dtype,
    )
    if not torch.isfinite(values.hidden_states.float()).all():
        raise ValueError("hidden_states must be finite")
    if not torch.isfinite(values.router_weights.float()).all():
        raise ValueError("router_weights must be finite")


def router_projection_reference(
    values: RouterProjectionInputValues,
) -> torch.Tensor:
    """Return fp32 router logits for ``hidden_states @ router_weights.T``."""

    _check_router_projection_values(values)
    return (values.hidden_states.float() @ values.router_weights.float().T).float()


@dataclass
class LMHeadProjectionInputValues:
    """Generated values for LM-head projection.

    ``hidden_states`` contains token activations. ``weight`` contains one row
    per vocabulary entry, or per vocabulary shard for tensor-parallel adapters.
    The represented operation computes logits as ``hidden_states @ weight.T``.
    """

    hidden_states: torch.Tensor
    weight: torch.Tensor


@dataclass
class LMHeadProjectionInputConfig:
    """Initialization parameters for ``LMHeadProjectionInputs``.

    The generator models the final language-model head projection from hidden
    states to vocabulary logits. Generated weights are scaled by
    ``1 / sqrt(hidden_dim)`` by default so logits remain in a numerically useful
    range as the reduction dimension changes.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows to project.
    num_tokens: int

    # Required: hidden-state width and projection reduction dimension.
    hidden_dim: int

    # Required: number of vocabulary rows generated. Tensor-parallel adapters
    # can use this as the local vocabulary shard size.
    vocab_size: int

    # ------------------------------------------------------------------
    # Optional dtype and value-distribution configuration.
    # ------------------------------------------------------------------

    # Optional: dtype for generated hidden-state activations.
    hidden_dtype: torch.dtype = torch.bfloat16

    # Optional: dtype for generated LM-head weights.
    weight_dtype: torch.dtype = torch.bfloat16

    # Optional: multiplicative scale applied to generated hidden states.
    hidden_scale: float = 1.0

    # Optional: multiplicative scale applied to generated weights. When
    # omitted, generation uses 1 / sqrt(hidden_dim).
    weight_scale: float | None = None

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class LMHeadProjectionInputs(NumericsInputGenerator):
    """Generator for LM-head projection inputs.

    The represented operation is:

    ``logits = hidden_states @ weight.T``

    The generator does not encode backend-specific constraints such as compiled
    vocab-shard sizes or fused-kernel token limits. Those checks belong in the
    adapter for a specific implementation.
    """

    config: LMHeadProjectionInputConfig
    hidden_states_input: TensorInput | None
    weight_input: TensorInput | None

    def __init__(self, config: LMHeadProjectionInputConfig) -> None:
        self.config = config
        self.hidden_states_input = None
        self.weight_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = int(self.config.num_tokens)
        self.config.hidden_dim = int(self.config.hidden_dim)
        self.config.vocab_size = int(self.config.vocab_size)
        if self.config.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config.hidden_dtype = _check_projection_dtype(
            "hidden_dtype",
            self.config.hidden_dtype,
        )
        self.config.weight_dtype = _check_projection_dtype(
            "weight_dtype",
            self.config.weight_dtype,
        )
        self.config.hidden_scale = _check_positive_float(
            "hidden_scale",
            self.config.hidden_scale,
        )
        if self.config.weight_scale is not None:
            self.config.weight_scale = _check_positive_float(
                "weight_scale",
                self.config.weight_scale,
            )
        self.hidden_states_input = self.hidden_states_input or TensorInput(
            (self.config.num_tokens, self.config.hidden_dim),
            self.config.hidden_dtype,
            device=self.config.device,
        )
        self.weight_input = self.weight_input or TensorInput(
            (self.config.vocab_size, self.config.hidden_dim),
            self.config.weight_dtype,
            device=self.config.device,
        )
        self.hidden_states_input.shape = (
            self.config.num_tokens,
            self.config.hidden_dim,
        )
        self.hidden_states_input.dtype = self.config.hidden_dtype
        self.hidden_states_input.device = self.config.device
        self.weight_input.shape = (
            self.config.vocab_size,
            self.config.hidden_dim,
        )
        self.weight_input.dtype = self.config.weight_dtype
        self.weight_input.device = self.config.device

    def _weight_scale(self) -> float:
        if self.config.weight_scale is not None:
            return self.config.weight_scale
        return 1.0 / math.sqrt(self.config.hidden_dim)

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> LMHeadProjectionInputValues:
        self.__post_init__()
        if self.hidden_states_input is None or self.weight_input is None:
            raise ValueError("LM-head projection child generators must be initialized")

        hidden_states = self.hidden_states_input.generate(
            seed=_child_seed(seed, 1),
            device=device,
        ).values
        weight = self.weight_input.generate(
            seed=_child_seed(seed, 2),
            device=device,
        ).values
        if hidden_states is None or weight is None:
            raise ValueError("LM-head projection tensors must not be skipped")

        hidden_states = (
            hidden_states.float()
            .mul(self.config.hidden_scale)
            .to(self.config.hidden_dtype)
            .contiguous()
        )
        weight = (
            weight.float()
            .mul(self._weight_scale())
            .to(self.config.weight_dtype)
            .contiguous()
        )
        values = LMHeadProjectionInputValues(
            hidden_states=hidden_states,
            weight=weight,
        )
        _check_lm_head_projection_values(values)
        return values


def _check_lm_head_projection_values(values: LMHeadProjectionInputValues) -> None:
    if values.hidden_states.ndim != 2:
        raise ValueError("hidden_states must be rank-2 [num_tokens, hidden_dim]")
    if values.weight.ndim != 2:
        raise ValueError("weight must be rank-2 [vocab_size, hidden_dim]")
    if values.hidden_states.shape[0] <= 0:
        raise ValueError("hidden_states must contain at least one token row")
    if values.hidden_states.shape[1] <= 0:
        raise ValueError("hidden_dim must be positive")
    if values.weight.shape[0] <= 0:
        raise ValueError("weight must contain at least one vocabulary row")
    if values.weight.shape[1] != values.hidden_states.shape[1]:
        raise ValueError(
            "LM-head projection hidden dimensions must match; "
            f"hidden_states={tuple(values.hidden_states.shape)}, "
            f"weight={tuple(values.weight.shape)}"
        )
    _check_projection_dtype("hidden_states dtype", values.hidden_states.dtype)
    _check_projection_dtype("weight dtype", values.weight.dtype)
    if not torch.isfinite(values.hidden_states.float()).all():
        raise ValueError("hidden_states must be finite")
    if not torch.isfinite(values.weight.float()).all():
        raise ValueError("weight must be finite")


def lm_head_projection_reference(
    values: LMHeadProjectionInputValues,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return LM-head logits for ``hidden_states @ weight.T``."""

    _check_lm_head_projection_values(values)
    _check_projection_dtype("out_dtype", out_dtype)
    return (values.hidden_states.float() @ values.weight.float().T).to(out_dtype)


def mxfp4_gemm_input_config(
    *,
    M: int,
    N: int,
    K: int,
    c_dtype: torch.dtype,
    scale_dtype: InputDType = None,
    a_dtype: InputDType = CustomDType.MXFP4,
    b_dtype: InputDType = CustomDType.MXFP4,
    a_layout: GemmLayout = "MK",
    b_layout: GemmLayout = "NK",
    batch_shape: tuple[int, ...] = (),
    block_size: int = _DEFAULT_MXFP4_BLOCK_SIZE,
) -> GemmInputConfig:
    """Build a GEMM config for MXFP4 values with UE8M0 scales."""

    return GemmInputConfig(
        M=M,
        N=N,
        K=K,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        a_scale_dtype=None if a_dtype is None else scale_dtype,
        b_scale_dtype=None if b_dtype is None else scale_dtype,
        c_dtype=c_dtype,
        a_layout=a_layout,
        b_layout=b_layout,
        batch_shape=batch_shape,
        a_scale_shape=(
            None
            if a_dtype is None
            else gemm_scale_shape(
                "block",
                "a",
                M=M,
                N=N,
                K=K,
                batch_shape=batch_shape,
                block_shape=(block_size,),
            )
        ),
        b_scale_shape=(
            None
            if b_dtype is None
            else gemm_scale_shape(
                "block",
                "b",
                M=M,
                N=N,
                K=K,
                batch_shape=batch_shape,
                block_shape=(block_size,),
            )
        ),
    )


def mxfp8_gemm_input_config(
    *,
    M: int,
    N: int,
    K: int,
    c_dtype: torch.dtype,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_dtype: torch.dtype = torch.float32,
    block_shape: tuple[int, int] = (128, 128),
    a_layout: GemmLayout = "MK",
    b_layout: GemmLayout = "NK",
    batch_shape: tuple[int, ...] = (),
    a_device: DeviceLike = None,
    b_device: DeviceLike = None,
    a_scale_device: DeviceLike = None,
    b_scale_device: DeviceLike = None,
    c_device: DeviceLike = None,
) -> GemmInputConfig:
    """Build a standard block-scaled FP8 GEMM input config.

    The represented operation is still the logical row-major ``A @ B.T`` GEMM.
    ``A_scales`` contains one scale per logical ``M`` row and ``K`` block.
    ``B_scales`` contains one scale per logical ``N`` block and ``K`` block.
    The helper enforces regular block grids so references can expand scales
    without backend-specific padding rules.
    """

    M = int(M)
    N = int(N)
    K = int(K)
    block_shape = tuple(int(dim) for dim in block_shape)
    if len(block_shape) != 2 or any(dim <= 0 for dim in block_shape):
        raise ValueError("mxfp8 block_shape must contain two positive dimensions")
    block_n, block_k = block_shape
    if K % block_k != 0:
        raise ValueError(
            "mxfp8 GEMM requires K to be divisible by block_shape[1]; "
            f"got K={K}, block_k={block_k}"
        )
    if N % block_n != 0:
        raise ValueError(
            "mxfp8 GEMM requires N to be divisible by block_shape[0]; "
            f"got N={N}, block_n={block_n}"
        )
    if a_layout != "MK":
        raise ValueError("mxfp8 A operands require a_layout='MK'")
    if b_layout != "NK":
        raise ValueError("mxfp8 B operands require b_layout='NK'")

    return GemmInputConfig(
        M=M,
        N=N,
        K=K,
        a_dtype=fp8_dtype,
        b_dtype=fp8_dtype,
        c_dtype=c_dtype,
        a_layout=a_layout,
        b_layout=b_layout,
        batch_shape=batch_shape,
        a_scale_shape=gemm_scale_shape(
            "block",
            "a",
            M=M,
            N=N,
            K=K,
            batch_shape=batch_shape,
            block_shape=block_shape,
        ),
        b_scale_shape=gemm_scale_shape(
            "block",
            "b",
            M=M,
            N=N,
            K=K,
            batch_shape=batch_shape,
            block_shape=block_shape,
        ),
        a_scale_dtype=scale_dtype,
        b_scale_dtype=scale_dtype,
        a_device=a_device,
        b_device=b_device,
        a_scale_device=a_scale_device,
        b_scale_device=b_scale_device,
        c_device=c_device,
    )


def mxint4_gemm_input_config(
    *,
    M: int,
    N: int,
    K: int,
    c_dtype: torch.dtype,
    scale_dtype: InputDType = None,
    a_dtype: InputDType = CustomDType.MXINT4,
    b_dtype: InputDType = CustomDType.MXINT4,
    a_layout: GemmLayout = "MK",
    b_layout: GemmLayout = "NK",
    batch_shape: tuple[int, ...] = (),
    block_size: int = _DEFAULT_MXINT4_BLOCK_SIZE,
) -> GemmInputConfig:
    """Build a GEMM config for signed INT4 values with BF16 group scales."""

    return GemmInputConfig(
        M=M,
        N=N,
        K=K,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        a_scale_dtype=None if a_dtype is None else scale_dtype,
        b_scale_dtype=None if b_dtype is None else scale_dtype,
        c_dtype=c_dtype,
        a_layout=a_layout,
        b_layout=b_layout,
        batch_shape=batch_shape,
        a_scale_shape=(
            None
            if a_dtype is None
            else gemm_scale_shape(
                "block",
                "a",
                M=M,
                N=N,
                K=K,
                batch_shape=batch_shape,
                block_shape=(block_size,),
            )
        ),
        b_scale_shape=(
            None
            if b_dtype is None
            else gemm_scale_shape(
                "block",
                "b",
                M=M,
                N=N,
                K=K,
                batch_shape=batch_shape,
                block_shape=(block_size,),
            )
        ),
    )


@dataclass
class NVFP4GemmSwiGLUNVFP4QuantInputValues:
    """Generated values for fused NVFP4 GEMM + SwiGLU + NVFP4 quantization."""

    x: torch.Tensor
    w1: torch.Tensor
    x_fp4: torch.Tensor
    x_scale: torch.Tensor
    w1_fp4: torch.Tensor
    w1_scale: torch.Tensor
    x_global_scale: torch.Tensor
    w1_global_scale: torch.Tensor
    fc1_alpha: torch.Tensor
    output_global_scale: torch.Tensor
    output_global_scale_inv: torch.Tensor
    scale_size: int


@dataclass
class NVFP4GemmSwiGLUNVFP4QuantInputConfig:
    """Initialization parameters for fused NVFP4 GEMM/SwiGLU/quant inputs.

    The represented operation dequantizes packed NVFP4 activations ``x_fp4``
    and gate/up weights ``w1_fp4``, computes ``x @ w1.T``, applies
    ``silu(gate) * up`` after splitting the GEMM result in half, then quantizes
    the activated output back to NVFP4.
    """

    # Required: logical number of input token rows.
    M: int

    # Required: logical input/reduction width.
    K: int

    # Required: hidden width after the gate/up split. The FC1 output width is
    # 2 * intermediate_size.
    intermediate_size: int

    # Required: generated floating source dtype before NVFP4 quantization.
    dtype: torch.dtype

    # Optional: number of values covered by each NVFP4 FP8 scale. TokenSpeed
    # NVFP4 kernels currently use 16.
    scale_size: int = _DEFAULT_NVFP4_BLOCK_SIZE

    # Optional: tensor-wide scale used when quantizing the SwiGLU output back
    # to NVFP4. The default keeps typical generated MLP activations in range
    # without requiring generation to run the full GEMM.
    output_global_scale: float = 0.01

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class NVFP4GemmSwiGLUNVFP4QuantInputs(NumericsInputGenerator):
    """Generator for fused NVFP4 GEMM + SwiGLU + NVFP4 quantization inputs."""

    config: NVFP4GemmSwiGLUNVFP4QuantInputConfig
    x_input: TensorInput | None
    w1_input: TensorInput | None

    def __init__(self, config: NVFP4GemmSwiGLUNVFP4QuantInputConfig) -> None:
        self.config = config
        self.x_input = None
        self.w1_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.M = int(self.config.M)
        self.config.K = int(self.config.K)
        self.config.intermediate_size = int(self.config.intermediate_size)
        self.config.scale_size = int(self.config.scale_size)
        self.config.output_global_scale = float(self.config.output_global_scale)
        if self.config.M <= 0:
            raise ValueError("M must be positive")
        if self.config.K <= 0:
            raise ValueError("K must be positive")
        if self.config.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.config.scale_size != _DEFAULT_NVFP4_BLOCK_SIZE:
            raise ValueError(
                f"NVFP4 fused GEMM currently requires scale_size={_DEFAULT_NVFP4_BLOCK_SIZE}"
            )
        if self.config.K % self.config.scale_size != 0:
            raise ValueError("K must be divisible by scale_size")
        if self.config.intermediate_size % self.config.scale_size != 0:
            raise ValueError("intermediate_size must be divisible by scale_size")
        if self.config.output_global_scale <= 0.0:
            raise ValueError("output_global_scale must be positive")
        self.config.dtype = _check_nvfp4_source_dtype(self.config.dtype)
        self.x_input = self.x_input or TensorInput(
            (self.config.M, self.config.K),
            self.config.dtype,
            device=self.config.device,
        )
        self.w1_input = self.w1_input or TensorInput(
            (2 * self.config.intermediate_size, self.config.K),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> NVFP4GemmSwiGLUNVFP4QuantInputValues:
        self.__post_init__()
        if self.x_input is None or self.w1_input is None:
            raise ValueError(
                "NVFP4GemmSwiGLUNVFP4QuantInputs child generators must be initialized"
            )
        self.x_input.shape = (self.config.M, self.config.K)
        self.x_input.dtype = self.config.dtype
        self.w1_input.shape = (2 * self.config.intermediate_size, self.config.K)
        self.w1_input.dtype = self.config.dtype
        x = self.x_input.generate(seed=_child_seed(seed, 1), device=device).values
        w1 = self.w1_input.generate(seed=_child_seed(seed, 2), device=device).values
        if x is None or w1 is None:
            raise ValueError("x and w1 generation must not be skipped")
        x = x.contiguous()
        w1 = (w1.float() / math.sqrt(self.config.K)).to(self.config.dtype).contiguous()
        x_global_scale = _nvfp4_global_scale_for(x)
        w1_global_scale = _nvfp4_global_scale_for(w1)
        x_fp4, x_scale = nvfp4_quantization_reference(
            x,
            scale=x_global_scale,
            scale_size=self.config.scale_size,
        )
        w1_fp4, w1_scale = nvfp4_quantization_reference(
            w1,
            scale=w1_global_scale,
            scale_size=self.config.scale_size,
        )
        output_global_scale = torch.tensor(
            [self.config.output_global_scale],
            dtype=torch.float32,
            device=x.device,
        )
        return NVFP4GemmSwiGLUNVFP4QuantInputValues(
            x=x,
            w1=w1,
            x_fp4=x_fp4,
            x_scale=x_scale,
            w1_fp4=w1_fp4,
            w1_scale=w1_scale,
            x_global_scale=x_global_scale,
            w1_global_scale=w1_global_scale,
            fc1_alpha=x_global_scale * w1_global_scale,
            output_global_scale=output_global_scale,
            output_global_scale_inv=1.0 / output_global_scale,
            scale_size=self.config.scale_size,
        )


def nvfp4_gemm_swiglu_nvfp4_quant_reference(
    values: NVFP4GemmSwiGLUNVFP4QuantInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed NVFP4 output for fused NVFP4 GEMM + SwiGLU."""

    if values.x_fp4.ndim != 2 or values.w1_fp4.ndim != 2:
        raise ValueError("x_fp4 and w1_fp4 must be rank-2 packed NVFP4 tensors")
    if values.w1_fp4.shape[0] % 2 != 0:
        raise ValueError("w1_fp4 must contain an even gate/up row count")
    if values.x_fp4.shape[1] != values.w1_fp4.shape[1]:
        raise ValueError("x_fp4 and w1_fp4 packed K dimensions must match")
    if values.output_global_scale.numel() != 1:
        raise ValueError("output_global_scale must be scalar")
    x = nvfp4_dequantization_reference(
        values.x_fp4,
        values.x_scale,
        scale=values.x_global_scale,
        scale_size=values.scale_size,
    )
    w1 = nvfp4_dequantization_reference(
        values.w1_fp4,
        values.w1_scale,
        scale=values.w1_global_scale,
        scale_size=values.scale_size,
    )
    activated = _silu_and_mul(x @ w1.T)
    return nvfp4_quantization_reference(
        activated,
        scale=values.output_global_scale,
        scale_size=values.scale_size,
    )
