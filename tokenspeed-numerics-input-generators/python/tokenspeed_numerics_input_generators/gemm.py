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
    _generate_scale_tensor,
    _normalize_shape,
    _packed_mxfp4_shape,
    _packed_mxint4_shape,
    _packed_nvfp4_shape,
    _resolve_device,
    _rng_for_device,
)
from tokenspeed_numerics_input_generators.quantization import (
    nvfp4_dequantization_reference,
)

__all__ = [
    "GemmInputs",
    "GemmInputConfig",
    "GemmInputValues",
    "gemm_reference",
    "gemm_scale_shape",
]

GemmLayout = Literal["MK", "KM", "NK", "KN"]
ScaleGranularity = Literal["tensor", "channel", "block"]

_DEFAULT_MXFP4_BLOCK_SIZE = 32
_DEFAULT_MXINT4_BLOCK_SIZE = 32
_DEFAULT_NVFP4_BLOCK_SIZE = 16
_GEMM_LAYOUTS = frozenset({"MK", "KM", "NK", "KN"})
_NVFP4_SCALE_DTYPES = frozenset({torch.float32, torch.float8_e4m3fn, torch.uint8})


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
    if dtype in (CustomDType.NVFP4, torch.float4_e2m1fn_x2):
        shape = _packed_nvfp4_shape(shape, packed_dim=packed_dim)
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


def _scale_shape_prefix_compatible(
    shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
) -> bool:
    if len(scale_shape) > len(shape):
        return False
    for scale_dim, value_dim in zip(scale_shape, shape, strict=False):
        if scale_dim == value_dim or scale_dim == 1:
            continue
        if scale_dim <= 0 or scale_dim > value_dim or value_dim % scale_dim != 0:
            return False
    return True


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


def _linear_block_scale_shape_matches(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int,
) -> bool:
    if packed.ndim < 2 or scales.ndim < 1:
        return False
    logical_k = packed.shape[-1] * 2
    expected_groups = math.ceil(logical_k / block_size)
    return (
        scales.shape[:-1] == packed.shape[:-1] and scales.shape[-1] == expected_groups
    )


def _infer_packed_gemm_format(
    values: torch.Tensor,
    scales: torch.Tensor | None,
) -> Literal["mxfp4", "nvfp4", "mxint4"] | None:
    if scales is None:
        return None
    if values.dtype == torch.float4_e2m1fn_x2:
        if scales.dtype not in _NVFP4_SCALE_DTYPES:
            raise ValueError(
                "torch.float4_e2m1fn_x2 GEMM operands require NVFP4 scale "
                f"storage, got {scales.dtype}"
            )
        return "nvfp4"
    if values.dtype != torch.uint8:
        return None
    if scales.dtype == torch.bfloat16:
        return "mxint4"
    if scales.dtype in (torch.float32, torch.float8_e4m3fn):
        return "nvfp4"
    if scales.dtype != torch.uint8:
        return None

    matches_mxfp4 = _linear_block_scale_shape_matches(
        values,
        scales,
        block_size=_DEFAULT_MXFP4_BLOCK_SIZE,
    )
    matches_nvfp4 = _linear_block_scale_shape_matches(
        values,
        scales,
        block_size=_DEFAULT_NVFP4_BLOCK_SIZE,
    )
    if matches_nvfp4 and not matches_mxfp4:
        return "nvfp4"
    if matches_mxfp4 and not matches_nvfp4:
        return "mxfp4"
    if matches_mxfp4 and matches_nvfp4:
        raise ValueError(
            "uint8 FP4 GEMM scales are ambiguous between MXFP4 block size "
            f"{_DEFAULT_MXFP4_BLOCK_SIZE} and NVFP4 block size "
            f"{_DEFAULT_NVFP4_BLOCK_SIZE}; use a larger K dimension or a "
            "non-uint8 NVFP4 scale dtype"
        )
    raise ValueError(
        "uint8 FP4 GEMM scale shape does not match MXFP4 or NVFP4 block "
        f"groups; values={tuple(values.shape)}, scales={tuple(scales.shape)}"
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


def _logical_operand(
    values: torch.Tensor,
    scales: torch.Tensor | None,
    *,
    layout: GemmLayout,
    is_mxfp4: bool,
    is_nvfp4: bool = False,
    is_mxint4: bool = False,
) -> torch.Tensor:
    if sum((is_mxfp4, is_nvfp4, is_mxint4)) > 1:
        raise ValueError("GEMM operand cannot use multiple packed formats")
    if is_mxfp4:
        if layout not in ("MK", "NK"):
            raise ValueError(
                "MXFP4 GEMM reference currently supports row-major MK/NK operands"
            )
        if scales is None:
            raise ValueError("MXFP4 GEMM reference requires scales")
        logical = _dequantize_mxfp4_linear(values, scales)
    elif is_nvfp4:
        if layout not in ("MK", "NK"):
            raise ValueError(
                "NVFP4 GEMM reference currently supports row-major MK/NK operands"
            )
        if scales is None:
            raise ValueError("NVFP4 GEMM reference requires scales")
        logical = nvfp4_dequantization_reference(values, scales, scale=1.0)
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
            scale_values = scales.float()
            if (
                layout in ("KM", "KN")
                and scale_values.ndim == 1
                and scale_values.shape[0] == logical.shape[-1]
            ):
                logical = logical * scale_values.reshape(
                    (1,) * (logical.ndim - 1) + (logical.shape[-1],)
                )
            else:
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
        a_is_nvfp4 = self.config.a_dtype in (
            CustomDType.NVFP4,
            torch.float4_e2m1fn_x2,
        )
        b_is_nvfp4 = self.config.b_dtype in (
            CustomDType.NVFP4,
            torch.float4_e2m1fn_x2,
        )
        if a_is_nvfp4 and self.config.a_layout != "MK":
            raise ValueError("NVFP4 A operands currently require a_layout='MK'")
        if b_is_nvfp4 and self.config.b_layout != "NK":
            raise ValueError("NVFP4 B operands currently require b_layout='NK'")
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
        value_shape = self._value_shape(role)
        dtype = self.config.a_dtype if is_a else self.config.b_dtype
        scale_shape = self.config.a_scale_shape if is_a else self.config.b_scale_shape
        scale_dtype = self.config.a_scale_dtype if is_a else self.config.b_scale_dtype
        value_device = self.config.a_device if is_a else self.config.b_device
        scale_device = (
            self.config.a_scale_device if is_a else self.config.b_scale_device
        )
        if (
            dtype is not None
            and not isinstance(dtype, CustomDType)
            and scale_shape is not None
            and not _scale_shape_prefix_compatible(value_shape, scale_shape)
        ):
            if scale_dtype is None:
                raise ValueError(
                    "scale_shape and scale_dtype must be provided together"
                )
            values = TensorInput(
                value_shape,
                dtype,
                device=value_device,
            ).generate(seed=seed, device=device)
            scale_target_device = _resolve_device(scale_device, device)
            scale_generator = _rng_for_device(
                scale_target_device,
                _child_seed(seed, 2),
            )
            return TensorValues(
                values=values.values,
                scales=_generate_scale_tensor(
                    scale_shape,
                    scale_dtype,
                    device=scale_target_device,
                    generator=scale_generator,
                    max_value=1.0,
                ),
            )
        return TensorInput(
            value_shape,
            dtype,
            scale_shape=scale_shape,
            scale_dtype=scale_dtype,
            device=value_device,
            scale_device=scale_device,
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
    E2M1x2 bytes and dequantized with UE8M0 scales. NVFP4 operands are unpacked
    from E2M1x2 bytes and dequantized with FP8 E4M3 local scales. MXINT4
    operands are unpacked from signed INT4 nibbles and dequantized with BF16
    group scales. The returned dtype defaults to the generated ``C`` dtype,
    matching TokenSpeed's ``out_dtype`` role.
    """

    a_layout = _check_gemm_layout("a_layout", a_layout)
    b_layout = _check_gemm_layout("b_layout", b_layout)
    if values.A is None or values.B is None:
        raise ValueError("gemm_reference requires generated A and B operands")
    out_dtype = values.C.dtype if out_dtype is None else out_dtype
    if not isinstance(out_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")

    a_format = _infer_packed_gemm_format(values.A, values.A_scales)
    b_format = _infer_packed_gemm_format(values.B, values.B_scales)
    A = _logical_operand(
        values.A,
        values.A_scales,
        layout=a_layout,
        is_mxfp4=a_format == "mxfp4",
        is_nvfp4=a_format == "nvfp4",
        is_mxint4=a_format == "mxint4",
    )
    B = _logical_operand(
        values.B,
        values.B_scales,
        layout=b_layout,
        is_mxfp4=b_format == "mxfp4",
        is_nvfp4=b_format == "nvfp4",
        is_mxint4=b_format == "mxint4",
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
