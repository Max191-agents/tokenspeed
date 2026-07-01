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

"""Quantization-family input generators for numerical correctness tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _is_floating_storage_dtype,
    _normalize_shape,
    _resolve_device,
    _rng_for_device,
)

__all__ = [
    "FP8QuantizationInputConfig",
    "FP8QuantizationInputs",
    "FP8QuantizationInputValues",
    "MXFP4QuantizationInputConfig",
    "MXFP4QuantizationInputs",
    "MXFP4QuantizationInputValues",
    "fp8_quantization_reference",
    "fp8_scale_shape",
    "mxfp4_quantization_reference",
    "mxfp4_scale_shape",
]

FP8ScaleGranularity = Literal["none", "tensor", "token", "token_group"]
FP8ScaleEncoding = Literal["float32", "ue8m0", "packed_ue8m0"]
MXFP4ScaleLayout = Literal["linear"]

_E2M1_NIBBLES = torch.tensor(
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
    dtype=torch.uint8,
)
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    shape = _normalize_shape(shape)
    if not shape:
        raise ValueError("quantization input shape must have at least one dimension")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"quantization input dimensions must be positive, got {shape}")
    return shape


def _check_input_dtype(dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    if not _is_floating_storage_dtype(dtype):
        raise ValueError(f"dtype must be a floating torch dtype, got {dtype}")
    if dtype in {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    }:
        raise ValueError("FP8 quantization inputs should be generated before FP8 cast")
    return dtype


def _check_mxfp4_input_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"MXFP4 input dtype must be bf16 or fp16, got {dtype}")
    return dtype


def _check_fp8_dtype(dtype: torch.dtype) -> torch.dtype:
    fp8_dtypes = {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    }
    if dtype not in fp8_dtypes:
        raise ValueError(f"output_dtype must be an FP8 torch dtype, got {dtype}")
    return dtype


def mxfp4_scale_shape(
    shape: tuple[int, ...] | list[int],
    *,
    scale_size: int = 32,
) -> tuple[int, ...]:
    """Return the semantic UE8M0 scale shape for MXFP4 quantization."""

    shape = _check_shape(shape)
    scale_size = _check_positive("scale_size", scale_size)
    if scale_size != 32:
        raise ValueError(f"MXFP4 currently requires scale_size=32, got {scale_size}")
    if shape[-1] % scale_size != 0:
        raise ValueError(
            f"last dimension must be divisible by scale_size, got K={shape[-1]}, "
            f"scale_size={scale_size}"
        )
    return shape[:-1] + (shape[-1] // scale_size,)


def _e2m1_values_from_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    magnitude_bits = nibbles & 0x7
    exponent = (magnitude_bits >> 1).to(torch.float32)
    mantissa = (magnitude_bits & 0x1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent - 1.0)
    subnormal = 0.5 * mantissa
    magnitude = torch.where(exponent == 0, subnormal, normal)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
    return magnitude * sign


def _pack_e2m1_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def _nearest_e2m1_nibbles(values: torch.Tensor) -> torch.Tensor:
    table_values = _E2M1_VALUES.to(device=values.device)
    table_nibbles = _E2M1_NIBBLES.to(device=values.device)
    distances = (values.unsqueeze(-1) - table_values).abs()
    return table_nibbles[distances.argmin(dim=-1)]


def fp8_scale_shape(
    shape: tuple[int, ...] | list[int],
    *,
    granularity: FP8ScaleGranularity,
    group_size: int | None = None,
) -> tuple[int, ...] | None:
    """Return the semantic scale shape for FP8 quantization."""

    shape = _check_shape(shape)
    if granularity == "none":
        return None
    if granularity == "tensor":
        return (1,)
    row_shape = shape[:-1]
    if granularity == "token":
        return row_shape + (1,)
    if granularity != "token_group":
        raise ValueError(f"unsupported FP8 scale granularity={granularity!r}")
    if group_size is None:
        raise ValueError("token_group granularity requires group_size")
    group_size = _check_positive("group_size", group_size)
    K = shape[-1]
    if K % group_size != 0:
        raise ValueError(
            f"last dimension must be divisible by group_size, got K={K}, "
            f"group_size={group_size}"
        )
    return row_shape + (K // group_size,)


@dataclass
class FP8QuantizationInputValues:
    """Generated values for ``FP8QuantizationInputs``."""

    x: torch.Tensor
    scale: torch.Tensor | None


@dataclass
class FP8QuantizationInputConfig:
    """Initialization parameters for ``FP8QuantizationInputs``.

    The represented operation casts a floating point tensor to FP8, optionally
    using scales computed at tensor, token, or token-group granularity.
    """

    # Required: full input tensor shape. The last dimension is the quantized
    # channel/group axis.
    shape: tuple[int, ...]

    # Required: generated input dtype, typically bf16 or fp16.
    dtype: torch.dtype

    # Optional: target FP8 dtype for reference computations.
    output_dtype: torch.dtype = torch.float8_e4m3fn

    # Optional: semantic scale granularity. "none" represents a pure FP8 cast.
    granularity: FP8ScaleGranularity = "none"

    # Optional: required for token_group granularity.
    group_size: int | None = None

    # Optional: scale encoding requested by a consumer. The generator returns
    # float32 scales for the semantic reference; encoded layouts are adapter
    # concerns unless a family adds an encoded-value generator.
    scale_encoding: FP8ScaleEncoding = "float32"

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class FP8QuantizationInputs(NumericsInputGenerator):
    """Generator for FP8 quantization input tensors and semantic scales."""

    config: FP8QuantizationInputConfig
    x_input: TensorInput | None

    def __init__(self, config: FP8QuantizationInputConfig) -> None:
        self.config = config
        self.x_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.shape = _check_shape(self.config.shape)
        self.config.dtype = _check_input_dtype(self.config.dtype)
        self.config.output_dtype = _check_fp8_dtype(self.config.output_dtype)
        if self.config.granularity not in ("none", "tensor", "token", "token_group"):
            raise ValueError(
                f"unsupported FP8 scale granularity={self.config.granularity!r}"
            )
        if self.config.scale_encoding not in ("float32", "ue8m0", "packed_ue8m0"):
            raise ValueError(
                f"unsupported scale_encoding={self.config.scale_encoding!r}"
            )
        if self.config.scale_encoding != "float32" and (
            self.config.granularity != "token_group"
        ):
            raise ValueError(
                "non-float FP8 scale encodings are only meaningful for token_group"
            )
        fp8_scale_shape(
            self.config.shape,
            granularity=self.config.granularity,
            group_size=self.config.group_size,
        )
        self.x_input = self.x_input or TensorInput(
            self.config.shape,
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> FP8QuantizationInputValues:
        if self.x_input is None:
            raise ValueError(
                "FP8QuantizationInputs child generator must be initialized"
            )
        x = self.x_input.generate(seed=_child_seed(seed, 1), device=device).values
        if x is None:
            raise ValueError("x generation unexpectedly returned None")
        scale = _fp8_scale(
            x.float(),
            granularity=self.config.granularity,
            group_size=self.config.group_size,
            output_dtype=self.config.output_dtype,
        )
        return FP8QuantizationInputValues(x=x, scale=scale)


@dataclass
class MXFP4QuantizationInputValues:
    """Generated values for ``MXFP4QuantizationInputs``."""

    x: torch.Tensor
    scale_size: int
    scale_layout: MXFP4ScaleLayout
    global_scale: None = None


@dataclass
class MXFP4QuantizationInputConfig:
    """Initialization parameters for ``MXFP4QuantizationInputs``.

    The represented operation quantizes a bf16/fp16 tensor into packed E2M1
    values with one UE8M0 scale byte for each contiguous group of 32 values.
    Generated values are chosen from exactly representable MXFP4 values so the
    quantized output is deterministic and not dominated by rounding choices.
    """

    # Required: full input tensor shape. The last dimension is grouped into
    # contiguous MXFP4 scale groups.
    shape: tuple[int, ...]

    # Required: generated input dtype. MXFP4 kernels currently consume bf16 or
    # fp16 source tensors.
    dtype: torch.dtype

    # Optional: number of values per UE8M0 group. Currently fixed to 32.
    scale_size: int = 32

    # Optional: scale layout for generated/reference values. Currently linear.
    scale_layout: MXFP4ScaleLayout = "linear"

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MXFP4QuantizationInputs(NumericsInputGenerator):
    """Generator for MXFP4 quantization input tensors."""

    config: MXFP4QuantizationInputConfig

    def __init__(self, config: MXFP4QuantizationInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.shape = _check_shape(self.config.shape)
        self.config.dtype = _check_mxfp4_input_dtype(self.config.dtype)
        mxfp4_scale_shape(self.config.shape, scale_size=self.config.scale_size)
        if self.config.scale_layout != "linear":
            raise ValueError(
                f"MXFP4 generator currently supports scale_layout='linear', "
                f"got {self.config.scale_layout!r}"
            )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MXFP4QuantizationInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        generator = _rng_for_device(target_device, seed)
        scale_shape = mxfp4_scale_shape(
            self.config.shape,
            scale_size=self.config.scale_size,
        )
        num_groups = scale_shape[-1]
        group_shape = self.config.shape[:-1] + (num_groups, self.config.scale_size)
        nibble_indices = torch.randint(
            0,
            len(_E2M1_NIBBLES),
            group_shape,
            dtype=torch.int64,
            device=target_device,
            generator=generator,
        )
        nibble_table = _E2M1_NIBBLES.to(device=target_device)
        nibbles = nibble_table[nibble_indices]
        nibbles[..., 0] = 0x7
        scale_bytes = torch.randint(
            123,
            131,
            scale_shape,
            dtype=torch.int32,
            device=target_device,
            generator=generator,
        )
        e2m1_values = _e2m1_values_from_nibbles(nibbles)
        scale_values = torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=target_device),
            scale_bytes.to(torch.float32) - 127.0,
        )
        x = (e2m1_values * scale_values.unsqueeze(-1)).reshape(self.config.shape)
        return MXFP4QuantizationInputValues(
            x=x.to(self.config.dtype).contiguous(),
            scale_size=self.config.scale_size,
            scale_layout=self.config.scale_layout,
        )


def _fp8_scale(
    x_fp32: torch.Tensor,
    *,
    granularity: FP8ScaleGranularity,
    group_size: int | None,
    output_dtype: torch.dtype,
) -> torch.Tensor | None:
    if granularity == "none":
        return None
    fp8_max = torch.finfo(output_dtype).max
    if granularity == "tensor":
        max_abs = x_fp32.abs().amax().reshape(1)
    elif granularity == "token":
        max_abs = x_fp32.abs().amax(dim=-1, keepdim=True)
    elif granularity == "token_group":
        if group_size is None:
            raise ValueError("token_group granularity requires group_size")
        grouped = x_fp32.reshape(
            *x_fp32.shape[:-1], x_fp32.shape[-1] // group_size, group_size
        )
        max_abs = grouped.abs().amax(dim=-1)
    else:
        raise ValueError(f"unsupported FP8 scale granularity={granularity!r}")
    return (max_abs / fp8_max).clamp(min=1e-10).to(torch.float32)


def fp8_quantization_reference(
    x: torch.Tensor,
    *,
    granularity: FP8ScaleGranularity,
    group_size: int | None = None,
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP8 quantized values cast back to float32 for comparison."""

    x_fp32 = x.float()
    if scale is None:
        scale = _fp8_scale(
            x_fp32,
            granularity=granularity,
            group_size=group_size,
            output_dtype=output_dtype,
        )
    if scale is None:
        return x_fp32.to(output_dtype).float()
    if granularity == "token_group":
        if group_size is None:
            raise ValueError("token_group granularity requires group_size")
        scaled = x_fp32.reshape(
            *x_fp32.shape[:-1],
            x_fp32.shape[-1] // group_size,
            group_size,
        )
        quantized = (scaled / scale.unsqueeze(-1)).clamp(
            min=torch.finfo(output_dtype).min,
            max=torch.finfo(output_dtype).max,
        )
        return quantized.to(output_dtype).float().reshape_as(x_fp32)
    quantized = (x_fp32 / scale).clamp(
        min=torch.finfo(output_dtype).min,
        max=torch.finfo(output_dtype).max,
    )
    return quantized.to(output_dtype).float()


def mxfp4_quantization_reference(
    x: torch.Tensor,
    *,
    scale_size: int = 32,
    scale_layout: MXFP4ScaleLayout = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed MXFP4 bytes and UE8M0 scale bytes for ``x``.

    The reference models the operation-level MXFP4 representation. It chooses
    one power-of-two scale for every contiguous group of ``scale_size`` values,
    quantizes each scaled value to the nearest E2M1 value, then packs pairs of
    nibbles into uint8 storage.
    """

    if scale_layout != "linear":
        raise ValueError(
            f"MXFP4 reference currently supports scale_layout='linear', got {scale_layout!r}"
        )
    shape = _check_shape(tuple(x.shape))
    scale_shape = mxfp4_scale_shape(shape, scale_size=scale_size)
    x_groups = x.to(torch.float32).reshape(*scale_shape, scale_size)
    max_abs = x_groups.abs().amax(dim=-1)
    zero_groups = max_abs == 0
    scale_exp = (
        torch.floor(torch.log2(max_abs.clamp(min=torch.finfo(torch.float32).tiny)))
        - 2.0
    )
    scale_exp = torch.where(zero_groups, torch.full_like(scale_exp, -127.0), scale_exp)
    scale_exp = scale_exp.clamp(min=-127.0, max=127.0)
    scale_values = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=x.device),
        scale_exp,
    )
    scaled = x_groups / scale_values.unsqueeze(-1)
    nibbles = _nearest_e2m1_nibbles(scaled)
    packed = _pack_e2m1_nibbles(nibbles).reshape(*shape[:-1], shape[-1] // 2)
    scales = (scale_exp.to(torch.int32) + 127).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()
