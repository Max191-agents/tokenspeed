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

"""Core tensor input generators for numerical correctness tests."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

__all__ = [
    "CustomDType",
    "DeviceLike",
    "InputDType",
    "NumericsInputGenerator",
    "TensorInput",
    "TensorValues",
]

DeviceLike = str | torch.device | None


class CustomDType(str, Enum):
    """Custom numerical dtype handled by core input generators."""

    # Packed MXFP4 values. Storage is torch.uint8 with two E2M1 nibbles per byte;
    # TensorInput requires a scale_shape and generates paired UE8M0 scales.
    MXFP4 = "mxfp4"

    # Raw UE8M0 scale bytes. Storage is torch.uint8 exponent values interpreted
    # by MXFP4 kernels as unsigned exponent-only FP8 scales.
    UE8M0 = "ue8m0"

    # Weight-only signed INT4 values with group scales. Storage is torch.uint8
    # with two two's-complement INT4 nibbles per byte; TensorInput requires a
    # scale_shape and generates paired BF16 group scales.
    MXINT4 = "mxint4"


InputDType = torch.dtype | CustomDType | None

_DEFAULT_DEVICE = torch.device("cpu")
_MXFP4_VALUES_PER_BYTE = 2
_MXFP4_SCALE_MAX = 0.125
_MXINT4_VALUES_PER_BYTE = 2
_MXINT4_SCALE_MAX = 0.25
_UE8M0_SCALE_EXPONENTS = (121, 122, 123, 124)


class NumericsInputGenerator(ABC):
    """Base class for typed numerical input generators."""

    @abstractmethod
    def generate(self, *, seed: int, device: DeviceLike = None) -> Any:
        """Generate values and return a values object."""


def _normalize_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    normalized = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in normalized):
        raise ValueError(f"shape dimensions must be non-negative, got {shape}")
    return normalized


def _resolve_device(configured: DeviceLike, default: DeviceLike) -> torch.device:
    if configured is not None:
        return torch.device(configured)
    if default is not None:
        return torch.device(default)
    return _DEFAULT_DEVICE


def _rng_for_device(device: torch.device, seed: int) -> torch.Generator:
    rng_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=rng_device).manual_seed(seed)


def _child_seed(seed: int, offset: int) -> int:
    return seed + offset * 1_000_003


def _is_floating_storage_dtype(dtype: torch.dtype) -> bool:
    dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    }
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        dtypes.add(e8m0_dtype)
    return dtype in dtypes


def _packed_mxfp4_shape(
    shape: tuple[int, ...],
    *,
    packed_dim: int,
) -> tuple[int, ...]:
    if not shape:
        raise ValueError("mxfp4 tensors need at least one dimension to pack")
    packed_dim = packed_dim % len(shape)
    packed = list(shape)
    packed[packed_dim] = math.ceil(packed[packed_dim] / _MXFP4_VALUES_PER_BYTE)
    return tuple(packed)


def _packed_mxint4_shape(
    shape: tuple[int, ...],
    *,
    packed_dim: int,
) -> tuple[int, ...]:
    if not shape:
        raise ValueError("mxint4 tensors need at least one dimension to pack")
    packed_dim = packed_dim % len(shape)
    packed = list(shape)
    packed[packed_dim] = math.ceil(packed[packed_dim] / _MXINT4_VALUES_PER_BYTE)
    return tuple(packed)


def _generate_mxfp4_packed(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    low = torch.randint(
        0,
        16,
        shape,
        device=device,
        dtype=torch.uint8,
        generator=generator,
    )
    high = torch.randint(
        0,
        16,
        shape,
        device=device,
        dtype=torch.uint8,
        generator=generator,
    )
    return low | (high << 4)


def _generate_mxint4_packed(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    low = torch.randint(
        -7,
        8,
        shape,
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    high = torch.randint(
        -7,
        8,
        shape,
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    return ((low & 0xF) | ((high & 0xF) << 4)).to(torch.uint8)


def _generate_torch_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not _is_floating_storage_dtype(dtype):
        raise ValueError(
            "TensorInput only supports floating torch dtypes; use a dedicated "
            "metadata generator for index tensors or a CustomDType for "
            "custom quantized storage"
        )

    values = torch.randn(
        *shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    return values.to(dtype)


def _generate_scaled_torch_values(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.randn(
        *shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).clamp_(-1.0, 1.0)
    return values.to(dtype)


def _generate_scale_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype | CustomDType,
    *,
    device: torch.device,
    generator: torch.Generator,
    max_value: float,
) -> torch.Tensor:
    if dtype == CustomDType.UE8M0:
        exponents = torch.tensor(
            _UE8M0_SCALE_EXPONENTS,
            dtype=torch.uint8,
            device=device,
        )
        indices = torch.randint(
            0,
            len(_UE8M0_SCALE_EXPONENTS),
            shape,
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
        return exponents[indices]
    if isinstance(dtype, CustomDType):
        raise ValueError(f"unsupported scale custom dtype={dtype!r}")
    if not _is_floating_storage_dtype(dtype):
        raise ValueError("scale tensors must use a floating torch dtype")
    values = torch.rand(
        *shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    values = values * (max_value * 0.5) + (max_value * 0.5)
    return values.to(dtype)


def _check_scale_shape(
    shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
) -> None:
    if len(scale_shape) > len(shape):
        raise ValueError(
            "scale_shape must be prefix-compatible with shape; got "
            f"scale_shape={scale_shape}, shape={shape}"
        )
    for scale_dim, value_dim in zip(scale_shape, shape, strict=False):
        if scale_dim == value_dim or scale_dim == 1:
            continue
        if scale_dim <= 0 or scale_dim > value_dim or value_dim % scale_dim != 0:
            raise ValueError(
                "scale_shape must be prefix-compatible with shape; got "
                f"scale_shape={scale_shape}, shape={shape}"
            )


@dataclass
class TensorValues:
    """Generated values for a ``TensorInput``."""

    values: torch.Tensor | None
    scales: torch.Tensor | None


@dataclass(init=False)
class TensorInput(NumericsInputGenerator):
    """Generator for one tensor operand and optional scale sidecar."""

    shape: tuple[int, ...]
    dtype: InputDType
    scale_shape: tuple[int, ...] | None
    scale_dtype: InputDType
    device: DeviceLike
    scale_device: DeviceLike

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: InputDType = None,
        *,
        scale_shape: tuple[int, ...] | None = None,
        scale_dtype: InputDType = None,
        device: DeviceLike = None,
        scale_device: DeviceLike = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.scale_shape = scale_shape
        self.scale_dtype = scale_dtype
        self.device = device
        self.scale_device = scale_device
        self.__post_init__()

    def __post_init__(self) -> None:
        self.shape = _normalize_shape(self.shape)
        if self.dtype == CustomDType.MXFP4:
            if self.scale_shape is None:
                raise ValueError("mxfp4 tensors require scale_shape")
            if self.scale_dtype is None:
                self.scale_dtype = CustomDType.UE8M0
            elif self.scale_dtype != CustomDType.UE8M0:
                raise ValueError("mxfp4 scale_dtype must be CustomDType.UE8M0 or None")
        elif self.dtype == CustomDType.MXINT4:
            if self.scale_shape is None:
                raise ValueError("mxint4 tensors require scale_shape")
            if self.scale_dtype is None:
                self.scale_dtype = torch.bfloat16
            elif self.scale_dtype != torch.bfloat16:
                raise ValueError("mxint4 scale_dtype must be torch.bfloat16 or None")
        elif (self.scale_shape is None) != (self.scale_dtype is None):
            raise ValueError("scale_shape and scale_dtype must be provided together")
        if self.scale_shape is not None:
            self.scale_shape = _normalize_shape(self.scale_shape)
            _check_scale_shape(self.shape, self.scale_shape)

    def generate(self, *, seed: int, device: DeviceLike = None) -> TensorValues:
        self.__post_init__()
        values = self._generate_values(seed=_child_seed(seed, 1), device=device)
        scales = self._generate_scales(seed=_child_seed(seed, 2), device=device)
        return TensorValues(values=values, scales=scales)

    def _generate_values(
        self,
        *,
        seed: int,
        device: DeviceLike,
    ) -> torch.Tensor | None:
        if self.dtype is None:
            return None

        target_device = _resolve_device(self.device, device)
        generator = _rng_for_device(target_device, seed)
        if self.dtype == CustomDType.MXFP4:
            return _generate_mxfp4_packed(
                self.shape,
                device=target_device,
                generator=generator,
            )
        if self.dtype == CustomDType.MXINT4:
            return _generate_mxint4_packed(
                self.shape,
                device=target_device,
                generator=generator,
            )
        if isinstance(self.dtype, torch.dtype):
            if self.scale_shape is None:
                return _generate_torch_tensor(
                    self.shape,
                    self.dtype,
                    device=target_device,
                    generator=generator,
                )
            return _generate_scaled_torch_values(
                self.shape,
                self.dtype,
                device=target_device,
                generator=generator,
            )
        raise TypeError(f"unsupported tensor dtype={self.dtype!r}")

    def _generate_scales(
        self,
        *,
        seed: int,
        device: DeviceLike,
    ) -> torch.Tensor | None:
        if self.scale_shape is None:
            return None
        if self.scale_dtype is None:
            raise ValueError("scale_shape and scale_dtype must be provided together")

        scale_device = _resolve_device(self.scale_device, device)
        scale_generator = _rng_for_device(scale_device, seed)
        return _generate_scale_tensor(
            self.scale_shape,
            self.scale_dtype,
            device=scale_device,
            generator=scale_generator,
            max_value=self._scale_max_value(),
        )

    def _scale_max_value(self) -> float:
        if self.dtype == CustomDType.MXFP4:
            return _MXFP4_SCALE_MAX
        if self.dtype == CustomDType.MXINT4:
            return _MXINT4_SCALE_MAX
        return 1.0
