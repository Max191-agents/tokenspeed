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
    "ScaledTensorInputConfig",
    "ScaledTensorInput",
    "ScaledTensorValues",
    "TensorInputConfig",
    "TensorInput",
]

DeviceLike = str | torch.device | None


class CustomDType(str, Enum):
    """Custom numerical dtype handled by core input generators."""

    MXFP4 = "mxfp4"


InputDType = torch.dtype | CustomDType | None

_DEFAULT_DEVICE = torch.device("cpu")
_MXFP4_VALUES_PER_BYTE = 2
_MXFP4_SCALE_MAX = 0.125


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
    return dtype in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    }


def _storage_dtype(dtype: torch.dtype | CustomDType) -> torch.dtype:
    if dtype == CustomDType.MXFP4:
        return torch.uint8
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError(f"unsupported dtype={dtype!r}")


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
            "metadata generator for index tensors or ScaledTensorInput for "
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
    dtype: torch.dtype,
    *,
    device: torch.device,
    generator: torch.Generator,
    max_value: float,
) -> torch.Tensor:
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


@dataclass
class TensorInputConfig:
    """Initialization parameters for ``TensorInput``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: physical tensor shape to generate.
    shape: tuple[int, ...]

    # Required: builtin torch dtype. ``None`` skips generation and returns
    # ``None``.
    dtype: torch.dtype | None

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: device override for this tensor. If omitted, the
    # caller-supplied ``generate(..., device=...)`` device is used; if that is
    # also omitted, generation defaults to CPU.
    device: DeviceLike = None


@dataclass(init=False)
class TensorInput(NumericsInputGenerator):
    """Generator for one tensor operand."""

    config: TensorInputConfig

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
        *,
        device: DeviceLike = None,
    ) -> None:
        self.config = TensorInputConfig(shape=shape, dtype=dtype, device=device)
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.shape = _normalize_shape(self.config.shape)

    def generate(self, *, seed: int, device: DeviceLike = None) -> torch.Tensor | None:
        if self.config.dtype is None:
            return None

        target_device = _resolve_device(self.config.device, device)
        generator = _rng_for_device(target_device, seed)
        return _generate_torch_tensor(
            self.config.shape,
            self.config.dtype,
            device=target_device,
            generator=generator,
        )


def _tensor_input_from_config(config: TensorInputConfig) -> TensorInput:
    return TensorInput(config.shape, config.dtype, device=config.device)


@dataclass
class ScaledTensorValues:
    """Generated values for a ``ScaledTensorInput``."""

    values: torch.Tensor | None
    scales: torch.Tensor | None


@dataclass
class ScaledTensorInputConfig:
    """Initialization parameters for ``ScaledTensorInput``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: logical unscaled value shape. Custom packed dtypes may adjust
    # the physical shape used by child tensor generation.
    value_shape: tuple[int, ...]

    # Required: generated value dtype. ``CustomDType`` values use custom
    # storage handling; ``None`` skips value generation.
    value_dtype: InputDType

    # Required: generated scale shape. ``None`` skips scale child generation.
    scale_shape: tuple[int, ...] | None

    # Required: generated scale dtype. ``None`` skips scale values.
    scale_dtype: torch.dtype | None

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: value tensor device override.
    value_device: DeviceLike = None

    # Optional: scale tensor device override.
    scale_device: DeviceLike = None

    # ------------------------------------------------------------------
    # Optional child generator configuration.
    # ------------------------------------------------------------------

    # Optional: nested config for generated tensor values.
    values_input: TensorInputConfig | None = None

    # Optional: nested config for generated tensor scales.
    scales_input: TensorInputConfig | None = None


@dataclass(init=False)
class ScaledTensorInput(NumericsInputGenerator):
    """Generator for one tensor plus an optional scale sidecar.

    ``values`` and ``scales`` are returned by ``generate``. Either side can be
    skipped by setting its dtype to ``None``. This is useful for composite
    layers where, for example, a weight tensor is generated but the activation
    tensor is supplied by an earlier operation.

    Child tensor generators are initialized when this generator is constructed
    and reused by ``generate``. Tests can mutate those child generator
    configurations before generating through a parent.
    """

    config: ScaledTensorInputConfig
    values_input: TensorInput | None
    scales_input: TensorInput | None

    def __init__(
        self,
        config: ScaledTensorInputConfig,
    ) -> None:
        self.config = config
        self.values_input = None
        self.scales_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.value_shape = _normalize_shape(self.config.value_shape)
        if self.config.scale_shape is not None:
            self.config.scale_shape = _normalize_shape(self.config.scale_shape)
        if self.values_input is None:
            values_config = self.config.values_input or TensorInputConfig(
                self.config.value_shape,
                None
                if self.config.value_dtype is None
                else _storage_dtype(self.config.value_dtype),
                device=self.config.value_device,
            )
            self.values_input = _tensor_input_from_config(values_config)
        if self.scales_input is None and self.config.scale_shape is not None:
            scales_config = self.config.scales_input or TensorInputConfig(
                self.config.scale_shape,
                self.config.scale_dtype,
                device=self.config.scale_device,
            )
            self.scales_input = _tensor_input_from_config(scales_config)
        self.config.values_input = (
            None if self.values_input is None else self.values_input.config
        )
        self.config.scales_input = (
            None if self.scales_input is None else self.scales_input.config
        )

    def generate(self, *, seed: int, device: DeviceLike = None) -> ScaledTensorValues:
        if self.values_input is None:
            raise ValueError("ScaledTensorInput requires a values_input generator")
        value_dtype = self._resolved_value_dtype()
        value_device = _resolve_device(self.config.value_device, device)
        value_generator = _rng_for_device(value_device, _child_seed(seed, 1))
        values = self._generate_values(
            dtype=value_dtype,
            device=value_device,
            generator=value_generator,
        )

        if self.scales_input is None:
            return ScaledTensorValues(
                values=values,
                scales=None,
            )

        scale_dtype = self.scales_input.config.dtype
        if scale_dtype is None:
            return ScaledTensorValues(
                values=values,
                scales=None,
            )

        scale_device = _resolve_device(self.config.scale_device, device)
        scale_generator = _rng_for_device(scale_device, _child_seed(seed, 2))
        scales = _generate_scale_tensor(
            self.scales_input.config.shape,
            scale_dtype,
            device=scale_device,
            generator=scale_generator,
            max_value=self._scale_max_value(),
        )
        return ScaledTensorValues(
            values=values,
            scales=scales,
        )

    def _generate_values(
        self,
        *,
        dtype: InputDType,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        if dtype is None:
            return None
        storage_shape = (
            self.values_input.config.shape
            if self.values_input
            else self.config.value_shape
        )
        if dtype == CustomDType.MXFP4:
            return _generate_mxfp4_packed(
                storage_shape,
                device=device,
                generator=generator,
            )
        if isinstance(dtype, torch.dtype):
            return _generate_scaled_torch_values(
                storage_shape,
                dtype,
                device=device,
                generator=generator,
            )
        raise TypeError(f"unsupported scaled tensor dtype={dtype!r}")

    def _scale_max_value(self) -> float:
        if self.config.value_dtype == CustomDType.MXFP4:
            return _MXFP4_SCALE_MAX
        return 1.0

    def _resolved_value_dtype(self) -> InputDType:
        if self.config.value_dtype == CustomDType.MXFP4:
            return CustomDType.MXFP4
        if self.values_input is not None:
            return self.values_input.config.dtype
        return self.config.value_dtype
