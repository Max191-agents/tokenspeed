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

"""Input generators for standalone tensor transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _normalize_shape,
)

__all__ = [
    "HadamardTransformInputConfig",
    "HadamardTransformInputs",
    "HadamardTransformInputValues",
    "hadamard_transform_reference",
]

_HADAMARD_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_power_of_two(name: str, value: int) -> int:
    value = _check_positive(name, value)
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")
    return value


def _check_hadamard_dtype(name: str, value: torch.dtype) -> torch.dtype:
    if value not in _HADAMARD_DTYPES:
        raise ValueError(
            f"{name} must be one of {sorted(str(dtype) for dtype in _HADAMARD_DTYPES)}; "
            f"got {value}"
        )
    return value


def _resolve_hadamard_scale(transform_dim: int, scale: float | None) -> float:
    if scale is None:
        return transform_dim**-0.5
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale}")
    return scale


@dataclass
class HadamardTransformInputValues:
    """Generated values for ``HadamardTransformInputs``.

    ``x`` is the input tensor whose last dimension is transformed. ``scale`` is
    the concrete multiplier applied after the unnormalized Walsh-Hadamard
    transform.
    """

    x: torch.Tensor
    scale: float


@dataclass
class HadamardTransformInputConfig:
    """Initialization parameters for Walsh-Hadamard transform inputs.

    The represented operation applies a Walsh-Hadamard transform independently
    to every row formed by flattening ``batch_shape`` and using
    ``transform_dim`` as the last dimension. ``transform_dim`` is required to be
    a power of two because the transform is defined recursively by splitting
    each row into equal halves.
    """

    # Required: leading dimensions before the transformed axis. Dimensions may
    # be zero to model empty batches.
    batch_shape: tuple[int, ...]

    # Required: transformed last dimension. Must be a positive power of two.
    transform_dim: int

    # Optional: dtype for the generated input and transformed output.
    dtype: torch.dtype = torch.bfloat16

    # Optional: output scale. If omitted, the orthonormal scale
    # ``transform_dim ** -0.5`` is used, matching TokenSpeed attention/indexer
    # callsites that preserve row norm through the transform.
    scale: float | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class HadamardTransformInputs(NumericsInputGenerator):
    """Generator for standalone Walsh-Hadamard transform inputs."""

    config: HadamardTransformInputConfig
    x_input: TensorInput

    def __init__(self, config: HadamardTransformInputConfig) -> None:
        self.config = config
        self.x_input = TensorInput(
            (1,),
            self.config.dtype,
            device=self.config.device,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.batch_shape = _normalize_shape(self.config.batch_shape)
        self.config.transform_dim = _check_power_of_two(
            "transform_dim", self.config.transform_dim
        )
        self.config.dtype = _check_hadamard_dtype("dtype", self.config.dtype)
        self.config.scale = _resolve_hadamard_scale(
            self.config.transform_dim,
            self.config.scale,
        )
        self.x_input.shape = self._x_shape()
        self.x_input.dtype = self.config.dtype
        self.x_input.device = self.config.device
        self.x_input.__post_init__()

    def _x_shape(self) -> tuple[int, ...]:
        return tuple(self.config.batch_shape) + (int(self.config.transform_dim),)

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> HadamardTransformInputValues:
        self.__post_init__()
        x = self.x_input.generate(seed=seed, device=device).values
        if x is None:
            raise ValueError("HadamardTransformInputs requires an input tensor")
        return HadamardTransformInputValues(
            x=x.contiguous(),
            scale=float(self.config.scale),
        )


def hadamard_transform_reference(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reference Walsh-Hadamard transform over the last dimension.

    The unnormalized transform is defined recursively:

    ```text
    H_1(x) = x
    H_2n([a, b]) = [H_n(a) + H_n(b), H_n(a) - H_n(b)]
    ```

    The returned tensor has the same shape and dtype as ``x``. Floating point
    work is performed in fp32 for fp16/bf16/fp32 inputs and fp64 for fp64
    inputs, then cast back to the input dtype to model same-dtype transform
    kernels.
    """

    if x.dim() == 0:
        raise ValueError("hadamard transform expects at least one dimension")
    transform_dim = _check_power_of_two("x.shape[-1]", int(x.shape[-1]))
    _check_hadamard_dtype("x.dtype", x.dtype)
    scale = _resolve_hadamard_scale(transform_dim, scale)

    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    out = x.to(compute_dtype).reshape(-1, transform_dim).clone()
    stride = 1
    while stride < transform_dim:
        view = out.reshape(-1, transform_dim // (2 * stride), 2, stride)
        left = view[:, :, 0, :].clone()
        right = view[:, :, 1, :].clone()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        stride *= 2

    if scale != 1.0:
        out = out * scale
    return out.reshape_as(x).to(x.dtype)
