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
)

__all__ = [
    "GemmInputs",
    "GemmInputConfig",
    "GemmInputValues",
    "gemm_scale_shape",
    "mxfp4_gemm_input_config",
]

GemmLayout = Literal["MK", "KM", "NK", "KN"]
ScaleGranularity = Literal["tensor", "channel", "block"]

_DEFAULT_MXFP4_BLOCK_SIZE = 32


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
            scale_shape=self.config.a_scale_shape
            if is_a
            else self.config.b_scale_shape,
            scale_dtype=self.config.a_scale_dtype
            if is_a
            else self.config.b_scale_dtype,
            device=self.config.a_device if is_a else self.config.b_device,
            scale_device=self.config.a_scale_device
            if is_a
            else self.config.b_scale_device,
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
