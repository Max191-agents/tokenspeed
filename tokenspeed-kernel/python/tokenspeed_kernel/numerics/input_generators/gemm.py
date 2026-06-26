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
from typing import Literal, Self

import torch
from tokenspeed_kernel.numerics.input_generators.core import (
    CustomDType,
    DeviceLike,
    InputDType,
    NumericsInputGenerator,
    ScaledTensorInput,
    ScaledTensorValues,
    TensorInput,
    _child_seed,
    _normalize_shape,
    _packed_mxfp4_shape,
)
from tokenspeed_kernel.signature import ScaleFormat, TensorFormat

__all__ = [
    "GemmInputs",
    "GemmInputConfig",
    "GemmInputValues",
    "ScaledGemmInputs",
    "ScaledGemmInputConfig",
    "ScaledGemmInputValues",
    "gemm_scale_shape",
]

GemmLayout = Literal["MK", "KM", "NK", "KN"]

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
    scale_format: ScaleFormat | None,
    role: Literal["a", "b"],
    *,
    M: int,
    N: int,
    K: int,
    batch_shape: tuple[int, ...] = (),
) -> tuple[int, ...] | None:
    """Return the physical scale shape for a GEMM operand role."""

    if scale_format is None:
        return None

    if scale_format.granularity == "tensor":
        return (1,)

    if scale_format.granularity == "channel":
        return batch_shape + ((M,) if role == "a" else (N,))

    if scale_format.granularity != "block":
        return (1,)

    if scale_format.block_shape is None:
        raise ValueError("block scale format requires concrete block_shape")

    if len(scale_format.block_shape) == 1:
        block_k = scale_format.block_shape[0]
        return batch_shape + (
            (M, math.ceil(K / block_k)) if role == "a" else (N, math.ceil(K / block_k))
        )

    if len(scale_format.block_shape) == 2:
        block_n, block_k = scale_format.block_shape
        if role == "a":
            return batch_shape + (M, math.ceil(K / block_k))
        return batch_shape + (math.ceil(N / block_n), math.ceil(K / block_k))

    raise ValueError(
        f"GEMM scale format supports 1D or 2D block shapes, got "
        f"{scale_format.block_shape}"
    )


def _input_dtype_from_format(tensor_format: TensorFormat) -> InputDType:
    if tensor_format.format == "mxfp4":
        if tensor_format.storage_dtype != torch.uint8:
            raise ValueError("mxfp4 values must use torch.uint8 storage")
        return CustomDType.MXFP4
    return tensor_format.storage_dtype


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
            ScaledTensorInput(
                value_shape=shape,
                value_dtype=dtype,
                scale_shape=None,
                scale_dtype=None,
                value_device=configured_device,
            )
            .generate(seed=seed, device=device)
            .values
        )
    if dtype is None or isinstance(dtype, torch.dtype):
        return TensorInput(
            shape,
            dtype,
            device=configured_device,
        ).generate(seed=seed, device=device)
    raise TypeError(f"unsupported GEMM dtype={dtype!r}")


@dataclass
class GemmInputValues:
    """Generated values for ``GemmInputs``."""

    A: torch.Tensor | None
    B: torch.Tensor | None
    C: torch.Tensor


@dataclass
class ScaledGemmInputValues:
    """Generated values for ``ScaledGemmInputs``."""

    A: ScaledTensorValues
    B: ScaledTensorValues
    C: torch.Tensor


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

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: generated A device override.
    a_device: DeviceLike = None

    # Optional: generated B device override.
    b_device: DeviceLike = None

    # Optional: generated C device override.
    c_device: DeviceLike = None


@dataclass(init=False)
class GemmInputs(NumericsInputGenerator):
    """Typed input generator for dense GEMM-like calls.

    ``M``, ``N``, and ``K`` are the logical matrix dimensions for ``A @ B.T``
    style kernels used by TokenSpeed. ``generate`` returns generated ``A``,
    ``B``, and ``C`` tensors. ``A`` and ``B`` may use dtype ``None`` to skip
    generation when a layer generator only needs one side of the GEMM, but
    ``C`` is always generated.

    Mutate ``generator.config`` to adjust dtype, layout, batch shape, or device
    before calling ``generate``.
    """

    config: GemmInputConfig

    def __init__(
        self,
        config: GemmInputConfig | None = None,
        **kwargs: object,
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword parameters, not both")

        self.config = config or GemmInputConfig(**kwargs)  # type: ignore[arg-type]
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

    def generate(self, *, seed: int, device: DeviceLike = None) -> GemmInputValues:
        A = _generate_tensor(
            shape=self._value_shape("a"),
            dtype=self.config.a_dtype,
            seed=_child_seed(seed, 1),
            configured_device=self.config.a_device,
            device=device,
        )
        B = _generate_tensor(
            shape=self._value_shape("b"),
            dtype=self.config.b_dtype,
            seed=_child_seed(seed, 2),
            configured_device=self.config.b_device,
            device=device,
        )
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
            A=A,
            B=B,
            C=C,
        )


@dataclass
class ScaledGemmInputConfig:
    """Initialization parameters for ``ScaledGemmInputs``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: logical M dimension for A @ B.T style GEMM generation.
    M: int

    # Required: logical N dimension for A @ B.T style GEMM generation.
    N: int

    # Required: logical K reduction dimension.
    K: int

    # Required: generated A value dtype. ``None`` skips A values/scales.
    a_dtype: InputDType

    # Required: generated B value dtype. ``None`` skips B values/scales.
    b_dtype: InputDType

    # Required: generated A scale dtype. ``None`` skips A scales.
    a_scale_dtype: torch.dtype | None

    # Required: generated B scale dtype. ``None`` skips B scales.
    b_scale_dtype: torch.dtype | None

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

    # Optional: physical A scale shape.
    a_scale_shape: tuple[int, ...] | None = (1,)

    # Optional: physical B scale shape.
    b_scale_shape: tuple[int, ...] | None = (1,)

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: generated A value device override.
    a_device: DeviceLike = None

    # Optional: generated B value device override.
    b_device: DeviceLike = None

    # Optional: generated A scale device override.
    a_scale_device: DeviceLike = None

    # Optional: generated B scale device override.
    b_scale_device: DeviceLike = None

    # Optional: generated C device override.
    c_device: DeviceLike = None


@dataclass(init=False)
class ScaledGemmInputs(NumericsInputGenerator):
    """Typed input generator for GEMM with scaled ``A`` and ``B`` operands.

    ``generate`` returns nested values where tensors are available as
    ``A.values``, ``A.scales``, ``B.values``, and ``B.scales``. ``A`` and
    ``B`` may skip their values/scales when their dtype or scale dtype is
    ``None``. ``C`` is always generated.

    Mutate ``generator.config`` to adjust dtype, scale dtype, shape, layout, or
    device before calling ``generate``.
    """

    config: ScaledGemmInputConfig

    def __init__(
        self,
        config: ScaledGemmInputConfig | None = None,
        **kwargs: object,
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword parameters, not both")

        self.config = config or ScaledGemmInputConfig(**kwargs)  # type: ignore[arg-type]
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

    @classmethod
    def from_formats(
        cls,
        *,
        M: int,
        N: int,
        K: int,
        a_tensor_format: TensorFormat,
        b_tensor_format: TensorFormat,
        c_dtype: torch.dtype,
        a_layout: GemmLayout = "MK",
        b_layout: GemmLayout = "NK",
        batch_shape: tuple[int, ...] = (),
    ) -> Self:
        """Build a scaled GEMM generator from signature metadata."""

        a_scale_shape = gemm_scale_shape(
            a_tensor_format.scale,
            "a",
            M=M,
            N=N,
            K=K,
            batch_shape=batch_shape,
        )
        b_scale_shape = gemm_scale_shape(
            b_tensor_format.scale,
            "b",
            M=M,
            N=N,
            K=K,
            batch_shape=batch_shape,
        )
        return cls(
            M=M,
            N=N,
            K=K,
            a_dtype=_input_dtype_from_format(a_tensor_format),
            b_dtype=_input_dtype_from_format(b_tensor_format),
            a_scale_dtype=(
                None
                if a_tensor_format.scale is None
                else a_tensor_format.scale.storage_dtype
            ),
            b_scale_dtype=(
                None
                if b_tensor_format.scale is None
                else b_tensor_format.scale.storage_dtype
            ),
            c_dtype=c_dtype,
            a_layout=a_layout,
            b_layout=b_layout,
            batch_shape=batch_shape,
            a_scale_shape=a_scale_shape,
            b_scale_shape=b_scale_shape,
        )

    @classmethod
    def mxfp4(
        cls,
        *,
        M: int,
        N: int,
        K: int,
        c_dtype: torch.dtype,
        scale_dtype: torch.dtype = torch.float8_e4m3fn,
        a_dtype: InputDType = CustomDType.MXFP4,
        b_dtype: InputDType = CustomDType.MXFP4,
        a_layout: GemmLayout = "MK",
        b_layout: GemmLayout = "NK",
        batch_shape: tuple[int, ...] = (),
        block_size: int = _DEFAULT_MXFP4_BLOCK_SIZE,
    ) -> Self:
        """Build an mxfp4 scaled GEMM generator with fp8-compatible scales."""

        scale = ScaleFormat(
            storage_dtype=scale_dtype,
            granularity="block",
            block_shape=(block_size,),
        )
        return cls(
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
                    scale,
                    "a",
                    M=M,
                    N=N,
                    K=K,
                    batch_shape=batch_shape,
                )
            ),
            b_scale_shape=(
                None
                if b_dtype is None
                else gemm_scale_shape(
                    scale,
                    "b",
                    M=M,
                    N=N,
                    K=K,
                    batch_shape=batch_shape,
                )
            ),
        )

    def _scaled_value_shape(
        self,
        role: Literal["a", "b"],
    ) -> tuple[int, ...]:
        is_a = role == "a"
        return _gemm_value_shape(
            role=role,
            M=self.config.M,
            N=self.config.N,
            K=self.config.K,
            batch_shape=self.config.batch_shape,
            layout=self.config.a_layout if is_a else self.config.b_layout,
            dtype=self.config.a_dtype if is_a else self.config.b_dtype,
        )

    def _c_shape(self) -> tuple[int, ...]:
        return _gemm_value_shape(
            role="c",
            M=self.config.M,
            N=self.config.N,
            K=self.config.K,
            batch_shape=self.config.batch_shape,
            layout="MK",
        )

    def _generate_scaled(
        self,
        role: Literal["a", "b"],
        *,
        seed: int,
        device: DeviceLike,
    ) -> ScaledTensorValues:
        is_a = role == "a"
        return ScaledTensorInput(
            value_shape=self._scaled_value_shape(role),
            value_dtype=self.config.a_dtype if is_a else self.config.b_dtype,
            scale_shape=self.config.a_scale_shape
            if is_a
            else self.config.b_scale_shape,
            scale_dtype=self.config.a_scale_dtype
            if is_a
            else self.config.b_scale_dtype,
            value_device=self.config.a_device if is_a else self.config.b_device,
            scale_device=self.config.a_scale_device
            if is_a
            else self.config.b_scale_device,
        ).generate(seed=seed, device=device)

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> ScaledGemmInputValues:
        a_values = self._generate_scaled("a", seed=_child_seed(seed, 1), device=device)
        b_values = self._generate_scaled("b", seed=_child_seed(seed, 2), device=device)
        C = _generate_tensor(
            shape=self._c_shape(),
            dtype=self.config.c_dtype,
            seed=_child_seed(seed, 3),
            configured_device=self.config.c_device,
            device=device,
        )
        if C is None:
            raise ValueError("c_dtype is required")
        return ScaledGemmInputValues(
            A=a_values,
            B=b_values,
            C=C,
        )
