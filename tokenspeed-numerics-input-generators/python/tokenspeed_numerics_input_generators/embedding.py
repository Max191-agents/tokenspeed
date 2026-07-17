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

"""Embedding-family input generators for rotary positional embeddings.

This family models position-dependent rotations over query/key tensors,
including decomposed MLA query/key slices. The generated values describe the
mathematical inputs; output allocation, quantization, backend-specific wrapper
names, and keyword conventions remain adapter concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _resolve_device,
    _rng_for_device,
)
from tokenspeed_numerics_input_generators.rotary import build_rope_cos_sin_cache

__all__ = [
    "MLARopeInputConfig",
    "MLARopeInputs",
    "MLARopeInputValues",
    "RopeFusedKVInputValues",
    "RopeInputConfig",
    "RopeInputValues",
    "RopeInputs",
]

MLARopeKRank = Literal[2, 3]

_REGULAR_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_MLA_ROPE_DTYPES = _REGULAR_FLOAT_DTYPES | {
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
}


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


def _check_float_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _REGULAR_FLOAT_DTYPES:
        raise ValueError(f"{name} must be a regular floating torch dtype, got {dtype}")
    return dtype


def _check_mla_rope_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _MLA_ROPE_DTYPES:
        raise ValueError(
            f"{name} must be a supported floating torch dtype, got {dtype}"
        )
    return dtype


def _check_index_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be torch.int32 or torch.int64, got {dtype}")
    return dtype


def _check_rope_dims(*, head_size: int, rotary_dim: int) -> None:
    if rotary_dim <= 0:
        raise ValueError(f"rotary_dim must be positive, got {rotary_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if rotary_dim > head_size:
        raise ValueError(
            f"rotary_dim must be <= head_size, got {rotary_dim} > {head_size}"
        )


def _require_tensor(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} generation unexpectedly returned None")
    return values


def _generate_positions(
    *,
    num_tokens: int,
    max_position: int,
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    generator = _rng_for_device(target_device, seed)
    return torch.randint(
        0,
        max_position,
        (num_tokens,),
        dtype=dtype,
        device=target_device,
        generator=generator,
    )


def _generate_cache_locs(
    *,
    num_tokens: int,
    cache_size: int,
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    if num_tokens == 0:
        return torch.empty(0, dtype=dtype, device=target_device)
    generator = _rng_for_device(target_device, seed)
    permutation = torch.randperm(cache_size, device=target_device, generator=generator)
    return permutation[:num_tokens].to(dtype)


@dataclass
class RopeFusedKVInputValues:
    """Generated values for optional fused KV-cache writes in RoPE."""

    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    cache_loc: torch.Tensor
    k_scale: None = None
    v_scale: None = None


@dataclass
class RopeInputValues:
    """Generated values for ``RopeInputs``."""

    positions: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    cos_sin_cache: torch.Tensor
    fused_kv: RopeFusedKVInputValues | None
    output_q_rope: torch.Tensor | None
    output_k_rope: torch.Tensor | None


@dataclass
class MLARopeInputValues:
    """Generated values for rotary embedding over decomposed MLA Q/K.

    ``q_nope`` and ``k_nope`` are the non-rotary query/key slices. ``q_rope``
    and ``k_rope`` are the position-encoded slices before rotation.
    """

    q_rope: torch.Tensor
    k_rope: torch.Tensor
    q_nope: torch.Tensor
    k_nope: torch.Tensor
    cos_sin_cache: torch.Tensor
    positions: torch.Tensor


@dataclass
class MLARopeInputConfig:
    """Initialization parameters for decomposed MLA rotary embedding inputs.

    The represented operation applies rotary embedding to the PE slices of
    query/key tensors while leaving the NOPE slices unchanged. The K tensors
    may use the 2D MLA shape with a shared key head, or a 3D shape with an
    explicit KV-head axis.
    """

    # Required: number of token rows. Zero is valid.
    num_tokens: int

    # Required: number of query heads.
    num_q_heads: int

    # Required: non-rotary query/key width.
    qk_nope_head_dim: int

    # Required: rotary query/key width. Must be positive and even.
    qk_rope_head_dim: int

    # Required: generated dtype for all Q/K slices.
    dtype: torch.dtype

    # Optional: K tensor rank. Rank-2 is the MLA shared-K shape; rank-3 has an
    # explicit KV-head axis.
    k_rank: MLARopeKRank = 2

    # Optional: number of KV heads. Rank-2 uses the implicit shared KV head and
    # therefore requires this to remain 1.
    num_kv_heads: int = 1

    # Optional: number of rows in the generated RoPE cache.
    max_position: int = 1024

    # Optional: RoPE frequency base used to build the cache.
    rope_base: float = 10000.0

    # Optional: True for NEOX half-split rotation; False for GPT-J interleaved
    # pair rotation.
    is_neox: bool = True

    # Optional: dtype for positions.
    position_dtype: torch.dtype = torch.int64

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class RopeInputConfig:
    """Initialization parameters for rotary positional embedding inputs.

    The represented operation rotates the first ``rotary_dim`` channels of
    every query and key head using position-indexed cosine and sine values.
    Channels after ``rotary_dim`` are copied through unchanged.
    """

    # Required: number of token rows. Zero is valid.
    num_tokens: int

    # Required: number of query heads.
    num_q_heads: int

    # Required: number of key/value heads.
    num_kv_heads: int

    # Required: per-head width.
    head_size: int

    # Required: generated dtype for query, key, and optional value/cache.
    dtype: torch.dtype

    # Optional: number of rotated channels. Defaults to full-head RoPE.
    rotary_dim: int | None = None

    # Optional: number of rows in the generated RoPE cache.
    max_position: int = 1024

    # Optional: RoPE frequency base used to build the cache.
    rope_base: float = 10000.0

    # Optional: True for NEOX half-split rotation; False for GPT-J interleaved
    # pair rotation.
    is_neox: bool = True

    # Optional: dtype for positions.
    position_dtype: torch.dtype = torch.int64

    # Optional: generate fused KV-cache write inputs.
    with_fused_kv: bool = False

    # Optional: cache row count for fused KV writes. Defaults to num_tokens
    # when fused KV is enabled.
    cache_size: int | None = None

    # Optional: dtype for cache locations.
    cache_loc_dtype: torch.dtype = torch.int32

    # Optional: generate a separate output buffer for query.
    with_q_output: bool = False

    # Optional: generate a separate output buffer for key.
    with_k_output: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class RopeInputs(NumericsInputGenerator):
    """Generator for rotary positional embedding inputs."""

    config: RopeInputConfig
    query_input: TensorInput | None
    key_input: TensorInput | None
    value_input: TensorInput | None

    def __init__(self, config: RopeInputConfig) -> None:
        self.config = config
        self.query_input = None
        self.key_input = None
        self.value_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.head_size = _check_positive("head_size", self.config.head_size)
        if self.config.rotary_dim is None:
            self.config.rotary_dim = self.config.head_size
        self.config.rotary_dim = int(self.config.rotary_dim)
        _check_rope_dims(
            head_size=self.config.head_size,
            rotary_dim=self.config.rotary_dim,
        )
        self.config.max_position = _check_positive(
            "max_position", self.config.max_position
        )
        if self.config.rope_base <= 0.0:
            raise ValueError(f"rope_base must be positive, got {self.config.rope_base}")
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.position_dtype = _check_index_dtype(
            "position_dtype", self.config.position_dtype
        )
        self.config.cache_loc_dtype = _check_index_dtype(
            "cache_loc_dtype", self.config.cache_loc_dtype
        )
        if self.config.with_fused_kv:
            if self.config.cache_size is None:
                self.config.cache_size = max(1, self.config.num_tokens)
            self.config.cache_size = _check_positive(
                "cache_size", self.config.cache_size
            )
            if self.config.cache_size < self.config.num_tokens:
                raise ValueError(
                    "cache_size must be >= num_tokens for unique cache locations; "
                    f"got cache_size={self.config.cache_size}, "
                    f"num_tokens={self.config.num_tokens}"
                )

        query_size = self.config.num_q_heads * self.config.head_size
        key_size = self.config.num_kv_heads * self.config.head_size
        self.query_input = self.query_input or TensorInput(
            (self.config.num_tokens, query_size),
            self.config.dtype,
            device=self.config.device,
        )
        self.key_input = self.key_input or TensorInput(
            (self.config.num_tokens, key_size),
            self.config.dtype,
            device=self.config.device,
        )
        if self.config.with_fused_kv:
            self.value_input = self.value_input or TensorInput(
                (
                    self.config.num_tokens,
                    self.config.num_kv_heads,
                    self.config.head_size,
                ),
                self.config.dtype,
                device=self.config.device,
            )
        else:
            self.value_input = None

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> RopeInputValues:
        self.__post_init__()
        if self.query_input is None or self.key_input is None:
            raise ValueError("RopeInputs child generators must be initialized")
        query = _require_tensor(
            self.query_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "query",
        )
        key = _require_tensor(
            self.key_input.generate(seed=_child_seed(seed, 2), device=device).values,
            "key",
        )
        target_device = _resolve_device(self.config.device, device)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        positions = _generate_positions(
            num_tokens=self.config.num_tokens,
            max_position=self.config.max_position,
            dtype=self.config.position_dtype,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        cos_sin_cache = build_rope_cos_sin_cache(
            rotary_dim=int(self.config.rotary_dim),
            max_position=self.config.max_position,
            base=self.config.rope_base,
            device=target_device,
        )
        fused_kv = None
        if self.config.with_fused_kv:
            if self.value_input is None:
                raise ValueError("fused KV value generator must be initialized")
            value = _require_tensor(
                self.value_input.generate(
                    seed=_child_seed(seed, 3), device=device
                ).values,
                "value",
            )
            assert self.config.cache_size is not None
            cache_shape = (
                self.config.cache_size,
                self.config.num_kv_heads * self.config.head_size,
            )
            k_buffer = torch.zeros(
                cache_shape, dtype=self.config.dtype, device=target_device
            )
            v_buffer = torch.zeros_like(k_buffer)
            cache_loc = _generate_cache_locs(
                num_tokens=self.config.num_tokens,
                cache_size=self.config.cache_size,
                dtype=self.config.cache_loc_dtype,
                seed=_child_seed(metadata_base_seed, 2),
                device=device,
                configured_device=self.config.device,
            )
            fused_kv = RopeFusedKVInputValues(
                value=value.contiguous(),
                k_buffer=k_buffer.contiguous(),
                v_buffer=v_buffer.contiguous(),
                cache_loc=cache_loc.contiguous(),
            )

        output_q_rope = torch.empty_like(query) if self.config.with_q_output else None
        output_k_rope = torch.empty_like(key) if self.config.with_k_output else None
        return RopeInputValues(
            positions=positions.contiguous(),
            query=query.contiguous(),
            key=key.contiguous(),
            cos_sin_cache=cos_sin_cache.contiguous(),
            fused_kv=fused_kv,
            output_q_rope=output_q_rope,
            output_k_rope=output_k_rope,
        )


@dataclass(init=False)
class MLARopeInputs(NumericsInputGenerator):
    """Generator for rotary embedding over decomposed MLA Q/K inputs."""

    config: MLARopeInputConfig
    q_rope_input: TensorInput | None
    k_rope_input: TensorInput | None
    q_nope_input: TensorInput | None
    k_nope_input: TensorInput | None

    def __init__(self, config: MLARopeInputConfig) -> None:
        self.config = config
        self.q_rope_input = None
        self.k_rope_input = None
        self.q_nope_input = None
        self.k_nope_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.qk_nope_head_dim = _check_positive(
            "qk_nope_head_dim", self.config.qk_nope_head_dim
        )
        self.config.qk_rope_head_dim = _check_positive(
            "qk_rope_head_dim", self.config.qk_rope_head_dim
        )
        _check_rope_dims(
            head_size=self.config.qk_rope_head_dim,
            rotary_dim=self.config.qk_rope_head_dim,
        )
        self.config.dtype = _check_mla_rope_dtype("dtype", self.config.dtype)
        if self.config.k_rank not in (2, 3):
            raise ValueError(f"k_rank must be 2 or 3, got {self.config.k_rank}")
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        if self.config.k_rank == 2 and self.config.num_kv_heads != 1:
            raise ValueError(
                "rank-2 MLA K tensors use an implicit shared KV head; "
                f"num_kv_heads must be 1, got {self.config.num_kv_heads}"
            )
        self.config.max_position = _check_positive(
            "max_position", self.config.max_position
        )
        if self.config.rope_base <= 0.0:
            raise ValueError(f"rope_base must be positive, got {self.config.rope_base}")
        self.config.position_dtype = _check_index_dtype(
            "position_dtype", self.config.position_dtype
        )
        q_rope_shape = (
            self.config.num_tokens,
            self.config.num_q_heads,
            self.config.qk_rope_head_dim,
        )
        q_nope_shape = (
            self.config.num_tokens,
            self.config.num_q_heads,
            self.config.qk_nope_head_dim,
        )
        if self.config.k_rank == 2:
            k_rope_shape = (
                self.config.num_tokens,
                self.config.qk_rope_head_dim,
            )
            k_nope_shape = (
                self.config.num_tokens,
                self.config.qk_nope_head_dim,
            )
        else:
            k_rope_shape = (
                self.config.num_tokens,
                self.config.num_kv_heads,
                self.config.qk_rope_head_dim,
            )
            k_nope_shape = (
                self.config.num_tokens,
                self.config.num_kv_heads,
                self.config.qk_nope_head_dim,
            )

        self.q_rope_input = self.q_rope_input or TensorInput(
            q_rope_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.q_nope_input = self.q_nope_input or TensorInput(
            q_nope_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.k_rope_input = self.k_rope_input or TensorInput(
            k_rope_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.k_nope_input = self.k_nope_input or TensorInput(
            k_nope_shape,
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MLARopeInputValues:
        self.__post_init__()
        if (
            self.q_rope_input is None
            or self.k_rope_input is None
            or self.q_nope_input is None
            or self.k_nope_input is None
        ):
            raise ValueError("MLARopeInputs child generators must be initialized")
        target_device = _resolve_device(self.config.device, device)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        q_rope = _require_tensor(
            self.q_rope_input.generate(
                seed=_child_seed(seed, 1), device=target_device
            ).values,
            "q_rope",
        ).contiguous()
        k_rope = _require_tensor(
            self.k_rope_input.generate(
                seed=_child_seed(seed, 2), device=target_device
            ).values,
            "k_rope",
        ).contiguous()
        q_nope = _require_tensor(
            self.q_nope_input.generate(
                seed=_child_seed(seed, 3), device=target_device
            ).values,
            "q_nope",
        ).contiguous()
        k_nope = _require_tensor(
            self.k_nope_input.generate(
                seed=_child_seed(seed, 4), device=target_device
            ).values,
            "k_nope",
        ).contiguous()
        positions = _generate_positions(
            num_tokens=self.config.num_tokens,
            max_position=self.config.max_position,
            dtype=self.config.position_dtype,
            seed=_child_seed(metadata_base_seed, 1),
            device=target_device,
            configured_device=self.config.device,
        )
        cos_sin_cache = build_rope_cos_sin_cache(
            rotary_dim=self.config.qk_rope_head_dim,
            max_position=self.config.max_position,
            base=self.config.rope_base,
            device=target_device,
        )
        return MLARopeInputValues(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache.contiguous(),
            positions=positions.contiguous(),
        )
