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

This family models position-dependent rotations over query/key tensors and
MLA-style fused RoPE plus FP8 quantization. The generated values describe the
mathematical tensor inputs and output buffers; backend-specific wrapper names
or keyword conventions remain adapter concerns.
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
    "MLARopeQuantizeFP8InputConfig",
    "MLARopeQuantizeFP8Inputs",
    "MLARopeQuantizeFP8InputValues",
    "MLARopeQuantizeFP8ReferenceValues",
    "RopeFusedKVInputValues",
    "RopeInputConfig",
    "RopeInputValues",
    "RopeInputs",
    "mla_rope_quantize_fp8_reference",
    "rope_reference",
]

MLARopeKRank = Literal[2, 3]

_REGULAR_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
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


def _check_fp16_or_bf16_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{name} must be torch.float16 or torch.bfloat16, got {dtype}")
    return dtype


def _check_fp8_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"{name} must be an FP8 torch dtype, got {dtype}")
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


def _apply_rope_to_pe_slice(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    is_neox: bool,
) -> torch.Tensor:
    """Apply RoPE to a PE-only tensor whose last dimension is rotary width."""

    rotary_dim = x.shape[-1]
    _check_rope_dims(head_size=rotary_dim, rotary_dim=rotary_dim)
    if positions.shape != (x.shape[0],):
        raise ValueError(
            f"positions must have shape {(x.shape[0],)}, got {tuple(positions.shape)}"
        )
    if cos_sin_cache.shape[-1] != rotary_dim:
        raise ValueError(
            "cos_sin_cache last dimension must equal rotary width; got "
            f"{cos_sin_cache.shape[-1]} and {rotary_dim}"
        )
    if positions.numel() > 0:
        min_pos = int(positions.min().item())
        max_pos = int(positions.max().item())
        if min_pos < 0 or max_pos >= cos_sin_cache.shape[0]:
            raise ValueError(
                "positions must index the generated cos_sin_cache; got range "
                f"[{min_pos}, {max_pos}] for cache size {cos_sin_cache.shape[0]}"
            )

    original_shape = x.shape
    x_view = x.reshape(x.shape[0], -1, rotary_dim)
    half = rotary_dim // 2
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64))
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(torch.float32)
    sin = sin.unsqueeze(-2).to(torch.float32)
    x_rot = x_view.to(torch.float32)
    if is_neox:
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        rotated = rotated.flatten(-2)
    return rotated.to(x.dtype).reshape(original_shape)


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
class MLARopeQuantizeFP8InputValues:
    """Generated values for fused RoPE plus FP8 quantization.

    ``q_nope`` and ``k_nope`` are the non-rotary query/key slices. ``q_rope``
    and ``k_rope`` are the position-encoded slices before rotation. The output
    tensors are preallocated FP8 buffers for the mutating kernel API; the
    reference also combines the corresponding NOPE and RoPE outputs into full
    query/key tensors.
    """

    q_rope: torch.Tensor
    k_rope: torch.Tensor
    q_nope: torch.Tensor
    k_nope: torch.Tensor
    cos_sin_cache: torch.Tensor
    positions: torch.Tensor
    q_rope_out: torch.Tensor
    k_rope_out: torch.Tensor
    q_nope_out: torch.Tensor
    k_nope_out: torch.Tensor
    quant_scale_q: float
    quant_scale_kv: float
    is_neox: bool
    fp8_dtype: torch.dtype


@dataclass
class MLARopeQuantizeFP8ReferenceValues:
    """Reference outputs for fused RoPE plus FP8 quantization."""

    query: torch.Tensor
    key: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor


@dataclass
class MLARopeQuantizeFP8InputConfig:
    """Initialization parameters for fused MLA RoPE + FP8 quantization.

    The represented operation applies rotary embedding to the PE slices of
    query/key tensors, scales the rotated PE slices and unrotated NOPE slices,
    and casts the results to FP8. The K tensors may use the 2D MLA shape with a
    shared key head, or the 3D GQA/MHA shape with an explicit KV-head axis.
    """

    # Required: number of token rows. Zero is valid.
    num_tokens: int

    # Required: number of query heads.
    num_q_heads: int

    # Required: non-rotary query/key width.
    qk_nope_head_dim: int

    # Required: rotary query/key width. Must be positive and even.
    qk_rope_head_dim: int

    # Required: generated dtype for Q/K inputs. FlashInfer supports fp16/bf16.
    input_dtype: torch.dtype

    # Optional: K tensor rank. Rank-2 is the MLA shared-K shape; rank-3 has an
    # explicit KV-head axis.
    k_rank: MLARopeKRank = 2

    # Optional: number of KV heads. Rank-2 uses the implicit shared KV head and
    # therefore requires this to remain 1.
    num_kv_heads: int = 1

    # Optional: FP8 output dtype.
    fp8_dtype: torch.dtype = torch.float8_e4m3fn

    # Optional: query output multiplier before FP8 cast.
    quant_scale_q: float = 1.0

    # Optional: key output multiplier before FP8 cast.
    quant_scale_kv: float = 1.0

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
class MLARopeQuantizeFP8Inputs(NumericsInputGenerator):
    """Generator for fused MLA/GQA RoPE plus FP8 quantization inputs."""

    config: MLARopeQuantizeFP8InputConfig
    q_rope_input: TensorInput | None
    k_rope_input: TensorInput | None
    q_nope_input: TensorInput | None
    k_nope_input: TensorInput | None

    def __init__(self, config: MLARopeQuantizeFP8InputConfig) -> None:
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
        self.config.input_dtype = _check_fp16_or_bf16_dtype(
            "input_dtype", self.config.input_dtype
        )
        self.config.fp8_dtype = _check_fp8_dtype("fp8_dtype", self.config.fp8_dtype)
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
        self.config.quant_scale_q = float(self.config.quant_scale_q)
        self.config.quant_scale_kv = float(self.config.quant_scale_kv)
        if self.config.quant_scale_q <= 0.0:
            raise ValueError(
                f"quant_scale_q must be positive, got {self.config.quant_scale_q}"
            )
        if self.config.quant_scale_kv <= 0.0:
            raise ValueError(
                f"quant_scale_kv must be positive, got {self.config.quant_scale_kv}"
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
            self.config.input_dtype,
            device=self.config.device,
        )
        self.q_nope_input = self.q_nope_input or TensorInput(
            q_nope_shape,
            self.config.input_dtype,
            device=self.config.device,
        )
        self.k_rope_input = self.k_rope_input or TensorInput(
            k_rope_shape,
            self.config.input_dtype,
            device=self.config.device,
        )
        self.k_nope_input = self.k_nope_input or TensorInput(
            k_nope_shape,
            self.config.input_dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MLARopeQuantizeFP8InputValues:
        self.__post_init__()
        if (
            self.q_rope_input is None
            or self.k_rope_input is None
            or self.q_nope_input is None
            or self.k_nope_input is None
        ):
            raise ValueError(
                "MLARopeQuantizeFP8Inputs child generators must be initialized"
            )
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
        return MLARopeQuantizeFP8InputValues(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache.contiguous(),
            positions=positions.contiguous(),
            q_rope_out=torch.empty_like(q_rope, dtype=self.config.fp8_dtype),
            k_rope_out=torch.empty_like(k_rope, dtype=self.config.fp8_dtype),
            q_nope_out=torch.empty_like(q_nope, dtype=self.config.fp8_dtype),
            k_nope_out=torch.empty_like(k_nope, dtype=self.config.fp8_dtype),
            quant_scale_q=self.config.quant_scale_q,
            quant_scale_kv=self.config.quant_scale_kv,
            is_neox=self.config.is_neox,
            fp8_dtype=self.config.fp8_dtype,
        )


def rope_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query/key after rotary positional embedding."""

    head_size = _check_positive("head_size", head_size)
    if rotary_dim is None:
        rotary_dim = int(cos_sin_cache.shape[-1])
    rotary_dim = int(rotary_dim)
    _check_rope_dims(head_size=head_size, rotary_dim=rotary_dim)
    if query.shape[0] != key.shape[0]:
        raise ValueError(
            f"query/key token count mismatch: {query.shape[0]} vs {key.shape[0]}"
        )
    if positions.shape[0] != query.shape[0]:
        raise ValueError(
            "positions must have one entry per token; got "
            f"{positions.shape[0]} for {query.shape[0]} tokens"
        )
    if query.shape[-1] % head_size != 0 or key.shape[-1] % head_size != 0:
        raise ValueError("query/key last dimensions must be divisible by head_size")
    if cos_sin_cache.shape[-1] != rotary_dim:
        raise ValueError(
            "cos_sin_cache last dimension must equal rotary_dim; got "
            f"{cos_sin_cache.shape[-1]} and {rotary_dim}"
        )

    num_tokens = query.shape[0]
    half = rotary_dim // 2
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64))
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(torch.float32)
    sin = sin.unsqueeze(-2).to(torch.float32)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x_view = x.view(num_tokens, -1, head_size)
        x_rot = x_view[..., :rotary_dim].to(torch.float32)
        x_tail = x_view[..., rotary_dim:]
        if is_neox:
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        else:
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
            rotated = rotated.flatten(-2)
        out = torch.cat((rotated.to(x.dtype), x_tail), dim=-1)
        return out.reshape_as(x)

    return _apply(query), _apply(key)


def mla_rope_quantize_fp8_reference(
    values: MLARopeQuantizeFP8InputValues,
) -> MLARopeQuantizeFP8ReferenceValues:
    """Return FP8 query/key outputs for fused MLA RoPE quantization."""

    for name, tensor in (
        ("q_rope", values.q_rope),
        ("k_rope", values.k_rope),
        ("q_nope", values.q_nope),
        ("k_nope", values.k_nope),
    ):
        _check_fp16_or_bf16_dtype(f"{name} dtype", tensor.dtype)
    if (
        values.q_rope.dtype != values.k_rope.dtype
        or values.q_rope.dtype != values.q_nope.dtype
        or values.q_rope.dtype != values.k_nope.dtype
    ):
        raise ValueError("q/k rope and nope inputs must have the same dtype")
    if values.q_rope.ndim != 3 or values.q_nope.ndim != 3:
        raise ValueError("q_rope and q_nope must be rank-3")
    if values.k_rope.ndim not in (2, 3) or values.k_nope.ndim != values.k_rope.ndim:
        raise ValueError("k_rope and k_nope must both be rank-2 or both be rank-3")
    if values.q_rope.shape[:2] != values.q_nope.shape[:2]:
        raise ValueError(
            "q_rope and q_nope must have matching token/head dimensions; got "
            f"{tuple(values.q_rope.shape)} and {tuple(values.q_nope.shape)}"
        )
    if values.k_rope.shape[:-1] != values.k_nope.shape[:-1]:
        raise ValueError(
            "k_rope and k_nope must have matching leading dimensions; got "
            f"{tuple(values.k_rope.shape)} and {tuple(values.k_nope.shape)}"
        )
    if values.k_rope.shape[0] != values.q_rope.shape[0]:
        raise ValueError(
            "q/k token dimensions must match; got "
            f"{values.q_rope.shape[0]} and {values.k_rope.shape[0]}"
        )
    if values.k_rope.shape[-1] != values.q_rope.shape[-1]:
        raise ValueError(
            "q/k rope dimensions must match; got "
            f"{values.q_rope.shape[-1]} and {values.k_rope.shape[-1]}"
        )
    if values.k_nope.shape[-1] != values.q_nope.shape[-1]:
        raise ValueError(
            "q/k nope dimensions must match; got "
            f"{values.q_nope.shape[-1]} and {values.k_nope.shape[-1]}"
        )
    fp8_dtype = _check_fp8_dtype("fp8_dtype", values.fp8_dtype)
    for name, tensor, expected_shape in (
        ("q_rope_out", values.q_rope_out, values.q_rope.shape),
        ("k_rope_out", values.k_rope_out, values.k_rope.shape),
        ("q_nope_out", values.q_nope_out, values.q_nope.shape),
        ("k_nope_out", values.k_nope_out, values.k_nope.shape),
    ):
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != fp8_dtype:
            raise ValueError(f"{name} must have dtype {fp8_dtype}, got {tensor.dtype}")
    if values.positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"positions must be int32 or int64, got {values.positions.dtype}"
        )
    if values.quant_scale_q <= 0.0:
        raise ValueError(f"quant_scale_q must be positive, got {values.quant_scale_q}")
    if values.quant_scale_kv <= 0.0:
        raise ValueError(
            f"quant_scale_kv must be positive, got {values.quant_scale_kv}"
        )

    q_rope = _apply_rope_to_pe_slice(
        values.q_rope,
        values.positions,
        values.cos_sin_cache,
        is_neox=values.is_neox,
    )
    k_rope = _apply_rope_to_pe_slice(
        values.k_rope,
        values.positions,
        values.cos_sin_cache,
        is_neox=values.is_neox,
    )
    q_nope = (values.q_nope.float() * values.quant_scale_q).to(fp8_dtype).contiguous()
    q_rope = (q_rope.float() * values.quant_scale_q).to(fp8_dtype).contiguous()
    k_nope = (values.k_nope.float() * values.quant_scale_kv).to(fp8_dtype).contiguous()
    k_rope = (k_rope.float() * values.quant_scale_kv).to(fp8_dtype).contiguous()
    return MLARopeQuantizeFP8ReferenceValues(
        query=torch.cat((q_nope, q_rope), dim=-1).contiguous(),
        key=torch.cat((k_nope, k_rope), dim=-1).contiguous(),
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=k_nope,
        k_rope=k_rope,
    )
