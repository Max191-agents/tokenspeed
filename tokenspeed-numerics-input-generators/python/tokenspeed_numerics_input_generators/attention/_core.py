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

"""Attention-family input generators for numerical correctness tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

import torch
from tokenspeed_numerics_input_generators.attention_cache import (
    KVCacheInput,
    KVCacheInputConfig,
    KVCacheValues,
    MLAKVCacheInput,
    MLAKVCacheInputConfig,
    MLAKVCacheValues,
    PageTableInput,
    PageTableInputConfig,
    PageTableValues,
)
from tokenspeed_numerics_input_generators.attention_metadata import (
    CacheLayout,
    LengthMode,
    MHARequestMetadataInput,
    MHARequestMetadataInputConfig,
    MHARequestMetadataValues,
    PageTableIndexing,
    SlotMappingInput,
    SlotMappingInputConfig,
)
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _resolve_device,
    _rng_for_device,
)
from tokenspeed_numerics_input_generators.rotary import build_rope_cos_sin_cache

_AttentionCacheT = TypeVar("_AttentionCacheT")
_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK = 128
_DSA_SPARSE_DECODE_FP8_SCALE_BYTES = 4
_DSA_SPARSE_DECODE_BF16_BYTES = 2
_DSA_SPARSE_DECODE_FP8_E4M3_MAX = 448.0
_DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT = 128
_DEEPSEEK_V4_HEAD_DIM = 512
_DEEPSEEK_V4_ROPE_DIM = 64
_DEEPSEEK_V4_NOPE_DIM = _DEEPSEEK_V4_HEAD_DIM - _DEEPSEEK_V4_ROPE_DIM
_DEEPSEEK_V4_FP8_QUANT_BLOCK = 64
_DEEPSEEK_V4_FP8_MAX = 448.0
_DEEPSEEK_V4_SWA_TOKEN_STRIDE = _DEEPSEEK_V4_NOPE_DIM + (_DEEPSEEK_V4_ROPE_DIM * 2)
_DEEPSEEK_V4_SWA_SCALE_DIM = _DEEPSEEK_V4_NOPE_DIM // _DEEPSEEK_V4_FP8_QUANT_BLOCK + 1
_DEEPSEEK_V4_INDEXER_DIM = 128
_DEEPSEEK_V4_INDEXER_MXFP4_BLOCK_SIZE = 32
_DEEPSEEK_V4_INDEXER_MXFP4_HALF_BLOCK = _DEEPSEEK_V4_INDEXER_MXFP4_BLOCK_SIZE // 2
_DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES = _DEEPSEEK_V4_INDEXER_DIM // 2
_DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES = (
    _DEEPSEEK_V4_INDEXER_DIM // _DEEPSEEK_V4_INDEXER_MXFP4_BLOCK_SIZE
)
_GDN_CHUNK_SIZE = 64


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _is_power_of_2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _check_power_of_2(name: str, value: int) -> int:
    value = _check_positive(name, value)
    if not _is_power_of_2(value):
        raise ValueError(f"{name} must be a power of two, got {value}")
    return value


def _check_float_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"{name} must be a regular floating torch dtype, got {dtype}")
    return dtype


def _fp8_dtypes() -> set[torch.dtype]:
    dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
    e4m3fnuz = getattr(torch, "float8_e4m3fnuz", None)
    e5m2fnuz = getattr(torch, "float8_e5m2fnuz", None)
    if e4m3fnuz is not None:
        dtypes.add(e4m3fnuz)
    if e5m2fnuz is not None:
        dtypes.add(e5m2fnuz)
    return dtypes


def _check_fp8_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _fp8_dtypes():
        raise ValueError(f"{name} must be an FP8 torch dtype, got {dtype}")
    return dtype


def _check_cache_layout(value: str) -> CacheLayout:
    if value not in ("none", "dense", "paged"):
        raise ValueError(
            f"cache_layout must be 'none', 'dense', or 'paged', got {value!r}"
        )
    return value  # type: ignore[return-value]


def _check_page_table_indexing(value: str | None) -> PageTableIndexing | None:
    if value is None:
        return None
    if value not in ("identity", "random"):
        raise ValueError(f"indexing must be 'identity' or 'random', got {value!r}")
    return value  # type: ignore[return-value]


def _check_matches(
    *,
    parent_name: str,
    child_name: str,
    parent_value: object,
    child_value: object,
) -> None:
    if parent_value != child_value:
        raise ValueError(
            f"{child_name}={child_value!r} must match {parent_name}={parent_value!r}"
        )


def _metadata_config(
    metadata_input: MHARequestMetadataInput | None,
) -> MHARequestMetadataInputConfig:
    if metadata_input is None:
        raise ValueError("metadata_input must be initialized")
    return metadata_input.config


class _GenerateCacheFn(Protocol[_AttentionCacheT]):
    def __call__(
        self,
        *,
        metadata: MHARequestMetadataValues,
        value_seed: int,
        page_table_seed: int,
        device: torch.device,
    ) -> _AttentionCacheT: ...


@dataclass
class AttentionGeneratedValues(Generic[_AttentionCacheT]):
    """Tensor/cache values produced by the shared attention generation path."""

    q: torch.Tensor | None
    k: torch.Tensor | None
    v: torch.Tensor | None
    sinks: torch.Tensor | None
    cache: _AttentionCacheT | None


def _resolve_attention_seeds(
    *,
    seed: int | None,
    metadata_seed: int | None,
    value_seed: int | None,
) -> tuple[int, int]:
    if seed is None and (metadata_seed is None or value_seed is None):
        raise ValueError("provide either seed or both metadata_seed and value_seed")
    if seed is not None:
        if metadata_seed is None:
            metadata_seed = _child_seed(seed, 1)
        if value_seed is None:
            value_seed = _child_seed(seed, 2)
    assert metadata_seed is not None
    assert value_seed is not None
    return metadata_seed, value_seed


def _require_tensor(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} generation unexpectedly returned None")
    return values


def _generate_slot_mapping_tensor(
    *,
    num_rows: int,
    total_slots: int,
    negative_count: int,
    seed: int,
    device: torch.device,
    unique: bool = True,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Generate row-to-flat-cache-slot metadata."""

    return SlotMappingInput(
        SlotMappingInputConfig(
            num_rows=num_rows,
            total_slots=total_slots,
            negative_count=negative_count,
            unique=unique,
            dtype=dtype,
        )
    ).generate(seed=seed, device=device).slot_mapping


def _generate_valid_mask(
    *,
    num_rows: int,
    false_count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a boolean row-validity mask with exactly ``false_count`` false rows."""

    false_count = _check_nonnegative("false_count", false_count)
    if false_count > num_rows:
        raise ValueError("false_count must be <= num_rows")
    valid = torch.ones((num_rows,), dtype=torch.bool)
    if false_count:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(num_rows, generator=rng)
        valid[order[:false_count]] = False
    return valid.to(device)


def _generate_random_positions(
    *,
    num_tokens: int,
    max_position: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate per-token absolute positions inside a RoPE cache range."""

    if dtype not in (torch.int32, torch.int64):
        raise TypeError(f"position dtype must be int32 or int64, got {dtype}")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    positions = torch.randint(
        0,
        max_position,
        (num_tokens,),
        dtype=dtype,
        generator=rng,
    )
    return positions.to(device)


def _generate_token_to_request_indices(
    *,
    num_tokens: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate per-token request ids."""

    rng = torch.Generator(device="cpu").manual_seed(seed)
    reqs = torch.randint(
        0,
        batch_size,
        (num_tokens,),
        dtype=torch.int64,
        generator=rng,
    )
    return reqs.to(device)


def _compression_boundary_positions(
    *,
    max_seq_len: int,
    compress_ratio: int,
) -> torch.Tensor:
    return torch.arange(
        compress_ratio - 1,
        max_seq_len,
        compress_ratio,
        dtype=torch.int64,
    )


def _compression_non_boundary_positions(
    *,
    max_seq_len: int,
    compress_ratio: int,
) -> torch.Tensor:
    positions = torch.arange(max_seq_len, dtype=torch.int64)
    return positions[(positions + 1) % compress_ratio != 0]


def _generate_compression_positions(
    *,
    num_tokens: int,
    max_seq_len: int,
    compress_ratio: int,
    non_boundary_token_count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate positions with a configured mix of compression-boundary rows."""

    rng = torch.Generator(device="cpu").manual_seed(seed)
    positions = torch.empty((num_tokens,), dtype=torch.int64)
    boundary_count = num_tokens - non_boundary_token_count
    if boundary_count:
        boundary = _compression_boundary_positions(
            max_seq_len=max_seq_len,
            compress_ratio=compress_ratio,
        )
        choices = torch.randint(0, boundary.numel(), (boundary_count,), generator=rng)
        positions[:boundary_count] = boundary[choices]
    if non_boundary_token_count:
        non_boundary = _compression_non_boundary_positions(
            max_seq_len=max_seq_len,
            compress_ratio=compress_ratio,
        )
        choices = torch.randint(
            0,
            non_boundary.numel(),
            (non_boundary_token_count,),
            generator=rng,
        )
        positions[boundary_count:] = non_boundary[choices]
    if num_tokens:
        order = torch.randperm(num_tokens, generator=rng)
        positions = positions[order]
    return positions.to(device)


def _generate_random_uint8_tensor(
    *,
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate arbitrary byte storage for cache rows."""

    return torch.randint(
        0,
        256,
        shape,
        dtype=torch.uint8,
        device=device,
        generator=_rng_for_device(device, seed),
    )


def _generate_optional_tensor(
    *,
    tensor_input: TensorInput | None,
    shape: tuple[int, ...] | None,
    seed: int,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor_input is None:
        if shape is not None:
            raise ValueError("shape requires a tensor_input")
        return None
    if shape is None:
        raise ValueError("tensor_input requires a shape")
    tensor_input.shape = shape
    return tensor_input.generate(seed=seed, device=device).values


def _generate_packed_rows(
    *,
    tensor_input: TensorInput | None,
    shape: tuple[int, int],
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Generate a packed projection tensor with one row per token."""

    if tensor_input is None:
        raise ValueError(f"{name}_input must be initialized")
    tensor_input.shape = shape
    tensor_input.dtype = dtype
    return _require_tensor(
        tensor_input.generate(seed=seed, device=device).values,
        name,
    ).contiguous()


def attention_generate(
    *,
    metadata: MHARequestMetadataValues,
    value_seed: int,
    metadata_seed: int,
    device: torch.device,
    cache_layout: CacheLayout,
    q_input: TensorInput,
    q_shape: tuple[int, ...],
    k_input: TensorInput | None = None,
    k_shape: tuple[int, ...] | None = None,
    v_input: TensorInput | None = None,
    v_shape: tuple[int, ...] | None = None,
    sinks_input: TensorInput | None = None,
    sinks_shape: tuple[int, ...] | None = None,
    sinks_dtype: torch.dtype | None = None,
    generate_cache: _GenerateCacheFn[_AttentionCacheT] | None = None,
    cache_seed_index: int = 4,
    page_table_seed_index: int = 4,
    fixed_q_length_error: str | None = None,
) -> AttentionGeneratedValues[_AttentionCacheT]:
    """Generate common attention tensors and optional cache.

    Family-specific generators own the shape choices and cache type. This
    shared path owns applying those shapes to child tensor generators, deriving
    deterministic child seeds, and invoking cache generation.
    """

    if fixed_q_length_error is not None and cache_layout != "none":
        if len(set(metadata.new_q_lens_cpu)) != 1:
            raise ValueError(fixed_q_length_error)

    q_input.shape = q_shape
    q = q_input.generate(seed=_child_seed(value_seed, 1), device=device).values
    k = _generate_optional_tensor(
        tensor_input=k_input,
        shape=k_shape,
        seed=_child_seed(value_seed, 2),
        device=device,
    )
    v = _generate_optional_tensor(
        tensor_input=v_input,
        shape=v_shape,
        seed=_child_seed(value_seed, 3),
        device=device,
    )

    if sinks_input is not None:
        sinks_input.dtype = sinks_dtype
    sinks = _generate_optional_tensor(
        tensor_input=sinks_input,
        shape=sinks_shape,
        seed=_child_seed(value_seed, 4),
        device=device,
    )

    if cache_layout == "none":
        return AttentionGeneratedValues(q=q, k=k, v=v, sinks=sinks, cache=None)

    if generate_cache is None:
        raise ValueError("cache_layout != 'none' requires generate_cache")
    cache = generate_cache(
        metadata=metadata,
        value_seed=_child_seed(value_seed, cache_seed_index),
        page_table_seed=_child_seed(metadata_seed, page_table_seed_index),
        device=device,
    )
    return AttentionGeneratedValues(
        q=q,
        k=k,
        v=v,
        sinks=sinks,
        cache=cache,
    )

__all__ = [
    "AttentionGeneratedValues",
    "CacheLayout",
    "DeviceLike",
    "KVCacheInput",
    "KVCacheInputConfig",
    "KVCacheValues",
    "LengthMode",
    "Literal",
    "MHARequestMetadataInput",
    "MHARequestMetadataInputConfig",
    "MHARequestMetadataValues",
    "MLAKVCacheInput",
    "MLAKVCacheInputConfig",
    "MLAKVCacheValues",
    "NumericsInputGenerator",
    "PageTableIndexing",
    "PageTableInput",
    "PageTableInputConfig",
    "PageTableValues",
    "TensorInput",
    "_DEEPSEEK_V4_FP8_MAX",
    "_DEEPSEEK_V4_FP8_QUANT_BLOCK",
    "_DEEPSEEK_V4_HEAD_DIM",
    "_DEEPSEEK_V4_INDEXER_DIM",
    "_DEEPSEEK_V4_INDEXER_MXFP4_BLOCK_SIZE",
    "_DEEPSEEK_V4_INDEXER_MXFP4_HALF_BLOCK",
    "_DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES",
    "_DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES",
    "_DEEPSEEK_V4_NOPE_DIM",
    "_DEEPSEEK_V4_ROPE_DIM",
    "_DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT",
    "_DEEPSEEK_V4_SWA_SCALE_DIM",
    "_DEEPSEEK_V4_SWA_TOKEN_STRIDE",
    "_DSA_SPARSE_DECODE_BF16_BYTES",
    "_DSA_SPARSE_DECODE_FP8_E4M3_MAX",
    "_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK",
    "_DSA_SPARSE_DECODE_FP8_SCALE_BYTES",
    "_GDN_CHUNK_SIZE",
    "_align_up",
    "_check_cache_layout",
    "_check_float_dtype",
    "_check_matches",
    "_check_nonnegative",
    "_check_page_table_indexing",
    "_check_positive",
    "_check_power_of_2",
    "_child_seed",
    "_compression_boundary_positions",
    "_compression_non_boundary_positions",
    "_fp8_dtypes",
    "_generate_compression_positions",
    "_generate_packed_rows",
    "_generate_random_positions",
    "_generate_random_uint8_tensor",
    "_generate_slot_mapping_tensor",
    "_generate_token_to_request_indices",
    "_generate_valid_mask",
    "_metadata_config",
    "_require_tensor",
    "_resolve_attention_seeds",
    "_resolve_device",
    "_rng_for_device",
    "attention_generate",
    "build_rope_cos_sin_cache",
    "dataclass",
    "math",
    "torch",
]
