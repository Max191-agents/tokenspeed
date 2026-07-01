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
from typing import Generic, Protocol, TypeVar

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

__all__ = [
    "AttentionMergeStateInputConfig",
    "AttentionMergeStateInputs",
    "AttentionMergeStateInputValues",
    "attention_generate",
    "attention_merge_state_reference",
    "DSASparseDecodeKVPackInputConfig",
    "DSASparseDecodeKVPackInputs",
    "DSASparseDecodeKVPackInputValues",
    "DSATopKSlotInputConfig",
    "DSATopKSlotInputs",
    "DSATopKSlotInputValues",
    "dsa_sparse_decode_kv_pack_reference",
    "dsa_sparse_decode_row_bytes",
    "dsa_full_context_topk_to_global_slots_reference",
    "dsa_local_topk_to_global_slots_reference",
    "GDNQKVSplitInputConfig",
    "GDNQKVSplitInputs",
    "GDNQKVSplitInputValues",
    "GDNChunkPrefillInputConfig",
    "GDNChunkPrefillInputs",
    "GDNChunkPrefillInputValues",
    "GDNChunkPrefillReferenceValues",
    "gdn_chunk_prefill_reference",
    "gdn_qkv_split_reference",
    "DeepSeekV4CompressorStateInputConfig",
    "DeepSeekV4CompressorStateInputs",
    "DeepSeekV4CompressorStateInputValues",
    "deepseek_v4_save_compressor_state_reference",
    "DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig",
    "DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs",
    "DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues",
    "deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference",
    "DeepSeekV4IndexerMXFP4CacheWriteInputConfig",
    "DeepSeekV4IndexerMXFP4CacheWriteInputs",
    "DeepSeekV4IndexerMXFP4CacheWriteInputValues",
    "deepseek_v4_indexer_mxfp4_cache_write_reference",
    "DeepSeekV4IndexerMXFP4CacheGatherInputConfig",
    "DeepSeekV4IndexerMXFP4CacheGatherInputs",
    "DeepSeekV4IndexerMXFP4CacheGatherInputValues",
    "deepseek_v4_indexer_mxfp4_cache_gather_reference",
    "DeepSeekV4KCacheGatherInputConfig",
    "DeepSeekV4KCacheGatherInputs",
    "DeepSeekV4KCacheGatherInputValues",
    "deepseek_v4_dequantize_and_gather_k_cache_reference",
    "DeepSeekV4PagedIndexInputConfig",
    "DeepSeekV4PagedIndexInputs",
    "DeepSeekV4PagedIndexValues",
    "deepseek_v4_compressed_slot_mapping_reference",
    "deepseek_v4_compute_global_topk_indices_and_lens_reference",
    "deepseek_v4_decode_swa_indices_and_lens_reference",
    "deepseek_v4_indexer_decode_metadata_reference",
    "DeepSeekV4SparsePrefillIndexInputConfig",
    "DeepSeekV4SparsePrefillIndexInputs",
    "DeepSeekV4SparsePrefillIndexValues",
    "deepseek_v4_build_dense_prefill_local_compressed_indices_reference",
    "deepseek_v4_combine_dense_swa_indices_reference",
    "deepseek_v4_combine_topk_swa_indices_reference",
    "MLAKVPackQuantizeFP8InputConfig",
    "MLAKVPackQuantizeFP8Inputs",
    "MLAKVPackQuantizeFP8InputValues",
    "mla_kv_pack_quantize_fp8_reference",
    "MHAInputConfig",
    "MHAInputValues",
    "MHAInputs",
    "MLAInputConfig",
    "MLAInputValues",
    "MLAInputs",
    "PackedQKVComplexRotaryInputConfig",
    "PackedQKVComplexRotaryInputs",
    "PackedQKVComplexRotaryInputValues",
    "packed_qkv_complex_rotary_reference",
]

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


@dataclass
class AttentionMergeStateInputValues:
    """Generated values for merging two partial attention states."""

    out_a: torch.Tensor
    lse_a: torch.Tensor
    out_b: torch.Tensor
    lse_b: torch.Tensor
    lse_scale_log2: float


@dataclass
class AttentionMergeStateInputConfig:
    """Initialization parameters for attention merge-state inputs.

    The represented operation merges two partial attention outputs and their
    log-sum-exp values into one equivalent state. ``lse_scale_log2`` converts
    the LSE inputs into log2 space before the numerically stable merge.
    """

    # Required: total query rows being merged.
    total_q: int

    # Required: number of attention heads.
    num_heads: int

    # Required: per-head output dimension.
    head_dim: int

    # Optional: generated dtype for partial output tensors.
    dtype: torch.dtype = torch.bfloat16

    # Optional: scalar that maps input LSE values to log2 space. The default
    # corresponds to natural-log LSE inputs.
    lse_scale_log2: float = math.log2(math.e)

    # Optional: generated LSE values are sampled from [-lse_bound, lse_bound].
    lse_bound: float = 6.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class AttentionMergeStateInputs(NumericsInputGenerator):
    """Generator for attention merge-state inputs."""

    config: AttentionMergeStateInputConfig
    out_a_input: TensorInput | None
    out_b_input: TensorInput | None

    def __init__(self, config: AttentionMergeStateInputConfig) -> None:
        self.config = config
        self.out_a_input = None
        self.out_b_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.total_q = _check_nonnegative("total_q", self.config.total_q)
        self.config.num_heads = _check_positive("num_heads", self.config.num_heads)
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        if self.config.lse_scale_log2 <= 0.0:
            raise ValueError(
                f"lse_scale_log2 must be positive, got {self.config.lse_scale_log2}"
            )
        if self.config.lse_bound <= 0.0:
            raise ValueError(f"lse_bound must be positive, got {self.config.lse_bound}")
        output_shape = (
            self.config.total_q,
            self.config.num_heads,
            self.config.head_dim,
        )
        self.out_a_input = self.out_a_input or TensorInput(
            output_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.out_b_input = self.out_b_input or TensorInput(
            output_shape,
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> AttentionMergeStateInputValues:
        del metadata_seed
        self.__post_init__()
        if self.out_a_input is None or self.out_b_input is None:
            raise ValueError(
                "AttentionMergeStateInputs child generators must be initialized"
            )
        target_device = _resolve_device(self.config.device, device)
        lse_generator = torch.Generator(
            device="cuda" if target_device.type == "cuda" else "cpu"
        ).manual_seed(_child_seed(seed, 3))
        lse_shape = (self.config.total_q, self.config.num_heads)
        lse_a = (
            torch.rand(
                lse_shape,
                dtype=torch.float32,
                device=target_device,
                generator=lse_generator,
            )
            * (2.0 * self.config.lse_bound)
            - self.config.lse_bound
        )
        lse_b = (
            torch.rand(
                lse_shape,
                dtype=torch.float32,
                device=target_device,
                generator=lse_generator,
            )
            * (2.0 * self.config.lse_bound)
            - self.config.lse_bound
        )
        return AttentionMergeStateInputValues(
            out_a=_require_tensor(
                self.out_a_input.generate(
                    seed=_child_seed(seed, 1), device=device
                ).values,
                "out_a",
            ).contiguous(),
            lse_a=lse_a.contiguous(),
            out_b=_require_tensor(
                self.out_b_input.generate(
                    seed=_child_seed(seed, 2), device=device
                ).values,
                "out_b",
            ).contiguous(),
            lse_b=lse_b.contiguous(),
            lse_scale_log2=float(self.config.lse_scale_log2),
        )


def attention_merge_state_reference(
    values: AttentionMergeStateInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the merged partial attention output and LSE state."""

    if values.out_a.shape != values.out_b.shape:
        raise ValueError(
            f"out_a/out_b shape mismatch: {tuple(values.out_a.shape)} vs {tuple(values.out_b.shape)}"
        )
    if values.out_a.ndim != 3:
        raise ValueError(f"out tensors must be rank 3, got {values.out_a.ndim}")
    expected_lse_shape = values.out_a.shape[:2]
    if (
        values.lse_a.shape != expected_lse_shape
        or values.lse_b.shape != expected_lse_shape
    ):
        raise ValueError(
            "lse tensors must match out tensor leading dimensions; "
            f"expected {tuple(expected_lse_shape)}, "
            f"got {tuple(values.lse_a.shape)} and {tuple(values.lse_b.shape)}"
        )
    if values.lse_scale_log2 <= 0.0:
        raise ValueError(
            f"lse_scale_log2 must be positive, got {values.lse_scale_log2}"
        )
    lse_a_log2 = values.lse_a.to(torch.float32) * values.lse_scale_log2
    lse_b_log2 = values.lse_b.to(torch.float32) * values.lse_scale_log2
    lse_max_log2 = torch.maximum(lse_a_log2, lse_b_log2)
    weight_a = torch.exp2(lse_a_log2 - lse_max_log2)
    weight_b = torch.exp2(lse_b_log2 - lse_max_log2)
    denom = weight_a + weight_b
    out = (
        values.out_a.to(torch.float32) * weight_a[..., None]
        + values.out_b.to(torch.float32) * weight_b[..., None]
    ) / denom[..., None]
    lse = (lse_max_log2 + torch.log2(denom)) / values.lse_scale_log2
    return out.to(values.out_a.dtype), lse


@dataclass
class MLAKVPackQuantizeFP8InputValues:
    """Generated values for MLA K/V pack and FP8 quantization."""

    k_nope: torch.Tensor
    k_pe: torch.Tensor
    v: torch.Tensor
    k_scale_inv: float
    v_scale_inv: float
    fp8_dtype: torch.dtype


@dataclass
class MLAKVPackQuantizeFP8InputConfig:
    """Initialization parameters for MLA K/V pack and FP8 quantization.

    The represented operation forms the materialized K tensor used by MLA by
    broadcasting the RoPE key component across KV heads, concatenating it with
    the non-RoPE key component, and independently scaling/casting K and V into
    FP8 storage.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows.
    num_tokens: int

    # Required: number of KV heads in k_nope and v.
    num_kv_heads: int

    # Required: non-RoPE key width per KV head.
    qk_nope_head_dim: int

    # Required: RoPE key width shared across KV heads.
    qk_rope_head_dim: int

    # Required: value width per KV head.
    v_head_dim: int

    # Required: generated input dtype for k_nope, k_pe, and v.
    input_dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional generation and quantization configuration.
    # ------------------------------------------------------------------

    # Optional: shape rank for k_pe. Rank 3 generates [tokens, 1, rope_dim];
    # rank 2 generates [tokens, rope_dim].
    k_pe_rank: int = 3

    # Optional: inverse scale applied before casting packed K to FP8.
    k_scale_inv: float = 1.0

    # Optional: inverse scale applied before casting V to FP8.
    v_scale_inv: float = 1.0

    # Optional: generated output storage dtype.
    fp8_dtype: torch.dtype = torch.float8_e4m3fn

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MLAKVPackQuantizeFP8Inputs(NumericsInputGenerator):
    """Generator for the MLA K/V pack and FP8 quantization primitive."""

    config: MLAKVPackQuantizeFP8InputConfig
    k_nope_input: TensorInput | None
    k_pe_input: TensorInput | None
    v_input: TensorInput | None

    def __init__(self, config: MLAKVPackQuantizeFP8InputConfig) -> None:
        self.config = config
        self.k_nope_input = None
        self.k_pe_input = None
        self.v_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.qk_nope_head_dim = _check_positive(
            "qk_nope_head_dim", self.config.qk_nope_head_dim
        )
        self.config.qk_rope_head_dim = _check_positive(
            "qk_rope_head_dim", self.config.qk_rope_head_dim
        )
        self.config.v_head_dim = _check_positive("v_head_dim", self.config.v_head_dim)
        self.config.input_dtype = _check_float_dtype(
            "input_dtype", self.config.input_dtype
        )
        self.config.k_pe_rank = int(self.config.k_pe_rank)
        if self.config.k_pe_rank not in (2, 3):
            raise ValueError(f"k_pe_rank must be 2 or 3, got {self.config.k_pe_rank}")
        self.config.k_scale_inv = float(self.config.k_scale_inv)
        self.config.v_scale_inv = float(self.config.v_scale_inv)
        if self.config.k_scale_inv <= 0.0:
            raise ValueError(
                f"k_scale_inv must be positive, got {self.config.k_scale_inv}"
            )
        if self.config.v_scale_inv <= 0.0:
            raise ValueError(
                f"v_scale_inv must be positive, got {self.config.v_scale_inv}"
            )
        self.config.fp8_dtype = _check_fp8_dtype("fp8_dtype", self.config.fp8_dtype)
        self.k_nope_input = self.k_nope_input or TensorInput(
            (
                self.config.num_tokens,
                self.config.num_kv_heads,
                self.config.qk_nope_head_dim,
            ),
            self.config.input_dtype,
            device=self.config.device,
        )
        k_pe_shape = (
            (self.config.num_tokens, self.config.qk_rope_head_dim)
            if self.config.k_pe_rank == 2
            else (self.config.num_tokens, 1, self.config.qk_rope_head_dim)
        )
        self.k_pe_input = self.k_pe_input or TensorInput(
            k_pe_shape,
            self.config.input_dtype,
            device=self.config.device,
        )
        self.v_input = self.v_input or TensorInput(
            (
                self.config.num_tokens,
                self.config.num_kv_heads,
                self.config.v_head_dim,
            ),
            self.config.input_dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MLAKVPackQuantizeFP8InputValues:
        del metadata_seed
        self.__post_init__()
        if self.k_nope_input is None or self.k_pe_input is None or self.v_input is None:
            raise ValueError(
                "MLAKVPackQuantizeFP8Inputs child generators must be initialized"
            )
        target_device = _resolve_device(self.config.device, device)
        self.k_nope_input.shape = (
            self.config.num_tokens,
            self.config.num_kv_heads,
            self.config.qk_nope_head_dim,
        )
        self.k_nope_input.dtype = self.config.input_dtype
        self.k_pe_input.shape = (
            (self.config.num_tokens, self.config.qk_rope_head_dim)
            if self.config.k_pe_rank == 2
            else (self.config.num_tokens, 1, self.config.qk_rope_head_dim)
        )
        self.k_pe_input.dtype = self.config.input_dtype
        self.v_input.shape = (
            self.config.num_tokens,
            self.config.num_kv_heads,
            self.config.v_head_dim,
        )
        self.v_input.dtype = self.config.input_dtype
        return MLAKVPackQuantizeFP8InputValues(
            k_nope=_require_tensor(
                self.k_nope_input.generate(
                    seed=_child_seed(seed, 1), device=target_device
                ).values,
                "k_nope",
            ).contiguous(),
            k_pe=_require_tensor(
                self.k_pe_input.generate(
                    seed=_child_seed(seed, 2), device=target_device
                ).values,
                "k_pe",
            ).contiguous(),
            v=_require_tensor(
                self.v_input.generate(
                    seed=_child_seed(seed, 3), device=target_device
                ).values,
                "v",
            ).contiguous(),
            k_scale_inv=self.config.k_scale_inv,
            v_scale_inv=self.config.v_scale_inv,
            fp8_dtype=self.config.fp8_dtype,
        )


def mla_kv_pack_quantize_fp8_reference(
    values: MLAKVPackQuantizeFP8InputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed K and V tensors after inverse scaling and FP8 cast."""

    if values.k_nope.ndim != 3:
        raise ValueError(f"k_nope must be rank-3, got {values.k_nope.ndim}")
    if values.v.ndim != 3:
        raise ValueError(f"v must be rank-3, got {values.v.ndim}")
    if values.k_pe.ndim not in (2, 3):
        raise ValueError(f"k_pe must be rank-2 or rank-3, got {values.k_pe.ndim}")
    if values.k_nope.shape[:2] != values.v.shape[:2]:
        raise ValueError(
            "k_nope and v must have matching token/head dimensions; "
            f"k_nope={tuple(values.k_nope.shape)}, v={tuple(values.v.shape)}"
        )
    if values.k_pe.shape[0] != values.k_nope.shape[0]:
        raise ValueError(
            "k_pe token dimension must match k_nope; "
            f"k_pe={tuple(values.k_pe.shape)}, k_nope={tuple(values.k_nope.shape)}"
        )
    if values.k_pe.ndim == 3 and values.k_pe.shape[1] != 1:
        raise ValueError(
            "rank-3 k_pe must have singleton head dimension; "
            f"got {tuple(values.k_pe.shape)}"
        )
    if (
        values.k_nope.device != values.k_pe.device
        or values.k_nope.device != values.v.device
    ):
        raise ValueError("k_nope, k_pe, and v must be on the same device")
    for name, tensor in (
        ("k_nope", values.k_nope),
        ("k_pe", values.k_pe),
        ("v", values.v),
    ):
        _check_float_dtype(f"{name} dtype", tensor.dtype)
    if (
        values.k_nope.dtype != values.k_pe.dtype
        or values.k_nope.dtype != values.v.dtype
    ):
        raise ValueError("k_nope, k_pe, and v must have the same dtype")
    if values.k_scale_inv <= 0.0:
        raise ValueError(f"k_scale_inv must be positive, got {values.k_scale_inv}")
    if values.v_scale_inv <= 0.0:
        raise ValueError(f"v_scale_inv must be positive, got {values.v_scale_inv}")
    fp8_dtype = _check_fp8_dtype("fp8_dtype", values.fp8_dtype)

    k_pe_2d = values.k_pe.squeeze(1) if values.k_pe.ndim == 3 else values.k_pe
    k_pe_heads = k_pe_2d.unsqueeze(1).expand(-1, values.k_nope.shape[1], -1)
    k = torch.cat((values.k_nope, k_pe_heads), dim=-1)
    k_fp8 = (k.float() * values.k_scale_inv).to(fp8_dtype)
    v_fp8 = (values.v.float() * values.v_scale_inv).to(fp8_dtype)
    return k_fp8.contiguous(), v_fp8.contiguous()


@dataclass
class GDNQKVSplitInputValues:
    """Generated values for packed GDN QKV splitting."""

    mixed_qkv: torch.Tensor
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    head_q: int
    head_k: int
    head_v: int
    fuse_l2norm: bool
    l2norm_eps: float


@dataclass
class GDNQKVSplitInputConfig:
    """Initialization parameters for packed GDN QKV splitting.

    The represented operation splits a packed post-projection QKV row into
    contiguous query, key, and value tensors. When ``fuse_l2norm`` is true, Q
    and K are independently normalized over each head dimension before they are
    returned; V is always copied as-is.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of packed token rows.
    num_tokens: int

    # Required: number of Q heads in the packed Q segment.
    num_q_heads: int

    # Required: number of K heads in the packed K segment.
    num_k_heads: int

    # Required: number of V heads in the packed V segment.
    num_v_heads: int

    # Required: per-head Q dimension.
    head_q: int

    # Required: per-head K dimension.
    head_k: int

    # Required: per-head V dimension.
    head_v: int

    # Required: dtype for generated packed QKV values.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: normalize Q and K per head after splitting. This models the
    # fused fast path used by some GDN prefill implementations.
    fuse_l2norm: bool = False

    # Optional: epsilon used by the per-head L2 normalization.
    l2norm_eps: float = 1.0e-6

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class GDNQKVSplitInputs(NumericsInputGenerator):
    """Generator for packed GDN QKV split inputs."""

    config: GDNQKVSplitInputConfig
    mixed_qkv_input: TensorInput | None

    def __init__(self, config: GDNQKVSplitInputConfig) -> None:
        self.config = config
        self.mixed_qkv_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.mixed_qkv_input = self.mixed_qkv_input or TensorInput(
            (self.config.num_tokens, self._qkv_dim()),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> GDNQKVSplitInputValues:
        self.__post_init__()
        if self.mixed_qkv_input is None:
            raise ValueError("mixed_qkv_input must be initialized")
        target_device = _resolve_device(self.config.device, device)
        self.mixed_qkv_input.shape = (self.config.num_tokens, self._qkv_dim())
        self.mixed_qkv_input.dtype = self.config.dtype
        mixed_qkv = _require_tensor(
            self.mixed_qkv_input.generate(seed=seed, device=target_device).values,
            "mixed_qkv",
        )
        return GDNQKVSplitInputValues(
            mixed_qkv=mixed_qkv.contiguous(),
            num_q_heads=self.config.num_q_heads,
            num_k_heads=self.config.num_k_heads,
            num_v_heads=self.config.num_v_heads,
            head_q=self.config.head_q,
            head_k=self.config.head_k,
            head_v=self.config.head_v,
            fuse_l2norm=bool(self.config.fuse_l2norm),
            l2norm_eps=float(self.config.l2norm_eps),
        )

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.num_k_heads = _check_positive(
            "num_k_heads", self.config.num_k_heads
        )
        self.config.num_v_heads = _check_positive(
            "num_v_heads", self.config.num_v_heads
        )
        self.config.head_q = _check_positive("head_q", self.config.head_q)
        self.config.head_k = _check_positive("head_k", self.config.head_k)
        self.config.head_v = _check_positive("head_v", self.config.head_v)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.l2norm_eps = float(self.config.l2norm_eps)
        if self.config.l2norm_eps <= 0.0:
            raise ValueError(
                f"l2norm_eps must be positive, got {self.config.l2norm_eps}"
            )

    def _qkv_dim(self) -> int:
        return (
            self.config.num_q_heads * self.config.head_q
            + self.config.num_k_heads * self.config.head_k
            + self.config.num_v_heads * self.config.head_v
        )


def gdn_qkv_split_reference(
    values: GDNQKVSplitInputValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return split Q/K/V tensors, optionally with per-head Q/K L2 norm."""

    if values.mixed_qkv.ndim != 2:
        raise ValueError(f"mixed_qkv must be rank-2, got {values.mixed_qkv.ndim}")
    for name, value in (
        ("num_q_heads", values.num_q_heads),
        ("num_k_heads", values.num_k_heads),
        ("num_v_heads", values.num_v_heads),
        ("head_q", values.head_q),
        ("head_k", values.head_k),
        ("head_v", values.head_v),
    ):
        _check_positive(name, value)
    if values.l2norm_eps <= 0.0:
        raise ValueError(f"l2norm_eps must be positive, got {values.l2norm_eps}")

    q_dim = values.num_q_heads * values.head_q
    k_dim = values.num_k_heads * values.head_k
    v_dim = values.num_v_heads * values.head_v
    qkv_dim = q_dim + k_dim + v_dim
    if values.mixed_qkv.shape[1] != qkv_dim:
        raise ValueError(
            f"mixed_qkv last dimension must be {qkv_dim}, "
            f"got {values.mixed_qkv.shape[1]}"
        )

    tokens = values.mixed_qkv.shape[0]
    q = values.mixed_qkv[:, :q_dim].reshape(tokens, values.num_q_heads, values.head_q)
    k = values.mixed_qkv[:, q_dim : q_dim + k_dim].reshape(
        tokens,
        values.num_k_heads,
        values.head_k,
    )
    v = values.mixed_qkv[:, q_dim + k_dim :].reshape(
        tokens,
        values.num_v_heads,
        values.head_v,
    )
    if values.fuse_l2norm:
        q_norm = torch.sqrt(
            (q.float() * q.float()).sum(dim=-1, keepdim=True) + values.l2norm_eps
        )
        k_norm = torch.sqrt(
            (k.float() * k.float()).sum(dim=-1, keepdim=True) + values.l2norm_eps
        )
        q = (q.float() / q_norm).to(values.mixed_qkv.dtype)
        k = (k.float() / k_norm).to(values.mixed_qkv.dtype)
    return (
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
    )


@dataclass
class GDNChunkPrefillInputValues:
    """Generated values for gated-delta-rule chunked prefill.

    The operation consumes a flattened prompt stream partitioned by
    ``cu_seqlens``. Q and K are already L2-normalized, ``g`` is the log-space
    state decay, and ``beta`` gates the delta update written into the recurrent
    state.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens_cpu: list[int]
    scale: float | None
    output_h: bool


@dataclass
class GDNChunkPrefillReferenceValues:
    """Reference outputs for gated-delta-rule chunked prefill."""

    out: torch.Tensor
    final_state: torch.Tensor
    state_checkpoints: torch.Tensor | None = None
    checkpoint_cu_starts: torch.Tensor | None = None


@dataclass
class GDNChunkPrefillInputConfig:
    """Initialization parameters for GDN chunked prefill.

    This generator represents the prompt-side recurrent scan for Gated
    DeltaNet-style linear attention. It generates normalized Q/K rows, V rows,
    log-space decay gates, beta update gates, initial recurrent state, and
    cumulative sequence metadata for a flattened variable-length prompt stream.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of independent prompt sequences in cu_seqlens.
    batch_size: int

    # Required: total flattened prefill tokens across all sequences.
    total_tokens: int

    # Required: number of query/key heads. K shares the Q head count.
    num_q_heads: int

    # Required: number of value/state heads. Must be >= num_q_heads and an
    # integer multiple of num_q_heads for the generated GVA mapping.
    num_v_heads: int

    # Required: per-head Q/K/V dimension.
    head_dim: int

    # Required: generated dtype for Q, K, and V.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional request-layout configuration.
    # ------------------------------------------------------------------

    # Optional: split policy for total_tokens across batch_size sequences.
    length_mode: LengthMode = "ragged"

    # Optional: upper bound for generated tokens in one sequence.
    max_tokens_per_sequence: int | None = None

    # Optional: include a leading singleton batch axis accepted by TokenSpeed's
    # wrapper. The operation semantics remain a flattened cu_seqlens stream.
    include_batch_dim: bool = False

    # ------------------------------------------------------------------
    # Optional recurrence/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: explicit output scale. Defaults to 1 / sqrt(head_dim).
    scale: float | None = None

    # Optional: dtype for the generated initial recurrent state.
    initial_state_dtype: torch.dtype = torch.float32

    # Optional: factor applied to the generated initial state to keep the
    # recurrence in a numerically useful range.
    initial_state_scale: float = 0.05

    # Optional: lower bound for exp(g), where g is generated in log space.
    min_decay: float = 0.75

    # Optional: upper bound for exp(g), where g is generated in log space.
    max_decay: float = 0.99

    # Optional: epsilon used when normalizing generated Q and K rows.
    l2norm_eps: float = 1.0e-6

    # Optional: request intermediate recurrent-state checkpoints after each
    # full 64-token chunk in every sequence.
    output_h: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class GDNChunkPrefillInputs(NumericsInputGenerator):
    """Generator for gated-delta-rule chunked prefill inputs."""

    config: GDNChunkPrefillInputConfig
    q_input: TensorInput | None
    k_input: TensorInput | None
    v_input: TensorInput | None
    initial_state_input: TensorInput | None

    def __init__(self, config: GDNChunkPrefillInputConfig) -> None:
        self.config = config
        self.q_input = None
        self.k_input = None
        self.v_input = None
        self.initial_state_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.q_input = self.q_input or TensorInput(
            self._qk_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.k_input = self.k_input or TensorInput(
            self._qk_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.v_input = self.v_input or TensorInput(
            self._v_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.initial_state_input = self.initial_state_input or TensorInput(
            self._state_shape(),
            self.config.initial_state_dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> GDNChunkPrefillInputValues:
        self.__post_init__()
        if (
            self.q_input is None
            or self.k_input is None
            or self.v_input is None
            or self.initial_state_input is None
        ):
            raise ValueError(
                "GDNChunkPrefillInputs child generators must be initialized"
            )
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        metadata = self._generate_metadata(
            seed=metadata_seed,
            device=target_device,
        )

        self.q_input.shape = self._qk_shape()
        self.q_input.dtype = self.config.dtype
        self.k_input.shape = self._qk_shape()
        self.k_input.dtype = self.config.dtype
        self.v_input.shape = self._v_shape()
        self.v_input.dtype = self.config.dtype
        self.initial_state_input.shape = self._state_shape()
        self.initial_state_input.dtype = self.config.initial_state_dtype

        q = self._normalize_qk(
            _require_tensor(
                self.q_input.generate(
                    seed=_child_seed(value_seed, 1),
                    device=target_device,
                ).values,
                "q",
            )
        )
        k = self._normalize_qk(
            _require_tensor(
                self.k_input.generate(
                    seed=_child_seed(value_seed, 2),
                    device=target_device,
                ).values,
                "k",
            )
        )
        v = _require_tensor(
            self.v_input.generate(
                seed=_child_seed(value_seed, 3),
                device=target_device,
            ).values,
            "v",
        ).contiguous()
        initial_state = _require_tensor(
            self.initial_state_input.generate(
                seed=_child_seed(value_seed, 4),
                device=target_device,
            ).values,
            "initial_state",
        )
        initial_state = (
            initial_state.float() * float(self.config.initial_state_scale)
        ).to(self.config.initial_state_dtype)
        g, beta = self._generate_gates(
            seed=_child_seed(value_seed, 5),
            device=target_device,
        )

        if self.config.include_batch_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

        values = GDNChunkPrefillInputValues(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            initial_state=initial_state.contiguous(),
            cu_seqlens=metadata.cu_seqlens_q.contiguous(),
            seq_lens_cpu=list(metadata.new_q_lens_cpu),
            scale=self.config.scale,
            output_h=bool(self.config.output_h),
        )
        _validate_gdn_chunk_prefill_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.total_tokens = _check_positive(
            "total_tokens",
            self.config.total_tokens,
        )
        if self.config.total_tokens < self.config.batch_size:
            raise ValueError(
                "total_tokens must be >= batch_size so every sequence has at "
                f"least one token, got total_tokens={self.config.total_tokens}, "
                f"batch_size={self.config.batch_size}"
            )
        self.config.num_q_heads = _check_positive(
            "num_q_heads",
            self.config.num_q_heads,
        )
        self.config.num_v_heads = _check_positive(
            "num_v_heads",
            self.config.num_v_heads,
        )
        if self.config.num_v_heads < self.config.num_q_heads:
            raise ValueError(
                "num_v_heads must be >= num_q_heads for the supported GVA/equal-head "
                f"mapping, got num_v_heads={self.config.num_v_heads}, "
                f"num_q_heads={self.config.num_q_heads}"
            )
        if self.config.num_v_heads % self.config.num_q_heads != 0:
            raise ValueError(
                "num_v_heads must be an integer multiple of num_q_heads, got "
                f"num_v_heads={self.config.num_v_heads}, "
                f"num_q_heads={self.config.num_q_heads}"
            )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.initial_state_dtype = _check_float_dtype(
            "initial_state_dtype",
            self.config.initial_state_dtype,
        )
        if self.config.length_mode not in ("ragged", "regular", "fixed_per_request"):
            raise ValueError(
                "length_mode must be 'ragged', 'regular', or 'fixed_per_request', "
                f"got {self.config.length_mode!r}"
            )
        if self.config.max_tokens_per_sequence is not None:
            self.config.max_tokens_per_sequence = _check_positive(
                "max_tokens_per_sequence",
                self.config.max_tokens_per_sequence,
            )
        if self.config.scale is not None:
            self.config.scale = float(self.config.scale)
            if self.config.scale <= 0.0:
                raise ValueError(f"scale must be positive, got {self.config.scale}")
        self.config.initial_state_scale = float(self.config.initial_state_scale)
        if self.config.initial_state_scale < 0.0:
            raise ValueError(
                "initial_state_scale must be non-negative, got "
                f"{self.config.initial_state_scale}"
            )
        self.config.min_decay = float(self.config.min_decay)
        self.config.max_decay = float(self.config.max_decay)
        if not (0.0 < self.config.min_decay <= self.config.max_decay <= 1.0):
            raise ValueError(
                "decay bounds must satisfy 0 < min_decay <= max_decay <= 1, got "
                f"min_decay={self.config.min_decay}, "
                f"max_decay={self.config.max_decay}"
            )
        self.config.l2norm_eps = float(self.config.l2norm_eps)
        if self.config.l2norm_eps <= 0.0:
            raise ValueError(
                f"l2norm_eps must be positive, got {self.config.l2norm_eps}"
            )
        self.config.include_batch_dim = bool(self.config.include_batch_dim)
        self.config.output_h = bool(self.config.output_h)

    def _qk_shape(self) -> tuple[int, int, int]:
        return (
            self.config.total_tokens,
            self.config.num_q_heads,
            self.config.head_dim,
        )

    def _v_shape(self) -> tuple[int, int, int]:
        return (
            self.config.total_tokens,
            self.config.num_v_heads,
            self.config.head_dim,
        )

    def _state_shape(self) -> tuple[int, int, int, int]:
        return (
            self.config.batch_size,
            self.config.num_v_heads,
            self.config.head_dim,
            self.config.head_dim,
        )

    def _generate_metadata(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> MHARequestMetadataValues:
        metadata_input = MHARequestMetadataInput(
            MHARequestMetadataInputConfig(
                batch_size=self.config.batch_size,
                total_cached_tokens=0,
                total_new_q_tokens=self.config.total_tokens,
                new_q_length_mode=self.config.length_mode,
                max_new_q_tokens_per_request=self.config.max_tokens_per_sequence,
                cache_layout="none",
                device=self.config.device,
            )
        )
        return metadata_input.generate(seed=seed, device=device)

    def _normalize_qk(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_f32 = tensor.float()
        norm = torch.sqrt(
            (tensor_f32 * tensor_f32).sum(dim=-1, keepdim=True) + self.config.l2norm_eps
        )
        return (tensor_f32 / norm).to(tensor.dtype).contiguous()

    def _generate_gates(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        gate_shape = (
            self.config.total_tokens,
            self.config.num_v_heads,
        )
        alpha = torch.rand(
            gate_shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        alpha = alpha * (self.config.max_decay - self.config.min_decay)
        alpha = alpha + self.config.min_decay
        g = torch.log(alpha)
        beta = torch.rand(
            gate_shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        beta = beta * 0.9 + 0.05
        return g.contiguous(), beta.contiguous()


def _gdn_as_3d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} rank-4 form must have leading batch size 1")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be rank-3 or rank-4, got {tensor.ndim}")
    return tensor.contiguous()


def _gdn_gate_as_2d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} rank-3 form must have leading batch size 1")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2 or rank-3, got {tensor.ndim}")
    return tensor.contiguous()


def _validate_gdn_chunk_prefill_values(
    values: GDNChunkPrefillInputValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _gdn_as_3d("q", values.q)
    k = _gdn_as_3d("k", values.k)
    v = _gdn_as_3d("v", values.v)
    g = _gdn_gate_as_2d("g", values.g)
    beta = _gdn_gate_as_2d("beta", values.beta)

    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have matching shapes, got {q.shape} and {k.shape}"
        )
    if q.shape[0] != v.shape[0]:
        raise ValueError("q/k and v must have matching token dimensions")
    if q.shape[-1] != v.shape[-1]:
        raise ValueError("q/k and v must have matching head_dim")
    num_q_heads = q.shape[1]
    num_v_heads = v.shape[1]
    if num_v_heads < num_q_heads:
        raise ValueError("num_v_heads must be >= num_q_heads")
    if num_v_heads % num_q_heads != 0:
        raise ValueError("num_v_heads must be an integer multiple of num_q_heads")
    if g.shape != (v.shape[0], num_v_heads):
        raise ValueError(
            f"g must have shape {(v.shape[0], num_v_heads)}, got {tuple(g.shape)}"
        )
    if beta.shape != g.shape:
        raise ValueError(
            f"beta must have shape {tuple(g.shape)}, got {tuple(beta.shape)}"
        )
    if values.initial_state.shape != (
        values.cu_seqlens.numel() - 1,
        num_v_heads,
        q.shape[-1],
        v.shape[-1],
    ):
        raise ValueError(
            "initial_state must have shape "
            f"{(values.cu_seqlens.numel() - 1, num_v_heads, q.shape[-1], v.shape[-1])}, "
            f"got {tuple(values.initial_state.shape)}"
        )
    if values.cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be rank-1, got {values.cu_seqlens.ndim}")
    if values.cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"cu_seqlens must use an integer dtype, got {values.cu_seqlens.dtype}"
        )
    cu = values.cu_seqlens.detach().cpu().to(torch.int64)
    if cu.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if int(cu[0].item()) != 0:
        raise ValueError("cu_seqlens must start at zero")
    if int(cu[-1].item()) != q.shape[0]:
        raise ValueError(
            f"cu_seqlens must end at total token count {q.shape[0]}, "
            f"got {int(cu[-1].item())}"
        )
    if bool((cu[1:] < cu[:-1]).any().item()):
        raise ValueError("cu_seqlens must be monotonically nondecreasing")
    seq_lens = (cu[1:] - cu[:-1]).tolist()
    if values.seq_lens_cpu != [int(length) for length in seq_lens]:
        raise ValueError("seq_lens_cpu must match cu_seqlens differences")
    if any(length <= 0 for length in values.seq_lens_cpu):
        raise ValueError(
            "all generated GDN prefill sequences must have at least one token"
        )
    if values.scale is not None and values.scale <= 0.0:
        raise ValueError(f"scale must be positive, got {values.scale}")
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", values.initial_state),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if values.initial_state.device != q.device:
        raise ValueError("initial_state must be on the same device as q")
    if (
        k.device != q.device
        or v.device != q.device
        or g.device != q.device
        or beta.device != q.device
    ):
        raise ValueError("q, k, v, g, and beta must share a device")
    if values.cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be on the same device as q")
    return q, k, v, g, beta


def gdn_chunk_prefill_reference(
    values: GDNChunkPrefillInputValues,
) -> GDNChunkPrefillReferenceValues:
    """Return a sequential reference for GDN chunked prefill.

    The reference follows the recurrent delta-rule form for each sequence and
    value head:

    ```text
    state = exp(g_t) * state
    delta = beta_t * (v_t - k_t @ state)
    state = state + outer(k_t, delta)
    out_t = scale * (q_t @ state)
    ```
    """

    q, k, v, g, beta = _validate_gdn_chunk_prefill_values(values)
    scale = values.scale if values.scale is not None else q.shape[-1] ** -0.5
    state = values.initial_state.float().clone()
    out = torch.empty_like(v)
    heads_per_q = v.shape[1] // q.shape[1]
    cu = values.cu_seqlens.detach().cpu().to(torch.int64).tolist()

    checkpoint_counts: list[int] = []
    checkpoints: list[torch.Tensor] = []
    for seq_idx, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=True)):
        start = int(start)
        end = int(end)
        checkpoint_count = 0
        for local_idx, token_idx in enumerate(range(start, end), start=1):
            for v_head in range(v.shape[1]):
                q_head = v_head // heads_per_q
                state_view = state[seq_idx, v_head]
                state_view.mul_(torch.exp(g[token_idx, v_head].float()))
                k_vec = k[token_idx, q_head].float()
                v_vec = v[token_idx, v_head].float()
                read = torch.matmul(k_vec, state_view)
                delta = (v_vec - read) * beta[token_idx, v_head].float()
                state_view.add_(torch.outer(k_vec, delta))
                out[token_idx, v_head] = (
                    torch.matmul(q[token_idx, q_head].float(), state_view) * scale
                ).to(out.dtype)
            if values.output_h and local_idx % _GDN_CHUNK_SIZE == 0:
                checkpoints.append(state[seq_idx].clone())
                checkpoint_count += 1
        checkpoint_counts.append(checkpoint_count)

    if values.q.ndim == 4:
        out = out.unsqueeze(0)

    if not values.output_h:
        return GDNChunkPrefillReferenceValues(
            out=out.contiguous(),
            final_state=state.contiguous(),
        )

    if checkpoints:
        state_checkpoints = torch.stack(checkpoints, dim=0).contiguous()
    else:
        state_checkpoints = torch.empty(
            (0, v.shape[1], q.shape[-1], v.shape[-1]),
            dtype=torch.float32,
            device=q.device,
        )
    checkpoint_cu = [0]
    for count in checkpoint_counts:
        checkpoint_cu.append(checkpoint_cu[-1] + count)
    checkpoint_cu_starts = torch.tensor(
        checkpoint_cu,
        dtype=torch.int64,
        device=q.device,
    )
    return GDNChunkPrefillReferenceValues(
        out=out.contiguous(),
        final_state=state.contiguous(),
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
    )


@dataclass
class PackedQKVComplexRotaryInputValues:
    """Generated values for packed QKV complex rotary splitting."""

    qkv: torch.Tensor
    freqs_cis: torch.Tensor
    num_heads: int
    head_dim: int
    copy_v: bool


@dataclass
class PackedQKVComplexRotaryInputConfig:
    """Initialization parameters for packed QKV complex RoPE.

    The represented operation splits a packed QKV tensor into equal-width Q, K,
    and V segments. Q and K are interpreted as pairs of real channels and are
    multiplied by unit complex rotary frequencies. V is not rotated.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of packed token rows.
    num_tokens: int

    # Required: number of Q/K/V heads. This utility expects all three packed
    # segments to use the same head count.
    num_heads: int

    # Required: per-head Q/K/V dimension. Must be even for complex pairs.
    head_dim: int

    # Required: dtype for generated packed QKV values.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: materialize V as an output tensor instead of returning a view of
    # the packed V segment. The mathematical values are the same either way.
    copy_v: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class PackedQKVComplexRotaryInputs(NumericsInputGenerator):
    """Generator for packed QKV complex rotary inputs."""

    config: PackedQKVComplexRotaryInputConfig
    qkv_input: TensorInput | None

    def __init__(self, config: PackedQKVComplexRotaryInputConfig) -> None:
        self.config = config
        self.qkv_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.qkv_input = self.qkv_input or TensorInput(
            (self.config.num_tokens, self._packed_dim()),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> PackedQKVComplexRotaryInputValues:
        self.__post_init__()
        if self.qkv_input is None:
            raise ValueError("qkv_input must be initialized")
        target_device = _resolve_device(self.config.device, device)
        self.qkv_input.shape = (self.config.num_tokens, self._packed_dim())
        self.qkv_input.dtype = self.config.dtype
        qkv = _require_tensor(
            self.qkv_input.generate(
                seed=_child_seed(seed, 1), device=target_device
            ).values,
            "qkv",
        )
        freqs_cis = self._generate_freqs_cis(
            seed=_child_seed(seed, 2),
            device=target_device,
        )
        return PackedQKVComplexRotaryInputValues(
            qkv=qkv.contiguous(),
            freqs_cis=freqs_cis.contiguous(),
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            copy_v=bool(self.config.copy_v),
        )

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_heads = _check_positive("num_heads", self.config.num_heads)
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        if self.config.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {self.config.head_dim}")
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)

    def _packed_dim(self) -> int:
        return 3 * self.config.num_heads * self.config.head_dim

    def _generate_freqs_cis(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        angles = (
            torch.rand(
                (self.config.num_tokens, self.config.head_dim // 2),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * (2.0 * math.pi)
            - math.pi
        )
        return torch.complex(torch.cos(angles), torch.sin(angles))


def _apply_complex_rotary(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    real = freqs_cis.real[:, None, :].float()
    imag = freqs_cis.imag[:, None, :].float()
    out = torch.empty_like(x.float())
    out[..., 0::2] = x_even * real - x_odd * imag
    out[..., 1::2] = x_odd * real + x_even * imag
    return out.to(x.dtype)


def packed_qkv_complex_rotary_reference(
    values: PackedQKVComplexRotaryInputValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q/K after complex RoPE and unrotated V."""

    if values.qkv.ndim != 2:
        raise ValueError(f"qkv must be rank-2, got {values.qkv.ndim}")
    _check_positive("num_heads", values.num_heads)
    _check_positive("head_dim", values.head_dim)
    if values.head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {values.head_dim}")
    if not values.freqs_cis.is_complex():
        raise TypeError("freqs_cis must be a complex tensor")
    if values.qkv.device != values.freqs_cis.device:
        raise ValueError("qkv and freqs_cis must be on the same device")
    total_tokens = values.qkv.shape[0]
    q_size = values.num_heads * values.head_dim
    kv_size = q_size
    packed_dim = q_size + 2 * kv_size
    if values.qkv.shape[1] != packed_dim:
        raise ValueError(
            f"qkv last dimension must be {packed_dim}, got {values.qkv.shape[1]}"
        )
    expected_freqs_shape = (total_tokens, values.head_dim // 2)
    if values.freqs_cis.shape != expected_freqs_shape:
        raise ValueError(
            f"freqs_cis must have shape {expected_freqs_shape}, "
            f"got {tuple(values.freqs_cis.shape)}"
        )

    q = values.qkv[:, :q_size].reshape(total_tokens, values.num_heads, values.head_dim)
    k = values.qkv[:, q_size : q_size + kv_size].reshape(
        total_tokens,
        values.num_heads,
        values.head_dim,
    )
    v = values.qkv[:, q_size + kv_size :].reshape(
        total_tokens,
        values.num_heads,
        values.head_dim,
    )
    q_out = _apply_complex_rotary(q, values.freqs_cis).contiguous()
    k_out = _apply_complex_rotary(k, values.freqs_cis).contiguous()
    v_out = v.clone().contiguous() if values.copy_v else v
    return q_out, k_out, v_out


def dsa_sparse_decode_row_bytes(nope_dim: int, rope_dim: int) -> int:
    """Return bytes per packed sparse-decode KV row."""

    nope_dim = int(nope_dim)
    rope_dim = int(rope_dim)
    if nope_dim % _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK != 0:
        raise ValueError(
            "DSA sparse decode NoPE dim must be divisible by "
            f"{_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK}, got {nope_dim}"
        )
    return (
        nope_dim
        + nope_dim
        // _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK
        * _DSA_SPARSE_DECODE_FP8_SCALE_BYTES
        + rope_dim * _DSA_SPARSE_DECODE_BF16_BYTES
    )


@dataclass
class DSASparseDecodeKVPackInputValues:
    """Generated values for DSA sparse decode KV row packing.

    The represented output row layout is:

    ```text
    [NoPE FP8 E4M3 bytes][FP32 scale bytes per 128 NoPE channels][RoPE BF16 bytes]
    ```
    """

    out: torch.Tensor
    loc: torch.Tensor
    cache_k_nope: torch.Tensor
    cache_k_rope: torch.Tensor


@dataclass
class DSASparseDecodeKVPackInputConfig:
    """Initialization parameters for DSA sparse decode KV row packing.

    The operation packs per-token sparse-decode K-cache rows into physical
    cache slots. Non-RoPE key channels are dynamically scaled per 128-channel
    block and stored as FP8 E4M3 bytes. RoPE key channels are copied as raw
    BF16 bytes after the FP32 scale sidecar.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows to pack.
    num_tokens: int

    # Required: number of physical output rows available for sparse decode.
    num_slots: int

    # Required: non-RoPE key width. Must be a power of two and divisible by 128.
    nope_dim: int

    # Required: RoPE key width. Must be a power of two.
    rope_dim: int

    # ------------------------------------------------------------------
    # Optional generation configuration.
    # ------------------------------------------------------------------

    # Optional: include the singleton KV-head axis accepted by some callers,
    # generating [tokens, 1, dim] source tensors instead of [tokens, dim].
    include_head_axis: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DSASparseDecodeKVPackInputs(NumericsInputGenerator):
    """Generator for DSA sparse decode KV row packing."""

    config: DSASparseDecodeKVPackInputConfig
    cache_k_nope_input: TensorInput | None
    cache_k_rope_input: TensorInput | None

    def __init__(self, config: DSASparseDecodeKVPackInputConfig) -> None:
        self.config = config
        self.cache_k_nope_input = None
        self.cache_k_rope_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_slots = _check_positive("num_slots", self.config.num_slots)
        if self.config.num_tokens > self.config.num_slots:
            raise ValueError(
                "num_tokens must be <= num_slots so generated slot locations "
                f"are unique, got num_tokens={self.config.num_tokens}, "
                f"num_slots={self.config.num_slots}"
            )
        self.config.nope_dim = _check_power_of_2("nope_dim", self.config.nope_dim)
        if self.config.nope_dim % _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK != 0:
            raise ValueError(
                "nope_dim must be divisible by "
                f"{_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK}, got {self.config.nope_dim}"
            )
        self.config.rope_dim = _check_power_of_2("rope_dim", self.config.rope_dim)
        self.config.include_head_axis = bool(self.config.include_head_axis)
        self.cache_k_nope_input = self.cache_k_nope_input or TensorInput(
            self._source_shape(self.config.nope_dim),
            torch.bfloat16,
            device=self.config.device,
        )
        self.cache_k_rope_input = self.cache_k_rope_input or TensorInput(
            self._source_shape(self.config.rope_dim),
            torch.bfloat16,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DSASparseDecodeKVPackInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.cache_k_nope_input is None or self.cache_k_rope_input is None:
            raise ValueError(
                "DSASparseDecodeKVPackInputs child generators must be initialized"
            )

        self.cache_k_nope_input.shape = self._source_shape(self.config.nope_dim)
        self.cache_k_nope_input.dtype = torch.bfloat16
        self.cache_k_nope_input.device = self.config.device
        self.cache_k_rope_input.shape = self._source_shape(self.config.rope_dim)
        self.cache_k_rope_input.dtype = torch.bfloat16
        self.cache_k_rope_input.device = self.config.device

        cache_k_nope = self.cache_k_nope_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        ).values
        cache_k_rope = self.cache_k_rope_input.generate(
            seed=_child_seed(seed, 2),
            device=target_device,
        ).values
        if cache_k_nope is None or cache_k_rope is None:
            raise ValueError("DSA sparse decode K source tensors must be generated")

        out = self._generate_out(seed=_child_seed(seed, 3), device=target_device)
        loc_seed = seed if metadata_seed is None else metadata_seed
        loc = self._generate_loc(seed=_child_seed(loc_seed, 4), device=target_device)
        values = DSASparseDecodeKVPackInputValues(
            out=out,
            loc=loc,
            cache_k_nope=cache_k_nope,
            cache_k_rope=cache_k_rope,
        )
        _validate_dsa_sparse_decode_kv_pack_values(values)
        return values

    def _source_shape(self, dim: int) -> tuple[int, ...]:
        if self.config.include_head_axis:
            return (self.config.num_tokens, 1, int(dim))
        return (self.config.num_tokens, int(dim))

    def _generate_out(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        row_bytes = dsa_sparse_decode_row_bytes(
            self.config.nope_dim,
            self.config.rope_dim,
        )
        return torch.randint(
            0,
            256,
            (self.config.num_slots, row_bytes),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def _generate_loc(self, *, seed: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randperm(
            self.config.num_slots,
            dtype=torch.int64,
            generator=generator,
        )[: self.config.num_tokens].to(device)


def _dsa_sparse_decode_source_2d(
    name: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[1] != 1:
            raise ValueError(f"{name} rank-3 form must have one KV head")
        tensor = tensor.squeeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2 or rank-3, got {tensor.ndim}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16, got {tensor.dtype}")
    return tensor.contiguous()


def _validate_dsa_sparse_decode_kv_pack_values(
    values: DSASparseDecodeKVPackInputValues,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if values.out.dtype != torch.uint8:
        raise TypeError(f"out must be uint8, got {values.out.dtype}")
    if values.out.ndim != 2:
        raise ValueError(f"out must be rank-2, got {values.out.ndim}")
    if values.loc.ndim != 1:
        raise ValueError(f"loc must be rank-1, got {values.loc.ndim}")
    if values.loc.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"loc must use an integer dtype, got {values.loc.dtype}")
    if values.loc.device != values.out.device:
        raise ValueError("loc and out must be on the same device")

    cache_k_nope = _dsa_sparse_decode_source_2d(
        "cache_k_nope",
        values.cache_k_nope,
    )
    cache_k_rope = _dsa_sparse_decode_source_2d(
        "cache_k_rope",
        values.cache_k_rope,
    )
    if (
        cache_k_nope.device != values.out.device
        or cache_k_rope.device != values.out.device
    ):
        raise ValueError("cache_k_nope, cache_k_rope, loc, and out must share device")
    if cache_k_nope.shape[0] != values.loc.numel():
        raise ValueError("cache_k_nope token dimension must match loc length")
    if cache_k_rope.shape[0] != values.loc.numel():
        raise ValueError("cache_k_rope token dimension must match loc length")

    nope_dim = _check_power_of_2("nope_dim", int(cache_k_nope.shape[1]))
    if nope_dim % _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK != 0:
        raise ValueError(
            "nope_dim must be divisible by "
            f"{_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK}, got {nope_dim}"
        )
    rope_dim = _check_power_of_2("rope_dim", int(cache_k_rope.shape[1]))
    expected_row_bytes = dsa_sparse_decode_row_bytes(nope_dim, rope_dim)
    if values.out.shape[1] != expected_row_bytes:
        raise ValueError(
            f"out row width must be {expected_row_bytes}, got {values.out.shape[1]}"
        )

    if values.loc.numel() > 0:
        loc = values.loc.to(torch.int64)
        if int(loc.min().item()) < 0 or int(loc.max().item()) >= values.out.shape[0]:
            raise ValueError("loc entries must be valid output row indices")
        unique_count = int(torch.unique(loc).numel())
        if unique_count != values.loc.numel():
            raise ValueError("loc entries must be unique to avoid output row races")

    return cache_k_nope, cache_k_rope, nope_dim, rope_dim


def dsa_sparse_decode_kv_pack_reference(
    values: DSASparseDecodeKVPackInputValues,
) -> torch.Tensor:
    """Return sparse-decode KV rows packed into the generated output buffer."""

    cache_k_nope, cache_k_rope, nope_dim, rope_dim = (
        _validate_dsa_sparse_decode_kv_pack_values(values)
    )
    out = values.out.clone()
    if values.loc.numel() == 0:
        return out

    num_nope_blocks = nope_dim // _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK
    nope_blocks = cache_k_nope.float().reshape(
        values.loc.numel(),
        num_nope_blocks,
        _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK,
    )
    scales = nope_blocks.abs().amax(dim=-1) / _DSA_SPARSE_DECODE_FP8_E4M3_MAX
    scales = torch.clamp(scales, min=1.0e-26)
    fp8_nope = torch.clamp(
        nope_blocks / scales.unsqueeze(-1),
        -_DSA_SPARSE_DECODE_FP8_E4M3_MAX,
        _DSA_SPARSE_DECODE_FP8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    fp8_nope_bytes = (
        fp8_nope.contiguous()
        .view(torch.uint8)
        .reshape(
            values.loc.numel(),
            nope_dim,
        )
    )
    scale_bytes = (
        scales.float()
        .contiguous()
        .view(torch.uint8)
        .reshape(
            values.loc.numel(),
            num_nope_blocks * _DSA_SPARSE_DECODE_FP8_SCALE_BYTES,
        )
    )
    rope_bytes = (
        cache_k_rope.contiguous()
        .view(torch.uint8)
        .reshape(
            values.loc.numel(),
            rope_dim * _DSA_SPARSE_DECODE_BF16_BYTES,
        )
    )

    loc = values.loc.to(torch.int64)
    scale_offset = nope_dim
    rope_offset = scale_offset + scale_bytes.shape[1]
    out[loc, :nope_dim] = fp8_nope_bytes
    out[loc, scale_offset:rope_offset] = scale_bytes
    out[loc, rope_offset:] = rope_bytes
    return out


@dataclass
class DSATopKSlotInputValues:
    """Generated values for DSA sparse top-k slot conversion."""

    local_topk_offsets: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    block_table_cpu: list[list[int]]
    block_table_values: PageTableValues
    block_size: int
    topk: int


@dataclass
class DSATopKSlotInputConfig:
    """Initialization parameters for DSA sparse top-k slot conversion.

    The represented metadata operation maps per-token local offsets through a
    token-row page table. Local offsets address logical positions in the
    per-token context; global slots address physical cache rows.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows.
    num_tokens: int

    # Required: number of top-k offsets per token row.
    topk: int

    # Required: physical page size used for local-offset to page/offset mapping.
    block_size: int

    # Required: page-table width per token row.
    max_pages_per_token: int

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: upper bound for generated per-token sequence lengths. Defaults
    # to max_pages_per_token * block_size.
    max_seq_len: int | None = None

    # Optional: physical-page assignment policy for the token-row block table.
    indexing: PageTableIndexing = "random"

    # Optional: nested page-table config. If supplied, its batch size must match
    # num_tokens; its width is raised as needed during generation.
    page_table_input: PageTableInputConfig | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DSATopKSlotInputs(NumericsInputGenerator):
    """Generator for DSA sparse top-k slot-conversion metadata."""

    config: DSATopKSlotInputConfig
    page_table_input: PageTableInput | None

    def __init__(self, config: DSATopKSlotInputConfig) -> None:
        self.config = config
        self.page_table_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.page_table_input = self.page_table_input or PageTableInput(
            self.config.page_table_input or self._make_page_table_config()
        )
        self._verify_page_table_config_matches_parent()
        self.config.page_table_input = self.page_table_input.config

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> DSATopKSlotInputValues:
        self.__post_init__()
        if self.page_table_input is None:
            raise ValueError("page_table_input must be initialized")
        target_device = _resolve_device(self.config.device, device)
        self.page_table_input.config.batch_size = max(1, self.config.num_tokens)
        self.page_table_input.config.max_pages_per_request = max(
            self.page_table_input.config.max_pages_per_request,
            self.config.max_pages_per_token,
        )
        self.page_table_input.config.indexing = self.config.indexing
        page_table_values = self.page_table_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        )
        max_context_len = page_table_values.page_table.shape[1] * self.config.block_size
        max_seq_len = (
            max_context_len
            if self.config.max_seq_len is None
            else min(self.config.max_seq_len, max_context_len)
        )
        seq_lens_cpu = self._generate_seq_lens(
            seed=_child_seed(seed, 2),
            max_seq_len=max_seq_len,
        )
        local_topk_cpu = self._generate_local_topk_offsets(
            seed=_child_seed(seed, 3),
            seq_lens=seq_lens_cpu,
        )

        return DSATopKSlotInputValues(
            local_topk_offsets=local_topk_cpu.to(target_device),
            seq_lens=torch.tensor(
                seq_lens_cpu, dtype=torch.int32, device=target_device
            ),
            block_table=page_table_values.page_table,
            block_table_cpu=page_table_values.page_table_cpu,
            block_table_values=page_table_values,
            block_size=self.config.block_size,
            topk=self.config.topk,
        )

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.topk = _check_positive("topk", self.config.topk)
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.max_pages_per_token = _check_positive(
            "max_pages_per_token", self.config.max_pages_per_token
        )
        if self.config.max_seq_len is not None:
            self.config.max_seq_len = _check_positive(
                "max_seq_len", self.config.max_seq_len
            )
        self.config.indexing = _check_page_table_indexing(self.config.indexing)

    def _make_page_table_config(self) -> PageTableInputConfig:
        return PageTableInputConfig(
            batch_size=max(1, self.config.num_tokens),
            max_pages_per_request=self.config.max_pages_per_token,
            indexing=self.config.indexing,
            device=self.config.device,
        )

    def _verify_page_table_config_matches_parent(self) -> None:
        if self.page_table_input is None:
            raise ValueError("page_table_input must be initialized")
        _check_matches(
            parent_name="DSATopKSlotInputConfig.num_tokens",
            child_name="page_table_input.batch_size",
            parent_value=max(1, self.config.num_tokens),
            child_value=self.page_table_input.config.batch_size,
        )

    def _generate_seq_lens(self, *, seed: int, max_seq_len: int) -> list[int]:
        if self.config.num_tokens == 0:
            return []
        rng = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(
            1,
            max_seq_len + 1,
            (self.config.num_tokens,),
            dtype=torch.int32,
            generator=rng,
        ).tolist()

    def _generate_local_topk_offsets(
        self,
        *,
        seed: int,
        seq_lens: list[int],
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        local_topk = torch.full(
            (self.config.num_tokens, self.config.topk),
            -1,
            dtype=torch.int32,
        )
        for token_idx, seq_len in enumerate(seq_lens):
            valid_count = int(
                torch.randint(
                    1,
                    min(int(seq_len), self.config.topk) + 1,
                    (1,),
                    generator=rng,
                ).item()
            )
            selected = torch.randperm(int(seq_len), generator=rng)[:valid_count].to(
                torch.int32
            )
            local_topk[token_idx, :valid_count] = selected
        return local_topk


def _validate_dsa_topk_slot_values(values: DSATopKSlotInputValues) -> None:
    if values.local_topk_offsets.dtype != torch.int32:
        raise TypeError(
            f"local_topk_offsets must be int32, got {values.local_topk_offsets.dtype}"
        )
    if values.seq_lens.dtype != torch.int32:
        raise TypeError(f"seq_lens must be int32, got {values.seq_lens.dtype}")
    if values.block_table.dtype != torch.int32:
        raise TypeError(f"block_table must be int32, got {values.block_table.dtype}")
    if values.local_topk_offsets.ndim != 2:
        raise ValueError("local_topk_offsets must be rank-2")
    num_tokens, topk = values.local_topk_offsets.shape
    if values.topk != topk:
        raise ValueError("topk must match local_topk_offsets width")
    if values.seq_lens.shape != (num_tokens,):
        raise ValueError("seq_lens must have shape [num_tokens]")
    if values.block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    if values.block_table.shape[0] < num_tokens:
        raise ValueError("block_table must have at least one row per token")
    if values.block_table.shape[1] <= 0:
        raise ValueError("block_table must have at least one page column")
    if values.block_size <= 0:
        raise ValueError("block_size must be positive")
    if values.topk <= 0:
        raise ValueError("topk must be positive")
    if (
        values.local_topk_offsets.device != values.seq_lens.device
        or values.local_topk_offsets.device != values.block_table.device
    ):
        raise ValueError(
            "local_topk_offsets, seq_lens, and block_table must share device"
        )


def dsa_local_topk_to_global_slots_reference(
    values: DSATopKSlotInputValues,
    *,
    use_seq_lens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global cache slots for generated DSA local top-k offsets."""

    _validate_dsa_topk_slot_values(values)
    local = values.local_topk_offsets
    device = local.device
    num_tokens, topk = local.shape
    global_slots = torch.full_like(local, -1)
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    block_table_cpu = values.block_table.cpu()
    seq_lens_cpu = values.seq_lens.cpu().tolist()
    max_context_len = values.block_table.shape[1] * values.block_size
    for token_idx in range(num_tokens):
        seq_len = int(seq_lens_cpu[token_idx]) if use_seq_lens else max_context_len
        count = 0
        for slot in range(topk):
            local_idx = int(local[token_idx, slot].item())
            if local_idx < 0 or local_idx >= seq_len:
                continue
            block_idx = local_idx // values.block_size
            if block_idx < 0 or block_idx >= values.block_table.shape[1]:
                continue
            block_offset = local_idx % values.block_size
            page = int(block_table_cpu[token_idx, block_idx].item())
            global_slots[token_idx, slot] = page * values.block_size + block_offset
            count += 1
        lens[token_idx] = count
    return global_slots, lens


def dsa_full_context_topk_to_global_slots_reference(
    values: DSATopKSlotInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global slots for the first top-k positions of each token context."""

    _validate_dsa_topk_slot_values(values)
    device = values.seq_lens.device
    num_tokens = int(values.seq_lens.numel())
    topk = int(values.topk)
    global_slots = torch.full(
        (num_tokens, topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    block_table_cpu = values.block_table.cpu()
    seq_lens_cpu = values.seq_lens.cpu().tolist()
    max_context_len = values.block_table.shape[1] * values.block_size
    for token_idx, seq_len in enumerate(seq_lens_cpu):
        capped_seq_len = min(int(seq_len), max_context_len)
        lens[token_idx] = min(capped_seq_len, topk)
        for offset in range(topk):
            block_idx = offset // values.block_size
            if offset >= int(seq_len) or block_idx >= values.block_table.shape[1]:
                continue
            block_offset = offset % values.block_size
            page = int(block_table_cpu[token_idx, block_idx].item())
            global_slots[token_idx, offset] = page * values.block_size + block_offset
    return global_slots, lens


@dataclass
class DeepSeekV4CompressorStateInputValues:
    """Generated values for saving DeepSeek V4 compressor state rows."""

    kv: torch.Tensor
    score: torch.Tensor
    ape: torch.Tensor
    state_cache: torch.Tensor
    slot_mapping: torch.Tensor
    positions: torch.Tensor
    block_size: int
    compress_ratio: int


@dataclass
class DeepSeekV4CompressorStateInputConfig:
    """Initialization parameters for DeepSeek V4 compressor-state writes.

    The represented operation writes one generated token row into a paged
    compressor-state cache. The first half of the cache row stores the K/V
    state. The second half stores score state plus the absolute-position
    embedding row selected by ``position % compress_ratio``.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows that may write compressor state.
    num_tokens: int

    # Required: width of each generated K/V and score state row.
    state_width: int

    # Required: number of physical state-cache pages.
    num_cache_blocks: int

    # Required: number of state rows per physical cache page.
    block_size: int

    # Required: compression ratio controlling APE row selection.
    compress_ratio: int

    # Required: generated dtype for K/V and score state rows.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: number of generated token rows with slot_mapping == -1. These
    # rows model padded or otherwise skipped tokens and leave cache rows intact.
    invalid_token_count: int = 0

    # Optional: first absolute token position used for generated positions.
    position_start: int = 0

    # Optional: factor applied to generated score rows.
    score_scale: float = 0.1

    # Optional: factor applied to generated APE rows.
    ape_scale: float = 0.01

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DeepSeekV4CompressorStateInputs(NumericsInputGenerator):
    """Generator for DeepSeek V4 compressor-state save inputs."""

    config: DeepSeekV4CompressorStateInputConfig
    kv_input: TensorInput | None
    score_input: TensorInput | None
    ape_input: TensorInput | None
    state_cache_input: TensorInput | None

    def __init__(self, config: DeepSeekV4CompressorStateInputConfig) -> None:
        self.config = config
        self.kv_input = None
        self.score_input = None
        self.ape_input = None
        self.state_cache_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.kv_input = self.kv_input or TensorInput(
            self._state_row_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.score_input = self.score_input or TensorInput(
            self._state_row_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.ape_input = self.ape_input or TensorInput(
            self._ape_shape(),
            torch.float32,
            device=self.config.device,
        )
        self.state_cache_input = self.state_cache_input or TensorInput(
            self._state_cache_shape(),
            torch.float32,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4CompressorStateInputValues:
        self.__post_init__()
        if (
            self.kv_input is None
            or self.score_input is None
            or self.ape_input is None
            or self.state_cache_input is None
        ):
            raise ValueError(
                "DeepSeekV4CompressorStateInputs child generators must be initialized"
            )
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        self.kv_input.shape = self._state_row_shape()
        self.kv_input.dtype = self.config.dtype
        self.score_input.shape = self._state_row_shape()
        self.score_input.dtype = self.config.dtype
        self.ape_input.shape = self._ape_shape()
        self.ape_input.dtype = torch.float32
        self.state_cache_input.shape = self._state_cache_shape()
        self.state_cache_input.dtype = torch.float32

        kv = _require_tensor(
            self.kv_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "kv",
        )
        score = _require_tensor(
            self.score_input.generate(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ).values,
            "score",
        )
        ape = _require_tensor(
            self.ape_input.generate(
                seed=_child_seed(value_seed, 3),
                device=target_device,
            ).values,
            "ape",
        )
        state_cache = _require_tensor(
            self.state_cache_input.generate(
                seed=_child_seed(value_seed, 4),
                device=target_device,
            ).values,
            "state_cache",
        )
        values = DeepSeekV4CompressorStateInputValues(
            kv=kv.contiguous(),
            score=(score.float() * self.config.score_scale)
            .to(score.dtype)
            .contiguous(),
            ape=(ape.float() * self.config.ape_scale).contiguous(),
            state_cache=state_cache.contiguous(),
            slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            positions=self._generate_positions(device=target_device),
            block_size=self.config.block_size,
            compress_ratio=self.config.compress_ratio,
        )
        _validate_deepseek_v4_compressor_state_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens",
            self.config.num_tokens,
        )
        self.config.state_width = _check_positive(
            "state_width",
            self.config.state_width,
        )
        self.config.num_cache_blocks = _check_positive(
            "num_cache_blocks",
            self.config.num_cache_blocks,
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.compress_ratio = _check_positive(
            "compress_ratio",
            self.config.compress_ratio,
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.invalid_token_count = _check_nonnegative(
            "invalid_token_count",
            self.config.invalid_token_count,
        )
        if self.config.invalid_token_count > self.config.num_tokens:
            raise ValueError("invalid_token_count must be <= num_tokens")
        valid_count = self.config.num_tokens - self.config.invalid_token_count
        total_slots = self.config.num_cache_blocks * self.config.block_size
        if valid_count > total_slots:
            raise ValueError(
                "valid generated tokens must fit in the state cache, got "
                f"valid_count={valid_count}, total_slots={total_slots}"
            )
        self.config.position_start = _check_nonnegative(
            "position_start",
            self.config.position_start,
        )
        self.config.score_scale = float(self.config.score_scale)
        if self.config.score_scale < 0.0:
            raise ValueError(
                f"score_scale must be non-negative, got {self.config.score_scale}"
            )
        self.config.ape_scale = float(self.config.ape_scale)
        if self.config.ape_scale < 0.0:
            raise ValueError(
                f"ape_scale must be non-negative, got {self.config.ape_scale}"
            )

    def _state_row_shape(self) -> tuple[int, int]:
        return (self.config.num_tokens, self.config.state_width)

    def _ape_shape(self) -> tuple[int, int]:
        return (self.config.compress_ratio, self.config.state_width)

    def _state_cache_shape(self) -> tuple[int, int, int]:
        return (
            self.config.num_cache_blocks,
            self.config.block_size,
            self.config.state_width * 2,
        )

    def _generate_positions(self, *, device: torch.device) -> torch.Tensor:
        return torch.arange(
            self.config.position_start,
            self.config.position_start + self.config.num_tokens,
            dtype=torch.int64,
            device=device,
        )

    def _generate_slot_mapping(
        self, *, seed: int, device: torch.device
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        slot_mapping = torch.full(
            (self.config.num_tokens,),
            -1,
            dtype=torch.int64,
        )
        if self.config.num_tokens == 0:
            return slot_mapping.to(device)
        order = torch.randperm(self.config.num_tokens, generator=rng)
        valid_count = self.config.num_tokens - self.config.invalid_token_count
        if valid_count:
            total_slots = self.config.num_cache_blocks * self.config.block_size
            slots = torch.randperm(total_slots, generator=rng)[:valid_count]
            slot_mapping[order[:valid_count]] = slots.to(torch.int64)
        return slot_mapping.to(device)


def _validate_deepseek_v4_compressor_state_values(
    values: DeepSeekV4CompressorStateInputValues,
) -> None:
    if values.kv.shape != values.score.shape:
        raise ValueError(
            f"kv and score shapes must match, got {values.kv.shape} vs {values.score.shape}"
        )
    if values.kv.ndim != 2:
        raise ValueError(f"kv/score must be rank-2, got {values.kv.ndim}")
    if values.state_cache.ndim != 3:
        raise ValueError(f"state_cache must be rank-3, got {values.state_cache.ndim}")
    block_size = _check_positive("block_size", values.block_size)
    compress_ratio = _check_positive("compress_ratio", values.compress_ratio)
    if values.state_cache.shape[1] != block_size:
        raise ValueError(
            f"block_size={block_size} must match state_cache.shape[1]="
            f"{values.state_cache.shape[1]}"
        )
    state_width = values.kv.shape[1]
    if values.state_cache.shape[-1] != state_width * 2:
        raise ValueError(
            f"state_cache last dimension must be {state_width * 2}, "
            f"got {values.state_cache.shape[-1]}"
        )
    if values.ape.shape != (compress_ratio, state_width):
        raise ValueError(
            f"ape must have shape {(compress_ratio, state_width)}, "
            f"got {tuple(values.ape.shape)}"
        )
    if values.slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be rank-1")
    if values.positions.ndim != 1:
        raise ValueError("positions must be rank-1")
    if values.slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"slot_mapping must be integer, got {values.slot_mapping.dtype}"
        )
    if values.positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be integer, got {values.positions.dtype}")
    if values.slot_mapping.numel() != values.kv.shape[0]:
        raise ValueError("slot_mapping length must match kv token dimension")
    if values.positions.numel() < values.kv.shape[0]:
        raise ValueError("positions length must be at least the kv token dimension")
    for name, tensor in (
        ("kv", values.kv),
        ("score", values.score),
        ("ape", values.ape),
        ("state_cache", values.state_cache),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
        if tensor.device != values.kv.device:
            raise ValueError("all generated compressor-state tensors must share device")
    if (
        values.slot_mapping.device != values.kv.device
        or values.positions.device != values.kv.device
    ):
        raise ValueError("slot_mapping and positions must share the tensor device")

    slots = values.slot_mapping.to(torch.int64)
    valid = slots >= 0
    if not bool(valid.any().item()):
        return
    total_slots = values.state_cache.shape[0] * block_size
    valid_slots = slots[valid]
    if int(valid_slots.max().item()) >= total_slots:
        raise ValueError(
            f"slot_mapping entries must be < {total_slots}, got "
            f"{int(valid_slots.max().item())}"
        )
    unique_count = int(torch.unique(valid_slots).numel())
    if unique_count != int(valid_slots.numel()):
        raise ValueError("valid slot_mapping entries must be unique")


def _deepseek_v4_compressor_state_ape_row(
    *,
    ape: torch.Tensor,
    ape_row: int,
    state_width: int,
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio == 4 and state_width == ape.shape[1] and state_width % 2 == 0:
        head_dim = state_width // 2
        flat = ape.reshape(-1)
        first = flat[ape_row * head_dim : (ape_row + 1) * head_dim]
        second_start = (ape_row + compress_ratio) * head_dim
        second = flat[second_start : second_start + head_dim]
        return torch.cat((first, second), dim=0)
    return ape[ape_row]


def deepseek_v4_save_compressor_state_reference(
    values: DeepSeekV4CompressorStateInputValues,
) -> torch.Tensor:
    """Return state cache after applying DeepSeek V4 compressor-state writes."""

    _validate_deepseek_v4_compressor_state_values(values)
    out = values.state_cache.clone()
    state_width = values.kv.shape[1]
    slots = values.slot_mapping.to(torch.int64)
    positions = values.positions.to(torch.int64)
    for token_idx in range(values.kv.shape[0]):
        slot = int(slots[token_idx].item())
        if slot < 0:
            continue
        block_idx = slot // values.block_size
        pos_in_block = slot % values.block_size
        ape_row = int(positions[token_idx].item()) % values.compress_ratio
        ape = _deepseek_v4_compressor_state_ape_row(
            ape=values.ape.float(),
            ape_row=ape_row,
            state_width=state_width,
            compress_ratio=values.compress_ratio,
        )
        out[block_idx, pos_in_block, :state_width] = values.kv[token_idx].float()
        out[block_idx, pos_in_block, state_width:] = (
            values.score[token_idx].float() + ape
        )
    return out.contiguous()


@dataclass
class DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues:
    """Generated values for DeepSeek V4 indexer-Q RoPE/Hadamard/MXFP4."""

    index_q: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    weights: torch.Tensor
    softmax_scale: float
    head_scale: float


@dataclass
class DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig:
    """Initialization parameters for DeepSeek V4 indexer-Q transforms.

    The represented operation transforms 128-channel per-head indexer Q rows:
    RoPE is applied to the final 64 channels, the rotated row is projected by a
    normalized Hadamard sign matrix, and each 32-channel block is quantized into
    packed MXFP4 bytes with one UE8M0 scale byte.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of packed query-token rows.
    num_tokens: int

    # Required: number of indexer heads per query token.
    num_heads: int

    # Required: generated dtype for indexer Q rows.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: number of RoPE cache rows. Generated positions are in
    # [0, max_position), so this controls the generated absolute-position range.
    max_position: int = 1024

    # Optional: RoPE frequency base used to build the generated cos/sin cache.
    rope_base: float = 10000.0

    # Optional: dtype for generated position metadata.
    position_dtype: torch.dtype = torch.int64

    # Optional: generated dtype for per-token/per-head weights.
    weights_dtype: torch.dtype = torch.float32

    # Optional: scale applied to generated indexer Q rows before RoPE.
    value_scale: float = 1.0

    # Optional: scale applied to generated weights.
    weights_value_scale: float = 1.0

    # Optional: scalar multiplier applied to weights in the represented op.
    softmax_scale: float = 1.0

    # Optional: scalar multiplier applied to weights in the represented op.
    head_scale: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(NumericsInputGenerator):
    """Generator for DeepSeek V4 indexer-Q RoPE/Hadamard/MXFP4 inputs."""

    config: DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig
    index_q_input: TensorInput | None
    weights_input: TensorInput | None

    def __init__(
        self,
        config: DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig,
    ) -> None:
        self.config = config
        self.index_q_input = None
        self.weights_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.index_q_input = self.index_q_input or TensorInput(
            self._index_q_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.weights_input = self.weights_input or TensorInput(
            self._weights_shape(),
            self.config.weights_dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues:
        self.__post_init__()
        if self.index_q_input is None:
            raise ValueError("index_q_input must be initialized")
        if self.weights_input is None:
            raise ValueError("weights_input must be initialized")
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        self.index_q_input.shape = self._index_q_shape()
        self.index_q_input.dtype = self.config.dtype
        self.weights_input.shape = self._weights_shape()
        self.weights_input.dtype = self.config.weights_dtype

        index_q = _require_tensor(
            self.index_q_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "index_q",
        )
        weights = _require_tensor(
            self.weights_input.generate(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ).values,
            "weights",
        )
        values = DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues(
            index_q=(index_q.float() * self.config.value_scale)
            .to(index_q.dtype)
            .contiguous(),
            positions=self._generate_positions(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            cos_sin_cache=build_rope_cos_sin_cache(
                rotary_dim=_DEEPSEEK_V4_ROPE_DIM,
                max_position=self.config.max_position,
                base=self.config.rope_base,
                device=target_device,
            ),
            weights=(weights.float() * self.config.weights_value_scale)
            .to(weights.dtype)
            .contiguous(),
            softmax_scale=self.config.softmax_scale,
            head_scale=self.config.head_scale,
        )
        _validate_deepseek_v4_indexer_q_rope_hadamard_mxfp4_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens",
            self.config.num_tokens,
        )
        self.config.num_heads = _check_positive("num_heads", self.config.num_heads)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.max_position = _check_positive(
            "max_position",
            self.config.max_position,
        )
        self.config.weights_dtype = _check_float_dtype(
            "weights_dtype",
            self.config.weights_dtype,
        )
        if self.config.position_dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "position_dtype must be torch.int32 or torch.int64, got "
                f"{self.config.position_dtype}"
            )
        self.config.rope_base = float(self.config.rope_base)
        if self.config.rope_base <= 0.0 or not math.isfinite(self.config.rope_base):
            raise ValueError(f"rope_base must be positive, got {self.config.rope_base}")
        self.config.value_scale = float(self.config.value_scale)
        if self.config.value_scale < 0.0 or not math.isfinite(self.config.value_scale):
            raise ValueError(
                f"value_scale must be finite and non-negative, got {self.config.value_scale}"
            )
        self.config.weights_value_scale = float(self.config.weights_value_scale)
        if self.config.weights_value_scale < 0.0 or not math.isfinite(
            self.config.weights_value_scale
        ):
            raise ValueError(
                "weights_value_scale must be finite and non-negative, got "
                f"{self.config.weights_value_scale}"
            )
        self.config.softmax_scale = float(self.config.softmax_scale)
        self.config.head_scale = float(self.config.head_scale)
        if not math.isfinite(self.config.softmax_scale):
            raise ValueError(
                f"softmax_scale must be finite, got {self.config.softmax_scale}"
            )
        if not math.isfinite(self.config.head_scale):
            raise ValueError(f"head_scale must be finite, got {self.config.head_scale}")

    def _index_q_shape(self) -> tuple[int, int, int]:
        return (
            self.config.num_tokens,
            self.config.num_heads,
            _DEEPSEEK_V4_INDEXER_DIM,
        )

    def _weights_shape(self) -> tuple[int, int]:
        return (self.config.num_tokens, self.config.num_heads)

    def _generate_positions(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        positions = torch.randint(
            0,
            self.config.max_position,
            (self.config.num_tokens,),
            dtype=self.config.position_dtype,
            generator=rng,
        )
        return positions.to(device)


def _validate_deepseek_v4_indexer_q_rope_hadamard_mxfp4_values(
    values: DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues,
) -> None:
    if values.index_q.ndim != 3:
        raise ValueError(f"index_q must be rank-3, got {values.index_q.ndim}")
    if values.index_q.shape[-1] != _DEEPSEEK_V4_INDEXER_DIM:
        raise ValueError(
            f"index_q width must be {_DEEPSEEK_V4_INDEXER_DIM}, "
            f"got {values.index_q.shape[-1]}"
        )
    if not values.index_q.is_floating_point():
        raise TypeError(f"index_q must be floating point, got {values.index_q.dtype}")
    if values.positions.ndim != 1:
        raise ValueError("positions must be rank-1")
    if values.positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be integer, got {values.positions.dtype}")
    if values.positions.numel() != values.index_q.shape[0]:
        raise ValueError("positions length must match index_q token dimension")
    if values.cos_sin_cache.ndim != 2:
        raise ValueError("cos_sin_cache must be rank-2")
    if values.cos_sin_cache.shape[1] != _DEEPSEEK_V4_ROPE_DIM:
        raise ValueError(
            f"cos_sin_cache width must be {_DEEPSEEK_V4_ROPE_DIM}, "
            f"got {values.cos_sin_cache.shape[1]}"
        )
    if values.weights.shape != values.index_q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(values.index_q.shape[:2])}, "
            f"got {tuple(values.weights.shape)}"
        )
    if not values.weights.is_floating_point():
        raise TypeError(f"weights must be floating point, got {values.weights.dtype}")
    if (
        values.positions.device != values.index_q.device
        or values.cos_sin_cache.device != values.index_q.device
        or values.weights.device != values.index_q.device
    ):
        raise ValueError(
            "index_q, positions, cos_sin_cache, and weights must share device"
        )
    if values.positions.numel():
        min_pos = int(values.positions.min().item())
        max_pos = int(values.positions.max().item())
        if min_pos < 0 or max_pos >= values.cos_sin_cache.shape[0]:
            raise ValueError(
                "positions must be within cos_sin_cache rows, got range "
                f"[{min_pos}, {max_pos}] for cache length {values.cos_sin_cache.shape[0]}"
            )
    if not math.isfinite(values.softmax_scale):
        raise ValueError(f"softmax_scale must be finite, got {values.softmax_scale}")
    if not math.isfinite(values.head_scale):
        raise ValueError(f"head_scale must be finite, got {values.head_scale}")


def _deepseek_v4_apply_indexer_q_rope(
    values: DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues,
) -> torch.Tensor:
    q = values.index_q.float()
    nope_dim = _DEEPSEEK_V4_INDEXER_DIM - _DEEPSEEK_V4_ROPE_DIM
    half_rope = _DEEPSEEK_V4_ROPE_DIM // 2
    rope = q[..., nope_dim:]
    rope_even = rope[..., 0::2]
    rope_odd = rope[..., 1::2]
    cos_sin = values.cos_sin_cache[values.positions.to(torch.int64)]
    cos_v = cos_sin[:, None, :half_rope].float()
    sin_v = cos_sin[:, None, half_rope:].float()
    rotated_rope = torch.empty_like(rope)
    rotated_rope[..., 0::2] = rope_even * cos_v - rope_odd * sin_v
    rotated_rope[..., 1::2] = rope_odd * cos_v + rope_even * sin_v
    rotated = torch.cat((q[..., :nope_dim], rotated_rope), dim=-1)
    return rotated.to(torch.bfloat16).to(torch.float32)


def _deepseek_v4_indexer_hadamard_signs(device: torch.device) -> torch.Tensor:
    indices = torch.arange(
        _DEEPSEEK_V4_INDEXER_DIM,
        dtype=torch.int32,
        device=device,
    )
    bits = indices[:, None] & indices[None, :]
    parity = bits ^ (bits >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    parity = parity & 1
    return torch.where(
        parity == 0,
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(-1.0, dtype=torch.float32, device=device),
    )


def _deepseek_v4_indexer_q_hadamard(
    rotated: torch.Tensor,
) -> torch.Tensor:
    signs = _deepseek_v4_indexer_hadamard_signs(rotated.device)
    flat = rotated.reshape(-1, _DEEPSEEK_V4_INDEXER_DIM)
    projected = flat @ signs
    projected = projected * (_DEEPSEEK_V4_INDEXER_DIM**-0.5)
    projected = projected.reshape_as(rotated)
    return projected.to(torch.bfloat16).to(torch.float32)


def deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference(
    values: DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return packed MXFP4 indexer Q and scaled weights for DeepSeek V4."""

    _validate_deepseek_v4_indexer_q_rope_hadamard_mxfp4_values(values)
    num_tokens, num_heads, _ = values.index_q.shape
    q_packed = torch.empty(
        (
            num_tokens,
            num_heads,
            _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
        ),
        dtype=torch.uint8,
        device=values.index_q.device,
    )
    q_scale_bytes = torch.empty(
        (
            num_tokens,
            num_heads,
            _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES,
        ),
        dtype=torch.uint8,
        device=values.index_q.device,
    )
    weights_out = (
        values.weights.float() * values.softmax_scale * values.head_scale
    ).contiguous()
    if num_tokens == 0:
        return (
            q_packed,
            q_scale_bytes.view(torch.int32).squeeze(-1).contiguous(),
        ), weights_out

    rotated = _deepseek_v4_apply_indexer_q_rope(values)
    hadamard = _deepseek_v4_indexer_q_hadamard(rotated)
    packed_rows = q_packed.reshape(-1, _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES)
    scale_rows = q_scale_bytes.reshape(-1, _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES)
    for row_idx, row in enumerate(hadamard.reshape(-1, _DEEPSEEK_V4_INDEXER_DIM)):
        packed, scales = _deepseek_v4_indexer_mxfp4_row_reference(row)
        packed_rows[row_idx] = packed
        scale_rows[row_idx] = scales
    return (
        q_packed.contiguous(),
        q_scale_bytes.view(torch.int32).squeeze(-1).contiguous(),
    ), weights_out


@dataclass
class DeepSeekV4IndexerMXFP4CacheWriteInputValues:
    """Generated values for writing DeepSeek V4 indexer K rows to MXFP4 cache."""

    index_k: torch.Tensor
    cache_2d: torch.Tensor
    slot_mapping: torch.Tensor
    valid: torch.Tensor
    block_size: int


@dataclass
class DeepSeekV4IndexerMXFP4CacheWriteInputConfig:
    """Initialization parameters for DeepSeek V4 indexer MXFP4 cache writes.

    The represented operation quantizes 128-channel indexer K rows into MXFP4
    storage and writes them into a paged byte cache. Cache rows are selected by
    ``slot_mapping``. A row is written only when ``valid`` is true and its slot
    is non-negative.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of generated indexer K rows.
    num_rows: int

    # Required: number of physical MXFP4 cache pages.
    num_cache_blocks: int

    # Required: number of indexer rows in each physical cache page.
    block_size: int

    # Required: generated dtype for indexer K rows.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: rows with slot_mapping == -1. These rows are skipped.
    negative_slot_count: int = 0

    # Optional: rows with valid == False. These rows are skipped even when they
    # have a non-negative slot.
    masked_row_count: int = 0

    # Optional: factor applied to generated indexer K rows to keep MXFP4 scales
    # in a representative range.
    value_scale: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DeepSeekV4IndexerMXFP4CacheWriteInputs(NumericsInputGenerator):
    """Generator for DeepSeek V4 indexer MXFP4 cache-write inputs."""

    config: DeepSeekV4IndexerMXFP4CacheWriteInputConfig
    index_k_input: TensorInput | None

    def __init__(self, config: DeepSeekV4IndexerMXFP4CacheWriteInputConfig) -> None:
        self.config = config
        self.index_k_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.index_k_input = self.index_k_input or TensorInput(
            self._index_k_shape(),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4IndexerMXFP4CacheWriteInputValues:
        self.__post_init__()
        if self.index_k_input is None:
            raise ValueError("index_k_input must be initialized")
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        self.index_k_input.shape = self._index_k_shape()
        self.index_k_input.dtype = self.config.dtype
        index_k = _require_tensor(
            self.index_k_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "index_k",
        )
        values = DeepSeekV4IndexerMXFP4CacheWriteInputValues(
            index_k=(index_k.float() * self.config.value_scale)
            .to(index_k.dtype)
            .contiguous(),
            cache_2d=self._generate_cache(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ),
            slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            valid=self._generate_valid(
                seed=_child_seed(metadata_seed, 2),
                device=target_device,
            ),
            block_size=self.config.block_size,
        )
        _validate_deepseek_v4_indexer_mxfp4_cache_write_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.num_cache_blocks = _check_positive(
            "num_cache_blocks",
            self.config.num_cache_blocks,
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.negative_slot_count = _check_nonnegative(
            "negative_slot_count",
            self.config.negative_slot_count,
        )
        self.config.masked_row_count = _check_nonnegative(
            "masked_row_count",
            self.config.masked_row_count,
        )
        if self.config.negative_slot_count > self.config.num_rows:
            raise ValueError("negative_slot_count must be <= num_rows")
        if self.config.masked_row_count > self.config.num_rows:
            raise ValueError("masked_row_count must be <= num_rows")
        non_negative_count = self.config.num_rows - self.config.negative_slot_count
        total_slots = self.config.num_cache_blocks * self.config.block_size
        if non_negative_count > total_slots:
            raise ValueError(
                "generated non-negative slots must fit in the MXFP4 cache, got "
                f"non_negative_count={non_negative_count}, total_slots={total_slots}"
            )
        self.config.value_scale = float(self.config.value_scale)
        if self.config.value_scale < 0.0:
            raise ValueError(
                f"value_scale must be non-negative, got {self.config.value_scale}"
            )

    def _index_k_shape(self) -> tuple[int, int]:
        return (self.config.num_rows, _DEEPSEEK_V4_INDEXER_DIM)

    def _cache_row_bytes(self) -> int:
        return self.config.block_size * (
            _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
            + _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES
        )

    def _generate_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        return torch.randint(
            0,
            256,
            (self.config.num_cache_blocks, self._cache_row_bytes()),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def _generate_slot_mapping(
        self, *, seed: int, device: torch.device
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        slots = torch.full((self.config.num_rows,), -1, dtype=torch.int64)
        if self.config.num_rows == 0:
            return slots.to(device)
        order = torch.randperm(self.config.num_rows, generator=rng)
        non_negative_count = self.config.num_rows - self.config.negative_slot_count
        if non_negative_count:
            total_slots = self.config.num_cache_blocks * self.config.block_size
            slots[order[:non_negative_count]] = torch.randperm(
                total_slots,
                generator=rng,
            )[:non_negative_count].to(torch.int64)
        return slots.to(device)

    def _generate_valid(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        valid = torch.ones((self.config.num_rows,), dtype=torch.bool)
        if self.config.masked_row_count:
            order = torch.randperm(self.config.num_rows, generator=rng)
            valid[order[: self.config.masked_row_count]] = False
        return valid.to(device)


def _deepseek_v4_mxfp4_nibble_reference(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.minimum(x.abs(), torch.tensor(6.0, device=x.device))
    code = torch.where(
        abs_x <= 0.25,
        torch.zeros_like(abs_x, dtype=torch.uint8),
        torch.where(
            abs_x <= 0.75,
            torch.ones_like(abs_x, dtype=torch.uint8),
            torch.where(
                abs_x <= 1.25,
                torch.full_like(abs_x, 2, dtype=torch.uint8),
                torch.where(
                    abs_x <= 1.75,
                    torch.full_like(abs_x, 3, dtype=torch.uint8),
                    torch.where(
                        abs_x <= 2.5,
                        torch.full_like(abs_x, 4, dtype=torch.uint8),
                        torch.where(
                            abs_x <= 3.5,
                            torch.full_like(abs_x, 5, dtype=torch.uint8),
                            torch.where(
                                abs_x <= 5.0,
                                torch.full_like(abs_x, 6, dtype=torch.uint8),
                                torch.full_like(abs_x, 7, dtype=torch.uint8),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    sign = ((x < 0) & (code != 0)).to(torch.uint8)
    return code | (sign << 3)


def _deepseek_v4_indexer_mxfp4_row_reference(
    row: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    row = row.float()
    packed = torch.empty(
        (_DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,),
        dtype=torch.uint8,
        device=row.device,
    )
    scales = torch.empty(
        (_DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES,),
        dtype=torch.uint8,
        device=row.device,
    )
    for block_idx in range(_DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES):
        block_base = block_idx * _DEEPSEEK_V4_INDEXER_MXFP4_BLOCK_SIZE
        block = row[block_base : block_base + _DEEPSEEK_V4_INDEXER_MXFP4_BLOCK_SIZE]
        lo = block[0::2]
        hi = block[1::2]
        amax = torch.maximum(
            torch.maximum(lo.abs().max(), hi.abs().max()),
            torch.tensor(1.0e-4, dtype=torch.float32, device=row.device),
        )
        exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
        inv_scale = torch.exp2(-exponent)
        lo_nibbles = _deepseek_v4_mxfp4_nibble_reference(lo * inv_scale)
        hi_nibbles = _deepseek_v4_mxfp4_nibble_reference(hi * inv_scale)
        start = block_idx * _DEEPSEEK_V4_INDEXER_MXFP4_HALF_BLOCK
        packed[start : start + _DEEPSEEK_V4_INDEXER_MXFP4_HALF_BLOCK] = lo_nibbles | (
            hi_nibbles << 4
        )
        scales[block_idx] = int(exponent.item()) + 127
    return packed, scales


def _validate_deepseek_v4_indexer_mxfp4_cache_write_values(
    values: DeepSeekV4IndexerMXFP4CacheWriteInputValues,
) -> None:
    if values.index_k.ndim != 2:
        raise ValueError(f"index_k must be rank-2, got {values.index_k.ndim}")
    if values.index_k.shape[1] != _DEEPSEEK_V4_INDEXER_DIM:
        raise ValueError(
            f"index_k width must be {_DEEPSEEK_V4_INDEXER_DIM}, "
            f"got {values.index_k.shape[1]}"
        )
    if values.cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {values.cache_2d.dtype}")
    if values.cache_2d.ndim != 2:
        raise ValueError(f"cache_2d must be rank-2, got {values.cache_2d.ndim}")
    block_size = _check_positive("block_size", values.block_size)
    min_row_bytes = block_size * (
        _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES + _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES
    )
    if values.cache_2d.shape[1] < min_row_bytes:
        raise ValueError(
            f"cache_2d row width must be at least {min_row_bytes}, "
            f"got {values.cache_2d.shape[1]}"
        )
    if values.slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be rank-1")
    if values.valid.ndim != 1:
        raise ValueError("valid must be rank-1")
    if values.slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"slot_mapping must be integer, got {values.slot_mapping.dtype}"
        )
    if values.valid.dtype != torch.bool:
        raise TypeError(f"valid must be bool, got {values.valid.dtype}")
    num_rows = min(
        values.index_k.shape[0],
        values.slot_mapping.numel(),
        values.valid.numel(),
    )
    if values.slot_mapping.numel() != values.index_k.shape[0]:
        raise ValueError("slot_mapping length must match index_k rows")
    if values.valid.numel() != values.index_k.shape[0]:
        raise ValueError("valid length must match index_k rows")
    if not values.index_k.is_floating_point():
        raise TypeError(f"index_k must be floating point, got {values.index_k.dtype}")
    if (
        values.cache_2d.device != values.index_k.device
        or values.slot_mapping.device != values.index_k.device
        or values.valid.device != values.index_k.device
    ):
        raise ValueError("index_k, cache_2d, slot_mapping, and valid must share device")
    slots = values.slot_mapping[:num_rows].to(torch.int64)
    writable = values.valid[:num_rows] & (slots >= 0)
    if not bool(writable.any().item()):
        return
    total_slots = values.cache_2d.shape[0] * block_size
    write_slots = slots[writable]
    if int(write_slots.max().item()) >= total_slots:
        raise ValueError(
            f"writable slot_mapping entries must be < {total_slots}, "
            f"got {int(write_slots.max().item())}"
        )
    unique_count = int(torch.unique(write_slots).numel())
    if unique_count != int(write_slots.numel()):
        raise ValueError("writable slot_mapping entries must be unique")


def deepseek_v4_indexer_mxfp4_cache_write_reference(
    values: DeepSeekV4IndexerMXFP4CacheWriteInputValues,
) -> torch.Tensor:
    """Return cache bytes after applying DeepSeek V4 indexer MXFP4 writes."""

    _validate_deepseek_v4_indexer_mxfp4_cache_write_values(values)
    out = values.cache_2d.clone()
    num_rows = min(
        values.index_k.shape[0],
        values.slot_mapping.numel(),
        values.valid.numel(),
    )
    slots = values.slot_mapping[:num_rows].to(torch.int64)
    for row_idx in range(num_rows):
        if not bool(values.valid[row_idx].item()):
            continue
        slot = int(slots[row_idx].item())
        if slot < 0:
            continue
        page = slot // values.block_size
        pos = slot % values.block_size
        page_base = page * out.stride(0)
        value_base = page_base + pos * _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
        scale_base = (
            page_base
            + values.block_size * _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
            + pos * _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES
        )
        packed, scales = _deepseek_v4_indexer_mxfp4_row_reference(
            values.index_k[row_idx]
        )
        flat = out.reshape(-1)
        flat[value_base : value_base + _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES] = packed
        flat[scale_base : scale_base + _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES] = scales
    return out.contiguous()


@dataclass
class DeepSeekV4IndexerMXFP4CacheGatherInputValues:
    """Generated values for gathering DeepSeek V4 indexer MXFP4 cache rows."""

    cache_2d: torch.Tensor
    slot_mapping: torch.Tensor
    values_out: torch.Tensor
    scales_out: torch.Tensor
    block_size: int


@dataclass
class DeepSeekV4IndexerMXFP4CacheGatherInputConfig:
    """Initialization parameters for DeepSeek V4 indexer MXFP4 cache gathers.

    The represented operation reads packed MXFP4 value bytes and UE8M0 scale
    bytes from a paged cache into dense workspaces. Negative slot ids produce
    zero-filled output rows.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of rows to gather.
    num_rows: int

    # Required: number of physical MXFP4 cache pages.
    num_cache_blocks: int

    # Required: number of indexer rows in each physical cache page.
    block_size: int

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: rows with slot_mapping == -1. These rows gather zeros.
    negative_slot_count: int = 0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DeepSeekV4IndexerMXFP4CacheGatherInputs(NumericsInputGenerator):
    """Generator for DeepSeek V4 indexer MXFP4 cache-gather inputs."""

    config: DeepSeekV4IndexerMXFP4CacheGatherInputConfig

    def __init__(self, config: DeepSeekV4IndexerMXFP4CacheGatherInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4IndexerMXFP4CacheGatherInputValues:
        self.__post_init__()
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        values = DeepSeekV4IndexerMXFP4CacheGatherInputValues(
            cache_2d=self._generate_cache(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ),
            slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            values_out=self._generate_output(
                seed=_child_seed(value_seed, 2),
                shape=(self.config.num_rows, _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES),
                device=target_device,
            ),
            scales_out=self._generate_output(
                seed=_child_seed(value_seed, 3),
                shape=(self.config.num_rows, _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES),
                device=target_device,
            ),
            block_size=self.config.block_size,
        )
        _validate_deepseek_v4_indexer_mxfp4_cache_gather_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.num_cache_blocks = _check_positive(
            "num_cache_blocks",
            self.config.num_cache_blocks,
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.negative_slot_count = _check_nonnegative(
            "negative_slot_count",
            self.config.negative_slot_count,
        )
        if self.config.negative_slot_count > self.config.num_rows:
            raise ValueError("negative_slot_count must be <= num_rows")
        total_slots = self.config.num_cache_blocks * self.config.block_size
        if self.config.num_rows > 0 and total_slots <= 0:
            raise ValueError("gather cache must contain at least one slot")

    def _cache_row_bytes(self) -> int:
        return self.config.block_size * (
            _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
            + _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES
        )

    def _generate_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        return torch.randint(
            0,
            256,
            (self.config.num_cache_blocks, self._cache_row_bytes()),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def _generate_output(
        self,
        *,
        seed: int,
        shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        return torch.randint(
            0,
            256,
            shape,
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def _generate_slot_mapping(
        self, *, seed: int, device: torch.device
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        slots = torch.full((self.config.num_rows,), -1, dtype=torch.int64)
        non_negative_count = self.config.num_rows - self.config.negative_slot_count
        if non_negative_count:
            total_slots = self.config.num_cache_blocks * self.config.block_size
            slots[:non_negative_count] = torch.randint(
                0,
                total_slots,
                (non_negative_count,),
                dtype=torch.int64,
                generator=rng,
            )
            order = torch.randperm(self.config.num_rows, generator=rng)
            slots = slots[order]
        return slots.to(device)


def _validate_deepseek_v4_indexer_mxfp4_cache_gather_values(
    values: DeepSeekV4IndexerMXFP4CacheGatherInputValues,
) -> None:
    if values.cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {values.cache_2d.dtype}")
    if values.values_out.dtype != torch.uint8:
        raise TypeError(f"values_out must be uint8, got {values.values_out.dtype}")
    if values.scales_out.dtype != torch.uint8:
        raise TypeError(f"scales_out must be uint8, got {values.scales_out.dtype}")
    if values.cache_2d.ndim != 2:
        raise ValueError(f"cache_2d must be rank-2, got {values.cache_2d.ndim}")
    if values.slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be rank-1")
    if values.values_out.ndim != 2 or values.scales_out.ndim != 2:
        raise ValueError("values_out and scales_out must be rank-2")
    block_size = _check_positive("block_size", values.block_size)
    min_row_bytes = block_size * (
        _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES + _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES
    )
    if values.cache_2d.shape[1] < min_row_bytes:
        raise ValueError(
            f"cache_2d row width must be at least {min_row_bytes}, "
            f"got {values.cache_2d.shape[1]}"
        )
    if values.cache_2d.stride(1) != 1:
        raise ValueError("cache_2d must be contiguous in the byte dimension")
    rows = values.slot_mapping.numel()
    if values.values_out.shape[0] < rows or values.scales_out.shape[0] < rows:
        raise ValueError(
            "gather output workspaces must have at least slot_mapping rows"
        )
    if values.values_out.shape[1] < _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES:
        raise ValueError("values_out has insufficient value bytes")
    if values.scales_out.shape[1] < _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES:
        raise ValueError("scales_out has insufficient scale bytes")
    if values.values_out.stride(1) != 1 or values.scales_out.stride(1) != 1:
        raise ValueError(
            "gather output workspaces must be contiguous in the byte dimension"
        )
    if values.slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"slot_mapping must be integer, got {values.slot_mapping.dtype}"
        )
    if (
        values.slot_mapping.device != values.cache_2d.device
        or values.values_out.device != values.cache_2d.device
        or values.scales_out.device != values.cache_2d.device
    ):
        raise ValueError(
            "cache_2d, slot_mapping, values_out, and scales_out must share device"
        )
    slots = values.slot_mapping.to(torch.int64)
    valid = slots >= 0
    if not bool(valid.any().item()):
        return
    total_slots = values.cache_2d.shape[0] * block_size
    valid_slots = slots[valid]
    if int(valid_slots.max().item()) >= total_slots:
        raise ValueError(
            f"slot_mapping entries must be < {total_slots}, "
            f"got {int(valid_slots.max().item())}"
        )


def deepseek_v4_indexer_mxfp4_cache_gather_reference(
    values: DeepSeekV4IndexerMXFP4CacheGatherInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense value and scale byte workspaces gathered from MXFP4 cache."""

    _validate_deepseek_v4_indexer_mxfp4_cache_gather_values(values)
    values_out = values.values_out.clone()
    scales_out = values.scales_out.clone()
    flat_cache = values.cache_2d.reshape(-1)
    slots = values.slot_mapping.to(torch.int64)
    for row_idx, slot_value in enumerate(slots.tolist()):
        if slot_value < 0:
            values_out[row_idx, :_DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES].zero_()
            scales_out[row_idx, :_DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES].zero_()
            continue
        page = slot_value // values.block_size
        pos = slot_value % values.block_size
        page_base = page * values.cache_2d.stride(0)
        value_base = page_base + pos * _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
        scale_base = (
            page_base
            + values.block_size * _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
            + pos * _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES
        )
        values_out[
            row_idx,
            :_DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
        ] = flat_cache[value_base : value_base + _DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES]
        scales_out[
            row_idx,
            :_DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES,
        ] = flat_cache[scale_base : scale_base + _DEEPSEEK_V4_INDEXER_MXFP4_SCALE_BYTES]
    return values_out.contiguous(), scales_out.contiguous()


@dataclass
class DeepSeekV4KCacheGatherInputValues:
    """Generated values for DeepSeek V4 sparse K-cache gather/dequantization."""

    out: torch.Tensor
    cache_2d: torch.Tensor
    seq_lens: torch.Tensor
    gather_lens: torch.Tensor | None
    block_table: torch.Tensor
    block_size: int
    offset: int
    block_table_base_offsets: torch.Tensor | None = None


@dataclass
class DeepSeekV4KCacheGatherInputConfig:
    """Initialization parameters for DeepSeek V4 K-cache gather/dequantization.

    The represented operation reads paged sparse-window attention K-cache rows.
    Each cache page stores per-token FP8 NoPE bytes, BF16 RoPE bytes, and UE8M0
    NoPE scale bytes. The generated metadata selects the suffix of each
    request to gather into a BF16 output workspace.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of independent request rows.
    batch_size: int

    # Required: maximum visible K-token sequence length per request.
    max_seq_len: int

    # Required: number of tokens in each physical cache page.
    block_size: int

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: maximum suffix length gathered for any request. Defaults to
    # max_seq_len. Actual per-request gather lengths are generated <= seq_len.
    max_gather_len: int | None = None

    # Optional: output-token offset where gathered rows are written.
    offset: int = 0

    # Optional: physical cache page count. Defaults to one unique page-table
    # entry for every generated request/page coordinate.
    num_cache_blocks: int | None = None

    # Optional: generate explicit gather_lens. When false, the operation
    # gathers the full generated seq_lens suffix for each request.
    include_gather_lens: bool = True

    # Optional: include block-table base offsets. Generated offsets are zero so
    # the metadata remains a valid unsliced page table while exercising the API.
    include_block_table_base_offsets: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DeepSeekV4KCacheGatherInputs(NumericsInputGenerator):
    """Generator for DeepSeek V4 paged K-cache gather/dequantization inputs."""

    config: DeepSeekV4KCacheGatherInputConfig

    def __init__(self, config: DeepSeekV4KCacheGatherInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4KCacheGatherInputValues:
        self.__post_init__()
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        metadata_generator = _rng_for_device(
            target_device, _child_seed(metadata_seed, 1)
        )
        seq_lens = self._generate_seq_lens(
            generator=metadata_generator,
            device=target_device,
        )
        gather_lens = self._generate_gather_lens(
            seq_lens=seq_lens,
            generator=metadata_generator,
        )
        block_table = self._generate_block_table(
            generator=metadata_generator,
            device=target_device,
        )
        base_offsets = None
        if self.config.include_block_table_base_offsets:
            base_offsets = torch.zeros(
                (self.config.batch_size,),
                dtype=torch.int32,
                device=target_device,
            )
        out_rows = self.config.offset + self._out_gather_width()
        values = DeepSeekV4KCacheGatherInputValues(
            out=self._generate_out(
                seed=_child_seed(value_seed, 1),
                out_rows=out_rows,
                device=target_device,
            ),
            cache_2d=self._generate_cache(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ),
            seq_lens=seq_lens.contiguous(),
            gather_lens=None if gather_lens is None else gather_lens.contiguous(),
            block_table=block_table.contiguous(),
            block_size=self.config.block_size,
            offset=self.config.offset,
            block_table_base_offsets=(
                None if base_offsets is None else base_offsets.contiguous()
            ),
        )
        _validate_deepseek_v4_k_cache_gather_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_nonnegative(
            "batch_size", self.config.batch_size
        )
        self.config.max_seq_len = _check_positive(
            "max_seq_len", self.config.max_seq_len
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        if self.config.max_gather_len is None:
            self.config.max_gather_len = self.config.max_seq_len
        self.config.max_gather_len = _check_positive(
            "max_gather_len", self.config.max_gather_len
        )
        if self.config.max_gather_len > self.config.max_seq_len:
            raise ValueError(
                "max_gather_len must be <= max_seq_len; got "
                f"{self.config.max_gather_len} > {self.config.max_seq_len}"
            )
        self.config.offset = _check_nonnegative("offset", self.config.offset)
        required_pages = self._required_page_table_entries()
        if self.config.num_cache_blocks is None:
            self.config.num_cache_blocks = max(1, required_pages)
        self.config.num_cache_blocks = _check_positive(
            "num_cache_blocks", self.config.num_cache_blocks
        )
        if self.config.num_cache_blocks < required_pages:
            raise ValueError(
                "num_cache_blocks must cover generated page-table entries; got "
                f"{self.config.num_cache_blocks} < {required_pages}"
            )

    def _max_blocks_per_seq(self) -> int:
        return math.ceil(self.config.max_seq_len / self.config.block_size)

    def _required_page_table_entries(self) -> int:
        return self.config.batch_size * self._max_blocks_per_seq()

    def _cache_row_bytes(self) -> int:
        return self.config.block_size * (
            _DEEPSEEK_V4_SWA_TOKEN_STRIDE + _DEEPSEEK_V4_SWA_SCALE_DIM
        )

    def _out_gather_width(self) -> int:
        if self.config.include_gather_lens:
            return int(self.config.max_gather_len)
        return int(self.config.max_seq_len)

    def _generate_seq_lens(
        self,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        if self.config.batch_size == 0:
            return torch.empty((0,), dtype=torch.int32, device=device)
        return torch.randint(
            1,
            self.config.max_seq_len + 1,
            (self.config.batch_size,),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )

    def _generate_gather_lens(
        self,
        *,
        seq_lens: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        if not self.config.include_gather_lens:
            return None
        gather_lens = torch.empty_like(seq_lens)
        for row, seq_len in enumerate(seq_lens.tolist()):
            upper = min(int(seq_len), int(self.config.max_gather_len))
            if upper <= 1:
                gather_lens[row] = upper
            else:
                gather_lens[row] = torch.randint(
                    1,
                    upper + 1,
                    (),
                    dtype=torch.int32,
                    device=seq_lens.device,
                    generator=generator,
                )
        return gather_lens

    def _generate_block_table(
        self,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        width = self._max_blocks_per_seq()
        if self.config.batch_size == 0:
            return torch.empty((0, width), dtype=torch.int32, device=device)
        assert self.config.num_cache_blocks is not None
        page_ids = torch.randperm(
            self.config.num_cache_blocks,
            dtype=torch.int64,
            device=device,
            generator=generator,
        )[: self._required_page_table_entries()]
        return page_ids.reshape(self.config.batch_size, width).to(torch.int32)

    def _generate_out(
        self,
        *,
        seed: int,
        out_rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        generator = _rng_for_device(device, seed)
        out = torch.randn(
            (self.config.batch_size, out_rows, _DEEPSEEK_V4_HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        return out.to(torch.bfloat16).contiguous()

    def _generate_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        assert self.config.num_cache_blocks is not None
        generator = _rng_for_device(device, seed)
        cache = torch.empty(
            (self.config.num_cache_blocks, self._cache_row_bytes()),
            dtype=torch.uint8,
            device=device,
        )
        token_area = cache[:, : self.config.block_size * _DEEPSEEK_V4_SWA_TOKEN_STRIDE]
        token_area = token_area.reshape(
            self.config.num_cache_blocks,
            self.config.block_size,
            _DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        )
        scale_area = cache[:, self.config.block_size * _DEEPSEEK_V4_SWA_TOKEN_STRIDE :]
        scale_area = scale_area.reshape(
            self.config.num_cache_blocks,
            self.config.block_size,
            _DEEPSEEK_V4_SWA_SCALE_DIM,
        )
        nope_fp8 = (
            torch.randn(
                (
                    self.config.num_cache_blocks,
                    self.config.block_size,
                    _DEEPSEEK_V4_NOPE_DIM,
                ),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            .clamp(-4.0, 4.0)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        rope_bf16 = (
            torch.randn(
                (
                    self.config.num_cache_blocks,
                    self.config.block_size,
                    _DEEPSEEK_V4_ROPE_DIM,
                ),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            .to(torch.bfloat16)
            .view(torch.uint8)
        )
        scale_bytes = torch.randint(
            123,
            131,
            (
                self.config.num_cache_blocks,
                self.config.block_size,
                _DEEPSEEK_V4_SWA_SCALE_DIM,
            ),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        scale_bytes[..., -1] = 127
        token_area[..., :_DEEPSEEK_V4_NOPE_DIM] = nope_fp8
        token_area[
            ...,
            _DEEPSEEK_V4_NOPE_DIM:_DEEPSEEK_V4_SWA_TOKEN_STRIDE,
        ] = rope_bf16
        scale_area.copy_(scale_bytes)
        return cache.contiguous()


def _validate_deepseek_v4_k_cache_gather_values(
    values: DeepSeekV4KCacheGatherInputValues,
) -> None:
    if values.out.dtype != torch.bfloat16:
        raise TypeError(f"out must be bfloat16, got {values.out.dtype}")
    if values.cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {values.cache_2d.dtype}")
    if values.out.ndim != 3:
        raise ValueError(f"out must be rank-3, got {values.out.ndim}")
    if values.out.shape[-1] != _DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(
            f"out hidden width must be {_DEEPSEEK_V4_HEAD_DIM}, "
            f"got {values.out.shape[-1]}"
        )
    if values.out.stride(-1) != 1:
        raise ValueError("out must be contiguous in the hidden dimension")
    if values.cache_2d.ndim != 2:
        raise ValueError(f"cache_2d must be rank-2, got {values.cache_2d.ndim}")
    block_size = _check_positive("block_size", values.block_size)
    row_bytes = block_size * (
        _DEEPSEEK_V4_SWA_TOKEN_STRIDE + _DEEPSEEK_V4_SWA_SCALE_DIM
    )
    if values.cache_2d.shape[1] < row_bytes:
        raise ValueError(
            f"cache_2d row width must be at least {row_bytes}, "
            f"got {values.cache_2d.shape[1]}"
        )
    if values.cache_2d.stride(1) != 1:
        raise ValueError("cache_2d must be contiguous in the byte dimension")
    if values.seq_lens.ndim != 1:
        raise ValueError("seq_lens must be rank-1")
    if values.seq_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"seq_lens must be integer, got {values.seq_lens.dtype}")
    if values.gather_lens is not None:
        if values.gather_lens.ndim != 1:
            raise ValueError("gather_lens must be rank-1")
        if values.gather_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"gather_lens must be integer, got {values.gather_lens.dtype}"
            )
        if values.gather_lens.shape != values.seq_lens.shape:
            raise ValueError("gather_lens shape must match seq_lens")
    if values.block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    batch_size = values.seq_lens.numel()
    if values.out.shape[0] != batch_size or values.block_table.shape[0] != batch_size:
        raise ValueError("out, block_table, and seq_lens batch dimensions must match")
    if values.block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"block_table must be integer, got {values.block_table.dtype}")
    if values.block_table_base_offsets is not None:
        if values.block_table_base_offsets.shape != values.seq_lens.shape:
            raise ValueError("block_table_base_offsets shape must match seq_lens")
        if values.block_table_base_offsets.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "block_table_base_offsets must be integer, got "
                f"{values.block_table_base_offsets.dtype}"
            )
    if (
        values.out.device != values.cache_2d.device
        or values.seq_lens.device != values.cache_2d.device
        or values.block_table.device != values.cache_2d.device
        or (
            values.gather_lens is not None
            and values.gather_lens.device != values.cache_2d.device
        )
        or (
            values.block_table_base_offsets is not None
            and values.block_table_base_offsets.device != values.cache_2d.device
        )
    ):
        raise ValueError("K-cache gather values must share a device")
    offset = _check_nonnegative("offset", values.offset)
    seq_lens = values.seq_lens.to(torch.int64)
    gather_lens = (
        seq_lens if values.gather_lens is None else values.gather_lens.to(torch.int64)
    )
    if bool((seq_lens < 0).any().item()):
        raise ValueError("seq_lens entries must be non-negative")
    if bool((gather_lens < 0).any().item()):
        raise ValueError("gather_lens entries must be non-negative")
    if bool((gather_lens > seq_lens).any().item()):
        raise ValueError("gather_lens entries must be <= seq_lens")
    if (
        gather_lens.numel()
        and offset + int(gather_lens.max().item()) > values.out.shape[1]
    ):
        raise ValueError("out token dimension is too small for offset + gather_lens")
    if values.block_table.numel():
        if bool((values.block_table < 0).any().item()):
            raise ValueError("block_table entries must be non-negative")
        if int(values.block_table.max().item()) >= values.cache_2d.shape[0]:
            raise ValueError("block_table entries must index cache_2d rows")
    base_offsets = (
        torch.zeros_like(seq_lens)
        if values.block_table_base_offsets is None
        else values.block_table_base_offsets.to(torch.int64)
    )
    if bool((base_offsets < 0).any().item()):
        raise ValueError("block_table_base_offsets entries must be non-negative")
    max_blocks = values.block_table.shape[1]
    for row in range(batch_size):
        seq_len = int(seq_lens[row].item())
        gather_len = int(gather_lens[row].item())
        start_pos = seq_len - gather_len
        base = int(base_offsets[row].item())
        for pos in range(start_pos, seq_len):
            table_idx = pos // block_size - base
            if table_idx < 0 or table_idx >= max_blocks:
                raise ValueError("block_table is too narrow for generated positions")


def deepseek_v4_dequantize_and_gather_k_cache_reference(
    values: DeepSeekV4KCacheGatherInputValues,
) -> torch.Tensor:
    """Return BF16 K rows gathered/dequantized from DeepSeek V4 paged cache."""

    _validate_deepseek_v4_k_cache_gather_values(values)
    out = values.out.clone()
    block_size = values.block_size
    seq_lens = values.seq_lens.to(torch.int64)
    gather_lens = (
        seq_lens if values.gather_lens is None else values.gather_lens.to(torch.int64)
    )
    base_offsets = (
        torch.zeros_like(seq_lens)
        if values.block_table_base_offsets is None
        else values.block_table_base_offsets.to(torch.int64)
    )
    n_quant_blocks = _DEEPSEEK_V4_NOPE_DIM // _DEEPSEEK_V4_FP8_QUANT_BLOCK
    for batch_idx in range(seq_lens.numel()):
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = int(gather_lens[batch_idx].item())
        start_pos = seq_len - gather_len
        base = int(base_offsets[batch_idx].item())
        for gather_idx in range(gather_len):
            pos = start_pos + gather_idx
            table_idx = pos // block_size - base
            out_row = out[batch_idx, values.offset + gather_idx]
            physical = int(values.block_table[batch_idx, table_idx].item())
            pos_in_block = pos % block_size
            cache_row = values.cache_2d[physical]
            token_base = pos_in_block * _DEEPSEEK_V4_SWA_TOKEN_STRIDE
            scale_base = (
                block_size * _DEEPSEEK_V4_SWA_TOKEN_STRIDE
                + pos_in_block * _DEEPSEEK_V4_SWA_SCALE_DIM
            )
            for qblock in range(n_quant_blocks):
                qstart = token_base + qblock * _DEEPSEEK_V4_FP8_QUANT_BLOCK
                qend = qstart + _DEEPSEEK_V4_FP8_QUANT_BLOCK
                x_fp8 = cache_row[qstart:qend].view(torch.float8_e4m3fn).float()
                scale_exp = int(cache_row[scale_base + qblock].item()) - 127
                scale = math.pow(2.0, scale_exp)
                out_row[
                    qblock
                    * _DEEPSEEK_V4_FP8_QUANT_BLOCK : (qblock + 1)
                    * _DEEPSEEK_V4_FP8_QUANT_BLOCK
                ] = (x_fp8 * scale).to(torch.bfloat16)
            rope_start = token_base + _DEEPSEEK_V4_NOPE_DIM
            rope_end = rope_start + _DEEPSEEK_V4_ROPE_DIM * 2
            out_row[_DEEPSEEK_V4_NOPE_DIM:] = cache_row[rope_start:rope_end].view(
                torch.bfloat16
            )
    return out.contiguous()


@dataclass
class DeepSeekV4PagedIndexValues:
    """Generated values for DeepSeek V4 paged index metadata operations."""

    metadata: MHARequestMetadataValues
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    local_topk_indices: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    block_table_cpu: list[list[int]]
    block_table_values: PageTableValues
    block_table_base_offsets: torch.Tensor | None
    is_valid_token: torch.Tensor | None
    block_size: int
    compress_ratio: int
    window_size: int
    topk: int
    max_blocks: int


@dataclass
class DeepSeekV4PagedIndexInputConfig:
    """Initialization parameters for DeepSeek V4 paged index metadata.

    The represented operation family maps request-local token or compressed
    token positions through a paged KV cache block table. The generated values
    are shared by DeepSeek V4 helpers that build global top-k slot ids, decode
    sliding-window slot ids, compressed KV slot mappings, and decode-indexer
    block-table/context metadata.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of request sequences represented by generated metadata.
    batch_size: int

    # Required: total already-resident KV tokens across all requests.
    total_cached_tokens: int

    # Required: total new query/KV tokens across all requests.
    total_new_q_tokens: int

    # Required: page/block size used to map logical token positions to slots.
    block_size: int

    # Required: compression ratio for compressed KV slot/indexer metadata.
    compress_ratio: int

    # Required: sliding-window token count used by decode SWA metadata.
    window_size: int

    # Required: number of request-local top-k indices generated per token.
    topk: int

    # ------------------------------------------------------------------
    # Optional metadata/page-table configuration.
    # ------------------------------------------------------------------

    # Optional: output block-table width for decode-indexer metadata. Defaults
    # to the generated page-table width.
    max_blocks: int | None = None

    # Optional: physical-page assignment policy for the generated block table.
    indexing: PageTableIndexing = "random"

    # Optional: generate block-table base offsets for helper APIs that model a
    # row as starting at a non-zero logical page.
    include_block_table_base_offsets: bool = False

    # Optional: generate an is_valid_token mask consumed by global-top-k and SWA
    # helpers. Invalid rows are kept in the tensors but should produce length 0.
    include_valid_token_mask: bool = False

    # Optional: approximate probability that one generated token is invalid
    # when include_valid_token_mask is true.
    invalid_token_probability: float = 0.25

    # Optional: nested request metadata config. If supplied, its batch size and
    # token totals must match this parent config.
    metadata_input: MHARequestMetadataInputConfig | None = None

    # Optional: nested page-table config. If supplied, its batch size must match
    # this parent config; its width is raised as needed during generation.
    page_table_input: PageTableInputConfig | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DeepSeekV4PagedIndexInputs(NumericsInputGenerator):
    """Generator for DeepSeek V4 paged index metadata inputs."""

    config: DeepSeekV4PagedIndexInputConfig
    metadata_input: MHARequestMetadataInput | None
    page_table_input: PageTableInput | None

    def __init__(self, config: DeepSeekV4PagedIndexInputConfig) -> None:
        self.config = config
        self.metadata_input = None
        self.page_table_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.metadata_input = self.metadata_input or MHARequestMetadataInput(
            self.config.metadata_input or self._make_metadata_config()
        )
        self._verify_metadata_config_matches_parent()
        self.config.metadata_input = self.metadata_input.config
        self.page_table_input = self.page_table_input or PageTableInput(
            self.config.page_table_input or self._make_page_table_config()
        )
        self._verify_page_table_config_matches_parent()
        self.config.page_table_input = self.page_table_input.config

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> DeepSeekV4PagedIndexValues:
        self.__post_init__()
        if self.metadata_input is None or self.page_table_input is None:
            raise ValueError("metadata/page-table child generators must be initialized")
        target_device = _resolve_device(self.config.device, device)
        metadata = self.metadata_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        )
        if metadata.new_q_lens_cpu != metadata.new_kv_lens_cpu:
            raise ValueError("DeepSeek V4 paged indices require tied Q/KV lengths")

        max_visible = max(metadata.visible_kv_lens_cpu)
        required_pages = max(1, math.ceil(max_visible / self.config.block_size))
        self.page_table_input.config.batch_size = self.config.batch_size
        self.page_table_input.config.max_pages_per_request = max(
            self.page_table_input.config.max_pages_per_request,
            required_pages,
        )
        self.page_table_input.config.indexing = self.config.indexing
        page_table_values = self.page_table_input.generate(
            seed=_child_seed(seed, 2),
            device=target_device,
        )
        max_blocks = (
            self.config.max_blocks
            if self.config.max_blocks is not None
            else page_table_values.page_table.shape[1]
        )

        positions_cpu: list[int] = []
        token_to_req_cpu: list[int] = []
        for req, (query_len, seq_len) in enumerate(
            zip(metadata.new_q_lens_cpu, metadata.visible_kv_lens_cpu, strict=True)
        ):
            start_pos = seq_len - query_len
            for offset in range(query_len):
                positions_cpu.append(start_pos + offset)
                token_to_req_cpu.append(req)

        local_topk_cpu = self._generate_local_topk_indices(
            seed=_child_seed(seed, 3),
            positions=positions_cpu,
        )
        is_valid_token = self._generate_valid_token_mask(
            seed=_child_seed(seed, 4),
            num_tokens=len(positions_cpu),
            device=target_device,
        )
        block_table_base_offsets = self._generate_block_table_base_offsets(
            seed=_child_seed(seed, 5),
            metadata=metadata,
            page_table_width=page_table_values.page_table.shape[1],
            device=target_device,
        )

        return DeepSeekV4PagedIndexValues(
            metadata=metadata,
            positions=torch.tensor(
                positions_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            token_to_req_indices=torch.tensor(
                token_to_req_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            local_topk_indices=local_topk_cpu.to(target_device),
            seq_lens=torch.tensor(
                metadata.visible_kv_lens_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            block_table=page_table_values.page_table,
            block_table_cpu=page_table_values.page_table_cpu,
            block_table_values=page_table_values,
            block_table_base_offsets=block_table_base_offsets,
            is_valid_token=is_valid_token,
            block_size=self.config.block_size,
            compress_ratio=self.config.compress_ratio,
            window_size=self.config.window_size,
            topk=self.config.topk,
            max_blocks=max_blocks,
        )

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.total_cached_tokens = _check_nonnegative(
            "total_cached_tokens",
            self.config.total_cached_tokens,
        )
        self.config.total_new_q_tokens = _check_positive(
            "total_new_q_tokens",
            self.config.total_new_q_tokens,
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.compress_ratio = _check_positive(
            "compress_ratio", self.config.compress_ratio
        )
        if self.config.compress_ratio <= 1:
            raise ValueError(
                "compress_ratio must be greater than 1 for compressed KV metadata, "
                f"got {self.config.compress_ratio}"
            )
        self.config.window_size = _check_positive(
            "window_size", self.config.window_size
        )
        self.config.topk = _check_positive("topk", self.config.topk)
        if self.config.max_blocks is not None:
            self.config.max_blocks = _check_positive(
                "max_blocks", self.config.max_blocks
            )
        self.config.indexing = _check_page_table_indexing(self.config.indexing)
        self.config.invalid_token_probability = float(
            self.config.invalid_token_probability
        )
        if not 0.0 <= self.config.invalid_token_probability < 1.0:
            raise ValueError("invalid_token_probability must be in [0, 1)")

    def _make_metadata_config(self) -> MHARequestMetadataInputConfig:
        return MHARequestMetadataInputConfig(
            batch_size=self.config.batch_size,
            total_cached_tokens=self.config.total_cached_tokens,
            total_new_q_tokens=self.config.total_new_q_tokens,
            cache_layout="paged",
            device=self.config.device,
        )

    def _make_page_table_config(self) -> PageTableInputConfig:
        return PageTableInputConfig(
            batch_size=self.config.batch_size,
            max_pages_per_request=1,
            indexing=self.config.indexing,
            device=self.config.device,
        )

    def _verify_metadata_config_matches_parent(self) -> None:
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        metadata_config = self.metadata_input.config
        for name in ("batch_size", "total_cached_tokens", "total_new_q_tokens"):
            _check_matches(
                parent_name=f"DeepSeekV4PagedIndexInputConfig.{name}",
                child_name=f"metadata_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(metadata_config, name),
            )
        if metadata_config.total_new_kv_tokens is not None:
            _check_matches(
                parent_name="DeepSeekV4PagedIndexInputConfig.total_new_q_tokens",
                child_name="metadata_input.total_new_kv_tokens",
                parent_value=self.config.total_new_q_tokens,
                child_value=metadata_config.total_new_kv_tokens,
            )
        if not metadata_config.tie_new_kv_to_query:
            raise ValueError(
                "metadata_input.tie_new_kv_to_query must be true for paged indices"
            )

    def _verify_page_table_config_matches_parent(self) -> None:
        if self.page_table_input is None:
            raise ValueError("page_table_input must be initialized")
        _check_matches(
            parent_name="DeepSeekV4PagedIndexInputConfig.batch_size",
            child_name="page_table_input.batch_size",
            parent_value=self.config.batch_size,
            child_value=self.page_table_input.config.batch_size,
        )

    def _generate_local_topk_indices(
        self,
        *,
        seed: int,
        positions: list[int],
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        local_topk = torch.full(
            (len(positions), self.config.topk),
            -1,
            dtype=torch.int32,
        )
        for token_idx, pos in enumerate(positions):
            available = int(pos) + 1
            topk_len = min(available, self.config.topk)
            if topk_len == 0:
                continue
            selected = torch.randperm(available, generator=rng)[:topk_len].to(
                torch.int32
            )
            local_topk[token_idx, :topk_len] = selected
        return local_topk

    def _generate_valid_token_mask(
        self,
        *,
        seed: int,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.config.include_valid_token_mask:
            return None
        rng = torch.Generator(device="cpu").manual_seed(seed)
        valid_cpu = (
            torch.rand(num_tokens, generator=rng)
            >= self.config.invalid_token_probability
        )
        if num_tokens:
            valid_cpu[0] = True
        return valid_cpu.to(device=device, dtype=torch.bool)

    def _generate_block_table_base_offsets(
        self,
        *,
        seed: int,
        metadata: MHARequestMetadataValues,
        page_table_width: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.config.include_block_table_base_offsets:
            return None
        rng = torch.Generator(device="cpu").manual_seed(seed)
        offsets: list[int] = []
        for seq_len in metadata.visible_kv_lens_cpu:
            max_logical_page = max(0, (seq_len - 1) // self.config.block_size)
            max_offset = min(max_logical_page, max(0, page_table_width - 1))
            if max_offset == 0:
                offsets.append(0)
            else:
                offsets.append(
                    int(torch.randint(0, max_offset + 1, (1,), generator=rng))
                )
        return torch.tensor(offsets, dtype=torch.int32, device=device)


def _validate_deepseek_v4_paged_index_values(
    values: DeepSeekV4PagedIndexValues,
) -> None:
    if values.positions.dtype != torch.int32:
        raise TypeError(f"positions must be int32, got {values.positions.dtype}")
    if values.token_to_req_indices.dtype != torch.int32:
        raise TypeError(
            "token_to_req_indices must be int32, "
            f"got {values.token_to_req_indices.dtype}"
        )
    if values.local_topk_indices.dtype != torch.int32:
        raise TypeError(
            f"local_topk_indices must be int32, got {values.local_topk_indices.dtype}"
        )
    if values.seq_lens.dtype != torch.int32:
        raise TypeError(f"seq_lens must be int32, got {values.seq_lens.dtype}")
    if values.block_table.dtype != torch.int32:
        raise TypeError(f"block_table must be int32, got {values.block_table.dtype}")
    if values.positions.ndim != 1:
        raise ValueError("positions must be rank-1")
    num_tokens = int(values.positions.numel())
    if values.token_to_req_indices.shape != (num_tokens,):
        raise ValueError("token_to_req_indices must have shape [num_tokens]")
    if values.local_topk_indices.shape != (num_tokens, values.topk):
        raise ValueError("local_topk_indices must have shape [num_tokens, topk]")
    if values.block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    batch_size = len(values.metadata.new_q_lens_cpu)
    if values.seq_lens.shape != (batch_size,):
        raise ValueError("seq_lens must have one value per request")
    if values.block_table.shape[0] != batch_size:
        raise ValueError("block_table must have one row per request")
    if values.is_valid_token is not None and values.is_valid_token.shape != (
        num_tokens,
    ):
        raise ValueError("is_valid_token must have shape [num_tokens]")
    if values.block_table_base_offsets is not None and (
        values.block_table_base_offsets.shape != (batch_size,)
        or values.block_table_base_offsets.dtype != torch.int32
    ):
        raise ValueError("block_table_base_offsets must be int32 [batch_size]")
    if values.block_size <= 0:
        raise ValueError("block_size must be positive")
    if values.compress_ratio <= 1:
        raise ValueError("compress_ratio must be greater than 1")
    if values.window_size <= 0:
        raise ValueError("window_size must be positive")
    if values.topk <= 0:
        raise ValueError("topk must be positive")
    if values.max_blocks <= 0:
        raise ValueError("max_blocks must be positive")


def deepseek_v4_compute_global_topk_indices_and_lens_reference(
    values: DeepSeekV4PagedIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global KV slot ids for request-local top-k indices."""

    _validate_deepseek_v4_paged_index_values(values)
    device = values.local_topk_indices.device
    topk = values.local_topk_indices
    num_tokens = int(topk.shape[0])
    out = torch.full_like(topk, -1)
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    reqs = values.token_to_req_indices.cpu().tolist()
    block_table_cpu = values.block_table.cpu()
    valid_mask = (
        [True] * num_tokens
        if values.is_valid_token is None
        else values.is_valid_token.cpu().tolist()
    )
    rows, cols = values.block_table.shape
    for token_idx, req in enumerate(reqs):
        if not valid_mask[token_idx] or req < 0 or req >= rows:
            continue
        count = 0
        for slot in range(values.topk):
            local_idx = int(topk[token_idx, slot].item())
            if local_idx < 0:
                continue
            block_idx = local_idx // values.block_size
            if block_idx < 0 or block_idx >= cols:
                continue
            block_number = int(block_table_cpu[req, block_idx].item())
            if block_number < 0:
                continue
            out[token_idx, slot] = block_number * values.block_size + (
                local_idx % values.block_size
            )
            count += 1
        lens[token_idx] = count
    return out, lens


def deepseek_v4_decode_swa_indices_and_lens_reference(
    values: DeepSeekV4PagedIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decode sliding-window KV slot ids and row lengths."""

    _validate_deepseek_v4_paged_index_values(values)
    device = values.positions.device
    num_tokens = int(values.positions.numel())
    out = torch.full(
        (num_tokens, values.window_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    reqs = values.token_to_req_indices.cpu().tolist()
    query_offsets = values.metadata.cu_seqlens_q_cpu
    seq_lens_cpu = values.seq_lens.cpu().tolist()
    block_table_cpu = values.block_table.cpu()
    base_offsets = (
        [0] * len(seq_lens_cpu)
        if values.block_table_base_offsets is None
        else values.block_table_base_offsets.cpu().tolist()
    )
    valid_mask = (
        [True] * num_tokens
        if values.is_valid_token is None
        else values.is_valid_token.cpu().tolist()
    )
    rows, cols = values.block_table.shape
    for token_idx, req in enumerate(reqs):
        if not valid_mask[token_idx] or req < 0 or req >= rows:
            continue
        query_start = query_offsets[req]
        query_end = query_offsets[req + 1]
        query_len = query_end - query_start
        prefix_len = int(seq_lens_cpu[req]) - query_len
        pos = prefix_len + token_idx - query_start
        start_pos = max(pos - values.window_size + 1, 0)
        swa_len = pos + 1 - start_pos
        lens[token_idx] = swa_len
        for offset in range(swa_len):
            pos_offset = start_pos + offset
            block_idx = pos_offset // values.block_size - int(base_offsets[req])
            if block_idx < 0 or block_idx >= cols:
                continue
            block_number = int(block_table_cpu[req, block_idx].item())
            if block_number < 0:
                continue
            out[token_idx, offset] = block_number * values.block_size + (
                pos_offset % values.block_size
            )
    return out, lens


def deepseek_v4_compressed_slot_mapping_reference(
    values: DeepSeekV4PagedIndexValues,
) -> torch.Tensor:
    """Return compressed KV slot ids for newly materialized compressed tokens."""

    _validate_deepseek_v4_paged_index_values(values)
    device = values.positions.device
    num_tokens = int(values.positions.numel())
    out = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    block_table_cpu = values.block_table.cpu()
    rows, cols = values.block_table.shape
    for req, seq_len in enumerate(values.metadata.visible_kv_lens_cpu):
        if req >= rows:
            continue
        query_start = values.metadata.cu_seqlens_q_cpu[req]
        query_end = values.metadata.cu_seqlens_q_cpu[req + 1]
        query_len = query_end - query_start
        start_pos = seq_len - query_len
        for offset, token_idx in enumerate(range(query_start, query_end)):
            pos = start_pos + offset
            if (pos + 1) % values.compress_ratio != 0:
                continue
            compressed_pos = pos // values.compress_ratio
            block_id = compressed_pos // values.block_size
            if block_id < 0 or block_id >= cols:
                continue
            block_number = int(block_table_cpu[req, block_id].item())
            if block_number < 0:
                continue
            out[token_idx] = block_number * values.block_size + (
                compressed_pos % values.block_size
            )
    return out


def deepseek_v4_indexer_decode_metadata_reference(
    values: DeepSeekV4PagedIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decode-indexer context lengths and block tables."""

    _validate_deepseek_v4_paged_index_values(values)
    device = values.positions.device
    num_tokens = int(values.positions.numel())
    out_context_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    out_block_tables = torch.zeros(
        (num_tokens, values.max_blocks),
        dtype=torch.int32,
        device=device,
    )
    positions_cpu = values.positions.cpu().tolist()
    reqs_cpu = values.token_to_req_indices.cpu().tolist()
    block_table_cpu = values.block_table.cpu()
    rows, cols = values.block_table.shape
    base_offsets = (
        [0] * rows
        if values.block_table_base_offsets is None
        else values.block_table_base_offsets.cpu().tolist()
    )
    for token_idx, (pos, req) in enumerate(zip(positions_cpu, reqs_cpu, strict=True)):
        if req < 0 or req >= rows:
            continue
        num_valid_pages = 0
        for col in range(values.max_blocks):
            if col >= cols:
                continue
            block_number = int(block_table_cpu[req, col].item())
            if block_number < 0:
                continue
            out_block_tables[token_idx, col] = block_number
            num_valid_pages += 1
        compressed_len = max(
            (int(pos) + 1) // values.compress_ratio
            - int(base_offsets[req]) * values.block_size,
            0,
        )
        out_context_lens[token_idx] = min(
            compressed_len,
            num_valid_pages * values.block_size,
        )
    return out_context_lens, out_block_tables


@dataclass
class DeepSeekV4SparsePrefillIndexValues:
    """Generated values for DeepSeek V4 sparse-prefill index construction.

    The tensors describe one packed batch of new query tokens. For each query
    token, the operation can build candidate KV workspace indices from a
    compressed prefix, a sparse top-k compressed-prefix selection, and a local
    sliding-window attention span.
    """

    metadata: MHARequestMetadataValues
    topk_indices: torch.Tensor
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    seq_lens: torch.Tensor
    compressed_lens: torch.Tensor
    gather_lens: torch.Tensor
    window_size: int
    compress_ratio: int
    topk: int
    workspace_width: int
    compressed_base: int


@dataclass
class DeepSeekV4SparsePrefillIndexInputConfig:
    """Initialization parameters for DeepSeek V4 sparse-prefill indices.

    The represented operation builds integer candidate lists for sparse
    prefill attention. Each request has a per-request workspace. The compressed
    prefix occupies local workspace slots ``[0, compressed_base)`` and the
    gathered sliding-window span starts at ``compressed_base``.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of request sequences represented by generated metadata.
    batch_size: int

    # Required: total already-resident KV tokens across all requests.
    total_cached_tokens: int

    # Required: total new query/KV tokens across all requests.
    total_new_q_tokens: int

    # Required: maximum number of sparse compressed-prefix candidates per token.
    topk: int

    # Required: sliding-window token count included for each query position.
    window_size: int

    # Required: number of original sequence positions represented by one
    # compressed prefix position. DeepSeek V4 sparse prefill uses a compressed
    # prefix plus an uncompressed sliding-window span, so this must be > 1.
    compress_ratio: int

    # ------------------------------------------------------------------
    # Optional workspace and metadata configuration.
    # ------------------------------------------------------------------

    # Optional: number of per-request workspace slots reserved for compressed
    # prefix entries. Defaults to the maximum generated compressed prefix length.
    compressed_base: int | None = None

    # Optional: total per-request workspace width. Defaults to
    # compressed_base + max(gather_lens).
    workspace_width: int | None = None

    # Optional: ordering policy for generated top-k compressed-prefix indices.
    # "identity" uses the first valid compressed positions; "random" samples a
    # metadata-seeded permutation from the valid compressed-prefix range.
    topk_indexing: PageTableIndexing = "random"

    # Optional: nested request metadata config. If supplied, its batch size and
    # token totals must match this parent config.
    metadata_input: MHARequestMetadataInputConfig | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class DeepSeekV4SparsePrefillIndexInputs(NumericsInputGenerator):
    """Generator for DeepSeek V4 sparse-prefill index-construction inputs."""

    config: DeepSeekV4SparsePrefillIndexInputConfig
    metadata_input: MHARequestMetadataInput | None

    def __init__(self, config: DeepSeekV4SparsePrefillIndexInputConfig) -> None:
        self.config = config
        self.metadata_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.metadata_input = self.metadata_input or MHARequestMetadataInput(
            self.config.metadata_input or self._make_metadata_config()
        )
        self._verify_metadata_config_matches_parent()
        self.config.metadata_input = self.metadata_input.config

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> DeepSeekV4SparsePrefillIndexValues:
        self.__post_init__()
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        target_device = _resolve_device(self.config.device, device)
        metadata = self.metadata_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        )
        if metadata.new_q_lens_cpu != metadata.new_kv_lens_cpu:
            raise ValueError("DeepSeek V4 sparse prefill requires tied Q/KV lengths")

        positions_cpu: list[int] = []
        token_to_req_cpu: list[int] = []
        gather_lens_cpu: list[int] = []
        compressed_lens_cpu: list[int] = []
        max_compressed_len = 0
        max_gather_len = 0
        for req, (query_len, seq_len) in enumerate(
            zip(metadata.new_q_lens_cpu, metadata.visible_kv_lens_cpu, strict=True)
        ):
            start_pos = seq_len - query_len
            gather_len = min(seq_len, self.config.window_size + query_len - 1)
            compressed_len = seq_len // self.config.compress_ratio
            gather_lens_cpu.append(gather_len)
            compressed_lens_cpu.append(compressed_len)
            max_gather_len = max(max_gather_len, gather_len)
            max_compressed_len = max(max_compressed_len, compressed_len)
            for offset in range(query_len):
                positions_cpu.append(start_pos + offset)
                token_to_req_cpu.append(req)

        compressed_base = (
            max_compressed_len
            if self.config.compressed_base is None
            else self.config.compressed_base
        )
        required_topk_prefix = min(max_compressed_len, self.config.topk)
        if compressed_base < required_topk_prefix:
            raise ValueError(
                "compressed_base is too small for generated top-k indices; "
                f"need at least {required_topk_prefix}, got {compressed_base}"
            )
        compressed_lens_cpu = [
            min(compressed_len, compressed_base)
            for compressed_len in compressed_lens_cpu
        ]
        workspace_width = (
            compressed_base + max_gather_len
            if self.config.workspace_width is None
            else self.config.workspace_width
        )
        if workspace_width < compressed_base + max_gather_len:
            raise ValueError(
                "workspace_width must cover compressed prefix slots plus gathered "
                f"SWA slots; need at least {compressed_base + max_gather_len}, "
                f"got {workspace_width}"
            )

        topk_indices_cpu = self._generate_topk_indices(
            seed=_child_seed(seed, 2),
            positions=positions_cpu,
            compressed_base=compressed_base,
        )

        return DeepSeekV4SparsePrefillIndexValues(
            metadata=metadata,
            topk_indices=topk_indices_cpu.to(target_device),
            positions=torch.tensor(
                positions_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            token_to_req_indices=torch.tensor(
                token_to_req_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            seq_lens=torch.tensor(
                metadata.visible_kv_lens_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            compressed_lens=torch.tensor(
                compressed_lens_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            gather_lens=torch.tensor(
                gather_lens_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            window_size=self.config.window_size,
            compress_ratio=self.config.compress_ratio,
            topk=self.config.topk,
            workspace_width=workspace_width,
            compressed_base=compressed_base,
        )

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.total_cached_tokens = _check_nonnegative(
            "total_cached_tokens",
            self.config.total_cached_tokens,
        )
        self.config.total_new_q_tokens = _check_positive(
            "total_new_q_tokens",
            self.config.total_new_q_tokens,
        )
        self.config.topk = _check_positive("topk", self.config.topk)
        self.config.window_size = _check_positive(
            "window_size", self.config.window_size
        )
        self.config.compress_ratio = _check_positive(
            "compress_ratio", self.config.compress_ratio
        )
        if self.config.compress_ratio <= 1:
            raise ValueError(
                "compress_ratio must be greater than 1 for compressed-prefix "
                f"sparse prefill, got {self.config.compress_ratio}"
            )
        if self.config.compressed_base is not None:
            self.config.compressed_base = _check_nonnegative(
                "compressed_base",
                self.config.compressed_base,
            )
        if self.config.workspace_width is not None:
            self.config.workspace_width = _check_positive(
                "workspace_width",
                self.config.workspace_width,
            )
        self.config.topk_indexing = _check_page_table_indexing(
            self.config.topk_indexing
        )

    def _make_metadata_config(self) -> MHARequestMetadataInputConfig:
        return MHARequestMetadataInputConfig(
            batch_size=self.config.batch_size,
            total_cached_tokens=self.config.total_cached_tokens,
            total_new_q_tokens=self.config.total_new_q_tokens,
            cache_layout=("dense" if self.config.total_cached_tokens else "none"),
            device=self.config.device,
        )

    def _verify_metadata_config_matches_parent(self) -> None:
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        metadata_config = self.metadata_input.config
        for name in ("batch_size", "total_cached_tokens", "total_new_q_tokens"):
            _check_matches(
                parent_name=f"DeepSeekV4SparsePrefillIndexInputConfig.{name}",
                child_name=f"metadata_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(metadata_config, name),
            )
        if metadata_config.total_new_kv_tokens is not None:
            _check_matches(
                parent_name=(
                    "DeepSeekV4SparsePrefillIndexInputConfig.total_new_q_tokens"
                ),
                child_name="metadata_input.total_new_kv_tokens",
                parent_value=self.config.total_new_q_tokens,
                child_value=metadata_config.total_new_kv_tokens,
            )
        if not metadata_config.tie_new_kv_to_query:
            raise ValueError(
                "metadata_input.tie_new_kv_to_query must be true for sparse prefill"
            )

    def _generate_topk_indices(
        self,
        *,
        seed: int,
        positions: list[int],
        compressed_base: int,
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        topk_indices = torch.full(
            (len(positions), self.config.topk),
            -1,
            dtype=torch.int32,
        )
        for token_idx, pos in enumerate(positions):
            available = min((pos + 1) // self.config.compress_ratio, compressed_base)
            topk_len = min(available, self.config.topk)
            if topk_len == 0:
                continue
            if self.config.topk_indexing == "identity":
                selected = torch.arange(topk_len, dtype=torch.int32)
            else:
                selected = torch.randperm(available, generator=rng)[:topk_len].to(
                    torch.int32
                )
            topk_indices[token_idx, :topk_len] = selected
        return topk_indices


def _validate_deepseek_v4_sparse_prefill_values(
    values: DeepSeekV4SparsePrefillIndexValues,
) -> None:
    if values.topk_indices.dtype != torch.int32:
        raise TypeError(f"topk_indices must be int32, got {values.topk_indices.dtype}")
    if values.positions.dtype != torch.int32:
        raise TypeError(f"positions must be int32, got {values.positions.dtype}")
    if values.token_to_req_indices.dtype != torch.int32:
        raise TypeError(
            "token_to_req_indices must be int32, "
            f"got {values.token_to_req_indices.dtype}"
        )
    for name, tensor in (
        ("seq_lens", values.seq_lens),
        ("compressed_lens", values.compressed_lens),
        ("gather_lens", values.gather_lens),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {tensor.dtype}")
    if values.topk_indices.ndim != 2:
        raise ValueError("topk_indices must be rank-2")
    num_tokens = values.topk_indices.shape[0]
    if values.topk_indices.shape[1] != values.topk:
        raise ValueError("topk scalar must match topk_indices width")
    if values.positions.shape != (num_tokens,):
        raise ValueError("positions must have shape [num_tokens]")
    if values.token_to_req_indices.shape != (num_tokens,):
        raise ValueError("token_to_req_indices must have shape [num_tokens]")
    batch_size = len(values.metadata.new_q_lens_cpu)
    if values.seq_lens.shape != (batch_size,):
        raise ValueError("seq_lens must have one value per request")
    if values.compressed_lens.shape != (batch_size,):
        raise ValueError("compressed_lens must have one value per request")
    if values.gather_lens.shape != (batch_size,):
        raise ValueError("gather_lens must have one value per request")
    if values.window_size <= 0:
        raise ValueError("window_size must be positive")
    if values.compress_ratio <= 1:
        raise ValueError("compress_ratio must be greater than 1")
    if values.topk <= 0:
        raise ValueError("topk must be positive")
    if values.compressed_base < 0:
        raise ValueError("compressed_base must be non-negative")
    if values.workspace_width <= 0:
        raise ValueError("workspace_width must be positive")
    max_gather_len = int(values.gather_lens.max().item()) if batch_size else 0
    if values.workspace_width < values.compressed_base + max_gather_len:
        raise ValueError("workspace_width does not cover generated workspace indices")


def deepseek_v4_build_dense_prefill_local_compressed_indices_reference(
    values: DeepSeekV4SparsePrefillIndexValues,
    *,
    width: int | None = None,
) -> torch.Tensor:
    """Return per-token local compressed-prefix indices for dense prefill."""

    _validate_deepseek_v4_sparse_prefill_values(values)
    width = (
        values.compressed_base if width is None else _check_nonnegative("width", width)
    )
    positions = values.positions.to(torch.int64)
    offsets = torch.arange(width, dtype=torch.int64, device=positions.device)
    compressed_lens = torch.div(
        positions + 1,
        values.compress_ratio,
        rounding_mode="floor",
    ).clamp(0, width)
    local = offsets[None, :].expand(positions.numel(), -1)
    valid = offsets[None, :] < compressed_lens[:, None]
    return torch.where(valid, local, torch.full_like(local, -1)).to(torch.int32)


def deepseek_v4_combine_topk_swa_indices_reference(
    values: DeepSeekV4SparsePrefillIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse-prefill candidate indices from top-k prefix plus SWA."""

    _validate_deepseek_v4_sparse_prefill_values(values)
    device = values.topk_indices.device
    num_tokens = int(values.topk_indices.shape[0])
    combined_topk = _align_up(
        values.topk + values.window_size,
        _DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT,
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    combined_lens = torch.empty(num_tokens, dtype=torch.int32, device=device)

    base = values.metadata.cu_seqlens_q_cpu[0]
    topk_cpu = values.topk_indices.cpu()
    for req, seq_len in enumerate(values.metadata.visible_kv_lens_cpu):
        query_start = values.metadata.cu_seqlens_q_cpu[req] - base
        query_end = values.metadata.cu_seqlens_q_cpu[req + 1] - base
        query_len = query_end - query_start
        start_pos = seq_len - query_len
        gather_start = seq_len - int(values.gather_lens[req].item())
        for token_idx in range(query_start, query_end):
            token_offset = token_idx - query_start
            pos = start_pos + token_offset
            topk_len = min((pos + 1) // values.compress_ratio, values.topk)
            swa_len = min(pos + 1, values.window_size)
            if topk_len:
                topk_values = topk_cpu[token_idx, :topk_len].to(device=device)
                if (topk_values < 0).any() or (
                    topk_values >= values.compressed_base
                ).any():
                    raise ValueError("topk_indices contain invalid compressed slots")
                combined_indices[token_idx, :topk_len] = (
                    topk_values + values.workspace_width * req
                )
            swa_values = (
                values.workspace_width * req
                + values.compressed_base
                + torch.arange(swa_len, dtype=torch.int32, device=device)
                + pos
                - swa_len
                + 1
                - gather_start
            )
            combined_indices[token_idx, topk_len : topk_len + swa_len] = swa_values
            combined_lens[token_idx] = topk_len + swa_len

    return combined_indices, combined_lens


def deepseek_v4_combine_dense_swa_indices_reference(
    values: DeepSeekV4SparsePrefillIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse-prefill candidate indices from dense prefix plus SWA."""

    _validate_deepseek_v4_sparse_prefill_values(values)
    device = values.positions.device
    num_tokens = int(values.positions.numel())
    combined_topk = _align_up(
        max(values.compressed_base + values.window_size, 1),
        _DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT,
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    combined_lens = torch.empty(num_tokens, dtype=torch.int32, device=device)

    positions_cpu = values.positions.cpu().tolist()
    reqs_cpu = values.token_to_req_indices.cpu().tolist()
    seq_lens_cpu = values.seq_lens.cpu().tolist()
    compressed_lens_cpu = values.compressed_lens.cpu().tolist()
    gather_lens_cpu = values.gather_lens.cpu().tolist()
    for token_idx, (pos, req) in enumerate(zip(positions_cpu, reqs_cpu, strict=True)):
        if req < 0 or req >= len(seq_lens_cpu):
            raise ValueError("token_to_req_indices contains an invalid request id")
        seq_len = int(seq_lens_cpu[req])
        gather_start = seq_len - int(gather_lens_cpu[req])
        compressed_len = min(
            (int(pos) + 1) // values.compress_ratio,
            int(compressed_lens_cpu[req]),
        )
        swa_len = min(int(pos) + 1, values.window_size)
        request_base = values.workspace_width * int(req)
        if compressed_len:
            combined_indices[token_idx, :compressed_len] = request_base + torch.arange(
                compressed_len, dtype=torch.int32, device=device
            )
        swa_values = (
            request_base
            + values.compressed_base
            + torch.arange(swa_len, dtype=torch.int32, device=device)
            + int(pos)
            - swa_len
            + 1
            - gather_start
        )
        combined_indices[token_idx, compressed_len : compressed_len + swa_len] = (
            swa_values
        )
        combined_lens[token_idx] = compressed_len + swa_len

    return combined_indices, combined_lens


@dataclass
class MHAInputValues:
    """Generated values for ``MHAInputs``."""

    metadata: MHARequestMetadataValues
    q: torch.Tensor | None
    k: torch.Tensor | None
    v: torch.Tensor | None
    sinks: torch.Tensor | None
    cache: KVCacheValues | None = None

    def dense_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return regular batch-major Q/K/V views for dense non-cached tests."""

        if self.cache is not None:
            raise ValueError("dense_qkv requires cache_layout='none'")
        if (
            len(set(self.metadata.new_q_lens_cpu)) != 1
            or self.metadata.new_q_lens_cpu != self.metadata.new_kv_lens_cpu
        ):
            raise ValueError("dense_qkv requires equal fixed per-request Q/KV lengths")
        if self.q is None or self.k is None or self.v is None:
            raise ValueError("generated Q/K/V tensors are incomplete")
        batch_size = len(self.metadata.new_q_lens_cpu)
        seqlen = self.metadata.new_q_lens_cpu[0]
        return (
            self.q.view(batch_size, seqlen, self.q.shape[-2], self.q.shape[-1]),
            self.k.view(batch_size, seqlen, self.k.shape[-2], self.k.shape[-1]),
            self.v.view(batch_size, seqlen, self.v.shape[-2], self.v.shape[-1]),
        )


@dataclass
class MHAInputConfig:
    """Initialization parameters for ``MHAInputs``.

    Common request and cache controls live here so basic attention tests do not
    need to construct child configs. Detailed request-length distribution
    options live in ``metadata_input``. Detailed dense/paged cache storage
    options live in ``cache_input`` and its nested page-table config.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of request sequences represented by generated metadata.
    batch_size: int

    # Required: total already-resident KV tokens across all requests.
    total_cached_tokens: int

    # Required: total query tokens that produce outputs in this call.
    total_new_q_tokens: int

    # Required: query head count. May be larger than num_kv_heads for GQA/MQA.
    num_q_heads: int

    # Required: KV head count. Must divide num_q_heads.
    num_kv_heads: int

    # Required: per-head Q/K/V dimension for standard MHA kernels.
    head_dim: int

    # Required: dtype for generated query values.
    q_dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional common cache/page configuration.
    # ------------------------------------------------------------------

    # Optional: K/V storage surface represented by this attention invocation.
    cache_layout: CacheLayout = "none"

    # Optional: paged-cache physical page size. Required when cache_layout is
    # "paged" and no cache_input override supplies a page size.
    page_size: int | None = None

    # Optional: page-table physical-page assignment policy. Relevant only for
    # paged cache; if omitted, PageTableInput's default policy is used.
    indexing: PageTableIndexing | None = None

    # ------------------------------------------------------------------
    # Optional generated tensor dtype and device configuration.
    # ------------------------------------------------------------------

    # Optional: dtype for generated explicit K values. Defaults to q_dtype.
    k_dtype: torch.dtype | None = None

    # Optional: dtype for generated explicit V values. Defaults to k_dtype.
    v_dtype: torch.dtype | None = None

    # Optional: default generation device. Caller-supplied generate(device=...)
    # overrides omitted per-tensor devices.
    device: DeviceLike = None

    # ------------------------------------------------------------------
    # Optional auxiliary generated tensor configuration.
    # ------------------------------------------------------------------

    # Optional: generate attention sinks, an extra per-Q-head tensor supported
    # by some MHA prefill/decode kernels.
    include_sinks: bool = False

    # Optional: dtype for attention sinks. Defaults to q_dtype.
    sink_dtype: torch.dtype | None = None

    # ------------------------------------------------------------------
    # Optional child generator overrides.
    # ------------------------------------------------------------------

    # Optional: nested config for request lengths, cumulative offsets, cache
    # sequence lengths, detailed length modes, and max K length.
    metadata_input: MHARequestMetadataInputConfig | None = None

    # Optional: nested config for dense/paged KV-cache storage and page table.
    # Required when metadata_input.cache_layout != "none" and cache defaults
    # are insufficient, for example paged cache needs page_size.
    cache_input: KVCacheInputConfig | None = None


@dataclass(init=False)
class MHAInputs(NumericsInputGenerator):
    """Typed input generator for MHA attention surfaces.

    Child tensor/cache/metadata generators are created during initialization
    from ``MHAInputConfig`` and reused by ``generate``. Tests may mutate those
    child generator fields before generation.
    """

    config: MHAInputConfig
    metadata_input: MHARequestMetadataInput | None
    q_input: TensorInput | None
    k_input: TensorInput | None
    v_input: TensorInput | None
    cache_input: KVCacheInput | None
    sinks_input: TensorInput | None

    def __init__(
        self,
        config: MHAInputConfig,
    ) -> None:
        self.config = config
        self.metadata_input = None
        self.q_input = None
        self.k_input = None
        self.v_input = None
        self.cache_input = None
        self.sinks_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.metadata_input = self.metadata_input or MHARequestMetadataInput(
            self.config.metadata_input or self._make_metadata_config()
        )
        self._verify_metadata_config_matches_parent()
        self.config.metadata_input = self.metadata_input.config

        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        if self.config.num_q_heads % self.config.num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads, got "
                f"{self.config.num_q_heads} and {self.config.num_kv_heads}"
            )
        if self.config.k_dtype is None:
            self.config.k_dtype = self.config.q_dtype
        if self.config.v_dtype is None:
            self.config.v_dtype = self.config.k_dtype
        if self.config.sink_dtype is None:
            self.config.sink_dtype = self.config.q_dtype

        if self.config.cache_layout == "none":
            if self.config.cache_input is not None or self.cache_input is not None:
                raise ValueError("cache_input requires cache_layout != 'none'")
            self.cache_input = None
        elif self.cache_input is None:
            cache_config = self._prepare_cache_config(
                self.config.cache_input or self._make_cache_config()
            )
            self.cache_input = KVCacheInput(
                cache_config,
            )
        else:
            self._prepare_cache_config(self.cache_input.config)
        if self.cache_input is not None:
            self._verify_cache_config_matches_parent()

        self.q_input = self.q_input or TensorInput(
            (0, self.config.num_q_heads, self.config.head_dim),
            self.config.q_dtype,
            device=self.config.device,
        )
        self.k_input = self.k_input or TensorInput(
            (0, self.config.num_kv_heads, self.config.head_dim),
            self.config.k_dtype,
            device=self.config.device,
        )
        self.v_input = self.v_input or TensorInput(
            (0, self.config.num_kv_heads, self.config.head_dim),
            self.config.v_dtype,
            device=self.config.device,
        )
        self.sinks_input = self.sinks_input or TensorInput(
            (self.config.num_q_heads,),
            self.config.sink_dtype if self.config.include_sinks else None,
            device=self.config.device,
        )
        self.config.cache_input = (
            None if self.cache_input is None else self.cache_input.config
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MHAInputValues:
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        metadata = self._generate_metadata(metadata_seed, target_device)
        generated = self._generate_attention_values(
            metadata=metadata,
            value_seed=value_seed,
            metadata_seed=metadata_seed,
            device=target_device,
        )
        return self._make_values(
            metadata=metadata,
            q=generated.q,
            k=generated.k,
            v=generated.v,
            sinks=generated.sinks,
            cache=generated.cache,
        )

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.total_cached_tokens = _check_nonnegative(
            "total_cached_tokens",
            self.config.total_cached_tokens,
        )
        self.config.total_new_q_tokens = _check_positive(
            "total_new_q_tokens",
            self.config.total_new_q_tokens,
        )
        self.config.cache_layout = _check_cache_layout(str(self.config.cache_layout))
        if self.config.page_size is not None:
            self.config.page_size = _check_positive("page_size", self.config.page_size)
        self.config.indexing = _check_page_table_indexing(
            None if self.config.indexing is None else str(self.config.indexing)
        )
        if self.config.indexing is not None and self.config.cache_layout != "paged":
            raise ValueError("indexing requires cache_layout='paged'")
        if self.config.cache_layout == "none" and self.config.total_cached_tokens != 0:
            raise ValueError("cache_layout='none' requires total_cached_tokens == 0")

    def _make_metadata_config(self) -> MHARequestMetadataInputConfig:
        return MHARequestMetadataInputConfig(
            batch_size=self.config.batch_size,
            total_cached_tokens=self.config.total_cached_tokens,
            total_new_q_tokens=self.config.total_new_q_tokens,
            cache_layout=self.config.cache_layout,
            device=self.config.device,
        )

    def _verify_metadata_config_matches_parent(self) -> None:
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        metadata = self.metadata_input
        for name in (
            "batch_size",
            "total_cached_tokens",
            "total_new_q_tokens",
            "cache_layout",
        ):
            _check_matches(
                parent_name=f"MHAInputConfig.{name}",
                child_name=f"metadata_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(metadata.config, name),
            )

    def _prepare_cache_config(
        self,
        cache_config: KVCacheInputConfig,
    ) -> KVCacheInputConfig:
        if self.config.page_size is not None and cache_config.page_size is None:
            cache_config.page_size = self.config.page_size
        if self.config.indexing is not None and cache_config.page_table_input is None:
            cache_config.page_table_input = PageTableInputConfig(
                batch_size=self.config.batch_size,
                max_pages_per_request=1,
                indexing=self.config.indexing,
                device=cache_config.device,
            )
        return cache_config

    def _verify_cache_config_matches_parent(self) -> None:
        if self.cache_input is None:
            raise ValueError("cache_input must be initialized")
        cache_input = self.cache_input
        for name in ("cache_layout", "batch_size", "num_kv_heads", "head_dim"):
            _check_matches(
                parent_name=f"MHAInputConfig.{name}",
                child_name=f"cache_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(cache_input.config, name),
            )
        if self.config.page_size is not None:
            _check_matches(
                parent_name="MHAInputConfig.page_size",
                child_name="cache_input.page_size",
                parent_value=self.config.page_size,
                child_value=cache_input.config.page_size,
            )
        if self.config.indexing is not None:
            if cache_input.page_table_input is None:
                raise ValueError("indexing requires cache_input.page_table_input")
            _check_matches(
                parent_name="MHAInputConfig.indexing",
                child_name="cache_input.page_table_input.indexing",
                parent_value=self.config.indexing,
                child_value=cache_input.page_table_input.config.indexing,
            )

    def _generate_metadata(
        self,
        seed: int,
        device: torch.device,
    ) -> MHARequestMetadataValues:
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        self._verify_metadata_config_matches_parent()
        metadata = self.metadata_input.generate(seed=seed, device=device)
        self._verify_metadata_config_matches_parent()
        return metadata

    def _generate_attention_values(
        self,
        *,
        metadata: MHARequestMetadataValues,
        value_seed: int,
        metadata_seed: int,
        device: torch.device,
    ) -> AttentionGeneratedValues[KVCacheValues]:
        if (
            self.q_input is None
            or self.k_input is None
            or self.v_input is None
            or self.sinks_input is None
        ):
            raise ValueError("MHAInputs child generators must be initialized")
        if self.config.cache_layout != "none" and self.cache_input is None:
            self.cache_input = KVCacheInput(self._make_cache_config())
        if self.config.cache_layout == "none":
            self.cache_input = None

        total_q = sum(metadata.new_q_lens_cpu)
        total_new_kv = sum(metadata.new_kv_lens_cpu)

        generated = attention_generate(
            metadata=metadata,
            value_seed=value_seed,
            metadata_seed=metadata_seed,
            device=device,
            cache_layout=self.config.cache_layout,
            q_input=self.q_input,
            q_shape=(total_q, self.config.num_q_heads, self.config.head_dim),
            k_input=self.k_input,
            k_shape=(total_new_kv, self.config.num_kv_heads, self.config.head_dim),
            v_input=self.v_input,
            v_shape=(total_new_kv, self.config.num_kv_heads, self.config.head_dim),
            sinks_input=self.sinks_input,
            sinks_shape=(self.config.num_q_heads,),
            sinks_dtype=self.config.sink_dtype if self.config.include_sinks else None,
            generate_cache=self._generate_cache,
            cache_seed_index=5,
            page_table_seed_index=5,
        )
        if generated.cache is not None:
            self._copy_new_kv_into_cache(
                metadata=metadata,
                k=generated.k,
                v=generated.v,
                cache=generated.cache,
            )
        return generated

    def _generate_cache(
        self,
        *,
        metadata: MHARequestMetadataValues,
        value_seed: int,
        page_table_seed: int,
        device: torch.device,
    ) -> KVCacheValues:
        if self.cache_input is None:
            self.cache_input = KVCacheInput(self._make_cache_config())
        self._verify_cache_config_matches_parent()
        self.cache_input.config.max_seqlen_k = metadata.resolved_max_seqlen_k
        cache = self.cache_input.generate(
            seed=value_seed,
            page_table_seed=page_table_seed,
            device=device,
        )

        self.config.cache_input = self.cache_input.config
        return cache

    def _make_cache_config(self) -> KVCacheInputConfig:
        if self.config.cache_layout == "none":
            raise ValueError("cache_layout='none' does not have a cache_input")
        return KVCacheInputConfig(
            cache_layout=self.config.cache_layout,
            batch_size=self.config.batch_size,
            max_seqlen_k=_metadata_config(self.metadata_input).max_seqlen_k or 0,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.k_dtype,
            page_size=self.config.page_size,
            page_table_input=(
                None
                if self.config.indexing is None
                else PageTableInputConfig(
                    batch_size=self.config.batch_size,
                    max_pages_per_request=1,
                    indexing=self.config.indexing,
                    device=self.config.device,
                )
            ),
            device=self.config.device,
        )

    def _copy_new_kv_into_cache(
        self,
        *,
        metadata: MHARequestMetadataValues,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        cache: KVCacheValues,
    ) -> None:
        if (
            self.config.cache_layout == "none"
            or k is None
            or v is None
            or cache.k_cache is None
            or cache.v_cache is None
        ):
            return

        kv_offset = 0
        for batch_idx, (cached_len, new_kv_len) in enumerate(
            zip(metadata.cached_lens_cpu, metadata.new_kv_lens_cpu, strict=True)
        ):
            for token_idx in range(new_kv_len):
                logical_pos = cached_len + token_idx
                src = kv_offset + token_idx
                if self.config.cache_layout == "dense":
                    cache.k_cache[batch_idx, logical_pos].copy_(
                        k[src].to(cache.k_cache.dtype)
                    )
                    cache.v_cache[batch_idx, logical_pos].copy_(
                        v[src].to(cache.v_cache.dtype)
                    )
                else:
                    assert cache.page_table_cpu is not None
                    assert self.cache_input is not None
                    assert self.cache_input.config.page_size is not None
                    page_col = logical_pos // self.cache_input.config.page_size
                    page_offset = logical_pos % self.cache_input.config.page_size
                    physical_page = cache.page_table_cpu[batch_idx][page_col]
                    cache.k_cache[physical_page, page_offset].copy_(
                        k[src].to(cache.k_cache.dtype)
                    )
                    cache.v_cache[physical_page, page_offset].copy_(
                        v[src].to(cache.v_cache.dtype)
                    )
            kv_offset += new_kv_len

    def _make_values(
        self,
        *,
        metadata: MHARequestMetadataValues,
        q: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        sinks: torch.Tensor | None,
        cache: KVCacheValues | None,
    ) -> MHAInputValues:
        return MHAInputValues(
            metadata=metadata,
            q=q,
            k=k,
            v=v,
            sinks=sinks,
            cache=cache,
        )


@dataclass
class MLAInputValues:
    """Generated values for ``MLAInputs``.

    MLA kernel entry points need several scalar shape parameters in addition to
    tensors. Those scalars are included here because they are part of the
    generated kernel input bundle, not just generator configuration metadata.
    """

    metadata: MHARequestMetadataValues
    q: torch.Tensor | None
    k: torch.Tensor | None
    v: torch.Tensor | None
    cache: MLAKVCacheValues | None
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    softmax_scale: float


@dataclass
class MLAInputConfig:
    """Initialization parameters for ``MLAInputs``.

    The request/cache controls mirror ``MHAInputConfig``. The model-shape
    controls use MLA dimensions: materialized prefill Q/K use
    ``qk_nope_head_dim + qk_rope_head_dim`` while cached decode Q/cache use
    ``kv_lora_rank + qk_rope_head_dim``.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of request sequences represented by generated metadata.
    batch_size: int

    # Required: total already-resident KV tokens across all requests.
    total_cached_tokens: int

    # Required: total query tokens that produce outputs in this call.
    total_new_q_tokens: int

    # Required: query head count.
    num_q_heads: int

    # Required: original non-RoPE Q/K head dimension used for MLA scaling and
    # kernel specialization. In absorbed decode this does not appear directly
    # in the generated Q/cache last dimension.
    qk_nope_head_dim: int

    # Required: RoPE Q/K head dimension. Stored in materialized prefill Q/K and
    # in compressed decode Q/cache rows.
    qk_rope_head_dim: int

    # Required: latent KV rank stored by compressed MLA decode cache.
    kv_lora_rank: int

    # Required: materialized value/output dimension used by MLA prefill.
    v_head_dim: int

    # Required: dtype for generated query values.
    q_dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional common cache/page configuration.
    # ------------------------------------------------------------------

    # Optional: explicit non-cached prefill or compressed cached decode.
    cache_layout: CacheLayout = "none"

    # Optional: paged-cache physical page size. Required when cache_layout is
    # "paged" and no cache_input override supplies a page size.
    page_size: int | None = None

    # Optional: page-table physical-page assignment policy. Relevant only for
    # paged cache; if omitted, PageTableInput's default policy is used.
    indexing: PageTableIndexing | None = None

    # ------------------------------------------------------------------
    # Optional generated tensor dtype and device configuration.
    # ------------------------------------------------------------------

    # Optional: KV head count for materialized MLA prefill. Defaults to
    # num_q_heads, which models expanded per-head K/V.
    num_kv_heads: int | None = None

    # Optional: dtype for generated explicit K values. Defaults to q_dtype.
    k_dtype: torch.dtype | None = None

    # Optional: dtype for generated explicit V values. Defaults to q_dtype.
    v_dtype: torch.dtype | None = None

    # Optional: dtype for generated compressed MLA cache. Defaults to q_dtype.
    kv_cache_dtype: torch.dtype | None = None

    # Optional: override softmax scale. Defaults to
    # 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim).
    softmax_scale: float | None = None

    # Optional: default generation device. Caller-supplied generate(device=...)
    # overrides omitted per-tensor devices.
    device: DeviceLike = None

    # ------------------------------------------------------------------
    # Optional child generator overrides.
    # ------------------------------------------------------------------

    # Optional: nested config for request lengths, cumulative offsets, cache
    # sequence lengths, detailed length modes, and max K length.
    metadata_input: MHARequestMetadataInputConfig | None = None

    # Optional: nested config for compressed dense/paged MLA cache storage and
    # page table.
    cache_input: MLAKVCacheInputConfig | None = None


@dataclass(init=False)
class MLAInputs(NumericsInputGenerator):
    """Typed input generator for MLA attention surfaces."""

    config: MLAInputConfig
    metadata_input: MHARequestMetadataInput | None
    q_input: TensorInput | None
    k_input: TensorInput | None
    v_input: TensorInput | None
    cache_input: MLAKVCacheInput | None

    def __init__(
        self,
        config: MLAInputConfig,
    ) -> None:
        self.config = config
        self.metadata_input = None
        self.q_input = None
        self.k_input = None
        self.v_input = None
        self.cache_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.metadata_input = self.metadata_input or MHARequestMetadataInput(
            self.config.metadata_input or self._make_metadata_config()
        )
        self._verify_metadata_config_matches_parent()
        self.config.metadata_input = self.metadata_input.config

        if self.config.num_kv_heads is None:
            self.config.num_kv_heads = self.config.num_q_heads
        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        if self.config.num_q_heads % self.config.num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads, got "
                f"{self.config.num_q_heads} and {self.config.num_kv_heads}"
            )
        self.config.qk_nope_head_dim = _check_positive(
            "qk_nope_head_dim", self.config.qk_nope_head_dim
        )
        self.config.qk_rope_head_dim = _check_positive(
            "qk_rope_head_dim", self.config.qk_rope_head_dim
        )
        self.config.kv_lora_rank = _check_positive(
            "kv_lora_rank", self.config.kv_lora_rank
        )
        self.config.v_head_dim = _check_positive("v_head_dim", self.config.v_head_dim)
        if self.config.k_dtype is None:
            self.config.k_dtype = self.config.q_dtype
        if self.config.v_dtype is None:
            self.config.v_dtype = self.config.q_dtype
        if self.config.kv_cache_dtype is None:
            self.config.kv_cache_dtype = self.config.q_dtype
        if self.config.softmax_scale is None:
            self.config.softmax_scale = 1.0 / math.sqrt(self._prefill_qk_head_dim())

        if self.config.cache_layout == "none":
            if self.config.cache_input is not None or self.cache_input is not None:
                raise ValueError("cache_input requires cache_layout != 'none'")
            self.cache_input = None
        elif self.cache_input is None:
            cache_config = self._prepare_cache_config(
                self.config.cache_input or self._make_cache_config()
            )
            self.cache_input = MLAKVCacheInput(cache_config)
        else:
            self._prepare_cache_config(self.cache_input.config)
        if self.cache_input is not None:
            self._verify_cache_config_matches_parent()

        self.q_input = self.q_input or TensorInput(
            (0, self.config.num_q_heads, self._prefill_qk_head_dim()),
            self.config.q_dtype,
            device=self.config.device,
        )
        self.k_input = self.k_input or TensorInput(
            (0, self.config.num_kv_heads, self._prefill_qk_head_dim()),
            self.config.k_dtype,
            device=self.config.device,
        )
        self.v_input = self.v_input or TensorInput(
            (0, self.config.num_kv_heads, self.config.v_head_dim),
            self.config.v_dtype,
            device=self.config.device,
        )
        self.config.cache_input = (
            None if self.cache_input is None else self.cache_input.config
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MLAInputValues:
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        metadata = self._generate_metadata(metadata_seed, target_device)
        generated = self._generate_attention_values(
            metadata=metadata,
            value_seed=value_seed,
            metadata_seed=metadata_seed,
            device=target_device,
        )
        return self._make_values(
            metadata=metadata,
            q=generated.q,
            k=generated.k,
            v=generated.v,
            cache=generated.cache,
        )

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.total_cached_tokens = _check_nonnegative(
            "total_cached_tokens",
            self.config.total_cached_tokens,
        )
        self.config.total_new_q_tokens = _check_positive(
            "total_new_q_tokens",
            self.config.total_new_q_tokens,
        )
        self.config.cache_layout = _check_cache_layout(str(self.config.cache_layout))
        if self.config.page_size is not None:
            self.config.page_size = _check_positive("page_size", self.config.page_size)
        self.config.indexing = _check_page_table_indexing(
            None if self.config.indexing is None else str(self.config.indexing)
        )
        if self.config.indexing is not None and self.config.cache_layout != "paged":
            raise ValueError("indexing requires cache_layout='paged'")
        if self.config.cache_layout == "none" and self.config.total_cached_tokens != 0:
            raise ValueError("cache_layout='none' requires total_cached_tokens == 0")

    def _make_metadata_config(self) -> MHARequestMetadataInputConfig:
        return MHARequestMetadataInputConfig(
            batch_size=self.config.batch_size,
            total_cached_tokens=self.config.total_cached_tokens,
            total_new_q_tokens=self.config.total_new_q_tokens,
            cache_layout=self.config.cache_layout,
            new_q_length_mode=(
                "fixed_per_request" if self.config.cache_layout != "none" else "ragged"
            ),
            allow_untied_non_cached_kv=True,
            device=self.config.device,
        )

    def _verify_metadata_config_matches_parent(self) -> None:
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        metadata = self.metadata_input
        for name in (
            "batch_size",
            "total_cached_tokens",
            "total_new_q_tokens",
            "cache_layout",
        ):
            _check_matches(
                parent_name=f"MLAInputConfig.{name}",
                child_name=f"metadata_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(metadata.config, name),
            )

    def _prepare_cache_config(
        self,
        cache_config: MLAKVCacheInputConfig,
    ) -> MLAKVCacheInputConfig:
        if self.config.page_size is not None and cache_config.page_size is None:
            cache_config.page_size = self.config.page_size
        if self.config.indexing is not None and cache_config.page_table_input is None:
            cache_config.page_table_input = PageTableInputConfig(
                batch_size=self.config.batch_size,
                max_pages_per_request=1,
                indexing=self.config.indexing,
                device=cache_config.device,
            )
        return cache_config

    def _verify_cache_config_matches_parent(self) -> None:
        if self.cache_input is None:
            raise ValueError("cache_input must be initialized")
        cache_input = self.cache_input
        for name in (
            "cache_layout",
            "batch_size",
            "kv_lora_rank",
            "qk_rope_head_dim",
        ):
            _check_matches(
                parent_name=f"MLAInputConfig.{name}",
                child_name=f"cache_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(cache_input.config, name),
            )
        if self.config.page_size is not None:
            _check_matches(
                parent_name="MLAInputConfig.page_size",
                child_name="cache_input.page_size",
                parent_value=self.config.page_size,
                child_value=cache_input.config.page_size,
            )
        if self.config.indexing is not None:
            if cache_input.page_table_input is None:
                raise ValueError("indexing requires cache_input.page_table_input")
            _check_matches(
                parent_name="MLAInputConfig.indexing",
                child_name="cache_input.page_table_input.indexing",
                parent_value=self.config.indexing,
                child_value=cache_input.page_table_input.config.indexing,
            )

    def _generate_metadata(
        self,
        seed: int,
        device: torch.device,
    ) -> MHARequestMetadataValues:
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        self._verify_metadata_config_matches_parent()
        metadata = self.metadata_input.generate(seed=seed, device=device)
        self._verify_metadata_config_matches_parent()
        return metadata

    def _generate_attention_values(
        self,
        *,
        metadata: MHARequestMetadataValues,
        value_seed: int,
        metadata_seed: int,
        device: torch.device,
    ) -> AttentionGeneratedValues[MLAKVCacheValues]:
        if self.q_input is None or self.k_input is None or self.v_input is None:
            raise ValueError("MLAInputs child generators must be initialized")

        if self.config.cache_layout == "none":
            total_q = sum(metadata.new_q_lens_cpu)
            total_kv = sum(metadata.new_kv_lens_cpu)
            q_shape = (total_q, self.config.num_q_heads, self._prefill_qk_head_dim())
            k_input = self.k_input
            k_shape = (
                total_kv,
                self.config.num_kv_heads,
                self._prefill_qk_head_dim(),
            )
            v_input = self.v_input
            v_shape = (
                total_kv,
                self.config.num_kv_heads,
                self.config.v_head_dim,
            )
            generate_cache = None
            fixed_q_length_error = None
        else:
            if self.cache_input is None:
                self.cache_input = MLAKVCacheInput(self._make_cache_config())
            q_len = metadata.new_q_lens_cpu[0]
            q_shape = (
                self.config.batch_size,
                q_len,
                self.config.num_q_heads,
                self._decode_qk_head_dim(),
            )
            k_input = None
            k_shape = None
            v_input = None
            v_shape = None
            generate_cache = self._generate_cache
            fixed_q_length_error = (
                "cached MLA generation requires a fixed query length per request"
            )

        return attention_generate(
            metadata=metadata,
            value_seed=value_seed,
            metadata_seed=metadata_seed,
            device=device,
            cache_layout=self.config.cache_layout,
            q_input=self.q_input,
            q_shape=q_shape,
            k_input=k_input,
            k_shape=k_shape,
            v_input=v_input,
            v_shape=v_shape,
            generate_cache=generate_cache,
            cache_seed_index=4,
            page_table_seed_index=4,
            fixed_q_length_error=fixed_q_length_error,
        )

    def _generate_cache(
        self,
        *,
        metadata: MHARequestMetadataValues,
        value_seed: int,
        page_table_seed: int,
        device: torch.device,
    ) -> MLAKVCacheValues:
        if self.cache_input is None:
            self.cache_input = MLAKVCacheInput(self._make_cache_config())
        self._verify_cache_config_matches_parent()
        self.cache_input.config.max_seqlen_k = metadata.resolved_max_seqlen_k
        cache = self.cache_input.generate(
            seed=value_seed,
            page_table_seed=page_table_seed,
            device=device,
        )
        self.config.cache_input = self.cache_input.config
        return cache

    def _make_cache_config(self) -> MLAKVCacheInputConfig:
        if self.config.cache_layout == "none":
            raise ValueError("cache_layout='none' does not have a cache_input")
        return MLAKVCacheInputConfig(
            cache_layout=self.config.cache_layout,
            batch_size=self.config.batch_size,
            max_seqlen_k=_metadata_config(self.metadata_input).max_seqlen_k or 0,
            kv_lora_rank=self.config.kv_lora_rank,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            dtype=self.config.kv_cache_dtype,
            page_size=self.config.page_size,
            page_table_input=(
                None
                if self.config.indexing is None
                else PageTableInputConfig(
                    batch_size=self.config.batch_size,
                    max_pages_per_request=1,
                    indexing=self.config.indexing,
                    device=self.config.device,
                )
            ),
            device=self.config.device,
        )

    def _make_values(
        self,
        *,
        metadata: MHARequestMetadataValues,
        q: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        cache: MLAKVCacheValues | None,
    ) -> MLAInputValues:
        if self.config.softmax_scale is None:
            raise ValueError("softmax_scale must be initialized")
        return MLAInputValues(
            metadata=metadata,
            q=q,
            k=k,
            v=v,
            cache=cache,
            qk_nope_head_dim=self.config.qk_nope_head_dim,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            kv_lora_rank=self.config.kv_lora_rank,
            softmax_scale=self.config.softmax_scale,
        )

    def _prefill_qk_head_dim(self) -> int:
        return self.config.qk_nope_head_dim + self.config.qk_rope_head_dim

    def _decode_qk_head_dim(self) -> int:
        return self.config.kv_lora_rank + self.config.qk_rope_head_dim
