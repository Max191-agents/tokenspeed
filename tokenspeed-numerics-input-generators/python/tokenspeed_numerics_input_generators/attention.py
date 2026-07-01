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
    PageTableInputConfig,
)
from tokenspeed_numerics_input_generators.attention_metadata import (
    CacheLayout,
    MHARequestMetadataInputConfig,
    MHARequestMetadataInput,
    MHARequestMetadataValues,
    PageTableIndexing,
)
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _resolve_device,
)

__all__ = [
    "AttentionMergeStateInputConfig",
    "AttentionMergeStateInputs",
    "AttentionMergeStateInputValues",
    "attention_generate",
    "attention_merge_state_reference",
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
]

_AttentionCacheT = TypeVar("_AttentionCacheT")


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
