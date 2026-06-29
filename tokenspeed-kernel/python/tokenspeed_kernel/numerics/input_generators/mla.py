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

"""MLA-family input generators for numerical correctness tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tokenspeed_kernel.numerics.input_generators.attention_cache import (
    MLAKVCacheInput,
    MLAKVCacheInputConfig,
    MLAKVCacheValues,
    PageTableInputConfig,
)
from tokenspeed_kernel.numerics.input_generators.attention_metadata import (
    CacheLayout,
    MHARequestMetadataInput,
    MHARequestMetadataInputConfig,
    MHARequestMetadataValues,
    PageTableIndexing,
)
from tokenspeed_kernel.numerics.input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    TensorInputConfig,
    _child_seed,
    _resolve_device,
    _tensor_input_from_config,
)

__all__ = ["MLAInputConfig", "MLAInputValues", "MLAInputs"]


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

    # Optional: nested config for Q. Initialized automatically so tests can
    # mutate shape-independent leaf settings before calling generate().
    q_input: TensorInputConfig | None = None

    # Optional: nested config for explicit non-cached K.
    k_input: TensorInputConfig | None = None

    # Optional: nested config for explicit non-cached V.
    v_input: TensorInputConfig | None = None

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
        self._refresh_metadata_state_from_child()
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

        q_config = self.config.q_input or TensorInputConfig(
            (0, self.config.num_q_heads, self._prefill_qk_head_dim()),
            self.config.q_dtype,
            device=self.config.device,
        )
        k_config = self.config.k_input or TensorInputConfig(
            (0, self.config.num_kv_heads, self._prefill_qk_head_dim()),
            self.config.k_dtype,
            device=self.config.device,
        )
        v_config = self.config.v_input or TensorInputConfig(
            (0, self.config.num_kv_heads, self.config.v_head_dim),
            self.config.v_dtype,
            device=self.config.device,
        )
        self.q_input = self.q_input or _tensor_input_from_config(q_config)
        self.k_input = self.k_input or _tensor_input_from_config(k_config)
        self.v_input = self.v_input or _tensor_input_from_config(v_config)
        self.config.q_input = self.q_input.config
        self.config.k_input = self.k_input.config
        self.config.v_input = self.v_input.config
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
        if seed is None and (metadata_seed is None or value_seed is None):
            raise ValueError("provide either seed or both metadata_seed and value_seed")
        if seed is not None:
            if metadata_seed is None:
                metadata_seed = _child_seed(seed, 1)
            if value_seed is None:
                value_seed = _child_seed(seed, 2)
        assert metadata_seed is not None
        assert value_seed is not None

        target_device = _resolve_device(self.config.device, device)
        metadata = self._generate_metadata(metadata_seed, target_device)
        return self._generate_values(
            metadata=metadata,
            value_seed=value_seed,
            metadata_seed=metadata_seed,
            device=target_device,
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

    def _refresh_metadata_state_from_child(self) -> None:
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        metadata = self.metadata_input
        self.config.batch_size = metadata.config.batch_size
        self.config.total_cached_tokens = metadata.config.total_cached_tokens
        self.config.total_new_q_tokens = metadata.config.total_new_q_tokens
        self.total_new_kv_tokens = metadata.config.total_new_kv_tokens
        self.config.cache_layout = metadata.config.cache_layout
        self.cached_length_mode = metadata.config.cached_length_mode
        self.max_cached_tokens_per_request = (
            metadata.config.max_cached_tokens_per_request
        )
        self.new_q_length_mode = metadata.config.new_q_length_mode
        self.max_new_q_tokens_per_request = metadata.config.max_new_q_tokens_per_request
        self.new_kv_length_mode = metadata.config.new_kv_length_mode
        self.max_new_kv_tokens_per_request = (
            metadata.config.max_new_kv_tokens_per_request
        )
        self.tie_new_kv_to_query = metadata.config.tie_new_kv_to_query
        self.max_seqlen_k = metadata.config.max_seqlen_k

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
        self._refresh_metadata_state_from_child()
        return metadata

    def _generate_values(
        self,
        *,
        metadata: MHARequestMetadataValues,
        value_seed: int,
        metadata_seed: int,
        device: torch.device,
    ) -> MLAInputValues:
        if self.q_input is None or self.k_input is None or self.v_input is None:
            raise ValueError("MLAInputs child generators must be initialized")

        if self.config.cache_layout == "none":
            total_q = sum(metadata.new_q_lens_cpu)
            total_kv = sum(metadata.new_kv_lens_cpu)
            self.q_input.config.shape = (
                total_q,
                self.config.num_q_heads,
                self._prefill_qk_head_dim(),
            )
            self.k_input.config.shape = (
                total_kv,
                self.config.num_kv_heads,
                self._prefill_qk_head_dim(),
            )
            self.v_input.config.shape = (
                total_kv,
                self.config.num_kv_heads,
                self.config.v_head_dim,
            )
            q = self.q_input.generate(seed=_child_seed(value_seed, 1), device=device)
            k = self.k_input.generate(seed=_child_seed(value_seed, 2), device=device)
            v = self.v_input.generate(seed=_child_seed(value_seed, 3), device=device)
            return self._make_values(metadata=metadata, q=q, k=k, v=v, cache=None)

        if len(set(metadata.new_q_lens_cpu)) != 1:
            raise ValueError(
                "cached MLA generation requires a fixed query length per request"
            )
        if self.cache_input is None:
            self.cache_input = MLAKVCacheInput(self._make_cache_config())
        q_len = metadata.new_q_lens_cpu[0]
        self.q_input.config.shape = (
            self.config.batch_size,
            q_len,
            self.config.num_q_heads,
            self._decode_qk_head_dim(),
        )
        q = self.q_input.generate(seed=_child_seed(value_seed, 1), device=device)
        cache = self._generate_cache(
            metadata=metadata,
            value_seed=_child_seed(value_seed, 4),
            page_table_seed=_child_seed(metadata_seed, 4),
            device=device,
        )
        return self._make_values(metadata=metadata, q=q, k=None, v=None, cache=cache)

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
            max_seqlen_k=self.max_seqlen_k or 0,
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
