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

"""MHA-family input generators for numerical correctness tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.numerics.input_generators.attention_cache import (
    KVCacheInputConfig,
    KVCacheInput,
    KVCacheValues,
    PageTableInputConfig,
)
from tokenspeed_kernel.numerics.input_generators.attention_metadata import (
    CacheLayout,
    MHARequestMetadataInputConfig,
    MHARequestMetadataInput,
    MHARequestMetadataValues,
    PageTableIndexing,
)
from tokenspeed_kernel.numerics.input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInputConfig,
    TensorInput,
    _child_seed,
    _resolve_device,
    _tensor_input_from_config,
)

__all__ = ["MHAInputConfig", "MHAInputValues", "MHAInputs"]


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

    # Optional: nested config for Q. Initialized automatically so tests can
    # mutate shape-independent leaf settings before calling generate().
    q_input: TensorInputConfig | None = None

    # Optional: nested config for explicit non-cached or newly inserted K.
    k_input: TensorInputConfig | None = None

    # Optional: nested config for explicit non-cached or newly inserted V.
    v_input: TensorInputConfig | None = None

    # Optional: nested config for dense/paged KV-cache storage and page table.
    # Required when metadata_input.cache_layout != "none" and cache defaults
    # are insufficient, for example paged cache needs page_size.
    cache_input: KVCacheInputConfig | None = None

    # Optional: nested config for attention sinks.
    sinks_input: TensorInputConfig | None = None


@dataclass(init=False)
class MHAInputs(NumericsInputGenerator):
    """Typed input generator for MHA attention surfaces.

    Child tensor/cache/metadata generators are created during initialization
    from ``MHAInputConfig`` and reused by ``generate``. Tests may mutate those
    child generator configs before generation.
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
        self._refresh_metadata_state_from_child()
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

        q_config = self.config.q_input or TensorInputConfig(
            (0, self.config.num_q_heads, self.config.head_dim),
            self.config.q_dtype,
            device=self.config.device,
        )
        k_config = self.config.k_input or TensorInputConfig(
            (0, self.config.num_kv_heads, self.config.head_dim),
            self.config.k_dtype,
            device=self.config.device,
        )
        v_config = self.config.v_input or TensorInputConfig(
            (0, self.config.num_kv_heads, self.config.head_dim),
            self.config.v_dtype,
            device=self.config.device,
        )
        sinks_config = self.config.sinks_input or TensorInputConfig(
            (self.config.num_q_heads,),
            self.config.sink_dtype if self.config.include_sinks else None,
            device=self.config.device,
        )
        self.q_input = self.q_input or _tensor_input_from_config(q_config)
        self.k_input = self.k_input or _tensor_input_from_config(k_config)
        self.v_input = self.v_input or _tensor_input_from_config(v_config)
        self.sinks_input = self.sinks_input or _tensor_input_from_config(sinks_config)
        self.config.q_input = self.q_input.config
        self.config.k_input = self.k_input.config
        self.config.v_input = self.v_input.config
        self.config.cache_input = (
            None if self.cache_input is None else self.cache_input.config
        )
        self.config.sinks_input = self.sinks_input.config

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MHAInputValues:
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
        self._refresh_metadata_state_from_child()
        return metadata

    def _generate_values(
        self,
        *,
        metadata: MHARequestMetadataValues,
        value_seed: int,
        metadata_seed: int,
        device: torch.device,
    ) -> MHAInputValues:
        if (
            self.q_input is None
            or self.k_input is None
            or self.v_input is None
            or self.sinks_input is None
        ):
            raise ValueError("MHAInputs child generators must be initialized")
        if self.config.cache_layout != "none" and self.cache_input is None:
            self.cache_input = KVCacheInput(self._make_cache_config())

        total_q = sum(metadata.new_q_lens_cpu)
        total_new_kv = sum(metadata.new_kv_lens_cpu)
        self.q_input.config.shape = (
            total_q,
            self.config.num_q_heads,
            self.config.head_dim,
        )
        self.k_input.config.shape = (
            total_new_kv,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        self.v_input.config.shape = (
            total_new_kv,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        self.sinks_input.config.shape = (self.config.num_q_heads,)
        self.sinks_input.config.dtype = (
            self.config.sink_dtype if self.config.include_sinks else None
        )

        q = self.q_input.generate(seed=_child_seed(value_seed, 1), device=device)
        k = self.k_input.generate(seed=_child_seed(value_seed, 2), device=device)
        v = self.v_input.generate(seed=_child_seed(value_seed, 3), device=device)
        sinks = self.sinks_input.generate(
            seed=_child_seed(value_seed, 4),
            device=device,
        )

        if self.config.cache_layout == "none":
            self.cache_input = None
            return self._make_values(
                metadata=metadata,
                q=q,
                k=k,
                v=v,
                sinks=sinks,
                cache=None,
            )

        cache = self._generate_cache(
            metadata=metadata,
            value_seed=_child_seed(value_seed, 5),
            page_table_seed=_child_seed(metadata_seed, 5),
            device=device,
        )
        self._copy_new_kv_into_cache(
            metadata=metadata,
            k=k,
            v=v,
            cache=cache,
        )
        return self._make_values(
            metadata=metadata,
            q=q,
            k=k,
            v=v,
            sinks=sinks,
            cache=cache,
        )

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
            max_seqlen_k=self.max_seqlen_k or 0,
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
