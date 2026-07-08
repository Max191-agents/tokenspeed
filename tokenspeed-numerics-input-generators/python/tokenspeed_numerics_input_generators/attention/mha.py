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

from ._core import (
    AttentionGeneratedValues,
    CacheLayout,
    DeviceLike,
    KVCacheInput,
    KVCacheInputConfig,
    KVCacheValues,
    MHARequestMetadataInput,
    MHARequestMetadataInputConfig,
    MHARequestMetadataValues,
    NumericsInputGenerator,
    PageTableIndexing,
    PageTableInputConfig,
    TensorInput,
    _check_cache_layout,
    _check_matches,
    _check_nonnegative,
    _check_page_table_indexing,
    _check_positive,
    _fp8_dtypes,
    _metadata_config,
    _resolve_attention_seeds,
    _resolve_device,
    attention_generate,
    dataclass,
    math,
    torch,
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
class MHAReferenceValues:
    """Reference output tensors for ``MHAInputValues``."""

    out: torch.Tensor
    lse: torch.Tensor


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


def _apply_attention_logit_cap(scores: torch.Tensor, logit_cap: float) -> torch.Tensor:
    logit_cap = float(logit_cap)
    if logit_cap == 0.0:
        return scores
    if logit_cap < 0.0:
        raise ValueError(f"logit_cap must be non-negative, got {logit_cap}")
    return torch.tanh(scores / logit_cap) * logit_cap


def _attention_output_dtype(input_dtype: torch.dtype) -> torch.dtype:
    return torch.bfloat16 if input_dtype in _fp8_dtypes() else input_dtype


def _expand_kv_heads_for_queries(
    tensor: torch.Tensor,
    *,
    num_q_heads: int,
    name: str,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be rank-3, got {tensor.ndim}")
    num_kv_heads = tensor.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads must be divisible by {name} heads; got "
            f"{num_q_heads} and {num_kv_heads}"
        )
    return tensor.repeat_interleave(num_q_heads // num_kv_heads, dim=1)


def _attention_scores_with_optional_sink(
    scores: torch.Tensor,
    sinks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sinks is None:
        probs = torch.softmax(scores, dim=1)
        lse = torch.logsumexp(scores, dim=1)
        return probs, lse
    if sinks.ndim != 1 or sinks.shape[0] != scores.shape[-1]:
        raise ValueError(
            "sinks must have shape [num_q_heads], got " f"{tuple(sinks.shape)}"
        )
    sink_scores = (
        sinks.to(torch.float32)
        .view(1, 1, -1)
        .expand(
            scores.shape[0],
            1,
            scores.shape[-1],
        )
    )
    scores_with_sink = torch.cat((scores, sink_scores), dim=1)
    probs_with_sink = torch.softmax(scores_with_sink, dim=1)
    return probs_with_sink[:, :-1], torch.logsumexp(scores_with_sink, dim=1)


def _apply_attention_mask(
    scores: torch.Tensor,
    *,
    q_len: int,
    kv_len: int,
    is_causal: bool,
    window_left: int,
) -> torch.Tensor:
    if window_left < -1:
        raise ValueError(f"window_left must be -1 or non-negative, got {window_left}")
    if not is_causal and window_left < 0:
        return scores

    q_idx = torch.arange(q_len, device=scores.device).view(-1, 1)
    k_idx = torch.arange(kv_len, device=scores.device).view(1, -1)
    offset = kv_len - q_len
    mask = torch.zeros((q_len, kv_len), dtype=torch.bool, device=scores.device)
    if is_causal:
        mask |= k_idx > q_idx + offset
    if window_left >= 0:
        mask |= k_idx < q_idx + offset - window_left
    return scores.masked_fill(mask.unsqueeze(-1), float("-inf"))


def _mha_attention_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor | None,
    is_causal: bool,
    window_left: int,
    logit_cap: float,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("MHA q, k, and v must be rank-3 tensors")
    if q.shape[-1] != k.shape[-1] or v.shape[-1] != q.shape[-1]:
        raise ValueError("MHA q, k, and v head dimensions must match")
    if k.shape[:2] != v.shape[:2]:
        raise ValueError("MHA k and v token/head dimensions must match")
    if softmax_scale <= 0.0:
        raise ValueError("softmax_scale must be positive")

    expanded_k = _expand_kv_heads_for_queries(
        k.float(),
        num_q_heads=q.shape[1],
        name="k",
    )
    expanded_v = _expand_kv_heads_for_queries(
        v.float(),
        num_q_heads=q.shape[1],
        name="v",
    )
    scores = torch.einsum("qhd,khd->qkh", q.float(), expanded_k) * softmax_scale
    scores = _apply_attention_logit_cap(scores, logit_cap)
    scores = _apply_attention_mask(
        scores,
        q_len=q.shape[0],
        kv_len=k.shape[0],
        is_causal=is_causal,
        window_left=window_left,
    )
    probs, lse = _attention_scores_with_optional_sink(scores, sinks)
    out = torch.einsum("qkh,khd->qhd", probs, expanded_v)
    return out.to(_attention_output_dtype(q.dtype)), lse.to(torch.float32)


def _gather_mha_cache_rows(
    cache: KVCacheValues,
    *,
    batch_idx: int,
    cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.k_cache is None or cache.v_cache is None:
        raise ValueError("MHA cache reference requires k_cache and v_cache")
    if cache_len <= 0:
        raise ValueError("MHA cached attention requires positive cache lengths")
    if cache.k_cache.shape != cache.v_cache.shape:
        raise ValueError("MHA k_cache and v_cache shapes must match")

    if cache.page_table_cpu is None:
        if cache.k_cache.ndim != 4:
            raise ValueError("dense MHA cache must have shape [batch, max_k, heads, d]")
        return (
            cache.k_cache[batch_idx, :cache_len],
            cache.v_cache[batch_idx, :cache_len],
        )

    if cache.k_cache.ndim != 4:
        raise ValueError("paged MHA cache must have shape [pages, page, heads, d]")
    page_size = cache.k_cache.shape[1]
    k_rows = []
    v_rows = []
    for pos in range(cache_len):
        page_col = pos // page_size
        page_offset = pos % page_size
        physical_page = cache.page_table_cpu[batch_idx][page_col]
        k_rows.append(cache.k_cache[physical_page, page_offset])
        v_rows.append(cache.v_cache[physical_page, page_offset])
    return torch.stack(k_rows, dim=0), torch.stack(v_rows, dim=0)


def _mha_prefill_reference(
    values: MHAInputValues,
    *,
    window_left: int,
    logit_cap: float,
) -> MHAReferenceValues:
    if values.q is None or values.k is None or values.v is None:
        raise ValueError("MHA prefill reference requires q, k, and v tensors")
    metadata = values.metadata
    if metadata.new_q_lens_cpu != metadata.new_kv_lens_cpu:
        raise ValueError("MHA prefill reference requires matching Q and KV lengths")
    q_offsets = metadata.cu_seqlens_q_cpu
    kv_offsets = metadata.cu_seqlens_q_cpu
    if q_offsets[0] != 0 or q_offsets[-1] != values.q.shape[0]:
        raise ValueError("MHA prefill q offsets must cover q")
    if kv_offsets[-1] != values.k.shape[0] or values.k.shape[0] != values.v.shape[0]:
        raise ValueError("MHA prefill kv offsets must cover k/v")

    outputs = []
    lses = []
    softmax_scale = 1.0 / math.sqrt(values.q.shape[-1])
    for request_idx in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[request_idx], q_offsets[request_idx + 1]
        kv_start, kv_end = kv_offsets[request_idx], kv_offsets[request_idx + 1]
        out_i, lse_i = _mha_attention_reference(
            q=values.q[q_start:q_end],
            k=values.k[kv_start:kv_end],
            v=values.v[kv_start:kv_end],
            sinks=values.sinks,
            is_causal=True,
            window_left=window_left,
            logit_cap=logit_cap,
            softmax_scale=softmax_scale,
        )
        outputs.append(out_i)
        lses.append(lse_i)
    return MHAReferenceValues(
        out=torch.cat(outputs, dim=0),
        lse=torch.cat(lses, dim=0).to(torch.float32),
    )


def _mha_cached_reference(
    values: MHAInputValues,
    *,
    is_causal: bool,
    window_left: int,
    logit_cap: float,
) -> MHAReferenceValues:
    if values.q is None or values.cache is None:
        raise ValueError("MHA cached reference requires q and cache tensors")
    metadata = values.metadata
    q_offsets = metadata.cu_seqlens_q_cpu
    if q_offsets[0] != 0 or q_offsets[-1] != values.q.shape[0]:
        raise ValueError("MHA cached q offsets must cover q")
    cache_lens = metadata.cache_seqlens.detach().cpu().tolist()

    outputs = []
    lses = []
    softmax_scale = 1.0 / math.sqrt(values.q.shape[-1])
    for batch_idx, cache_len_raw in enumerate(cache_lens):
        q_start, q_end = q_offsets[batch_idx], q_offsets[batch_idx + 1]
        k_i, v_i = _gather_mha_cache_rows(
            values.cache,
            batch_idx=batch_idx,
            cache_len=int(cache_len_raw),
        )
        out_i, lse_i = _mha_attention_reference(
            q=values.q[q_start:q_end],
            k=k_i,
            v=v_i,
            sinks=values.sinks,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            softmax_scale=softmax_scale,
        )
        outputs.append(out_i)
        lses.append(lse_i)
    return MHAReferenceValues(
        out=torch.cat(outputs, dim=0),
        lse=torch.cat(lses, dim=0).to(torch.float32),
    )


def mha_reference(
    values: MHAInputValues,
    *,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
) -> MHAReferenceValues:
    """Reference MHA output for generated prefill, extend, or decode inputs."""

    if values.cache is None:
        return _mha_prefill_reference(
            values,
            window_left=window_left,
            logit_cap=logit_cap,
        )
    return _mha_cached_reference(
        values,
        is_causal=is_causal,
        window_left=window_left,
        logit_cap=logit_cap,
    )
