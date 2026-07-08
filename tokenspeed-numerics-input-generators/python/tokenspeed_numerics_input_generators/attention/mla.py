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
    MHARequestMetadataInput,
    MHARequestMetadataInputConfig,
    MHARequestMetadataValues,
    MLAKVCacheInput,
    MLAKVCacheInputConfig,
    MLAKVCacheValues,
    NumericsInputGenerator,
    PageTableIndexing,
    PageTableInputConfig,
    TensorInput,
    _check_cache_layout,
    _check_matches,
    _check_nonnegative,
    _check_page_table_indexing,
    _check_positive,
    _metadata_config,
    _resolve_attention_seeds,
    _resolve_device,
    attention_generate,
    dataclass,
    math,
    torch,
)

from .mha import _apply_attention_logit_cap, _attention_output_dtype, _expand_kv_heads_for_queries


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
class MLAReferenceValues:
    """Reference output tensors for ``MLAInputValues``."""

    out: torch.Tensor
    lse: torch.Tensor


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



def _mla_prefill_reference(
    values: MLAInputValues,
    *,
    is_causal: bool,
    logit_cap: float,
) -> MLAReferenceValues:
    if values.q is None or values.k is None or values.v is None:
        raise ValueError("MLA prefill reference requires q, k, and v tensors")
    q = values.q
    k = values.k
    v = values.v
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("MLA prefill q, k, and v must be rank-3 tensors")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("MLA prefill q and k head dimensions must match")
    if k.shape[:2] != v.shape[:2]:
        raise ValueError("MLA prefill k and v token/head dimensions must match")
    if values.softmax_scale <= 0.0:
        raise ValueError("softmax_scale must be positive")
    metadata = values.metadata
    q_offsets = metadata.cu_seqlens_q_cpu
    kv_offsets = metadata.cu_seqlens_kv_cpu
    if len(q_offsets) != len(kv_offsets):
        raise ValueError("MLA prefill q and kv offsets must have the same length")
    if q_offsets[0] != 0 or kv_offsets[0] != 0:
        raise ValueError("MLA prefill offsets must start at zero")
    if q_offsets[-1] != q.shape[0]:
        raise ValueError("MLA prefill q offsets must end at q token count")
    if kv_offsets[-1] != k.shape[0]:
        raise ValueError("MLA prefill kv offsets must end at k token count")

    outputs = []
    lses = []
    for request_idx in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[request_idx], q_offsets[request_idx + 1]
        kv_start, kv_end = kv_offsets[request_idx], kv_offsets[request_idx + 1]
        cur_s_q = q_end - q_start
        cur_s_kv = kv_end - kv_start
        if cur_s_q <= 0 or cur_s_kv <= 0:
            raise ValueError("MLA prefill requests must have non-empty q and kv")
        if is_causal and cur_s_kv < cur_s_q:
            raise ValueError("causal MLA prefill requires kv length >= q length")

        q_i = q[q_start:q_end].float()
        k_i = _expand_kv_heads_for_queries(
            k[kv_start:kv_end].float(),
            num_q_heads=q.shape[1],
            name="k",
        )
        v_i = _expand_kv_heads_for_queries(
            v[kv_start:kv_end].float(),
            num_q_heads=q.shape[1],
            name="v",
        )
        scores = torch.einsum("qhd,khd->qkh", q_i, k_i) * values.softmax_scale
        scores = _apply_attention_logit_cap(scores, logit_cap)
        if is_causal:
            q_idx = torch.arange(cur_s_q, device=q.device).view(-1, 1)
            k_idx = torch.arange(cur_s_kv, device=q.device).view(1, -1)
            offset = cur_s_kv - cur_s_q
            mask = k_idx > q_idx + offset
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
        probs = torch.softmax(scores, dim=1)
        outputs.append(torch.einsum("qkh,khd->qhd", probs, v_i))
        lses.append(torch.logsumexp(scores, dim=1))
    return MLAReferenceValues(
        out=torch.cat(outputs, dim=0).to(_attention_output_dtype(q.dtype)),
        lse=torch.cat(lses, dim=0).to(torch.float32),
    )


def _gather_mla_paged_cache_rows(
    cache: MLAKVCacheValues,
    *,
    batch_idx: int,
    cache_len: int,
) -> torch.Tensor:
    if cache.kv_cache.ndim != 4 or cache.kv_cache.shape[2] != 1:
        raise ValueError("paged MLA cache must have shape [pages, page, 1, dim]")
    if cache.page_table_cpu is None:
        raise ValueError("paged MLA cache reference requires page_table_cpu")
    if cache_len <= 0:
        raise ValueError("MLA decode cache lengths must be positive")
    page_size = cache.kv_cache.shape[1]
    rows = []
    for pos in range(cache_len):
        page_col = pos // page_size
        page_offset = pos % page_size
        physical_page = cache.page_table_cpu[batch_idx][page_col]
        rows.append(cache.kv_cache[physical_page, page_offset, 0])
    return torch.stack(rows, dim=0)


def _mla_decode_reference(
    values: MLAInputValues,
    *,
    logit_cap: float,
) -> MLAReferenceValues:
    if values.q is None or values.cache is None:
        raise ValueError("MLA decode reference requires q and cache tensors")
    q = values.q
    cache = values.cache
    if q.ndim != 4:
        raise ValueError(f"MLA decode q must be rank-4, got {q.ndim}")
    if cache.kv_cache is None:
        raise ValueError("MLA decode reference requires kv_cache")
    if values.kv_lora_rank + values.qk_rope_head_dim != q.shape[-1]:
        raise ValueError("MLA decode q last dimension does not match MLA ranks")
    if values.kv_lora_rank <= 0:
        raise ValueError("kv_lora_rank must be positive")

    outputs = []
    lses = []
    cache_lens = values.metadata.cache_seqlens.detach().cpu().tolist()
    for batch_idx, cache_len_raw in enumerate(cache_lens):
        cache_len = int(cache_len_raw)
        kv = _gather_mla_paged_cache_rows(
            cache,
            batch_idx=batch_idx,
            cache_len=cache_len,
        ).float()
        q_i = q[batch_idx].float()
        scores = torch.einsum("qhd,kd->qhk", q_i, kv) * values.softmax_scale
        scores = _apply_attention_logit_cap(scores, logit_cap)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qhk,kr->qhr", probs, kv[:, : values.kv_lora_rank]))
        lses.append(torch.logsumexp(scores, dim=-1))
    return MLAReferenceValues(
        out=torch.stack(outputs, dim=0).to(_attention_output_dtype(q.dtype)),
        lse=torch.stack(lses, dim=0).to(torch.float32),
    )


def mla_reference(
    values: MLAInputValues,
    *,
    is_causal: bool = True,
    logit_cap: float = 0.0,
) -> MLAReferenceValues:
    """Reference MLA output for generated prefill or paged decode inputs."""

    if values.cache is None:
        return _mla_prefill_reference(
            values,
            is_causal=is_causal,
            logit_cap=logit_cap,
        )
    if values.cache.page_table is None:
        raise ValueError("MLA cache reference currently requires paged cache inputs")
    return _mla_decode_reference(values, logit_cap=logit_cap)
