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
    CacheLayout,
    DeviceLike,
    MHARequestMetadataInput,
    MHARequestMetadataInputConfig,
    MHARequestMetadataValues,
    NumericsInputGenerator,
    PageTableIndexing,
    PageTableInput,
    PageTableInputConfig,
    PageTableValues,
    TensorInput,
    _CSA_HEAD_DIM,
    _CSA_INDEXER_DIM,
    _CSA_ROPE_DIM,
    _CSA_SPARSE_PREFILL_TOPK_ALIGNMENT,
    _align_up,
    _check_float_dtype,
    _check_matches,
    _check_nonnegative,
    _check_page_table_indexing,
    _check_positive,
    _child_seed,
    _generate_random_positions,
    _generate_slot_mapping_tensor,
    _metadata_config,
    _require_tensor,
    _resolve_attention_seeds,
    _resolve_device,
    build_rope_cos_sin_cache,
    dataclass,
    math,
    torch,
)


@dataclass
class CompressedSequenceAttentionSlidingWindowValues:
    """Generated values for the always-present sliding-window attention portion."""

    q: torch.Tensor
    attn_sink: torch.Tensor
    metadata: MHARequestMetadataValues
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    seq_lens: torch.Tensor
    page_table: torch.Tensor
    page_table_values: PageTableValues
    block_size: int
    window_size: int


@dataclass
class CompressedSequenceAttentionHistoryConfig:
    """Configuration for optional compressed-history attention inputs.

    ``compress_ratio=4`` models CSA-style compressed sequence attention and
    defaults to overlapping compressor state. ``compress_ratio=128`` models
    HCA-style compressed history and defaults to non-overlapping compressor
    state.
    """

    # Required: number of original token positions represented by each
    # compressed-history row. The currently supported layouts use 4 for CSA or
    # 128 for HCA.
    compress_ratio: int

    # Optional: number of compressed-prefix candidates to generate per token.
    topk: int = 4

    # Optional: whether compressor state is the overlapping CSA layout. Defaults
    # from compress_ratio when omitted.
    overlap: bool | None = None

    # Optional: generated compressor-state storage size.
    num_state_cache_blocks: int = 4
    compressor_block_size: int | None = None

    # Optional: generate negative slot mappings for skipped rows.
    negative_compressor_slot_count: int = 0

    # Optional: generated block-table/base-offset and validity metadata.
    include_block_table_base_offsets: bool = False
    include_valid_token_mask: bool = False
    invalid_token_probability: float = 0.25

    # Optional: generated top-k ordering policy for sparse prefill.
    topk_indexing: PageTableIndexing = "random"



@dataclass
class CompressedSequenceAttentionIndexerConfig:
    """Configuration for optional CSA indexer inputs."""

    # Optional: number of indexer heads used for indexer-Q/weight generation.
    num_heads: int = 2

    # Optional: softmax/head scales used by the indexer query packing helper.
    softmax_scale: float = 1.0
    head_scale: float = 1.0

    # Optional: RoPE frequency base used for generated indexer query metadata.
    rope_base: float = 10000.0


@dataclass
class CompressedSequenceAttentionHistoryValues:
    """Generated optional compressed-history values."""

    paged_index: CSAPagedIndexValues
    sparse_prefill_index: CSASparsePrefillIndexValues
    compressor_state: CSACompressorStateValues


@dataclass
class CompressedSequenceAttentionIndexerValues:
    """Generated optional CSA indexer values."""

    query: CSAIndexerQueryValues


@dataclass
class CompressedSequenceAttentionInputValues:
    """Generated values for compressed sequence attention.

    ``sliding_window`` is always present. ``compressed`` is present for CSA/HCA
    layers. ``indexer`` is present only for CSA-style sparse compressed
    attention.
    """

    kind: str
    sliding_window: CompressedSequenceAttentionSlidingWindowValues
    compressed: CompressedSequenceAttentionHistoryValues | None
    indexer: CompressedSequenceAttentionIndexerValues | None


@dataclass
class CompressedSequenceAttentionInputConfig:
    """Initialization parameters for compressed sequence attention.

    The represented operation is sliding-window attention plus optional
    compressed-history attention and optional CSA indexer selection. The current
    implementation supports CSA/HCA layouts; TokenSpeed helper kernels should
    derive narrower argument bundles outside this generator package.
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

    # Required: generated floating dtype for Q and compressor-state values.
    dtype: torch.dtype

    # Required: number of local Q heads in generated attention queries.
    num_q_heads: int

    # Required: page size used for generated paged caches.
    page_size: int

    # Required: number of recent tokens included in the sliding-window portion.
    window_size: int

    # ------------------------------------------------------------------
    # Optional attention-shape and component configuration.
    # ------------------------------------------------------------------

    # Optional: attention head width. The current CSA/HCA references require
    # the model's fixed 512-wide layout.
    head_dim: int = _CSA_HEAD_DIM

    # Optional: RoPE width. The current CSA/HCA references require 64.
    rope_dim: int = _CSA_ROPE_DIM

    # Optional: compressed-history component. ``None`` generates SWA-only.
    compressed: CompressedSequenceAttentionHistoryConfig | None = None

    # Optional: CSA indexer component. Requires compressed.compress_ratio == 4.
    indexer: CompressedSequenceAttentionIndexerConfig | None = None

    # Optional: metadata/page-table controls shared by generated components.
    indexing: PageTableIndexing = "random"
    metadata_input: MHARequestMetadataInputConfig | None = None
    swa_page_table_input: PageTableInputConfig | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None

class CSAInputs(NumericsInputGenerator):
    """Generator for compressed sequence attention inputs."""

    config: CompressedSequenceAttentionInputConfig
    q_input: TensorInput | None
    metadata_input: MHARequestMetadataInput | None
    swa_page_table_input: PageTableInput | None

    def __init__(self, config: CompressedSequenceAttentionInputConfig) -> None:
        self.config = config
        self.q_input = None
        self.metadata_input = None
        self.swa_page_table_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.q_input = self.q_input or TensorInput(
            (
                self.config.total_new_q_tokens,
                self.config.num_q_heads,
                self.config.head_dim,
            ),
            self.config.dtype,
            device=self.config.device,
        )
        self.metadata_input = self.metadata_input or MHARequestMetadataInput(
            self.config.metadata_input or self._make_metadata_config()
        )
        self._verify_metadata_config_matches_parent()
        self.config.metadata_input = self.metadata_input.config
        self.swa_page_table_input = self.swa_page_table_input or PageTableInput(
            self.config.swa_page_table_input or self._make_swa_page_table_config()
        )
        self._verify_swa_page_table_config_matches_parent()
        self.config.swa_page_table_input = self.swa_page_table_input.config

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> CompressedSequenceAttentionInputValues:
        self.__post_init__()
        if (
            self.q_input is None
            or self.metadata_input is None
            or self.swa_page_table_input is None
        ):
            raise ValueError(
                "compressed-sequence-attention children must be initialized"
            )
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        metadata = self.metadata_input.generate(
            seed=_child_seed(metadata_seed, 1),
            device=target_device,
        )
        if metadata.new_q_lens_cpu != metadata.new_kv_lens_cpu:
            raise ValueError("compressed sequence attention requires tied Q/KV lengths")

        self.q_input.shape = (
            self.config.total_new_q_tokens,
            self.config.num_q_heads,
            self.config.head_dim,
        )
        self.q_input.dtype = self.config.dtype
        q = _require_tensor(
            self.q_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "q",
        )

        positions_cpu, token_to_req_cpu = self._positions_from_metadata(metadata)
        required_pages = max(
            1,
            math.ceil(max(metadata.visible_kv_lens_cpu) / self.config.page_size),
        )
        self.swa_page_table_input.config.batch_size = self.config.batch_size
        self.swa_page_table_input.config.max_pages_per_request = max(
            self.swa_page_table_input.config.max_pages_per_request,
            required_pages,
        )
        self.swa_page_table_input.config.indexing = self.config.indexing
        page_table = self.swa_page_table_input.generate(
            seed=_child_seed(metadata_seed, 2),
            device=target_device,
        )
        sliding = CompressedSequenceAttentionSlidingWindowValues(
            q=q,
            attn_sink=torch.zeros(
                (self.config.num_q_heads,),
                dtype=torch.float32,
                device=target_device,
            ),
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
            seq_lens=torch.tensor(
                metadata.visible_kv_lens_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            page_table=page_table.page_table,
            page_table_values=page_table,
            block_size=self.config.page_size,
            window_size=self.config.window_size,
        )

        compressed = self._generate_compressed_values(
            metadata_seed=metadata_seed,
            value_seed=value_seed,
            device=target_device,
        )
        indexer = self._generate_indexer_values(
            metadata_seed=metadata_seed,
            value_seed=value_seed,
            device=target_device,
        )
        kind: str
        if self.config.compressed is None:
            kind = "swa"
        elif self.config.compressed.compress_ratio == 4:
            kind = "csa"
        else:
            kind = "hca"
        values = CompressedSequenceAttentionInputValues(
            kind=kind,
            sliding_window=sliding,
            compressed=compressed,
            indexer=indexer,
        )
        self._validate_values(values)
        return values

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
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.num_q_heads = _check_positive(
            "num_q_heads",
            self.config.num_q_heads,
        )
        self.config.page_size = _check_positive("page_size", self.config.page_size)
        self.config.window_size = _check_positive(
            "window_size",
            self.config.window_size,
        )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.rope_dim = _check_positive("rope_dim", self.config.rope_dim)
        if self.config.head_dim != _CSA_HEAD_DIM:
            raise ValueError(
                "compressed sequence attention currently requires "
                f"current CSA/HCA head_dim={_CSA_HEAD_DIM}"
            )
        if self.config.rope_dim != _CSA_ROPE_DIM:
            raise ValueError(
                "compressed sequence attention currently requires "
                f"current CSA/HCA rope_dim={_CSA_ROPE_DIM}"
            )
        self.config.indexing = _check_page_table_indexing(self.config.indexing)
        if self.config.compressed is None:
            if self.config.indexer is not None:
                raise ValueError("indexer requires compressed attention")
            return
        compressed = self.config.compressed
        compressed.compress_ratio = _check_positive(
            "compressed.compress_ratio",
            compressed.compress_ratio,
        )
        if compressed.compress_ratio not in (4, 128):
            raise ValueError(
                "compressed sequence attention currently supports ratios 4 or 128"
            )
        compressed.topk = _check_positive("compressed.topk", compressed.topk)
        if compressed.overlap is None:
            compressed.overlap = compressed.compress_ratio == 4
        if compressed.compress_ratio == 4 and not compressed.overlap:
            raise ValueError("CSA compressed attention requires overlap=True")
        if compressed.compress_ratio == 128 and compressed.overlap:
            raise ValueError("HCA compressed attention requires overlap=False")
        compressed.num_state_cache_blocks = _check_positive(
            "compressed.num_state_cache_blocks",
            compressed.num_state_cache_blocks,
        )
        compressed.compressor_block_size = _check_positive(
            "compressed.compressor_block_size",
            compressed.compressor_block_size
            if compressed.compressor_block_size is not None
            else (4 if compressed.compress_ratio == 4 else 8),
        )
        compressed.negative_compressor_slot_count = _check_nonnegative(
            "compressed.negative_compressor_slot_count",
            compressed.negative_compressor_slot_count,
        )
        if compressed.negative_compressor_slot_count > self.config.total_new_q_tokens:
            raise ValueError(
                "compressed.negative_compressor_slot_count must be <= total_new_q_tokens"
            )
        compressed.topk_indexing = _check_page_table_indexing(compressed.topk_indexing)
        compressed.invalid_token_probability = float(compressed.invalid_token_probability)
        if not 0.0 <= compressed.invalid_token_probability < 1.0:
            raise ValueError("compressed.invalid_token_probability must be in [0, 1)")
        if self.config.indexer is None:
            return
        if compressed.compress_ratio != 4:
            raise ValueError("indexer inputs are only valid for CSA compress_ratio=4")
        indexer = self.config.indexer
        indexer.num_heads = _check_positive("indexer.num_heads", indexer.num_heads)
        indexer.rope_base = float(indexer.rope_base)
        if indexer.rope_base <= 0.0 or not math.isfinite(indexer.rope_base):
            raise ValueError("indexer.rope_base must be finite and positive")

    def _make_metadata_config(self) -> MHARequestMetadataInputConfig:
        return MHARequestMetadataInputConfig(
            batch_size=self.config.batch_size,
            total_cached_tokens=self.config.total_cached_tokens,
            total_new_q_tokens=self.config.total_new_q_tokens,
            cache_layout="paged",
        )

    def _verify_metadata_config_matches_parent(self) -> None:
        config = _metadata_config(self.metadata_input)
        for name, parent_value, child_value in (
            ("batch_size", self.config.batch_size, config.batch_size),
            (
                "total_cached_tokens",
                self.config.total_cached_tokens,
                config.total_cached_tokens,
            ),
            (
                "total_new_q_tokens",
                self.config.total_new_q_tokens,
                config.total_new_q_tokens,
            ),
        ):
            _check_matches(
                parent_name=f"CompressedSequenceAttentionInputConfig.{name}",
                child_name=f"metadata_input.{name}",
                parent_value=parent_value,
                child_value=child_value,
            )
        if config.total_new_kv_tokens != self.config.total_new_q_tokens:
            raise ValueError("metadata_input.total_new_kv_tokens must match total_new_q_tokens")
        if not config.tie_new_kv_to_query:
            raise ValueError("metadata_input.tie_new_kv_to_query must be true")

    def _make_swa_page_table_config(self) -> PageTableInputConfig:
        max_tokens = (
            self.config.total_cached_tokens + self.config.total_new_q_tokens
        )
        return PageTableInputConfig(
            batch_size=self.config.batch_size,
            max_pages_per_request=max(1, math.ceil(max_tokens / self.config.page_size)),
            indexing=self.config.indexing,
            device=self.config.device,
        )

    def _verify_swa_page_table_config_matches_parent(self) -> None:
        if self.swa_page_table_input is None:
            raise ValueError("swa_page_table_input must be initialized")
        _check_matches(
            parent_name="CompressedSequenceAttentionInputConfig.batch_size",
            child_name="swa_page_table_input.batch_size",
            parent_value=self.config.batch_size,
            child_value=self.swa_page_table_input.config.batch_size,
        )

    def _positions_from_metadata(
        self,
        metadata: MHARequestMetadataValues,
    ) -> tuple[list[int], list[int]]:
        positions: list[int] = []
        token_to_req: list[int] = []
        for req, (query_len, seq_len) in enumerate(
            zip(metadata.new_q_lens_cpu, metadata.visible_kv_lens_cpu, strict=True)
        ):
            start_pos = seq_len - query_len
            for offset in range(query_len):
                positions.append(start_pos + offset)
                token_to_req.append(req)
        return positions, token_to_req

    def _generate_compressed_values(
        self,
        *,
        metadata_seed: int,
        value_seed: int,
        device: torch.device,
    ) -> CompressedSequenceAttentionHistoryValues | None:
        compressed = self.config.compressed
        if compressed is None:
            return None
        metadata_config = self._shared_metadata_config(cache_layout="paged")
        dense_metadata_config = self._shared_metadata_config(cache_layout="dense")
        paged_index = _CSAPagedIndexBuilder(
            _CSAPagedIndexConfig(
                batch_size=self.config.batch_size,
                total_cached_tokens=self.config.total_cached_tokens,
                total_new_q_tokens=self.config.total_new_q_tokens,
                block_size=self.config.page_size,
                compress_ratio=compressed.compress_ratio,
                window_size=self.config.window_size,
                topk=compressed.topk,
                indexing=self.config.indexing,
                include_block_table_base_offsets=compressed.include_block_table_base_offsets,
                include_valid_token_mask=compressed.include_valid_token_mask,
                invalid_token_probability=compressed.invalid_token_probability,
                metadata_input=metadata_config,
                page_table_input=self.config.swa_page_table_input,
                device=self.config.device,
            )
        ).generate(seed=metadata_seed, device=device)
        sparse_prefill_index = _CSASparsePrefillIndexBuilder(
            _CSASparsePrefillIndexConfig(
                batch_size=self.config.batch_size,
                total_cached_tokens=self.config.total_cached_tokens,
                total_new_q_tokens=self.config.total_new_q_tokens,
                topk=compressed.topk,
                window_size=self.config.window_size,
                compress_ratio=compressed.compress_ratio,
                topk_indexing=compressed.topk_indexing,
                metadata_input=dense_metadata_config,
                device=self.config.device,
            )
        ).generate(seed=metadata_seed, device=device)
        compressor_state = _CSACompressorStateBuilder(
            _CSACompressorStateConfig(
                num_tokens=self.config.total_new_q_tokens,
                state_width=_CSA_HEAD_DIM * (2 if compressed.overlap else 1),
                num_cache_blocks=compressed.num_state_cache_blocks,
                block_size=compressed.compressor_block_size,
                compress_ratio=compressed.compress_ratio,
                dtype=self.config.dtype,
                invalid_token_count=compressed.negative_compressor_slot_count,
                device=self.config.device,
            )
        ).generate(
            metadata_seed=_child_seed(metadata_seed, 12),
            value_seed=_child_seed(value_seed, 12),
            device=device,
        )
        return CompressedSequenceAttentionHistoryValues(
            paged_index=paged_index,
            sparse_prefill_index=sparse_prefill_index,
            compressor_state=compressor_state,
        )

    def _generate_indexer_values(
        self,
        *,
        metadata_seed: int,
        value_seed: int,
        device: torch.device,
    ) -> CompressedSequenceAttentionIndexerValues | None:
        compressed = self.config.compressed
        indexer = self.config.indexer
        if compressed is None or indexer is None:
            return None
        max_seq_len = max(1, self.config.total_cached_tokens + self.config.total_new_q_tokens)
        q_rope = _CSAIndexerQueryBuilder(
            _CSAIndexerQueryConfig(
                num_tokens=self.config.total_new_q_tokens,
                num_heads=indexer.num_heads,
                dtype=self.config.dtype,
                max_position=max_seq_len,
                softmax_scale=indexer.softmax_scale,
                head_scale=indexer.head_scale,
                rope_base=indexer.rope_base,
                device=self.config.device,
            )
        ).generate(
            metadata_seed=_child_seed(metadata_seed, 20),
            value_seed=_child_seed(value_seed, 20),
            device=device,
        )
        return CompressedSequenceAttentionIndexerValues(query=q_rope)

    def _shared_metadata_config(self, *, cache_layout: CacheLayout) -> MHARequestMetadataInputConfig:
        parent = self.config.metadata_input
        if parent is None:
            return MHARequestMetadataInputConfig(
                batch_size=self.config.batch_size,
                total_cached_tokens=self.config.total_cached_tokens,
                total_new_q_tokens=self.config.total_new_q_tokens,
                cache_layout=cache_layout,
            )
        return MHARequestMetadataInputConfig(
            batch_size=parent.batch_size,
            total_cached_tokens=parent.total_cached_tokens,
            total_new_q_tokens=parent.total_new_q_tokens,
            total_new_kv_tokens=parent.total_new_kv_tokens,
            cache_layout=cache_layout,
            cached_length_mode=parent.cached_length_mode,
            new_q_length_mode=parent.new_q_length_mode,
            new_kv_length_mode=parent.new_kv_length_mode,
            max_cached_tokens_per_request=parent.max_cached_tokens_per_request,
            max_new_q_tokens_per_request=parent.max_new_q_tokens_per_request,
            max_new_kv_tokens_per_request=parent.max_new_kv_tokens_per_request,
            tie_new_kv_to_query=parent.tie_new_kv_to_query,
            allow_untied_non_cached_kv=parent.allow_untied_non_cached_kv,
            max_seqlen_k=parent.max_seqlen_k,
            device=parent.device,
        )

    def _validate_values(self, values: CompressedSequenceAttentionInputValues) -> None:
        sliding = values.sliding_window
        if sliding.q.shape != (
            self.config.total_new_q_tokens,
            self.config.num_q_heads,
            self.config.head_dim,
        ):
            raise ValueError("sliding_window.q shape does not match config")
        if sliding.positions.shape != (self.config.total_new_q_tokens,):
            raise ValueError("sliding_window.positions must have one row per query")
        if sliding.token_to_req_indices.shape != (self.config.total_new_q_tokens,):
            raise ValueError("sliding_window.token_to_req_indices must have one row per query")
        if sliding.seq_lens.shape != (self.config.batch_size,):
            raise ValueError("sliding_window.seq_lens must have one row per request")
        if sliding.page_table.shape[0] != self.config.batch_size:
            raise ValueError("sliding_window.page_table must have one row per request")
        if self.config.compressed is None:
            if values.compressed is not None or values.indexer is not None:
                raise ValueError("SWA-only values must not include compressed/indexer data")
            return
        if values.compressed is None:
            raise ValueError("compressed config must generate compressed values")
        if self.config.indexer is None:
            if values.indexer is not None:
                raise ValueError("indexer values require indexer config")
        elif values.indexer is None:
            raise ValueError("indexer config must generate indexer values")


@dataclass
class CSACompressorStateValues:
    """Generated values for saving CSA compressor state rows."""

    kv: torch.Tensor
    score: torch.Tensor
    ape: torch.Tensor
    state_cache: torch.Tensor
    slot_mapping: torch.Tensor
    positions: torch.Tensor
    block_size: int
    compress_ratio: int


@dataclass
class _CSACompressorStateConfig:
    """Initialization parameters for CSA compressor-state writes.

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
class _CSACompressorStateBuilder:
    """Generator for CSA compressor-state save inputs."""

    config: _CSACompressorStateConfig
    kv_input: TensorInput | None
    score_input: TensorInput | None
    ape_input: TensorInput | None
    state_cache_input: TensorInput | None

    def __init__(self, config: _CSACompressorStateConfig) -> None:
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
    ) -> CSACompressorStateValues:
        self.__post_init__()
        if (
            self.kv_input is None
            or self.score_input is None
            or self.ape_input is None
            or self.state_cache_input is None
        ):
            raise ValueError(
                "_CSACompressorStateBuilder child generators must be initialized"
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
        values = CSACompressorStateValues(
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
        _validate_csa_compressor_state_values(values)
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
        return _generate_slot_mapping_tensor(
            num_rows=self.config.num_tokens,
            total_slots=self.config.num_cache_blocks * self.config.block_size,
            negative_count=self.config.invalid_token_count,
            seed=seed,
            device=device,
        )


def _validate_csa_compressor_state_values(
    values: CSACompressorStateValues,
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


def _csa_compressor_state_ape_row(
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


def csa_save_compressor_state_reference(
    values: CSACompressorStateValues,
) -> torch.Tensor:
    """Return state cache after applying CSA compressor-state writes."""

    _validate_csa_compressor_state_values(values)
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
        ape = _csa_compressor_state_ape_row(
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
class CSAIndexerQueryValues:
    """Generated values for CSA indexer query rows and metadata."""

    index_q: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    weights: torch.Tensor
    softmax_scale: float
    head_scale: float


@dataclass
class _CSAIndexerQueryConfig:
    """Initialization parameters for CSA indexer query values.

    The generated values are the operation-level query rows, position metadata,
    RoPE cache, and per-token/per-head weights used by the CSA indexer. Backend
    adapters can apply implementation-specific transforms or packing outside
    the input generator package.
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
class _CSAIndexerQueryBuilder:
    """Generator for CSA indexer query inputs."""

    config: _CSAIndexerQueryConfig
    index_q_input: TensorInput | None
    weights_input: TensorInput | None

    def __init__(
        self,
        config: _CSAIndexerQueryConfig,
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
    ) -> CSAIndexerQueryValues:
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
        values = CSAIndexerQueryValues(
            index_q=(index_q.float() * self.config.value_scale)
            .to(index_q.dtype)
            .contiguous(),
            positions=self._generate_positions(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            cos_sin_cache=build_rope_cos_sin_cache(
                rotary_dim=_CSA_ROPE_DIM,
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
        _validate_csa_indexer_query_values(values)
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
            _CSA_INDEXER_DIM,
        )

    def _weights_shape(self) -> tuple[int, int]:
        return (self.config.num_tokens, self.config.num_heads)

    def _generate_positions(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_random_positions(
            num_tokens=self.config.num_tokens,
            max_position=self.config.max_position,
            dtype=self.config.position_dtype,
            seed=seed,
            device=device,
        )


def _validate_csa_indexer_query_values(
    values: CSAIndexerQueryValues,
) -> None:
    if values.index_q.ndim != 3:
        raise ValueError(f"index_q must be rank-3, got {values.index_q.ndim}")
    if values.index_q.shape[-1] != _CSA_INDEXER_DIM:
        raise ValueError(
            f"index_q width must be {_CSA_INDEXER_DIM}, "
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
    if values.cos_sin_cache.shape[1] != _CSA_ROPE_DIM:
        raise ValueError(
            f"cos_sin_cache width must be {_CSA_ROPE_DIM}, "
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


@dataclass
class CSAPagedIndexValues:
    """Generated values for CSA paged index metadata operations."""

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
class _CSAPagedIndexConfig:
    """Initialization parameters for CSA paged index metadata.

    The represented operation family maps request-local token or compressed
    token positions through a paged KV cache block table. The generated values
    are shared by CSA helpers that build global top-k slot ids, decode
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
class _CSAPagedIndexBuilder:
    """Generator for CSA paged index metadata inputs."""

    config: _CSAPagedIndexConfig
    metadata_input: MHARequestMetadataInput | None
    page_table_input: PageTableInput | None

    def __init__(self, config: _CSAPagedIndexConfig) -> None:
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
    ) -> CSAPagedIndexValues:
        self.__post_init__()
        if self.metadata_input is None or self.page_table_input is None:
            raise ValueError("metadata/page-table child generators must be initialized")
        target_device = _resolve_device(self.config.device, device)
        metadata = self.metadata_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        )
        if metadata.new_q_lens_cpu != metadata.new_kv_lens_cpu:
            raise ValueError("CSA paged indices require tied Q/KV lengths")

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

        return CSAPagedIndexValues(
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
                parent_name=f"_CSAPagedIndexConfig.{name}",
                child_name=f"metadata_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(metadata_config, name),
            )
        if metadata_config.total_new_kv_tokens is not None:
            _check_matches(
                parent_name="_CSAPagedIndexConfig.total_new_q_tokens",
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
            parent_name="_CSAPagedIndexConfig.batch_size",
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


def _validate_csa_paged_index_values(
    values: CSAPagedIndexValues,
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


def csa_compute_global_topk_indices_and_lens_reference(
    values: CSAPagedIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global KV slot ids for request-local top-k indices."""

    _validate_csa_paged_index_values(values)
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


def csa_decode_swa_indices_and_lens_reference(
    values: CSAPagedIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decode sliding-window KV slot ids and row lengths."""

    _validate_csa_paged_index_values(values)
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


def csa_compressed_slot_mapping_reference(
    values: CSAPagedIndexValues,
) -> torch.Tensor:
    """Return compressed KV slot ids for newly materialized compressed tokens."""

    _validate_csa_paged_index_values(values)
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


def csa_indexer_decode_metadata_reference(
    values: CSAPagedIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decode-indexer context lengths and block tables."""

    _validate_csa_paged_index_values(values)
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
class CSASparsePrefillIndexValues:
    """Generated values for CSA sparse-prefill index construction.

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
class _CSASparsePrefillIndexConfig:
    """Initialization parameters for CSA sparse-prefill indices.

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
    # compressed prefix position. CSA sparse prefill uses a compressed
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
class _CSASparsePrefillIndexBuilder:
    """Generator for CSA sparse-prefill index-construction inputs."""

    config: _CSASparsePrefillIndexConfig
    metadata_input: MHARequestMetadataInput | None

    def __init__(self, config: _CSASparsePrefillIndexConfig) -> None:
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
    ) -> CSASparsePrefillIndexValues:
        self.__post_init__()
        if self.metadata_input is None:
            raise ValueError("metadata_input must be initialized")
        target_device = _resolve_device(self.config.device, device)
        metadata = self.metadata_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        )
        if metadata.new_q_lens_cpu != metadata.new_kv_lens_cpu:
            raise ValueError("CSA sparse prefill requires tied Q/KV lengths")

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

        return CSASparsePrefillIndexValues(
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
                parent_name=f"_CSASparsePrefillIndexConfig.{name}",
                child_name=f"metadata_input.{name}",
                parent_value=getattr(self.config, name),
                child_value=getattr(metadata_config, name),
            )
        if metadata_config.total_new_kv_tokens is not None:
            _check_matches(
                parent_name=(
                    "_CSASparsePrefillIndexConfig.total_new_q_tokens"
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


def _validate_csa_sparse_prefill_values(
    values: CSASparsePrefillIndexValues,
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


def csa_build_dense_prefill_local_compressed_indices_reference(
    values: CSASparsePrefillIndexValues,
    *,
    width: int | None = None,
) -> torch.Tensor:
    """Return per-token local compressed-prefix indices for dense prefill."""

    _validate_csa_sparse_prefill_values(values)
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


def csa_combine_topk_swa_indices_reference(
    values: CSASparsePrefillIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse-prefill candidate indices from top-k prefix plus SWA."""

    _validate_csa_sparse_prefill_values(values)
    device = values.topk_indices.device
    num_tokens = int(values.topk_indices.shape[0])
    combined_topk = _align_up(
        values.topk + values.window_size,
        _CSA_SPARSE_PREFILL_TOPK_ALIGNMENT,
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


def csa_combine_dense_swa_indices_reference(
    values: CSASparsePrefillIndexValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse-prefill candidate indices from dense prefix plus SWA."""

    _validate_csa_sparse_prefill_values(values)
    device = values.positions.device
    num_tokens = int(values.positions.numel())
    combined_topk = _align_up(
        max(values.compressed_base + values.window_size, 1),
        _CSA_SPARSE_PREFILL_TOPK_ALIGNMENT,
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


CSAHistoryConfig = CompressedSequenceAttentionHistoryConfig
CSAHistoryValues = CompressedSequenceAttentionHistoryValues
CSAIndexerConfig = CompressedSequenceAttentionIndexerConfig
CSAIndexerValues = CompressedSequenceAttentionIndexerValues
CSAInputConfig = CompressedSequenceAttentionInputConfig
CSAInputValues = CompressedSequenceAttentionInputValues
CSASlidingWindowValues = CompressedSequenceAttentionSlidingWindowValues
