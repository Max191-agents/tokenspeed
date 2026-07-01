# Attention Input Generators

Attention generators cover operation-level inputs for multi-head attention
families and related attention utility primitives. They describe query/K/V
tensors, request metadata, optional cache contents, and merge states without
depending on a specific backend API.

## Operation Semantics

### MHA

`MHAInputs` represents multi-head attention with query heads, key/value heads,
per-head dimensions, request metadata, optional new K/V tensors, optional
attention sinks, and optional dense or paged KV cache contents.

The generated metadata controls the number of cached tokens and new query
tokens. For ragged request layouts, per-request lengths are generated from the
configured totals and batch size so callers do not need to hand-author
consistent cumulative sequence metadata.

### MLA

`MLAInputs` represents multi-head latent attention. It shares the same request
metadata and cache-layout concepts as MHA, but uses MLA-specific operand shapes:
query-nope, query positional embedding channels, compressed KV cache contents,
and optional new compressed KV inputs.

### MLA K/V Pack And FP8 Quantize

`MLAKVPackQuantizeFP8Inputs` represents the utility operation that materializes
MLA K/V tensors for FP8 cache storage. The operation takes non-RoPE key channels
`k_nope`, RoPE key channels `k_pe`, and value channels `v`:

```text
k_pe_heads = broadcast(k_pe, across=kv_heads)
k = concat(k_nope, k_pe_heads, dim=-1)
k_fp8 = cast_fp8(k * k_scale_inv)
v_fp8 = cast_fp8(v * v_scale_inv)
```

`k_nope` and `v` have shape `[num_tokens, num_kv_heads, dim]`. `k_pe` may be
`[num_tokens, qk_rope_head_dim]` or `[num_tokens, 1, qk_rope_head_dim]`; both
forms represent the same per-token RoPE key component shared by all KV heads.

### DeepSeek V4 Sparse-Prefill Indices

`DeepSeekV4SparsePrefillIndexInputs` represents the metadata/index construction
used by DeepSeek V4 sparse prefill attention. The operation does not generate
attention values. It builds integer candidate lists into a per-request KV
workspace:

```text
workspace[request] =
  compressed_prefix_slots[0:compressed_base] +
  gathered_sliding_window_slots[compressed_base:workspace_width]
```

For each generated query token, `positions` gives the absolute token position in
the request and `token_to_req_indices` maps the packed token row back to the
request. The generated `topk_indices` select sparse compressed-prefix slots for
the top-k path. Dense-compressed references instead enumerate compressed prefix
slots from zero up to the per-token compressed length. Both paths append local
sliding-window attention slots derived from `window_size`, `gather_lens`, and
the request sequence lengths.

The generator keeps request lengths, compressed-prefix capacity, gathered SWA
capacity, and workspace width consistent so the generated indices are valid
operation-level inputs for the local-compressed, top-k+SWA, and
dense-compressed+SWA index builders.

### DeepSeek V4 Paged Indices

`DeepSeekV4PagedIndexInputs` represents paged-cache metadata used by DeepSeek V4
decode and indexer helpers. The generated object describes packed query tokens,
their absolute request positions, a request id for each packed token row, and a
per-request block table:

```text
logical_token_position -> logical_block = position // block_size
logical_block -> physical_block = block_table[request, logical_block]
slot_id = physical_block * block_size + position % block_size
```

The same generated values support several related metadata operations:

- mapping request-local top-k indices to global paged-cache slot ids
- building per-token decode sliding-window slot lists and row lengths
- building compressed KV slot mappings for newly materialized compressed tokens
- building decode-indexer block tables and compressed context lengths

The generator owns the shape relationships between request metadata, token
positions, page-table width, optional valid-token masks, optional block-table
base offsets, and the compressed/paged indexing parameters. These are metadata
operations, so no attention value tensors are generated.

### Merge State

`AttentionMergeStateInputs` represents merging two partial attention outputs and
their log-sum-exp states:

```text
lse_a_log2 = lse_a * lse_scale_log2
lse_b_log2 = lse_b * lse_scale_log2
lse_max = max(lse_a_log2, lse_b_log2)
w_a = 2 ** (lse_a_log2 - lse_max)
w_b = 2 ** (lse_b_log2 - lse_max)
out = (out_a * w_a + out_b * w_b) / (w_a + w_b)
lse = (lse_max + log2(w_a + w_b)) / lse_scale_log2
```

This operation is useful when an attention computation is split into partial
contexts. It is independent of how those partial states were produced.

## Validation Contract

The attention generators reject invalid operation inputs before returning
values:

- token totals must be non-negative
- batch sizes, head counts, head dimensions, and page sizes must be positive
- generated request-length metadata is internally consistent
- cache configuration must match the requested cache layout
- paged caches require consistent page-table configuration
- MHA query heads must be compatible with KV heads for grouped attention
- MLA K/V pack inputs require matching token/head dimensions for `k_nope` and
  `v`, positive inverse scales, a broadcastable RoPE key tensor, and an FP8
  output dtype
- DeepSeek V4 sparse-prefill index inputs require tied Q/KV request lengths, a
  compressed-prefix ratio greater than one, valid top-k compressed-prefix slots,
  and a workspace width that covers compressed-prefix plus gathered SWA slots
- DeepSeek V4 paged-index inputs require tied Q/KV request lengths, a block
  table wide enough for generated visible KV positions, positive block/window
  sizes, a compressed-prefix ratio greater than one, and valid token-to-request
  metadata
- merge-state outputs must have shape `[total_q, num_heads, head_dim]`
- merge-state LSE tensors must have shape `[total_q, num_heads]` and use fp32
  generated values
- merge-state `lse_scale_log2` and generated LSE bounds must be positive

Metadata-oriented values such as request lengths and page mappings are generated
from metadata seeds. Numerical tensors are generated from value seeds so tests
can reuse the same request/cache layout across different value draws.

## TokenSpeed API Mapping

TokenSpeed has several attention registry entry points: MHA prefill, MHA
extend/decode with KV cache, MLA prefill, MLA decode with KV cache, and
attention merge-state. TokenSpeed also exposes an MLA K/V pack+quantize helper
that maps directly to `MLAKVPackQuantizeFP8InputValues`, plus DeepSeek V4
sparse-prefill index helpers that accept the tensors and scalar workspace
parameters generated by `DeepSeekV4SparsePrefillIndexInputs`. DeepSeek V4
paged-cache index helpers similarly consume `DeepSeekV4PagedIndexInputs` through
small adapters. The generator values are operation-level values. Tests or
adapters are responsible for converting generated values into the exact keyword
arguments expected by a selected TokenSpeed backend.
