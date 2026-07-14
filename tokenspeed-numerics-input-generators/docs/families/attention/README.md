# Attention Generators

Attention input generation is organized around five operation families:

- `MHAInputs`: multi-head attention with optional new K/V tensors, dense cache,
  or paged KV cache.
- `MLAInputs`: multi-head latent attention with MLA-specific query/cache
  shapes and the same request/cache metadata concepts as MHA.
- `CSAInputs`: compressed sequence attention. This includes sliding-window
  attention and optional compressed-history/indexer inputs for DeepSeek
  V4-style CSA/HCA paths.
- `DSAInputs`: dynamic sparse attention decode values, including source K
  rows, sparse-index logits, selected offsets, slot mappings, and page-table
  metadata.
- `GDNInputs`: Gated DeltaNet recurrent-scan values, including Q/K/V tensors,
  gates, beta, initial state, and sequence metadata.

These are the public attention-family generator entry points. Configuration
objects configure operation-level shape, cache, and metadata relationships.
Backend-specific helper-kernel argument bundles should be derived outside the
generator package.

## Shared Components

The attention generators reuse a few lower-level metadata components:

- `MHARequestMetadataInput`: request token lengths, cumulative offsets, visible
  KV lengths, and cache sequence lengths.
- `PageTableInput`: paged-cache page/block tables.
- `SlotMappingInput`: flat token-row to cache-slot mappings.
- `KVCacheInput` and `MLAKVCacheInput`: dense or paged cache storage for MHA
  and MLA cache layouts.

Kernel-specific names such as `block_table`, `slot_mapping`,
`kv_slot_mapping`, or `compressor_slot_mapping` should map back to these shared
metadata concepts when their semantics match. Kernel registry adapters are
responsible for flattening generated values into backend-specific kwargs.

Helper kernels are not separate generator families. For example, TokenSpeed
tests can pack GDN Q/K/V into a QKV split input, adapt `DSAInputs` into sparse
decode pack or top-k slot arguments, and derive DeepSeek V4-style packed
cache/indexer helper values from `CSAInputs` in TokenSpeed adapter code.

## Generated Values

Generated attention values keep structured metadata separate from numerical
tensor values. Metadata seeds can be held fixed independently from value seeds
so different numerical draws can share the same request layout, page mapping,
or sparse-index structure.

Cache/page generation verifies valid page tables, request lengths, and cache
slot relationships before values are returned. The family generator should be
the place where operation-level shape and metadata relationships are enforced.
