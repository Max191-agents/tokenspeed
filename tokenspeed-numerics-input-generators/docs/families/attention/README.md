# Attention Generators

Attention input generation is organized around five operation families:

- `MHAInputs`: multi-head attention with optional new K/V tensors, dense cache,
  or paged KV cache.
- `MLAInputs`: multi-head latent attention with MLA-specific query/cache
  shapes and the same request/cache metadata concepts as MHA.
- `CSAInputs`: compressed sequence attention. This includes sliding-window
  attention and optional compressed-history/indexer inputs for DeepSeek
  V4-style CSA/HCA paths.
- `DSAInputs`: dynamic sparse attention helper inputs, including sparse decode
  KV packing, sparse slot mapping, and deterministic decode top-k selection.
- `GDNInputs`: Gated DeltaNet inputs, including packed QKV split and chunked
  prefill configurations.

These are the public attention-family generator entry points. Configuration
objects select the concrete mode within each family, similar to how MHA and MLA
configs select prefill, decode, cache layout, and ragged metadata behavior.

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

## Generated Values

Generated attention values keep structured metadata separate from numerical
tensor values. Metadata seeds can be held fixed independently from value seeds
so different numerical draws can share the same request layout, page mapping,
or sparse-index structure.

Cache/page generation verifies valid page tables, request lengths, and cache
slot relationships before values are returned. The family generator should be
the place where operation-level shape and metadata relationships are enforced.
