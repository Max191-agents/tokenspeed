# Attention Generators

Attention generators describe the tensor and metadata inputs for multi-head
attention variants. The core operation combines query tokens with key/value
context, where the context may be dense new tokens, dense cached tokens, or a
paged cache addressed through metadata. MHA stores K and V separately; MLA uses
compressed/cache-specific representations with different per-token shapes.

## Generators

- `MHAInputs`: generates Q, optional new K/V, request metadata, and optional
  dense or paged KV cache state for MHA.
- `MLAInputs`: generates MLA query/cache inputs using the same metadata/cache
  concepts with MLA-specific operand shapes.
- `AttentionMergeStateInputs`: generates merge-state tensors for combining
  partial attention results.
- `MLAKVPackQuantizeFP8Inputs`: generates MLA KV pack and FP8 quantization
  inputs.
- `GDNQKVSplitInputs`: generates packed QKV inputs for split operations.
- `GDNChunkPrefillInputs`: generates Gated DeltaNet chunked-prefill tensors,
  log-space gates, recurrent state, and sequence metadata.
- `PackedQKVComplexRotaryInputs`: generates packed QKV and rotary metadata for
  complex rotary transforms.
- `DSASparseDecodeKVPackInputs`, `DSATopKSlotInputs`,
  `DeepSeekV4PagedIndexInputs`, and `DeepSeekV4SparsePrefillIndexInputs`:
  generate DeepSeek-style sparse decode/prefill indexing and cache metadata.

## Generated Values

Generated attention values keep structured metadata separate from numerical
tensor values. Metadata seeds can be held fixed independently from value seeds
so different numerical draws can share the same request layout, page mapping,
or sparse-index structure.

Cache/page generation enforces valid page tables, request lengths, and cache
slot relationships. Kernel-specific names such as `block_tables` or flattened
registry arguments belong in adapters outside the generator.
