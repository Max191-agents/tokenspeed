# Attention Generator Deduplication Report

This report reviews the current attention-family generator surface for cases
where the generator names are tied too closely to TokenSpeed kernel variants or
operation fusions. It does not propose deleting every specialized helper. Some
helpers represent real operations with their own input contracts. The main goal
is to avoid creating new generators for renamed shapes or fused pipelines when
composition would describe the inputs more clearly.

## Configuration Helper Boundary

The standalone numerics package should expose canonical generator configs,
values, and reusable component generators. Named presets or builders that map
TokenSpeed model/kernel terminology onto those configs should live in
TokenSpeed tests or adapters if they are useful. In this report,
"configuration" means capabilities on canonical configs, not new
TokenSpeed-specific convenience constructors in the standalone package.

## Generators That Should Remain Canonical

- `MHAInputs`: canonical generator for multi-head attention with configurable
  request metadata, new Q/K/V tensors, optional sinks, and optional dense or
  paged KV cache.
- `MLAInputs`: canonical generator for MLA. It shares request/cache concepts
  with MHA but has different operand and cache row shapes, so it should stay
  distinct.
- `AttentionMergeStateInputs`: distinct operation for merging partial attention
  outputs and log-sum-exp states.
- `GDNChunkPrefillInputs`: distinct recurrent linear-attention style operation,
  not a softmax-attention variant.

## Strong Deduplication Candidates

Current shared primitives already cover request metadata, page/block tables,
flat slot mappings, packed-row tensor generation, and common byte-cache helper
logic. The remaining candidates below are still useful as migration targets:
they name fused TokenSpeed operations, but their constituent input concepts are
now increasingly represented by narrower generators and helpers.

### `MLAPrefillFP8Inputs`

This is mostly `MLAInputs` in an uncached prefill configuration with materialized
Q/K/V tensors stored in FP8 and tied Q/KV request lengths. The generator exists
because a TokenSpeed kernel accepts that exact input bundle.

Recommended direction: fold this into `MLAInputs` as configuration for
materialized prefill storage dtype/source dtype. If TokenSpeed wants a named
convenience builder for a specific kernel/test shape, that helper should live
in TokenSpeed tests/adapters, not in the standalone numerics package.

### `MLAKVPackQuantizeFP8Inputs`

This fuses two concepts:

- MLA K/V assembly: broadcast `k_pe` across KV heads and concatenate it with
  `k_nope`.
- FP8 quantization of the assembled K and V tensors.

Recommended direction: split this into reusable MLA K/V assembly inputs plus
quantization-family utilities. The `packed_slices` option can remain an adapter
or storage-layout mode, but the generator should not be named as a fused pack
and quantize kernel.

### `GDNQKVSplitInputs` And `PackedQKVComplexRotaryInputs`

Both generate a packed QKV projection tensor and then describe a split plus an
optional transform. One applies optional per-head L2 normalization to Q/K. The
other applies complex RoPE to Q/K.

Recommended direction: introduce a reusable packed-QKV generator that owns
segment widths and head reshaping. Compose it with normalization or rotary
generators/references where needed.

### DeepSeek V4 Transform + Quantization Generators

`DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs` and
`DeepSeekV4InvRoPEFP8QuantInputs` combine reusable operations:

- position and RoPE cache generation
- RoPE or inverse RoPE
- Hadamard transform
- grouping/reshaping
- MXFP4 or FP8 quantization

Recommended direction: keep DeepSeek-specific shape presets or config builders
outside the standalone package, in TokenSpeed tests/adapters if needed. The
numerics package should expose canonical component generators for embedding,
transforms, and quantization; TokenSpeed code can assemble them for specific
kernels.

### DeepSeek V4 Compression + Cache Insert Generators

`DeepSeekV4CSAIndexerMXFP4CacheInsertInputs` and
`DeepSeekV4SparseCompressCacheInsertInputs` are large fused pipelines. They
generate compressor-state cache contents, request/page metadata, compression
window metadata, RMSNorm weights, RoPE cache, output cache bytes, and output
slot mappings.

Current direction: `CompressedSequenceAttentionInputs` is the canonical
operation-level generator for compressed sequence attention. It currently
models DeepSeek V4-style layouts. It always generates the sliding-window
portion and optionally generates compressed-history and CSA indexer nested
values. TokenSpeed helper tests can adapt those nested values to narrower
kernel entry points.

Further cleanup should split the compatibility/helper bundles into reusable
lower-level generators where that makes the implementation easier to follow:

- compressor-state cache rows
- compression-window metadata
- page/block-table metadata, using `PageTableInput` where the table semantics
  are a physical page lookup
- RMSNorm parameters
- RoPE metadata
- flat compressor and KV slot mappings, using `SlotMappingInput`
- indexer MXFP4 cache rows
- sparse K-cache byte rows

The fused DeepSeek tests can then compose those pieces and call a fused
reference. That preserves coverage without making a fused kernel name the input
generation abstraction.

### DeepSeek V4 Cache Write/Gather Generators

`DeepSeekV4IndexerMXFP4CacheWriteInputs` and
`DeepSeekV4IndexerMXFP4CacheGatherInputs` both encode the same paged MXFP4
indexer cache byte layout. `DeepSeekV4KCacheGatherInputs` and
`DeepSeekV4SparseCompressCacheInsertInputs` similarly share the sparse K-cache
byte layout.

Recommended direction: move byte-cache layout generation into cache-oriented
generators, then have write/gather tests compose cache rows with slot mappings.
This follows the same pattern as GEMM: the storage format is reusable, while
write and gather are operation references over that storage.

Current status: the repeated flat slot-mapping generation has been factored
through `SlotMappingInput`, page/block-table generation can target larger
physical page pools through `PageTableInput`, and compressed sequence attention
now has one canonical high-level generator instead of a separate generator per
CSA/HCA helper path.

## Metadata Deduplication Candidates

The following generators are metadata-heavy and share concepts:

- `DSATopKSlotInputs`
- compressed sequence attention nested paged/sparse index values

They all generate request/token positions, sequence lengths, page tables or
block tables, and local-to-global cache slot relationships. They should not
necessarily collapse into one generator because the output metadata operations
are different, but their common pieces should move into reusable metadata
generators.

Recommended shared pieces:

- token-to-request mapping
- per-token absolute position generation
- local top-k/local-offset generation
- compressed-prefix length generation
- sliding-window length generation
- page/block-table generation

## Lower Priority Or Keep As-Is

- `DSADecodeTopKInputs`: distinct deterministic top-k operation over pre-masked
  logits. It may eventually belong closer to sampling/metadata, but it is not a
  duplicate of MHA/MLA.
- `DSASparseDecodeKVPackInputs`: distinct byte-row packing operation. It may
  share future cache-row layout utilities with DeepSeek sparse K-cache
  generators, but its row format is not identical.
- `DeepSeekV4CompressorStateInputs`: distinct state-cache write operation.
  It should share compressor-state metadata/layout generation with the fused
  compression generators, but it is not just a renamed MHA/MLA case.

## Suggested Cleanup Order

1. Add reusable packed-QKV and metadata/cache-row primitives.
2. Convert `MLAPrefillFP8Inputs` into `MLAInputs` configuration. Any
   TokenSpeed convenience builder should live outside the numerics package.
3. Split `MLAKVPackQuantizeFP8Inputs` into MLA K/V assembly plus quantization
   composition.
4. Continue refactoring DeepSeek V4 cache layouts and compression-window
   metadata into shared generators where that removes real implementation
   duplication.
5. Move TokenSpeed-specific convenience builders and compatibility adapters out
   of the standalone numerics package once the kernel tests no longer import
   helper-specific DeepSeek generator names directly.
