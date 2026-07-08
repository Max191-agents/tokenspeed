# Attention Input Generators

Attention input generation is organized around five operation families:
`MHAInputs`, `MLAInputs`, `CSAInputs`, `DSAInputs`, and `GDNInputs`. These are
the public generator entry points. Config objects select the concrete mode
within a family, so variants such as prefill/decode, paged/dense cache, sparse
decode packing, and GDN QKV split are represented as configurations of the
family generator rather than separate generator families.

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
query-NoPE channels, query positional embedding channels, compressed KV cache
contents, and optional new compressed KV inputs.

`mla_reference` evaluates generated MLA values at the operation level. Uncached
inputs use explicit varlen Q/K/V attention, including grouped KV heads when the
generated head counts require expansion. Paged-cache inputs gather compressed
cache rows through the generated page table and compute absorbed MLA decode
outputs over the latent KV channels.

### CSA

`CSAInputs` represents compressed sequence attention input generation. The
current layout support is DeepSeek V4-style. The sliding-window attention
portion is always generated; optional compressed-history and indexer configs
add the generated values needed by CSA/HCA cache, sparse-index, and gather
paths.

The compressed-history config covers both supported compression ratios:

- `compress_ratio == 128`: HCA-style compressed history.
- `compress_ratio == 4`: CSA-style compressed history with optional indexer
  inputs.

CSA values intentionally contain nested bundles for the narrower helper
operations consumed by TokenSpeed tests and references. Callers should still
construct those values through `CSAInputs` instead of using one generator per
helper kernel.

### DSA

`DSAInputs` represents DeepSeek sparse attention metadata and cache helper
inputs. Its configs cover:

- sparse decode KV packing, where BF16 NoPE/RoPE key rows are packed into
  physical cache slots with FP8 E4M3 NoPE blocks and FP32 per-block scales;
- sparse top-k slot mapping, where token-local context offsets are translated
  through a block table into physical KV-cache slots;
- deterministic decode top-k selection over pre-masked indexer logits.

The selected top-k indices are token-local context offsets. Slot-mapping
configs convert those offsets into physical cache slots when a backend needs
cache addresses.

### GDN

`GDNInputs` represents Gated DeltaNet inputs. Its configs cover packed QKV
split and chunked prefill.

Packed QKV split starts from a post-projection tensor whose last dimension is
`q_width + k_width + v_width`, then returns separate Q, K, and V views or
outputs. Some GDN paths also fuse per-head L2 normalization of Q and K into the
split; V is copied without normalization.

Chunked prefill represents the prompt-side recurrent scan for Gated
DeltaNet-style linear attention. Unlike softmax attention, it does not
materialize pairwise attention over all prior tokens. It maintains a
per-sequence, per-value-head matrix state, applies log-space decay gates,
applies a delta-rule correction, and reads the updated state with the query
vector.

## Shared Components

The attention family generators reuse lower-level metadata and cache
components:

- `MHARequestMetadataInput`: request token lengths, cumulative offsets, visible
  KV lengths, and cache sequence lengths.
- `PageTableInput`: paged-cache page/block tables.
- `SlotMappingInput`: flat token-row to cache-slot mappings.
- `KVCacheInput` and `MLAKVCacheInput`: dense or paged cache storage for MHA
  and MLA cache layouts.

Kernel-facing names such as `block_table`, `slot_mapping`,
`kv_slot_mapping`, or `compressor_slot_mapping` should map back to these shared
metadata concepts when their semantics match. Backend adapters are responsible
for converting generated values into exact registry kwargs.

## Validation Contract

The attention generators reject invalid operation inputs before returning
values:

- token totals must be non-negative;
- batch sizes, head counts, head dimensions, and page sizes must be positive;
- generated request-length metadata must be internally consistent;
- cache configuration must match the requested cache layout;
- paged caches require consistent page-table configuration;
- MHA query heads must be compatible with KV heads for grouped attention;
- MLA prefill references require matching Q/K dimensions and compatible
  grouped K/V heads;
- DeepSeek sparse attention packing and slot-mapping configs require integer
  slot metadata, valid packed row widths, and no out-of-bounds cache addresses;
- CSA compressed-history and indexer configs require supported compression
  ratios, page metadata wide enough for visible KV positions, and cache/indexer
  layouts that match the selected mode;
- GDN packed split configs require positive head counts/dimensions and a packed
  last dimension equal to `q_width + k_width + v_width`;
- GDN chunked-prefill configs require consistent recurrent-state shapes,
  positive chunk metadata, and valid per-sequence cumulative lengths.

Metadata-oriented values such as request lengths and page mappings are
generated from metadata seeds. Numerical tensors are generated from value seeds
so tests can reuse the same request/cache layout across different value draws.

## TokenSpeed API Mapping

TokenSpeed exposes several helper kernels whose input bundles are narrower than
the five family generators. Tests should construct the corresponding family
generator and then use the nested generated values required by the helper
kernel. For example, DeepSeek sparse attention decode packing comes from
`DSAInputs`, GDN QKV split comes from `GDNInputs`, and DeepSeek V4-style
compressed-cache/indexer helpers come from `CSAInputs`.

Generator values are operation-level values. Tests or adapters are responsible
for converting generated values into the exact keyword arguments expected by a
selected TokenSpeed backend.
