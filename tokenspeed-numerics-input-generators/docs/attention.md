# Attention Input Generators

Attention input generation is organized around five operation families:
`MHAInputs`, `MLAInputs`, `CSAInputs`, `DSAInputs`, and `GDNInputs`. These are
the public generator entry points. They model operation-level inputs, not every
TokenSpeed helper-kernel argument bundle. Frontend or kernel tests should adapt
the generated values when a helper kernel wants a packed view, a cache-write
subset, or backend-specific metadata names.

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
add the core values needed for CSA/HCA compressed history and sparse indexer
state.

The compressed-history config covers both supported compression ratios:

- `compress_ratio == 128`: HCA-style compressed history.
- `compress_ratio == 4`: CSA-style compressed history with optional indexer
  inputs.

CSA values may contain nested structured values that TokenSpeed adapters can
use for narrower helper kernels, but those helpers are not separate generator
families.

### DSA

`DSAInputs` represents dynamic sparse attention decode inputs. It generates the
core values shared by DSA helper kernels:

- source NoPE/RoPE key rows and physical cache slots;
- masked indexer logits plus top-k output storage;
- token-local offsets, sequence lengths, and page-table metadata.

TokenSpeed adapters derive sparse KV pack, decode top-k, and local-offset to
global-slot argument bundles from these core values.

### GDN

`GDNInputs` represents Gated DeltaNet inputs for the prompt-side recurrent
scan. Unlike softmax attention, it does not materialize pairwise attention over
all prior tokens. It maintains a per-sequence, per-value-head matrix state,
applies log-space decay gates, applies a delta-rule correction, and reads the
updated state with the query vector.

TokenSpeed helper kernels that consume packed QKV can pack the generated Q, K,
and V values outside the generator.

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
- Dynamic sparse attention configs require integer slot metadata, valid row
  widths, and no out-of-bounds cache addresses;
- CSA compressed-history and indexer configs require supported compression
  ratios, page metadata wide enough for visible KV positions, and cache/indexer
  layouts that match the selected CSA/HCA components;
- GDN configs require consistent recurrent-state shapes, positive chunk
  metadata, and valid per-sequence cumulative lengths.

Metadata-oriented values such as request lengths and page mappings are
generated from metadata seeds. Numerical tensors are generated from value seeds
so tests can reuse the same request/cache layout across different value draws.

## TokenSpeed API Mapping

TokenSpeed exposes several helper kernels whose input bundles are narrower than
the five family generators. Tests should construct the closest operation-family
generator and adapt from its generated values. For example, dynamic sparse
attention decode packing is derived from `DSAInputs`, GDN QKV split can be
derived by packing `GDNInputs` Q/K/V tensors, and DeepSeek V4-style
compressed-cache/indexer helper bundles come from nested `CSAInputs` values.

Generator values are operation-level values. Tests or adapters are responsible
for converting generated values into the exact keyword arguments expected by a
selected TokenSpeed backend.
