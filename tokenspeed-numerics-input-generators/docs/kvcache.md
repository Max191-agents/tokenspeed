# KV Cache Input Generators

KV-cache generators cover operations that manipulate cache storage for
attention. This slice models per-token cache store, cache-row transfer, and
active page-table gather-with-padding. It intentionally does not yet cover every
TokenSpeed KV-cache kernel, such as FP8 cache writes.

## Operation Semantics

### K/V Cache Store

`KVCacheStoreInputs` represents the per-token scatter used to write newly
computed K/V rows into cache slots:

```text
k_dst[loc[i]] = k_src[i]
v_dst[loc[i]] = v_src[i]
```

The source token shape is `[num_tokens, num_kv_heads, head_dim]`. The
destination cache shape is `[num_slots, num_kv_heads, head_dim]`. Generated
locations are unique so the result is deterministic and not affected by write
ordering.

### K/V Cache Transfer

`KVCacheTransferInputs` represents row transfer for conventional attention
caches with separate K and V tensors per layer:

```text
dst_k[layer][dst_indices[i]] = src_k[layer][src_indices[i]]
dst_v[layer][dst_indices[i]] = src_v[layer][src_indices[i]]
```

The cache slot shape is `[num_kv_heads, head_dim]`, and every layer has the same
slot count and slot shape.

### MLA Cache Transfer

`MLAKVCacheTransferInputs` represents the same row-transfer operation for MLA
caches, where each layer has one compressed cache tensor instead of separate K
and V tensors:

```text
dst[layer][dst_indices[i]] = src[layer][src_indices[i]]
```

The generated MLA cache shape is `[num_slots, 1, kv_cache_dim]`, matching the
compressed cache layout used by the TokenSpeed transfer kernels.

### Page-Table Gather With Padding

`PageTableGatherInputs` represents gathering active request rows from a source
request-to-page table and writing a compact active page table:

```text
n_pages = ceil(seq_lens[row] / page_size)
out[row, :n_pages] = req_to_page[req_pool_indices[row], :n_pages]
out[row, n_pages:] = dummy_slot
```

`req_pool_indices` selects which source request rows are active. `seq_lens`
contains the active KV length for each gathered row. Padding columns are filled
with `dummy_slot` so consumers do not accidentally read stale page IDs beyond
the valid page range.

## Validation Contract

The generators reject invalid inputs before values are returned:

- slot count, head counts, head dimensions, page counts, page size, layer count,
  and MLA cache width must be positive where applicable
- store and transfer counts must be non-negative and no larger than the slot
  count when unique locations are required
- generated destination locations and indices are unique, so the result is
  deterministic and not affected by write ordering
- generated source indices are unique for easier debugging and coverage
- page-table gather batch size must fit within the source request table
- generated sequence lengths imply no more pages than the output can hold
- cache dtype must be a regular floating torch dtype
- index dtype must be int32 or int64

Locations, indices, and request lengths are generated from `metadata_seed`,
while cache contents and page table contents are generated from `seed`. This
allows tests to reuse the same metadata layout across different random values.

## TokenSpeed API Mapping

TokenSpeed has per-layer and all-layer transfer kernels for both conventional
K/V caches and MLA caches. The generated transfer values map directly to the
per-layer APIs. For all-layer APIs, tests or adapters can convert the generated
list of layer tensors into pointer tensors. The store and page-table gather
values map directly to the current TokenSpeed Triton helpers.

The generator returns operation-level values only. Pointer tensors and other
implementation-specific launch arguments are intentionally left to TokenSpeed
test adapters.
