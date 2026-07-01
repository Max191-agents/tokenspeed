# KV Cache Input Generators

KV-cache generators cover operations that manipulate cache storage for
attention. This slice models cache-row transfer: copying selected cache slots
from one cache to another. It intentionally does not yet cover every
TokenSpeed KV-cache kernel, such as FP8 cache writes or page-table gather.

## Operation Semantics

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

## Validation Contract

The generators reject invalid transfer inputs before values are returned:

- layer count, slot count, head counts, head dimensions, and MLA cache width
  must be positive
- transfer count must be non-negative and no larger than the slot count
- generated destination indices are unique, so the transfer result is
  deterministic and not affected by write ordering
- generated source indices are unique for easier debugging and coverage
- cache dtype must be a regular floating torch dtype
- index dtype must be int32 or int64

Source and destination indices are generated from `metadata_seed`, while cache
contents are generated from `seed`. This allows tests to reuse the same row-copy
mapping across different random cache contents.

## TokenSpeed API Mapping

TokenSpeed has per-layer and all-layer transfer kernels for both conventional
K/V caches and MLA caches. The generated values map directly to the per-layer
APIs. For all-layer APIs, tests or adapters can convert the generated list of
layer tensors into pointer tensors.

The generator returns operation-level values only. Pointer tensors and other
implementation-specific launch arguments are intentionally left to TokenSpeed
test adapters.
