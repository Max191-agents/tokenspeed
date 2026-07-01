# KV-Cache Generators

KV-cache generators model operations that write, gather, or transfer rows in
attention cache storage. The cache may store separate K/V tensors, compressed
MLA rows, FP8 values, dense slots, or page-table metadata depending on the
operation being tested.

## Generators

- `FP8KVCacheWriteInputs`: generates K/V token rows, FP8 cache tensors, cache
  locations, and optional scalar scales for quantized cache writes.
- `KVCacheStoreInputs`: generates per-token K/V source rows, destination cache
  tensors, and unique store locations.
- `PageTableGatherInputs`: generates source page tables, selected request rows,
  sequence lengths, and padded output buffers.
- `KVCacheTransferInputs`: generates source/destination K/V cache layers plus
  unique source and destination transfer indices.
- `MLAKVCacheTransferInputs`: generates compressed MLA cache layers and row
  transfer indices.

## Generated Values

Generated metadata uses unique destination indices where write order would
otherwise affect the result. Page-table metadata is generated consistently with
request lengths and page counts. References clone the destination cache state
and apply the operation-level scatter, gather, or transfer semantics.

Backend-specific pointer arrays and registry kwargs are adapter details; the
generator owns only cache shapes, row identities, page metadata, and value
validity.
