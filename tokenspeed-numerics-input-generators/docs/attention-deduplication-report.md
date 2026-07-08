# Attention Deduplication Status

Attention input generation is intentionally deduplicated to five public family
generators:

- `MHAInputs`
- `MLAInputs`
- `CSAInputs`
- `DSAInputs`
- `GDNInputs`

The old helper-kernel split made separate generators for fused operations,
renamed metadata, or narrower utility kernels. That added API surface without
adding new operation semantics. The current model keeps those details as
configs or nested generated value bundles under the closest operation family.

## Current Mapping

- MHA prefill, decode, extend, dense cache, paged cache, and ragged metadata
  are configurations of `MHAInputs`.
- MLA prefill and cache/decode modes are configurations of `MLAInputs`.
- Sliding-window compressed sequence attention, compressed history, HCA/CSA
  compression-ratio differences, and DeepSeek V4-style compressed-cache/indexer
  helper bundles are configurations or nested values of `CSAInputs`.
- Sparse decode KV packing, sparse top-k slot mapping, and deterministic
  decode top-k selection are configurations of `DSAInputs`.
- GDN packed QKV split and GDN chunked prefill are configurations of
  `GDNInputs`.

## Rule Of Thumb

A new attention generator should be added only when it represents a new
operation family. Differences in dtype, quantization format, shape naming,
cache layout, decode/prefill mode, or fused helper kernels should normally be
represented as configuration on one of the five existing families.
