# Embedding Generators

Embedding generators currently focus on rotary positional embedding. Rotary
embedding applies position-dependent 2D rotations to pairs of hidden dimensions,
with optional fused handling for K/V tensors and decomposed MLA query/key
inputs.

## Generators

- `RopeInputs`: generates query/key style tensors, position ids, and cosine/sine
  caches for rotary embedding.
- `RopeFusedKVInputValues`: represents generated values for fused K/V rotary
  scenarios.
- `MLARopeInputs`: generates decomposed MLA/GQA query-key slices, position
  metadata, and a RoPE cache using a configurable floating dtype.

The shared `build_rope_cos_sin_cache` utility constructs deterministic rotary
tables for a requested context length and rotary dimension.

## Generated Values

Generated values include input tensors, position metadata, precomputed
cosine/sine caches, and optional output buffers. Consumers execute the
operation or translate those values into implementation-specific layouts;
reference implementations are not part of this package.

Verification keeps position ids within the generated cache range and enforces
compatible rotary dimensions. For MLA RoPE, verification also enforces the
PE/NOPE shape relationships, generated floating dtype, and the distinction
between rank-2 shared-K MLA tensors and rank-3 explicit-KV-head tensors.
Quantization configuration belongs to consumers of the generated inputs.
