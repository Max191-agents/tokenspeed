# Embedding Generators

Embedding generators currently focus on rotary positional embedding. Rotary
embedding applies position-dependent 2D rotations to pairs of hidden dimensions,
with optional fused handling for K/V tensors and FP8 quantization paths in
attention frontends.

## Generators

- `RopeInputs`: generates query/key style tensors, position ids, and cosine/sine
  caches for rotary embedding.
- `RopeFusedKVInputValues`: represents generated values for fused K/V rotary
  scenarios.
- `MLARopeQuantizeFP8Inputs`: generates decomposed MLA/GQA query-key slices,
  position metadata, RoPE cache, and FP8 output buffers for fused RoPE plus
  quantization.

The shared `build_rope_cos_sin_cache` utility constructs deterministic rotary
tables for a requested context length and rotary dimension.

## Generated Values

Generated values include input tensors, position metadata, precomputed
cosine/sine caches, and optional output buffers. Consumers execute the
operation or translate those values into implementation-specific layouts;
reference implementations are not part of this package.

Verification keeps position ids within the generated cache range and enforces
compatible rotary dimensions. For fused RoPE quantization, verification also
enforces the PE/NOPE shape relationships, FP8 output dtype, positive
quantization scales, and the distinction between rank-2 shared-K MLA tensors
and rank-3 explicit-KV-head tensors.
