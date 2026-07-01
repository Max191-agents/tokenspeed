# Embedding Generators

Embedding generators currently focus on rotary positional embedding. Rotary
embedding applies position-dependent 2D rotations to pairs of hidden dimensions,
with optional fused handling for K/V tensors in attention frontends.

## Generators

- `RopeInputs`: generates query/key style tensors, position ids, and cosine/sine
  caches for rotary embedding.
- `RopeFusedKVInputValues`: represents generated values for fused K/V rotary
  scenarios.

The shared `build_rope_cos_sin_cache` utility constructs deterministic rotary
tables for a requested context length and rotary dimension.

## Generated Values

Generated values include the input tensor, position metadata, and precomputed
cosine/sine cache. The reference applies rotary math directly over the generated
values, while backend adapters can translate those values into kernel-specific
layouts.

Verification keeps position ids within the generated cache range and enforces
compatible rotary dimensions so consumers receive valid embedding inputs.
