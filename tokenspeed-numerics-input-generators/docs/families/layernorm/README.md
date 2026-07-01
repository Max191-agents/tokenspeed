# LayerNorm Generators

LayerNorm generators cover RMSNorm-style normalization and fused variants used
around attention projections. RMSNorm normalizes each row by the root mean
square of its hidden dimension and applies learned weights. Gemma RMSNorm uses
the same normalization input structure with `1 + weight` scaling. Fused
variants add Q/K splitting, rotary embedding, gates, residuals, or parallel
reductions.

## Generators

- `RMSNormInputs`: generates input and weight tensors for ordinary or Gemma
  RMSNorm.
- `QKRMSNormInputs`: generates Q/K tensors and weights for separate Q/K RMSNorm.
- `FusedQKRMSNormRopeGateInputs`: generates Q/K RMSNorm, RoPE, and gate inputs
  for fused attention setup.
- `ParallelRMSNormInputs`: generates per-rank inputs for parallel RMSNorm
  scenarios.

## Generated Values

Generated tensors preserve row/hidden-dimension relationships and use bounded
floating values appropriate for reductions. References compute normalization in
torch using the operation-level epsilon and weight semantics.

Consumers should use adapters for backend-specific fused argument order. The
generator contract is the normalized operation input structure and not a
particular kernel wrapper.
