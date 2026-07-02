# GEMM Generators

GEMM generators model matrix multiplication inputs for `A @ B.T`-style kernels
and fused GEMM-like operations. The operation is defined by logical dimensions
`M`, `N`, and `K`, physical operand layouts, optional batch prefixes, and
optional scale sidecars that define the represented numerical values.

## Generators

- `GemmInputs`: generates dense, scaled, FP8, MXFP4, MXINT4, or skipped
  operands for GEMM-style tests.
- `RouterProjectionInputs`: generates hidden states and router weights for the
  MoE router projection `hidden_states @ router_weights.T`, with default weight
  scaling chosen to keep fp32 router logits in a useful range.
- `LMHeadProjectionInputs`: generates hidden states and vocabulary projection
  weights for `hidden_states @ weight.T`, with default weight scaling chosen to
  keep LM-head logits in a useful range.
- `NVFP4GemmSwiGLUNVFP4QuantInputs`: generates inputs for a fused NVFP4 GEMM,
  SwiGLU activation, and NVFP4 output quantization operation.

Helper functions such as `gemm_scale_shape`, `mxfp4_gemm_input_config`, and
`mxint4_gemm_input_config` build valid scale shapes and custom operand configs
from the logical GEMM dimensions.

## Generated Values

`GemmInputValues` returns `A`, `B`, always-generated `C`, and optional scale
tensors. Plain tensors are returned directly; quantized packed formats use
storage tensors plus scale sidecars. References dequantize or apply scales at
the operation level before computing the mathematical result.

The generator verifies layout compatibility, custom dtype scale requirements,
and shape relationships so tests do not need to reconstruct GEMM validity rules.
Router projection generation additionally verifies positive token, hidden, and
expert dimensions, supported floating-point dtypes, finite positive generation
scales, and matching hidden dimensions. LM-head projection generation verifies
the same projection invariants with vocabulary rows in place of expert rows.
