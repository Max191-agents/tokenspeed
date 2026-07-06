# GEMM Generators

GEMM generators model inputs for the operation-level computation `A @ B.T`.
The operation is defined by logical dimensions `M`, `N`, and `K`, physical
operand layouts, optional batch prefixes, and optional scale sidecars.

## Generators

- `GemmInputs`: generates dense, scaled, FP8, MXFP4, NVFP4, MXINT4, or skipped
  operands for GEMM-style tests.

Helper functions such as `gemm_scale_shape`, `mxfp8_gemm_input_config`,
`mxfp4_gemm_input_config`, `nvfp4_gemm_input_config`, and
`mxint4_gemm_input_config` build valid scale shapes and custom operand configs
from logical GEMM dimensions.

Layer-level projections such as router projection and LM-head projection should
map their domain names onto `M`, `N`, and `K` instead of using separate GEMM
generators. Fused kernels should compose GEMM inputs with the other operation
families they fuse, such as activation or quantization.

## Generated Values

`GemmInputValues` returns `A`, `B`, always-generated `C`, and optional scale
tensors. Plain tensors are returned directly; quantized packed formats use
storage tensors plus scale sidecars. References dequantize or apply scales at
the operation level before computing the mathematical result.

The generator verifies layout compatibility, custom dtype scale requirements,
and shape relationships so tests do not need to reconstruct GEMM validity rules.
