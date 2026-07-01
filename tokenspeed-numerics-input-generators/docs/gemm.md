# GEMM Input Generators

GEMM-family generators produce operands for matrix multiplication operations
whose logical computation is based on an `M x K` left-hand operand multiplied by
an `N x K` right-hand operand viewed as transposed, producing an `M x N` result.
The same generator also covers scaled and quantized operand storage when a
tensor's numerical value is represented by storage values plus scale tensors.

The family is intentionally described in terms of matrix operands, layouts,
dtypes, and scale relationships rather than a specific TokenSpeed kernel call.
Adapters can translate the generated values into registry arguments such as
`A`, `B`, `C`, `A_scales`, `B_scales`, output dtype metadata, or backend-specific
layout flags.

## Generators

`GemmInputs` generates:

- `A`: optional left-hand operand storage.
- `B`: optional right-hand operand storage.
- `C`: generated output/accumulator tensor.
- `A_scales` and `B_scales`: optional scale tensors attached to the corresponding
  operands.

The required configuration is the logical `M`, `N`, and `K` shape plus the dtypes
for `A`, `B`, and `C`. `A` or `B` may use dtype `None` when a parent operation,
such as a layer-level MoE generator, only needs one side of the GEMM. `C` is
required and is always generated.

`gemm_reference` computes the operation-level `A @ B.T` result from generated
values. It normalizes dense operand layouts, applies scale sidecars for scaled
tensors, and dequantizes MXFP4 storage with UE8M0 scales before multiplying.
Kernel tests can use it as the semantic target while keeping backend-specific
argument mapping in the adapter or test.

## Scale And Quantized Storage

Dense floating-point tensors use the core tensor generator. Scaled tensors use
the same tensor generator with explicit scale shape and scale dtype. Custom
dtypes, such as MXFP4 storage with UE8M0 scales, are represented through the
shared custom dtype enum so the format-specific constraints are centralized.

`gemm_scale_shape` provides common scale-shape calculations for tensor,
channel, and block granularities. `mxfp4_gemm_input_config` builds a GEMM config
for MXFP4 operands with block scales.

## Verification

The generator verifies logical dimensions, required output dtype, layout-derived
physical shapes, and scale requirements. It rejects configurations that would
produce meaningless inputs, such as scales for skipped operands or MXFP4 storage
without compatible scales. Layout names are validated explicitly. The current
MXFP4 GEMM definition is the row-major form consumed by the TokenSpeed Triton
MXFP4 GEMM path, so MXFP4 `A` operands require `MK` layout and MXFP4 `B`
operands require `NK` layout.
