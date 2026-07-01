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

Some backend entry points consume a different physical layout than the
operation-level default. For example, the TokenSpeed Triton scaled-FP8 GEMM path
accepts `B` as `[K, N]`, while `GemmInputs` defaults to the semantic right-hand
operand shape `[N, K]` used by `A @ B.T`. A kernel adapter can transpose the
generated `B` tensor for that call while still comparing against
`gemm_reference(values)`. That layout translation belongs at the consumer
boundary, not in the generator definition.

## Scale And Quantized Storage

Dense floating-point tensors use the core tensor generator. Scaled tensors use
the same tensor generator with explicit scale shape and scale dtype. Custom
dtypes, such as MXFP4 storage with UE8M0 scales, are represented through the
shared custom dtype enum so the format-specific constraints are centralized.

`gemm_scale_shape` provides common scale-shape calculations for tensor,
channel, and block granularities. `mxfp4_gemm_input_config` builds a GEMM config
for MXFP4 operands with block scales.

For block-scaled FP8 GEMM, generated storage values are interpreted as FP8
values multiplied by a scale grid. `A` commonly uses one scale row per logical
`M` row and one scale group per `K` block, while `B` may use a 2D grid over
logical `N` blocks and `K` blocks. The reference expands those grids over the
logical operand before computing `A @ B.T`. The current generator path expects
regular scale grids that evenly partition the generated physical operand shape.

## Fused NVFP4 GEMM And SwiGLU

`NVFP4GemmSwiGLUNVFP4QuantInputs` represents the fused dense MLP primitive:

```text
gate_up = dequant_nvfp4(x_fp4) @ dequant_nvfp4(w1_fp4).T
gate, up = split(gate_up)
activated = silu(gate) * up
out_fp4, out_scale = quantize_nvfp4(activated)
```

The generator produces model-like floating source tensors, quantizes them into
packed NVFP4 values plus linear FP8 group scales, and returns the global scales
needed by NVFP4 GEMM adapters. The output NVFP4 global scale is explicit in the
config so generation does not need to run the full GEMM just to infer a scale.
TokenSpeed kernels may require backend-specific scale swizzles or gate/up
weight interleaving; those layouts are adapter concerns.

`nvfp4_gemm_swiglu_nvfp4_quant_reference` implements the operation using the
same NVFP4 dequantization and quantization semantics as the quantization
family.

## Verification

The generator verifies logical dimensions, required output dtype, layout-derived
physical shapes, and scale requirements. It rejects configurations that would
produce meaningless inputs, such as scales for skipped operands or MXFP4 storage
without compatible scales. Layout names are validated explicitly. The current
MXFP4 GEMM definition is the row-major form consumed by the TokenSpeed Triton
MXFP4 GEMM path, so MXFP4 `A` operands require `MK` layout and MXFP4 `B`
operands require `NK` layout.

The fused NVFP4 generator also verifies the fixed NVFP4 scale group width,
source dtype, divisibility of K and intermediate dimensions by the scale group,
and scalar global-scale relationships.
