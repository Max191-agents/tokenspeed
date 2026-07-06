# GEMM Input Generators

GEMM-family generation is defined around one operation:

```text
output = A @ B.T
```

`A` is a logical `M x K` operand, `B` is a logical `N x K` operand, and the
result is `M x N`. Physical operand layouts, batch prefixes, scale sidecars, and
packed custom dtypes describe storage, not different operations.

The family intentionally does not introduce separate generators for layer names
such as router projection or LM-head projection. Those are GEMMs with different
domain names for `M`, `N`, and `K`. Consumers should translate their layer
metadata into `GemmInputConfig` at the adapter or test boundary.

The family also does not define fused-operation generators. For example, a fused
NVFP4 GEMM + SwiGLU + NVFP4 quantization test can compose:

- `GemmInputs` for the NVFP4 `A @ B.T` inputs.
- `gemm_reference` for the GEMM result.
- activation or quantization family references for the following SwiGLU and
  output-quantization operation.

This keeps each generator tied to one operation-level input contract while still
making fused kernel tests easy to write.

## Generator

`GemmInputs` generates:

- `A`: optional left-hand operand storage.
- `B`: optional right-hand operand storage.
- `C`: always-generated output/accumulator tensor.
- `A_scales` and `B_scales`: optional scale tensors attached to `A` and `B`.

The required configuration is logical `M`, `N`, `K`, the dtype for `A`, the
dtype for `B`, and the dtype for `C`. `A` or `B` may use dtype `None` when a
parent generator only needs one side of the GEMM, but `C` is required and is
always generated.

`gemm_reference` computes the semantic `A @ B.T` result. It normalizes dense
operand layouts, applies scale sidecars, dequantizes MXFP4 storage with UE8M0
scales, dequantizes NVFP4 storage with per-block local scales, and dequantizes
MXINT4 storage with BF16 group scales before multiplying.

## Shape Mapping

Layer-level names map directly to GEMM dimensions:

```text
router_logits = hidden_states @ router_weights.T
M = num_tokens
N = num_experts
K = hidden_dim
```

```text
lm_head_logits = hidden_states @ vocab_weight.T
M = num_tokens
N = vocab_size_or_local_vocab_shard
K = hidden_dim
```

If a layer wants a distribution-specific tweak, such as scaling projection
weights by `1 / sqrt(hidden_dim)`, that tweak should be applied by the caller or
layer-level parent generator after the GEMM values are generated. It should not
require a renamed GEMM generator.

## Scale And Quantized Storage

Dense floating-point tensors use the core tensor generator. Scaled tensors use
the same tensor generator with explicit scale shape and scale dtype. Custom
dtypes, such as MXFP4 storage with UE8M0 scales, NVFP4 storage with per-block
local scales, and MXINT4 storage with BF16 group scales, are represented through
the shared custom dtype enum so format-specific constraints stay centralized.

`gemm_scale_shape` provides common scale-shape calculations for tensor,
channel, and block granularities. Quantized or scaled GEMMs are configured by
constructing `GemmInputConfig` directly with the appropriate custom dtype,
scale shape, and scale dtype. The tensor generator owns format-specific dtype
defaults, such as UE8M0 scales for MXFP4 and BF16 group scales for MXINT4.

Some backend entry points consume a physical layout different from the
operation-level default. For example, a backend may expect `B` as `[K, N]` while
`GemmInputs` defaults to the semantic right-hand operand shape `[N, K]`. That
layout translation belongs at the consumer boundary while the comparison can
still use `gemm_reference(values)`.

## Verification

The generator verifies logical dimensions, required output dtype,
layout-derived physical shapes, scale sidecar relationships, and custom dtype
scale requirements. It rejects configurations that would produce meaningless
inputs, such as scales for skipped operands, MXFP4 storage without compatible
UE8M0 scales, NVFP4 storage without compatible local scales, or MXINT4 storage
without compatible BF16 group scales.
