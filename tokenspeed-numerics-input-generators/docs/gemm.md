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
tensors, dequantizes MXFP4 storage with UE8M0 scales, dequantizes NVFP4
storage with per-block local scales, and dequantizes MXINT4 storage with BF16
group scales before multiplying. Kernel tests can use it as the semantic target
while keeping backend-specific argument mapping in the adapter or test.

Some backend entry points consume a different physical layout than the
operation-level default. For example, the TokenSpeed Triton scaled-FP8 GEMM path
accepts `B` as `[K, N]`, while `GemmInputs` defaults to the semantic right-hand
operand shape `[N, K]` used by `A @ B.T`. A kernel adapter can transpose the
generated `B` tensor for that call while still comparing against
`gemm_reference(values)`. That layout translation belongs at the consumer
boundary, not in the generator definition.

`RouterProjectionInputs` represents the MoE routing projection:

```text
router_logits = hidden_states @ router_weights.T
```

`hidden_states` has shape `[num_tokens, hidden_dim]`, `router_weights` has
shape `[num_experts, hidden_dim]`, and the reference returns fp32 logits with
shape `[num_tokens, num_experts]`. The generator is still operation-level: it
does not encode a particular router kernel launch shape or backend ABI. TokenSpeed
adapters can pass the generated tensors to CUDA helpers such as fp32 router GEMM
or DeepSeek-V3 router GEMM when their backend-specific dtype and shape
requirements are satisfied.

Router projection generation uses ordinary floating-point tensor generation for
hidden states and weights, then scales weights by `1 / sqrt(hidden_dim)` by
default. That keeps generated logits roughly O(1), which is more useful for
downstream routing checks than arbitrary unscaled random weights whose logits
grow with the reduction dimension. Callers may override the hidden or weight
scale when intentionally testing a different distribution.

`router_projection_reference` computes the fp32 projection reference directly
from the generated values.

`LMHeadProjectionInputs` represents the final language-model head projection:

```text
logits = hidden_states @ weight.T
```

`hidden_states` has shape `[num_tokens, hidden_dim]`, `weight` has shape
`[vocab_size, hidden_dim]`, and the reference returns logits with shape
`[num_tokens, vocab_size]`. Tensor-parallel consumers can interpret
`vocab_size` as the local vocabulary shard size. TokenSpeed's fused CUDA
LM-head path currently consumes contiguous BF16 tensors and returns BF16 output,
but the generator describes the projection operation rather than baking in the
compiled backend shape table.

Like router projection, LM-head generation scales weights by
`1 / sqrt(hidden_dim)` by default so logits stay O(1) as hidden width changes.
This keeps sampled vocabulary projections numerically useful without requiring
tests to manually tune distributions for each model shape.

`lm_head_projection_reference` computes the projection in fp32 and casts to the
requested output dtype, defaulting to BF16 to match the fused TokenSpeed
LM-head wrapper.

## Scale And Quantized Storage

Dense floating-point tensors use the core tensor generator. Scaled tensors use
the same tensor generator with explicit scale shape and scale dtype. Custom
dtypes, such as MXFP4 storage with UE8M0 scales, NVFP4 storage with per-block
local scales, and MXINT4 storage with BF16 group scales, are represented through
the shared custom dtype enum so the format-specific constraints are
centralized.

The dense case is the baseline operation: `A` and `B` are generated as ordinary
floating-point tensors with no sidecar scales, and `gemm_reference` computes the
same `A @ B.T` operation. Scaled and quantized modes should preserve that
operation-level meaning while changing only how the operand values are stored
and interpreted.

`gemm_scale_shape` provides common scale-shape calculations for tensor,
channel, and block granularities. `mxfp4_gemm_input_config` builds a GEMM config
for MXFP4 operands with block scales. `mxint4_gemm_input_config` builds the
same row-major operation-level layout for signed INT4 operands with BF16 group
scales.

MXINT4 is modeled as weight-like signed INT4 storage: every byte stores two
two's-complement 4-bit values, and each BF16 scale covers one logical 32-value
K group. This is the semantic representation before backend adapters perform
checkpoint packing, block-major conversion, or scale interleaving.

For block-scaled FP8 GEMM, generated storage values are interpreted as FP8
values multiplied by a scale grid. `A` commonly uses one scale row per logical
`M` row and one scale group per `K` block, while `B` may use a 2D grid over
logical `N` blocks and `K` blocks. The reference expands those grids over the
logical operand before computing `A @ B.T`. The current generator path expects
regular scale grids that evenly partition the generated physical operand shape.

`mxfp8_gemm_input_config` builds this standard row-major block-scaled FP8
configuration. It requires `K` to be divisible by the K block and `N` to be
divisible by the N block so the scale grids describe complete logical operand
regions without backend-specific padding. TokenSpeed Triton and DeepGEMM
adapters can consume the generated `A`, `B`, `A_scales`, and `B_scales`
directly, while keeping any required scale-layout conversion at the adapter
boundary.

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
produce meaningless inputs, such as scales for skipped operands, MXFP4 storage
without compatible UE8M0 scales, or MXINT4 storage without compatible BF16
group scales. Layout names are validated explicitly. The current MXFP4 and
MXINT4 GEMM definitions are row-major operation layouts, so custom `A` operands
require `MK` layout and custom `B` operands require `NK` layout.

The fused NVFP4 generator also verifies the fixed NVFP4 scale group width,
source dtype, divisibility of K and intermediate dimensions by the scale group,
and scalar global-scale relationships.

Router projection verification checks positive token, hidden, and expert
dimensions; supported floating-point dtypes for hidden states and weights;
finite positive generation scales; rank-2 generated tensors; matching hidden
dimensions; and finite generated values.

LM-head projection verification checks positive token, hidden, and vocabulary
dimensions; supported floating-point dtypes for hidden states, weights, and
reference output; finite positive generation scales; rank-2 generated tensors;
matching hidden dimensions; and finite generated values.
