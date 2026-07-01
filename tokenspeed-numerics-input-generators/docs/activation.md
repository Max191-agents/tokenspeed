# Activation Input Generators

Activation generators cover elementwise and fused activation helpers whose
inputs are ordinary floating point tensors plus, in one case, quantization
metadata implied by the operation shape.

## Operation Semantics

### Sigmoid Multiply

`SigmoidMulInputs` represents:

```text
y = x * sigmoid(gate)
```

`x` is a contiguous `[num_tokens, hidden_dim]` tensor. `gate` may either have
the same dense shape or be the strided `[num_tokens, num_heads, head_dim]` view
that comes from splitting packed QKV-style storage. In the strided case,
`num_heads * head_dim` must equal `hidden_dim`, and the generated backing
storage is kept in the values object so the view remains valid.

### Split Gated Activations

`GatedActivationInputs` represents split activations such as:

```text
gate, up = split(x, 2, dim=-1)
y = activation(gate) * up
```

The generator supports `silu`, exact `gelu`, and tanh-approximate `gelu_tanh`.
The generated `x` tensor has shape `[num_tokens, 2 * hidden_dim]`; the output
width is `hidden_dim`.

### Fused Gate Sigmoid Multiply Add

`FusedGateSigmoidMulAddInputs` represents:

```text
gate = sigmoid(hidden_states @ gate_weight)
y = final_hidden_states + gate * shared_output
```

`hidden_states`, `shared_output`, and `final_hidden_states` all have shape
`[num_tokens, hidden_dim]`. `gate_weight` has shape `[hidden_dim]`. The generator
scales `gate_weight` by `1 / sqrt(hidden_dim)` so the dot product remains in a
range where the sigmoid usually carries useful variation instead of saturating
for large hidden dimensions.

### Fused SwiGLU FP8 UE8M0

`FusedSwiGLUFP8UE8M0Inputs` represents:

```text
gate, up = split(gate_up, 2, dim=-1)
y = silu(gate) * up
q, scales = fp8_e4m3_quantize_with_ue8m0_group_scales(y)
```

The generated `gate_up` tensor has shape `[num_tokens, 2 * hidden_dim]`.
`hidden_dim` must be divisible by `group_size` because the quantized operation
computes one scale per token group.

The reference returns `(fp8_out, packed_scales)`. `fp8_out` has shape
`[num_tokens, hidden_dim]` and dtype FP8 E4M3. `packed_scales` has shape
`[num_tokens, ceil(num_groups / 4)]` with dtype int32, where each int32 packs up
to four UE8M0 biased exponent bytes for consecutive groups.

### Fused SwiGLU FP8 Block Quant

`FusedSwiGLUFP8BlockQuantInputs` represents:

```text
gate, up = split(gate_up, 2, dim=-1)
y = silu(gate) * up
q, scales = fp8_e4m3_block_quantize(y)
```

The dense variant generates `gate_up[num_tokens, 2 * hidden_dim]` and a
preallocated float32 `scale_out[num_tokens, hidden_dim / group_size]` buffer.
The expert-parallel variant generates
`gate_up[num_experts, max_tokens_per_expert, 2 * hidden_dim]` and
`scale_out[num_experts, max_tokens_per_expert, hidden_dim / group_size]`.
`num_tokens_per_expert` marks the valid token prefix for each expert, and
`num_tokens_hint` records the per-expert token capacity.

The reference returns `(fp8_out, scales)`. `fp8_out` has the same leading
dimensions as `gate_up` and last dimension `hidden_dim`; `scales` matches
`scale_out`. Invalid expert-padding rows are zeroed in the reference output.

### Fused SwiGLU NVFP4 Quant

`FusedSwiGLUNVFP4QuantInputs` represents:

```text
gate, up = split(gate_up, 2, dim=-1)
y = silu(gate) * up
packed, scales = nvfp4_quantize(y, global_scale)
```

The generated `gate_up` tensor has shape `[num_tokens, 2 * hidden_dim]`.
`global_scale` stores the inverse of the configured input scale because that is
the representation consumed by the TokenSpeed CUDA helper. The reference
returns packed E2M1 NVFP4 bytes with shape `[num_tokens, hidden_dim / 2]` and
linear FP8 E4M3 group scales with shape
`[num_tokens, hidden_dim / scale_size]`.

## Validation Contract

The activation generators reject invalid operation inputs before generation:

- token counts must be non-negative
- hidden dimensions, head dimensions, head counts, and group sizes must be
  positive
- qkv-split sigmoid gates require `num_heads * head_dim == hidden_dim`
- qkv-split sigmoid gates require enough KV-head metadata to construct the
  backing QKV row stride
- fused SwiGLU FP8/UE8M0 requires `hidden_dim % group_size == 0`
- fused SwiGLU FP8/UE8M0 references require a 2D `gate_up` tensor with an even
  last dimension and a positive `group_size`
- fused SwiGLU FP8 block quant requires `hidden_dim % group_size == 0` and a
  float32 scale buffer matching the generated block layout
- expert-parallel FP8 block quant requires one valid token count per expert,
  with counts inside the generated per-expert token capacity
- fused SwiGLU NVFP4 requires positive input scales, an even hidden dimension
  for packed output, and `hidden_dim % scale_size == 0`
- all generated tensor dtypes must be `torch.dtype` values accepted by the core
  tensor generator

These checks describe operation-level validity. Kernel-specific constraints,
such as which dtypes a particular backend accepts, remain adapter or test
concerns.

## TokenSpeed API Mapping

TokenSpeed currently exposes activation helpers directly rather than through
the registry numerics harness:

- `sigmoid_mul(x, gate)` consumes `SigmoidMulInputValues.x` and `.gate`
- `silu_and_mul(x)` consumes `GatedActivationInputValues.x` with
  `activation="silu"`
- FlashInfer `silu_and_mul(x)`, `gelu_and_mul(x)`, and
  `gelu_tanh_and_mul(x)` consume `GatedActivationInputValues.x` with the
  corresponding split-gated activation configuration
- `fused_gate_sigmoid_mul_add(...)` consumes the four tensors from
  `FusedGateSigmoidMulAddInputValues`
- `fused_swiglu_fp8_ue8m0(gate_up, swiglu_limit)` consumes
  `FusedSwiGLUFP8UE8M0InputValues.gate_up` and `.swiglu_limit`
- `silu_and_mul_fuse_block_quant(gate_up, scale_out, ...)` consumes
  `FusedSwiGLUFP8BlockQuantInputValues.gate_up`, `.scale_out`, and, for
  expert-parallel calls, `.num_tokens_per_expert`, `.num_tokens_hint`, and
  `.num_experts`
- `silu_and_mul_fuse_nvfp4_quant(gate_up, global_scale)` consumes
  `FusedSwiGLUNVFP4QuantInputValues.gate_up` and `.global_scale`

Adapters should remain small because the generated values already match the
operation-level tensor inputs.
