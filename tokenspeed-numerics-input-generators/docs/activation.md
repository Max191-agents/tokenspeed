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
- `fused_gate_sigmoid_mul_add(...)` consumes the four tensors from
  `FusedGateSigmoidMulAddInputValues`
- `fused_swiglu_fp8_ue8m0(gate_up, swiglu_limit)` consumes
  `FusedSwiGLUFP8UE8M0InputValues.gate_up` and `.swiglu_limit`

Adapters should remain small because the generated values already match the
operation-level tensor inputs.
