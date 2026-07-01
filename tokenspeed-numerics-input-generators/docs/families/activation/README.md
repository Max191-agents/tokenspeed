# Activation Generators

Activation generators model pointwise nonlinear functions used inside inference
kernels. The operation semantics are elementwise: generated tensor operands must
share the shape relationships required by the activation formula, and references
compute the same mathematical expression without assuming a specific backend
kernel signature.

## Generators

- `SigmoidMulInputs`: generates `x` and `gate` for `x * sigmoid(gate)`.
- `GatedActivationInputs`: generates gate/up operands for `silu`, `gelu`, or
  `swiglu`-style gated activations.
- `FusedGateSigmoidMulAddInputs`: generates gate, up, and residual operands for
  a fused gate/sigmoid/multiply/add expression.
- `FusedSwiGLUFP8UE8M0Inputs`: generates SwiGLU inputs plus FP8 output scale
  metadata for fused activation and quantization paths.
- `FusedSwiGLUFP8BlockQuantInputs`: generates SiLU+Mul inputs plus preallocated
  float32 block-scale buffers for FP8 E4M3 block quantization, including the
  expert-parallel tensor/metadata variant.
- `FusedSwiGLUNVFP4QuantInputs`: generates dense SiLU+Mul inputs and the global
  inverse input scale used by NVFP4 quantization paths.

## Fused Quantized SiLU+Mul

The fused quantized activation generators model a two-stage operation:

```text
activated = silu(gate) * up
quantized = quantize(activated)
```

`FusedSwiGLUFP8BlockQuantInputs` uses one float32 scale for each contiguous
`group_size` block of activated values. The expert-parallel variant uses
`gate_up[num_experts, max_tokens_per_expert, 2 * hidden_dim]` plus
`num_tokens_per_expert` metadata to describe the valid token prefix for each
expert.

`FusedSwiGLUNVFP4QuantInputs` uses packed E2M1 NVFP4 values with one FP8 E4M3
scale for each group of 16 values and a tensor-wide inverse input scale. The
reference returns linear group scales; backend-specific swizzled/padded scale
layouts are adapter concerns.

## Generated Values

Plain activation values are returned as tensors. Fused quantized paths return
both floating-point activation inputs and the scale metadata needed to interpret
the quantized output. Value ranges are intentionally bounded where the operation
feeds quantization so tests exercise the fused computation rather than avoidable
overflow or saturation.

Consumers should pass generated values through small adapters when a kernel uses
backend-specific argument names. The generator contract is the mathematical
activation input shape and dtype relationship, not the kernel ABI.
