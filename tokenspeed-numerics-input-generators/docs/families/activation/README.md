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

## Generated Values

Plain activation values are returned as tensors. Fused quantized paths return
both floating-point activation inputs and the scale metadata needed to interpret
the quantized output. Value ranges are intentionally bounded where the operation
feeds quantization so tests exercise the fused computation rather than avoidable
overflow or saturation.

Consumers should pass generated values through small adapters when a kernel uses
backend-specific argument names. The generator contract is the mathematical
activation input shape and dtype relationship, not the kernel ABI.
