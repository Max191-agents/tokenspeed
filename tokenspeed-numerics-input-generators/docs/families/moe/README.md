# MoE Family

The MoE family models the four fundamental stages of routed expert
computation and the external operands of the complete fused operation.

## Generators

- `MoeRoutingInputs`: hidden states, router projection weights, and optional
  router and correction biases. It does not execute routing.
- `MoeDispatchInputs`: hidden states and precomputed top-k routing metadata
  entering token-to-expert dispatch.
- `MoeGateUpInputs`: routed hidden-state rows, selected expert IDs, W13 expert
  weights, and optional gate/up bias and quantization scales.
- `MoeDownCombineInputs`: activated expert rows, top-k routing metadata, W2
  expert weights, and optional down bias, quantization scales, and shared
  output.
- `MoeInputs`: all external operands for a full fused routed MoE, composed from
  `MoeRoutingInputs` and expert W13/W2 weight generation.

Router logits and top-k selections are derived results and are intentionally not
returned by `MoeRoutingInputs` or `MoeInputs`. Consumers execute their routing
strategy after generation. Expert weights support regular torch dtypes and the
quantized custom dtypes handled by core tensor generation.

Kernel-specific staging buffers, block-alignment metadata, permutation
layouts, and finalization buffers are adapter concerns. Tests for those kernels
should derive them from the relevant core stage generator.

See the detailed MoE documentation and class docstrings for tensor shapes,
configuration fields, and verification guarantees.
