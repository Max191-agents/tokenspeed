# MoE Generators

MoE generators describe routed mixture-of-experts layers and metadata
preparation. A routed MoE layer maps each token to one or more experts, applies
expert-local gate/up and down projections, and combines the expert outputs using
route weights.

## Generators

- `MoeInputs`: generates hidden states, router logits, top-k routing ids and
  weights, expert W13/W2 GEMM operands, optional biases, and optional
  per-expert activation scales for quantized projection inputs.
- `MoeAlignBlockSizeInputs`: generates top-k expert ids and block-size metadata
  for expert-local token grouping and padding.
- `MoESoftmaxTopKRoutingInputs`: generates FP32 router logits, correction bias,
  and output buffers for softmax/correction-bias top-k routing with padded
  zero-expert masking.
- `MoEBiasedGroupedTopKInputs`: generates hidden-state rows, sigmoid router
  logits, correction bias, optional expert-id maps, and optional padding
  cutoffs for grouped top-k routing.

`MoeInputs` composes the GEMM generator for expert weights. Dense, MXFP4, and
MXINT4 expert weights are operation-level choices; backend preprocessing such
as preshuffling, checkpoint repacking, or registry precision configs belongs in
adapters.

## Generated Values

Generated routing uses router logits to derive valid top-k ids and normalized
route weights. References validate that selected experts are in range, route
weights are finite and normalized, and expert weight shapes agree with hidden
and intermediate dimensions. Optional activation scales are generated as
positive per-expert tensors and are intended for adapters that quantize the
projection activations before calling a backend kernel.

MXINT4 MoE weights are generated as packed signed INT4 bytes with BF16 group
scales. Backend adapters can repack those bytes into checkpoint or block-major
layouts without changing the operation-level generator contract.

Softmax top-k routing values describe the standalone router operation used to
produce selected expert ids and scaled route weights. Expert ids greater than
or equal to `num_experts_real` are generated and referenced as padded experts
that map to `-1` in the output ids.

Biased grouped top-k routing values describe the MiniMax/DeepSeek-style router
that scores experts with `sigmoid(gating_output) + correction_bias`, filters
candidates by selected expert groups, and returns selected ids plus weights from
the original sigmoid scores.

These generators intentionally describe operation-level routed computations
rather than any one fused MoE kernel signature.
