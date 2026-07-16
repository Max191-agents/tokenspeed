# MoE Input Generators

The MoE family is organized around the mathematical stages of a routed
SwiGLU-style mixture of experts. Let:

- `T` be the number of input tokens.
- `H` be the model hidden size.
- `E` be the number of routed experts.
- `K` be the number of selected experts per token.
- `I` be the expert intermediate size.
- `R = T * K` be the number of routed token-expert rows.

The package provides one generator for each fundamental stage and one generator
for the external operands of the full fused operation.

## Routing

`MoeRoutingInputs` generates:

- `hidden_states`: `[T, H]`.
- `router_weight`: `[E, H]`.
- Optional `router_bias`: `[E]`.
- Optional `correction_bias`: `[E]`, used for selection but not route weights.

Router weights are generated with a `1 / sqrt(H)` scale so executing the router
projection produces a useful score distribution as hidden size grows. The
generator does not execute that projection or top-k selection. Router logits,
selected expert IDs, and route weights are derived results owned by the caller's
reference or implementation.

## Dispatch

`MoeDispatchInputs` generates the inputs at the token-dispatch boundary:

- `hidden_states`: `[T, H]`.
- `topk_ids`: `[T, K]` unique IDs in `[0, E)`.
- `topk_weights`: `[T, K]` normalized route weights.

The top-k tensors are valid generated inputs here because routing has already
happened at the dispatch boundary. They are generated independently rather than
being computed from `hidden_states`.

The generator does not choose a physical dispatch representation. Flattened
slot IDs, expert offsets, block padding, permutations, inverse permutations,
and distributed exchange layouts are implementation concerns derived from
these values.

## Gate And Up

`MoeGateUpInputs` generates the inputs to the first expert projection:

- `routed_hidden_states`: `[R, H]`.
- `expert_ids`: `[R]`.
- `w13`: nested GEMM values containing expert weights logically shaped
  `[E, 2I, H]` plus optional scale sidecars.
- Optional `w13_bias`: `[E, 2I]`.
- Optional per-expert activation scale: `[E]`.

The represented computation splits the W13 result into gate and up halves and
forms an activation such as `silu(gate) * up`. That resulting `[R, I]` tensor
is an output of this stage, so it is generated as an input by the next stage
rather than returned here.

## Down And Combine

`MoeDownCombineInputs` generates:

- `activations`: `[R, I]`, representing post-gate/up activation rows.
- `topk_ids`: `[T, K]`.
- `topk_weights`: `[T, K]`.
- `w2`: nested GEMM values containing expert weights logically shaped
  `[E, H, I]` plus optional scale sidecars.
- Optional `w2_bias`: `[E, H]`.
- Optional per-expert activation scale: `[E]`.
- Optional `shared_output`: `[T, H]`.

Flattening `topk_ids` associates each activation row with its expert. The down
projection produces `[R, H]` expert outputs. Route weights then scale those
rows before they are reduced back into `[T, H]` token order. A shared output,
when present, is added after the routed reduction.

## Full Fused MoE

`MoeInputs` generates the external operands for the complete fused routed MoE:

- Nested `routing` values containing hidden states, router projection tensors,
  and optional routing biases.
- `w13` and `w2` expert weight operands with optional scale sidecars.
- Optional W13 and W2 expert biases.
- Optional per-expert W13 and W2 activation scales.

The full generator calls `MoeRoutingInputs` and reuses GEMM generation for the
expert weights. It does not generate router logits, selected routes, routed
hidden states, or post-gate/up activations because those are results produced
during full execution.

## Dtypes And Quantization

Regular floating torch dtypes are supported for activations, routing tensors,
and dense expert weights. Quantized expert weights use the same custom dtypes
as core tensor and GEMM generation. Scale storage and block shapes are inferred
from the custom dtype:

- `CustomDType.MXFP4`: packed E2M1 values with UE8M0 scales over 32 values.
- `CustomDType.MXINT4`: packed signed INT4 values with BF16 scales over 32
  values.
- `CustomDType.NVFP4`: packed E2M1 values with FP8 scales over 16 values.

Scaled regular torch weights can provide an explicit scale dtype and block
size. Projection input dimensions must be divisible by the selected scale
block size so generated storage cannot represent a partial quantization group.

## TokenSpeed Adapters

TokenSpeed's fused MoE API begins after the router projection: it consumes
hidden states, router logits or precomputed top-k values, and a runtime expert
weight module. Its numerics adapter therefore executes router projection and
top-k selection from `MoeInputs.routing`, then builds the runtime module from
`w13`, `w2`, biases, and scales.

Standalone helper kernels use the nearest core stage:

- Block alignment and FP8 staging start from `MoeDispatchInputs`.
- Router helpers start from `MoeRoutingInputs`; helper-specific output buffers,
  padded-expert masks, hash tables, or physical expert maps are supplied by the
  TokenSpeed test adapter.
- Finalization helpers start from `MoeDownCombineInputs`; the adapter executes
  or constructs down-projection outputs and converts logical route order into
  the helper's permutation format.

This keeps backend names, fused helper boundaries, output buffers, and physical
layouts out of the standalone input-generation package.

## Verification

All generators reject invalid token, hidden, intermediate, expert, and top-k
sizes. They verify floating and integer dtypes, selected ID ranges, weight scale
block divisibility, and the required shapes and finite values of every generated
tensor. Stage generators that accept top-k routing inputs ensure that each row
contains unique expert IDs and finite non-negative weights. Quantized weight
configuration is also validated by the shared tensor and GEMM generators before
values are returned.
