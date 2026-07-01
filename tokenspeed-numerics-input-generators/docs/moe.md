# MoE Input Generators

MoE-family generators cover routed mixture-of-experts inputs. The operation
family is defined by hidden states, router outputs, selected expert ids, expert
weights, and metadata used to dispatch token-expert work to expert-local GEMMs.
The generators describe those operation-level inputs independently of any one
fused MoE implementation.

TokenSpeed kernels currently guide the first covered modes, but TokenSpeed
registry signatures are adapter concerns. A consumer can use the generated
values to call a fused kernel, a reference implementation, or a layer-level test
that performs routing and expert computation itself.

## Generators

`MoeInputs` generates layer-level routed MoE inputs:

- `hidden_states`: token activations with shape `[num_tokens, hidden_size]`.
- `router_logits`: routing logits with shape `[num_tokens, num_experts]`.
- `topk_ids`: selected expert ids with shape `[num_tokens, top_k]`.
- `topk_weights`: normalized selected-expert routing weights with shape
  `[num_tokens, top_k]`.
- `w13` and `w2`: nested GEMM values for the gate/up and down expert weights.
- Optional expert biases.
- Optional per-expert activation scales for quantized W13 and W2 projection
  inputs.

Generated `topk_ids` and `topk_weights` are derived from `router_logits` by
softmax, top-k selection, and selected-weight renormalization. This keeps the
default generated routing metadata consistent with the logits while still
making the explicit top-k tensors available to consumers that test precomputed
routing paths.

The nested weight generators reuse the GEMM family so dense, scaled, MXFP4, and
MXINT4 weights share the same dtype and scale handling as standalone GEMMs.
Activation operands are skipped for the nested weight GEMMs because the layer
computation produces those intermediate activations.

Quantized MoE implementations may need scale tensors for activations entering
the gate/up projection and the down projection. `MoeInputs` can generate those
per-expert positive scale tensors when `activation_scale_dtype` is configured.
The scales describe quantized projection inputs at the operation boundary;
packing, precision-config objects, or backend-specific scale layout transforms
remain adapter concerns.

`MoeAlignBlockSizeInputs` generates inputs for the expert block-alignment
metadata operation. This operation flattens top-k expert selections, groups the
flattened token-expert slots by expert, and pads each expert group to a multiple
of `block_size`. The output metadata identifies which flattened token slots are
processed by each expert-local GEMM block.

`MoESoftmaxTopKRoutingInputs` generates inputs for a router operation that
converts router logits to probabilities, selects experts using a correction
bias, and emits top-k expert ids and route weights. The operation is:

1. `probs = softmax(logits, dim=-1)`.
2. Select top-k expert ids from `probs + correction_bias`.
3. Gather output weights from the original `probs`, not the biased scores.
4. Optionally renormalize selected weights by their selected-weight sum.
5. Multiply selected weights by `scaling_factor`.
6. Replace selected expert ids greater than or equal to `num_experts_real` with
   `-1` to represent padded zero experts.

The generated correction bias can intentionally force a padded expert into the
selected top-k set. That gives consumers coverage for both ordinary expert
selection and zero-expert masking without making the generator depend on any
particular fused routing kernel.

`MoEBiasedGroupedTopKInputs` generates inputs for sigmoid/correction-bias MoE
routing with optional expert-group filtering. The operation is:

1. `scores = sigmoid(gating_output)`.
2. `selection_scores = scores + correction_bias`.
3. Partition experts into `num_expert_groups` equal-size groups.
4. Score each group by summing the top two `selection_scores` in that group.
5. Select `top_k_groups` groups and mask experts outside those groups.
6. Select `top_k` experts from the remaining `selection_scores`.
7. Gather route weights from the original sigmoid `scores`.
8. If `renormalize` is enabled, normalize selected weights by their selected
   sum and multiply by `routed_scaling_factor`.
9. Optionally map logical expert ids through a generated physical-id
   permutation and optionally mark padded token rows with output ids `-1`.

The generated `hidden_states` tensor represents the token rows associated with
the router logits. Its values do not affect the grouped top-k computation, but
it keeps the generated values aligned with implementation APIs that route a
hidden-state batch.

`MoESoftplusSqrtTopKRoutingInputs` generates inputs for a DeepSeek-style router
that transforms each logit with `sqrt(softplus(x))`, normalizes selected
weights, and applies a routed scaling factor. The generator supports two
selection modes:

- In correction-bias mode, select top-k experts from
  `sqrt(softplus(logits)) + correction_bias`.
- In hash mode, use `input_ids` to gather selected expert ids from
  `hash_indices_table`; the transformed logits only determine the normalized
  weights for those selected experts.

Both modes gather output weights from the un-biased transformed scores,
normalize the selected weights by their selected sum, and multiply by
`routed_scaling_factor`.

## References

`moe_reference` implements the routed layer computation for generated MoE
values. For each token and selected expert it applies the expert gate/up
projection, computes the gated activation, applies the expert down projection,
scales the result by the selected routing weight, and sums across selected
experts. Dense weights and generated scaled, MXFP4, or MXINT4 weight values are
normalized through the GEMM operand semantics before the reference matmuls.

`moe_align_block_size_reference` implements the block-alignment metadata
semantics directly:

1. Flatten `topk_ids` so each selected token-expert pair has one slot id.
2. Collect slot ids for each expert.
3. Pad each expert's slot list with the sentinel id `topk_ids.numel()` until the
   expert's slot count is divisible by `block_size`.
4. Emit concatenated `sorted_token_ids`, one `expert_id` per block, and the
   total padded token count.

`canonicalize_moe_align_block_size` packs those outputs into a deterministic
comparison tensor while ignoring intra-block ordering differences that can arise
from parallel implementations.

`moe_softmax_topk_routing_reference` implements the softmax, correction-bias
selection, optional selected-weight renormalization, scaling, and padded-expert
masking semantics described above. Ties are resolved by selecting the smaller
expert id first so reference output is deterministic.

`moe_biased_grouped_topk_reference` implements sigmoid scoring, grouped
candidate filtering, top-k expert selection, optional renormalization/scaling,
logical-to-physical expert id mapping, and padded-token output id masking.

`moe_softplus_sqrt_topk_routing_reference` implements both correction-bias and
hash-table softplus-sqrt routing. Non-hash ties are resolved by smaller expert
id first for deterministic references. Hash-table routing preserves the expert
order stored in the selected table row.

## TokenSpeed API Mapping

TokenSpeed fused MoE kernels consume a runtime weight module plus a plan created
by `moe_plan`. That module is a backend adapter concern. `MoeInputs` produces
the operation-level tensors: hidden states, router logits, explicit top-k
routing results, expert weights with optional scale sidecars, optional biases,
and optional activation scales. Tests that exercise TokenSpeed kernels should
build a small adapter module from those values, call `moe_process_weights` for
the selected backend, and then pass the generated hidden/routing tensors to
`moe_apply`.

For example, the MXFP4 Triton precomputed-routing path uses `MoeInputs` with
MXFP4 expert weights and passes generated `topk_ids` and `topk_weights`
directly to `moe_apply`. The generator does not own Triton-specific weight
swizzling, precision-config objects, or module attributes such as `top_k` and
`num_experts`; those remain in the TokenSpeed adapter/test layer.

Dense TokenSpeed MoE adapters follow the same pattern with ordinary torch
weight tensors. `MoeInputs` provides the semantic expert weights in
`w13.B` and `w2.B`; the adapter attaches them to a small module, calls
`moe_process_weights` for the selected backend, and then compares `moe_apply`
against `moe_reference(values)`. Backend-specific gate/up reordering is owned
by `moe_process_weights`, not by the generator.

The FlashInfer TRT-LLM MXINT4 path is a weight-only INT4 variant with BF16
group scales. `MoeInputs(weight_format="mxint4")` generates the operation-level
packed signed INT4 bytes and BF16 scales. A TokenSpeed adapter can convert
those bytes into checkpoint-style int32 words, attach the BF16 scale tensors to
the runtime weight module, and let `moe_process_weights` handle backend block
layout conversion and scale interleaving.

The CUDA `routing_flash` helper maps directly to
`MoESoftmaxTopKRoutingInputs`: generated `logits`, `correction_bias`, output
buffers, `num_experts_real`, `scaling_factor`, and `renormalize` become the
helper arguments. The generator keeps the routing operation independent of that
helper's supported expert counts and extension-loading details; those remain
adapter/test concerns.

The Triton `minimax_biased_grouped_topk` helper maps directly to
`MoEBiasedGroupedTopKInputs`: generated `hidden_states`, `gating_output`,
`correction_bias`, `top_k`, `renormalize`, `num_expert_groups`,
`top_k_groups`, `routed_scaling_factor`, optional
`num_token_non_padded`, and optional `logical_to_physical_map` become the
helper arguments. The generator describes the grouped routing operation; the
wrapper's fast-path restrictions, such as specific `top_k` and group settings,
remain implementation-specific details.

The CUDA `softplus_sqrt_topk_flash` and `hash_softplus_sqrt_topk_flash` helpers
map to `MoESoftplusSqrtTopKRoutingInputs`. The non-hash helper consumes
generated `logits`, `correction_bias`, output buffers, `routed_scaling_factor`,
and `renormalize`. The hash helper consumes generated `logits`, `input_ids`,
`hash_indices_table`, output buffers, `routed_scaling_factor`, and
`renormalize`. TokenSpeed's CUDA helpers currently require FP32 logits, int32
output ids, `top_k=6`, `renormalize=True`, and 256 or 384 experts; those are
adapter requirements rather than additional operation semantics.

## Verification

MoE configs verify token counts, hidden/intermediate widths, expert counts,
top-k constraints, block sizes, integer routing dtypes, and generated id
ranges. Optional activation scales must use regular floating dtypes and
positive finite scalar fill values. MXINT4 weights require BF16 group scales
with shapes derived from the expert projection widths. The layer reference
checks that selected expert ids and weights are rank-2, shape-consistent,
finite, non-negative, duplicate-free per token, and normalized across each
token's selected experts. The align-block-size reference also checks that
provided top-k ids are rank-2 and within `[0, num_experts)`, so invalid routing
metadata fails before reaching a kernel adapter.

Softmax top-k routing verifies that logits and correction bias are finite FP32
tensors with matching expert width, output buffers are rank-2 with matching
token/top-k shape, output indices use `torch.int32` or `torch.int64`, output
weights use FP32, `num_experts_real` identifies a proper prefix of real experts,
and `scaling_factor` is positive and finite.

Biased grouped top-k routing verifies finite floating hidden/router tensors,
finite FP32 correction bias, matching token/expert dimensions, equal-size expert
groups with at least two experts per group, selected-group capacity sufficient
for `top_k`, positive finite routed scaling, permutation validity for optional
logical-to-physical maps, and in-range scalar padding cutoffs.

Softplus-sqrt top-k routing verifies finite FP32 logits, rank-2 output buffers,
int32 output ids, FP32 output weights, positive finite routed scaling, and
`renormalize=True`. Correction-bias mode requires one finite FP32 bias per
expert. Hash mode requires int32/int64 input ids, an int32 hash table with
one unique in-range expert id per selected slot, and input ids that index valid
hash-table rows.
