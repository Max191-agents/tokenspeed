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
