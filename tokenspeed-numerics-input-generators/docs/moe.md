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
- `w13` and `w2`: nested GEMM values for the gate/up and down expert weights.
- Optional expert biases.

The nested weight generators reuse the GEMM family so dense, scaled, and MXFP4
weights share the same dtype and scale handling as standalone GEMMs. Activation
operands are skipped for the nested weight GEMMs because the layer computation
produces those intermediate activations.

`MoeAlignBlockSizeInputs` generates inputs for the expert block-alignment
metadata operation. This operation flattens top-k expert selections, groups the
flattened token-expert slots by expert, and pads each expert group to a multiple
of `block_size`. The output metadata identifies which flattened token slots are
processed by each expert-local GEMM block.

## References

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

## Verification

MoE configs verify token counts, expert counts, top-k constraints, block sizes,
integer routing dtypes, and generated id ranges. The align-block-size reference
also checks that provided top-k ids are rank-2 and within `[0, num_experts)`, so
invalid routing metadata fails before reaching a kernel adapter.
