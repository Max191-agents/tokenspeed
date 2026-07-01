# Communication Input Generators

Communication generators cover collective and routing operations over a group
of ranks. The operation semantics are independent of any particular transport
or backend: each rank contributes rank-local tensors or metadata, and the
communication operation defines how those inputs are combined and redistributed.

This slice covers sum all-reduce, all-gather, sum reduce-scatter, fused sum
all-reduce plus residual RMSNorm, fused reduce-scatter plus residual RMSNorm,
fused all-gather plus dual RMSNorm, MiniMax Q/K all-reduce plus RMSNorm,
expert-parallel MoE dispatch/combine routing, and batch-DP sampling
communication.

## Operation Semantics

### Sum All-Reduce

`AllReduceInputs` represents a collective where every rank contributes one
tensor with the same shape and dtype:

```text
out[rank] = sum(input[peer] for peer in ranks)
```

Every rank receives the same result. The reference returns that shared tensor.

### Fused Sum All-Reduce + Residual RMSNorm

`AllReduceResidualRMSNormInputs` represents a common distributed normalization
pattern where every rank contributes one rank-local activation tensor, the
activation tensors are summed across ranks, and each rank then combines the
shared reduced value with its own residual before applying RMSNorm:

```text
reduced = sum(input[peer] for peer in ranks)
residual_out[rank] = reduced + residual[rank]
norm_out[rank] = residual_out[rank]
                 * rsqrt(mean(residual_out[rank]^2, dim=-1) + eps)
                 * weight
```

The rank-local input and residual tensors have shape
`[num_tokens, hidden_size]`. The RMSNorm weight has shape `[hidden_size]` and is
shared by all ranks. The reference returns both per-rank outputs because fused
implementations commonly expose the normalized output and the updated residual.

### All-Gather

`AllGatherInputs` represents gathering token shards from every rank:

```text
out = concat(input[0], input[1], ..., input[world_size - 1], dim=0)
```

Each rank's local shard has shape `[tokens_per_rank[rank], hidden_size]`.
`tokens_per_rank` is generated from `total_tokens`, `world_size`, and optional
`max_tokens_per_rank` using `metadata_seed`.

### Fused All-Gather + Dual RMSNorm

`AllGatherDualRMSNormInputs` represents the MLA-style pattern where every rank
contributes Q/KV/RoPE token shards, all shards are gathered, and two low-rank
slices are normalized independently:

```text
gathered = concat(qkv[0], qkv[1], ..., qkv[world_size - 1], dim=0)
q_norm = rmsnorm(gathered[:, :q_lora_rank], q_weight, eps_q)
kv_norm = rmsnorm(
  gathered[:, q_lora_rank : q_lora_rank + kv_lora_rank],
  kv_weight,
  eps_kv,
)
```

The generated row layout is
`[q_lora_rank, kv_lora_rank, qk_rope_head_dim]`. The reference returns the
gathered tensor with the normalized KV slice written back into place, plus the
separate normalized Q and KV outputs. Backend-specific workspace setup and
optional FP8 quantization epilogues are adapter concerns.

### Sum Reduce-Scatter

`ReduceScatterInputs` represents a collective where every rank contributes a
full token tensor, all tensors are summed, and each rank receives one contiguous
token shard:

```text
reduced = sum(input[peer] for peer in ranks)
out[rank] = reduced[offset[rank] : offset[rank] + tokens_per_rank[rank]]
```

`tokens_per_rank` is generated using the same metadata policy as all-gather, so
tests can exercise uneven token distributions while preserving a fixed total
amount of work.

### Fused Sum Reduce-Scatter + Residual RMSNorm

`ReduceScatterResidualRMSNormInputs` represents a fused reduce-scatter epilogue
where the summed token tensor is scattered using the standard near-even rank
partition, then each rank adds local residual state and applies RMSNorm:

```text
reduced = sum(input[peer] for peer in ranks)
shard[rank] = reduced[offset[rank] : offset[rank] + tokens_per_rank[rank]]
residual_out[rank] = shard[rank] + residual[rank] + optional_add_in[rank]
norm_out[rank] = rmsnorm(residual_out[rank], weight, eps)
```

The deterministic near-even partition matches common reduce-scatter wrapper
contracts where `total_tokens` alone determines each rank's output row count.
The optional `add_in` tensor models fused add+residual modes without changing
the core operation definition.

### MiniMax Q/K All-Reduce + RMSNorm

`MiniMaxAllReduceQKRMSNormInputs` represents the MiniMax M2 communication
pattern where Q and K projection shards are all-reduced independently before
RMSNorm:

```text
reduced_q = sum(q_input[peer] for peer in ranks)
reduced_k = sum(k_input[peer] for peer in ranks)
q_norm[rank] = rmsnorm(reduced_q, q_weight, eps)
k_norm[rank] = rmsnorm(reduced_k, k_weight, eps)
```

The all-reduce result is shared by every rank, so the reference returns the
same normalized Q/K tensors for each rank. TokenSpeed's MiniMax path is
specialized for global Q width 6144 and global K width 1024, with those widths
sharded evenly across `world_size` in `{2, 4, 8, 16}`. RMSNorm weights are bf16
because that is the operation ABI for the fused path. The generator can also
produce padded-row-stride Q/K views to exercise the API mode where Q and K are
slices of larger projection storage.

### Expert-Parallel MoE Dispatch/Combine

`ExpertParallelRoutingInputs` represents the routing semantics behind
expert-parallel MoE all-to-all communication. Each source rank owns a set of
input token hidden states. Each token has top-k selected expert ids and top-k
weights. Experts are partitioned evenly and contiguously across ranks:

```text
owner_rank(expert_id) = expert_id // (num_experts / world_size)
```

Dispatch sends each token once to every rank that owns at least one of the
token's selected experts. If two selected experts are on the same target rank,
the token is still dispatched once to that rank, with metadata identifying the
local selected expert slots.

Combine returns expert outputs to the token's source rank and computes the
weighted MoE contribution:

```text
combined[source_rank, token] =
  sum(expert_output[source_rank, token, slot] * topk_weight[source_rank, token, slot])
```

The generator also creates synthetic per-slot expert outputs so combine can be
tested independently of the expert MLP implementation.

## Validation Contract

The generators reject invalid collective descriptions before returning values:

- `world_size`, tensor dimensions, and hidden size must be positive
- total token counts must be non-negative
- generated token distributions sum to `total_tokens`
- optional per-rank token caps must be large enough to hold `total_tokens`
- collectives currently generate regular floating torch dtypes only
- all rank-local tensors for a collective have compatible shapes and dtypes
- fused residual RMSNorm requires matching input and residual shapes, a
  rank-1 weight with length `hidden_size`, and non-negative `eps`
- fused reduce-scatter residual RMSNorm requires the summed input row count to
  match `sum(tokens_per_rank)` and each rank's residual/add tensors to match
  that rank's scattered shard shape
- fused all-gather dual RMSNorm requires Q/KV/RoPE widths to match every
  gathered row, rank-local shard sizes to match `tokens_per_rank`, and separate
  rank-1 Q/KV RMSNorm weights with matching lengths
- MiniMax Q/K all-reduce RMSNorm requires supported world sizes, fixed global
  Q/K widths, bf16 RMSNorm weights, and valid row strides for dense or padded
  Q/K views
- expert-parallel routing requires an even expert partition across ranks,
  unique in-range top-k ids per token, normalized top-k weights, and compatible
  hidden/expert-output shapes

Tensor values are generated from `seed`, while token-shard metadata is generated
from `metadata_seed`. This allows tests to compare different numerical draws
with the same rank/token layout. Expert-parallel routing also uses
`metadata_seed` for token distribution and top-k expert ids; hidden states,
top-k weights, and synthetic expert outputs come from `seed`.

## TokenSpeed API Mapping

TokenSpeed provides backend-specific communication implementations for
all-reduce, all-gather, reduce-scatter, fused all-reduce residual RMSNorm,
fused reduce-scatter residual RMSNorm, fused all-gather dual RMSNorm, MiniMax
Q/K all-reduce RMSNorm, and DeepEP-based expert-parallel dispatch/combine. The
generated values map to those APIs through a small rank-local adapter: each distributed worker
generates the same operation-level values, selects `rank_inputs[rank]` and any
other rank-local tensors such as `residuals[rank]`, invokes the backend kernel,
and compares against the reference result for that rank.

For fused all-reduce residual RMSNorm, the adapter passes
`rank_inputs[rank]`, `residuals[rank]`, the shared `weight`, and `eps` to the
backend. The semantic comparison checks both returned tensors:
`residual_out[rank]` after the reduced input is added to the local residual,
and `norm_out[rank]` after RMSNorm is applied.

For fused reduce-scatter residual RMSNorm, the adapter passes each rank's full
contribution tensor plus the local residual and optional add tensor. The
reference performs the sum, slices the near-even rank shard, and compares that
rank's residual and normalized outputs.

For fused all-gather dual RMSNorm, the adapter passes each rank's local QKV
shard and the shared Q/KV RMSNorm weights. The reference compares the gathered
output with normalized KV slice, the normalized Q output, and the normalized KV
view.

For MiniMax Q/K all-reduce RMSNorm, the adapter passes
`q_rank_inputs[rank]`, `k_rank_inputs[rank]`, the shared Q/K RMSNorm weights,
and `eps` to the fused MiniMax backend. The reference compares that rank's
normalized Q and K outputs.

For DeepEP-style consumers, `rank_hidden_states[rank]`, `topk_ids[rank]`, and
`topk_weights[rank]` are the rank-local dispatch inputs. The reference
`dispatch_records`, receive tensors, and per-expert counts provide a semantic
target for adapter tests without baking a specific DeepEP buffer layout into
the generator.

The generator does not create process groups, symmetric-memory workspaces, IPC
handles, or backend state objects. Those details are implementation-specific and
belong in TokenSpeed tests or adapters.
