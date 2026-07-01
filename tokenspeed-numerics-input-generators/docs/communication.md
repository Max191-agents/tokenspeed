# Communication Input Generators

Communication generators cover collective and routing operations over a group
of ranks. The operation semantics are independent of any particular transport
or backend: each rank contributes rank-local tensors or metadata, and the
communication operation defines how those inputs are combined and redistributed.

This slice covers sum all-reduce, all-gather, sum reduce-scatter, fused sum
all-reduce plus residual RMSNorm, and expert-parallel MoE dispatch/combine
routing.

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
all-reduce, all-gather, reduce-scatter, fused all-reduce residual RMSNorm, and
DeepEP-based expert-parallel dispatch/combine. The generated values map to
those APIs through a small rank-local adapter: each distributed worker
generates the same operation-level values, selects `rank_inputs[rank]` and any
other rank-local tensors such as `residuals[rank]`, invokes the backend kernel,
and compares against the reference result for that rank.

For fused all-reduce residual RMSNorm, the adapter passes
`rank_inputs[rank]`, `residuals[rank]`, the shared `weight`, and `eps` to the
backend. The semantic comparison checks both returned tensors:
`residual_out[rank]` after the reduced input is added to the local residual,
and `norm_out[rank]` after RMSNorm is applied.

For DeepEP-style consumers, `rank_hidden_states[rank]`, `topk_ids[rank]`, and
`topk_weights[rank]` are the rank-local dispatch inputs. The reference
`dispatch_records`, receive tensors, and per-expert counts provide a semantic
target for adapter tests without baking a specific DeepEP buffer layout into
the generator.

The generator does not create process groups, symmetric-memory workspaces, IPC
handles, or backend state objects. Those details are implementation-specific and
belong in TokenSpeed tests or adapters.
