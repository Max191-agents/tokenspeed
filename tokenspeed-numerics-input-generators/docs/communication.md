# Communication Input Generators

Communication generators cover collective operations over a group of ranks. The
operation semantics are independent of any particular transport or backend:
each rank contributes one tensor, and the collective defines how rank-local
inputs are combined and redistributed.

This slice covers sum all-reduce, all-gather, sum reduce-scatter, and fused
sum all-reduce plus residual RMSNorm. It does not yet cover expert all-to-all
dispatch.

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

Tensor values are generated from `seed`, while token-shard metadata is generated
from `metadata_seed`. This allows tests to compare different numerical draws
with the same rank/token layout.

## TokenSpeed API Mapping

TokenSpeed provides backend-specific communication implementations for
all-reduce, all-gather, reduce-scatter, and fused all-reduce residual RMSNorm.
The generated values map to those APIs through a small rank-local adapter: each
distributed worker generates the same operation-level values, selects
`rank_inputs[rank]` and any other rank-local tensors such as
`residuals[rank]`, invokes the backend kernel, and compares against the
reference result for that rank.

The generator does not create process groups, symmetric-memory workspaces, IPC
handles, or backend state objects. Those details are implementation-specific and
belong in TokenSpeed tests or adapters.
