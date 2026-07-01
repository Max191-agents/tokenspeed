# Communication Generators

Communication generators model collective-style tensor movement used around
distributed inference kernels. The operation semantics are expressed as tensor
partitioning, summation, gathering, scattering, and expert-parallel routing
rather than any specific communication library API.

## Generators

- `AllReduceInputs`: generates per-rank tensors for elementwise sum reduction.
- `AllReduceResidualRMSNormInputs`: generates residual/RMSNorm values around an
  all-reduce style sum.
- `AllGatherInputs`: generates rank-local shards for concatenating along a
  configured dimension.
- `ReduceScatterInputs`: generates full tensors and expected shard metadata for
  reduce-scatter sum.
- `ExpertParallelRoutingInputs`: generates routing assignments and token tensors
  for expert-parallel dispatch/combination tests.
- `DPSamplingInputs`: generates the batch-DP sampling communication layout used
  by speculative verification, including vocab-sharded logits, rank-local
  verify outputs, and references for the logits swap and verify-output gather.

## Batch-DP Sampling

Batch-DP sampling is a communication pattern for speculative verification with
tensor-parallel vocabulary shards. Each rank starts with logits for the full
padded batch, but only for its local vocabulary shard:

```text
local_logits[rank]: [pad_batch_size * N, vocab_size / world_size]
```

The logits swap redistributes those shards so each rank owns a request shard
with full vocabulary logits:

```text
swapped_logits[rank]: [pad_batch_size / world_size * N, vocab_size]
```

After verification, each rank has local request-shard outputs:

```text
predict_local[rank]:       [pad_batch_size / world_size, N]
accept_index_local[rank]:  [pad_batch_size / world_size, N]
accept_length_local[rank]: [pad_batch_size / world_size]
```

The gather reference concatenates those local verify outputs in source-rank
order to form full padded-batch tensors. The generator verifies that
`pad_batch_size` and `vocab_size` are divisible by `world_size`, generated token
ids stay within the padded vocabulary, accept indices are either `-1` or valid
rank-local flattened positions, and accept lengths match the generated
non-negative accept-index entries.

## Generated Values

Values include both local tensors and metadata describing rank count, rank id,
partition dimension, and routing layout. References compute operation-level
results in torch so consumer tests can compare kernel or adapter behavior
without duplicating generic collective validity checks.

The generator verifies shapes, divisibility, rank bounds, and routing metadata
so generated communication inputs are meaningful before a backend adapter sees
them.
