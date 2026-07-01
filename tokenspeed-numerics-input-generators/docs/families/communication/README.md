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

## Generated Values

Values include both local tensors and metadata describing rank count, rank id,
partition dimension, and routing layout. References compute operation-level
results in torch so consumer tests can compare kernel or adapter behavior
without duplicating generic collective validity checks.

The generator verifies shapes, divisibility, rank bounds, and routing metadata
so generated communication inputs are meaningful before a backend adapter sees
them.
