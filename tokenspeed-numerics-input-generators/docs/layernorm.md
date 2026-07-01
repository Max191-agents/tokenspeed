# Layernorm Input Generators

Layernorm-family generators cover normalization operations that reduce across a
feature dimension and rescale by learned weights. The current family focuses on
RMSNorm variants because those are the normalization operations used by the
TokenSpeed layernorm kernels today.

## Operation Semantics

### RMSNorm

`RMSNormInputs` represents row-wise root-mean-square normalization:

```text
variance = mean(x * x, dim=-1, keepdim=True)
y = x * rsqrt(variance + eps) * weight
```

When `with_residual` is enabled, the normalized value is `x + residual`, and
that summed residual value is also part of the operation output.

### Gemma RMSNorm

Gemma RMSNorm uses the same generated input structure as `RMSNormInputs`, but
the learned weight is interpreted as an offset from one:

```text
variance = mean(x * x, dim=-1, keepdim=True)
y = x * rsqrt(variance + eps) * (1 + weight)
```

The residual form follows the same residual-add rule as ordinary RMSNorm. The
input generator does not need a separate config because the operation has the
same tensor inputs and shape constraints; the distinction is captured by
`gemma_rmsnorm_reference`.

### QK RMSNorm

`QKRMSNormInputs` represents two independent RMSNorm operations over query and
key heads. The tensors are flattened as `[tokens, heads * head_dim]`, but the
normalization groups are the individual `[head_dim]` heads. The generator can
produce ordinary dense `q`/`k` tensors or strided `q`/`k` views from packed
`[q, k, v]` storage.

### Fused QK RMSNorm + RoPE + Gate

`FusedQKRMSNormRopeGateInputs` represents the fused operation used by QK paths
where the query operand arrives packed per head as `[q | gate]`. The operation:

- splits `q_gate` into query and gate tensors
- applies per-head RMSNorm to query and key
- applies RoPE to the first `rotary_dim` values of each normalized head
- copies the gate tensor unchanged

The RoPE cache uses per-position `[cos | sin]` layout so consumers can pass it
directly to kernels that expect that semantic layout. The generator accepts an
optional `metadata_seed` so position metadata can be held fixed across different
random tensor draws.

### Parallel RMSNorm

`ParallelRMSNormInputs` represents two independent RMSNorms with a shared row
count. It is useful for fused-parallel kernels where the implementation computes
both normalizations in one launch, but the operation semantics remain two
ordinary RMSNorms.

## Validation Contract

The generators reject invalid normalization inputs before values are returned:

- token counts must be non-negative
- hidden dimensions, head counts, head dimensions, and cache sizes must be
  positive
- input and weight dtypes must be regular floating torch dtypes
- `eps` must be positive
- Q/K flattened widths are derived from head counts and `head_dim`
- `rotary_dim` must be positive, even, and no larger than `head_dim`
- generated RoPE positions are in range for the generated cache

These are operation-level constraints. Whether a specific kernel supports a
particular dtype, contiguity, or launch shape remains an adapter or test
concern.

## TokenSpeed API Mapping

TokenSpeed currently exposes Triton layernorm functions for ordinary RMSNorm,
residual RMSNorm, QK RMSNorm, fused QK RMSNorm + RoPE + gate, and fused-parallel
RMSNorm. It also exposes FlashInfer wrappers for ordinary RMSNorm, Gemma
RMSNorm, and their in-place fused add variants. The generator values map
directly to those APIs with small adapters:

- `RMSNormInputValues` provides `x`, `weight`, and optional `residual`
- `QKRMSNormInputValues` provides `q`, `k`, `q_weight`, and `k_weight`
- `FusedQKRMSNormRopeGateInputValues` provides the packed `q_gate`, `k`,
  weights, RoPE cache, and positions
- `ParallelRMSNormInputValues` provides both input/weight/output triples

The reference helpers in the package model the mathematical operation and can
be used by kernel tests without tying the generators to TokenSpeed registry
details.
