# Attention Input Generators

Attention generators cover operation-level inputs for multi-head attention
families and related attention utility primitives. They describe query/K/V
tensors, request metadata, optional cache contents, and merge states without
depending on a specific backend API.

## Operation Semantics

### MHA

`MHAInputs` represents multi-head attention with query heads, key/value heads,
per-head dimensions, request metadata, optional new K/V tensors, optional
attention sinks, and optional dense or paged KV cache contents.

The generated metadata controls the number of cached tokens and new query
tokens. For ragged request layouts, per-request lengths are generated from the
configured totals and batch size so callers do not need to hand-author
consistent cumulative sequence metadata.

### MLA

`MLAInputs` represents multi-head latent attention. It shares the same request
metadata and cache-layout concepts as MHA, but uses MLA-specific operand shapes:
query-nope, query positional embedding channels, compressed KV cache contents,
and optional new compressed KV inputs.

### Merge State

`AttentionMergeStateInputs` represents merging two partial attention outputs and
their log-sum-exp states:

```text
lse_a_log2 = lse_a * lse_scale_log2
lse_b_log2 = lse_b * lse_scale_log2
lse_max = max(lse_a_log2, lse_b_log2)
w_a = 2 ** (lse_a_log2 - lse_max)
w_b = 2 ** (lse_b_log2 - lse_max)
out = (out_a * w_a + out_b * w_b) / (w_a + w_b)
lse = (lse_max + log2(w_a + w_b)) / lse_scale_log2
```

This operation is useful when an attention computation is split into partial
contexts. It is independent of how those partial states were produced.

## Validation Contract

The attention generators reject invalid operation inputs before returning
values:

- token totals must be non-negative
- batch sizes, head counts, head dimensions, and page sizes must be positive
- generated request-length metadata is internally consistent
- cache configuration must match the requested cache layout
- paged caches require consistent page-table configuration
- MHA query heads must be compatible with KV heads for grouped attention
- merge-state outputs must have shape `[total_q, num_heads, head_dim]`
- merge-state LSE tensors must have shape `[total_q, num_heads]` and use fp32
  generated values
- merge-state `lse_scale_log2` and generated LSE bounds must be positive

Metadata-oriented values such as request lengths and page mappings are generated
from metadata seeds. Numerical tensors are generated from value seeds so tests
can reuse the same request/cache layout across different value draws.

## TokenSpeed API Mapping

TokenSpeed has several attention registry entry points: MHA prefill, MHA
extend/decode with KV cache, MLA prefill, MLA decode with KV cache, and
attention merge-state. The generator values are operation-level values. Tests
or adapters are responsible for converting generated values into the exact
keyword arguments expected by a selected TokenSpeed backend.
