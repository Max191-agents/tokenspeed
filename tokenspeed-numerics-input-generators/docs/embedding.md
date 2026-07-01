# Embedding Input Generators

The current TokenSpeed embedding family is rotary positional embedding (RoPE)
over query and key heads. It is not a token-lookup embedding table generator.
The generator therefore models the Q/K tensors, position metadata, and RoPE
cache needed to apply position-dependent rotations.

## Operation Semantics

`RopeInputs` represents rotary embedding applied independently to every query
and key head. Query and key are flattened as `[tokens, heads * head_size]`, but
the operation views them as `[tokens, heads, head_size]`.

For each token position and head, the first `rotary_dim` channels are rotated by
cosine and sine values from `cos_sin_cache[position]`. Channels after
`rotary_dim` pass through unchanged.

Two rotation layouts are supported:

- `is_neox=True`: split the rotated channels into two halves `[x1 | x2]`
- `is_neox=False`: rotate interleaved GPT-J pairs `[x0, x1]`

The generator can also produce optional output buffers. When output buffers are
present, the operation writes rotated values there instead of mutating the
corresponding input.

## Fused KV Writes

Some RoPE implementations fuse key rotation with a KV-cache write. The
operation-level inputs for this are:

- `value`: the V tensor to copy into the V cache
- `k_buffer`: the destination K cache
- `v_buffer`: the destination V cache
- `cache_loc`: one destination row per token

The generator returns these as `RopeFusedKVInputValues`. Kernel-specific wrapper
objects, such as TokenSpeed's `FusedSetKVBufferArg`, are adapter concerns.

## Validation Contract

The generator rejects invalid RoPE inputs before values are returned:

- token counts must be non-negative
- head counts, `head_size`, `rotary_dim`, and `max_position` must be positive
- `rotary_dim` must be even and no larger than `head_size`
- query/key flattened widths are derived from head counts and `head_size`
- input dtype must be a regular floating torch dtype
- position and cache-location dtypes must be int32 or int64
- fused KV cache size must be large enough for unique generated cache locations

Positions and cache locations are generated from `metadata_seed`, while query,
key, and value tensors are generated from `seed`. This lets callers keep the
same positional/cache metadata across different random tensor draws.

## TokenSpeed API Mapping

TokenSpeed's `embedding.rope` API takes `positions`, `query`, `key`,
`head_size`, `cos_sin_cache`, layout flags, optional output buffers, and an
optional fused KV wrapper. `RopeInputValues` maps directly to these arguments.
Tests can construct the TokenSpeed wrapper from `values.fused_kv` without
placing that wrapper in the standalone generator package.

The same generated values are valid for the Triton and CUDA RoPE backends when
the backend supports the requested traits, such as head size, partial-rotary
mode, layout, fused KV writes, and output buffers. Backend selection is an
adapter concern; the generator only defines the operation-level Q/K tensors,
positions, RoPE cache, and optional cache-write values.
