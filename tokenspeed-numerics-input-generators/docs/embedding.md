# Embedding Input Generators

The current TokenSpeed embedding family is rotary positional embedding (RoPE)
over query and key heads. It is not a token-lookup embedding table generator.
The generators therefore model Q/K tensors, position metadata, RoPE caches, and
the MLA-style fused RoPE plus FP8 quantization path used before attention.

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

### MLA RoPE FP8 Quantization

`MLARopeQuantizeFP8Inputs` represents a fused operation over decomposed query
and key tensors:

```text
q_rope_rot = rope(q_rope, positions, cos_sin_cache)
k_rope_rot = rope(k_rope, positions, cos_sin_cache)
q_nope_fp8 = fp8(q_nope * quant_scale_q)
q_rope_fp8 = fp8(q_rope_rot * quant_scale_q)
k_nope_fp8 = fp8(k_nope * quant_scale_kv)
k_rope_fp8 = fp8(k_rope_rot * quant_scale_kv)
query = concat(q_nope_fp8, q_rope_fp8, dim=-1)
key = concat(k_nope_fp8, k_rope_fp8, dim=-1)
```

The query inputs are rank-3 tensors:

- `q_nope[num_tokens, num_q_heads, qk_nope_head_dim]`
- `q_rope[num_tokens, num_q_heads, qk_rope_head_dim]`

The key inputs support two operation shapes. Rank-2 key tensors model MLA's
shared key head:

- `k_nope[num_tokens, qk_nope_head_dim]`
- `k_rope[num_tokens, qk_rope_head_dim]`

Rank-3 key tensors model explicit GQA/MHA KV heads:

- `k_nope[num_tokens, num_kv_heads, qk_nope_head_dim]`
- `k_rope[num_tokens, num_kv_heads, qk_rope_head_dim]`

The generated output buffers have the same shapes as their corresponding input
slices and FP8 dtype. Executing the represented operation produces the rotated,
quantized slices; concatenating each NOPE/RoPE pair produces the full query or
key tensor.

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
- MLA RoPE FP8 input dtype must be fp16 or bf16
- MLA RoPE FP8 output dtype must be FP8 E4M3 or FP8 E5M2
- MLA RoPE FP8 PE dimensions must be even and match between query and key
- rank-2 MLA key tensors use an implicit shared KV head
- RoPE FP8 quantization scales must be positive

Positions and cache locations are generated from `metadata_seed`, while query,
key, value, and MLA RoPE FP8 tensors are generated from `seed`. This lets
callers keep the same positional/cache metadata across different random tensor
draws.

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

TokenSpeed's FlashInfer `mla_rope_quantize_fp8(...)` wrapper consumes the MLA
RoPE FP8 values directly: the generated input slices, `cos_sin_cache`,
`positions`, FP8 output buffers, layout flag, and quantization scales. The
generator does not depend on FlashInfer; tests can adapt the values into that
wrapper when the backend is available.

Reference implementations are consumer concerns and are intentionally not
part of the input-generator package. TokenSpeed keeps its RoPE references in
its numerical validation layer.
