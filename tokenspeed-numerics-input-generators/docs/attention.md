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

### MLA K/V Pack And FP8 Quantize

`MLAKVPackQuantizeFP8Inputs` represents the utility operation that materializes
MLA K/V tensors for FP8 cache storage. The operation takes non-RoPE key channels
`k_nope`, RoPE key channels `k_pe`, and value channels `v`:

```text
k_pe_heads = broadcast(k_pe, across=kv_heads)
k = concat(k_nope, k_pe_heads, dim=-1)
k_fp8 = cast_fp8(k * k_scale_inv)
v_fp8 = cast_fp8(v * v_scale_inv)
```

`k_nope` and `v` have shape `[num_tokens, num_kv_heads, dim]`. `k_pe` may be
`[num_tokens, qk_rope_head_dim]` or `[num_tokens, 1, qk_rope_head_dim]`; both
forms represent the same per-token RoPE key component shared by all KV heads.

### GDN Packed QKV Split

`GDNQKVSplitInputs` represents splitting a packed post-projection QKV tensor
into query, key, and value tensors:

```text
mixed_qkv[t] = concat(q[t], k[t], v[t])
q -> [1, tokens, num_q_heads, head_q]
k -> [1, tokens, num_k_heads, head_k]
v -> [1, tokens, num_v_heads, head_v]
```

Some GDN prefill paths also fuse per-head L2 normalization of Q and K into this
split:

```text
q[t, h] = q[t, h] / sqrt(sum(q[t, h] ** 2) + eps)
k[t, h] = k[t, h] / sqrt(sum(k[t, h] ** 2) + eps)
```

V is always copied without normalization. The generator owns the packed-width
relationships between head counts and per-head dimensions, and its reference
implements both the plain split and fused-normalization variants.

### GDN Chunked Prefill

`GDNChunkPrefillInputs` represents the prompt-side recurrent scan for
Gated DeltaNet-style linear attention. Unlike softmax attention, the operation
does not materialize pairwise attention over all prior tokens. It maintains a
per-sequence, per-value-head matrix state with shape `[head_dim, head_dim]`.
Each token decays the previous state, applies a delta-rule correction, writes
the corrected value back through the key vector, and reads the updated state
with the query vector:

```text
state = exp(g_t) * state
delta = beta_t * (v_t - k_t @ state)
state = state + outer(k_t, delta)
out_t = scale * (q_t @ state)
```

The generated Q and K rows are L2-normalized because the chunked prefill fast
paths expect normalized Q/K inputs. `g` is generated in log space, so `exp(g)`
is the multiplicative state decay. `beta` is generated as a bounded update gate
in `[0.05, 0.95]`. Sequence metadata is represented by `cu_seqlens`, which
partitions the flattened prompt stream into independent recurrent scans.

The generator supports equal Q/V head counts and grouped-value attention where
`num_v_heads` is an integer multiple of `num_q_heads`. In the grouped-value
case, multiple value/state heads share one Q/K head. The optional
`include_batch_dim` setting only controls whether tensors include the leading
singleton batch axis accepted by TokenSpeed's wrapper; the operation is still
defined by the flattened token stream and `cu_seqlens`.

When `output_h` is requested, the reference returns recurrent-state
checkpoints after each full 64-token chunk in every sequence, along with
`checkpoint_cu_starts` metadata describing how many checkpoints belong to each
sequence.

### Packed QKV Complex Rotary

`PackedQKVComplexRotaryInputs` represents splitting equal-width packed Q/K/V
segments while applying complex RoPE to Q and K:

```text
qkv[t] = concat(q[t], k[t], v[t])
z_q = q_even + i * q_odd
z_k = k_even + i * k_odd
q_rot = z_q * freqs_cis[t]
k_rot = z_k * freqs_cis[t]
v_out = v
```

`freqs_cis` is generated as unit complex values with shape
`[tokens, head_dim / 2]`. The generator requires an even `head_dim` because
adjacent real channels form one complex pair. The `copy_v` option controls
whether consumers should materialize V or can treat the V output as a view of
the packed V segment; the represented V values are unchanged either way.

### DSA Sparse Decode KV Pack

`DSASparseDecodeKVPackInputs` represents packing per-token sparse-decode K
rows into physical cache slots. Each generated token row contains BF16 NoPE key
channels and BF16 RoPE key channels. The packed physical row layout is:

```text
[NoPE FP8 E4M3 bytes][FP32 scale bytes per 128 NoPE channels][RoPE BF16 bytes]
```

The NoPE channels are split into 128-channel blocks. Each block gets one FP32
scale equal to `amax(abs(block)) / 448`, clamped away from zero, and stores the
scaled values as FP8 E4M3 bytes. RoPE channels are copied as raw BF16 bytes.
Generated slot locations are unique so the represented operation has no
write-after-write race between token rows.

### DSA Sparse Top-K Slots

`DSATopKSlotInputs` represents sparse DSA metadata that maps token-local
context offsets through a token-row page table into physical KV-cache slots:

```text
block = local_offset // block_size
offset = local_offset % block_size
slot = block_table[token, block] * block_size + offset
```

The same generated values support two related operations. Local top-k mode uses
generated `local_topk_offsets[token, topk]` and counts valid offsets for each
token. Full-context mode ignores `local_topk_offsets` and maps the first `topk`
logical positions for each token context. In both cases, `seq_lens` bounds the
valid logical context range and the block table bounds the physical page range.

### DeepSeek V4 Compressor State Save

`DeepSeekV4CompressorStateInputs` represents the write that saves intermediate
compressor state into a paged DeepSeek V4 state cache. Each valid token row has
a physical `slot_mapping` entry. The slot selects a cache page and row:

```text
page = slot // block_size
row = slot % block_size
state_cache[page, row, 0:state_width] = kv[token]
state_cache[page, row, state_width:] = score[token] + ape[position % compress_ratio]
```

Rows with `slot_mapping == -1` are skipped and leave the generated cache
unchanged. The generator ensures non-negative slots are unique and in range, so
generated writes do not race with each other and always address valid cache
rows.

For the C4 overlap path (`compress_ratio == 4` with an even state width), the
kernel treats the APE tensor as a flat overlapping layout: the first half of the
APE vector comes from row `position % 4`, and the second half comes from the
corresponding row in the second half of the flattened APE buffer. The reference
models this layout explicitly so tests can validate both standard and overlap
state-cache writes.

### DeepSeek V4 Indexer MXFP4 Cache Write

`DeepSeekV4IndexerMXFP4CacheWriteInputs` represents writing 128-channel indexer
K rows into a paged MXFP4 byte cache. Each cache page stores all packed value
bytes for its rows first, followed by one scale byte per 32-channel MXFP4 block:

```text
value_base = page * page_stride + row * 64
scale_base = page * page_stride + block_size * 64 + row * 4
```

For each writable input row, the operation splits the 128 channels into four
32-channel blocks. Every pair of channels is packed into one E2M1 MXFP4 byte,
and every block gets one encoded exponent scale byte. A row is writable only
when `valid[row]` is true and `slot_mapping[row] >= 0`; otherwise the cache
contents for that row remain unchanged. The generator gives writable rows
unique in-range slots so generated inputs cannot describe write races or
out-of-bounds cache accesses.

### DeepSeek V4 Indexer MXFP4 Cache Gather

`DeepSeekV4IndexerMXFP4CacheGatherInputs` represents reading 128-channel indexer
K rows from the same paged MXFP4 byte-cache layout used by the write generator.
For each requested row, `slot_mapping` selects the physical page and row within
that page. The operation copies 64 packed value bytes and 4 scale bytes into
dense output workspaces:

```text
value_base = page * page_stride + row * 64
scale_base = page * page_stride + block_size * 64 + row * 4
```

Rows with `slot_mapping < 0` gather zeros. Non-negative slots are generated
within the physical cache range, and generated cache/output byte workspaces use
layouts compatible with this packed cache representation.

### DeepSeek V4 K-Cache Gather And Dequantize

`DeepSeekV4KCacheGatherInputs` represents reading sparse-window K rows from a
paged DeepSeek V4 cache into a dense BF16 workspace. Each physical cache page
stores all token payload bytes first, followed by one scale-byte row per token:

```text
token_base = page * page_stride + row * 576
scale_base = page * page_stride + block_size * 576 + row * 8
```

The first 448 token payload bytes are FP8 E4M3 NoPE values, with one UE8M0
scale byte per contiguous 64-channel group. The final 128 payload bytes are the
64 BF16 RoPE channels. For each request, `seq_lens` and optional `gather_lens`
select a suffix of visible K tokens. `block_table` maps each logical cache page
to a physical page, and generated metadata keeps all gathered positions within
the table.

### DeepSeek V4 Sparse-Prefill Indices

`DeepSeekV4SparsePrefillIndexInputs` represents the metadata/index construction
used by DeepSeek V4 sparse prefill attention. The operation does not generate
attention values. It builds integer candidate lists into a per-request KV
workspace:

```text
workspace[request] =
  compressed_prefix_slots[0:compressed_base] +
  gathered_sliding_window_slots[compressed_base:workspace_width]
```

For each generated query token, `positions` gives the absolute token position in
the request and `token_to_req_indices` maps the packed token row back to the
request. The generated `topk_indices` select sparse compressed-prefix slots for
the top-k path. Dense-compressed references instead enumerate compressed prefix
slots from zero up to the per-token compressed length. Both paths append local
sliding-window attention slots derived from `window_size`, `gather_lens`, and
the request sequence lengths.

The generator keeps request lengths, compressed-prefix capacity, gathered SWA
capacity, and workspace width consistent so the generated indices are valid
operation-level inputs for the local-compressed, top-k+SWA, and
dense-compressed+SWA index builders.

### DeepSeek V4 Paged Indices

`DeepSeekV4PagedIndexInputs` represents paged-cache metadata used by DeepSeek V4
decode and indexer helpers. The generated object describes packed query tokens,
their absolute request positions, a request id for each packed token row, and a
per-request block table:

```text
logical_token_position -> logical_block = position // block_size
logical_block -> physical_block = block_table[request, logical_block]
slot_id = physical_block * block_size + position % block_size
```

The same generated values support several related metadata operations:

- mapping request-local top-k indices to global paged-cache slot ids
- building per-token decode sliding-window slot lists and row lengths
- building compressed KV slot mappings for newly materialized compressed tokens
- building decode-indexer block tables and compressed context lengths

The generator owns the shape relationships between request metadata, token
positions, page-table width, optional valid-token masks, optional block-table
base offsets, and the compressed/paged indexing parameters. These are metadata
operations, so no attention value tensors are generated.

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
- GDN QKV split inputs require positive head counts/dimensions, a packed last
  dimension equal to `q_dim + k_dim + v_dim`, and a positive L2-normalization
  epsilon when the fused normalization path is used
- packed QKV complex-rotary inputs require equal Q/K/V packed widths, an even
  head dimension, and complex frequencies with one value per token and channel
  pair
- DSA sparse top-k slot inputs require int32 local offsets, sequence lengths,
  and block tables with one row per token and enough page columns to cover the
  generated local context lengths
- DSA sparse decode KV pack inputs require BF16 source tensors, a 1-D integer
  slot-location tensor with unique valid rows, a uint8 output buffer whose row
  width matches the packed layout, a NoPE dimension that is a power of two and
  divisible by 128, and a RoPE dimension that is a power of two
- MLA K/V pack inputs require matching token/head dimensions for `k_nope` and
  `v`, positive inverse scales, a broadcastable RoPE key tensor, and an FP8
  output dtype
- DeepSeek V4 sparse-prefill index inputs require tied Q/KV request lengths, a
  compressed-prefix ratio greater than one, valid top-k compressed-prefix slots,
  and a workspace width that covers compressed-prefix plus gathered SWA slots
- DeepSeek V4 paged-index inputs require tied Q/KV request lengths, a block
  table wide enough for generated visible KV positions, positive block/window
  sizes, a compressed-prefix ratio greater than one, and valid token-to-request
  metadata
- DeepSeek V4 K-cache gather inputs require a paged byte-cache layout with
  FP8 NoPE bytes, BF16 RoPE bytes, UE8M0 scale bytes, valid gather lengths, and
  block-table entries covering every gathered token position
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
attention merge-state. TokenSpeed also exposes GDN QKV split, packed QKV rotary,
DSA sparse decode KV packing, DSA sparse slot conversion, and MLA K/V
pack+quantize helpers that map directly to `GDNQKVSplitInputValues`,
`PackedQKVComplexRotaryInputValues`, `DSASparseDecodeKVPackInputValues`,
`DSATopKSlotInputValues`, and `MLAKVPackQuantizeFP8InputValues`, plus DeepSeek
V4 sparse-prefill index helpers that accept the tensors and scalar workspace
parameters generated by `DeepSeekV4SparsePrefillIndexInputs`. DeepSeek V4
paged-cache index helpers similarly consume `DeepSeekV4PagedIndexInputs`
through small adapters. The DeepSeek V4 K-cache gather/dequantize helper
consumes `DeepSeekV4KCacheGatherInputs` directly. The generator values are
operation-level values. Tests or adapters are responsible for converting
generated values into the exact keyword arguments expected by a selected
TokenSpeed backend.
