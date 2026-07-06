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

`mla_reference` evaluates generated MLA values at the operation level. Uncached
inputs use explicit varlen Q/K/V attention, including grouped KV heads when the
generated head counts require expansion. Paged-cache inputs gather compressed
cache rows through the generated page table and compute absorbed MLA decode
outputs over the latent KV channels.

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
`k_nope` and `v` can be generated as separate contiguous tensors or as
non-contiguous views into one packed `[tokens, kv_heads, qk_nope + v_dim]`
projection tensor. Both layouts represent the same logical inputs; the packed
layout exists so consumers can exercise production-style slice views while the
reference continues to operate on the logical tensors.

### MLA FP8 Prefill

`MLAPrefillFP8Inputs` represents varlen MLA prefill attention where
materialized query, key, and value tensors are already stored in FP8:

```text
scores = query @ key.T * softmax_scale
probs = softmax(scores)
out = probs @ value
```

The generator ties Q and K/V request lengths per sequence, which models true
prefill and keeps causal masking valid by construction. Sequence metadata is
generated with the same cumulative-offset machinery used by MHA/MLA request
generators, while numerical values are generated in a regular floating source
dtype and cast to the requested FP8 storage dtype.

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

### DSA Decode Top-K

`DSADecodeTopKInputs` represents deterministic sparse DSA decode top-k
selection over pre-masked indexer logits:

```text
valid_logits = logits[row, 0:valid_len[row]]
logits[row, valid_len[row]:] = -inf
indices = topk(valid_logits, k=topk, tie_break=smallest_index)
```

The selected indices are token-local context offsets, not physical cache slots.
The generator returns `valid_lens` to document and validate the masking
relationship, while the operation input consumed by a top-k implementation is
the already-masked `logits` tensor plus the output index buffer and `topk`.
By default, generated rows include an equal-logit boundary case so references
and kernels must use a deterministic smallest-index tie-break.

### Compressed Sequence Attention

`CompressedSequenceAttentionInputs` is the canonical generator for compressed
sequence attention input generation. It represents sliding-window attention
plus optional compressed-history attention and optional CSA indexer inputs. The
current shape/layout support is DeepSeek V4-style.

The sliding-window portion is always generated. It contains Q, attention sink,
request metadata, absolute token positions, token-to-request indices, visible
sequence lengths, a page table, and a paged byte cache.

The optional `compressed` config adds compressed-history inputs. A
`compress_ratio` of `128` represents HCA-style compressed history, while a
`compress_ratio` of `4` represents CSA-style compressed history. The generated
compressed values include paged/sparse index metadata, compressor-state cache
values, sparse K-cache insert values, and sparse K-cache gather values.

The optional `indexer` config is valid only for CSA. It adds the generated
indexer-Q RoPE/Hadamard/MXFP4 transform values, CSA indexer cache-insert
values, and indexer cache write/gather values.

The nested values intentionally expose the narrower helper-operation bundles
that TokenSpeed tests and references already consume, but callers should start
from the single compressed sequence attention generator instead of constructing
one generator per helper kernel.

DeepSeek V4 inverse-RoPE FP8 quantization remains a separate generator because
it is an output-projection preparation utility, not part of compressed
attention input generation.

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
- MLA prefill references require matching Q/K dimensions and compatible
  grouped K/V heads; MLA paged-decode references require valid page-table
  metadata and compressed cache rows with one cached KV row per logical token
- GDN QKV split inputs require positive head counts/dimensions, a packed last
  dimension equal to `q_dim + k_dim + v_dim`, and a positive L2-normalization
  epsilon when the fused normalization path is used
- packed QKV complex-rotary inputs require equal Q/K/V packed widths, an even
  head dimension, and complex frequencies with one value per token and channel
  pair
- DSA sparse top-k slot inputs require int32 local offsets, sequence lengths,
  and block tables with one row per token and enough page columns to cover the
  generated local context lengths
- DSA decode top-k inputs require floating pre-masked logits, int32 output
  buffers, `topk <= vocab_size`, per-row valid lengths in `[topk, vocab_size]`,
  finite logits before each valid length, and `-inf` logits after each valid
  length
- DSA sparse decode KV pack inputs require BF16 source tensors, a 1-D integer
  slot-location tensor with unique valid rows, a uint8 output buffer whose row
  width matches the packed layout, a NoPE dimension that is a power of two and
  divisible by 128, and a RoPE dimension that is a power of two
- MLA K/V pack inputs require matching token/head dimensions for `k_nope` and
  `v`, positive inverse scales, a broadcastable RoPE key tensor, and an FP8
  output dtype
- MLA FP8 prefill inputs require tied non-empty Q and K/V request lengths, FP8
  Q/K/V storage, matching head counts, matching Q/K head dimensions, and a
  positive softmax scale
- compressed sequence attention inputs require tied Q/KV request lengths, the
  currently supported DeepSeek V4-style 512-wide attention-head layout with a
  64-channel RoPE suffix, positive paged-cache dimensions, and page tables wide
  enough for the generated visible KV positions
- compressed-history inputs require `compress_ratio` equal to 4 for CSA or 128
  for HCA; CSA requires overlapping compressor state, and HCA requires
  non-overlapping compressor state
- CSA indexer inputs require compressed-history inputs with
  `compress_ratio == 4`, valid MXFP4 indexer-cache layouts, and generated
  slot mappings that avoid out-of-bounds cache accesses
- DeepSeek V4 inverse-RoPE FP8 quantization inputs require grouped head counts
  matching the attention-output head dimension, an even rotary suffix, a head
  dimension divisible by the quantization group size, and a rotary suffix that
  fits in the final quantization group
- merge-state outputs must have shape `[total_q, num_heads, head_dim]`
- merge-state LSE tensors must have shape `[total_q, num_heads]` and use fp32
  generated values
- merge-state `lse_scale_log2` and generated LSE bounds must be positive

Metadata-oriented values such as request lengths and page mappings are generated
from metadata seeds. Numerical tensors are generated from value seeds so tests
can reuse the same request/cache layout across different value draws.

`MHARequestMetadataInput`, `PageTableInput`, and `SlotMappingInput` are the
shared metadata primitives for attention-style generators. Use them for request
lengths/cumulative offsets, page or block tables, and row-to-cache-slot
mappings before adding operation-specific metadata generation. Many
TokenSpeed-facing names are aliases for these concepts: `block_table` is a
page table when it maps request-local pages to physical cache pages, and
`slot_mapping`, `kv_slot_mapping`, and `compressor_slot_mapping` are flat slot
mappings with different consumers.

## TokenSpeed API Mapping

TokenSpeed has several attention registry entry points: MHA prefill, MHA
extend/decode with KV cache, MLA prefill, MLA decode with KV cache, and
attention merge-state. TokenSpeed also exposes GDN QKV split, packed QKV rotary,
DSA sparse decode KV packing, DSA sparse slot conversion, deterministic DSA
decode top-k selection, and MLA K/V pack+quantize helpers that map directly to
`GDNQKVSplitInputValues`, `PackedQKVComplexRotaryInputValues`,
`DSASparseDecodeKVPackInputValues`, `DSATopKSlotInputValues`,
`DSADecodeTopKInputValues`, `MLAKVPackQuantizeFP8InputValues`, and
`MLAPrefillFP8InputValues`.

Compressed sequence attention tests should start from
`CompressedSequenceAttentionInputValues`. Its nested `sliding_window`,
`compressed`, and `indexer` values contain the narrower bundles consumed by
TokenSpeed's SWA, HCA, CSA, indexer, cache-insert, cache-gather, and sparse
index helpers for DeepSeek V4-style kernels. The DeepSeek V4 inverse-RoPE FP8
quantization helper remains a separate utility and consumes
`DeepSeekV4InvRoPEFP8QuantInputValues`.

Generator values are operation-level values. Tests or adapters are responsible
for converting generated values into the exact keyword arguments expected by a
selected TokenSpeed backend.
