# Sampling Input Generators

Sampling generators cover deterministic helper operations used around token
selection. They do not generate stochastic draws from a distribution. Instead,
they model the tensor transforms that TokenSpeed kernels currently implement:
row-wise argmax, packed max/index argmax pairs, scalar metadata
gather/broadcast, softmax, speculative greedy verification, min-p
renormalization, and top-k/top-p renormalization.

## Operation Semantics

### Argmax

`ArgmaxInputs` represents row-wise argmax over logits. The semantic result is
the lowest column index whose value is maximal in each row. By default the
generator plants a known dominant maximum in every row, so correctness tests do
not depend on accidental random maxima or ties.

### Argmax Pair

`ArgmaxPairInputs` represents row-wise max plus argmax packed into a float32
tensor with shape `[rows, 2]`:

```text
out[row, 0] = max(logits[row])
out[row, 1] = first argmax index for logits[row]
```

This models direct helper APIs that need both the selected value and selected
index. The generator can create the optional caller-provided output buffer used
by in-place kernel paths.

### Softmax

`SoftmaxInputs` represents row-wise softmax over logits:

```text
out = softmax(logits / temperature)
```

`temperature` can be omitted, shared by every row as a scalar, or generated as
one positive fp32 value per row. The reference always returns fp32 normalized
probability rows.

### Speculative Greedy Verification

`SpeculativeGreedyVerifyInputs` represents deterministic verification of a
chain-speculative draft sequence under greedy target-model sampling. For each
request row, the accepted draft-token count is the length of the matching
prefix:

```text
candidates[row, i + 1] == target_predict[row, i]
```

The verifier writes the target predictions into the `predicts` output buffer,
writes flat accepted positions plus the final bonus-token position into
`accept_index`, and writes the accepted draft-token count into
`accept_token_num`.

### Gather And Expand Scalars

`GatherExpandScalarsInputs` represents:

```text
out = pool[index].repeat_interleave(n)
```

for sampling scalar pools such as temperature, top-k, top-p, min-p, seeds, and
offsets. Optional streams can be omitted to model call paths that do not use
min-p or RNG metadata.

### Min-P Renormalization

`MinPRenormInputs` represents filtering probabilities by a per-row min-p ratio:

```text
threshold = min_p[row] * max(probs[row])
out = where(probs >= threshold, probs, 0)
out = out / sum(out)
```

Generated probability rows are positive and normalized before filtering.

### Top-K + Top-P Renormalization

`TopKTopPRenormInputs` represents top-k filtering followed by top-p filtering
and renormalization. Top-k values can include disabled rows by setting `k` equal
to the vocabulary size.

## Validation Contract

The generators reject invalid sampling inputs before values are returned:

- row counts and batch sizes must be non-negative
- vocabulary sizes, scalar pool sizes, and repeat counts must be positive
- logits dtypes must be fp16, bf16, or fp32
- softmax temperatures must be positive when present
- probability tensors are generated as fp32 normalized rows
- output/index dtypes must be int32 or int64 where relevant
- argmax-pair output buffers are float32 with shape `[rows, 2]`
- speculative greedy verification token IDs are in range and force coherent
  matching/mismatching prefixes
- speculative greedy output buffers have int32 storage and shapes matching the
  accepted-prefix contract
- min-p and top-p values are generated in valid probability ranges
- planted argmax rows have a unique known maximum

Metadata such as argmax planted indices, scalar gather indices, top-k/top-p
controls, speculative accepted-prefix lengths, and min-p thresholds are
generated from `metadata_seed`. Tensor values such as logits, probability
distributions, and generated token IDs are generated from `seed`.

## TokenSpeed API Mapping

TokenSpeed exposes sampling kernels through several modules:

- `sampling.argmax` for row-wise argmax
- `sampling.cute_dsl.argmax_pair` for packed row-wise max/index pairs
- FlashInfer `sampling.softmax` consumes `SoftmaxInputValues.logits` and
  optional `.temperature`
- `sampling.triton.gather_and_expand_scalars`
- `sampling.triton.min_p_renorm_prob`
- NVIDIA-only CUDA `verify_chain_greedy` consumes
  `SpeculativeGreedyVerifyInputValues`
- NVIDIA-only fused top-k/top-p renormalization helpers
- FlashInfer `top_k_renorm_prob` followed by deterministic
  `top_p_renorm_prob` consumes `TopKTopPRenormInputValues.probs`, `.top_k`,
  and `.top_p`

The standalone generators return operation-level values and references. Tests
or adapters are responsible for selecting TokenSpeed solutions, passing `out=`
buffers, and handling backend-specific availability.
