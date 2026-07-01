# Sampling Input Generators

Sampling generators cover deterministic helper operations used around token
selection. They do not generate stochastic draws from a distribution. Instead,
they model the tensor transforms that TokenSpeed kernels currently implement:
row-wise argmax, packed max/index argmax pairs, scalar metadata
gather/broadcast, softmax, speculative greedy verification, target-only chain
speculative sampling, min-p renormalization, exact-checkable min-p sampling,
top-p renormalization, top-k/top-p renormalization, and exact-checkable
top-k/top-p sampling from probabilities or logits.

## Operation Semantics

### Argmax

`ArgmaxInputs` represents row-wise argmax over logits. The semantic result is
the lowest column index whose value is maximal in each row. By default the
generator plants a known dominant maximum in every row, so correctness tests do
not depend on accidental random maxima or ties. It can also generate rows with
two equal maxima to exercise the first-index tie rule, or leave random logits
untouched and derive expected indices from the reference.

For kernels that define NaN handling, the argmax reference ignores NaN values
when any non-NaN value is present and returns `-1` for all-NaN rows. The
generator can produce all-NaN rows or mixed batches containing finite maxima
among NaNs, all-valid `-inf` rows, tied finite maxima among NaNs, and all-NaN
rows.

### Argmax Pair

`ArgmaxPairInputs` represents row-wise max plus argmax packed into a float32
tensor with shape `[rows, 2]`:

```text
out[row, 0] = max(logits[row])
out[row, 1] = first argmax index for logits[row]
```

This models direct helper APIs that need both the selected value and selected
index. The generator can create the optional caller-provided output buffer used
by in-place kernel paths. Its logit generation supports the same unique,
tied, and random maximum patterns as `ArgmaxInputs`.

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

### Target-Only Chain Speculative Sampling

`SpeculativeChainSamplingInputs` represents chain-speculative verification when
the target model provides probability rows. Position 0 is accepted by
construction. Later draft tokens are accepted while either:

```text
target_prob(candidate) >= threshold_single
```

or:

```text
uniform <= target_prob(candidate) / threshold_acc
```

The first rejected slot, or the final bonus slot when all draft tokens are
accepted, is sampled from `relu(target_probs - draft_probs)`. When
`draft_probs` is omitted, the operation becomes the target-only runtime path and
the sample is drawn from target probabilities with the rejected token removed.

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

### Min-P Sampling

`MinPSamplingInputs` represents categorical sampling after the same min-p
filtering rule:

```text
threshold = min_p[row] * max(probs[row])
filtered = where(probs >= threshold, probs, 0)
sample ~ categorical(filtered / sum(filtered))
```

General categorical sampling requires statistical validation, so this generator
currently produces exact-checkable rows where exactly one token survives the
min-p filter. The reference returns that deterministic token and a valid mask.
This lets kernel smoke tests verify API wiring and filtering semantics without
claiming to validate the full random-sampling distribution.

### Top-P Renormalization

`TopPRenormInputs` represents nucleus filtering and renormalization:

```text
sorted = sort(probs[row], descending=True)
keep = smallest prefix whose cumulative probability reaches top_p[row]
out = where(probs is in keep, probs, 0)
out = out / sum(out)
```

The generated `top_p` tensor has one threshold per probability row. This
operation is deterministic filtering only; stochastic token sampling from the
renormalized row is outside this generator's contract.

### Top-K + Top-P Renormalization

`TopKTopPRenormInputs` represents top-k filtering followed by top-p filtering
and renormalization. Top-k values can include disabled rows by setting `k` equal
to the vocabulary size or any larger disabled-top-k sentinel. The generator can
produce batches where only top-k is active, only top-p is active, both filters
are active, or different rows use different modes.

### Top-K + Top-P Sampling

`TopKTopPSamplingInputs` represents categorical sampling after top-k and top-p
filtering from probability rows:

```text
filtered = apply_top_p(apply_top_k(probs, top_k), top_p)
sample ~ categorical(filtered / sum(filtered))
```

`TopKTopPLogitsSamplingInputs` represents the same filtering and sampling after
first applying row-wise softmax to logits:

```text
probs = softmax(logits)
filtered = apply_top_p(apply_top_k(probs, top_k), top_p)
sample ~ categorical(filtered / sum(filtered))
```

As with min-p sampling, the general operation is stochastic. The generator
currently creates exact-checkable rows where one token survives both filters. In
the logits variant, this is done by generating one finite logit per row. The
reference can then return a deterministic sample and valid mask. This covers the
operation-level filtering/API contract while leaving full distributional
sampling validation to a future statistical harness.

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
- chain speculative sampling probabilities are fp32, candidate IDs are in
  range, uniform samples are in `[0, 1)`, thresholds are coherent, and generated
  rows force the configured accepted-prefix range
- chain speculative sampling output buffers use int32 storage and invalid slots
  are initialized to a sentinel so consumers can mask with `accept_index`
- min-p and top-p values are generated in valid probability ranges
- min-p sampling rows have exactly one survivor after filtering for exact
  reference comparison
- standalone top-p thresholds have one value per probability row and are in
  `(0, 1]`
- top-k/top-p renormalization metadata uses either finite positive top-k values
  bounded by the configured maximum or a validated disabled-top-k value greater
  than or equal to the vocabulary size
- top-k/top-p sampling rows have one survivor after both filters for exact
  reference comparison, with logits rows containing one finite token before
  softmax
- planted argmax rows have either a unique known maximum or two tied known
  maxima whose lower index is the expected result
- NaN-focused argmax rows either produce the documented `-1` all-NaN sentinel
  or include enough valid non-NaN values to make the expected first argmax
  index unambiguous

Metadata such as argmax planted indices, scalar gather indices, top-k/top-p
controls, speculative accepted-prefix lengths, and min-p thresholds are generated
from `metadata_seed`. Tensor values such as logits, probability distributions,
generated token IDs, and acceptance coins are generated from `seed`.

## TokenSpeed API Mapping

TokenSpeed exposes sampling kernels through several modules:

- `sampling.argmax` for row-wise argmax
- `sampling.cute_dsl.argmax_pair` for packed row-wise max/index pairs
- FlashInfer `sampling.softmax` consumes `SoftmaxInputValues.logits` and
  optional `.temperature`
- `sampling.triton.gather_and_expand_scalars`
- `sampling.triton.min_p_renorm_prob`
- FlashInfer `min_p_sampling_from_probs` consumes `MinPSamplingInputValues`
- NVIDIA-only CUDA `verify_chain_greedy` consumes
  `SpeculativeGreedyVerifyInputValues`
- NVIDIA-only CUDA `chain_speculative_sampling_target_only` consumes
  `SpeculativeChainSamplingInputValues`
- NVIDIA-only fused top-k/top-p renormalization helpers
- FlashInfer `top_k_top_p_sampling_from_probs` consumes
  `TopKTopPSamplingInputValues`
- FlashInfer `top_k_top_p_sampling_from_logits` consumes
  `TopKTopPLogitsSamplingInputValues`
- FlashInfer `top_p_renorm_probs` consumes `TopPRenormInputValues.probs` and
  `.top_p`
- FlashInfer `top_k_renorm_prob` followed by deterministic
  `top_p_renorm_prob` consumes `TopKTopPRenormInputValues.probs`, `.top_k`,
  and `.top_p`

The standalone generators return operation-level values and references. Tests
or adapters are responsible for selecting TokenSpeed solutions, passing `out=`
buffers, and handling backend-specific availability.
