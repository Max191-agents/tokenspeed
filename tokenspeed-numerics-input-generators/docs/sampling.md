# Sampling Input Generators

Sampling generators cover deterministic helper operations used around token
selection. They do not generate stochastic draws from a distribution. Instead,
they model the tensor transforms that TokenSpeed kernels currently implement:
row-wise argmax, scalar metadata gather/broadcast, min-p renormalization, and
top-k/top-p renormalization.

## Operation Semantics

### Argmax

`ArgmaxInputs` represents row-wise argmax over logits. The semantic result is
the lowest column index whose value is maximal in each row. By default the
generator plants a known dominant maximum in every row, so correctness tests do
not depend on accidental random maxima or ties.

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
- probability tensors are generated as fp32 normalized rows
- output/index dtypes must be int32 or int64 where relevant
- min-p and top-p values are generated in valid probability ranges
- planted argmax rows have a unique known maximum

Metadata such as argmax planted indices, scalar gather indices, top-k/top-p
controls, and min-p thresholds are generated from `metadata_seed`. Tensor values
such as logits and probability distributions are generated from `seed`.

## TokenSpeed API Mapping

TokenSpeed exposes sampling kernels through several modules:

- `sampling.argmax` for row-wise argmax
- `sampling.triton.gather_and_expand_scalars`
- `sampling.triton.min_p_renorm_prob`
- NVIDIA-only fused top-k/top-p renormalization helpers

The standalone generators return operation-level values and references. Tests
or adapters are responsible for selecting TokenSpeed solutions, passing `out=`
buffers, and handling backend-specific availability.
