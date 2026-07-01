# Sampling Generators

Sampling generators cover row-wise selection and probability filtering used in
token sampling. The operation semantics include deterministic argmax behavior,
metadata gather/broadcast, softmax, deterministic speculative greedy
verification, target-only chain speculative sampling, min-p filtering, and
exact-checkable min-p sampling, top-p or top-k followed by top-p
renormalization, and exact-checkable top-k/top-p sampling.

## Generators

- `ArgmaxInputs`: generates logits with optional planted unique maxima and
  optional output buffers.
- `ArgmaxPairInputs`: generates logits for `(max_value, argmax_index)` outputs.
- `GatherExpandScalarsInputs`: generates scalar pools, request indices, and
  optional min-p/seed/offset streams.
- `SoftmaxInputs`: generates logits plus optional scalar or per-row temperature
  values for row-wise softmax.
- `SpeculativeGreedyVerifyInputs`: generates chain-speculative candidate token
  IDs, target greedy predictions, and verifier output buffers.
- `SpeculativeChainSamplingInputs`: generates chain-speculative candidate token
  IDs, target probability rows, optional draft probabilities, acceptance coins,
  and verifier output buffers.
- `MinPRenormInputs`: generates normalized probability rows and min-p
  thresholds.
- `MinPSamplingInputs`: generates probability rows whose min-p filtered
  support has one deterministic sampled token.
- `TopPRenormInputs`: generates normalized probability rows and top-p
  thresholds for standalone nucleus filtering.
- `TopKTopPRenormInputs`: generates normalized probability rows, top-k values,
  and top-p thresholds.
- `TopKTopPSamplingInputs`: generates probability rows whose top-k/top-p
  filtered support has one deterministic sampled token.

## Generated Values

Softmax references and probability generators return fp32 rows normalized to
sum to one. Argmax generators can plant unique maxima so correctness tests are
not dominated by random tie behavior. Metadata generators keep request indices
in range and expose optional streams explicitly.

References model the sampling operation directly and are suitable for comparing
different kernel implementations with the same generated inputs.
