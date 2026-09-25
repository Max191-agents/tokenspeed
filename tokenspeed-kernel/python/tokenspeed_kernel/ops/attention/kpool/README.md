# KPool attention kernels

KPool selection scores compressed index keys, selects logical pools, expands
each selected pool into raw FlatKV slots, and appends the visible incomplete
pool tail.

The GLM-5.3-Flash specialization has fixed geometry:

- 32 query heads with head dimension 128
- 4 raw tokens per compressed pool
- 16 compressed rows per index-cache page
- 512 selected pools
- BF16 queries, signed BF16 or FP32 head weights, and scaled FP8 E4M3 keys
- weighted per-head ReLU scoring and global FlatKV-slot output

## Prefill selection

The hybrid Gluon implementation keeps one orchestration path across AMD
architectures. The architecture backend supplies three stages:

1. Score a bounded pool window through either the request page table or the
   precomputed physical-slot plan.
2. Select logical columns with the architecture's radix top-k for long or
   merged windows.
3. Reuse the portable Triton payload gather, deterministic short-window sort,
   and pool-to-FlatKV expansion.

GFX950 uses Wave64 MFMA scoring and its logical radix selector. GFX1250 uses
Wave32 WMMA-v3 scoring and its existing Wave32 radix selector. Short
single-window rows of at most 2048 pools fold signed head contributions in
logical head order before deterministic sorting, preserving stable pool IDs
for equal scores.

Production prefill supplies `pool_workspace_slots`, `row_starts`, and
`row_ends`. These tensors preserve request-local logical pool order while
addressing physical cache rows directly. The eager-only compatibility path
reconstructs request IDs and causal lengths and scores through the index page
table.

Scoring workspaces are row-tiled under `max_logits_bytes`. The cap includes
the persistent sort or radix intermediates as well as logits; one row remains
legal when its workspace exceeds the cap.

## Decode selection on gfx950

For GLM-5.3's four-query verification steps at batch sizes 2, 4, 8, and 16,
the gfx950 path converts BF16 queries to FP8, scores the scaled FP8 index
cache, and combines top-k selection with FlatKV slot mapping. It requires the
model's 1/64 score scale. The registry selects by query shape and dtype, then
the wrapper delegates other score scales to the portable decode path. Batch
size and page-table capacity set graph-static workspace geometry; live
sequence lengths and page-table contents remain device inputs so a
captured graph can replay at different context lengths.
