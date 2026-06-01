# RMSNorm Gluon IR Verification

This note records IR/assembly checks for the experimental RMSNorm Gluon
strategies. GPU runs use `HIP_VISIBLE_DEVICES=3`.

## `gluon_rmsnorm_block_full`

Command:

```bash
rm -rf /tmp/tokenspeed_rmsnorm_block_full_cache
HIP_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/tokenspeed_rmsnorm_block_full_cache \
PYTHONPATH=tokenspeed-kernel/python \
../.venv/bin/python ../claude_tmp/compile_block_full_ir.py
```

The compile script ran two GPT-OSS-shaped bf16 cases:

- non-residual: `x=[1, 2880]`, `weight=[2880]`;
- residual: `x=[1, 2880]`, `residual=[1, 2880]`, `weight=[2880]`.

Generated artifacts:

- non-residual:
  `/tmp/tokenspeed_rmsnorm_block_full_cache/5AXG7TQQFWOAWN6FH4GFQ42WP33VADTF6P4ZF56KAJTXD35CP77A/_rmsnorm_block_full_kernel.{ttgir,llir,amdgcn,s}`;
- residual:
  `/tmp/tokenspeed_rmsnorm_block_full_cache/NXHDRBET55LKUAX7DPABQD3IUMJEDSSE73T3ZHULYAKX236ZX6RQ/_rmsnorm_block_full_kernel.{ttgir,llir,amdgcn,s}`.

Evidence:

- Both metadata files target `hip:gfx950`, use `num_ctas=1`,
  `num_warps=4`, and `warp_size=64`.
- TTGIR uses one row program id:
  `tt.get_program_id x`, then `row * 2880 + offsets`.
- TTGIR layout is a single 4096-element block layout:
  `#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>`.
- TTGIR contains one full-row `tt.reduce` over
  `tensor<4096xf32, #blocked>` for `sum(x * x)`.
- The residual variant loads `x` and `residual`, adds them, stores
  `residual_out`, then performs the same single full-row reduction.
- There is no multi-CTA row splitting: metadata has `num_ctas=1`, and the
  kernel only uses the row program id on the x dimension.
- There is no streaming two-pass chunk loop: the row is loaded once into the
  4096-element block tensor, reduced once, then normalized and stored.

Assembly metadata:

- non-residual: `vgpr_count=35`, `vgpr_spill_count=0`,
  `sgpr_spill_count=0`, `private_segment_fixed_size=0`;
- residual: `vgpr_count=38`, `vgpr_spill_count=0`,
  `sgpr_spill_count=0`, `private_segment_fixed_size=0`.

## `gluon_rmsnorm_wave_row`

Command:

```bash
rm -rf /tmp/tokenspeed_rmsnorm_wave_row_cache
HIP_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/tokenspeed_rmsnorm_wave_row_cache \
PYTHONPATH=tokenspeed-kernel/python \
../.venv/bin/python ../claude_tmp/compile_wave_row_ir.py
```

The compile script ran two GPT-OSS-shaped bf16 cases:

- non-residual: `x=[1, 2880]`, `weight=[2880]`;
- residual: `x=[1, 2880]`, `residual=[1, 2880]`, `weight=[2880]`.

Generated artifacts:

- non-residual:
  `/tmp/tokenspeed_rmsnorm_wave_row_cache/E762QEBQMO6IUB6AKEQJU6NRPPS27X7CIUKGLTMWBMTJR7ED6JMA/_rmsnorm_wave_row_kernel.{ttgir,llir,amdgcn,s}`;
- residual:
  `/tmp/tokenspeed_rmsnorm_wave_row_cache/6W5JDIV2M7EVQUPXNRQ5TY5OVWPFLOZZNM627QAQ2AXLLQTU7IUA/_rmsnorm_wave_row_kernel.{ttgir,llir,amdgcn,s}`.

Evidence:

- Both metadata files target `hip:gfx950`, use `num_ctas=1`,
  `num_warps=1`, and `warp_size=64`.
- TTGIR uses one row program id:
  `tt.get_program_id x`, then `row * 2880 + offsets`.
- TTGIR layout is a single-wave 4096-element block layout:
  `#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>`.
- TTGIR contains one `tt.reduce` over `tensor<4096xf32, #blocked>`.
  Because `warpsPerCTA=[1]` and `num_warps=1`, this reduction is confined
  to one hardware wave for the row.
- The residual variant loads `x` and `residual`, adds them, stores
  `residual_out`, then performs the same wave-local row reduction.
- There is no inter-wave/block reduction for a row: one CTA contains one
  wave, and the assembly reports `.max_flat_workgroup_size: 64`.
- There is no multi-CTA row splitting: metadata has `num_ctas=1`.
- There is no streaming two-pass chunk loop: the row is loaded as one
  4096-element tensor, reduced once, then normalized and stored.

Assembly metadata:

- non-residual: `vgpr_count=181`, `vgpr_spill_count=0`,
  `sgpr_spill_count=0`, `private_segment_fixed_size=0`;
- residual: `vgpr_count=195`, `vgpr_spill_count=0`,
  `sgpr_spill_count=0`, `private_segment_fixed_size=0`.

## `gluon_rmsnorm_streaming_block`

Command:

```bash
rm -rf /tmp/tokenspeed_rmsnorm_streaming_block_cache
HIP_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/tokenspeed_rmsnorm_streaming_block_cache \
PYTHONPATH=tokenspeed-kernel/python \
../.venv/bin/python ../claude_tmp/compile_streaming_block_ir.py
```

The compile script ran two GPT-OSS-shaped bf16 cases:

- non-residual: `x=[1, 2880]`, `weight=[2880]`;
- residual: `x=[1, 2880]`, `residual=[1, 2880]`, `weight=[2880]`.

Generated artifacts:

- non-residual:
  `/tmp/tokenspeed_rmsnorm_streaming_block_cache/TTSCFFYQ7XVYNS5C2LBLHQTXTX7YW3AZOS6AGOO5PSK5NBL5XV3Q/_rmsnorm_streaming_block_kernel.{ttgir,llir,amdgcn,s}`;
- residual:
  `/tmp/tokenspeed_rmsnorm_streaming_block_cache/ZGESFITIIMEG3RQ4WXN7MYNNCJ7W732A5QS3LA5NMIAD5Y7PMDNA/_rmsnorm_streaming_block_kernel.{ttgir,llir,amdgcn,s}`.

Evidence:

- Both metadata files target `hip:gfx950`, use `num_ctas=1`,
  `num_warps=4`, and `warp_size=64`.
- TTGIR uses one row program id:
  `tt.get_program_id x`, then `row * 2880 + chunk_offsets`.
- TTGIR uses chunk tensors, not full-row tensors. The visible layout is
  `tensor<1024xf32, #blocked>`, and there are no `tensor<2880...>` or
  `tensor<4096...>` values in the source-level IR.
- For `hidden_size=2880` and `COL_BLOCK=1024`, the static loop unrolls into
  chunk bases `0`, `1024`, and `2048`.
- The first pass has three sum calls over `tensor<1024xf32, #blocked>`.
  The non-residual variant has 9 `tt.load` operations and 3 `tt.store`
  operations; the residual variant has 12 `tt.load` operations and 6
  `tt.store` operations.
- `rstd` is computed only after those three chunk reductions. The second pass
  then splats the scalar `rstd` into each 1024-wide chunk and stores the
  normalized output chunk by chunk.
- Residual policy: the residual variant stores `x + residual` to
  `residual_out` during the first pass, then the second pass reloads
  `residual_out` for the normalized output. It does not reload both `x` and
  `residual` in the second pass.
- There is no multi-CTA row splitting: metadata has `num_ctas=1`, and the
  kernel only uses the row program id on the x dimension.

Assembly metadata:

- non-residual: `.max_flat_workgroup_size=256`, `vgpr_count=30`,
  `vgpr_spill_count=0`, `sgpr_spill_count=0`,
  `private_segment_fixed_size=0`;
- residual: `.max_flat_workgroup_size=256`, `vgpr_count=31`,
  `vgpr_spill_count=0`, `sgpr_spill_count=0`,
  `private_segment_fixed_size=0`.
