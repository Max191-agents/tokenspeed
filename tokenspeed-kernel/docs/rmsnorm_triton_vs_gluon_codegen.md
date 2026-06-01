# RMSNorm Triton vs Gluon Codegen Notes

GPU compile and benchmark checks used `HIP_VISIBLE_DEVICES=3`.

## Representative Cases

The generated artifacts compare Triton against the best bf16 Gluon candidates
from the expanded GPT-OSS sweep for `hidden_size=2880`.

| case | tokens | residual | candidate | p50 us | p90 us |
|---|---:|---|---|---:|---:|
| small non-residual baseline | 1 | false | `triton_rmsnorm` | 16.44 | 18.89 |
| small non-residual best Gluon | 1 | false | `block_full_spt4_w4` | 20.12 | 21.17 |
| small residual baseline | 1 | true | `triton_rmsnorm` | 19.18 | 20.38 |
| small residual best Gluon | 1 | true | `stream_c512_spt4_w4` | 21.26 | 23.48 |
| large non-residual baseline | 8192 | false | `triton_rmsnorm` | 16.96 | 17.39 |
| large non-residual best Gluon | 8192 | false | `stream_c2048_spt8_w4` | 18.46 | 21.15 |
| large residual baseline | 8192 | true | `triton_rmsnorm` | 29.76 | 30.26 |
| large residual best Gluon | 8192 | true | `block_full_spt4_w1` | 29.20 | 29.65 |

For fixed hidden size, dtype, residual mode, and candidate parameters, small
and large token counts compile to the same kernel body. Token count only
changes the launch grid. This was confirmed by identical Triton cache hashes
for `tokens=1` and `tokens=8192`:

- non-residual: `4FCYV7B3NJMO6EX6DM62MZS75OHGJZLXFY3EAV7ME62PEP44HVAA`
- residual: `ECB2UKHWNIP45TEDMOYUN7464KFQPF4653V4HLYQKOESZAR7ZREA`

## Artifact Generation

The same set of specializations can be regenerated through the tuning CLI:

```bash
rm -rf /tmp/tokenspeed_rmsnorm_compare
HIP_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/tokenspeed_rmsnorm_compare \
PYTHONPATH=tokenspeed-kernel/python \
../.venv/bin/python -m tokenspeed_kernel.benchmark.rmsnorm_tuning \
  --dtype bf16 --token-counts 1,8192 \
  --candidates triton_rmsnorm,block_full_spt4_w4,stream_c512_spt4_w4,stream_c2048_spt8_w4,block_full_spt4_w1,wave_row_spt4 \
  --warmup-iters 0 --bench-iters 1 --top-k 10 \
  --export ../claude_tmp/rmsnorm-codegen-compare.json
```

For this investigation I used separate `/tmp/tokenspeed_rmsnorm_compare_*`
cache directories per representative case so each artifact directory mapped to
one candidate/residual mode without further filtering.

## High-Level Algorithm

The Triton RMSNorm source and `block_full`/`wave_row` Gluon kernels implement
the same one-CTA-per-row algorithm:

1. `BLOCK = next_power_of_2(hidden_size) = 4096`.
2. Load the row with a mask for valid columns `< 2880`.
3. If residual mode is enabled, load residual, add it, and store
   `residual_out`.
4. Reduce `sum(x * x)` over the full row.
5. Compute `rsqrt(sum / hidden_size + eps)`.
6. Load f32 weights and store bf16 normalized output.

The streaming Gluon kernel deliberately changes this:

1. Split the row into fixed column chunks.
2. First pass: load each chunk and accumulate chunk reductions.
3. Compute one scalar `rstd`.
4. Second pass: reload input, or reload `residual_out` for residual mode,
   then store normalized chunks.

## Tile And Layout Differences

| candidate | TTGIR layout | row tensor | chunks | workgroup | shared | VGPR |
|---|---|---:|---:|---:|---:|---:|
| Triton non-residual | `sizePerThread=[8]`, `warpsPerCTA=[4]` | 4096 | 1 | 256 | 16 | 40 |
| Triton residual | `sizePerThread=[8]`, `warpsPerCTA=[4]` | 4096 | 1 | 256 | 16 | 40 |
| `block_full_spt4_w4` | `sizePerThread=[4]`, `warpsPerCTA=[4]` | 4096 | 1 | 256 | 16 | 32 |
| `stream_c512_spt4_w4` | `sizePerThread=[4]`, `warpsPerCTA=[4]` | 512 | 6 | 256 | 8 | 30 |
| `stream_c2048_spt8_w4` | `sizePerThread=[8]`, `warpsPerCTA=[4]` | 2048 | 2 | 256 | 16 | 32 |
| `block_full_spt4_w1` | `sizePerThread=[4]`, `warpsPerCTA=[1]` | 4096 | 1 | 64 | 0 | 122 |
| `wave_row_spt4` | `sizePerThread=[4]`, `warpsPerCTA=[1]` | 4096 | 1 | 64 | 0 | 122 |

Important differences:

- Triton independently chooses `sizePerThread=[8]` with four waves for the
  full-row kernel.
- The small non-residual Gluon winner uses four waves but only SPT 4.
- The large residual Gluon winner is effectively a one-wave full-row kernel.
  It removes LDS/shared-memory inter-wave reduction overhead but raises VGPR
  pressure substantially.
- Streaming lowers VGPR pressure, but it creates multiple chunk reductions and
  a second memory pass.

## TTGIR Differences

Triton's TTGIR is already AMD-specialized for memory:

- non-residual: 2 `amdg.buffer_load`, 1 `amdg.buffer_store`
- residual: 3 `amdg.buffer_load`, 2 `amdg.buffer_store`
- x/out bf16 accesses carry `contiguity = 8`
- weight f32 accesses carry `contiguity = 4`

Gluon TTGIR stayed at generic `tt.load`/`tt.store` for these kernels:

- `block_full_spt4_w4`: 2 `tt.load`, 1 `tt.store`
- `stream_c512_spt4_w4`: 24 `tt.load`, 12 `tt.store`
- `stream_c2048_spt8_w4`: 5 `tt.load`, 2 `tt.store`
- `block_full_spt4_w1`: 3 `tt.load`, 2 `tt.store`

The extra streaming operations are expected: for `COL_BLOCK=512`, the static
loop unrolls six chunks for a 2880-wide row; residual mode also stores
`residual_out` in pass 1 and reloads it in pass 2.

## ISA Differences

Instruction counts below are static counts from the `.amdgcn` text. They are
useful for comparing code shape, not dynamic instruction counts.

| candidate | ISA body lines | memory op shape | LDS/barriers | VGPR |
|---|---:|---|---|---:|
| Triton non-residual | 153 | 6 `buffer_load_dwordx4`, 2 `buffer_store_dwordx4` | 1 LDS write/read + 1 barrier | 40 |
| Triton residual | 199 | 8 `buffer_load_dwordx4`, 4 `buffer_store_dwordx4` | 1 LDS write/read + 1 barrier | 40 |
| `block_full_spt4_w4` | 138 | mixed `global_load_dwordx2/x4`, `global_store_dwordx2` | 1 LDS write/read + 1 barrier | 32 |
| `stream_c512_spt4_w4` | 497 | many `global_load_dwordx2/x4`, `global_store_dwordx2` | 6 LDS write/read + 11 barriers | 30 |
| `stream_c2048_spt8_w4` | 229 | 7 `global_load_dwordx4`, 2 `global_store_dwordx4` | 2 LDS write/read + 3 barriers | 32 |
| `block_full_spt4_w1` | 436 | many `global_load_dwordx2/x4`, `global_store_dwordx2` | no LDS/barrier | 122 |

Key observations:

- Triton reaches compact `buffer_*_dwordx4` ISA for row and weight traffic.
  This follows from the AMD buffer ops and contiguity annotations in TTGIR.
- Gluon generally lowers to `global_*` ISA. Some variants still get dwordx4
  loads, but stores are often dwordx2 and the full-row one-wave path emits many
  static memory instructions.
- Four-wave full-row reductions use one LDS handoff/barrier. One-wave
  full-row reductions avoid LDS/barriers but pay high VGPR and larger code.
- Streaming does reduce live row footprint and VGPR count, but each chunk adds
  reduction machinery. `COL_BLOCK=512` is particularly expensive in residual
  mode: six unrolled chunks, second-pass reloads, and 11 barriers in the final
  ISA.

## Takeaways

Triton is faster in most GPT-OSS bf16/fp16 cases because its generated kernel is
the compact full-row strategy we wanted, with a stronger memory lowering path:
four waves, SPT 8, one full-row reduction, AMD buffer ops, vectorized dwordx4
memory instructions, and only one inter-wave reduction barrier.

The Gluon kernels are exploring useful algorithmic alternatives, but their
current best cases win for different reasons:

- Small shapes: the best Gluon candidates are still slower. Token count does
  not change kernel code, so small-shape performance is mostly launch overhead
  plus one row of work. Triton's compact buffer-op kernel has the advantage.
- Large non-residual: `stream_c2048_spt8_w4` gets close, but the second pass
  and extra reductions still lose to Triton's one-pass full-row implementation.
- Large residual: `block_full_spt4_w1` narrowly wins. The likely advantage is
  eliminating LDS/barrier overhead with one wave per row once there are many
  rows in flight. The cost is high VGPR pressure and much larger static code,
  so it is not a clear default candidate.

Optimization implications:

- A high-priority experiment is getting Gluon to produce the same AMD buffer
  memory path as Triton for full-row kernels, including contiguity/vectorization
  comparable to Triton's SPT 8 layout.
- For small decode, avoid chunked streaming unless it materially reduces launch
  floor or codegen overhead; the chunked variants add too much static work for
  one row.
- For large residual, keep a one-wave/full-row variant in the tuning set, but
  treat it as a shape-specific candidate because VGPR/code size are high.
- For streaming, prefer larger chunks when using it at all. `COL_BLOCK=2048`
  has far less unrolled reduction/barrier overhead than `COL_BLOCK=512`.
