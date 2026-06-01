# RMSNorm Gluon GPT-OSS Benchmark Report

GPU runs used `HIP_VISIBLE_DEVICES=3`.

## Scope

Benchmarked GPT-OSS RMSNorm shapes:

- `hidden_size=2880`
- decode-like `num_tokens`: 1, 2, 4, 8, 16, 32
- prefill/extend-like `num_tokens`: 64, 128, 256, 512, 1024, 2048,
  4096, 8192
- `residual=false` and `residual=true`
- odd sanity shape: `num_tokens=11`, `hidden_size=2897`
- dtypes: bf16 and fp16

Command:

```bash
HIP_VISIBLE_DEVICES=3 \
PYTHONPATH=tokenspeed-kernel/python \
../.venv/bin/python -m tokenspeed_kernel.benchmark.rmsnorm_tuning \
  --dtype bf16 --include-odd --warmup-iters 10 --bench-iters 100 \
  --top-k 5 --export ../claude_tmp/rmsnorm-tuning-bf16-full.json
```

The fp16 run used the same command with `--dtype fp16` and exported
`../claude_tmp/rmsnorm-tuning-fp16-full.json`.

Those initial tables used the first tuned candidate set. The current tuning CLI
also sweeps `size_per_thread` for `block_full` and `streaming_block`; the
expanded SPT results are summarized below.

The normal benchmark CLI was also smoke-tested:

```bash
HIP_VISIBLE_DEVICES=3 \
PYTHONPATH=tokenspeed-kernel/python \
../.venv/bin/python -m tokenspeed_kernel.benchmark \
  --op norm.rmsnorm --dtype bf16 --dtype-role x \
  --shapes '[{"num_tokens":1,"hidden_size":2880,"residual":false}]' \
  --warmup-iters 1 --bench-iters 3 --no-verify
```

It listed `triton_rmsnorm`, `gluon_rmsnorm`,
`gluon_rmsnorm_block_full`, `gluon_rmsnorm_wave_row`, and
`gluon_rmsnorm_streaming_block`, with p50/p90/p99 and effective GB/s.

Effective bandwidth uses the op-level logical byte model in
`ThroughputCalculator`: x read, f32 weight read per row, output write, plus
residual read and residual_out write for residual mode. The streaming residual
kernel has one additional implementation-level residual_out read that is not
counted in this logical bandwidth number.

## Verification

- bf16 sweep: 630 rows, 0 codegen/runtime errors, 0 numerical failures.
- fp16 sweep: 630 rows, 0 codegen/runtime errors, 0 numerical failures.
- Odd shape `hidden_size=2897` passed for every tuned candidate in both dtypes
  and residual modes.

## BF16 Results

Best Gluon candidate compared with Triton:

| residual | tokens | Triton p50 us | Best Gluon | Gluon p50 us | Gluon/Triton |
|---|---:|---:|---|---:|---:|
| false | 1 | 16.48 | stream_c512_w8 | 21.48 | 1.30 |
| false | 2 | 17.64 | block_full_w4 | 21.84 | 1.24 |
| false | 4 | 17.94 | block_full_w8 | 21.58 | 1.20 |
| false | 8 | 17.80 | stream_c512_w2 | 20.52 | 1.15 |
| false | 16 | 17.94 | stream_c512_w8 | 20.52 | 1.14 |
| false | 32 | 18.02 | stream_c512_w4 | 20.48 | 1.14 |
| false | 64 | 17.70 | stream_c512_w8 | 20.46 | 1.16 |
| false | 128 | 17.82 | stream_c512_w8 | 20.76 | 1.17 |
| false | 256 | 18.12 | stream_c512_w8 | 20.58 | 1.14 |
| false | 512 | 17.72 | stream_c512_w2 | 20.48 | 1.16 |
| false | 1024 | 17.92 | stream_c1024_w2 | 20.72 | 1.16 |
| false | 2048 | 17.42 | block_full_w2 | 21.66 | 1.24 |
| false | 4096 | 17.80 | wave_row_spt2 | 22.00 | 1.24 |
| false | 8192 | 17.08 | stream_c2048_w4 | 20.48 | 1.20 |
| true | 1 | 19.98 | block_full_w8 | 22.46 | 1.12 |
| true | 2 | 20.90 | wave_row_spt1 | 22.96 | 1.10 |
| true | 4 | 20.78 | stream_c2048_w1 | 22.82 | 1.10 |
| true | 8 | 20.72 | stream_c512_w8 | 21.80 | 1.05 |
| true | 16 | 20.72 | stream_c512_w1 | 21.58 | 1.04 |
| true | 32 | 20.92 | stream_c512_w4 | 21.72 | 1.04 |
| true | 64 | 20.94 | stream_c512_w4 | 21.80 | 1.04 |
| true | 128 | 20.76 | stream_c512_w2 | 21.84 | 1.05 |
| true | 256 | 20.58 | stream_c512_w1 | 21.80 | 1.06 |
| true | 512 | 20.74 | stream_c1024_w2 | 21.88 | 1.05 |
| true | 1024 | 20.96 | block_full_w4 | 22.56 | 1.08 |
| true | 2048 | 20.68 | wave_row_spt2 | 23.12 | 1.12 |
| true | 4096 | 21.10 | stream_c1024_w2 | 21.46 | 1.02 |
| true | 8192 | 29.68 | wave_row_spt4 | 29.36 | 0.99 |

Best config per Gluon strategy/regime, selected by geometric mean p50 over the
shapes in that regime:

| dtype | residual | regime | strategy | best config | geom mean p50 us |
|---|---|---|---|---|---:|
| bf16 | false | decode | block_full | block_full_w8 | 21.80 |
| bf16 | false | decode | wave_row | wave_row_spt1 | 21.73 |
| bf16 | false | decode | streaming_block | stream_c512_w4 | 21.23 |
| bf16 | false | prefill | block_full | block_full_w4 | 21.89 |
| bf16 | false | prefill | wave_row | wave_row_spt2 | 21.88 |
| bf16 | false | prefill | streaming_block | stream_c512_w2 | 21.35 |
| bf16 | true | decode | block_full | block_full_w1 | 22.97 |
| bf16 | true | decode | wave_row | wave_row_spt1 | 22.93 |
| bf16 | true | decode | streaming_block | stream_c512_w2 | 22.38 |
| bf16 | true | prefill | block_full | block_full_w8 | 23.78 |
| bf16 | true | prefill | wave_row | wave_row_spt1 | 23.74 |
| bf16 | true | prefill | streaming_block | stream_c512_w2 | 23.88 |

## FP16 Cross-Check

fp16 did not change the recommendation. Triton won 27 of 28 GPT-OSS shapes.
The only Gluon win was again `num_tokens=8192`, `residual=true`, where
`wave_row_spt4` measured 29.04 us p50 versus Triton at 29.68 us p50.

The best Gluon configs by regime were similar:

| dtype | residual | regime | strategy | best config | geom mean p50 us |
|---|---|---|---|---|---:|
| fp16 | false | decode | block_full | block_full_w8 | 22.18 |
| fp16 | false | decode | wave_row | wave_row_spt4 | 22.16 |
| fp16 | false | decode | streaming_block | stream_c512_w4 | 21.53 |
| fp16 | false | prefill | block_full | block_full_w1 | 22.00 |
| fp16 | false | prefill | wave_row | wave_row_spt1 | 21.98 |
| fp16 | false | prefill | streaming_block | stream_c512_w2 | 21.62 |
| fp16 | true | decode | block_full | block_full_w1 | 23.40 |
| fp16 | true | decode | wave_row | wave_row_spt4 | 23.38 |
| fp16 | true | decode | streaming_block | stream_c512_w2 | 22.67 |
| fp16 | true | prefill | block_full | block_full_w1 | 24.12 |
| fp16 | true | prefill | wave_row | wave_row_spt4 | 24.00 |
| fp16 | true | prefill | streaming_block | stream_c512_w4 | 24.18 |

## Expanded SPT Sweep

The CLI now uses dtype-aware SPT defaults:

- bf16/fp16: `size_per_thread=1,2,4,8,16`
- fp32: `size_per_thread=1,2,4,8`

This corresponds to testing up to 256-bit per-thread fp16/bf16 vectors and
256-bit per-thread fp32 vectors. The expanded runs used the same GPT-OSS shape
set, `--include-odd`, `--warmup-iters 10`, and `--bench-iters 100`.

| dtype | rows | errors | numerical failures | best-candidate wins over 28 GPT-OSS shapes |
|---|---:|---:|---:|---|
| bf16 | 2610 | 0 | 0 | Triton 27, `block_full_spt4_w1` 1 |
| fp16 | 2610 | 0 | 0 | Triton 27, `block_full_spt4_w1` 1 |
| fp32 | 2100 | 0 | 0 | Triton 25, `block_full_w8` 2, `wave_row_spt4` 1 |

The only bf16/fp16 Gluon win remained the largest residual case:

| dtype | residual | tokens | Triton p50 us | Best Gluon | Gluon p50 us | Gluon p90 us |
|---|---|---:|---:|---|---:|---:|
| bf16 | true | 8192 | 29.76 | `block_full_spt4_w1` | 29.20 | 29.65 |
| fp16 | true | 8192 | 29.86 | `block_full_spt4_w1` | 29.20 | 29.72 |

For fp32, Gluon wins appeared only at large token counts:

| residual | tokens | Triton p50 us | Best Gluon | Gluon p50 us | Gluon p90 us |
|---|---:|---:|---|---:|---:|
| false | 8192 | 29.74 | `block_full_w8` | 29.16 | 29.42 |
| true | 4096 | 29.56 | `wave_row_spt4` | 28.64 | 29.32 |
| true | 8192 | 70.00 | `block_full_w8` | 68.92 | 71.93 |

The larger fp16 SPT values helped within the Gluon candidate set, especially
for streaming decode configurations, but they did not move the Gluon kernels
past Triton for most GPT-OSS shapes.

## Tuned IR Notes

Representative tuned configs were compiled with:

```bash
rm -rf /tmp/tokenspeed_rmsnorm_tuned_ir_cache
HIP_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/tokenspeed_rmsnorm_tuned_ir_cache \
PYTHONPATH=tokenspeed-kernel/python \
../.venv/bin/python -m tokenspeed_kernel.benchmark.rmsnorm_tuning \
  --dtype bf16 --token-counts 1 \
  --candidates block_full_w8,wave_row_spt4,stream_c512_w4 \
  --warmup-iters 0 --bench-iters 1 --top-k 3 \
  --export ../claude_tmp/rmsnorm-tuned-ir-compile.json
```

| candidate | residual | cache hash | num_warps | max workgroup | VGPR | VGPR spills | notes |
|---|---|---|---:|---:|---:|---:|---|
| block_full_w8 | false | C6STE63EHVK5KZH5LOADNK27OYJK7S4P4TMOKNIJPKYCUGZQF32A | 8 | 512 | 20 | 0 | `tensor<4096xf32>`, one full-row reduction |
| block_full_w8 | true | EZVIAC4HWIMQCSOG4PZT2KP7ZFQ74OKDVGMYZ524UTHNCY6SX46A | 8 | 512 | 22 | 0 | `tensor<4096xf32>`, one full-row reduction |
| wave_row_spt4 | false | 3CZIZI72ROGHV23QPYD6AANXV5EFCPJNR4QWR2NI57F7SRNGCMHA | 1 | 64 | 112 | 0 | `sizePerThread=[4]`, one wave, one full-row reduction |
| wave_row_spt4 | true | B27MIDV745ZSX2Y3GOJH7O6MITO4IJKEFD6NKITI63MJQ5HKLE7Q | 1 | 64 | 122 | 0 | `sizePerThread=[4]`, one wave, one full-row reduction |
| stream_c512_w4 | false | RYXJVYWLSLZIVRKSEVEWWNH6AUAVEJFTH6CPFEK26ATRMU6KWCZQ | 4 | 256 | 29 | 0 | six `tensor<512xf32>` chunk reductions |
| stream_c512_w4 | true | SIE7FNAFL3NWLF2R3RVMVE3E3SA57KPE5MHCUZXZKL5FGMRB2EBQ | 4 | 256 | 31 | 0 | six `tensor<512xf32>` chunk reductions |

No tuned representative spilled SGPRs or VGPRs, and all use one CTA per row.

## Recommendation

Do not make a Gluon RMSNorm variant the GPT-OSS default yet. The existing
`triton_rmsnorm` baseline is faster for nearly every measured shape:

- bf16: in the expanded SPT sweep, Triton won 27/28 GPT-OSS shapes; best Gluon
  only won the `num_tokens=8192`, `residual=true` corner by about 2%.
- fp16: same 27/28 result; the same corner favored `block_full_spt4_w1` by
  about 2%.
- fp32: Gluon won 3 large-token shapes, but fp32 is not the main GPT-OSS
  inference dtype target here.

For continued Gluon work, keep the streaming family as the main iteration
target, including the larger SPT variants. It is the best Gluon strategy over
many decode and mid-size prefill shapes, keeps the chunked strategy visible in
IR, and has low VGPR usage with no spills. The block/full-row family is still
worth tracking for large-token residual cases. `wave_row_spt4` is mostly
compelling for large fp32 residual cases and carries much higher VGPR pressure.

Open performance questions:

- The tuned Gluon variants sit around a 20-23 us floor for most H=2880 shapes,
  while Triton is around 16-21 us. That looks like codegen/runtime overhead or
  less efficient reduction/memory scheduling rather than spill pressure.
- Streaming with `COL_BLOCK=512` reduces the live row footprint, but the extra
  pass and repeated chunk loop do not beat the full-row Triton baseline.
- None of the current Gluon variants attacks launch overhead or uses multi-CTA
  split-row work; both were intentionally out of scope for this bead.
