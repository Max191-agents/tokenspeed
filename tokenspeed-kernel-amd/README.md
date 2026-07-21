# AMD-specific High Performance Kernels

This directory contains high-performance kernel implementations for AMD GPUs.

## DSA top-k workload benchmark

`benchmark/bench_dsa_topk_gfx950.py` compares the Gluon DSA top-k selector
with AITER on gfx950. The manifest models GLM agentic decode batching, MTP
query rows, prefix-cache replay, continuous prefill batching, and the selector
launches imposed by the default 512 MiB logits workspace cap. All cases use
`k=2048`.

The primary AITER comparison includes the API work fused into Gluon: physical
slot mapping and valid counts for decode, and valid counts for prefill. Timing
uses GPU events around graph replay, so it excludes Python and host launch
overhead. Prefill results aggregate all selector launches for one scheduler
forward before computing suite geomeans.

Run from the repository root with the TokenSpeed development environment and
an AITER checkout on `PYTHONPATH`:

```bash
PYTHONPATH=tokenspeed-kernel-amd/python:tokenspeed-kernel/python:<aiter-checkout> \
  ../.venv/bin/python \
  tokenspeed-kernel-amd/benchmark/bench_dsa_topk_gfx950.py list

HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=tokenspeed-kernel-amd/python:tokenspeed-kernel/python:<aiter-checkout> \
  ../.venv/bin/python \
  tokenspeed-kernel-amd/benchmark/bench_dsa_topk_gfx950.py run \
  --json-out /tmp/dsa-topk-workloads.json

PYTHONPATH=tokenspeed-kernel-amd/python:tokenspeed-kernel/python \
  ../.venv/bin/python \
  tokenspeed-kernel-amd/benchmark/bench_dsa_topk_gfx950.py report \
  /tmp/dsa-topk-workloads.json
```
