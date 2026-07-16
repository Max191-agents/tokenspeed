#!/usr/bin/env python3
"""Launch one reproducible DSA scoring cell for ROCm profiling."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Sequence
from pathlib import Path

import benchmark_dsa_scoring as benchmark


def _profile(args: argparse.Namespace) -> None:
    import torch

    if args.seq_len not in benchmark.SEQ_LENS:
        raise SystemExit(
            f"--seq-len must be one of {', '.join(map(str, benchmark.SEQ_LENS))}"
        )
    if args.warmups < 1 or args.launches < 1:
        raise SystemExit("--warmups and --launches must both be positive")

    visibility = benchmark._require_single_visible_gpu()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")

    fixture_dir = args.fixture_dir.resolve()
    manifest = benchmark._load_manifest(fixture_dir, verify=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    q, weights = benchmark._load_mode_inputs(
        args.backend, args.mode, fixture_dir, manifest, device
    )
    key_rows = (
        math.ceil(args.seq_len / benchmark.PAGE_SIZE) * benchmark.PAGE_SIZE
        if args.backend == "tokenspeed"
        else args.seq_len
    )
    key_fp8, key_scales = benchmark._load_key_inputs(
        fixture_dir,
        manifest,
        rows=key_rows,
        device=device,
    )

    with torch.inference_mode():
        if args.backend == "tokenspeed":
            packed_cache = benchmark._pack_tokenspeed_cache(key_fp8, key_scales)
            launch, logits, metadata = benchmark._tokenspeed_runners(
                mode=args.mode,
                seq_len=args.seq_len,
                q=q,
                weights=weights,
                packed_cache=packed_cache,
            )
            backend_source = None
        else:
            launch, logits, metadata, backend_source = benchmark._aiter_runner(
                mode=args.mode,
                seq_len=args.seq_len,
                q=q,
                weights=weights,
                key_fp8=key_fp8,
                key_scales=key_scales,
            )

        launch()
        torch.cuda.synchronize()
        for _ in range(args.warmups):
            launch()
        torch.cuda.synchronize()
        for _ in range(args.launches):
            launch()
        torch.cuda.synchronize()

    finite_count = int(torch.isfinite(logits).sum().item())
    payload = {
        "schema": "tokenspeed.dsa-scoring-profile.v1",
        "backend": benchmark.BACKEND_NAMES[args.backend],
        "mode": args.mode,
        "seq_len": args.seq_len,
        "fixture_id": manifest["fixture_id"],
        "visibility": visibility,
        "device": str(torch.cuda.get_device_name(torch.cuda.current_device())),
        "warmups": args.warmups,
        "profiled_launches": args.launches,
        "finite_output_count": finite_count,
        "output_shape": list(logits.shape),
        "launch": metadata,
        "backend_source": str(backend_source) if backend_source is not None else None,
    }
    if args.output is not None:
        benchmark._write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))

    del logits, key_fp8, key_scales, q, weights
    gc.collect()
    torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=tuple(benchmark.BACKEND_NAMES), required=True
    )
    parser.add_argument("--mode", choices=benchmark.MODES, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("benchmark-results/dsa_score_aiter/fixture"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(func=_profile)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
