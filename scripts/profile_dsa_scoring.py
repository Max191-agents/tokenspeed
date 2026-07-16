#!/usr/bin/env python3
"""Launch one reproducible DSA scoring cell for ROCm profiling."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import benchmark_dsa_scoring as benchmark

_TOKENSPEED_DECODE_MODULE = "tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_gfx950"
_TOKENSPEED_PREFILL_MODULE = (
    "tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_fp8_mfma_gfx950"
)
_AITER_KERNEL_MODULE = "aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits"
_PYTORCH_FILL_SOURCE = "torch.Tensor.fill_"
_PYTORCH_FILL_BLOCK_SIZE = 256
_PYTORCH_FILL_VECTOR_WIDTH = 4


def _backend_file_path(
    backend_files: Sequence[dict[str, str]], module_name: str
) -> str:
    for backend_file in backend_files:
        if backend_file["module"] == module_name:
            return backend_file["path"]
    raise ValueError(f"backend metadata is missing source module {module_name!r}")


def _selected_backend_source(
    *, backend: str, mode: str, backend_files: Sequence[dict[str, str]]
) -> str:
    module_name = (
        _TOKENSPEED_DECODE_MODULE
        if backend == "tokenspeed" and mode == "decode"
        else (
            _TOKENSPEED_PREFILL_MODULE
            if backend == "tokenspeed"
            else _AITER_KERNEL_MODULE
        )
    )
    return _backend_file_path(backend_files, module_name)


def _profile_dispatches(
    *,
    backend: str,
    mode: str,
    seq_len: int,
    launch: dict[str, Any],
    backend_files: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    if backend == "tokenspeed":
        if mode == "decode":
            source = _backend_file_path(backend_files, _TOKENSPEED_DECODE_MODULE)
            return [
                {
                    "stage": "decode_score",
                    "kernel_symbol": "_dsa_decode_logits_fp8_kernel",
                    "source": source,
                    "grid": list(launch["grid"]),
                    "dispatches_per_launch": 1,
                }
            ]

        source = _backend_file_path(backend_files, _TOKENSPEED_PREFILL_MODULE)
        score_grid = list(launch["grid"])
        if seq_len <= 2048:
            return [
                {
                    "stage": "fused_query_decomposition_and_packed_mfma_score",
                    "kernel_symbol": (
                        "_dsa_prefill_logits_fp8_tiled_fused_range_safe_kernel"
                    ),
                    "source": source,
                    "grid": score_grid,
                    "dispatches_per_launch": 1,
                }
            ]
        return [
            {
                "stage": "query_preprocess",
                "kernel_symbol": "_dsa_preprocess_prefill_query_fp8_kernel",
                "source": source,
                "grid": [benchmark.PREFILL_ROWS],
                "dispatches_per_launch": 1,
            },
            {
                "stage": "packed_mfma_score",
                "kernel_symbol": (
                    "_dsa_prefill_logits_fp8_tiled_multi_component_kernel"
                ),
                "source": source,
                "grid": score_grid,
                "dispatches_per_launch": 1,
            },
        ]

    source = _backend_file_path(backend_files, _AITER_KERNEL_MODULE)
    rows = 1 if mode == "decode" else benchmark.PREFILL_ROWS
    fill_elements = rows * int(launch["allocated_seq_len"])
    fill_blocks = math.ceil(
        fill_elements / (_PYTORCH_FILL_BLOCK_SIZE * _PYTORCH_FILL_VECTOR_WIDTH)
    )
    return [
        {
            "stage": "preallocated_neginf_fill",
            "kernel_symbol": "FillFunctor<float>",
            "source": _PYTORCH_FILL_SOURCE,
            "grid": [fill_blocks],
            "dispatches_per_launch": 1,
        },
        {
            "stage": "contiguous_fp8_mqa_score",
            "kernel_symbol": "_gluon_fp8_mqa_logits_kernel",
            "source": source,
            "grid": list(launch["grid"]),
            "dispatches_per_launch": 1,
        },
    ]


def _profile_counts(
    *,
    dispatches: Sequence[dict[str, Any]],
    warmups: int,
    profiled_launches: int,
) -> dict[str, int]:
    priming_launches = 1
    dispatches_per_launch = sum(
        int(dispatch["dispatches_per_launch"]) for dispatch in dispatches
    )
    total_callable_invocations = priming_launches + warmups + profiled_launches
    return {
        "priming_launches": priming_launches,
        "warmups": warmups,
        "profiled_launches": profiled_launches,
        "total_callable_invocations": total_callable_invocations,
        "dispatches_per_launch": dispatches_per_launch,
        "profiled_target_dispatches": profiled_launches * dispatches_per_launch,
        "total_target_dispatches": (total_callable_invocations * dispatches_per_launch),
    }


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
        else:
            launch, logits, metadata, _ = benchmark._aiter_runner(
                mode=args.mode,
                seq_len=args.seq_len,
                q=q,
                weights=weights,
                key_fp8=key_fp8,
                key_scales=key_scales,
            )

        backend_files = benchmark._backend_files(args.backend)
        dispatches = _profile_dispatches(
            backend=args.backend,
            mode=args.mode,
            seq_len=args.seq_len,
            launch=metadata,
            backend_files=backend_files,
        )
        backend_source = _selected_backend_source(
            backend=args.backend,
            mode=args.mode,
            backend_files=backend_files,
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
        **_profile_counts(
            dispatches=dispatches,
            warmups=args.warmups,
            profiled_launches=args.launches,
        ),
        "finite_output_count": finite_count,
        "output_shape": list(logits.shape),
        "launch": metadata,
        "dispatches": dispatches,
        "backend_source": backend_source,
        "backend_files": backend_files,
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
