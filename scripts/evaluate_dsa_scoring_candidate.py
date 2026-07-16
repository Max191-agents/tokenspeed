#!/usr/bin/env python3
"""Compare production and experimental MFMA TokenSpeed DSA scorers.

This evaluator reuses the neutral fixture and production runner from
``benchmark_dsa_scoring.py``. Both implementations write into preallocated
FP32 logits, use identical page/workspace metadata, and are timed in an
alternating order with HIP events after compilation and warmup.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import socket
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import benchmark_dsa_scoring as benchmark

REPORT_SCHEMA = "tokenspeed.dsa-scoring-candidate-evaluation.v1"
CANDIDATE_MODULE = "tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_mfma_gfx950"
BASELINE_NAME = "production_elementwise_gluon"
CANDIDATE_NAME = "experimental_persistent_bf16_mfma_gluon"


def _candidate_runner(
    *,
    mode: str,
    seq_len: int,
    q: Any,
    weights: Any,
    packed_cache: Any,
) -> tuple[Callable[[], None], Any, dict[str, Any]]:
    import torch

    module = importlib.import_module(CANDIDATE_MODULE)
    if mode == "decode":
        max_seq_len = math.ceil(seq_len / benchmark.PAGE_SIZE) * benchmark.PAGE_SIZE
        num_pages = max_seq_len // benchmark.PAGE_SIZE
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=q.device)
        block_table = torch.arange(
            num_pages, dtype=torch.int32, device=q.device
        ).reshape(1, num_pages)
        logits = torch.empty((1, max_seq_len), dtype=torch.float32, device=q.device)

        def launch() -> None:
            module.launch_dsa_decode_logits_mfma_gfx950(
                q,
                packed_cache,
                weights,
                seq_lens,
                block_table,
                logits,
                softmax_scale=benchmark.SOFTMAX_SCALE,
                q_len_per_req=1,
            )

        metadata = {
            "grid": [1],
            "block_n": 32,
            "num_warps": 1,
            "page_table": "identity_pages",
            "logical_seq_len": seq_len,
            "allocated_seq_len": max_seq_len,
        }
        return launch, logits, metadata

    workspace_slots = torch.arange(seq_len, dtype=torch.int64, device=q.device)
    row_starts = torch.zeros(benchmark.PREFILL_ROWS, dtype=torch.int32, device=q.device)
    row_ends = torch.arange(
        seq_len - benchmark.PREFILL_ROWS + 1,
        seq_len + 1,
        dtype=torch.int32,
        device=q.device,
    )
    logits = torch.empty(
        (benchmark.PREFILL_ROWS, seq_len), dtype=torch.float32, device=q.device
    )

    def launch() -> None:
        module.launch_dsa_prefill_logits_mfma_gfx950(
            q,
            packed_cache,
            weights,
            workspace_slots,
            row_starts,
            row_ends,
            logits,
            softmax_scale=benchmark.SOFTMAX_SCALE,
        )

    metadata = {
        "grid": [benchmark.PREFILL_ROWS],
        "block_n": 32,
        "num_warps": 1,
        "workspace_slots": "identity_slots",
        "row_starts": "all_zero",
        "row_ends": "seq_len-63_through_seq_len",
    }
    return launch, logits, metadata


def _compare_outputs(
    baseline: Any,
    candidate: Any,
    *,
    mode: str,
    seq_len: int,
    chunk_cols: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    import torch

    if baseline.shape != candidate.shape:
        return {
            "passed": False,
            "shape_equal": False,
            "baseline_shape": list(baseline.shape),
            "candidate_shape": list(candidate.shape),
        }

    if mode == "decode":
        row_ends = torch.tensor([seq_len], dtype=torch.int64, device=baseline.device)
    else:
        row_ends = torch.arange(
            seq_len - benchmark.PREFILL_ROWS + 1,
            seq_len + 1,
            dtype=torch.int64,
            device=baseline.device,
        )

    baseline_mask_mismatches = 0
    candidate_mask_mismatches = 0
    inter_impl_mask_mismatches = 0
    value_mismatches = 0
    max_abs_error = 0.0
    max_rel_error = 0.0
    compared_values = 0
    for start in range(0, baseline.shape[1], chunk_cols):
        end = min(start + chunk_cols, baseline.shape[1])
        positions = torch.arange(start, end, dtype=torch.int64, device=baseline.device)
        expected_valid = positions[None, :] < row_ends[:, None]
        expected_neg_inf = ~expected_valid
        baseline_chunk = baseline[:, start:end]
        candidate_chunk = candidate[:, start:end]
        baseline_bad_mask = (torch.isfinite(baseline_chunk) != expected_valid) | (
            torch.isneginf(baseline_chunk) != expected_neg_inf
        )
        candidate_bad_mask = (torch.isfinite(candidate_chunk) != expected_valid) | (
            torch.isneginf(candidate_chunk) != expected_neg_inf
        )
        baseline_mask_mismatches += int(baseline_bad_mask.sum().item())
        candidate_mask_mismatches += int(candidate_bad_mask.sum().item())
        inter_impl_mask_mismatches += int(
            (torch.isneginf(baseline_chunk) != torch.isneginf(candidate_chunk))
            .sum()
            .item()
        )

        baseline_values = baseline_chunk[expected_valid]
        candidate_values = candidate_chunk[expected_valid]
        compared_values += baseline_values.numel()
        close = torch.isclose(candidate_values, baseline_values, rtol=rtol, atol=atol)
        value_mismatches += int((~close).sum().item())
        finite = torch.isfinite(baseline_values) & torch.isfinite(candidate_values)
        if bool(finite.any()):
            absolute_error = (candidate_values[finite] - baseline_values[finite]).abs()
            relative_error = absolute_error / baseline_values[finite].abs().clamp_min(
                1.0e-12
            )
            max_abs_error = max(max_abs_error, float(absolute_error.max().item()))
            max_rel_error = max(max_rel_error, float(relative_error.max().item()))

    passed = (
        baseline_mask_mismatches == 0
        and candidate_mask_mismatches == 0
        and inter_impl_mask_mismatches == 0
        and value_mismatches == 0
    )
    return {
        "passed": passed,
        "shape_equal": True,
        "output_shape": list(baseline.shape),
        "compared_finite_values": compared_values,
        "baseline_expected_mask_mismatches": baseline_mask_mismatches,
        "candidate_expected_mask_mismatches": candidate_mask_mismatches,
        "inter_implementation_mask_mismatches": inter_impl_mask_mismatches,
        "value_mismatches": value_mismatches,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "rtol": rtol,
        "atol": atol,
    }


def _measure_pair(
    launches: dict[str, Callable[[], None]], *, warmups: int, samples: int
) -> tuple[dict[str, list[float]], float]:
    import torch

    names = (BASELINE_NAME, CANDIDATE_NAME)
    for name in names:
        launches[name]()
    torch.cuda.synchronize()
    for index in range(warmups):
        order = names if index % 2 == 0 else tuple(reversed(names))
        for name in order:
            launches[name]()
    torch.cuda.synchronize()

    elapsed = {name: [] for name in names}
    wall_start = time.monotonic()
    for index in range(samples):
        order = names if index % 2 == 0 else tuple(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launches[name]()
            end.record()
            end.synchronize()
            elapsed[name].append(float(start.elapsed_time(end)))
    return elapsed, time.monotonic() - wall_start


def _timing_result(samples_ms: Sequence[float]) -> dict[str, Any]:
    return {
        "sample_count": len(samples_ms),
        "samples_ms": list(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "p95_ms": benchmark._percentile(samples_ms, 0.95),
    }


def _evaluate_cell(
    *,
    mode: str,
    seq_len: int,
    q: Any,
    weights: Any,
    packed_cache: Any,
    warmups: int,
    samples: int,
    chunk_cols: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    import torch

    baseline_launch, baseline_logits, baseline_metadata = benchmark._tokenspeed_runners(
        mode=mode,
        seq_len=seq_len,
        q=q,
        weights=weights,
        packed_cache=packed_cache,
    )
    candidate_launch, candidate_logits, candidate_metadata = _candidate_runner(
        mode=mode,
        seq_len=seq_len,
        q=q,
        weights=weights,
        packed_cache=packed_cache,
    )
    baseline_launch()
    candidate_launch()
    torch.cuda.synchronize()
    correctness = _compare_outputs(
        baseline_logits,
        candidate_logits,
        mode=mode,
        seq_len=seq_len,
        chunk_cols=chunk_cols,
        rtol=rtol,
        atol=atol,
    )
    samples_by_impl, wall_seconds = _measure_pair(
        {
            BASELINE_NAME: baseline_launch,
            CANDIDATE_NAME: candidate_launch,
        },
        warmups=warmups,
        samples=samples,
    )
    baseline_timing = _timing_result(samples_by_impl[BASELINE_NAME])
    candidate_timing = _timing_result(samples_by_impl[CANDIDATE_NAME])
    speedup = baseline_timing["median_ms"] / candidate_timing["median_ms"]
    return {
        "cell_id": f"{mode}:{seq_len}",
        "mode": mode,
        "seq_len": seq_len,
        "q_shape": list(q.shape),
        "weights_shape": list(weights.shape),
        "correctness": correctness,
        "timing": {
            BASELINE_NAME: baseline_timing,
            CANDIDATE_NAME: candidate_timing,
            "measurement_wall_seconds": wall_seconds,
            "baseline_over_candidate": speedup,
            "candidate_is_faster": speedup > 1.0,
        },
        "launch": {
            BASELINE_NAME: baseline_metadata,
            CANDIDATE_NAME: candidate_metadata,
        },
    }


def _source_metadata() -> dict[str, Any]:
    module = importlib.import_module(CANDIDATE_MODULE)
    candidate_path = Path(module.__file__).resolve()
    return {
        BASELINE_NAME: benchmark._backend_files("tokenspeed"),
        CANDIDATE_NAME: [
            {
                "module": CANDIDATE_MODULE,
                "path": str(candidate_path),
                "sha256": benchmark._sha256_file(candidate_path),
            }
        ],
    }


def _run(args: argparse.Namespace) -> None:
    import torch

    if args.samples < benchmark.MIN_SAMPLES:
        raise SystemExit(
            f"candidate evaluation requires at least {benchmark.MIN_SAMPLES} "
            f"samples, got {args.samples}"
        )
    if args.warmups < 1:
        raise SystemExit("candidate evaluation requires at least one warmup")
    if args.compare_chunk_cols < 1:
        raise SystemExit("--compare-chunk-cols must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise SystemExit("--rtol and --atol must be nonnegative")

    visibility = benchmark._require_single_visible_gpu()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    torch.cuda.set_device(device)
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    arch = getattr(properties, "gcnArchName", "")
    if "gfx950" not in arch:
        raise SystemExit(f"GFX950 is required, got {arch or properties.name}")

    fixture_dir = args.fixture_dir.resolve()
    manifest = benchmark._load_manifest(fixture_dir, verify=True)
    cells = args.cells
    mode_inputs = {
        mode: benchmark._load_mode_inputs(
            "tokenspeed", mode, fixture_dir, manifest, device
        )
        for mode in benchmark.MODES
        if any(cell_mode == mode for cell_mode, _ in cells)
    }

    started_at_utc = benchmark._utc_now()
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for seq_len in benchmark.SEQ_LENS:
            requested_modes = [
                mode for mode in benchmark.MODES if (mode, seq_len) in cells
            ]
            if not requested_modes:
                continue
            key_rows = math.ceil(seq_len / benchmark.PAGE_SIZE) * benchmark.PAGE_SIZE
            key_fp8, key_scales = benchmark._load_key_inputs(
                fixture_dir,
                manifest,
                rows=key_rows,
                device=device,
            )
            packed_cache = benchmark._pack_tokenspeed_cache(key_fp8, key_scales)
            for mode in requested_modes:
                q, weights = mode_inputs[mode]
                try:
                    result = _evaluate_cell(
                        mode=mode,
                        seq_len=seq_len,
                        q=q,
                        weights=weights,
                        packed_cache=packed_cache,
                        warmups=args.warmups,
                        samples=args.samples,
                        chunk_cols=args.compare_chunk_cols,
                        rtol=args.rtol,
                        atol=args.atol,
                    )
                except Exception as error:  # Preserve results for the other cells.
                    result = {
                        "cell_id": f"{mode}:{seq_len}",
                        "mode": mode,
                        "seq_len": seq_len,
                        "correctness": {"passed": False},
                        "error": f"{type(error).__name__}: {error}",
                    }
                results.append(result)
                status = "PASS" if result["correctness"]["passed"] else "FAIL"
                print(f"{status} {result['cell_id']}", file=sys.stderr, flush=True)
            del packed_cache, key_fp8, key_scales
            gc.collect()
            torch.cuda.empty_cache()

    all_correct = len(results) == len(cells) and all(
        result["correctness"]["passed"] for result in results
    )
    speedups = [
        result["timing"]["baseline_over_candidate"]
        for result in results
        if "timing" in result
    ]
    payload = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "started_at_utc": started_at_utc,
        "finished_at_utc": benchmark._utc_now(),
        "fixture": {
            "fixture_id": manifest["fixture_id"],
            "manifest_sha256": benchmark._sha256_file(fixture_dir / "manifest.json"),
        },
        "evaluation": {
            "cells": [f"{mode}:{seq_len}" for mode, seq_len in cells],
            "warmups_per_implementation": args.warmups,
            "samples_per_implementation": args.samples,
            "event_timing": "torch.cuda.Event",
            "sample_order": "alternating_baseline_candidate",
            "compile_in_timing": False,
            "allocation_in_timing": False,
            "topk_in_timing": False,
            "compare_chunk_cols": args.compare_chunk_cols,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "summary": {
            "all_correct": all_correct,
            "passed_cells": sum(result["correctness"]["passed"] for result in results),
            "cell_count": len(results),
            "minimum_baseline_over_candidate": min(speedups) if speedups else None,
            "all_candidates_faster": bool(speedups)
            and all(speedup > 1.0 for speedup in speedups),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "hostname": socket.gethostname(),
            "device_name": properties.name,
            "device_arch": arch,
            "device_index_in_visible_set": device_index,
            "gpu_visibility": visibility,
            "git": benchmark._git_metadata(Path(__file__).resolve()),
            "sources": _source_metadata(),
            "harness_sha256": benchmark._sha256_file(Path(__file__).resolve()),
        },
        "results": results,
    }
    benchmark._write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_correct:
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("benchmark-results/dsa_score_aiter/fixture"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/dsa_score_aiter/candidate-evaluation.json"),
    )
    parser.add_argument(
        "--cells",
        type=benchmark._parse_cells,
        default=benchmark._parse_cells(None),
        help="comma-separated MODE:SEQ_LEN cells; defaults to the full grid",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=benchmark.MIN_SAMPLES)
    parser.add_argument("--compare-chunk-cols", type=int, default=8192)
    parser.add_argument("--rtol", type=float, default=3.0e-3)
    parser.add_argument("--atol", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _run(args)


if __name__ == "__main__":
    main()
