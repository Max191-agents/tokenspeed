#!/usr/bin/env python3
"""Benchmark structurally matched TokenSpeed and AITER DSA scoring cores.

This is deliberately not a production-layout comparison. Both implementations
consume identical contiguous FP8 query/key bytes, FP32 key scales, signed FP32
weights, and preallocated contiguous FP32 logits. Packed-cache translation,
BF16-to-FP8 conversion, production row masking, allocation, and TopK are outside
the measured scope.

Run each backend in its own environment in this serialized order, then combine
the six JSON files: TokenSpeed round 1, AITER round 1, TokenSpeed round 2,
AITER round 2, TokenSpeed round 3, AITER round 3. The combine command rejects
non-alternating or overlapping runs.
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
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import benchmark_dsa_scoring as shared

SCHEMA_VERSION = 1
ROUND_SCHEMA = "tokenspeed.dsa-scoring-matched-core-round.v1"
COMPARISON_SCHEMA = "tokenspeed.dsa-scoring-matched-core-comparison.v1"
COMPARISON_KIND = "matched_layout_computational_core"

TOKENSPEED_MODULE = (
    "tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_fp8_mfma_gfx950"
)
TOKENSPEED_HOOK = "launch_dsa_matched_core_fp8_gfx950"
AITER_WRAPPER_MODULE = "aiter.ops.triton.attention.fp8_mqa_logits"
AITER_KERNEL_MODULE = "aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits"

BACKEND_NAMES = {
    "tokenspeed": "tokenspeed_matched_contiguous_fp8_core",
    "aiter": "aiter_matched_contiguous_fp8_core",
}


def _module_file(module_name: str) -> dict[str, str]:
    module = importlib.import_module(module_name)
    path = Path(module.__file__).resolve()
    return {
        "module": module_name,
        "path": str(path),
        "sha256": shared._sha256_file(path),
    }


def _backend_files(backend: str) -> list[dict[str, str]]:
    modules = (
        (TOKENSPEED_MODULE,)
        if backend == "tokenspeed"
        else (AITER_WRAPPER_MODULE, AITER_KERNEL_MODULE)
    )
    return [_module_file(module_name) for module_name in modules]


def _load_mode_inputs(
    mode: str,
    fixture_dir: Path,
    manifest: dict[str, Any],
    device: Any,
) -> tuple[Any, Any]:
    import torch

    q_bits = shared._array_to_device(
        fixture_dir,
        manifest,
        f"{mode}_q_e4m3fn_bits",
        device=device,
    )
    q = q_bits.view(torch.float8_e4m3fn).contiguous()
    weights = shared._array_to_device(
        fixture_dir,
        manifest,
        f"{mode}_weights_f32",
        device=device,
    ).contiguous()
    return q, weights


def _validate_matched_inputs(
    q: Any,
    key_fp8: Any,
    key_scales: Any,
    weights: Any,
    seq_len: int,
) -> None:
    import torch

    expected_rows = q.shape[0]
    if q.dtype != torch.float8_e4m3fn or q.shape[1:] != (
        shared.NUM_HEADS,
        shared.HEAD_DIM,
    ):
        raise RuntimeError(
            "matched-core q must be float8_e4m3fn [rows, 32, 128], got "
            f"dtype={q.dtype}, shape={tuple(q.shape)}"
        )
    if key_fp8.dtype != torch.float8_e4m3fn or key_fp8.shape != (
        seq_len,
        shared.HEAD_DIM,
    ):
        raise RuntimeError(
            "matched-core K must be float8_e4m3fn [seq_len, 128], got "
            f"dtype={key_fp8.dtype}, shape={tuple(key_fp8.shape)}"
        )
    if key_scales.dtype != torch.float32 or key_scales.shape != (seq_len,):
        raise RuntimeError(
            "matched-core scales must be FP32 [seq_len], got "
            f"dtype={key_scales.dtype}, shape={tuple(key_scales.shape)}"
        )
    if weights.dtype != torch.float32 or weights.shape != (
        expected_rows,
        shared.NUM_HEADS,
    ):
        raise RuntimeError(
            "matched-core weights must be signed FP32 [rows, 32], got "
            f"dtype={weights.dtype}, shape={tuple(weights.shape)}"
        )
    tensors = (q, key_fp8, key_scales, weights)
    if any(tensor.device != q.device for tensor in tensors):
        raise RuntimeError("all matched-core inputs must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("all matched-core inputs must be contiguous")


def _tokenspeed_runner(
    *,
    q: Any,
    key_fp8: Any,
    key_scales: Any,
    weights: Any,
    seq_len: int,
) -> tuple[Callable[[], None], Any, dict[str, Any], Path]:
    import torch

    try:
        module = importlib.import_module(TOKENSPEED_MODULE)
    except ModuleNotFoundError as error:
        raise SystemExit(
            "TokenSpeed matched-core benchmark requires module "
            f"{TOKENSPEED_MODULE!r}; the production FP8 MFMA implementation "
            "has not provided the benchmark hook"
        ) from error
    hook = getattr(module, TOKENSPEED_HOOK, None)
    if not callable(hook):
        raise SystemExit(
            "TokenSpeed matched-core benchmark requires callable "
            f"{TOKENSPEED_MODULE}.{TOKENSPEED_HOOK}"
        )

    logits = torch.empty((q.shape[0], seq_len), dtype=torch.float32, device=q.device)

    def launch() -> None:
        hook(
            q,
            key_fp8,
            key_scales,
            weights,
            logits,
            seq_len=seq_len,
            softmax_scale=1.0,
        )

    metadata = {
        "entrypoint": f"{TOKENSPEED_MODULE}.{TOKENSPEED_HOOK}",
        "grid_policy": "implementation_defined",
        "query_layout": "contiguous_float8_e4m3fn_rows_heads_dim",
        "key_layout": "contiguous_float8_e4m3fn_rows_dim",
        "key_scale_layout": "contiguous_float32_per_key_row",
        "output_layout": "contiguous_float32_rows_by_seq_len",
        "logical_seq_len": seq_len,
        "softmax_scale": 1.0,
    }
    return launch, logits, metadata, Path(module.__file__).resolve()


def _aiter_runner(
    *,
    q: Any,
    key_fp8: Any,
    key_scales: Any,
    weights: Any,
    seq_len: int,
) -> tuple[Callable[[], None], Any, dict[str, Any], Path]:
    import torch

    module = importlib.import_module(AITER_WRAPPER_MODULE)
    if module.arch != "gfx950":
        raise SystemExit(f"AITER matched core requires gfx950, got {module.arch}")
    kernel = module._gluon_fp8_mqa_logits_kernel
    if kernel is None:
        raise SystemExit("AITER gfx950 Gluon FP8 MQA kernel is unavailable")

    rows = q.shape[0]
    starts = torch.zeros(rows, dtype=torch.int32, device=q.device)
    ends = torch.full((rows,), seq_len, dtype=torch.int32, device=q.device)
    logits = torch.empty((rows, seq_len), dtype=torch.float32, device=q.device)
    use_folded_reduction = module.FOLDED_REDUCTED_SUPPORT and shared.NUM_HEADS > 16
    num_chains = 4 if use_folded_reduction else 0
    use_buffer_load = key_fp8.numel() * key_fp8.element_size() < 2 * 1024**3
    use_buffer_store = logits.numel() * logits.element_size() < 2 * 1024**3
    use_padded_shared = module.ASYNC_COPY_SUPPORTS_DISTRIBUTED

    def launch() -> None:
        kernel[(rows,)](
            Q_ptr=q,
            KV_ptr=key_fp8,
            kv_scales_ptr=key_scales,
            weights_ptr=weights,
            cu_start_ptr=starts,
            cu_end_ptr=ends,
            logits_ptr=logits,
            seq_len=rows,
            seq_len_kv=seq_len,
            NUM_HEADS=shared.NUM_HEADS,
            HEAD_SIZE=shared.HEAD_DIM,
            stride_q_s=q.stride(0),
            stride_q_h=q.stride(1),
            stride_q_d=q.stride(2),
            stride_kv_s=key_fp8.stride(0),
            stride_kv_d=key_fp8.stride(1),
            stride_w_s=weights.stride(0),
            stride_w_h=weights.stride(1),
            stride_logits_s=logits.stride(0),
            stride_logits_k=logits.stride(1),
            BLOCK_KV=32,
            NUM_WARPS=1,
            NUM_BUFFERS=2,
            NUM_CHAINS=num_chains,
            USE_BUFFER_LOAD=use_buffer_load,
            USE_BUFFER_STORE=use_buffer_store,
            USE_PADDED_SHARED_LAYOUT=use_padded_shared,
            num_warps=1,
            waves_per_eu=3,
        )

    metadata = {
        "entrypoint": f"{AITER_WRAPPER_MODULE}._gluon_fp8_mqa_logits_kernel",
        "grid": [rows],
        "block_kv": 32,
        "num_warps": 1,
        "waves_per_eu": 3,
        "num_buffers": 2,
        "num_chains": num_chains,
        "folded_reduction": use_folded_reduction,
        "padded_shared_layout": use_padded_shared,
        "query_layout": "contiguous_float8_e4m3fn_rows_heads_dim",
        "key_layout": "contiguous_float8_e4m3fn_rows_dim",
        "key_scale_layout": "contiguous_float32_per_key_row",
        "output_layout": "contiguous_float32_rows_by_seq_len",
        "logical_seq_len": seq_len,
        "softmax_scale": 1.0,
    }
    return launch, logits, metadata, Path(module.__file__).resolve()


def _slice_identity(
    manifest: dict[str, Any], logical_name: str, rows: int | None = None
) -> dict[str, Any]:
    metadata = manifest["arrays"][logical_name]
    slice_spec = "all" if rows is None else f"rows[0:{rows}]"
    identity = {
        "fixture_id": manifest["fixture_id"],
        "logical_name": logical_name,
        "source_sha256": metadata["sha256"],
        "slice": slice_spec,
    }
    return {**identity, "slice_id": shared._canonical_hash(identity)}


def _cell_input_identity(
    manifest: dict[str, Any], mode: str, seq_len: int
) -> dict[str, Any]:
    identity = {
        "query": _slice_identity(manifest, f"{mode}_q_e4m3fn_bits"),
        "key": _slice_identity(manifest, "key_e4m3fn_bits", seq_len),
        "key_scale": _slice_identity(manifest, "key_scale_f32", seq_len),
        "weights": _slice_identity(manifest, f"{mode}_weights_f32"),
    }
    return {**identity, "combined_id": shared._canonical_hash(identity)}


def _probe_indices(limit: int) -> list[int]:
    candidates = {
        0,
        1,
        15,
        16,
        31,
        32,
        63,
        64,
        limit // 4,
        limit // 2,
        (3 * limit) // 4,
        limit - 2,
        limit - 1,
    }
    return sorted(index for index in candidates if 0 <= index < limit)


def _collect_probe(logits: Any) -> dict[str, Any]:
    import torch

    row_indices = _probe_indices(logits.shape[0])
    column_indices = _probe_indices(logits.shape[1])
    rows = torch.tensor(row_indices, dtype=torch.int64, device=logits.device)
    columns = torch.tensor(column_indices, dtype=torch.int64, device=logits.device)
    values = logits.index_select(0, rows).index_select(1, columns).float().cpu()
    return {
        "purpose": "untimed_cross_implementation_sanity_probe_not_full_correctness",
        "row_indices": row_indices,
        "column_indices": column_indices,
        "values": values.tolist(),
    }


def _cell_result(
    *,
    backend: str,
    mode: str,
    seq_len: int,
    samples_ms: Sequence[float],
    wall_seconds: float,
    output_shape: Sequence[int],
    launch: dict[str, Any],
    input_identity: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    rows = 1 if mode == "decode" else shared.PREFILL_ROWS
    return {
        "cell_id": f"{mode}:{seq_len}",
        "backend": BACKEND_NAMES[backend],
        "mode": mode,
        "seq_len": seq_len,
        "q_shape": [rows, shared.NUM_HEADS, shared.HEAD_DIM],
        "key_shape": [seq_len, shared.HEAD_DIM],
        "weights_shape": [rows, shared.NUM_HEADS],
        "output_shape": list(output_shape),
        "sample_count": len(samples_ms),
        "samples_ms": list(samples_ms),
        "round_median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "p95_ms": shared._percentile(samples_ms, 0.95),
        "measurement_wall_seconds": wall_seconds,
        "input_identity": input_identity,
        "launch": launch,
        "numerical_probe": probe,
    }


def _run_backend(args: argparse.Namespace) -> None:
    import torch

    if args.samples < shared.MIN_SAMPLES:
        raise SystemExit(
            f"matched-core benchmark requires at least {shared.MIN_SAMPLES} "
            f"samples, got {args.samples}"
        )
    if args.warmups < 1:
        raise SystemExit("matched-core benchmark requires at least one warmup")
    visibility = shared._require_single_visible_gpu()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")

    cells = shared._parse_cells(args.cells)
    fixture_dir = args.fixture_dir.resolve()
    manifest = shared._load_manifest(fixture_dir, verify=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    started_at_utc = shared._utc_now()
    started_at_ns = time.time_ns()
    results: list[dict[str, Any]] = []
    backend_source: Path | None = None

    mode_inputs = {
        mode: _load_mode_inputs(mode, fixture_dir, manifest, device)
        for mode in shared.MODES
        if any(cell_mode == mode for cell_mode, _ in cells)
    }

    with torch.inference_mode():
        for seq_len in shared.SEQ_LENS:
            requested_modes = [
                mode for mode in shared.MODES if (mode, seq_len) in cells
            ]
            if not requested_modes:
                continue
            key_fp8, key_scales = shared._load_key_inputs(
                fixture_dir,
                manifest,
                rows=seq_len,
                device=device,
            )
            for mode in requested_modes:
                q, weights = mode_inputs[mode]
                _validate_matched_inputs(q, key_fp8, key_scales, weights, seq_len)
                runner = (
                    _tokenspeed_runner
                    if args.backend == "tokenspeed"
                    else _aiter_runner
                )
                launch, logits, launch_metadata, backend_source = runner(
                    q=q,
                    key_fp8=key_fp8,
                    key_scales=key_scales,
                    weights=weights,
                    seq_len=seq_len,
                )
                samples_ms, wall_seconds = shared._measure(
                    launch, warmups=args.warmups, samples=args.samples
                )
                probe = _collect_probe(logits)
                results.append(
                    _cell_result(
                        backend=args.backend,
                        mode=mode,
                        seq_len=seq_len,
                        samples_ms=samples_ms,
                        wall_seconds=wall_seconds,
                        output_shape=logits.shape,
                        launch=launch_metadata,
                        input_identity=_cell_input_identity(manifest, mode, seq_len),
                        probe=probe,
                    )
                )
                del launch, logits
            del key_fp8, key_scales
            gc.collect()
            torch.cuda.empty_cache()

    finished_at_ns = time.time_ns()
    if backend_source is None:
        raise RuntimeError("no matched-core cells were executed")
    device_uuid = getattr(properties, "uuid", None)
    pci_bus_id = getattr(properties, "pci_bus_id", None)
    payload = {
        "schema": ROUND_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "comparison_kind": COMPARISON_KIND,
        "production_comparison": False,
        "backend": BACKEND_NAMES[args.backend],
        "round": args.round,
        "started_at_utc": started_at_utc,
        "finished_at_utc": shared._utc_now(),
        "started_at_ns": started_at_ns,
        "finished_at_ns": finished_at_ns,
        "fixture": {
            "fixture_id": manifest["fixture_id"],
            "manifest_sha256": shared._sha256_file(fixture_dir / "manifest.json"),
        },
        "benchmark": {
            "warmups": args.warmups,
            "samples_per_cell": args.samples,
            "compile_launches_per_cell": 1,
            "compile_in_timing": False,
            "allocation_in_timing": False,
            "output_initialization_in_timing": False,
            "topk_in_timing": False,
            "event_timing": "torch.cuda.Event",
            "softmax_scale": 1.0,
            "cells": [f"{mode}:{seq_len}" for mode, seq_len in cells],
        },
        "timing_scope": {
            "classification": COMPARISON_KIND,
            "warning": (
                "This isolates a matched contiguous FP8 scoring core and is "
                "not the complete TokenSpeed production scoring comparison."
            ),
            "identical_between_implementations": [
                "float8_e4m3fn query bit patterns",
                "contiguous float8_e4m3fn key bit patterns",
                "contiguous per-row FP32 key scales",
                "signed FP32 head weights",
                "contiguous preallocated FP32 output layout",
                "full rectangular rows_by_seq_len scoring work",
            ],
            "included": [
                "query and key loads",
                "per-row key scaling",
                "query-key dot products",
                "per-head ReLU",
                "signed head weighting and reduction",
                "FP32 output stores",
            ],
            "excluded": [
                "BF16 query to FP8 conversion",
                "packed-cache construction",
                "page-table lookup",
                "workspace-slot lookup",
                "causal or ragged row masking",
                "softmax scaling",
                "output allocation or initialization",
                "TopK",
            ],
            "prefill_policy": (
                "All 64 query rows score all S keys; realistic causal ranges "
                "are measured only by the production-layout benchmark."
            ),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "hostname": socket.gethostname(),
            "device_identity": {
                "name": properties.name,
                "uuid": str(device_uuid) if device_uuid is not None else None,
                "pci_bus_id": (str(pci_bus_id) if pci_bus_id is not None else None),
            },
            "device_index_in_visible_set": device_index,
            "gpu_visibility": visibility,
            "backend_source": str(backend_source),
            "backend_git": shared._git_metadata(backend_source),
            "backend_files": _backend_files(args.backend),
            "harness_sha256": shared._sha256_file(Path(__file__).resolve()),
            "shared_fixture_harness": {
                "path": str(Path(shared.__file__).resolve()),
                "sha256": shared._sha256_file(Path(shared.__file__).resolve()),
            },
        },
        "results": results,
    }
    shared._write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_round(path: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema") != ROUND_SCHEMA:
        raise SystemExit(f"unsupported matched-core round schema in {path}")
    if payload.get("comparison_kind") != COMPARISON_KIND:
        raise SystemExit(f"{path} is not a matched-core comparison round")
    if payload.get("production_comparison") is not False:
        raise SystemExit(
            f"{path} does not explicitly identify itself as non-production"
        )
    if payload.get("backend") not in BACKEND_NAMES.values():
        raise SystemExit(f"{path} has unknown backend {payload.get('backend')!r}")
    if payload.get("round") not in shared.REQUIRED_ROUNDS:
        raise SystemExit(f"{path} has invalid round {payload.get('round')!r}")
    if payload["benchmark"]["samples_per_cell"] < shared.MIN_SAMPLES:
        raise SystemExit(f"{path} has fewer than {shared.MIN_SAMPLES} samples per cell")
    result_ids = [result["cell_id"] for result in payload["results"]]
    if len(result_ids) != len(set(result_ids)):
        raise SystemExit(f"{path} contains duplicate cells")
    if set(result_ids) != set(payload["benchmark"]["cells"]):
        raise SystemExit(f"{path} results do not match its declared cells")
    for result in payload["results"]:
        samples = result["samples_ms"]
        if result["sample_count"] != len(samples) or len(samples) < shared.MIN_SAMPLES:
            raise SystemExit(
                f"{path} {result['cell_id']} does not contain "
                f"{shared.MIN_SAMPLES} samples"
            )
        if not math.isclose(
            result["round_median_ms"],
            statistics.median(samples),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise SystemExit(f"{path} {result['cell_id']} has an invalid round median")


def _validate_alternating_runs(payloads: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(payloads, key=lambda payload: payload["started_at_ns"])
    if len(ordered) != 6:
        raise SystemExit(
            "strict matched-core comparison requires six round files, got "
            f"{len(ordered)}"
        )
    first_backend = ordered[0]["backend"]
    other_backend = next(
        name for name in BACKEND_NAMES.values() if name != first_backend
    )
    expected: list[tuple[str, int]] = []
    for round_id in shared.REQUIRED_ROUNDS:
        expected.extend(((first_backend, round_id), (other_backend, round_id)))
    actual = [(payload["backend"], payload["round"]) for payload in ordered]
    if actual != expected:
        raise SystemExit(
            "matched-core runs were not alternating backend pairs for rounds "
            f"1, 2, 3: expected {expected}, got {actual}"
        )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous["finished_at_ns"] > current["started_at_ns"]:
            raise SystemExit(
                "matched-core round files overlap in wall time; runs must be serialized"
            )


def _probe_error(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if left["row_indices"] != right["row_indices"]:
        raise SystemExit("matched-core numerical probes use different row indices")
    if left["column_indices"] != right["column_indices"]:
        raise SystemExit("matched-core numerical probes use different column indices")
    left_values = [value for row in left["values"] for value in row]
    right_values = [value for row in right["values"] for value in row]
    if len(left_values) != len(right_values):
        raise SystemExit("matched-core numerical probes have different sizes")

    mismatches = 0
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for left_value, right_value in zip(left_values, right_values, strict=True):
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            if left_value != right_value:
                mismatches += 1
            continue
        absolute_error = abs(left_value - right_value)
        relative_error = absolute_error / max(abs(right_value), 1.0e-30)
        maximum_absolute_error = max(maximum_absolute_error, absolute_error)
        maximum_relative_error = max(maximum_relative_error, relative_error)
        if absolute_error > atol + rtol * abs(right_value):
            mismatches += 1
    return {
        "probe_value_count": len(left_values),
        "mismatch_count": mismatches,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "atol": atol,
        "rtol": rtol,
    }


def _combine(args: argparse.Namespace) -> None:
    payloads: list[dict[str, Any]] = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        _validate_round(path, payload)
        payloads.append(payload)

    if len(payloads) != 6:
        raise SystemExit(
            f"strict matched-core comparison requires six inputs, got {len(payloads)}"
        )
    _validate_alternating_runs(payloads)

    fixture_ids = {payload["fixture"]["fixture_id"] for payload in payloads}
    manifest_hashes = {payload["fixture"]["manifest_sha256"] for payload in payloads}
    harness_hashes = {payload["environment"]["harness_sha256"] for payload in payloads}
    shared_harness_hashes = {
        payload["environment"]["shared_fixture_harness"]["sha256"]
        for payload in payloads
    }
    if len(fixture_ids) != 1 or len(manifest_hashes) != 1:
        raise SystemExit("matched-core rounds do not use one identical fixture")
    if len(harness_hashes) != 1 or len(shared_harness_hashes) != 1:
        raise SystemExit("matched-core benchmark source changed between rounds")

    measurement_configs = {
        (
            payload["benchmark"]["warmups"],
            payload["benchmark"]["samples_per_cell"],
        )
        for payload in payloads
    }
    if len(measurement_configs) != 1:
        raise SystemExit("matched-core rounds use different measurement settings")
    gpu_identities = {
        json.dumps(
            {
                "hostname": payload["environment"]["hostname"],
                "device_identity": payload["environment"]["device_identity"],
                "gpu_visibility": payload["environment"]["gpu_visibility"],
            },
            sort_keys=True,
        )
        for payload in payloads
    }
    if len(gpu_identities) != 1:
        raise SystemExit("matched-core rounds did not use the same GPU selection")

    by_backend_round: dict[tuple[str, int], dict[str, Any]] = {}
    for payload in payloads:
        key = (payload["backend"], payload["round"])
        if key in by_backend_round:
            raise SystemExit(f"duplicate matched-core backend/round input: {key}")
        by_backend_round[key] = payload
    expected_runs = {
        (backend, round_id)
        for backend in BACKEND_NAMES.values()
        for round_id in shared.REQUIRED_ROUNDS
    }
    if set(by_backend_round) != expected_runs:
        raise SystemExit(
            f"expected matched-core runs {sorted(expected_runs)}, "
            f"got {sorted(by_backend_round)}"
        )

    for backend in BACKEND_NAMES.values():
        fingerprints = {
            tuple(
                (item["module"], item["sha256"])
                for item in payload["environment"]["backend_files"]
            )
            for payload in payloads
            if payload["backend"] == backend
        }
        if len(fingerprints) != 1:
            raise SystemExit(f"{backend} source changed between rounds")

    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for payload in payloads:
        for result in payload["results"]:
            grouped[(payload["backend"], result["cell_id"])].append(
                (payload["round"], result)
            )

    expected_cells = {
        f"{mode}:{seq_len}" for seq_len in shared.SEQ_LENS for mode in shared.MODES
    }
    found_cells = {cell_id for _, cell_id in grouped}
    if not args.allow_partial and found_cells != expected_cells:
        raise SystemExit(
            "strict matched-core comparison requires all 24 cells; "
            f"missing={sorted(expected_cells - found_cells)}, "
            f"extra={sorted(found_cells - expected_cells)}"
        )

    tokenspeed_name = BACKEND_NAMES["tokenspeed"]
    aiter_name = BACKEND_NAMES["aiter"]
    comparison_rows: list[dict[str, Any]] = []
    speedups: list[float] = []
    probe_mismatches = 0
    for cell_id in sorted(
        found_cells, key=lambda value: (int(value.split(":")[1]), value)
    ):
        rows_by_backend: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for backend in (tokenspeed_name, aiter_name):
            rows = grouped.get((backend, cell_id), [])
            rows.sort(key=lambda item: item[0])
            if tuple(round_id for round_id, _ in rows) != shared.REQUIRED_ROUNDS:
                raise SystemExit(f"{backend} {cell_id} does not cover rounds 1, 2, 3")
            rows_by_backend[backend] = rows

        input_ids = {
            result["input_identity"]["combined_id"]
            for backend_rows in rows_by_backend.values()
            for _, result in backend_rows
        }
        if len(input_ids) != 1:
            raise SystemExit(
                f"{cell_id} did not use identical matched-core input identities"
            )

        probe_results = []
        for round_index in range(len(shared.REQUIRED_ROUNDS)):
            probe_results.append(
                _probe_error(
                    rows_by_backend[tokenspeed_name][round_index][1]["numerical_probe"],
                    rows_by_backend[aiter_name][round_index][1]["numerical_probe"],
                    atol=args.probe_atol,
                    rtol=args.probe_rtol,
                )
            )
        cell_probe_mismatches = sum(
            result["mismatch_count"] for result in probe_results
        )
        probe_mismatches += cell_probe_mismatches

        tokenspeed_round_medians = [
            result["round_median_ms"] for _, result in rows_by_backend[tokenspeed_name]
        ]
        aiter_round_medians = [
            result["round_median_ms"] for _, result in rows_by_backend[aiter_name]
        ]
        tokenspeed_ms = statistics.median(tokenspeed_round_medians)
        aiter_ms = statistics.median(aiter_round_medians)
        speedup = aiter_ms / tokenspeed_ms
        speedups.append(speedup)
        mode, seq_len_text = cell_id.split(":")
        comparison_rows.append(
            {
                "cell_id": cell_id,
                "mode": mode,
                "seq_len": int(seq_len_text),
                "input_identity": next(iter(input_ids)),
                "tokenspeed_round_medians_ms": tokenspeed_round_medians,
                "aiter_round_medians_ms": aiter_round_medians,
                "tokenspeed_median_of_round_medians_ms": tokenspeed_ms,
                "aiter_median_of_round_medians_ms": aiter_ms,
                "aiter_over_tokenspeed": speedup,
                "tokenspeed_meets_or_beats_aiter": speedup >= 1.0,
                "numerical_probe": {
                    "rounds": probe_results,
                    "all_rounds_within_tolerance": cell_probe_mismatches == 0,
                },
            }
        )

    if probe_mismatches:
        raise SystemExit(
            "matched-core numerical probes differ beyond tolerance: "
            f"{probe_mismatches} values; this is not a valid performance comparison"
        )

    payload = {
        "schema": COMPARISON_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "comparison_kind": COMPARISON_KIND,
        "production_comparison": False,
        "scope_warning": (
            "These matched contiguous FP8 core timings exclude production BF16 "
            "conversion, packed addressing, and causal/ragged masking."
        ),
        "created_at_utc": shared._utc_now(),
        "fixture_id": next(iter(fixture_ids)),
        "aggregation": "median_of_three_round_medians",
        "alternating_serial_rounds_verified": True,
        "identical_input_identities_verified": True,
        "numerical_probes_within_tolerance": True,
        "speedup_definition": (
            "aiter_matched_contiguous_fp8_core_ms / "
            "tokenspeed_matched_contiguous_fp8_core_ms"
        ),
        "summary": {
            "cell_count": len(comparison_rows),
            "all_tokenspeed_meet_or_beat_aiter": all(
                speedup >= 1.0 for speedup in speedups
            ),
            "minimum_aiter_over_tokenspeed": min(speedups),
            "geomean_aiter_over_tokenspeed": math.exp(
                statistics.fmean(math.log(speedup) for speedup in speedups)
            ),
        },
        "cells": comparison_rows,
        "round_inputs": [str(path) for path in args.inputs],
    }
    shared._write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="run one implementation for one serialized alternating round",
    )
    run.add_argument("--backend", choices=tuple(BACKEND_NAMES), required=True)
    run.add_argument("--round", type=int, choices=shared.REQUIRED_ROUNDS, required=True)
    run.add_argument("--fixture-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--cells",
        help="comma-separated MODE:SEQ_LEN cells; defaults to all 24 cells",
    )
    run.add_argument("--warmups", type=int, default=10)
    run.add_argument("--samples", type=int, default=shared.MIN_SAMPLES)
    run.add_argument("--device", default="cuda:0")
    run.set_defaults(func=_run_backend)

    combine = subparsers.add_parser(
        "combine",
        help="validate and combine three alternating rounds per implementation",
    )
    combine.add_argument("--inputs", nargs="+", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--allow-partial", action="store_true")
    combine.add_argument("--probe-atol", type=float, default=5.0e-4)
    combine.add_argument("--probe-rtol", type=float, default=5.0e-3)
    combine.set_defaults(func=_combine)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
