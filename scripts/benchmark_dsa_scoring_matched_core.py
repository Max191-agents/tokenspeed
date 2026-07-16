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
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
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
    "tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_fp8_async_gfx950"
)
TOKENSPEED_HOOK = "launch_dsa_matched_core_fp8_async_gfx950"
AITER_WRAPPER_MODULE = "aiter.ops.triton.attention.fp8_mqa_logits"
AITER_KERNEL_MODULE = "aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits"

BACKEND_NAMES = {
    "tokenspeed": "tokenspeed_matched_contiguous_fp8_core",
    "aiter": "aiter_matched_contiguous_fp8_core",
}

REFERENCE_SEQ_LEN = min(shared.SEQ_LENS)
REFERENCE_ATOL = 5.0e-4
REFERENCE_RTOL = 5.0e-3


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


def _runtime_module_metadata(module: Any) -> dict[str, Any]:
    path_value = getattr(module, "__file__", None)
    path = Path(path_value).resolve() if path_value else None
    package = str(getattr(module, "__package__", None) or module.__name__)
    package_root = package.split(".", 1)[0]
    try:
        distribution_version = importlib.metadata.version(package_root)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "module": module.__name__,
        "module_version": getattr(module, "__version__", None),
        "distribution": package_root,
        "distribution_version": distribution_version,
        "path": str(path) if path is not None else None,
        "sha256": shared._sha256_file(path) if path is not None else None,
    }


def _compiler_runtime_metadata(backend: str) -> dict[str, Any]:
    import torch

    if backend == "tokenspeed":
        selector = importlib.import_module("tokenspeed_kernel_amd._triton")
        compiler_module = selector.triton
        selector_metadata = _runtime_module_metadata(selector)
    else:
        compiler_module = importlib.import_module("triton")
        selector_metadata = None

    metadata = {
        "python": {
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
        },
        "torch": {
            "version": torch.__version__,
            "git_version": getattr(torch.version, "git_version", None),
            "hip": getattr(torch.version, "hip", None),
            "debug_build": bool(getattr(torch.version, "debug", False)),
        },
        "kernel_compiler": _runtime_module_metadata(compiler_module),
        "tokenspeed_triton_selector": selector_metadata,
        "tokenspeed_triton_package_env": os.environ.get("TOKENSPEED_TRITON_PACKAGE"),
    }
    return {**metadata, "fingerprint": shared._canonical_hash(metadata)}


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
        "bounds_control": {
            "included_in_timing": True,
            "mechanism": "common seq_len argument and implementation predicates",
            "valid_range_per_row": [0, seq_len],
        },
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
        "bounds_control": {
            "included_in_timing": True,
            "mechanism": (
                "AITER loads per-row cu_start=0 and cu_end=seq_len and executes "
                "its rectangular bounds predicates"
            ),
            "valid_range_per_row": [0, seq_len],
        },
    }
    return launch, logits, metadata, Path(module.__file__).resolve()


def _slice_identity(
    fixture_dir: Path,
    manifest: dict[str, Any],
    logical_name: str,
    rows: int | None = None,
) -> dict[str, Any]:
    metadata = manifest["arrays"][logical_name]
    source_path = (fixture_dir / metadata["filename"]).resolve()
    slice_spec = "all" if rows is None else f"rows[0:{rows}]"
    identity = {
        "fixture_id": manifest["fixture_id"],
        "logical_name": logical_name,
        "source_path": str(source_path),
        "source_sha256": metadata["sha256"],
        "slice": slice_spec,
    }
    return {**identity, "slice_id": shared._canonical_hash(identity)}


def _cell_input_identity(
    fixture_dir: Path,
    manifest: dict[str, Any],
    mode: str,
    seq_len: int,
) -> dict[str, Any]:
    identity = {
        "query": _slice_identity(fixture_dir, manifest, f"{mode}_q_e4m3fn_bits"),
        "key": _slice_identity(fixture_dir, manifest, "key_e4m3fn_bits", seq_len),
        "key_scale": _slice_identity(fixture_dir, manifest, "key_scale_f32", seq_len),
        "weights": _slice_identity(fixture_dir, manifest, f"{mode}_weights_f32"),
    }
    return {**identity, "combined_id": shared._canonical_hash(identity)}


def _coverage(cells: Sequence[tuple[str, int]]) -> dict[str, Any]:
    expected = [
        f"{mode}:{seq_len}" for seq_len in shared.SEQ_LENS for mode in shared.MODES
    ]
    present_set = {f"{mode}:{seq_len}" for mode, seq_len in cells}
    present = [cell_id for cell_id in expected if cell_id in present_set]
    missing = [cell_id for cell_id in expected if cell_id not in present_set]
    return {
        "expected_cell_count": len(expected),
        "present_cell_count": len(present),
        "full_grid": not missing,
        "partial_comparison": bool(missing),
        "present_cells": present,
        "missing_cells": missing,
    }


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _independent_cpu_reference(
    q: Any,
    key_fp8: Any,
    key_scales: Any,
    weights: Any,
) -> Any:
    import torch

    q_bits = q.detach().view(torch.uint8).cpu().contiguous()
    key_bits = key_fp8.detach().view(torch.uint8).cpu().contiguous()
    q_fp32 = q_bits.view(torch.float8_e4m3fn).float()
    key_fp32 = key_bits.view(torch.float8_e4m3fn).float()
    scales_fp32 = key_scales.detach().float().cpu().contiguous()
    weights_fp32 = weights.detach().float().cpu().contiguous()
    per_head = torch.einsum("rhd,sd->rhs", q_fp32, key_fp32)
    per_head *= scales_fp32[None, None, :]
    return (torch.relu(per_head) * weights_fp32[:, :, None]).sum(dim=1)


def _full_reference_error(
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import torch

    actual_cpu = actual.detach().float().cpu().contiguous()
    expected_cpu = expected.detach().float().cpu().contiguous()
    if actual_cpu.shape != expected_cpu.shape:
        raise RuntimeError(
            "independent reference shape mismatch: "
            f"actual={tuple(actual_cpu.shape)}, expected={tuple(expected_cpu.shape)}"
        )
    finite_match = torch.isfinite(actual_cpu) == torch.isfinite(expected_cpu)
    close = torch.isclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol)
    matches = finite_match & close
    absolute_error = torch.abs(actual_cpu - expected_cpu)
    finite_error = torch.where(
        torch.isfinite(absolute_error), absolute_error, torch.zeros_like(absolute_error)
    )
    relative_error = finite_error / torch.clamp(torch.abs(expected_cpu), min=1.0e-30)
    mismatch_count = int((~matches).sum().item())
    return {
        "passed": mismatch_count == 0,
        "element_count": actual_cpu.numel(),
        "mismatch_count": mismatch_count,
        "maximum_absolute_error": float(finite_error.max().item()),
        "maximum_relative_error": float(relative_error.max().item()),
        "atol": atol,
        "rtol": rtol,
        "actual_fp32_sha256": _tensor_sha256(actual_cpu),
        "expected_fp32_sha256": _tensor_sha256(expected_cpu),
    }


def _run_independent_reference_checks(
    *,
    backend: str,
    fixture_dir: Path,
    manifest: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    import torch

    key_fp8, key_scales = shared._load_key_inputs(
        fixture_dir,
        manifest,
        rows=REFERENCE_SEQ_LEN,
        device=device,
    )
    checks = []
    runner = _tokenspeed_runner if backend == "tokenspeed" else _aiter_runner
    for mode in shared.MODES:
        q, weights = _load_mode_inputs(mode, fixture_dir, manifest, device)
        _validate_matched_inputs(q, key_fp8, key_scales, weights, REFERENCE_SEQ_LEN)
        launch, logits, _, _ = runner(
            q=q,
            key_fp8=key_fp8,
            key_scales=key_scales,
            weights=weights,
            seq_len=REFERENCE_SEQ_LEN,
        )
        launch()
        torch.cuda.synchronize()
        expected = _independent_cpu_reference(q, key_fp8, key_scales, weights)
        error = _full_reference_error(
            logits,
            expected,
            atol=REFERENCE_ATOL,
            rtol=REFERENCE_RTOL,
        )
        checks.append(
            {
                "cell_id": f"{mode}:{REFERENCE_SEQ_LEN}",
                "input_identity": _cell_input_identity(
                    fixture_dir, manifest, mode, REFERENCE_SEQ_LEN
                ),
                **error,
            }
        )
        del launch, logits, expected, q, weights
    del key_fp8, key_scales
    all_passed = all(check["passed"] for check in checks)
    artifact = {
        "kind": "independent_full_small_cell_cpu_reference",
        "reference_definition": (
            "CPU FP32 dot of decoded fixture FP8 q and K bytes, multiplied by "
            "FP32 per-row scales, per-head ReLU, signed FP32 head reduction"
        ),
        "measured_timing_includes_reference": False,
        "cells": checks,
        "all_passed": all_passed,
    }
    artifact["artifact_id"] = shared._canonical_hash(artifact)
    if not all_passed:
        failures = [check["cell_id"] for check in checks if not check["passed"]]
        raise SystemExit(
            "matched-core independent CPU reference failed for "
            f"{', '.join(failures)}"
        )
    return artifact


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
    launch_contract = {
        "backend": BACKEND_NAMES[backend],
        "mode": mode,
        "seq_len": seq_len,
        "q_shape": [rows, shared.NUM_HEADS, shared.HEAD_DIM],
        "key_shape": [seq_len, shared.HEAD_DIM],
        "output_shape": list(output_shape),
        "launch": launch,
    }
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
        "launch_fingerprint": shared._canonical_hash(launch_contract),
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
                        input_identity=_cell_input_identity(
                            fixture_dir, manifest, mode, seq_len
                        ),
                        probe=probe,
                    )
                )
                del launch, logits
            del key_fp8, key_scales
            gc.collect()
            torch.cuda.empty_cache()

        independent_correctness = _run_independent_reference_checks(
            backend=args.backend,
            fixture_dir=fixture_dir,
            manifest=manifest,
            device=device,
        )

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
            "manifest_path": str((fixture_dir / "manifest.json").resolve()),
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
        "coverage": _coverage(cells),
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
                "rectangular bounds-control loads and predicates",
            ],
            "excluded": [
                "BF16 query to FP8 conversion",
                "packed-cache construction",
                "page-table lookup",
                "workspace-slot lookup",
                "nontrivial causal or ragged row ranges",
                "softmax scaling",
                "output allocation or initialization",
                "TopK",
            ],
            "prefill_policy": (
                "All 64 query rows score all S keys; realistic causal ranges "
                "are measured only by the production-layout benchmark."
            ),
            "aiter_bounds_overhead": (
                "AITER still loads cu_start=0 and cu_end=S for every query row "
                "and executes its rectangular range predicates; that control "
                "work is included even though every requested output is valid."
            ),
        },
        "independent_correctness": independent_correctness,
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
            "compiler_runtime": _compiler_runtime_metadata(args.backend),
            "harness_path": str(Path(__file__).resolve()),
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
    expected_samples = payload["benchmark"]["samples_per_cell"]
    if expected_samples < shared.MIN_SAMPLES:
        raise SystemExit(f"{path} has fewer than {shared.MIN_SAMPLES} samples per cell")
    manifest_path = Path(payload["fixture"]["manifest_path"])
    if not manifest_path.is_absolute():
        raise SystemExit(f"{path} fixture manifest path is not resolved")
    compiler_runtime = dict(payload["environment"]["compiler_runtime"])
    compiler_fingerprint = compiler_runtime.pop("fingerprint", None)
    if compiler_fingerprint != shared._canonical_hash(compiler_runtime):
        raise SystemExit(f"{path} has an invalid compiler/runtime fingerprint")
    result_ids = [result["cell_id"] for result in payload["results"]]
    if len(result_ids) != len(set(result_ids)):
        raise SystemExit(f"{path} contains duplicate cells")
    if set(result_ids) != set(payload["benchmark"]["cells"]):
        raise SystemExit(f"{path} results do not match its declared cells")
    declared_cells = tuple(
        (cell_id.split(":", 1)[0], int(cell_id.split(":", 1)[1]))
        for cell_id in payload["benchmark"]["cells"]
    )
    if payload.get("coverage") != _coverage(declared_cells):
        raise SystemExit(f"{path} has invalid full-grid coverage metadata")

    correctness = dict(payload.get("independent_correctness", {}))
    correctness_id = correctness.pop("artifact_id", None)
    if correctness_id != shared._canonical_hash(correctness):
        raise SystemExit(f"{path} has an invalid independent correctness artifact")
    expected_reference_cells = {f"{mode}:{REFERENCE_SEQ_LEN}" for mode in shared.MODES}
    reference_cells = correctness.get("cells", [])
    if (
        correctness.get("all_passed") is not True
        or {check.get("cell_id") for check in reference_cells}
        != expected_reference_cells
    ):
        raise SystemExit(f"{path} lacks passing full-small-cell reference checks")
    if any(check.get("passed") is not True for check in reference_cells):
        raise SystemExit(f"{path} contains a failed independent reference check")

    for result in payload["results"]:
        samples = result["samples_ms"]
        if (
            result["sample_count"] != expected_samples
            or len(samples) != expected_samples
        ):
            raise SystemExit(
                f"{path} {result['cell_id']} does not contain exactly "
                f"{expected_samples} samples"
            )
        input_identity = dict(result["input_identity"])
        combined_id = input_identity.pop("combined_id", None)
        if combined_id != shared._canonical_hash(input_identity):
            raise SystemExit(
                f"{path} {result['cell_id']} has an invalid input identity"
            )
        for source in input_identity.values():
            source_without_id = dict(source)
            slice_id = source_without_id.pop("slice_id", None)
            if slice_id != shared._canonical_hash(source_without_id):
                raise SystemExit(
                    f"{path} {result['cell_id']} has an invalid input slice identity"
                )
            if not Path(source["source_path"]).is_absolute():
                raise SystemExit(
                    f"{path} {result['cell_id']} has an unresolved input path"
                )
            if len(source["source_sha256"]) != 64:
                raise SystemExit(
                    f"{path} {result['cell_id']} has an invalid input SHA256"
                )
        launch_contract = {
            "backend": payload["backend"],
            "mode": result["mode"],
            "seq_len": result["seq_len"],
            "q_shape": result["q_shape"],
            "key_shape": result["key_shape"],
            "output_shape": result["output_shape"],
            "launch": result["launch"],
        }
        if result.get("launch_fingerprint") != shared._canonical_hash(launch_contract):
            raise SystemExit(
                f"{path} {result['cell_id']} has an invalid launch fingerprint"
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
    manifest_paths = {payload["fixture"]["manifest_path"] for payload in payloads}
    manifest_hashes = {payload["fixture"]["manifest_sha256"] for payload in payloads}
    harness_paths = {payload["environment"]["harness_path"] for payload in payloads}
    harness_hashes = {payload["environment"]["harness_sha256"] for payload in payloads}
    shared_harness_paths = {
        payload["environment"]["shared_fixture_harness"]["path"] for payload in payloads
    }
    shared_harness_hashes = {
        payload["environment"]["shared_fixture_harness"]["sha256"]
        for payload in payloads
    }
    if len(fixture_ids) != 1 or len(manifest_paths) != 1 or len(manifest_hashes) != 1:
        raise SystemExit("matched-core rounds do not use one identical fixture")
    if (
        len(harness_paths) != 1
        or len(harness_hashes) != 1
        or len(shared_harness_paths) != 1
        or len(shared_harness_hashes) != 1
    ):
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
    timing_scope_fingerprints = {
        shared._canonical_hash(payload["timing_scope"]) for payload in payloads
    }
    if len(timing_scope_fingerprints) != 1:
        raise SystemExit("matched-core timing scope changed between rounds")
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
        compiler_fingerprints = {
            payload["environment"]["compiler_runtime"]["fingerprint"]
            for payload in payloads
            if payload["backend"] == backend
        }
        if len(compiler_fingerprints) != 1:
            raise SystemExit(f"{backend} compiler/runtime changed between rounds")

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
    extra_cells = found_cells - expected_cells
    missing_cells = expected_cells - found_cells
    if extra_cells:
        raise SystemExit(
            f"matched-core comparison contains unsupported cells: {sorted(extra_cells)}"
        )
    if not args.allow_partial and found_cells != expected_cells:
        raise SystemExit(
            "strict matched-core comparison requires all 24 cells; "
            f"missing={sorted(missing_cells)}"
        )
    aggregate_cells = tuple(
        (cell_id.split(":", 1)[0], int(cell_id.split(":", 1)[1]))
        for cell_id in found_cells
    )
    aggregate_coverage = _coverage(aggregate_cells)

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

        input_identities = {
            json.dumps(result["input_identity"], sort_keys=True)
            for backend_rows in rows_by_backend.values()
            for _, result in backend_rows
        }
        if len(input_identities) != 1:
            raise SystemExit(
                f"{cell_id} did not use identical matched-core input identities"
            )
        input_identity = json.loads(next(iter(input_identities)))

        launch_fingerprints: dict[str, str] = {}
        for backend, backend_rows in rows_by_backend.items():
            fingerprints = {result["launch_fingerprint"] for _, result in backend_rows}
            if len(fingerprints) != 1:
                raise SystemExit(f"{backend} {cell_id} launch changed between rounds")
            launch_fingerprints[backend] = next(iter(fingerprints))

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
                "input_identity": input_identity,
                "launch_fingerprints": launch_fingerprints,
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

    backend_provenance: dict[str, Any] = {}
    for backend in BACKEND_NAMES.values():
        backend_rounds = sorted(
            (payload for payload in payloads if payload["backend"] == backend),
            key=lambda payload: payload["round"],
        )
        first = backend_rounds[0]
        backend_provenance[backend] = {
            "backend_source": first["environment"]["backend_source"],
            "backend_git": first["environment"]["backend_git"],
            "backend_files": first["environment"]["backend_files"],
            "compiler_runtime": first["environment"]["compiler_runtime"],
            "independent_correctness_artifacts": [
                {
                    "round": payload["round"],
                    **payload["independent_correctness"],
                }
                for payload in backend_rounds
            ],
        }

    round_inputs = []
    for path, round_payload in zip(args.inputs, payloads, strict=True):
        resolved = path.resolve()
        round_inputs.append(
            {
                "path": str(resolved),
                "sha256": shared._sha256_file(resolved),
                "backend": round_payload["backend"],
                "round": round_payload["round"],
            }
        )

    warmups, samples_per_cell = next(iter(measurement_configs))
    common_provenance = {
        "fixture": {
            "fixture_id": next(iter(fixture_ids)),
            "manifest_path": next(iter(manifest_paths)),
            "manifest_sha256": next(iter(manifest_hashes)),
        },
        "matched_core_harness": {
            "path": next(iter(harness_paths)),
            "sha256": next(iter(harness_hashes)),
        },
        "shared_fixture_harness": {
            "path": next(iter(shared_harness_paths)),
            "sha256": next(iter(shared_harness_hashes)),
        },
        "gpu": json.loads(next(iter(gpu_identities))),
        "measurement": {
            "warmups": warmups,
            "samples_per_cell_exact": samples_per_cell,
            "round_count_per_backend": len(shared.REQUIRED_ROUNDS),
            "aggregation": "median_of_three_round_medians",
            "event_timing": "torch.cuda.Event",
        },
        "timing_scope": payloads[0]["timing_scope"],
        "timing_scope_fingerprint": next(iter(timing_scope_fingerprints)),
        "round_inputs": round_inputs,
    }

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
        "coverage": aggregate_coverage,
        "partial_comparison": aggregate_coverage["partial_comparison"],
        "aggregation": "median_of_three_round_medians",
        "alternating_serial_rounds_verified": True,
        "identical_input_identities_verified": True,
        "numerical_probes_within_tolerance": True,
        "speedup_definition": (
            "aiter_matched_contiguous_fp8_core_ms / "
            "tokenspeed_matched_contiguous_fp8_core_ms"
        ),
        "provenance": {
            "common": common_provenance,
            "backends": backend_provenance,
        },
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
        "round_inputs": round_inputs,
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
