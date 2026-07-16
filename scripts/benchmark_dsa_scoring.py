#!/usr/bin/env python3
"""Reproducible TokenSpeed/AITER GLM DSA scoring benchmark.

The fixture format contains only JSON metadata and NumPy arrays. BF16 and
FP8 values are stored as their unsigned integer bit patterns so independent
Torch installations consume exactly the same inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
FIXTURE_SCHEMA = "tokenspeed.dsa-scoring-fixture.v1"
ROUND_SCHEMA = "tokenspeed.dsa-scoring-round.v1"
COMPARISON_SCHEMA = "tokenspeed.dsa-scoring-comparison.v1"

SEQ_LENS = (
    512,
    1024,
    2048,
    8192,
    16384,
    32768,
    65536,
    90000,
    131072,
    262144,
    524288,
    1048576,
)
MODES = ("decode", "prefill")
NUM_HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64
PREFILL_ROWS = 64
SOFTMAX_SCALE = HEAD_DIM**-0.5
FP8_MAX = 448.0
KEY_CHUNK_ROWS = 8192
MIN_SAMPLES = 100
REQUIRED_ROUNDS = (1, 2, 3)

BACKEND_NAMES = {
    "tokenspeed": "tokenspeed_production_score",
    "aiter": "aiter_contiguous_score",
}

FIXTURE_ARRAYS = {
    "decode_q_bf16_bits": "decode_q_bf16_bits.npy",
    "decode_q_e4m3fn_bits": "decode_q_e4m3fn_bits.npy",
    "decode_weights_f32": "decode_weights_f32.npy",
    "prefill_q_bf16_bits": "prefill_q_bf16_bits.npy",
    "prefill_q_e4m3fn_bits": "prefill_q_e4m3fn_bits.npy",
    "prefill_weights_f32": "prefill_weights_f32.npy",
    "key_e4m3fn_bits": "key_e4m3fn_bits.npy",
    "key_scale_f32": "key_scale_f32.npy",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_npy(path: Path, array: Any) -> None:
    import numpy as np

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _bf16_and_fp8_bits(values: Any) -> tuple[Any, Any]:
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16)
    bf16_bits = tensor.view(torch.uint16).numpy().copy()
    fp8_bits = tensor.to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
    return bf16_bits, fp8_bits


def _generate_fixture(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        raise SystemExit("fixture generation requires torch.float8_e4m3fn")
    if args.q_std <= 0 or args.key_std <= 0:
        raise SystemExit("--q-std and --key-std must be positive")

    fixture_dir = args.fixture_dir.resolve()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fixture_dir / "manifest.json"
    existing = [
        manifest_path,
        *(fixture_dir / name for name in FIXTURE_ARRAYS.values()),
    ]
    if any(path.exists() for path in existing) and not args.force:
        raise SystemExit(
            f"fixture files already exist in {fixture_dir}; pass --force to replace them"
        )
    if args.force:
        manifest_path.unlink(missing_ok=True)

    rng = np.random.default_rng(args.seed)
    q_shapes = {
        "decode": (1, NUM_HEADS, HEAD_DIM),
        "prefill": (PREFILL_ROWS, NUM_HEADS, HEAD_DIM),
    }
    for mode, shape in q_shapes.items():
        q_source = rng.normal(0.0, args.q_std, size=shape).astype(np.float32)
        q_bf16_bits, q_fp8_bits = _bf16_and_fp8_bits(q_source)
        weights = rng.normal(0.0, NUM_HEADS**-0.5, size=shape[:2]).astype(np.float32)
        _write_npy(fixture_dir / FIXTURE_ARRAYS[f"{mode}_q_bf16_bits"], q_bf16_bits)
        _write_npy(fixture_dir / FIXTURE_ARRAYS[f"{mode}_q_e4m3fn_bits"], q_fp8_bits)
        _write_npy(fixture_dir / FIXTURE_ARRAYS[f"{mode}_weights_f32"], weights)

    max_seq_len = max(SEQ_LENS)
    key_path = fixture_dir / FIXTURE_ARRAYS["key_e4m3fn_bits"]
    scale_path = fixture_dir / FIXTURE_ARRAYS["key_scale_f32"]
    key_tmp = key_path.with_name(f".{key_path.name}.{os.getpid()}.tmp")
    scale_tmp = scale_path.with_name(f".{scale_path.name}.{os.getpid()}.tmp")
    key_bits = np.lib.format.open_memmap(
        key_tmp, mode="w+", dtype=np.uint8, shape=(max_seq_len, HEAD_DIM)
    )
    key_scales = np.lib.format.open_memmap(
        scale_tmp, mode="w+", dtype=np.float32, shape=(max_seq_len,)
    )
    for start in range(0, max_seq_len, KEY_CHUNK_ROWS):
        end = min(start + KEY_CHUNK_ROWS, max_seq_len)
        source = rng.normal(0.0, args.key_std, size=(end - start, HEAD_DIM)).astype(
            np.float32
        )
        source_bf16 = torch.from_numpy(source).to(torch.bfloat16).float()
        scales = source_bf16.abs().amax(dim=1).clamp_min(1.0e-6) / FP8_MAX
        quantized = (source_bf16 / scales[:, None]).clamp(-FP8_MAX, FP8_MAX)
        key_bits[start:end] = (
            quantized.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        )
        key_scales[start:end] = scales.numpy()
    key_bits.flush()
    key_scales.flush()
    del key_bits, key_scales
    os.replace(key_tmp, key_path)
    os.replace(scale_tmp, scale_path)

    arrays: dict[str, dict[str, Any]] = {}
    for logical_name, filename in FIXTURE_ARRAYS.items():
        path = fixture_dir / filename
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        arrays[logical_name] = {
            "filename": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _sha256_file(path),
        }

    identity_payload = {
        "schema": FIXTURE_SCHEMA,
        "seed": args.seed,
        "seq_lens": list(SEQ_LENS),
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "prefill_rows": PREFILL_ROWS,
        "q_std": args.q_std,
        "key_std": args.key_std,
        "arrays": arrays,
    }
    manifest = {
        **identity_payload,
        "schema_version": SCHEMA_VERSION,
        "fixture_id": _canonical_hash(identity_payload),
        "created_at_utc": _utc_now(),
        "serialization": {
            "container": "numpy_npy_allow_pickle_false",
            "bfloat16": "uint16_bit_pattern",
            "float8": "uint8_e4m3fn_bit_pattern",
            "byte_order": sys.byteorder,
        },
        "semantics": {
            "query_source_dtype": "bfloat16",
            "weights_dtype": "float32_signed",
            "key_dtype": "float8_e4m3fn",
            "key_scale_dtype": "float32_per_row",
            "key_dequantization": "key_e4m3fn * key_scale_f32",
            "softmax_scale": SOFTMAX_SCALE,
        },
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_manifest(fixture_dir: Path, *, verify: bool = True) -> dict[str, Any]:
    manifest_path = fixture_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != FIXTURE_SCHEMA:
        raise SystemExit(f"unsupported fixture schema in {manifest_path}")

    expected = {
        "seq_lens": list(SEQ_LENS),
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "prefill_rows": PREFILL_ROWS,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise SystemExit(
                f"fixture {field} mismatch: expected {value!r}, got {manifest.get(field)!r}"
            )
    if set(manifest.get("arrays", {})) != set(FIXTURE_ARRAYS):
        raise SystemExit(
            "fixture array set mismatch: expected "
            f"{sorted(FIXTURE_ARRAYS)}, got {sorted(manifest.get('arrays', {}))}"
        )
    identity_payload = {
        field: manifest[field]
        for field in (
            "schema",
            "seed",
            "seq_lens",
            "num_heads",
            "head_dim",
            "page_size",
            "prefill_rows",
            "q_std",
            "key_std",
            "arrays",
        )
    }
    actual_fixture_id = _canonical_hash(identity_payload)
    if manifest.get("fixture_id") != actual_fixture_id:
        raise SystemExit(
            "fixture identity mismatch: expected hash "
            f"{actual_fixture_id}, got {manifest.get('fixture_id')}"
        )
    if verify:
        for logical_name, metadata in manifest["arrays"].items():
            path = fixture_dir / metadata["filename"]
            actual = _sha256_file(path)
            if actual != metadata["sha256"]:
                raise SystemExit(
                    f"fixture checksum mismatch for {logical_name}: "
                    f"expected {metadata['sha256']}, got {actual}"
                )
    return manifest


def _parse_cells(value: str | None) -> tuple[tuple[str, int], ...]:
    if not value:
        return tuple((mode, seq_len) for seq_len in SEQ_LENS for mode in MODES)
    cells: list[tuple[str, int]] = []
    for item in value.split(","):
        try:
            mode, seq_len_text = item.strip().split(":", 1)
            seq_len = int(seq_len_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid cell {item!r}; expected MODE:SEQ_LEN"
            ) from error
        if mode not in MODES:
            raise argparse.ArgumentTypeError(
                f"invalid mode {mode!r}; expected one of {MODES}"
            )
        if seq_len not in SEQ_LENS:
            raise argparse.ArgumentTypeError(
                f"invalid sequence length {seq_len}; expected one of {SEQ_LENS}"
            )
        cell = (mode, seq_len)
        if cell not in cells:
            cells.append(cell)
    if not cells:
        raise argparse.ArgumentTypeError("at least one cell is required")
    return tuple(cells)


def _require_single_visible_gpu() -> dict[str, str]:
    visibility = {
        name: os.environ[name]
        for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        if os.environ.get(name)
    }
    if not visibility:
        raise SystemExit(
            "set ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES to one GPU after "
            "checking that it is idle"
        )
    for name, value in visibility.items():
        if "," in value:
            raise SystemExit(f"{name} must select exactly one GPU, got {value!r}")
    if len(set(visibility.values())) > 1:
        raise SystemExit(f"GPU visibility variables disagree: {visibility}")
    return visibility


def _git_metadata(path: Path) -> dict[str, Any]:
    working_directory = path if path.is_dir() else path.parent

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(working_directory), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    root = git("rev-parse", "--show-toplevel")
    return {
        "root": root,
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--short")),
    }


def _backend_files(backend: str) -> list[dict[str, str]]:
    if backend == "tokenspeed":
        module_names = (
            "tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_gfx950",
            "tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_fp8_mfma_gfx950",
            "tokenspeed_kernel_amd.ops.attention.gluon.dsa_topk_gfx950",
        )
    else:
        module_names = (
            "aiter.ops.triton.attention.fp8_mqa_logits",
            "aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits",
        )
    files = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        files.append(
            {
                "module": module_name,
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    return files


def _array_to_device(
    fixture_dir: Path,
    manifest: dict[str, Any],
    logical_name: str,
    *,
    device: Any,
    rows: int | None = None,
) -> Any:
    import numpy as np
    import torch

    metadata = manifest["arrays"][logical_name]
    source = np.load(
        fixture_dir / metadata["filename"], mmap_mode="r", allow_pickle=False
    )
    if rows is not None:
        source = source[:rows]
    host = torch.from_numpy(np.array(source, copy=True))
    return host.to(device=device, non_blocking=False)


def _load_mode_inputs(
    backend: str,
    mode: str,
    fixture_dir: Path,
    manifest: dict[str, Any],
    device: Any,
) -> tuple[Any, Any]:
    import torch

    q_kind = "bf16" if backend == "tokenspeed" else "e4m3fn"
    q = _array_to_device(
        fixture_dir,
        manifest,
        f"{mode}_q_{q_kind}_bits",
        device=device,
    )
    q = q.view(torch.bfloat16 if backend == "tokenspeed" else torch.float8_e4m3fn)
    weights = _array_to_device(
        fixture_dir,
        manifest,
        f"{mode}_weights_f32",
        device=device,
    )
    if backend == "aiter":
        # AITER has no softmax-scale argument. The positive scale commutes with
        # its per-head ReLU, so folding it into the weights preserves scoring.
        weights = weights * SOFTMAX_SCALE
    return q.contiguous(), weights.contiguous()


def _load_key_inputs(
    fixture_dir: Path,
    manifest: dict[str, Any],
    *,
    rows: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    key_bits = _array_to_device(
        fixture_dir,
        manifest,
        "key_e4m3fn_bits",
        device=device,
        rows=rows,
    )
    scales = _array_to_device(
        fixture_dir,
        manifest,
        "key_scale_f32",
        device=device,
        rows=rows,
    )
    return key_bits.view(torch.float8_e4m3fn).contiguous(), scales.contiguous()


def _pack_tokenspeed_cache(key_fp8: Any, scales: Any) -> Any:
    import torch

    slots = key_fp8.shape[0]
    if slots % PAGE_SIZE:
        raise ValueError(f"TokenSpeed cache must be page aligned, got {slots} slots")
    num_pages = slots // PAGE_SIZE
    row_bytes = HEAD_DIM + 4
    packed = torch.empty((slots, row_bytes), dtype=torch.uint8, device=key_fp8.device)
    flat = packed.reshape(-1)
    page_bytes = PAGE_SIZE * row_bytes
    fp8_view = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, PAGE_SIZE, HEAD_DIM),
        (page_bytes, HEAD_DIM, 1),
    )
    scale_view = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, PAGE_SIZE, 1),
        (page_bytes // 4, 1, 1),
        (PAGE_SIZE * HEAD_DIM) // 4,
    )
    fp8_view.copy_(key_fp8.reshape(num_pages, PAGE_SIZE, HEAD_DIM))
    scale_view.copy_(scales.reshape(num_pages, PAGE_SIZE, 1))
    return packed


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _measure(
    launch: Callable[[], None], *, warmups: int, samples: int
) -> tuple[list[float], float]:
    import torch

    launch()
    torch.cuda.synchronize()
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()

    elapsed: list[float] = []
    wall_start = time.monotonic()
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        elapsed.append(float(start.elapsed_time(end)))
    return elapsed, time.monotonic() - wall_start


def _cell_result(
    *,
    backend: str,
    mode: str,
    seq_len: int,
    samples_ms: Sequence[float],
    wall_seconds: float,
    output_shape: Sequence[int],
    launch: dict[str, Any],
) -> dict[str, Any]:
    rows = 1 if mode == "decode" else PREFILL_ROWS
    return {
        "cell_id": f"{mode}:{seq_len}",
        "backend": BACKEND_NAMES[backend],
        "mode": mode,
        "seq_len": seq_len,
        "q_shape": [rows, NUM_HEADS, HEAD_DIM],
        "weights_shape": [rows, NUM_HEADS],
        "output_shape": list(output_shape),
        "sample_count": len(samples_ms),
        "samples_ms": list(samples_ms),
        "round_median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
        "measurement_wall_seconds": wall_seconds,
        "launch": launch,
    }


def _tokenspeed_runners(
    *,
    mode: str,
    seq_len: int,
    q: Any,
    weights: Any,
    packed_cache: Any,
) -> tuple[Callable[[], None], Any, dict[str, Any]]:
    import torch
    from tokenspeed_kernel_amd.ops.attention.gluon.dsa_topk_gfx950 import (
        _launch_dsa_decode_logits_fp8,
        _launch_dsa_prefill_logits_fp8_production,
    )

    block_n = 32
    if mode == "decode":
        max_seq_len = math.ceil(seq_len / PAGE_SIZE) * PAGE_SIZE
        num_pages = max_seq_len // PAGE_SIZE
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=q.device)
        block_table = torch.arange(
            num_pages, dtype=torch.int32, device=q.device
        ).reshape(1, num_pages)
        logits = torch.empty((1, max_seq_len), dtype=torch.float32, device=q.device)

        def launch() -> None:
            _launch_dsa_decode_logits_fp8(
                q,
                packed_cache,
                weights,
                seq_lens,
                block_table,
                logits,
                page_size=PAGE_SIZE,
                row_bytes=HEAD_DIM + 4,
                softmax_scale=SOFTMAX_SCALE,
                q_len_per_req=1,
            )

        metadata = {
            "grid": [1, math.ceil(max_seq_len / block_n)],
            "block_n": block_n,
            "block_d": HEAD_DIM,
            "num_warps": 4,
            "page_table": "identity_pages",
            "logical_seq_len": seq_len,
            "allocated_seq_len": max_seq_len,
        }
        return launch, logits, metadata

    workspace_slots = torch.arange(seq_len, dtype=torch.int64, device=q.device)
    row_starts = torch.zeros(PREFILL_ROWS, dtype=torch.int32, device=q.device)
    row_ends = torch.arange(
        seq_len - PREFILL_ROWS + 1,
        seq_len + 1,
        dtype=torch.int32,
        device=q.device,
    )
    logits = torch.empty((PREFILL_ROWS, seq_len), dtype=torch.float32, device=q.device)
    query_fp8_scratch = torch.empty(
        (PREFILL_ROWS, 2, NUM_HEADS, HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    scaled_weights_scratch = torch.empty_like(weights, dtype=torch.float32)

    def launch() -> None:
        _launch_dsa_prefill_logits_fp8_production(
            q,
            packed_cache,
            weights,
            workspace_slots,
            row_starts,
            row_ends,
            logits,
            query_fp8_scratch,
            scaled_weights_scratch,
            page_size=PAGE_SIZE,
            row_bytes=HEAD_DIM + 4,
            softmax_scale=SOFTMAX_SCALE,
        )

    tiles_per_program = 2 if 512 < seq_len <= 2048 else 1
    short_prefill = seq_len <= 2048
    metadata = {
        "grid": [PREFILL_ROWS, math.ceil(seq_len / (block_n * tiles_per_program))],
        "block_n": block_n,
        "block_d": HEAD_DIM,
        "num_warps": 1,
        "waves_per_eu": 3,
        "query_decomposition": (
            "bit_guarded_e4m3_hi_plus_scaled_residual"
            if short_prefill
            else "range_normalized_e4m3_hi_plus_scaled_residual"
        ),
        "stages": (
            ["fused_query_decomposition_and_packed_mfma_score"]
            if short_prefill
            else ["query_preprocess", "packed_mfma_score"]
        ),
        "dispatches_per_launch": 1 if short_prefill else 2,
        "tiles_per_program": tiles_per_program,
        "workspace_slots": "identity_slots",
        "row_starts": "all_zero",
        "row_ends": "seq_len-63_through_seq_len",
    }
    return launch, logits, metadata


def _aiter_runner(
    *,
    mode: str,
    seq_len: int,
    q: Any,
    weights: Any,
    key_fp8: Any,
    key_scales: Any,
) -> tuple[Callable[[], None], Any, dict[str, Any], Path]:
    import torch

    module = importlib.import_module("aiter.ops.triton.attention.fp8_mqa_logits")
    if module.arch != "gfx950":
        raise RuntimeError(
            f"AITER contiguous Gluon baseline requires gfx950, got {module.arch}"
        )
    if module._gluon_fp8_mqa_logits_kernel is None:
        raise RuntimeError("AITER gfx950 Gluon fp8_mqa_logits kernel is unavailable")
    if q.dtype != torch.float8_e4m3fn or key_fp8.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "neutral fixture uses float8_e4m3fn; AITER must use the same dtype"
        )

    rows = q.shape[0]
    starts = torch.zeros(rows, dtype=torch.int32, device=q.device)
    if mode == "decode":
        ends = torch.full((1,), seq_len, dtype=torch.int32, device=q.device)
    else:
        ends = torch.arange(
            seq_len - PREFILL_ROWS + 1,
            seq_len + 1,
            dtype=torch.int32,
            device=q.device,
        )

    aligned_seq_len = math.ceil(seq_len / 256) * 256
    logits_storage = torch.empty(
        (rows, aligned_seq_len), dtype=torch.float32, device=q.device
    )
    logits = logits_storage[:, :seq_len]
    use_folded_reduction = module.FOLDED_REDUCTED_SUPPORT and NUM_HEADS > 16
    num_chains = 4 if use_folded_reduction else 0
    use_buffer_load = key_fp8.numel() * key_fp8.element_size() < 2 * 1024**3
    use_buffer_store = logits.numel() * logits.element_size() < 2 * 1024**3
    use_padded_shared = module.ASYNC_COPY_SUPPORTS_DISTRIBUTED

    def launch() -> None:
        # The AITER kernel only writes [start, end). Filling a preallocated
        # output makes exact -inf masking part of every timed invocation.
        logits_storage.fill_(-float("inf"))
        module._gluon_fp8_mqa_logits_kernel[(rows,)](
            Q_ptr=q,
            KV_ptr=key_fp8,
            kv_scales_ptr=key_scales,
            weights_ptr=weights,
            cu_start_ptr=starts,
            cu_end_ptr=ends,
            logits_ptr=logits,
            seq_len=rows,
            seq_len_kv=seq_len,
            NUM_HEADS=NUM_HEADS,
            HEAD_SIZE=HEAD_DIM,
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
        "grid": [rows],
        "block_kv": 32,
        "num_warps": 1,
        "waves_per_eu": 3,
        "num_buffers": 2,
        "num_chains": num_chains,
        "folded_reduction": use_folded_reduction,
        "padded_shared_layout": use_padded_shared,
        "contiguous_key": True,
        "timed_preallocated_neginf_fill": True,
        "logical_seq_len": seq_len,
        "allocated_seq_len": aligned_seq_len,
        "softmax_scale_folded_into_weights": SOFTMAX_SCALE,
    }
    return launch, logits, metadata, Path(module.__file__).resolve()


def _run_backend(args: argparse.Namespace) -> None:
    import torch

    if args.samples < MIN_SAMPLES:
        raise SystemExit(
            f"benchmark requires at least {MIN_SAMPLES} samples, got {args.samples}"
        )
    if args.warmups < 1:
        raise SystemExit("benchmark requires at least one warmup")
    visibility = _require_single_visible_gpu()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")

    cells = _parse_cells(args.cells)
    fixture_dir = args.fixture_dir.resolve()
    manifest = _load_manifest(fixture_dir, verify=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    started_at_utc = _utc_now()
    started_at_ns = time.time_ns()
    results: list[dict[str, Any]] = []
    backend_source: Path | None = None

    mode_inputs = {
        mode: _load_mode_inputs(args.backend, mode, fixture_dir, manifest, device)
        for mode in MODES
        if any(cell_mode == mode for cell_mode, _ in cells)
    }

    with torch.inference_mode():
        for seq_len in SEQ_LENS:
            requested_modes = [mode for mode in MODES if (mode, seq_len) in cells]
            if not requested_modes:
                continue
            key_rows = (
                math.ceil(seq_len / PAGE_SIZE) * PAGE_SIZE
                if args.backend == "tokenspeed"
                else seq_len
            )
            key_fp8, key_scales = _load_key_inputs(
                fixture_dir,
                manifest,
                rows=key_rows,
                device=device,
            )
            packed_cache = (
                _pack_tokenspeed_cache(key_fp8, key_scales)
                if args.backend == "tokenspeed"
                else None
            )
            for mode in requested_modes:
                q, weights = mode_inputs[mode]
                if args.backend == "tokenspeed":
                    launch, logits, launch_metadata = _tokenspeed_runners(
                        mode=mode,
                        seq_len=seq_len,
                        q=q,
                        weights=weights,
                        packed_cache=packed_cache,
                    )
                else:
                    launch, logits, launch_metadata, backend_source = _aiter_runner(
                        mode=mode,
                        seq_len=seq_len,
                        q=q,
                        weights=weights,
                        key_fp8=key_fp8,
                        key_scales=key_scales,
                    )
                samples_ms, wall_seconds = _measure(
                    launch, warmups=args.warmups, samples=args.samples
                )
                results.append(
                    _cell_result(
                        backend=args.backend,
                        mode=mode,
                        seq_len=seq_len,
                        samples_ms=samples_ms,
                        wall_seconds=wall_seconds,
                        output_shape=logits.shape,
                        launch=launch_metadata,
                    )
                )
                del launch, logits
            del key_fp8, key_scales, packed_cache
            gc.collect()
            torch.cuda.empty_cache()

    finished_at_ns = time.time_ns()
    module_path = (
        backend_source
        if backend_source is not None
        else Path(__file__).resolve().parents[1]
    )
    device_uuid = getattr(properties, "uuid", None)
    pci_bus_id = getattr(properties, "pci_bus_id", None)
    device_identity = {
        "name": properties.name,
        "uuid": str(device_uuid) if device_uuid is not None else None,
        "pci_bus_id": str(pci_bus_id) if pci_bus_id is not None else None,
    }
    payload = {
        "schema": ROUND_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND_NAMES[args.backend],
        "round": args.round,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
        "started_at_ns": started_at_ns,
        "finished_at_ns": finished_at_ns,
        "fixture": {
            "fixture_id": manifest["fixture_id"],
            "manifest_sha256": _sha256_file(fixture_dir / "manifest.json"),
        },
        "benchmark": {
            "warmups": args.warmups,
            "samples_per_cell": args.samples,
            "compile_launches_per_cell": 1,
            "compile_in_timing": False,
            "allocation_in_timing": False,
            "topk_in_timing": False,
            "event_timing": "torch.cuda.Event",
            "softmax_scale": SOFTMAX_SCALE,
            "cells": [f"{mode}:{seq_len}" for mode, seq_len in cells],
        },
        "timing_scope": {
            "tokenspeed_production_score": {
                "query_dtype": "bfloat16",
                "key_layout": "page64_packed_fp8_plus_per_row_fp32_scale",
                "included": [
                    "page_or_workspace_addressing",
                    "bfloat16_query_decomposition",
                    "query_and_key_loads",
                    "key_scaling",
                    "dot_products",
                    "per_head_relu",
                    "head_reduction",
                    "softmax_scale",
                    "exact_masking_and_store",
                ],
                "output_policy": "kernel_writes_every_logical_output",
            },
            "aiter_contiguous_score": {
                "query_dtype": "float8_e4m3fn_derived_from_fixture_bfloat16",
                "key_layout": "contiguous_fp8_plus_per_row_fp32_scale",
                "included": [
                    "preallocated_output_neginf_fill",
                    "query_and_key_loads",
                    "key_scaling",
                    "fp8_mfma_dot_products",
                    "per_head_relu",
                    "head_reduction",
                    "valid_range_store",
                ],
                "excluded": ["bfloat16_query_to_float8_conversion"],
                "softmax_scale": "folded_into_weights_outside_timing",
            },
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "hostname": socket.gethostname(),
            "device_identity": device_identity,
            "device_index_in_visible_set": device_index,
            "gpu_visibility": visibility,
            "backend_source": str(module_path),
            "backend_git": _git_metadata(module_path),
            "backend_files": _backend_files(args.backend),
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "results": results,
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_alternating_runs(payloads: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(payloads, key=lambda payload: payload["started_at_ns"])
    if len(ordered) != 6:
        raise SystemExit(
            f"strict comparison requires six round files, got {len(ordered)}"
        )
    first_backend = ordered[0]["backend"]
    other_backend = next(
        (name for name in BACKEND_NAMES.values() if name != first_backend), None
    )
    expected = []
    for round_id in REQUIRED_ROUNDS:
        expected.extend(((first_backend, round_id), (other_backend, round_id)))
    actual = [(payload["backend"], payload["round"]) for payload in ordered]
    if actual != expected:
        raise SystemExit(
            "runs were not alternating backend pairs for rounds 1, 2, 3: "
            f"expected {expected}, got {actual}"
        )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous["finished_at_ns"] > current["started_at_ns"]:
            raise SystemExit(
                "round files overlap in wall time; runs must be serialized"
            )


def _combine(args: argparse.Namespace) -> None:
    payloads = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema") != ROUND_SCHEMA:
            raise SystemExit(f"unsupported round schema in {path}")
        if payload["benchmark"]["samples_per_cell"] < MIN_SAMPLES:
            raise SystemExit(f"{path} has fewer than {MIN_SAMPLES} samples per cell")
        result_ids = [result["cell_id"] for result in payload["results"]]
        if len(result_ids) != len(set(result_ids)):
            raise SystemExit(f"{path} contains duplicate cells")
        if set(result_ids) != set(payload["benchmark"]["cells"]):
            raise SystemExit(f"{path} results do not match its declared cells")
        for result in payload["results"]:
            samples = result["samples_ms"]
            if result["sample_count"] != len(samples) or len(samples) < MIN_SAMPLES:
                raise SystemExit(
                    f"{path} {result['cell_id']} does not contain {MIN_SAMPLES} samples"
                )
            if not math.isclose(
                result["round_median_ms"],
                statistics.median(samples),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ):
                raise SystemExit(
                    f"{path} {result['cell_id']} has an invalid round median"
                )
        payloads.append(payload)

    fixture_ids = {payload["fixture"]["fixture_id"] for payload in payloads}
    if len(fixture_ids) != 1:
        raise SystemExit(f"rounds use different fixtures: {sorted(fixture_ids)}")
    manifest_hashes = {payload["fixture"]["manifest_sha256"] for payload in payloads}
    if len(manifest_hashes) != 1:
        raise SystemExit("rounds use different fixture manifests")
    harness_hashes = {payload["environment"]["harness_sha256"] for payload in payloads}
    if len(harness_hashes) != 1:
        raise SystemExit("rounds use different benchmark harness revisions")
    measurement_configs = {
        (
            payload["benchmark"]["warmups"],
            payload["benchmark"]["samples_per_cell"],
        )
        for payload in payloads
    }
    if len(measurement_configs) != 1:
        raise SystemExit("rounds use different warmup or sample counts")
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
        raise SystemExit("rounds did not use the same physical GPU selection")
    by_backend_round: dict[tuple[str, int], dict[str, Any]] = {}
    for payload in payloads:
        key = (payload["backend"], payload["round"])
        if key in by_backend_round:
            raise SystemExit(f"duplicate backend/round input: {key}")
        by_backend_round[key] = payload

    expected_runs = {
        (backend, round_id)
        for backend in BACKEND_NAMES.values()
        for round_id in REQUIRED_ROUNDS
    }
    if set(by_backend_round) != expected_runs:
        raise SystemExit(
            f"expected backend/round inputs {sorted(expected_runs)}, "
            f"got {sorted(by_backend_round)}"
        )
    _validate_alternating_runs(payloads)
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

    expected_cells = {f"{mode}:{seq_len}" for seq_len in SEQ_LENS for mode in MODES}
    found_cells = {cell_id for _, cell_id in grouped}
    if not args.allow_partial and found_cells != expected_cells:
        raise SystemExit(
            "strict comparison requires all 24 cells; "
            f"missing={sorted(expected_cells - found_cells)}, "
            f"extra={sorted(found_cells - expected_cells)}"
        )

    comparison_rows = []
    speedups = []
    tokenspeed_name = BACKEND_NAMES["tokenspeed"]
    aiter_name = BACKEND_NAMES["aiter"]
    for cell_id in sorted(
        found_cells, key=lambda value: (int(value.split(":")[1]), value)
    ):
        round_medians: dict[str, list[float]] = {}
        for backend in (tokenspeed_name, aiter_name):
            rows = grouped.get((backend, cell_id), [])
            if len(rows) != len(REQUIRED_ROUNDS):
                raise SystemExit(
                    f"{backend} {cell_id} has {len(rows)} rounds, expected 3"
                )
            rows.sort(key=lambda item: item[0])
            if tuple(round_id for round_id, _ in rows) != REQUIRED_ROUNDS:
                raise SystemExit(f"{backend} {cell_id} does not cover rounds 1, 2, 3")
            round_medians[backend] = [row["round_median_ms"] for _, row in rows]
        tokenspeed_ms = statistics.median(round_medians[tokenspeed_name])
        aiter_ms = statistics.median(round_medians[aiter_name])
        speedup = aiter_ms / tokenspeed_ms
        speedups.append(speedup)
        mode, seq_len_text = cell_id.split(":")
        comparison_rows.append(
            {
                "cell_id": cell_id,
                "mode": mode,
                "seq_len": int(seq_len_text),
                "tokenspeed_round_medians_ms": round_medians[tokenspeed_name],
                "aiter_round_medians_ms": round_medians[aiter_name],
                "tokenspeed_median_of_round_medians_ms": tokenspeed_ms,
                "aiter_median_of_round_medians_ms": aiter_ms,
                "aiter_over_tokenspeed": speedup,
                "tokenspeed_meets_or_beats_aiter": speedup >= 1.0,
            }
        )

    payload = {
        "schema": COMPARISON_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "fixture_id": next(iter(fixture_ids)),
        "aggregation": "median_of_three_round_medians",
        "alternating_serial_rounds_verified": True,
        "speedup_definition": "aiter_contiguous_ms / tokenspeed_production_ms",
        "summary": {
            "cell_count": len(comparison_rows),
            "all_tokenspeed_meet_or_beat_aiter": all(
                value >= 1.0 for value in speedups
            ),
            "minimum_aiter_over_tokenspeed": min(speedups),
            "geomean_aiter_over_tokenspeed": math.exp(
                statistics.fmean(math.log(value) for value in speedups)
            ),
        },
        "cells": comparison_rows,
        "round_inputs": [str(path) for path in args.inputs],
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate a portable CPU-side fixture"
    )
    generate.add_argument("--fixture-dir", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=20260716)
    generate.add_argument("--q-std", type=float, default=0.25)
    generate.add_argument("--key-std", type=float, default=0.25)
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=_generate_fixture)

    run = subparsers.add_parser(
        "run", help="run one backend for one of three alternating rounds"
    )
    run.add_argument("--backend", choices=tuple(BACKEND_NAMES), required=True)
    run.add_argument("--round", type=int, choices=REQUIRED_ROUNDS, required=True)
    run.add_argument("--fixture-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--cells",
        help="comma-separated MODE:SEQ_LEN cells; defaults to the full 24-cell grid",
    )
    run.add_argument("--warmups", type=int, default=10)
    run.add_argument("--samples", type=int, default=MIN_SAMPLES)
    run.add_argument("--device", default="cuda:0")
    run.set_defaults(func=_run_backend)

    combine = subparsers.add_parser(
        "combine", help="validate alternating rounds and aggregate their JSON"
    )
    combine.add_argument("--inputs", nargs="+", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--allow-partial", action="store_true")
    combine.set_defaults(func=_combine)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
