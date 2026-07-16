# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_dsa_scoring_matched_core as benchmark  # noqa: E402


def _canonical_with_id(payload: dict[str, Any], id_name: str) -> dict[str, Any]:
    return {**payload, id_name: benchmark.shared._canonical_hash(payload)}


def _input_identity(tmp_path: Path) -> dict[str, Any]:
    sources = {}
    for logical_name in ("query", "key", "key_scale", "weights"):
        source = {
            "fixture_id": "fixture-id",
            "logical_name": logical_name,
            "source_path": str((tmp_path / f"{logical_name}.npy").resolve()),
            "source_sha256": logical_name[0] * 64,
            "slice": "all",
        }
        sources[logical_name] = _canonical_with_id(source, "slice_id")
    return _canonical_with_id(sources, "combined_id")


def _correctness_artifact(tmp_path: Path) -> dict[str, Any]:
    checks = []
    for mode in benchmark.shared.MODES:
        checks.append(
            {
                "cell_id": f"{mode}:{benchmark.REFERENCE_SEQ_LEN}",
                "input_identity": _input_identity(tmp_path),
                "passed": True,
                "element_count": 32,
                "mismatch_count": 0,
                "maximum_absolute_error": 0.0,
                "maximum_relative_error": 0.0,
                "atol": benchmark.REFERENCE_ATOL,
                "rtol": benchmark.REFERENCE_RTOL,
                "actual_fp32_sha256": "a" * 64,
                "expected_fp32_sha256": "b" * 64,
            }
        )
    artifact = {
        "kind": "independent_full_small_cell_cpu_reference",
        "reference_definition": "synthetic CPU reference",
        "measured_timing_includes_reference": False,
        "cells": checks,
        "all_passed": True,
    }
    return _canonical_with_id(artifact, "artifact_id")


def _compiler_runtime(backend: str) -> dict[str, Any]:
    metadata = {
        "python": {"version": "test", "executable": "/python"},
        "torch": {
            "version": "test",
            "git_version": "test",
            "hip": "test",
            "debug_build": False,
        },
        "kernel_compiler": {"module": backend},
        "tokenspeed_triton_selector": None,
        "tokenspeed_triton_package_env": None,
    }
    return _canonical_with_id(metadata, "fingerprint")


def _round_payload(
    tmp_path: Path,
    *,
    backend_name: str,
    round_id: int,
    start_ns: int,
) -> dict[str, Any]:
    mode = "decode"
    seq_len = benchmark.REFERENCE_SEQ_LEN
    cell_id = f"{mode}:{seq_len}"
    launch = {
        "entrypoint": backend_name,
        "bounds_control": {"included_in_timing": True},
    }
    result = {
        "cell_id": cell_id,
        "backend": backend_name,
        "mode": mode,
        "seq_len": seq_len,
        "q_shape": [1, benchmark.shared.NUM_HEADS, benchmark.shared.HEAD_DIM],
        "key_shape": [seq_len, benchmark.shared.HEAD_DIM],
        "weights_shape": [1, benchmark.shared.NUM_HEADS],
        "output_shape": [1, seq_len],
        "sample_count": benchmark.shared.MIN_SAMPLES,
        "samples_ms": [1.0] * benchmark.shared.MIN_SAMPLES,
        "round_median_ms": 1.0,
        "minimum_ms": 1.0,
        "p95_ms": 1.0,
        "measurement_wall_seconds": 1.0,
        "input_identity": _input_identity(tmp_path),
        "launch": launch,
        "numerical_probe": {
            "purpose": "test",
            "row_indices": [0],
            "column_indices": [0],
            "values": [[1.0]],
        },
    }
    launch_contract = {
        "backend": backend_name,
        "mode": mode,
        "seq_len": seq_len,
        "q_shape": result["q_shape"],
        "key_shape": result["key_shape"],
        "output_shape": result["output_shape"],
        "launch": launch,
    }
    result["launch_fingerprint"] = benchmark.shared._canonical_hash(launch_contract)
    timing_scope = {
        "classification": benchmark.COMPARISON_KIND,
        "aiter_bounds_overhead": "included",
    }
    return {
        "schema": benchmark.ROUND_SCHEMA,
        "schema_version": benchmark.SCHEMA_VERSION,
        "comparison_kind": benchmark.COMPARISON_KIND,
        "production_comparison": False,
        "backend": backend_name,
        "round": round_id,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "finished_at_utc": "2026-01-01T00:00:01+00:00",
        "started_at_ns": start_ns,
        "finished_at_ns": start_ns + 1,
        "fixture": {
            "fixture_id": "fixture-id",
            "manifest_path": str((tmp_path / "manifest.json").resolve()),
            "manifest_sha256": "f" * 64,
        },
        "benchmark": {
            "warmups": 10,
            "samples_per_cell": benchmark.shared.MIN_SAMPLES,
            "compile_launches_per_cell": 1,
            "capture_launches_per_cell": 1,
            "host_submission_in_timing": False,
            "measured_execution": "captured_matched_core_graph_replay",
            "event_timing": "torch.cuda.Event_batched_until_cell_end",
            "cells": [cell_id],
        },
        "coverage": benchmark._coverage(((mode, seq_len),)),
        "timing_scope": timing_scope,
        "independent_correctness": _correctness_artifact(tmp_path),
        "environment": {
            "hostname": "test-host",
            "device_identity": {"name": "test-gpu", "uuid": "gpu"},
            "gpu_visibility": {"ROCR_VISIBLE_DEVICES": "0"},
            "backend_source": f"/{backend_name}.py",
            "backend_git": {"commit": "commit"},
            "backend_files": [
                {
                    "module": backend_name,
                    "path": f"/{backend_name}.py",
                    "sha256": backend_name[0] * 64,
                }
            ],
            "compiler_runtime": _compiler_runtime(backend_name),
            "harness_path": str(
                (ROOT / "scripts/benchmark_dsa_scoring_matched_core.py").resolve()
            ),
            "harness_sha256": "h" * 64,
            "shared_fixture_harness": {
                "path": str((ROOT / "scripts/benchmark_dsa_scoring.py").resolve()),
                "sha256": "s" * 64,
            },
        },
        "results": [result],
    }


def test_independent_cpu_reference_checks_all_values() -> None:
    q = torch.tensor([[[[1.0, -1.0], [2.0, 0.5]]]], dtype=torch.float32).reshape(
        1, 2, 2
    )
    key = torch.tensor([[1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
    scales = torch.tensor([2.0, 0.5], dtype=torch.float32)
    weights = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    actual = benchmark._independent_cpu_reference(
        q.to(torch.float8_e4m3fn),
        key.to(torch.float8_e4m3fn),
        scales,
        weights,
    )
    expected = torch.tensor([[-10.0, 0.0]], dtype=torch.float32)

    assert torch.equal(actual, expected)
    assert benchmark._full_reference_error(actual, expected, atol=0.0, rtol=0.0)[
        "passed"
    ]
    failed = benchmark._full_reference_error(actual + 1.0, expected, atol=0.0, rtol=0.0)
    assert failed["passed"] is False
    assert failed["mismatch_count"] == expected.numel()


def test_validate_round_requires_exact_sample_count(tmp_path: Path) -> None:
    payload = _round_payload(
        tmp_path,
        backend_name=benchmark.BACKEND_NAMES["tokenspeed"],
        round_id=1,
        start_ns=1,
    )
    benchmark._validate_round(tmp_path / "round.json", payload)
    payload["results"][0]["samples_ms"].pop()

    with pytest.raises(SystemExit, match="exactly 100 samples"):
        benchmark._validate_round(tmp_path / "round.json", payload)


def test_combine_preserves_provenance_and_labels_partial_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = []
    timestamp = 1
    for round_id in benchmark.shared.REQUIRED_ROUNDS:
        for backend_key in ("tokenspeed", "aiter"):
            backend_name = benchmark.BACKEND_NAMES[backend_key]
            payload = _round_payload(
                tmp_path,
                backend_name=backend_name,
                round_id=round_id,
                start_ns=timestamp,
            )
            path = tmp_path / f"{backend_key}-round-{round_id}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
            timestamp += 2

    output = tmp_path / "comparison.json"
    benchmark._combine(
        argparse.Namespace(
            inputs=paths,
            output=output,
            allow_partial=True,
            probe_atol=benchmark.REFERENCE_ATOL,
            probe_rtol=benchmark.REFERENCE_RTOL,
        )
    )
    capsys.readouterr()
    combined = json.loads(output.read_text(encoding="utf-8"))

    assert combined["partial_comparison"] is True
    assert combined["coverage"]["present_cells"] == [
        f"decode:{benchmark.REFERENCE_SEQ_LEN}"
    ]
    assert combined["coverage"]["missing_cells"]
    assert combined["provenance"]["common"]["fixture"]["manifest_path"] == str(
        (tmp_path / "manifest.json").resolve()
    )
    assert all(Path(item["path"]).is_absolute() for item in combined["round_inputs"])
    assert all(len(item["sha256"]) == 64 for item in combined["round_inputs"])
    assert combined["cells"][0]["input_identity"]["query"]["source_path"] == str(
        (tmp_path / "query.npy").resolve()
    )
    assert set(combined["provenance"]["backends"]) == set(
        benchmark.BACKEND_NAMES.values()
    )
    measurement = combined["provenance"]["common"]["measurement"]
    assert measurement["capture_launches_per_cell"] == 1
    assert measurement["host_submission_in_timing"] is False
    assert measurement["measured_execution"] == ("captured_matched_core_graph_replay")
    assert measurement["event_timing"] == ("torch.cuda.Event_batched_until_cell_end")
