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
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_dsa_scoring as benchmark  # noqa: E402


class _FakeEvent:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def record(self) -> None:
        self._trace.append("event.record")

    def synchronize(self) -> None:
        self._trace.append("event.synchronize")

    def elapsed_time(self, other: _FakeEvent) -> float:
        assert isinstance(other, _FakeEvent)
        return 0.25


class _FakeGraph:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def replay(self) -> None:
        self._trace.append("graph.replay")


class _FakeGraphContext:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def __enter__(self) -> None:
        self._trace.append("graph.capture.enter")

    def __exit__(self, *unused: object) -> None:
        self._trace.append("graph.capture.exit")


def _fake_torch(trace: list[str]) -> SimpleNamespace:
    graph = _FakeGraph(trace)
    cuda = SimpleNamespace(
        CUDAGraph=lambda: graph,
        Event=lambda **unused: _FakeEvent(trace),
        graph=lambda actual: _FakeGraphContext(trace),
        synchronize=lambda: trace.append("cuda.synchronize"),
    )
    return SimpleNamespace(cuda=cuda)


def test_measure_captures_static_gpu_work_and_batches_event_sync() -> None:
    trace: list[str] = []

    elapsed, wall_seconds = benchmark._measure_captured_gpu_work(
        lambda: trace.append("launch"),
        warmups=3,
        samples=5,
        torch_module=_fake_torch(trace),
    )

    assert elapsed == [0.25] * 5
    assert wall_seconds >= 0.0
    assert trace.count("launch") == 2
    assert trace.count("graph.replay") == 8
    assert trace.count("cuda.synchronize") == 3
    assert trace.count("event.record") == 10
    assert trace.count("event.synchronize") == 1
    assert trace.index("graph.capture.enter") < trace.index("graph.capture.exit")


@pytest.mark.parametrize("warmups,samples", [(0, 1), (1, 0)])
def test_measure_rejects_empty_sample_groups(warmups: int, samples: int) -> None:
    with pytest.raises(ValueError, match="warmups and samples must be positive"):
        benchmark._measure_captured_gpu_work(
            lambda: None,
            warmups=warmups,
            samples=samples,
            torch_module=_fake_torch([]),
        )


def test_prefill_metadata_matches_short_and_long_production_dispatches() -> None:
    short = benchmark._tokenspeed_prefill_launch_metadata(512)
    assert short["grid"] == [64, 4]
    assert short["block_n"] == 128
    assert short["num_warps"] == 4
    assert short["dispatches_per_launch"] == 1

    long = benchmark._tokenspeed_prefill_launch_metadata(8192)
    assert long["grid"] == [64, 256]
    assert long["block_n"] == 32
    assert long["num_warps"] == 1
    assert long["dispatches_per_launch"] == 2


def _round_payload(
    *, backend_name: str, round_id: int, started_at_ns: int
) -> dict[str, object]:
    cell_id = "decode:512"
    samples = [0.25] * benchmark.MIN_SAMPLES
    return {
        "schema": benchmark.ROUND_SCHEMA,
        "schema_version": benchmark.SCHEMA_VERSION,
        "backend": backend_name,
        "round": round_id,
        "started_at_ns": started_at_ns,
        "finished_at_ns": started_at_ns + 1,
        "fixture": {
            "fixture_id": "fixture-id",
            "manifest_sha256": "f" * 64,
        },
        "benchmark": {
            "warmups": 25,
            "samples_per_cell": benchmark.MIN_SAMPLES,
            "compile_launches_per_cell": 1,
            "capture_launches_per_cell": 1,
            "host_submission_in_timing": False,
            "measured_execution": "captured_production_graph_replay",
            "event_timing": "torch.cuda.Event_batched_until_cell_end",
            "cells": [cell_id],
        },
        "environment": {
            "hostname": "test-host",
            "device_identity": {"name": "test-gpu", "uuid": "gpu"},
            "gpu_visibility": {"ROCR_VISIBLE_DEVICES": "2"},
            "harness_sha256": "h" * 64,
            "backend_files": [
                {
                    "module": backend_name,
                    "path": f"/{backend_name}.py",
                    "sha256": backend_name[0] * 64,
                }
            ],
        },
        "results": [
            {
                "cell_id": cell_id,
                "mode": "decode",
                "seq_len": 512,
                "samples_ms": samples,
                "sample_count": len(samples),
                "round_median_ms": 0.25,
            }
        ],
    }


def _write_rounds(tmp_path: Path) -> list[Path]:
    paths = []
    started_at_ns = 1
    for round_id in benchmark.REQUIRED_ROUNDS:
        for backend_key in ("tokenspeed", "aiter"):
            payload = _round_payload(
                backend_name=benchmark.BACKEND_NAMES[backend_key],
                round_id=round_id,
                started_at_ns=started_at_ns,
            )
            path = tmp_path / f"{backend_key}-round-{round_id}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
            started_at_ns += 2
    return paths


def test_combine_preserves_graph_measurement_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_rounds(tmp_path)
    output = tmp_path / "comparison.json"

    benchmark._combine(
        argparse.Namespace(inputs=paths, output=output, allow_partial=True)
    )
    capsys.readouterr()
    combined = json.loads(output.read_text(encoding="utf-8"))

    measurement = combined["provenance"]["measurement"]
    assert measurement["compile_launches_per_cell"] == 1
    assert measurement["capture_launches_per_cell"] == 1
    assert measurement["host_submission_in_timing"] is False
    assert measurement["measured_execution"] == "captured_production_graph_replay"
    assert measurement["event_timing"] == (
        "torch.cuda.Event_batched_until_cell_end"
    )
    assert all(
        Path(item["path"]).is_absolute()
        for item in combined["provenance"]["round_inputs"]
    )
    assert all(
        len(item["sha256"]) == 64
        for item in combined["provenance"]["round_inputs"]
    )


def test_combine_rejects_mixed_measurement_protocols(tmp_path: Path) -> None:
    paths = _write_rounds(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["benchmark"]["host_submission_in_timing"] = True
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="different measurement settings"):
        benchmark._combine(
            argparse.Namespace(
                inputs=paths,
                output=tmp_path / "comparison.json",
                allow_partial=True,
            )
        )
