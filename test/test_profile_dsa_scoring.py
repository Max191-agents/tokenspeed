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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import profile_dsa_scoring as profile  # noqa: E402


def _backend_files(*modules: str) -> list[dict[str, str]]:
    return [
        {
            "module": module,
            "path": f"/{module.replace('.', '/')}.py",
            "sha256": module[0] * 64,
        }
        for module in modules
    ]


def test_tokenspeed_decode_dispatch_metadata() -> None:
    files = _backend_files(profile._TOKENSPEED_DECODE_MODULE)

    assert (
        profile._selected_backend_source(
            backend="tokenspeed", mode="decode", backend_files=files
        )
        == files[0]["path"]
    )
    dispatches = profile._profile_dispatches(
        backend="tokenspeed",
        mode="decode",
        seq_len=512,
        launch={"grid": [1, 16]},
        backend_files=files,
    )

    assert dispatches == [
        {
            "stage": "decode_score",
            "kernel_symbol": "_dsa_decode_logits_fp8_kernel",
            "source": files[0]["path"],
            "grid": [1, 16],
            "dispatches_per_launch": 1,
        }
    ]


def test_tokenspeed_short_prefill_dispatch_metadata() -> None:
    files = _backend_files(profile._TOKENSPEED_PREFILL_MODULE)

    assert (
        profile._selected_backend_source(
            backend="tokenspeed", mode="prefill", backend_files=files
        )
        == files[0]["path"]
    )
    dispatches = profile._profile_dispatches(
        backend="tokenspeed",
        mode="prefill",
        seq_len=1024,
        launch={"grid": [64, 8]},
        backend_files=files,
    )

    assert [dispatch["kernel_symbol"] for dispatch in dispatches] == [
        "_dsa_prefill_logits_fp8_tiled_fused_wide_kernel"
    ]
    assert dispatches[0]["stage"] == ("fused_query_decomposition_and_packed_mfma_score")
    assert dispatches[0]["grid"] == [64, 8]


def test_tokenspeed_long_prefill_dispatch_sequence_and_grids() -> None:
    files = _backend_files(profile._TOKENSPEED_PREFILL_MODULE)

    dispatches = profile._profile_dispatches(
        backend="tokenspeed",
        mode="prefill",
        seq_len=16384,
        launch={"grid": [64, 512]},
        backend_files=files,
    )

    assert [dispatch["kernel_symbol"] for dispatch in dispatches] == [
        "_dsa_preprocess_prefill_query_fp8_kernel",
        "_dsa_prefill_logits_fp8_tiled_multi_component_kernel",
    ]
    assert [dispatch["grid"] for dispatch in dispatches] == [[64], [64, 512]]
    assert all(dispatch["source"] == files[0]["path"] for dispatch in dispatches)


def test_aiter_dispatch_sequence_and_fill_grid() -> None:
    files = _backend_files(profile._AITER_KERNEL_MODULE)

    assert (
        profile._selected_backend_source(
            backend="aiter", mode="prefill", backend_files=files
        )
        == files[0]["path"]
    )
    dispatches = profile._profile_dispatches(
        backend="aiter",
        mode="prefill",
        seq_len=16384,
        launch={"allocated_seq_len": 16384, "grid": [64]},
        backend_files=files,
    )

    assert [dispatch["kernel_symbol"] for dispatch in dispatches] == [
        "FillFunctor<float>",
        "_gluon_fp8_mqa_logits_kernel",
    ]
    assert [dispatch["grid"] for dispatch in dispatches] == [[1024], [64]]
    assert dispatches[0]["source"] == "torch.Tensor.fill_"
    assert dispatches[1]["source"] == files[0]["path"]


def test_profile_counts_include_every_callable_invocation() -> None:
    dispatches = [
        {"dispatches_per_launch": 1},
        {"dispatches_per_launch": 1},
    ]

    counts = profile._profile_counts(
        dispatches=dispatches,
        warmups=5,
        profiled_launches=10,
    )

    assert counts == {
        "priming_launches": 1,
        "warmups": 5,
        "profiled_launches": 10,
        "total_callable_invocations": 16,
        "dispatches_per_launch": 2,
        "profiled_target_dispatches": 20,
        "total_target_dispatches": 32,
    }


def test_priming_launch_timer_includes_device_synchronization() -> None:
    trace: list[str] = []
    timestamps = iter((10.0, 10.125))

    elapsed = profile._prime_and_measure_wall_seconds(
        lambda: trace.append("launch"),
        lambda: trace.append("synchronize"),
        clock=lambda: next(timestamps),
    )

    assert elapsed == 0.125
    assert trace == ["launch", "synchronize"]
