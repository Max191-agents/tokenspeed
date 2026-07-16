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

import pytest
import torch
from tokenspeed_kernel_amd.ops.attention.gluon import dsa_topk_gfx950


def _scoring_inputs(heads: int, workspace_rows: int) -> tuple[torch.Tensor, ...]:
    q = torch.empty((1, heads, 128), dtype=torch.bfloat16)
    index_k_cache = torch.empty((64, 132), dtype=torch.uint8)
    weights = torch.empty((1, heads), dtype=torch.float32)
    kv_workspace_slots = torch.empty((workspace_rows,), dtype=torch.int64)
    row_starts = torch.zeros((1,), dtype=torch.int32)
    row_ends = torch.full((1,), workspace_rows, dtype=torch.int32)
    logits = torch.empty((1, workspace_rows), dtype=torch.float32)
    query_fp8_scratch = torch.empty(
        (1, 2, 32, 128), dtype=torch.float8_e4m3fn
    )
    scaled_weights_scratch = torch.empty((1, 32), dtype=torch.float32)
    return (
        q,
        index_k_cache,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        logits,
        query_fp8_scratch,
        scaled_weights_scratch,
    )


@pytest.mark.parametrize(
    "workspace_rows,expected_tiles",
    [(512, 1), (513, 2), (2048, 2), (2049, 1)],
)
def test_production_prefill_uses_tuned_mfma_configuration(
    monkeypatch: pytest.MonkeyPatch,
    workspace_rows: int,
    expected_tiles: int,
) -> None:
    inputs = _scoring_inputs(heads=32, workspace_rows=workspace_rows)
    logits = inputs[6]
    launch_kwargs: dict[str, object] = {}

    def fake_mfma(*args: object, **kwargs: object) -> torch.Tensor:
        assert args[6] is logits
        launch_kwargs.update(kwargs)
        return logits

    monkeypatch.setattr(
        dsa_topk_gfx950,
        "launch_dsa_prefill_logits_fp8_mfma_gfx950",
        fake_mfma,
    )

    result = dsa_topk_gfx950._launch_dsa_prefill_logits_fp8_production(
        *inputs,
        page_size=64,
        row_bytes=132,
        softmax_scale=0.125,
    )

    assert result is logits
    assert launch_kwargs == {
        "softmax_scale": 0.125,
        "tiles_per_program": expected_tiles,
    }


def test_production_prefill_preserves_scalar_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _scoring_inputs(heads=16, workspace_rows=4096)
    logits = inputs[6]
    scalar_kwargs: dict[str, object] = {}

    def fake_scalar(*args: object, **kwargs: object) -> torch.Tensor:
        assert args[-1] is logits
        scalar_kwargs.update(kwargs)
        return logits

    def fail_mfma(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError(
            "fixed-shape MFMA scorer must not handle other head counts"
        )

    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_launch_dsa_prefill_logits_fp8",
        fake_scalar,
    )
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "launch_dsa_prefill_logits_fp8_mfma_gfx950",
        fail_mfma,
    )

    result = dsa_topk_gfx950._launch_dsa_prefill_logits_fp8_production(
        *inputs,
        page_size=64,
        row_bytes=132,
        softmax_scale=0.125,
    )

    assert result is logits
    assert scalar_kwargs == {
        "page_size": 64,
        "row_bytes": 132,
        "softmax_scale": 0.125,
    }


@pytest.mark.parametrize("incompatibility", ["page_size", "row_bytes", "q_layout"])
def test_production_prefill_falls_back_outside_glm_specialization(
    monkeypatch: pytest.MonkeyPatch,
    incompatibility: str,
) -> None:
    inputs = list(_scoring_inputs(heads=32, workspace_rows=512))
    if incompatibility == "q_layout":
        inputs[0] = torch.empty((1, 32, 256), dtype=torch.bfloat16)[:, :, ::2]
        assert not inputs[0].is_contiguous()
    page_size = 32 if incompatibility == "page_size" else 64
    row_bytes = 136 if incompatibility == "row_bytes" else 132
    scalar_calls = 0

    def fake_scalar(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal scalar_calls
        scalar_calls += 1
        return args[-1]  # type: ignore[return-value]

    def fail_mfma(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("incompatible inputs must use the scalar scorer")

    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_launch_dsa_prefill_logits_fp8",
        fake_scalar,
    )
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "launch_dsa_prefill_logits_fp8_mfma_gfx950",
        fail_mfma,
    )

    result = dsa_topk_gfx950._launch_dsa_prefill_logits_fp8_production(
        *inputs,
        page_size=page_size,
        row_bytes=row_bytes,
        softmax_scale=0.125,
    )

    assert result is inputs[6]
    assert scalar_calls == 1
