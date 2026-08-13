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
"""Launch-policy tests for the GFX950 selected-attention kernel."""

from __future__ import annotations

import pytest
import torch

dsa_attention = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention",
    reason="tokenspeed-kernel-amd is required for DSA launch tuning tests",
)


def test_h16n128_wave_ownership_covers_each_matrix_fragment_once() -> None:
    num_waves = 4
    score_n_fragments = 128 // 16
    value_n_fragments = 512 // 16

    legacy_score_owners = {
        fragment: tuple(range(num_waves)) for fragment in range(score_n_fragments)
    }
    candidate_score_owners = {
        fragment: (fragment % num_waves,) for fragment in range(score_n_fragments)
    }
    candidate_value_owners = {
        fragment: (fragment % num_waves,) for fragment in range(value_n_fragments)
    }

    assert {len(owners) for owners in legacy_score_owners.values()} == {4}
    assert {len(owners) for owners in candidate_score_owners.values()} == {1}
    assert {len(owners) for owners in candidate_value_owners.values()} == {1}
    assert set(candidate_score_owners) == set(range(score_n_fragments))
    assert set(candidate_value_owners) == set(range(value_n_fragments))


def test_dsa_storage_alias_detection_is_conservative() -> None:
    storage = torch.empty(32, dtype=torch.uint8)
    separate = torch.empty(32, dtype=torch.uint8)

    assert dsa_attention._shares_storage(storage[:8], storage[16:])
    assert not dsa_attention._shares_storage(storage, separate, None)


def test_dsa_shared_output_storage_forces_scratch_then_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = torch.empty((1, 16, 576), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((64, 576), dtype=torch.float8_e4m3fn)
    slots = torch.zeros((1, 512), dtype=torch.int32)
    lens = torch.ones((1,), dtype=torch.int32)
    out = kv_cache.view(torch.bfloat16).reshape(-1)[: 16 * 512].view(1, 16, 512)
    recorded = {}

    def fake_run_dense_kv(*args, **kwargs):
        recorded["out"] = kwargs["out"]
        return torch.zeros((1, 16, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(dsa_attention, "_run_dense_kv", fake_run_dense_kv)

    result = dsa_attention._run_dsa(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=0.0625,
        page_size=64,
        k_scale=1.0,
        out=out,
        max_seqlen_k=512,
    )

    assert recorded["out"] is None
    assert result is out
    torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.parametrize(
    ("tokens", "expected_impl"),
    ((511, "legacy"), (512, "h16n128"), (878, "h16n128"), (8192, "h16n128")),
)
def test_dsa_prefill_routes_target_shape_to_h16n128(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    expected_impl: str,
) -> None:
    q = torch.empty((tokens, 16, 576), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((1, 576), dtype=torch.float8_e4m3fn)
    slots = torch.zeros((tokens, 512), dtype=torch.int32)
    lens = torch.ones((tokens,), dtype=torch.int32)
    called = []

    def fake_legacy(*args, **kwargs):
        called.append("legacy")
        return torch.zeros((tokens, 16, 512), dtype=torch.bfloat16)

    def fake_h16n128(*args, **kwargs):
        called.append("h16n128")
        return torch.zeros((tokens, 16, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(dsa_attention, "_run_dense_kv", fake_legacy)
    monkeypatch.setattr(dsa_attention, "_run_dense_kv_h16n128", fake_h16n128)

    result = dsa_attention.gluon_dsa_prefill_gfx950(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=80_000,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=0.0625,
        page_size=64,
    )

    assert called == [expected_impl]
    assert result.shape == (tokens, 16, 512)


def test_dsa_prefill_diagnostic_implementations_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = 8
    q = torch.empty((tokens, 16, 576), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((1, 576), dtype=torch.float8_e4m3fn)
    slots = torch.zeros((tokens, 512), dtype=torch.int32)
    lens = torch.ones((tokens,), dtype=torch.int32)
    called = []

    def fake_legacy(*args, **kwargs):
        called.append("legacy")
        return torch.zeros((tokens, 16, 512), dtype=torch.bfloat16)

    def fake_h16n128(*args, **kwargs):
        called.append("h16n128")
        return torch.zeros((tokens, 16, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(dsa_attention, "_run_dense_kv", fake_legacy)
    monkeypatch.setattr(dsa_attention, "_run_dense_kv_h16n128", fake_h16n128)
    kwargs = {
        "q": q,
        "kv_cache": kv_cache,
        "sparse_kv_cache": None,
        "topk_slots": slots,
        "topk_lens": lens,
        "max_seqlen_k": 80_000,
        "qk_nope_head_dim": 192,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "softmax_scale": 0.0625,
        "page_size": 64,
    }

    dsa_attention.gluon_dsa_prefill_legacy_gfx950(**kwargs)
    dsa_attention.gluon_dsa_prefill_h16n128_gfx950(**kwargs)

    assert called == ["legacy", "h16n128"]
