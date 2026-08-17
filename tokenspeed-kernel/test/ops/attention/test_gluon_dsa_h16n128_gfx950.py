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

import math

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for gfx950 DSA tests", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
    _DENSE_PREFILL_SCHEDULES,
    gluon_dsa_prefill_dense_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
    _require_selected_attention_schedule,
)

_NUM_HEADS = 16
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _ROPE_DIM
_TOPK = 2048
_SOFTMAX_SCALE = 1.0 / math.sqrt(256)


def _make_inputs(
    valid_lengths: tuple[int, ...],
    *,
    seed: int,
    num_heads: int = _NUM_HEADS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    num_slots = 2112
    q = torch.randn(
        len(valid_lengths),
        num_heads,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(
        num_slots,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    slots = torch.full(
        (len(valid_lengths), _TOPK), -1, device="cuda", dtype=torch.int32
    )
    lens = torch.tensor(valid_lengths, device="cuda", dtype=torch.int32)
    for row, valid_len in enumerate(valid_lengths):
        if valid_len == 0:
            continue
        values = torch.randperm(
            num_slots, device="cuda", dtype=torch.int32, generator=generator
        )[:valid_len]
        if valid_len > 2:
            values[1] = values[0]
        slots[row, :valid_len] = values
    return q, kv, slots, lens


def _run(
    implementation,
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    *,
    k_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return implementation(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=80_000,
        qk_nope_head_dim=192,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=_SOFTMAX_SCALE,
        page_size=64,
        k_scale=k_scale,
        out=out,
    )


def _online_fp8_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    *,
    k_scale: float,
) -> torch.Tensor:
    output = torch.zeros(
        (q.shape[0], q.shape[1], _KV_LORA_RANK),
        dtype=torch.float32,
        device=q.device,
    )
    for row in range(q.shape[0]):
        running_max = torch.full(
            (q.shape[1],), -float("inf"), device=q.device, dtype=torch.float32
        )
        denominator = torch.zeros_like(running_max)
        accumulator = torch.zeros_like(output[row])
        valid_len = max(0, min(int(lens[row].item()), slots.shape[1]))
        for start in range(0, valid_len, 128):
            selected = slots[row, start : min(start + 128, valid_len)].long()
            valid = selected >= 0
            safe_selected = torch.where(valid, selected, 0)
            keys = kv[safe_selected]
            scores = torch.matmul(q[row].float(), keys.float().T)
            scores *= _SOFTMAX_SCALE * k_scale
            scores[:, ~valid] = -float("inf")
            tile_max = scores.amax(dim=1)
            new_max = torch.maximum(running_max, tile_max)
            safe_max = torch.where(torch.isfinite(new_max), new_max, 0.0)
            old_scale = torch.exp(running_max - safe_max)
            probabilities = torch.exp(scores - safe_max[:, None])
            denominator = denominator * old_scale + probabilities.sum(dim=1)
            accumulator *= old_scale[:, None]
            accumulator += torch.matmul(
                probabilities.to(torch.float8_e4m3fn).float(),
                keys[:, :_KV_LORA_RANK].float(),
            )
            running_max = new_max
        output[row] = torch.where(
            denominator[:, None] > 0,
            accumulator
            / torch.where(denominator[:, None] > 0, denominator[:, None], 1.0),
            0.0,
        )
    return output.to(torch.bfloat16)


def test_h16n128_matches_online_reference_for_selected_slot_edge_cases() -> None:
    lengths = (0, 1, 127, 128, 129, 255, 256, 257, 1539, 2047, 2048)
    q, kv, slots, lens = _make_inputs(lengths, seed=10_201)
    slots[8, 32:64] = -1
    expected = _online_fp8_reference(q, kv, slots, lens, k_scale=0.75)
    actual = _run(gluon_dsa_prefill_dense_gfx950, q, kv, slots, lens, k_scale=0.75)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=2e-2)


def test_h16n128_matches_online_fp8_reference_with_invalid_tile() -> None:
    q, kv, slots, lens = _make_inputs((0, 129), seed=10_211)
    slots[1, :128] = -1
    expected = _online_fp8_reference(q, kv, slots, lens, k_scale=1.25)

    actual = _run(
        gluon_dsa_prefill_dense_gfx950,
        q,
        kv,
        slots,
        lens,
        k_scale=1.25,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=2e-3)


def test_h16n128_single_key_is_exact_and_repeatable() -> None:
    q, kv, slots, lens = _make_inputs((1, 1, 1, 1), seed=10_212)
    chosen = torch.tensor((0, 63, 64, 2111), device="cuda", dtype=torch.int32)
    slots[:, 0] = chosen
    expected = (
        kv[chosen.long(), :_KV_LORA_RANK]
        .unsqueeze(1)
        .expand(-1, _NUM_HEADS, -1)
        .to(torch.bfloat16)
    )

    first = _run(gluon_dsa_prefill_dense_gfx950, q, kv, slots, lens)
    second = _run(gluon_dsa_prefill_dense_gfx950, q, kv, slots, lens)

    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_h16n128_uses_scratch_when_output_aliases_cache() -> None:
    q, kv, slots, lens = _make_inputs((2048, 1539, 65, 0), seed=10_231)
    expected = _run(gluon_dsa_prefill_dense_gfx950, q, kv, slots, lens)
    aliased_out = (
        kv.view(torch.bfloat16).reshape(-1)[: expected.numel()].view_as(expected)
    )

    actual = _run(
        gluon_dsa_prefill_dense_gfx950,
        q,
        kv,
        slots,
        lens,
        out=aliased_out,
    )

    assert actual.data_ptr() == aliased_out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_heads", (15, 16, 17))
def test_selected_attention_supports_head_block_boundaries(num_heads: int) -> None:
    q, kv, slots, lens = _make_inputs((129, 257), seed=10_241, num_heads=num_heads)
    expected = _online_fp8_reference(q, kv, slots, lens, k_scale=1.0)

    actual = _run(gluon_dsa_prefill_dense_gfx950, q, kv, slots, lens)

    assert actual.shape == (2, num_heads, _KV_LORA_RANK)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=2e-2)


def test_dense_schedule_wave_ownership_is_complete_and_unique() -> None:
    expected = {
        "bf16_r128_h16n64": (8, 8, 1, 2),
        "bf16_r512_h16n64": (8, 8, 1, 8),
        "e4m3_r512_h16n128": (16, 8, 2, 8),
        "e5m2_r512_h16n128": (16, 8, 2, 8),
    }

    schedules = {
        schedule.name: schedule for schedule in _DENSE_PREFILL_SCHEDULES.values()
    }
    assert set(schedules) == set(expected)
    for name, schedule in schedules.items():
        assert schedule.waves_per_cta == (1, 4)
        assert schedule.num_warps == 4
        assert _require_selected_attention_schedule(
            q_dtype=schedule.q_dtype,
            kv_dtype=schedule.q_dtype,
            kv_lora_rank=schedule.kv_lora_rank,
            qk_rope_head_dim=_ROPE_DIM,
            block_h=schedule.block_h,
            block_n=schedule.block_n,
            waves_per_cta=schedule.waves_per_cta,
            num_warps=schedule.num_warps,
            qk_k_width=schedule.qk_k_width,
            pv_k_width=schedule.pv_k_width,
            pipeline_stages=schedule.pipeline_stages,
            kv_load_slices=schedule.kv_load_slices,
        ) == (
            schedule.block_h,
            schedule.block_n,
            schedule.waves_per_cta,
            schedule.num_warps,
            schedule.qk_k_width,
            schedule.pv_k_width,
            schedule.pipeline_stages,
            schedule.kv_load_slices,
        )
        assert (
            schedule.qk_k_width,
            schedule.pv_k_width,
            schedule.score_fragments_per_wave,
            schedule.value_fragments_per_wave,
        ) == expected[name]

        waves_m, waves_n = schedule.waves_per_cta
        for output_width, fragments_per_wave in (
            (schedule.block_n, schedule.score_fragments_per_wave),
            (schedule.kv_lora_rank, schedule.value_fragments_per_wave),
        ):
            owners = {
                (m, n): ((m % waves_m) * waves_n + n % waves_n,)
                for m in range(schedule.block_h // 16)
                for n in range(output_width // 16)
            }
            assert len(owners) == schedule.num_warps * fragments_per_wave
            assert all(len(fragment_owners) == 1 for fragment_owners in owners.values())
            assert [
                sum(owner[0] == wave for owner in owners.values())
                for wave in range(schedule.num_warps)
            ] == [fragments_per_wave] * schedule.num_warps
