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
from dataclasses import dataclass

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip("AMD GFX950 is required for Gluon DSA tests", allow_module_level=True)


from tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_fp8_mfma_gfx950 import (  # noqa: E402
    launch_dsa_prefill_logits_fp8_mfma_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_topk_gfx950 import (  # noqa: E402
    gluon_dsa_prefill_topk_fp8_gfx950,
)

_HEADS = 32
_HEAD_DIM = 128
_PAGE_SIZE = 64
_ROW_BYTES = _HEAD_DIM + 4
_SOFTMAX_SCALE = _HEAD_DIM**-0.5

# The retained kernels approximate each BF16 query with scaled E4M3 terms. These
# are the production validator's limits and remain below the TopK score gaps.
_RTOL = 3.0e-3
_ATOL = 3.0e-4


@dataclass(frozen=True)
class _PrefillCase:
    q: torch.Tensor
    weights: torch.Tensor
    packed_index_k: torch.Tensor
    index_k: torch.Tensor
    scales: torch.Tensor
    kv_workspace_slots: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _pack_index_k_cache(index_k: torch.Tensor) -> torch.Tensor:
    num_slots = int(index_k.shape[0])
    assert num_slots % _PAGE_SIZE == 0
    num_pages = num_slots // _PAGE_SIZE
    packed = torch.empty(
        (num_slots, _ROW_BYTES), device=index_k.device, dtype=torch.uint8
    )
    values = index_k.float()
    scales = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    fp8_values = (values / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    flat = packed.reshape(-1)
    page_bytes = _PAGE_SIZE * _ROW_BYTES
    fp8_view = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, _PAGE_SIZE, _HEAD_DIM),
        (page_bytes, _HEAD_DIM, 1),
    )
    scale_view = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, _PAGE_SIZE, 1),
        (page_bytes // 4, 1, 1),
        (_PAGE_SIZE * _HEAD_DIM) // 4,
    )
    fp8_view.copy_(fp8_values.reshape(num_pages, _PAGE_SIZE, _HEAD_DIM))
    scale_view.copy_(scales.reshape(num_pages, _PAGE_SIZE, 1))
    return packed


def _unpack_index_k_cache(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_slots = int(packed.shape[0])
    num_pages = num_slots // _PAGE_SIZE
    flat = packed.reshape(-1)
    page_bytes = _PAGE_SIZE * _ROW_BYTES
    fp8_values = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, _PAGE_SIZE, _HEAD_DIM),
        (page_bytes, _HEAD_DIM, 1),
    )
    scales = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, _PAGE_SIZE, 1),
        (page_bytes // 4, 1, 1),
        (_PAGE_SIZE * _HEAD_DIM) // 4,
    )
    index_k = fp8_values.float() * scales
    return index_k.reshape(num_slots, _HEAD_DIM), scales.reshape(num_slots)


def _make_random_case(
    *,
    tokens: int,
    seq_len_sum: int,
    row_starts: list[int],
    row_ends: list[int],
    seed: int,
) -> _PrefillCase:
    assert len(row_starts) == tokens
    assert len(row_ends) == tokens
    assert all(
        0 <= start < end <= seq_len_sum for start, end in zip(row_starts, row_ends)
    )

    generator = _generator(seed)
    q = (
        torch.randn(
            (tokens, _HEADS, _HEAD_DIM),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16)
    weights = torch.randn(
        (tokens, _HEADS),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) * (_HEADS**-0.5)

    num_pages = math.ceil(seq_len_sum / _PAGE_SIZE)
    num_slots = num_pages * _PAGE_SIZE
    row_amplitudes = torch.linspace(
        0.05,
        1.25,
        num_slots,
        device="cuda",
        dtype=torch.float32,
    )
    source_index_k = torch.randn(
        (num_slots, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    source_index_k = (source_index_k * row_amplitudes[:, None]).to(torch.bfloat16)
    packed_index_k = _pack_index_k_cache(source_index_k)
    index_k, scales = _unpack_index_k_cache(packed_index_k)
    kv_workspace_slots = torch.randperm(
        num_slots,
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )[:seq_len_sum].contiguous()
    starts = torch.tensor(row_starts, device="cuda", dtype=torch.int32)
    ends = torch.tensor(row_ends, device="cuda", dtype=torch.int32)
    case = _PrefillCase(
        q=q.contiguous(),
        weights=weights.contiguous(),
        packed_index_k=packed_index_k,
        index_k=index_k,
        scales=scales,
        kv_workspace_slots=kv_workspace_slots,
        row_starts=starts,
        row_ends=ends,
    )
    _assert_model_contract(case)
    return case


def _make_rank_case(*, seq_len_sum: int, seed: int) -> _PrefillCase:
    num_pages = math.ceil(seq_len_sum / _PAGE_SIZE)
    num_slots = num_pages * _PAGE_SIZE
    head_ids = torch.arange(_HEADS, device="cuda")
    head_scales = torch.where(
        head_ids < 16,
        1.0,
        torch.where(head_ids < 24, 0.5, -1.0),
    )
    q = head_scales[:, None].expand(_HEADS, _HEAD_DIM).unsqueeze(0)
    q = q.to(torch.bfloat16).contiguous()
    weights = torch.where(
        head_ids.remainder(2) == 0,
        0.25,
        -0.125,
    ).to(torch.float32)
    weights = (weights * (_HEADS**-0.5)).unsqueeze(0).contiguous()

    amplitudes = torch.linspace(
        0.05,
        1.25,
        num_slots,
        device="cuda",
        dtype=torch.float32,
    )
    source_index_k = amplitudes[:, None].expand(num_slots, _HEAD_DIM)
    packed_index_k = _pack_index_k_cache(source_index_k)
    index_k, scales = _unpack_index_k_cache(packed_index_k)
    kv_workspace_slots = torch.randperm(
        num_slots,
        device="cuda",
        dtype=torch.int64,
        generator=_generator(seed),
    )[:seq_len_sum].contiguous()
    row_start = 7
    row_end = seq_len_sum - 3
    case = _PrefillCase(
        q=q,
        weights=weights,
        packed_index_k=packed_index_k,
        index_k=index_k,
        scales=scales,
        kv_workspace_slots=kv_workspace_slots,
        row_starts=torch.tensor([row_start], device="cuda", dtype=torch.int32),
        row_ends=torch.tensor([row_end], device="cuda", dtype=torch.int32),
    )
    _assert_model_contract(case)
    assert row_end - row_start > 2048
    return case


def _make_high_dynamic_range_case(*, seq_len_sum: int, seed: int) -> _PrefillCase:
    num_pages = math.ceil(seq_len_sum / _PAGE_SIZE)
    num_slots = num_pages * _PAGE_SIZE
    base = torch.linspace(-1.0, 1.0, _HEAD_DIM, device="cuda", dtype=torch.float32)
    head_amplitudes = torch.logspace(
        math.log10(600.0),
        math.log10(1.0e10),
        _HEADS,
        device="cuda",
        dtype=torch.float32,
    )
    q = (head_amplitudes[:, None] * base).to(torch.bfloat16).unsqueeze(0)
    head_ids = torch.arange(_HEADS, device="cuda")
    signed_coefficients = torch.where(
        head_ids.remainder(2) == 0,
        0.25,
        -0.125,
    )
    weights = (
        signed_coefficients * head_amplitudes.reciprocal() * (_HEADS**-0.5)
    ).unsqueeze(0)

    row_amplitudes = torch.linspace(
        0.05,
        1.25,
        num_slots,
        device="cuda",
        dtype=torch.float32,
    )
    source_index_k = (row_amplitudes[:, None] * base).to(torch.bfloat16)
    packed_index_k = _pack_index_k_cache(source_index_k)
    index_k, scales = _unpack_index_k_cache(packed_index_k)
    kv_workspace_slots = torch.randperm(
        num_slots,
        device="cuda",
        dtype=torch.int64,
        generator=_generator(seed),
    )[:seq_len_sum].contiguous()
    case = _PrefillCase(
        q=q.contiguous(),
        weights=weights.to(torch.float32).contiguous(),
        packed_index_k=packed_index_k,
        index_k=index_k,
        scales=scales,
        kv_workspace_slots=kv_workspace_slots,
        row_starts=torch.tensor([3], device="cuda", dtype=torch.int32),
        row_ends=torch.tensor([seq_len_sum - 5], device="cuda", dtype=torch.int32),
    )
    _assert_model_contract(case)
    q_abs = case.q.float().abs()
    per_head_amax = q_abs.amax(dim=-1)
    assert bool(torch.isfinite(q_abs).all())
    assert float(per_head_amax.min().item()) >= 600.0
    assert float(per_head_amax.max().item()) >= 9.0e9
    return case


def _assert_model_contract(case: _PrefillCase) -> None:
    assert case.q.dtype == torch.bfloat16
    assert case.q.shape[1:] == (_HEADS, _HEAD_DIM)
    assert case.weights.dtype == torch.float32
    assert bool((case.weights < 0).any())
    assert bool((case.weights > 0).any())
    assert bool(torch.isfinite(case.scales).all())
    assert float(case.scales.min().item()) > 0.0
    assert float(case.scales.max().item()) > 4.0 * float(case.scales.min().item())
    assert not torch.equal(
        case.kv_workspace_slots,
        torch.arange(case.kv_workspace_slots.numel(), device="cuda", dtype=torch.int64),
    )
    assert bool((case.kv_workspace_slots.diff().abs() != 1).any())
    assert bool((case.row_starts > 0).all())
    assert bool((case.row_starts.remainder(_PAGE_SIZE) != 0).all())


def _reference_prefill_logits(case: _PrefillCase) -> torch.Tensor:
    seq_len_sum = int(case.kv_workspace_slots.numel())
    expected = torch.full(
        (case.q.shape[0], seq_len_sum),
        -float("inf"),
        device="cuda",
        dtype=torch.float32,
    )
    for token in range(case.q.shape[0]):
        row_start = int(case.row_starts[token].item())
        row_end = int(case.row_ends[token].item())
        rows = torch.arange(row_start, row_end, device="cuda", dtype=torch.long)
        slots = case.kv_workspace_slots.index_select(0, rows).long()
        keys = case.index_k.index_select(0, slots).float()
        per_head = keys @ case.q[token].float().transpose(0, 1)
        expected[token, row_start:row_end] = (
            torch.relu(per_head) * case.weights[token].float()
        ).sum(dim=1) * _SOFTMAX_SCALE
    return expected


def _launch(
    case: _PrefillCase,
    *,
    tiles_per_program: int,
    logits: torch.Tensor | None = None,
    query_fp8_scratch: torch.Tensor | None = None,
    scaled_weights_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits is None:
        logits = torch.full(
            (case.q.shape[0], case.kv_workspace_slots.numel()),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )
    if query_fp8_scratch is None:
        assert scaled_weights_scratch is None
        query_fp8_scratch = torch.empty(
            (case.q.shape[0], 2, _HEADS, _HEAD_DIM),
            device=case.q.device,
            dtype=torch.float8_e4m3fn,
        )
        scaled_weights_scratch = torch.empty(
            (case.q.shape[0], _HEADS),
            device=case.q.device,
            dtype=torch.float32,
        )
    assert scaled_weights_scratch is not None
    return launch_dsa_prefill_logits_fp8_mfma_gfx950(
        case.q,
        case.packed_index_k,
        case.weights,
        case.kv_workspace_slots,
        case.row_starts,
        case.row_ends,
        logits,
        query_fp8_scratch,
        scaled_weights_scratch,
        softmax_scale=_SOFTMAX_SCALE,
        tiles_per_program=tiles_per_program,
    )


def _assert_logits_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    expected_mask = torch.isneginf(expected)
    assert torch.equal(torch.isneginf(actual), expected_mask)
    assert not bool(torch.isnan(actual).any())
    torch.testing.assert_close(
        actual[~expected_mask],
        expected[~expected_mask],
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("seq_len_sum", [197, 2047])
def test_scaled_residual_two_tile_prefill_matches_bf16_reference(
    seq_len_sum: int,
) -> None:
    case = _make_random_case(
        tokens=4,
        seq_len_sum=seq_len_sum,
        row_starts=[3, 35, 67, 129],
        row_ends=[seq_len_sum, seq_len_sum - 1, seq_len_sum - 37, 182],
        seed=5200 + seq_len_sum,
    )
    expected = _reference_prefill_logits(case)
    actual = _launch(case, tiles_per_program=2)

    _assert_logits_match(actual, expected)


def test_scaled_residual_long_prefill_matches_bf16_reference() -> None:
    seq_len_sum = 4099
    case = _make_random_case(
        tokens=3,
        seq_len_sum=seq_len_sum,
        row_starts=[5, 69, 133],
        row_ends=[seq_len_sum, seq_len_sum - 1, seq_len_sum - 41],
        seed=6301,
    )
    expected = _reference_prefill_logits(case)
    actual = _launch(case, tiles_per_program=1)

    _assert_logits_match(actual, expected)


def test_range_safe_prefill_matches_high_dynamic_range_bf16_reference() -> None:
    case = _make_high_dynamic_range_case(seq_len_sum=509, seed=6351)
    expected = _reference_prefill_logits(case)
    actual = _launch(case, tiles_per_program=2)

    _assert_logits_match(actual, expected)


def test_scaled_residual_topk_set_and_rank_are_exact_above_2048_candidates() -> None:
    topk = 2048
    case = _make_rank_case(seq_len_sum=2179, seed=6401)
    expected = _reference_prefill_logits(case)
    actual = _launch(case, tiles_per_program=1)
    _assert_logits_match(actual, expected)

    row_start = int(case.row_starts[0].item())
    row_end = int(case.row_ends[0].item())
    expected_topk = torch.topk(expected[0, row_start:row_end], k=topk)
    actual_topk = torch.topk(actual[0, row_start:row_end], k=topk)
    expected_gaps = expected_topk.values[:-1] - expected_topk.values[1:]
    assert float(expected_gaps.min().item()) > _ATOL
    assert torch.equal(
        torch.sort(actual_topk.indices).values,
        torch.sort(expected_topk.indices).values,
    )
    assert torch.equal(actual_topk.indices, expected_topk.indices)
    torch.testing.assert_close(
        actual_topk.values,
        expected_topk.values,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_public_prefill_topk_matches_production_mfma_score_set() -> None:
    topk = 2048
    case = _make_rank_case(seq_len_sum=2179, seed=6402)
    expected = _reference_prefill_logits(case)
    expected_indices = torch.topk(expected[0], k=topk).indices

    actual_indices, actual_lens = gluon_dsa_prefill_topk_fp8_gfx950(
        case.q,
        case.weights,
        case.kv_workspace_slots,
        case.row_starts,
        case.row_ends,
        topk=topk,
        softmax_scale=_SOFTMAX_SCALE,
        index_k_cache=case.packed_index_k,
        page_size=_PAGE_SIZE,
    )

    assert int(actual_lens[0].item()) == topk
    assert actual_indices.shape == (1, topk)
    assert actual_indices.dtype == torch.int32
    assert int(torch.unique(actual_indices[0]).numel()) == topk
    assert bool((actual_indices[0] >= case.row_starts[0]).all())
    assert bool((actual_indices[0] < case.row_ends[0]).all())
    assert torch.equal(
        torch.sort(actual_indices[0]).values,
        torch.sort(expected_indices).values,
    )


def test_scaled_residual_prefill_uses_nondefault_concurrent_streams() -> None:
    short_case = _make_random_case(
        tokens=2,
        seq_len_sum=509,
        row_starts=[3, 71],
        row_ends=[509, 503],
        seed=6501,
    )
    long_case = _make_random_case(
        tokens=2,
        seq_len_sum=2113,
        row_starts=[5, 133],
        row_ends=[2113, 2089],
        seed=6502,
    )
    short_expected = _reference_prefill_logits(short_case)
    long_expected = _reference_prefill_logits(long_case)
    short_actual = torch.full_like(short_expected, float("nan"))
    long_actual = torch.full_like(long_expected, float("nan"))

    ready = torch.cuda.Event()
    ready.record()
    short_stream = torch.cuda.Stream()
    long_stream = torch.cuda.Stream()
    short_stream.wait_event(ready)
    long_stream.wait_event(ready)
    with torch.cuda.stream(short_stream):
        _launch(
            short_case,
            tiles_per_program=2,
            logits=short_actual,
        )
    with torch.cuda.stream(long_stream):
        _launch(
            long_case,
            tiles_per_program=1,
            logits=long_actual,
        )
    short_stream.synchronize()
    long_stream.synchronize()

    _assert_logits_match(short_actual, short_expected)
    _assert_logits_match(long_actual, long_expected)


@pytest.mark.parametrize(
    ("seq_len_sum", "tiles_per_program"),
    [(509, 2), (2113, 1)],
)
def test_scaled_residual_prefill_supports_cuda_graph_replay(
    seq_len_sum: int,
    tiles_per_program: int,
) -> None:
    if not hasattr(torch.cuda, "CUDAGraph"):
        pytest.skip("CUDA graph capture is not available")
    case = _make_random_case(
        tokens=2,
        seq_len_sum=seq_len_sum,
        row_starts=[3, 71],
        row_ends=[seq_len_sum, seq_len_sum - 5],
        seed=6600 + tiles_per_program,
    )
    expected = _reference_prefill_logits(case)
    captured = torch.full_like(expected, float("nan"))
    query_fp8_scratch = torch.empty(
        (case.q.shape[0], 2, _HEADS, _HEAD_DIM),
        device=case.q.device,
        dtype=torch.float8_e4m3fn,
    )
    scaled_weights_scratch = torch.empty(
        (case.q.shape[0], _HEADS),
        device=case.q.device,
        dtype=torch.float32,
    )

    _launch(
        case,
        tiles_per_program=tiles_per_program,
        logits=captured,
        query_fp8_scratch=query_fp8_scratch,
        scaled_weights_scratch=scaled_weights_scratch,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(
            case,
            tiles_per_program=tiles_per_program,
            logits=captured,
            query_fp8_scratch=query_fp8_scratch,
            scaled_weights_scratch=scaled_weights_scratch,
        )
    captured.fill_(float("nan"))
    query_fp8_scratch.fill_(float("nan"))
    scaled_weights_scratch.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_logits_match(captured, expected)
