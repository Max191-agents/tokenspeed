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

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip("AMD GFX950 is required for Gluon DSA tests", allow_module_level=True)


from tokenspeed_kernel_amd.ops.attention.gluon import (  # noqa: E402
    dsa_topk_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_gfx950 import (  # noqa: E402
    _check_packed_fp8_inputs,
)

_HEADS = 32
_HEAD_DIM = 128
_PAGE_SIZE = 64
_ROW_BYTES = _HEAD_DIM + 4
_SOFTMAX_SCALE = _HEAD_DIM**-0.5


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def _make_inputs(
    tokens: int,
    num_pages: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    num_slots = num_pages * _PAGE_SIZE
    row_amplitudes = torch.linspace(
        0.05,
        1.25,
        num_slots,
        device="cuda",
        dtype=torch.float32,
    )
    index_k = torch.randn(
        (num_slots, _HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    index_k = (index_k * row_amplitudes[:, None]).to(torch.bfloat16)
    return q.contiguous(), weights.contiguous(), _pack_index_k_cache(index_k)


def _pack_index_k_cache(index_k: torch.Tensor) -> torch.Tensor:
    num_slots = int(index_k.shape[0])
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
    values = fp8_values.float() * scales
    return values.reshape(num_slots, _HEAD_DIM), scales.reshape(num_slots)


def _reference_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
) -> torch.Tensor:
    per_head = index_k.float() @ q.float().transpose(0, 1)
    return (torch.relu(per_head) * weights.float()).sum(dim=1) * _SOFTMAX_SCALE


def _reference_decode_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    q_len_per_req: int,
) -> torch.Tensor:
    max_seq_len = int(block_table.shape[1]) * _PAGE_SIZE
    logits = torch.full(
        (q.shape[0], max_seq_len),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for token in range(q.shape[0]):
        req = token // q_len_per_req
        q_offset = token % q_len_per_req
        seq_len = max(
            int(seq_lens[req].item()) - (q_len_per_req - 1) + q_offset,
            0,
        )
        offsets = torch.arange(seq_len, device=q.device, dtype=torch.long)
        pages = block_table[req].long().index_select(0, offsets // _PAGE_SIZE)
        slots = pages * _PAGE_SIZE + offsets.remainder(_PAGE_SIZE)
        logits[token, :seq_len] = _reference_scores(
            q[token], weights[token], index_k.index_select(0, slots)
        )
    return logits


def _reference_prefill_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
) -> torch.Tensor:
    seq_len_sum = int(kv_workspace_slots.numel())
    logits = torch.full(
        (q.shape[0], seq_len_sum),
        -float("inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for token in range(q.shape[0]):
        row_start = int(row_starts[token].item())
        row_end = int(row_ends[token].item())
        rows = torch.arange(row_start, row_end, device=q.device, dtype=torch.long)
        slots = kv_workspace_slots.index_select(0, rows).long()
        logits[token, row_start:row_end] = _reference_scores(
            q[token], weights[token], index_k.index_select(0, slots)
        )
    return logits


def _assert_model_contract_inputs(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    assert q.dtype == torch.bfloat16
    assert weights.dtype == torch.float32
    assert bool((weights < 0).any())
    assert bool((weights > 0).any())
    assert float(scales.min().item()) > 0.0
    assert float(scales.max().item()) > 4.0 * float(scales.min().item())
    sample_per_head = index_k[:32].float() @ q[0].float().transpose(0, 1)
    assert bool((sample_per_head < 0).any())
    assert bool((sample_per_head > 0).any())


def _assert_logits_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    expected_mask = torch.isneginf(expected)
    assert torch.equal(torch.isneginf(actual), expected_mask)
    assert not bool(torch.isnan(actual).any())
    torch.testing.assert_close(
        actual[~expected_mask],
        expected[~expected_mask],
        rtol=3.0e-3,
        atol=3.0e-4,
    )


def _launch_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    packed_index_k: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    q_len_per_req: int,
    *,
    logits: torch.Tensor | None = None,
) -> torch.Tensor:
    row_bytes = _check_packed_fp8_inputs(q, packed_index_k, weights, _PAGE_SIZE)
    if logits is None:
        logits = torch.empty(
            (q.shape[0], block_table.shape[1] * _PAGE_SIZE),
            device=q.device,
            dtype=torch.float32,
        )
    return dsa_topk_gfx950._launch_dsa_decode_logits_fp8(
        q,
        packed_index_k,
        weights,
        seq_lens,
        block_table,
        logits,
        page_size=_PAGE_SIZE,
        row_bytes=row_bytes,
        softmax_scale=_SOFTMAX_SCALE,
        q_len_per_req=q_len_per_req,
    )


def _launch_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    packed_index_k: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
) -> torch.Tensor:
    row_bytes = _check_packed_fp8_inputs(q, packed_index_k, weights, _PAGE_SIZE)
    if logits is None:
        logits = torch.empty(
            (q.shape[0], kv_workspace_slots.numel()),
            device=q.device,
            dtype=torch.float32,
        )
    return dsa_topk_gfx950._launch_dsa_prefill_logits_fp8(
        q,
        packed_index_k,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        logits,
        page_size=_PAGE_SIZE,
        row_bytes=row_bytes,
        softmax_scale=_SOFTMAX_SCALE,
    )


@pytest.mark.parametrize("q_len_per_req", range(1, 7))
def test_decode_logits_match_reference_for_ragged_permuted_pages(
    q_len_per_req: int,
) -> None:
    block_table = torch.tensor(
        [[7, 2, 11, 0], [5, 9, 1, 12]],
        device="cuda",
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([173, 130], device="cuda", dtype=torch.int32)
    q, weights, packed_index_k = _make_inputs(
        2 * q_len_per_req, 13, seed=1000 + q_len_per_req
    )
    index_k, scales = _unpack_index_k_cache(packed_index_k)
    _assert_model_contract_inputs(q, weights, index_k, scales)

    actual = _launch_decode(
        q,
        weights,
        packed_index_k,
        seq_lens,
        block_table,
        q_len_per_req,
    )
    expected = _reference_decode_logits(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        q_len_per_req,
    )

    _assert_logits_match(actual, expected)


def test_prefill_logits_match_reference_on_nondefault_stream() -> None:
    tokens = 64
    seq_len_sum = 197
    num_pages = 8
    q, weights, packed_index_k = _make_inputs(tokens, num_pages, seed=2001)
    index_k, scales = _unpack_index_k_cache(packed_index_k)
    generator = _generator(2002)
    kv_workspace_slots = torch.randperm(
        num_pages * _PAGE_SIZE,
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )[:seq_len_sum].contiguous()
    token_offsets = torch.arange(tokens, device="cuda", dtype=torch.int32)
    row_starts = 3 + (token_offsets * 17).remainder(73)
    widths = 1 + (token_offsets * 29).remainder(121)
    row_ends = torch.minimum(
        row_starts + widths,
        torch.full_like(row_starts, seq_len_sum),
    )
    row_starts[-4:] = torch.tensor([127, 129, 191, 5], device="cuda")
    row_ends[-4:] = torch.tensor([197, 194, 197, 6], device="cuda")
    _assert_model_contract_inputs(q, weights, index_k, scales)
    assert not torch.equal(
        kv_workspace_slots,
        torch.sort(kv_workspace_slots).values,
    )
    assert bool((row_starts.remainder(_PAGE_SIZE) != 0).any())
    assert bool((row_ends == seq_len_sum).any())

    actual = torch.empty((tokens, seq_len_sum), device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch_prefill(
            q,
            weights,
            packed_index_k,
            kv_workspace_slots,
            row_starts,
            row_ends,
            logits=actual,
        )
    stream.synchronize()
    expected = _reference_prefill_logits(
        q,
        weights,
        index_k,
        kv_workspace_slots,
        row_starts,
        row_ends,
    )

    _assert_logits_match(actual, expected)


def test_signed_glm52_logits_and_topk_have_stable_ranks() -> None:
    q_len_per_req = 3
    topk = 32
    block_table = torch.tensor(
        [[7, 2, 11, 0], [5, 9, 1, 12]],
        device="cuda",
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([173, 130], device="cuda", dtype=torch.int32)
    q, weights, packed_index_k = _make_inputs(6, 13, seed=2501)
    index_k, scales = _unpack_index_k_cache(packed_index_k)
    _assert_model_contract_inputs(q, weights, index_k, scales)
    logits = _launch_decode(
        q,
        weights,
        packed_index_k,
        seq_lens,
        block_table,
        q_len_per_req,
    )
    expected = _reference_decode_logits(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        q_len_per_req,
    )
    _assert_logits_match(logits, expected)

    topk_slots, topk_lens = dsa_topk_gfx950.gluon_dsa_decode_topk_fp8_gfx950(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=_PAGE_SIZE,
        topk=topk,
        softmax_scale=_SOFTMAX_SCALE,
        q_len_per_req=q_len_per_req,
        index_k_cache=packed_index_k,
    )
    for token in range(q.shape[0]):
        req = token // q_len_per_req
        seq_len = int(seq_lens[req].item()) - (q_len_per_req - 1)
        seq_len += token % q_len_per_req
        assert int(topk_lens[token].item()) == topk
        offsets = torch.arange(seq_len, device="cuda", dtype=torch.long)
        pages = block_table[req].long().index_select(0, offsets // _PAGE_SIZE)
        candidate_slots = pages * _PAGE_SIZE + offsets.remainder(_PAGE_SIZE)
        selected = topk_slots[token, :topk]
        assert torch.unique(selected).numel() == topk
        matches = selected[:, None] == candidate_slots[None, :]
        assert bool(matches.any(dim=1).all())
        selected_offsets = matches.to(torch.int32).argmax(dim=1)
        selected_scores = logits[token].index_select(0, selected_offsets)
        expected_scores = torch.topk(logits[token, :seq_len], topk).values
        torch.testing.assert_close(
            torch.sort(selected_scores, descending=True).values,
            expected_scores,
            rtol=0.0,
            atol=0.0,
        )


def test_decode_logits_support_cuda_graph_replay() -> None:
    q_len_per_req = 2
    block_table = torch.tensor([[4, 1, 7], [2, 6, 0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([131, 95], device="cuda", dtype=torch.int32)
    q, weights, packed_index_k = _make_inputs(4, 8, seed=3001)
    index_k, _ = _unpack_index_k_cache(packed_index_k)
    expected = _reference_decode_logits(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        q_len_per_req,
    )
    captured_logits = torch.empty_like(expected)

    _launch_decode(
        q,
        weights,
        packed_index_k,
        seq_lens,
        block_table,
        q_len_per_req,
        logits=captured_logits,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_decode(
            q,
            weights,
            packed_index_k,
            seq_lens,
            block_table,
            q_len_per_req,
            logits=captured_logits,
        )
    graph.replay()
    torch.cuda.synchronize()

    _assert_logits_match(captured_logits, expected)
