# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import importlib.metadata
import inspect
import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip("AMD GFX950 is required for Gluon DSA tests", allow_module_level=True)


from tokenspeed_kernel_amd.ops.attention.gluon import dsa_topk_gfx950  # noqa: E402
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_gfx950 import (  # noqa: E402
    _trim_topk_slots_for_context,
    gluon_dsa_decode_gfx950,
    gluon_dsa_prefill_gfx950,
)
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_topk_gfx950 import (  # noqa: E402
    gluon_dsa_decode_topk_fp8_gfx950,
    gluon_dsa_prefill_topk_fp8_gfx950,
)

torch.manual_seed(42)


_COMPILED_RUNNER_CACHES = (
    dsa_topk_gfx950._compiled_runner_cache,
    dsa_topk_gfx950._trivial_decode_runner_plans,
    dsa_topk_gfx950._runtime_decode_runner_plans,
    dsa_topk_gfx950._manual_decode_runner_plans,
    dsa_topk_gfx950._trivial_prefill_runner_plans,
    dsa_topk_gfx950._runtime_prefill_runner_plans,
    dsa_topk_gfx950._manual_prefill_runner_plans,
    dsa_topk_gfx950._persistent_runner_plans,
)


def _clear_compiled_runner_caches() -> None:
    with dsa_topk_gfx950._compiled_runner_cache_lock:
        for cache in _COMPILED_RUNNER_CACHES:
            cache.clear()
        dsa_topk_gfx950._compiled_runner_environments.clear()


@pytest.fixture
def isolated_compiled_runner_caches():
    _clear_compiled_runner_caches()
    yield
    _clear_compiled_runner_caches()


@pytest.mark.parametrize(
    ("max_seqlen_k", "expected_topk"),
    ((25, 512), (608, 1024), (1537, 2048)),
)
def test_dsa_attention_trims_topk_to_registered_context_width(
    max_seqlen_k: int,
    expected_topk: int,
) -> None:
    topk_slots = torch.arange(2 * 2048, device="cuda", dtype=torch.int32).reshape(
        2, 2048
    )

    trimmed = _trim_topk_slots_for_context(topk_slots, max_seqlen_k)

    assert trimmed.shape == (2, expected_topk)
    torch.testing.assert_close(trimmed, topk_slots[:, :expected_topk])


@dataclass(frozen=True)
class _TopKDecodeCase:
    name: str
    seq_lens: tuple[int, ...]
    index_heads: int
    topk: int
    q_len_per_req: int
    seed: int


@dataclass(frozen=True)
class _TopKPrefillCase:
    name: str
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]
    index_heads: int
    topk: int
    seed: int


@dataclass(frozen=True)
class _DSACase:
    name: str
    mode: str
    kv_layout: str
    topk: int
    seed: int
    num_heads: int = 8
    qk_nope_head_dim: int = 192
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    q_len_per_req: int = 1
    visible_lens: tuple[int, ...] | None = None
    topk_lens: tuple[int, ...] | None = None
    prefix_lens: tuple[int, ...] | None = None
    extend_lens: tuple[int, ...] | None = None


_GLM52_TOPK_DECODE_CASES = (
    _TopKDecodeCase(
        "decode_batch_mixed_512",
        seq_lens=(128, 257, 511, 1024),
        index_heads=2,
        topk=512,
        q_len_per_req=1,
        seed=101,
    ),
    _TopKDecodeCase(
        "decode_q3_boundary_512",
        seq_lens=(510, 511, 512, 1022, 1023, 1024),
        index_heads=2,
        topk=512,
        q_len_per_req=3,
        seed=102,
    ),
    _TopKDecodeCase(
        "decode_long_1024",
        seq_lens=(2048, 3072, 4096),
        index_heads=4,
        topk=1024,
        q_len_per_req=1,
        seed=103,
    ),
    _TopKDecodeCase(
        "decode_long_2048",
        seq_lens=(1536, 4096),
        index_heads=2,
        topk=2048,
        q_len_per_req=1,
        seed=104,
    ),
)


_GLM52_TOPK_PREFILL_CASES = (
    _TopKPrefillCase(
        "prefill_short_512",
        prefix_lens=(64, 128),
        extend_lens=(16, 32),
        index_heads=2,
        topk=512,
        seed=201,
    ),
    _TopKPrefillCase(
        "prefill_chunk_512",
        prefix_lens=(512, 1024),
        extend_lens=(32, 32),
        index_heads=2,
        topk=512,
        seed=202,
    ),
    _TopKPrefillCase(
        "prefill_mixed_1024",
        prefix_lens=(256, 1024, 1536),
        extend_lens=(16, 24, 16),
        index_heads=4,
        topk=1024,
        seed=203,
    ),
    _TopKPrefillCase(
        "prefill_long_2048",
        prefix_lens=(1536, 2048),
        extend_lens=(16, 16),
        index_heads=2,
        topk=2048,
        seed=204,
    ),
)


_GLM52_DSA_CASES = (
    _DSACase(
        "decode_sparse_mixed_512",
        mode="decode",
        kv_layout="sparse",
        topk=512,
        visible_lens=(128, 257, 512, 1024),
        topk_lens=(64, 257, 512, 384),
        seed=301,
    ),
    _DSACase(
        "decode_dense_q3_512",
        mode="decode",
        kv_layout="dense",
        topk=512,
        q_len_per_req=3,
        visible_lens=(512, 513, 514, 1024, 1025, 1026),
        topk_lens=(128, 256, 512, 300, 511, 64),
        seed=302,
    ),
    _DSACase(
        "decode_sparse_long_1024",
        mode="decode",
        kv_layout="sparse",
        topk=1024,
        visible_lens=(2048, 3072, 4096),
        topk_lens=(640, 1024, 777),
        seed=303,
    ),
    _DSACase(
        "decode_dense_long_2048",
        mode="decode",
        kv_layout="dense",
        topk=2048,
        visible_lens=(2048, 4096),
        topk_lens=(1536, 2048),
        seed=304,
    ),
    _DSACase(
        "prefill_sparse_short_512",
        mode="prefill",
        kv_layout="sparse",
        topk=512,
        prefix_lens=(64, 128),
        extend_lens=(8, 8),
        topk_lens=(
            32,
            64,
            96,
            128,
            48,
            80,
            112,
            136,
            33,
            65,
            97,
            129,
            49,
            81,
            113,
            136,
        ),
        seed=305,
    ),
    _DSACase(
        "prefill_dense_chunk_512",
        mode="prefill",
        kv_layout="dense",
        topk=512,
        prefix_lens=(512, 1024),
        extend_lens=(8, 8),
        topk_lens=(
            128,
            192,
            256,
            320,
            384,
            448,
            512,
            256,
            96,
            160,
            224,
            288,
            352,
            416,
            480,
            512,
        ),
        seed=306,
    ),
    _DSACase(
        "prefill_sparse_mixed_1024",
        mode="prefill",
        kv_layout="sparse",
        topk=1024,
        prefix_lens=(256, 1024, 1536),
        extend_lens=(4, 6, 4),
        topk_lens=(
            128,
            256,
            384,
            260,
            512,
            640,
            768,
            896,
            1024,
            768,
            512,
            1024,
            768,
            1024,
        ),
        seed=307,
    ),
    _DSACase(
        "prefill_dense_long_2048",
        mode="prefill",
        kv_layout="dense",
        topk=2048,
        prefix_lens=(1536, 2048),
        extend_lens=(4, 4),
        topk_lens=(512, 1024, 1536, 1539, 1024, 1536, 2048, 2048),
        seed=308,
    ),
    _DSACase(
        "prefill_sparse_first_prompt_2048",
        mode="prefill",
        kv_layout="sparse",
        topk=2048,
        num_heads=16,
        prefix_lens=(0,),
        extend_lens=(25,),
        topk_lens=tuple(range(1, 26)),
        seed=309,
    ),
)


def _pack_index_k_cache(
    index_k: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = index_k.shape[1]
    num_groups = head_dim // 128
    row_bytes = head_dim + num_groups * 4
    num_slots = index_k.shape[0]
    num_pages = num_slots // page_size
    packed = torch.empty(
        (num_slots, row_bytes),
        device=index_k.device,
        dtype=torch.uint8,
    )
    x = index_k.float().reshape(num_slots, num_groups, 128)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    x_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    flat = packed.reshape(-1)
    page_bytes = page_size * row_bytes
    fp8_view = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, page_size, head_dim),
        (page_bytes, head_dim, 1),
    )
    scale_view = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, page_size, num_groups),
        (page_bytes // 4, num_groups, 1),
        (page_size * head_dim) // 4,
    )
    fp8_view.copy_(x_fp8.reshape(num_pages, page_size, head_dim))
    scale_view.copy_(scale.reshape(num_pages, page_size, num_groups))
    return packed, (x_fp8.float() * scale).reshape_as(index_k)


def _generator(device: str, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _randn_bf16(
    shape: Sequence[int],
    *,
    device: str,
    generator: torch.Generator,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        * scale
    ).to(torch.bfloat16)


def _normal_weights(
    shape: Sequence[int],
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    logits = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    return torch.softmax(logits, dim=-1).contiguous()


def _round_up_to_page(slots: int, page_size: int) -> int:
    return int(math.ceil(slots / page_size) * page_size)


def _make_decode_block_table(
    seq_lens: Sequence[int],
    page_size: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    max_pages = max(math.ceil(seq_len / page_size) for seq_len in seq_lens)
    pages = torch.arange(
        len(seq_lens) * max_pages, device=device, dtype=torch.int32
    ).reshape(len(seq_lens), max_pages)
    return pages, int(len(seq_lens) * max_pages * page_size)


def _make_prefill_workspace(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[range]]:
    kv_workspace_slots: list[int] = []
    row_starts: list[int] = []
    row_ends: list[int] = []
    visible_ranges: list[range] = []
    cursor = 0
    for prefix_len, extend_len in zip(prefix_lens, extend_lens, strict=True):
        req_start = cursor
        seq_len = int(prefix_len) + int(extend_len)
        kv_workspace_slots.extend(range(req_start, req_start + seq_len))
        for query_offset in range(int(extend_len)):
            visible_end = req_start + int(prefix_len) + query_offset + 1
            row_starts.append(req_start)
            row_ends.append(visible_end)
            visible_ranges.append(range(req_start, visible_end))
        cursor += seq_len

    return (
        torch.tensor(kv_workspace_slots, device=device, dtype=torch.int64),
        torch.tensor(row_starts, device=device, dtype=torch.int32),
        torch.tensor(row_ends, device=device, dtype=torch.int32),
        visible_ranges,
    )


def _index_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    per_head = index_k.float() @ q.float().transpose(0, 1)
    return (per_head * weights.float()).sum(dim=1) * softmax_scale


def _reference_decode_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    lens = torch.empty((q.shape[0],), device=q.device, dtype=torch.int32)
    for token in range(q.shape[0]):
        req = token // int(q_len_per_req)
        q_offset = token - req * int(q_len_per_req)
        seq_len = int(seq_lens[req].item())
        if q_len_per_req != 1:
            seq_len = seq_len - (int(q_len_per_req) - 1) + q_offset
        count = min(seq_len, int(topk))
        lens[token] = count
        if count == 0:
            continue
        offsets = torch.arange(seq_len, device=q.device, dtype=torch.long)
        pages = block_table[req].long().index_select(0, offsets // page_size)
        slots = pages * int(page_size) + offsets.remainder(page_size)
        scores = _index_scores(
            q[token],
            weights[token],
            index_k.index_select(0, slots),
            softmax_scale,
        )
        selected = torch.topk(scores, count).indices
        out[token, :count] = slots.index_select(0, selected).to(torch.int32)
    return out, lens


def _reference_prefill_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    candidate_lens = (row_ends - row_starts).clamp_min(0)
    lens = torch.minimum(candidate_lens, torch.full_like(candidate_lens, int(topk)))
    for token in range(q.shape[0]):
        count = int(lens[token].item())
        if count == 0:
            continue
        rows = torch.arange(
            int(row_starts[token].item()),
            int(row_ends[token].item()),
            device=q.device,
            dtype=torch.long,
        )
        slots = kv_workspace_slots.index_select(0, rows).long()
        scores = _index_scores(
            q[token],
            weights[token],
            index_k.index_select(0, slots),
            softmax_scale,
        )
        selected = torch.topk(scores, count).indices
        out[token, :count] = rows.index_select(0, selected).to(torch.int32)
    return out, lens


def _assert_topk_matches(
    actual: torch.Tensor,
    actual_lens: torch.Tensor,
    expected: torch.Tensor,
    expected_lens: torch.Tensor,
) -> None:
    torch.testing.assert_close(actual_lens.cpu(), expected_lens.cpu())
    for token in range(actual.shape[0]):
        count = int(expected_lens[token].item())
        actual_selected = torch.sort(actual[token, :count].cpu()).values
        expected_selected = torch.sort(expected[token, :count].cpu()).values
        torch.testing.assert_close(actual_selected, expected_selected)
        assert (actual[token, count:] == -1).all()


def _strided_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    backing = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] * 2),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    view = backing[..., ::2]
    view.copy_(tensor)
    return view


def _strided_1d(tensor: torch.Tensor) -> torch.Tensor:
    backing = torch.empty(
        (tensor.shape[0] * 2,),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    view = backing[::2]
    view.copy_(tensor)
    return view


@pytest.mark.parametrize(
    "case",
    _GLM52_TOPK_DECODE_CASES,
    ids=lambda case: case.name,
)
def test_dsa_decode_topk_fp8_glm52_cases(case: _TopKDecodeCase) -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5
    gen = _generator(device, case.seed)
    block_table, num_slots = _make_decode_block_table(case.seq_lens, page_size, device)
    tokens = len(case.seq_lens) * case.q_len_per_req
    q = _randn_bf16(
        (tokens, case.index_heads, head_dim),
        device=device,
        generator=gen,
    )
    weights = _normal_weights((tokens, case.index_heads), device=device, generator=gen)
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )
    seq_lens = torch.tensor(case.seq_lens, device=device, dtype=torch.int32)

    topk_slots, topk_lens = gluon_dsa_decode_topk_fp8_gfx950(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=case.topk,
        softmax_scale=softmax_scale,
        seq_lens_2d=seq_lens.unsqueeze(1).expand(-1, case.q_len_per_req),
        q_len_per_req=case.q_len_per_req,
        index_k_cache=packed_index_k,
    )
    expected_slots, expected_lens = _reference_decode_topk(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=case.topk,
        softmax_scale=softmax_scale,
        q_len_per_req=case.q_len_per_req,
    )

    _assert_topk_matches(topk_slots, topk_lens, expected_slots, expected_lens)


def test_dsa_decode_topk_fp8_accepts_strided_inputs() -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    topk = 512
    softmax_scale = head_dim**-0.5
    gen = _generator(device, 121)
    seq_lens_tuple = (640, 704)
    block_table, num_slots = _make_decode_block_table(seq_lens_tuple, page_size, device)
    tokens = len(seq_lens_tuple)
    q = _strided_last_dim(
        _randn_bf16((tokens, 1, head_dim), device=device, generator=gen)
    )
    weights = _strided_last_dim(
        _normal_weights((tokens, 1), device=device, generator=gen)
    )
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )
    seq_lens = _strided_1d(
        torch.tensor(seq_lens_tuple, device=device, dtype=torch.int32)
    )
    block_table = _strided_last_dim(block_table)
    packed_index_k = _strided_last_dim(packed_index_k)

    topk_slots, topk_lens = gluon_dsa_decode_topk_fp8_gfx950(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        q_len_per_req=1,
        index_k_cache=packed_index_k,
    )
    expected_slots, expected_lens = _reference_decode_topk(
        q,
        weights,
        index_k,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
    )

    _assert_topk_matches(topk_slots, topk_lens, expected_slots, expected_lens)


@pytest.mark.parametrize(
    "case",
    _GLM52_TOPK_PREFILL_CASES,
    ids=lambda case: case.name,
)
def test_dsa_prefill_topk_fp8_glm52_cases(case: _TopKPrefillCase) -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5
    gen = _generator(device, case.seed)
    kv_workspace_slots, row_starts, row_ends, _ = _make_prefill_workspace(
        case.prefix_lens, case.extend_lens, device=device
    )
    num_tokens = int(sum(case.extend_lens))
    num_slots = _round_up_to_page(int(kv_workspace_slots.numel()), page_size)
    q = _randn_bf16(
        (num_tokens, case.index_heads, head_dim),
        device=device,
        generator=gen,
    )
    weights = _normal_weights(
        (num_tokens, case.index_heads), device=device, generator=gen
    )
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )

    workspace_indices, topk_lens = gluon_dsa_prefill_topk_fp8_gfx950(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=case.topk,
        softmax_scale=softmax_scale,
        index_k_cache=packed_index_k,
        page_size=page_size,
    )
    expected_indices, expected_lens = _reference_prefill_topk(
        q,
        weights,
        index_k,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=case.topk,
        softmax_scale=softmax_scale,
    )

    _assert_topk_matches(workspace_indices, topk_lens, expected_indices, expected_lens)


def test_dsa_prefill_topk_fp8_accepts_strided_inputs() -> None:
    device = "cuda"
    page_size = 64
    head_dim = 128
    topk = 512
    softmax_scale = head_dim**-0.5
    gen = _generator(device, 221)
    kv_workspace_slots, row_starts, row_ends, _ = _make_prefill_workspace(
        (640,), (2,), device=device
    )
    num_tokens = int(row_starts.numel())
    num_slots = _round_up_to_page(int(kv_workspace_slots.numel()), page_size)
    q = _strided_last_dim(
        _randn_bf16((num_tokens, 1, head_dim), device=device, generator=gen)
    )
    weights = _strided_last_dim(
        _normal_weights((num_tokens, 1), device=device, generator=gen)
    )
    packed_index_k, index_k = _pack_index_k_cache(
        _randn_bf16((num_slots, head_dim), device=device, generator=gen),
        page_size,
    )
    kv_workspace_slots = _strided_1d(kv_workspace_slots)
    row_starts = _strided_1d(row_starts)
    row_ends = _strided_1d(row_ends)
    packed_index_k = _strided_last_dim(packed_index_k)

    workspace_indices, topk_lens = gluon_dsa_prefill_topk_fp8_gfx950(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=softmax_scale,
        index_k_cache=packed_index_k,
        page_size=page_size,
    )
    expected_indices, expected_lens = _reference_prefill_topk(
        q,
        weights,
        index_k,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=softmax_scale,
    )

    _assert_topk_matches(workspace_indices, topk_lens, expected_indices, expected_lens)


def test_dsa_prefill_select_topk_keeps_late_values_above_threshold() -> None:
    device = "cuda"
    cols = 16384
    topk = 2048
    logits = torch.full((1, cols), -10.0, device=device, dtype=torch.float32)
    equal_indices = torch.arange(0, 32, device=device, dtype=torch.int32)
    greater_indices = torch.cat(
        (
            torch.arange(4096, 4096 + topk - 2, device=device, dtype=torch.int32),
            torch.tensor([cols - 3], device=device, dtype=torch.int32),
        )
    )
    logits[0, equal_indices.long()] = 1.0
    logits[0, greater_indices.long()] = 2.0
    row_starts = torch.tensor([0], device=device, dtype=torch.int32)
    row_ends = torch.tensor([cols], device=device, dtype=torch.int32)
    out = torch.empty((1, topk), device=device, dtype=torch.int32)
    lens_out = torch.empty((1,), device=device, dtype=torch.int32)

    num_warps = 8
    block_n = dsa_topk_gfx950.triton.next_power_of_2(cols)
    dsa_topk_gfx950._dsa_prefill_select_topk_kernel[(1,)](
        logits,
        row_starts,
        row_ends,
        out,
        lens_out,
        logits.stride(0),
        out.stride(0),
        topk=topk,
        BLOCK_N=block_n,
        LOAD_ELEMS=dsa_topk_gfx950._load_elems(block_n, num_warps),
        TOPK_LOAD_ELEMS=dsa_topk_gfx950._load_elems(topk, num_warps),
        num_warps=num_warps,
    )
    torch.cuda.synchronize()

    selected = out[0, :topk]
    selected_set = set(selected.cpu().tolist())
    assert selected_set.issuperset(set(greater_indices.cpu().tolist()))
    assert len(selected_set.intersection(set(equal_indices.cpu().tolist()))) == 1
    torch.testing.assert_close(lens_out.cpu(), torch.tensor([topk], dtype=torch.int32))


def test_dsa_decode_select_topk_keeps_late_values_above_threshold() -> None:
    device = "cuda"
    page_size = 64
    cols = 16384
    topk = 2048
    logits = torch.full((1, cols), -10.0, device=device, dtype=torch.float32)
    equal_indices = torch.arange(0, 32, device=device, dtype=torch.int32)
    greater_indices = torch.cat(
        (
            torch.arange(4096, 4096 + topk - 2, device=device, dtype=torch.int32),
            torch.tensor([cols - 3], device=device, dtype=torch.int32),
        )
    )
    logits[0, equal_indices.long()] = 1.0
    logits[0, greater_indices.long()] = 2.0
    seq_lens = torch.tensor([cols], device=device, dtype=torch.int32)
    block_table = torch.arange(
        math.ceil(cols / page_size), device=device, dtype=torch.int32
    ).reshape(1, -1)
    out = torch.empty((1, topk), device=device, dtype=torch.int32)
    lens_out = torch.empty((1,), device=device, dtype=torch.int32)

    num_warps = 8
    block_n = dsa_topk_gfx950.triton.next_power_of_2(cols)
    dsa_topk_gfx950._dsa_decode_select_topk_kernel[(1,)](
        logits,
        block_table,
        seq_lens,
        out,
        lens_out,
        logits.stride(0),
        block_table.stride(0),
        out.stride(0),
        block_table.shape[1],
        page_size=page_size,
        topk=topk,
        q_len_per_req=1,
        BLOCK_N=block_n,
        LOAD_ELEMS=dsa_topk_gfx950._load_elems(block_n, num_warps),
        TOPK_LOAD_ELEMS=dsa_topk_gfx950._load_elems(topk, num_warps),
        num_warps=num_warps,
    )
    torch.cuda.synchronize()

    selected = out[0, :topk]
    selected_set = set(selected.cpu().tolist())
    assert selected_set.issuperset(set(greater_indices.cpu().tolist()))
    assert len(selected_set.intersection(set(equal_indices.cpu().tolist()))) == 1
    torch.testing.assert_close(lens_out.cpu(), torch.tensor([topk], dtype=torch.int32))


def test_dsa_decode_topk_gluon_long_row_uses_radix_path() -> None:
    device = "cuda"
    page_size = 64
    seq_len = 65536
    topk = 2048
    head_dim = 128
    q = torch.ones((1, 1, head_dim), device=device, dtype=torch.bfloat16)
    weights = torch.ones((1, 1), device=device, dtype=torch.float32)
    index_k = torch.zeros((seq_len, head_dim), device=device, dtype=torch.bfloat16)
    index_k[:topk].fill_(1.0)
    packed_index_k, _ = _pack_index_k_cache(index_k, page_size)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    block_table = torch.arange(
        seq_len // page_size, device=device, dtype=torch.int32
    ).reshape(1, -1)

    topk_slots, topk_lens = gluon_dsa_decode_topk_fp8_gfx950(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=head_dim**-0.5,
        index_k_cache=packed_index_k,
    )

    expected = torch.arange(topk, device=device, dtype=torch.int32)
    torch.testing.assert_close(topk_lens.cpu(), torch.tensor([topk], dtype=torch.int32))
    torch.testing.assert_close(torch.sort(topk_slots[0]).values.cpu(), expected.cpu())


def test_dsa_prefill_topk_gluon_long_row_uses_radix_path() -> None:
    device = "cuda"
    page_size = 64
    seq_len = 65536
    topk = 2048
    head_dim = 128
    q = torch.ones((1, 1, head_dim), device=device, dtype=torch.bfloat16)
    weights = torch.ones((1, 1), device=device, dtype=torch.float32)
    index_k = torch.zeros((seq_len, head_dim), device=device, dtype=torch.bfloat16)
    index_k[:topk].fill_(1.0)
    packed_index_k, _ = _pack_index_k_cache(index_k, page_size)
    kv_workspace_slots = torch.arange(seq_len, device=device, dtype=torch.int64)
    row_starts = torch.tensor([0], device=device, dtype=torch.int32)
    row_ends = torch.tensor([seq_len], device=device, dtype=torch.int32)

    workspace_indices, topk_lens = gluon_dsa_prefill_topk_fp8_gfx950(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=head_dim**-0.5,
        index_k_cache=packed_index_k,
        page_size=page_size,
    )

    expected = torch.arange(topk, device=device, dtype=torch.int32)
    torch.testing.assert_close(topk_lens.cpu(), torch.tensor([topk], dtype=torch.int32))
    torch.testing.assert_close(
        torch.sort(workspace_indices[0]).values.cpu(),
        expected.cpu(),
    )


def test_dsa_prefill_topk_oneblock_falls_back_for_large_tie_bucket() -> None:
    cols = 65536
    topk = 2048
    logits = torch.zeros((1, cols), device="cuda", dtype=torch.float32)
    row_starts = torch.zeros((1,), device="cuda", dtype=torch.int32)
    row_ends = torch.full((1,), cols, device="cuda", dtype=torch.int32)
    out = torch.empty((1, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    selected = out[0]
    torch.testing.assert_close(lens_out.cpu(), torch.tensor([topk], dtype=torch.int32))
    assert ((selected >= 0) & (selected < cols)).all()
    assert torch.unique(selected).numel() == topk


def _make_grouped_radix_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    cols: int,
    topk: int,
    seed: int = 1907,
) -> torch.Tensor:
    generator = _generator("cuda", seed)
    base_cols = min(cols, 65536)
    base = torch.empty(
        (row_starts.numel(), base_cols),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-1.0, 1.0, generator=generator)
    logits = base.repeat(1, math.ceil(cols / base_cols))[:, :cols].contiguous()
    high_count = topk - 32
    starts = row_starts.cpu().tolist()
    ends = row_ends.cpu().tolist()
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        assert end - start >= topk + 32
        high = torch.arange(start, start + high_count, device="cuda")
        ties = torch.arange(
            start + high_count,
            start + high_count + 64,
            device="cuda",
        )
        logits[row, high] = torch.linspace(4.0, 3.0, high_count, device="cuda")
        logits[row, ties] = 2.0
        neg_inf = torch.arange(start + high_count + 64, end, 257, device="cuda")
        logits[row, neg_inf] = -float("inf")
    return logits


def _assert_grouped_radix_topk(
    logits: torch.Tensor,
    actual: torch.Tensor,
    actual_lens: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
) -> None:
    expected_lens = torch.minimum(
        row_ends - row_starts,
        torch.full_like(row_ends, topk),
    )
    torch.testing.assert_close(actual_lens.cpu(), expected_lens.cpu())
    starts = row_starts.cpu().tolist()
    ends = row_ends.cpu().tolist()
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        count = int(expected_lens[row].item())
        selected = actual[row, :count].long()
        assert int(torch.unique(selected).numel()) == count
        assert bool(((selected >= start) & (selected < end)).all())
        actual_values = torch.sort(logits[row].index_select(0, selected)).values
        expected_values = torch.sort(
            torch.topk(logits[row, start:end], count).values
        ).values
        torch.testing.assert_close(actual_values.cpu(), expected_values.cpu())
        assert bool((actual[row, count:] == -1).all())


def _make_reversed_decode_block_table(
    requests: int,
    cols: int,
    page_size: int,
) -> torch.Tensor:
    pages = math.ceil(cols / page_size)
    logical_to_physical = torch.arange(
        pages - 1,
        -1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    return torch.stack(
        [logical_to_physical + request * pages for request in range(requests)]
    )


def _assert_decode_topk_slots(
    logits: torch.Tensor,
    actual: torch.Tensor,
    actual_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    q_len_per_req: int,
    topk: int,
) -> None:
    rows = logits.shape[0]
    for row in range(rows):
        req, q_offset = divmod(row, q_len_per_req)
        row_end = int(seq_lens[req].item()) - (q_len_per_req - 1) + q_offset
        count = min(max(row_end, 0), topk)
        assert int(actual_lens[row].item()) == count
        selected = actual[row, :count].long()
        inverse_pages = torch.empty(
            (int(block_table.max().item()) + 1,),
            device="cuda",
            dtype=torch.int64,
        )
        inverse_pages[block_table[req].long()] = torch.arange(
            block_table.shape[1],
            device="cuda",
        )
        logical = (
            inverse_pages[selected // page_size] * page_size + selected % page_size
        )
        assert bool(((logical >= 0) & (logical < row_end)).all())
        assert int(torch.unique(logical).numel()) == count
        actual_values = torch.sort(logits[row].index_select(0, logical)).values
        expected_values = torch.sort(
            torch.topk(logits[row, :row_end], count).values
        ).values
        torch.testing.assert_close(actual_values.cpu(), expected_values.cpu())
        assert bool((actual[row, count:] == -1).all())


_MANUAL_DECODE_QUALIFICATION_CONFIGS = (
    pytest.param((8192, 8), id="8192x8-production"),
    pytest.param((10240, 4), id="10240x4"),
    pytest.param((12288, 4), id="12288x4"),
    pytest.param((14336, 4), id="14336x4"),
)


def _make_dispatch_tensor(*shape: int) -> SimpleNamespace:
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    strides.reverse()
    return SimpleNamespace(
        shape=shape,
        device=SimpleNamespace(type="cuda", index=0),
        stride=lambda dim: strides[dim],
    )


def _record_decode_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    cols: int,
) -> tuple[object, tuple[object, ...], tuple[object, ...], dict[str, object]]:
    launches: list[
        tuple[object, tuple[object, ...], tuple[object, ...], dict[str, object]]
    ] = []

    def record_launch(
        kernel,
        grid,
        args,
        pointer_dtypes,
        specialization_key,
        **kwargs,
    ) -> None:
        del pointer_dtypes
        launches.append((kernel, args, specialization_key, kwargs | {"grid": grid}))

    monkeypatch.setattr(dsa_topk_gfx950, "_persistent_decode_groups", lambda *_: None)
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_get_cached_compiled_runner_plan",
        lambda *_: None,
    )
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_launch_warmed_compiled_kernel",
        record_launch,
    )
    rows = 2
    dsa_topk_gfx950._dsa_decode_topk_slots(
        _make_dispatch_tensor(rows, cols),
        _make_dispatch_tensor(rows, math.ceil(cols / 64)),
        _make_dispatch_tensor(rows),
        page_size=64,
        topk=2048,
        q_len_per_req=1,
        out=_make_dispatch_tensor(rows, 2048),
        lens_out=_make_dispatch_tensor(rows),
    )
    assert len(launches) == 1
    return launches[0]


def _record_prefill_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    cols: int,
) -> tuple[object, tuple[object, ...], tuple[object, ...], dict[str, object]]:
    launches: list[
        tuple[object, tuple[object, ...], tuple[object, ...], dict[str, object]]
    ] = []

    def record_launch(
        kernel,
        grid,
        args,
        pointer_dtypes,
        specialization_key,
        **kwargs,
    ) -> None:
        del pointer_dtypes
        launches.append((kernel, args, specialization_key, kwargs | {"grid": grid}))

    monkeypatch.setattr(dsa_topk_gfx950, "_persistent_prefill_groups", lambda *_: None)
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_launch_warmed_compiled_kernel",
        record_launch,
    )
    rows = 2
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        _make_dispatch_tensor(rows, cols),
        _make_dispatch_tensor(rows),
        _make_dispatch_tensor(rows),
        topk=2048,
        out=_make_dispatch_tensor(rows, 2048),
        lens_out=_make_dispatch_tensor(rows),
    )
    assert len(launches) == 1
    return launches[0]


def test_dsa_manual_decode_config_source_contract() -> None:
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_RUNTIME_CONFIG == (8192, 4)
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_LONG_RUNTIME_CONFIG == (8192, 8)
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_MANUAL_CONFIG == (8192, 8)
    assert dsa_topk_gfx950._ONEBLOCK_PREFILL_RADIX_BLOCK_N == 4096
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_EARLY_STOP_MIN_COLS == 65536
    assert dsa_topk_gfx950._ONEBLOCK_RADIX_MAX_COLS == 90000

    params = {
        param.name: param
        for param in dsa_topk_gfx950._dsa_oneblock_manual_radix_topk_kernel.params
    }
    constexpr_tail = (
        "MAX_BUCKETS",
        "BLOCK_N",
        "LOAD_ELEMS",
        "COMPACT_FINAL_BLOCK_N",
        "USE_COMPACT_FINAL",
        "USE_RADIX_EARLY_STOP",
    )
    assert tuple(params)[-len(constexpr_tail) :] == constexpr_tail
    assert all(params[name].is_constexpr for name in constexpr_tail)

    kernel_source = inspect.getsource(
        dsa_topk_gfx950._dsa_oneblock_manual_radix_topk_kernel.fn
    )
    assert "LOAD_ELEMS: gl.constexpr" in kernel_source
    assert (
        "_vector_layout(\n        BLOCK_N,\n        gl.num_warps(),\n        LOAD_ELEMS"
        in kernel_source
    )


@pytest.mark.parametrize("config", _MANUAL_DECODE_QUALIFICATION_CONFIGS)
@pytest.mark.parametrize("cols", (65536, 90000))
def test_dsa_manual_decode_config_dispatches_only_broad_region(
    monkeypatch: pytest.MonkeyPatch,
    config: tuple[int, int],
    cols: int,
) -> None:
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_ONEBLOCK_DECODE_MANUAL_CONFIG",
        config,
    )

    kernel, args, specialization_key, kwargs = _record_decode_dispatch(
        monkeypatch,
        cols,
    )

    assert kernel is dsa_topk_gfx950._dsa_oneblock_manual_radix_topk_kernel
    assert args[19:21] == config
    assert specialization_key == args[11:]
    assert specialization_key[-5:-3] == config
    assert kwargs["dispatch_cache"] is dsa_topk_gfx950._manual_decode_runner_plans
    assert kwargs["dispatch_key"] == (2, specialization_key)
    assert kwargs["native_scalar_count"] == 4
    assert kwargs["num_warps"] == 16
    assert kwargs["grid"] == (2, 1, 1)


@pytest.mark.parametrize(
    ("cols", "expected_config"),
    ((65535, (8192, 4)), (90001, (8192, 8))),
)
def test_dsa_manual_decode_config_does_not_change_runtime_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    cols: int,
    expected_config: tuple[int, int],
) -> None:
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_ONEBLOCK_DECODE_MANUAL_CONFIG",
        (14336, 4),
    )

    kernel, args, specialization_key, kwargs = _record_decode_dispatch(
        monkeypatch,
        cols,
    )

    assert kernel is dsa_topk_gfx950._dsa_runtime_radix_topk_kernel
    assert args[16:18] == expected_config
    assert specialization_key == args[10:]
    assert kwargs["dispatch_cache"] is dsa_topk_gfx950._runtime_decode_runner_plans


def test_dsa_manual_decode_config_does_not_change_prefill_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_ONEBLOCK_DECODE_MANUAL_CONFIG",
        (14336, 4),
    )

    kernel, args, specialization_key, kwargs = _record_prefill_dispatch(
        monkeypatch,
        65536,
    )

    assert kernel is dsa_topk_gfx950._dsa_oneblock_manual_radix_topk_kernel
    assert args[20:22] == (4096, 4)
    assert specialization_key == args[11:]
    assert kwargs["dispatch_cache"] is dsa_topk_gfx950._manual_prefill_runner_plans


def test_dsa_manual_decode_block_and_load_are_distinct_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specialization_keys = []
    for config in ((10240, 4), (12288, 4), (12288, 8), (14336, 4)):
        monkeypatch.setattr(
            dsa_topk_gfx950,
            "_ONEBLOCK_DECODE_MANUAL_CONFIG",
            config,
        )
        _, _, specialization_key, _ = _record_decode_dispatch(monkeypatch, 65536)
        assert specialization_key[-5:-3] == config
        specialization_keys.append(specialization_key)

    assert len(set(specialization_keys)) == len(specialization_keys)


@pytest.mark.parametrize(
    ("rows", "cols", "expected_groups"),
    (
        (32, 131072, 2),
        (64, 131072, 2),
        (64, 147493, 2),
        (64, 262144, 4),
        (64, 524288, 4),
        (64, 1048576, 4),
    ),
)
def test_dsa_prefill_topk_dispatches_persistent_groups_across_rows(
    rows: int,
    cols: int,
    expected_groups: int,
) -> None:
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 17
    row_ends = cols - (rows - 1 - row_ids) * 31
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    groups = dsa_topk_gfx950._persistent_prefill_groups(
        rows,
        cols,
        topk,
        logits.device,
    )
    assert groups == expected_groups
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )
    workspace = dsa_topk_gfx950._persistent_topk_workspace(rows, logits.device)
    assert all(int(torch.count_nonzero(t).item()) == 0 for t in workspace)


def test_dsa_persistent_prefill_topk_repeats_across_rows() -> None:
    rows = 64
    cols = 262144
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 17
    row_ends = cols - (rows - 1 - row_ids) * 31
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    tiles = dsa_topk_gfx950.triton.cdiv(cols, dsa_topk_gfx950._RADIX_TOPK_BLOCK_N)
    groups = dsa_topk_gfx950._radix_groups_per_row(rows, tiles, logits.device)
    assert groups < tiles
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )
    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )
    out.fill_(-1)
    lens_out.fill_(-1)
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )


def test_dsa_persistent_prefill_handles_oversized_ties_and_infinities() -> None:
    rows = 32
    cols = 131072
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 7
    row_ends = cols - (rows - 1 - row_ids) * 11
    logits = torch.zeros((rows, cols), device="cuda", dtype=torch.float32)
    starts = row_starts.cpu().tolist()
    ends = row_ends.cpu().tolist()
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if row == 0:
            logits[row, start:end] = float("inf")
        elif row == 1:
            logits[row, start:end] = -float("inf")
        else:
            logits[row, start : start + 37] = float("inf")
            logits[row, start + 37 : end : 257] = -float("inf")
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    assert (
        dsa_topk_gfx950._persistent_prefill_groups(
            rows,
            cols,
            topk,
            logits.device,
        )
        == 2
    )
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )
    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )
    out.fill_(-1)
    lens_out.fill_(-1)
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )


def test_dsa_persistent_prefill_topk_handles_ragged_rows() -> None:
    rows = 64
    cols = 131072
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 17
    row_ends = cols - (rows - 1 - row_ids) * 31
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )
    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )


def test_dsa_prefill_topk_90k_boundary_keeps_exact_values() -> None:
    rows = 4
    cols = 90000
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 19
    row_ends = cols - (rows - 1 - row_ids) * 29
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )


def test_dsa_persistent_prefill_groups_obey_residency_bound() -> None:
    device = torch.device("cuda")
    compute_units = torch.cuda.get_device_properties(device).multi_processor_count
    topk = 2048
    four_group_rows = compute_units // 4

    assert (
        dsa_topk_gfx950._persistent_prefill_groups(
            four_group_rows,
            262144,
            topk,
            device,
        )
        == 4
    )
    assert (
        dsa_topk_gfx950._persistent_prefill_groups(
            four_group_rows + 1,
            262144,
            topk,
            device,
        )
        == 2
    )
    assert (
        dsa_topk_gfx950._persistent_prefill_groups(
            compute_units // 2 + 1,
            262144,
            topk,
            device,
        )
        is None
    )
    assert (
        dsa_topk_gfx950._persistent_prefill_groups(
            dsa_topk_gfx950._PERSISTENT_PREFILL_MIN_ROWS - 1,
            131072,
            topk,
            device,
        )
        is None
    )


def test_dsa_decode_topk_dispatches_persistent_radix_for_batched_queries(
    isolated_compiled_runner_caches,
) -> None:
    page_size = 64
    q_len_per_req = 4
    requests = 16
    rows = requests * q_len_per_req
    cols = 131072
    topk = 2048
    seq_lens = cols - torch.arange(requests, device="cuda", dtype=torch.int32) * 53
    q_offsets = torch.arange(q_len_per_req, device="cuda", dtype=torch.int32)
    row_ends = (seq_lens[:, None] - (q_len_per_req - 1) + q_offsets[None, :]).reshape(
        -1
    )
    row_starts = torch.zeros((rows,), device="cuda", dtype=torch.int32)
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    block_table = torch.arange(
        math.ceil(cols / page_size),
        device="cuda",
        dtype=torch.int32,
    ).repeat(requests, 1)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )
    assert len(dsa_topk_gfx950._persistent_runner_plans) == 1
    plan = next(iter(dsa_topk_gfx950._persistent_runner_plans.values()))
    assert len(plan.pointer_refs) == 11


@pytest.mark.parametrize("cols", (512, 1024, 2048))
def test_dsa_decode_topk_trivial_2048_maps_grouped_queries_to_physical_slots(
    cols: int,
) -> None:
    page_size = 64
    q_len_per_req = 4
    requests = 2
    rows = requests * q_len_per_req
    topk = 2048
    pages = math.ceil(cols / page_size)
    seq_lens = torch.tensor(
        [cols, cols - 53],
        device="cuda",
        dtype=torch.int32,
    )
    logical_pages = torch.arange(pages, device="cuda", dtype=torch.int32)
    block_table = torch.stack(
        (
            4096 + logical_pages.flip(0),
            8192 + logical_pages.flip(0),
        )
    )
    logits = torch.zeros((rows, cols), device="cuda", dtype=torch.float32)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )

    expected = torch.full_like(out, -1)
    expected_lens = torch.empty_like(lens_out)
    for row in range(rows):
        req, q_offset = divmod(row, q_len_per_req)
        candidate_len = int(seq_lens[req].item()) - (q_len_per_req - 1) + q_offset
        offsets = torch.arange(candidate_len, device="cuda", dtype=torch.int32)
        physical_pages = block_table[req].index_select(0, offsets // page_size)
        expected[row, :candidate_len] = physical_pages * page_size + offsets % page_size
        expected_lens[row] = candidate_len

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(lens_out, expected_lens, rtol=0, atol=0)


def _make_trivial_decode_inputs(
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pages = math.ceil(cols / 64)
    logits = torch.empty((1, cols), device="cuda", dtype=torch.float32)
    block_table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
    seq_lens = torch.tensor([cols], device="cuda", dtype=torch.int32)
    out = torch.empty((1, 2048), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)
    return logits, block_table, seq_lens, out, lens_out


def _run_trivial_decode(
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    logits, block_table, seq_lens, out, lens_out = inputs
    dsa_topk_gfx950._dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=64,
        topk=2048,
        q_len_per_req=1,
        out=out,
        lens_out=lens_out,
    )


def _assert_trivial_decode(
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    logits, _, _, out, lens_out = inputs
    cols = logits.shape[1]
    torch.testing.assert_close(
        out[0, :cols],
        torch.arange(cols, device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert (out[0, cols:] == -1).all()
    assert lens_out.item() == cols


def _make_trivial_prefill_inputs(
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.empty((1, cols), device="cuda", dtype=torch.float32)
    row_starts = torch.zeros((1,), device="cuda", dtype=torch.int32)
    row_ends = torch.tensor([cols], device="cuda", dtype=torch.int32)
    out = torch.empty((1, 2048), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)
    return logits, row_starts, row_ends, out, lens_out


def _run_trivial_prefill(
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    logits, row_starts, row_ends, out, lens_out = inputs
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=2048,
        out=out,
        lens_out=lens_out,
    )


def _assert_trivial_prefill(
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    logits, row_starts, _, out, lens_out = inputs
    cols = logits.shape[1]
    start = int(row_starts.item())
    torch.testing.assert_close(
        out[0, :cols],
        torch.arange(start, start + cols, device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert (out[0, cols:] == -1).all()
    assert lens_out.item() == cols


def test_compiled_runner_abi_gate_matches_installed_distribution() -> None:
    assert dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED == (
        importlib.metadata.version("tokenspeed-triton") == "3.8.10.post20260709"
    )


def _make_fake_raw_compiled_plan(
    *,
    global_scratch_size: int = 0,
    profile_scratch_size: int = 0,
    descriptor_wrapper: bool = False,
    jit_kernel: object | None = None,
) -> tuple[
    dsa_topk_gfx950._CompiledRunnerPlan,
    object,
    list[tuple[object, ...]],
    list[tuple[object, ...]],
    list[int],
    object,
]:
    direct_launches: list[tuple[object, ...]] = []
    ordinary_launches: list[tuple[object, ...]] = []
    stream_devices: list[int] = []

    def direct_launch(*args: object) -> None:
        direct_launches.append(args)

    def descriptor_launch(*args: object) -> None:
        direct_launch(*args)

    def get_current_stream(device: int) -> int:
        stream_devices.append(device)
        return 707

    def unexpected_current_device() -> int:
        raise AssertionError("raw launch queried the current device")

    driver = SimpleNamespace(
        utils=SimpleNamespace(launch=direct_launch),
        get_current_stream=get_current_stream,
        get_current_device=unexpected_current_device,
    )
    launcher = SimpleNamespace(
        global_scratch_size=global_scratch_size,
        profile_scratch_size=profile_scratch_size,
        launch=descriptor_launch if descriptor_wrapper else direct_launch,
        launch_cooperative_grid=False,
        warp_size=64,
        arg_annotations=object(),
        kernel_signature=b"fake-signature",
    )
    if jit_kernel is None:
        jit_kernel = SimpleNamespace(launch_metadata=None)
    compiled = SimpleNamespace(
        run=launcher,
        function=object(),
        packed_metadata=(8, 1, 0),
        src=SimpleNamespace(fn=jit_kernel),
    )

    def ordinary_runner(*args: object) -> None:
        ordinary_launches.append(args)

    plan = dsa_topk_gfx950._CompiledRunnerPlan(
        ordinary_runner,
        driver,
        7,
        (),
        object(),
        ("hip", "gfx950", 64),
        dsa_topk_gfx950._COMPILED_RUNNER_CANONICAL_KNOBS,
        (),
        compiled=compiled,
        grid=(2, 3, 1),
    )
    return (
        plan,
        compiled,
        direct_launches,
        ordinary_launches,
        stream_devices,
        direct_launch,
    )


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_compiled_runner_raw_plan_calls_generated_launcher() -> None:
    (
        plan,
        compiled,
        direct_launches,
        ordinary_launches,
        stream_devices,
        direct_launch,
    ) = _make_fake_raw_compiled_plan()

    plan.runner("pointer", 17)

    assert plan.compiled is compiled
    assert plan.raw_launch is direct_launch
    assert not ordinary_launches
    assert stream_devices == [7]
    assert len(direct_launches) == 1
    launch = direct_launches[0]
    assert launch[:6] == (False, 2, 3, 1, 707, compiled.function)
    assert launch[6:8] == (None, None)
    assert launch[8] is compiled.packed_metadata
    assert launch[9:12] == (None, None, None)
    assert launch[12] == 64
    assert launch[13] is compiled.run.arg_annotations
    assert launch[14] == b"fake-signature"
    assert launch[15] == ("pointer", 17)


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
@pytest.mark.parametrize(
    "unsupported_gate",
    (
        "distribution_version",
        "global_scratch",
        "profile_scratch",
        "descriptor",
        "launch_metadata",
        "missing_launch_metadata",
    ),
)
def test_compiled_runner_raw_plan_falls_back_for_unsupported_launcher(
    unsupported_gate: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {}
    if unsupported_gate == "distribution_version":
        monkeypatch.setattr(
            dsa_topk_gfx950,
            "_COMPILED_RUNNER_ABI_SUPPORTED",
            False,
        )
    elif unsupported_gate == "global_scratch":
        options["global_scratch_size"] = 16
    elif unsupported_gate == "profile_scratch":
        options["profile_scratch_size"] = 16
    elif unsupported_gate == "launch_metadata":
        options["jit_kernel"] = SimpleNamespace(launch_metadata=lambda *_args: None)
    elif unsupported_gate == "missing_launch_metadata":
        options["jit_kernel"] = object()
    else:
        options["descriptor_wrapper"] = True

    plan, _, direct_launches, ordinary_launches, _, _ = _make_fake_raw_compiled_plan(
        **options
    )
    plan.runner("pointer")

    assert plan.raw_launch is None
    assert plan.runner is plan.ordinary_runner
    assert not direct_launches
    assert ordinary_launches == [("pointer",)]


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_compiled_runner_real_jit_launch_metadata_forces_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = dsa_topk_gfx950._dsa_trivial_decode_topk2048_kernel
    assert kernel.launch_metadata is None

    raw_plan, _, direct_launches, ordinary_launches, _, direct_launch = (
        _make_fake_raw_compiled_plan(jit_kernel=kernel)
    )
    raw_plan.runner("pointer")

    assert raw_plan.raw_launch is direct_launch
    assert len(direct_launches) == 1
    assert not ordinary_launches

    monkeypatch.setattr(kernel, "launch_metadata", lambda *_args: None)

    plan, _, direct_launches, ordinary_launches, _, _ = _make_fake_raw_compiled_plan(
        jit_kernel=kernel
    )
    plan.runner("pointer")

    assert plan.raw_launch is None
    assert plan.runner is plan.ordinary_runner
    assert not direct_launches
    assert ordinary_launches == [("pointer",)]


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_compiled_runner_reuses_trivial_plan_for_native_scalars(
    isolated_compiled_runner_caches,
) -> None:
    short_inputs = _make_trivial_decode_inputs(512)
    _run_trivial_decode(short_inputs)
    assert len(dsa_topk_gfx950._trivial_decode_runner_plans) == 1
    plan = next(iter(dsa_topk_gfx950._trivial_decode_runner_plans.values()))

    long_inputs = _make_trivial_decode_inputs(2048)
    _run_trivial_decode(long_inputs)

    assert len(dsa_topk_gfx950._trivial_decode_runner_plans) == 1
    assert next(iter(dsa_topk_gfx950._trivial_decode_runner_plans.values())) is plan
    _assert_trivial_decode(short_inputs)
    _assert_trivial_decode(long_inputs)


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_compiled_runner_uses_trivial_prefill_direct_path(
    isolated_compiled_runner_caches,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs_by_cols = tuple(
        _make_trivial_prefill_inputs(cols) for cols in (512, 1024, 2048)
    )
    plan = None
    for inputs in inputs_by_cols:
        _run_trivial_prefill(inputs)
        _assert_trivial_prefill(inputs)
        assert len(dsa_topk_gfx950._trivial_prefill_runner_plans) == 1
        current_plan = next(
            iter(dsa_topk_gfx950._trivial_prefill_runner_plans.values())
        )
        if plan is None:
            plan = current_plan
        else:
            assert current_plan is plan

    assert plan is not None
    assert plan.compiled is not None
    assert plan.raw_launch is plan.driver.utils.launch

    def fail_slow_path(*args, **kwargs) -> None:
        raise AssertionError("same-pointer trivial prefill missed the direct plan")

    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_launch_warmed_compiled_kernel",
        fail_slow_path,
    )
    inputs_by_cols[-1][3].fill_(-7)
    _run_trivial_prefill(inputs_by_cols[-1])
    _assert_trivial_prefill(inputs_by_cols[-1])


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_trivial_prefill_direct_runner_uses_current_stream_and_hooks(
    isolated_compiled_runner_caches,
) -> None:
    inputs = _make_trivial_prefill_inputs(2048)
    _run_trivial_prefill(inputs)
    assert dsa_topk_gfx950._trivial_prefill_runner_plans

    streams: list[int] = []

    def record_stream(metadata) -> None:
        streams.append(metadata.get()["stream"])

    hook = dsa_topk_gfx950.triton.knobs.runtime.launch_enter_hook
    current_stream = torch.cuda.current_stream()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    hook.add(record_stream)
    try:
        stream_a.wait_stream(current_stream)
        with torch.cuda.stream(stream_a):
            _run_trivial_prefill(inputs)
        event = stream_a.record_event()
        stream_b.wait_event(event)
        with torch.cuda.stream(stream_b):
            _run_trivial_prefill(inputs)
        current_stream.wait_stream(stream_b)
    finally:
        hook.remove(record_stream)

    assert streams == [stream_a.cuda_stream, stream_b.cuda_stream]
    _assert_trivial_prefill(inputs)


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_trivial_prefill_direct_runner_graph_replay(
    isolated_compiled_runner_caches,
) -> None:
    inputs = _make_trivial_prefill_inputs(2048)
    _run_trivial_prefill(inputs)
    plan = next(iter(dsa_topk_gfx950._trivial_prefill_runner_plans.values()))
    assert plan.raw_launch is plan.driver.utils.launch

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_trivial_prefill(inputs)
    inputs[3].fill_(-7)
    graph.replay()

    _assert_trivial_prefill(inputs)


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_compiled_runner_uses_current_stream_and_launch_hooks(
    isolated_compiled_runner_caches,
) -> None:
    inputs = _make_trivial_decode_inputs(2048)
    _run_trivial_decode(inputs)
    assert dsa_topk_gfx950._trivial_decode_runner_plans
    plan = next(iter(dsa_topk_gfx950._trivial_decode_runner_plans.values()))
    assert plan.raw_launch is plan.driver.utils.launch

    raw_streams: list[int] = []
    enter_streams: list[int] = []
    exit_streams: list[int] = []
    raw_launch = plan.raw_launch

    def record_raw_launch(*args: object) -> None:
        raw_streams.append(args[4])
        raw_launch(*args)

    def record_enter_stream(metadata) -> None:
        enter_streams.append(metadata.get()["stream"])

    def record_exit_stream(metadata) -> None:
        exit_streams.append(metadata.get()["stream"])

    runtime = dsa_topk_gfx950.triton.knobs.runtime
    enter_hook = runtime.launch_enter_hook
    exit_hook = runtime.launch_exit_hook
    current_stream = torch.cuda.current_stream()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    plan.raw_launch = record_raw_launch
    try:
        stream_a.wait_stream(current_stream)
        with torch.cuda.stream(stream_a):
            _run_trivial_decode(inputs)
        event = stream_a.record_event()
        stream_b.wait_event(event)
        with torch.cuda.stream(stream_b):
            _run_trivial_decode(inputs)
        current_stream.wait_stream(stream_b)
        current_stream.synchronize()
        assert raw_streams == [stream_a.cuda_stream, stream_b.cuda_stream]

        raw_streams.clear()
        enter_hook.add(record_enter_stream)
        exit_hook.add(record_exit_stream)
        with torch.cuda.stream(stream_a):
            _run_trivial_decode(inputs)
        event = stream_a.record_event()
        stream_b.wait_event(event)
        with torch.cuda.stream(stream_b):
            _run_trivial_decode(inputs)
        current_stream.wait_stream(stream_b)
        enter_hook.remove(record_enter_stream)
        _run_trivial_decode(inputs)

        exit_hook.remove(record_exit_stream)
        _run_trivial_decode(inputs)
        current_stream.synchronize()
    finally:
        enter_hook.remove(record_enter_stream)
        exit_hook.remove(record_exit_stream)
        plan.raw_launch = raw_launch

    assert raw_streams == [current_stream.cuda_stream]
    assert enter_streams == [
        stream_a.cuda_stream,
        stream_b.cuda_stream,
    ]
    assert exit_streams == [
        stream_a.cuda_stream,
        stream_b.cuda_stream,
        current_stream.cuda_stream,
    ]
    _assert_trivial_decode(inputs)


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_compiled_runner_graph_replay_and_capture_miss_fallback(
    isolated_compiled_runner_caches,
) -> None:
    inputs = _make_trivial_decode_inputs(2048)
    _run_trivial_decode(inputs)
    plan = next(iter(dsa_topk_gfx950._trivial_decode_runner_plans.values()))
    assert plan.raw_launch is plan.driver.utils.launch

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_trivial_decode(inputs)
    inputs[3].fill_(-7)
    graph.replay()
    _assert_trivial_decode(inputs)

    kernel_args = (
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[1].stride(0),
        inputs[3].stride(0),
        inputs[1].shape[1],
        64,
        1,
    )
    dsa_topk_gfx950._dsa_trivial_decode_topk2048_kernel[(1, 1, 1)](
        *kernel_args,
        num_warps=8,
    )
    _clear_compiled_runner_caches()

    miss_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(miss_graph):
        _run_trivial_decode(inputs)
    assert not dsa_topk_gfx950._compiled_runner_cache
    assert not dsa_topk_gfx950._trivial_decode_runner_plans
    inputs[3].fill_(-7)
    miss_graph.replay()
    _assert_trivial_decode(inputs)


class _FakeCompiledRunnerKernel:
    def __init__(self) -> None:
        self.debug = False
        self.pre_run_hooks: list = []
        self.used_global_vals: dict = {}
        self.normal_launches = 0

    def __getitem__(self, grid):
        del grid

        def launch(*args, **kwargs) -> None:
            self.normal_launches += 1
            for hook in self.pre_run_hooks:
                hook(*args, **kwargs)

        return launch


def _make_fake_compiled_plan(
    kernel: object,
    pointer_args: tuple[torch.Tensor, ...],
    runner,
) -> dsa_topk_gfx950._CompiledRunnerPlan:
    driver = dsa_topk_gfx950.triton.runtime.driver.active
    return dsa_topk_gfx950._CompiledRunnerPlan(
        runner,
        driver,
        driver.get_current_device(),
        dsa_topk_gfx950._compiled_runner_mutable_state(kernel),
        dsa_topk_gfx950._compiled_runner_target_getter(driver),
        dsa_topk_gfx950._compiled_runner_target_key(driver),
        dsa_topk_gfx950._compiled_runner_knob_key(),
        pointer_args,
    )


def _launch_fake_compiled_plan(
    kernel: _FakeCompiledRunnerKernel,
    pointer_args: tuple[torch.Tensor, ...],
    plan: dsa_topk_gfx950._CompiledRunnerPlan,
) -> None:
    dispatch_key = ("test",)
    dispatch_cache = OrderedDict(((dispatch_key, plan),))
    dsa_topk_gfx950._launch_warmed_compiled_kernel(
        kernel,
        (1, 1, 1),
        pointer_args,
        (torch.int32,) * len(pointer_args),
        (),
        dispatch_cache=dispatch_cache,
        dispatch_key=dispatch_key,
        num_warps=1,
    )


def _make_real_compiled_wrapper_case(name: str) -> Callable[[], None]:
    if name == "trivial_decode":
        inputs = _make_trivial_decode_inputs(2048)
        return lambda: _run_trivial_decode(inputs)

    if name == "runtime_decode":
        logits = torch.empty((1, 8192), device="cuda", dtype=torch.float32)
        block_table = torch.arange(128, device="cuda", dtype=torch.int32)[None, :]
        seq_lens = torch.tensor([8192], device="cuda", dtype=torch.int32)
        out = torch.empty((1, 2048), device="cuda", dtype=torch.int32)
        lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)

        def run_runtime_decode() -> None:
            dsa_topk_gfx950._dsa_decode_topk_slots(
                logits,
                block_table,
                seq_lens,
                page_size=64,
                topk=2048,
                q_len_per_req=1,
                out=out,
                lens_out=lens_out,
            )

        return run_runtime_decode

    if name == "persistent":
        logits = torch.empty((1, 131072), device="cuda", dtype=torch.float32)
        block_table = torch.arange(2048, device="cuda", dtype=torch.int32)[None, :]
        row_starts = torch.zeros((1,), device="cuda", dtype=torch.int32)
        row_ends = torch.tensor([131072], device="cuda", dtype=torch.int32)
        workspace = tuple(
            torch.zeros((1,), device="cuda", dtype=torch.int32) for _ in range(5)
        )
        out = torch.empty((1, 2048), device="cuda", dtype=torch.int32)
        lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)

        def run_persistent() -> None:
            dsa_topk_gfx950._dsa_persistent_radix_topk(
                logits,
                block_table,
                row_starts,
                row_ends,
                page_size=64,
                q_len_per_req=1,
                is_decode=True,
                topk=2048,
                groups=8,
                workspace=workspace,
                out=out,
                lens_out=lens_out,
            )

        return run_persistent

    if name == "trivial_prefill":
        inputs = _make_trivial_prefill_inputs(2048)
        return lambda: _run_trivial_prefill(inputs)

    raise AssertionError(f"unknown wrapper case: {name}")


def _install_recording_wrapper_plan(
    monkeypatch: pytest.MonkeyPatch,
    run_wrapper: Callable[[], None],
) -> tuple[
    object,
    dsa_topk_gfx950._CompiledRunnerPlan,
    list[tuple[object, ...]],
    list[tuple[object, ...]],
]:
    slow_launches: list[tuple[object, ...]] = []

    def record_slow_launch(
        kernel,
        grid,
        args,
        pointer_dtypes,
        specialization_key,
        **kwargs,
    ) -> None:
        slow_launches.append(
            (kernel, grid, args, pointer_dtypes, specialization_key, kwargs)
        )

    monkeypatch.setattr(
        dsa_topk_gfx950,
        "_launch_warmed_compiled_kernel",
        record_slow_launch,
    )
    run_wrapper()
    assert len(slow_launches) == 1
    kernel, _, args, pointer_dtypes, _, kwargs = slow_launches[0]
    direct_launches: list[tuple[object, ...]] = []
    plan = _make_fake_compiled_plan(
        kernel,
        args[: len(pointer_dtypes)],
        lambda *runner_args: direct_launches.append(runner_args),
    )
    dispatch_cache = kwargs["dispatch_cache"]
    dispatch_key = kwargs["dispatch_key"]
    dsa_topk_gfx950._cache_compiled_runner_plan(
        dispatch_cache,
        dispatch_key,
        plan,
    )
    return kernel, plan, slow_launches, direct_launches


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
@pytest.mark.parametrize(
    "wrapper_name",
    ("persistent", "trivial_decode", "runtime_decode", "trivial_prefill"),
)
def test_real_wrapper_same_pointer_plan_uses_direct_runner(
    wrapper_name: str,
    isolated_compiled_runner_caches,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_wrapper = _make_real_compiled_wrapper_case(wrapper_name)
    _, _, slow_launches, direct_launches = _install_recording_wrapper_plan(
        monkeypatch,
        run_wrapper,
    )

    run_wrapper()

    assert len(slow_launches) == 1
    assert len(direct_launches) == 1


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
@pytest.mark.parametrize(
    "wrapper_name",
    ("persistent", "trivial_decode", "runtime_decode", "trivial_prefill"),
)
@pytest.mark.parametrize(
    "unsupported_state",
    (
        "pre_run_hook",
        "used_global",
        "kernel_debug",
        "runtime_debug",
        "inspection_hook",
        "instrumentation",
        "compile_knob",
        "environment",
        "device",
        "target",
    ),
)
def test_real_wrapper_same_pointer_state_change_falls_back(
    wrapper_name: str,
    unsupported_state: str,
    isolated_compiled_runner_caches,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_wrapper = _make_real_compiled_wrapper_case(wrapper_name)
    kernel, plan, slow_launches, direct_launches = _install_recording_wrapper_plan(
        monkeypatch,
        run_wrapper,
    )
    knobs = dsa_topk_gfx950.triton.knobs
    expect_restored_direct = False

    if unsupported_state == "pre_run_hook":
        kernel.pre_run_hooks.append(lambda *args, **kwargs: None)
        try:
            run_wrapper()
        finally:
            kernel.pre_run_hooks.pop()
    elif unsupported_state == "used_global":
        kernel.used_global_vals["test"] = 1
        try:
            run_wrapper()
        finally:
            del kernel.used_global_vals["test"]
    elif unsupported_state == "kernel_debug":
        monkeypatch.setattr(kernel, "debug", True)
        run_wrapper()
    elif unsupported_state == "runtime_debug":
        with knobs.runtime.scope():
            knobs.runtime.debug = True
            run_wrapper()
    elif unsupported_state == "inspection_hook":
        with knobs.runtime.scope():
            knobs.runtime.add_stages_inspection_hook = lambda *args: None
            run_wrapper()
    elif unsupported_state == "instrumentation":
        with knobs.compilation.scope():
            knobs.compilation.instrumentation_mode = "test"
            run_wrapper()
    elif unsupported_state == "compile_knob":
        with knobs.amd.scope():
            knobs.amd.use_buffer_ops = False
            run_wrapper()
        run_wrapper()
        expect_restored_direct = True
    elif unsupported_state == "environment":
        monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
        run_wrapper()
    elif unsupported_state == "device":
        plan.device = plan.device + 1
        run_wrapper()
    else:
        with knobs.runtime.scope():
            knobs.runtime.override_arch = "gfx942"
            run_wrapper()

    assert len(slow_launches) == 2
    assert len(direct_launches) == int(expect_restored_direct)


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
@pytest.mark.parametrize("cache_level", ("dispatch", "route"))
def test_compiled_runner_caches_evict_least_recently_used_plan(
    cache_level: str,
    isolated_compiled_runner_caches,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dsa_topk_gfx950, "_COMPILED_RUNNER_CACHE_SIZE", 2)
    kernel = _FakeCompiledRunnerKernel()
    pointer_args = tuple(
        torch.zeros((4,), device="cuda", dtype=torch.int32) for _ in range(4)
    )
    direct_launches: list[bool] = []
    plan = _make_fake_compiled_plan(
        kernel,
        pointer_args,
        lambda *args: direct_launches.append(True),
    )
    active_key = ("active",)
    cold_key = ("cold",)
    new_key = ("new",)

    if cache_level == "dispatch":
        cache = OrderedDict(((active_key, plan), (cold_key, plan)))
        dsa_topk_gfx950._launch_warmed_compiled_kernel(
            kernel,
            (1, 1, 1),
            pointer_args,
            (torch.int32,) * len(pointer_args),
            (),
            dispatch_cache=cache,
            dispatch_key=active_key,
            num_warps=1,
        )
    else:
        cache = dsa_topk_gfx950._compiled_runner_cache
        driver = dsa_topk_gfx950.triton.runtime.driver.active
        active_key = dsa_topk_gfx950._compiled_runner_route_key(
            kernel,
            (1, 1, 1),
            (torch.int32,) * len(pointer_args),
            (),
            driver,
            driver.get_current_device(),
            0,
            1,
        )
        cache[active_key] = plan
        cache[cold_key] = plan
        dsa_topk_gfx950._launch_warmed_compiled_kernel(
            kernel,
            (1, 1, 1),
            pointer_args,
            (torch.int32,) * len(pointer_args),
            (),
            num_warps=1,
        )

    assert list(cache) == [cold_key, active_key]
    dsa_topk_gfx950._cache_compiled_runner_plan(cache, new_key, plan)
    assert list(cache) == [active_key, new_key]
    assert len(direct_launches) == 1


class _BlockingLRUCache(OrderedDict):
    def __init__(self, block_method: str) -> None:
        super().__init__()
        self.block_method = block_method
        self.operation_started = Event()
        self.operation_allowed = Event()

    def _block(self, method: str) -> None:
        if self.block_method == method:
            self.operation_started.set()
            assert self.operation_allowed.wait(timeout=5)

    def move_to_end(self, key, last: bool = True) -> None:
        self._block("move_to_end")
        super().move_to_end(key, last=last)

    def popitem(self, last: bool = True):
        self._block("popitem")
        return super().popitem(last=last)


@pytest.mark.parametrize(
    "race",
    ("lookup_move", "insert_move", "insert_evict"),
)
def test_compiled_runner_lru_tolerates_concurrent_clear(
    race: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_method = "popitem" if race == "insert_evict" else "move_to_end"
    cache = _BlockingLRUCache(block_method)
    key = ("active",)
    plan = object()
    if race != "insert_move":
        cache[key] = plan
    if race == "insert_evict":
        monkeypatch.setattr(dsa_topk_gfx950, "_COMPILED_RUNNER_CACHE_SIZE", 1)

    results: list[object | None] = []
    errors: list[BaseException] = []

    def access_cache() -> None:
        try:
            if race == "lookup_move":
                results.append(
                    dsa_topk_gfx950._get_cached_compiled_runner_plan(cache, key)
                )
            else:
                dsa_topk_gfx950._cache_compiled_runner_plan(
                    cache,
                    ("new",),
                    plan,
                )
                results.append(None)
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=access_cache)
    thread.start()
    try:
        assert cache.operation_started.wait(timeout=5)
        cache.clear()
    finally:
        cache.operation_allowed.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert results == [None]


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
@pytest.mark.parametrize(
    "unsupported_state",
    [
        "pre_run_hook",
        "used_global",
        "debug",
        "inspection_hook",
        "instrumentation",
        "compile_knob",
    ],
)
def test_compiled_runner_falls_back_for_unsupported_mutable_state(
    unsupported_state: str,
) -> None:
    kernel = _FakeCompiledRunnerKernel()
    original_args = tuple(
        torch.zeros((4,), device="cuda", dtype=torch.int32) for _ in range(4)
    )
    direct_launches: list[bool] = []
    plan = _make_fake_compiled_plan(
        kernel,
        original_args,
        lambda *args: direct_launches.append(True),
    )
    changed_args = tuple(
        torch.ones((4,), device="cuda", dtype=torch.int32) for _ in range(4)
    )
    knobs = dsa_topk_gfx950.triton.knobs

    if unsupported_state == "pre_run_hook":
        kernel.pre_run_hooks.append(lambda *args, **kwargs: None)
        _launch_fake_compiled_plan(kernel, changed_args, plan)
    elif unsupported_state == "used_global":
        kernel.used_global_vals["test"] = 1
        _launch_fake_compiled_plan(kernel, changed_args, plan)
    elif unsupported_state == "debug":
        kernel.debug = True
        _launch_fake_compiled_plan(kernel, changed_args, plan)
    elif unsupported_state == "inspection_hook":
        with knobs.runtime.scope():
            knobs.runtime.add_stages_inspection_hook = lambda *args: None
            _launch_fake_compiled_plan(kernel, changed_args, plan)
    elif unsupported_state == "instrumentation":
        with knobs.compilation.scope():
            knobs.compilation.instrumentation_mode = "test"
            _launch_fake_compiled_plan(kernel, changed_args, plan)
    else:
        with knobs.amd.scope():
            knobs.amd.use_buffer_ops = False
            _launch_fake_compiled_plan(kernel, changed_args, plan)

    assert kernel.normal_launches == 1
    assert not direct_launches


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
@pytest.mark.parametrize("unsupported_pointer", ["dtype", "alignment"])
def test_compiled_runner_falls_back_for_unsupported_pointer(
    unsupported_pointer: str,
) -> None:
    kernel = _FakeCompiledRunnerKernel()
    original_args = tuple(
        torch.zeros((4,), device="cuda", dtype=torch.int32) for _ in range(4)
    )
    direct_launches: list[bool] = []
    plan = _make_fake_compiled_plan(
        kernel,
        original_args,
        lambda *args: direct_launches.append(True),
    )
    changed_args = list(
        torch.ones((4,), device="cuda", dtype=torch.int32) for _ in range(4)
    )
    if unsupported_pointer == "dtype":
        changed_args[0] = torch.ones((4,), device="cuda", dtype=torch.int64)
    else:
        backing = torch.ones((5,), device="cuda", dtype=torch.int32)
        changed_args[0] = backing[1:]

    _launch_fake_compiled_plan(kernel, tuple(changed_args), plan)

    assert kernel.normal_launches == 1
    assert not direct_launches


@pytest.mark.skipif(
    not dsa_topk_gfx950._COMPILED_RUNNER_ABI_SUPPORTED,
    reason="requires the supported CompiledKernel runner ABI",
)
def test_runtime_decode_plan_reuses_native_strides(
    isolated_compiled_runner_caches,
) -> None:
    plan = None
    for cols in (8192, 16384):
        row_starts = torch.zeros((1,), device="cuda", dtype=torch.int32)
        row_ends = torch.full((1,), cols, device="cuda", dtype=torch.int32)
        logits = _make_grouped_radix_logits(
            row_starts,
            row_ends,
            cols=cols,
            topk=2048,
        )
        block_table = torch.arange(
            math.ceil(cols / 64), device="cuda", dtype=torch.int32
        )[None, :]
        out = torch.empty((1, 2048), device="cuda", dtype=torch.int32)
        lens_out = torch.empty((1,), device="cuda", dtype=torch.int32)
        dsa_topk_gfx950._dsa_decode_topk_slots(
            logits,
            block_table,
            row_ends,
            page_size=64,
            topk=2048,
            q_len_per_req=1,
            out=out,
            lens_out=lens_out,
        )
        _assert_grouped_radix_topk(
            logits,
            out,
            lens_out,
            row_starts,
            row_ends,
            topk=2048,
        )
        assert len(dsa_topk_gfx950._runtime_decode_runner_plans) == 1
        current_plan = next(iter(dsa_topk_gfx950._runtime_decode_runner_plans.values()))
        if plan is None:
            plan = current_plan
        else:
            assert current_plan is plan


@pytest.mark.parametrize("cols", [8192, 131072, 524288])
def test_dsa_decode_topk_maps_grouped_queries_to_physical_slots(
    cols: int,
) -> None:
    page_size = 64
    q_len_per_req = 4
    requests = 3
    rows = requests * q_len_per_req
    topk = 2048
    seq_lens = cols - torch.arange(requests, device="cuda", dtype=torch.int32) * 53
    q_offsets = torch.arange(q_len_per_req, device="cuda", dtype=torch.int32)
    row_ends = (seq_lens[:, None] - (q_len_per_req - 1) + q_offsets[None, :]).reshape(
        -1
    )
    row_starts = torch.zeros((rows,), device="cuda", dtype=torch.int32)
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    block_table = _make_reversed_decode_block_table(
        requests,
        cols,
        page_size,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )

    _assert_decode_topk_slots(
        logits,
        out,
        lens_out,
        seq_lens,
        block_table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        topk=topk,
    )


def test_dsa_persistent_decode_groups_obey_residency_bound() -> None:
    device = torch.device("cuda")
    compute_units = torch.cuda.get_device_properties(device).multi_processor_count
    topk = 2048

    assert dsa_topk_gfx950._persistent_decode_groups(1, 131072, topk, device) == 8
    assert (
        dsa_topk_gfx950._persistent_decode_groups(
            compute_units // 2,
            131072,
            topk,
            device,
        )
        == 2
    )
    assert (
        dsa_topk_gfx950._persistent_decode_groups(
            compute_units // 2 + 1,
            131072,
            topk,
            device,
        )
        is None
    )
    assert dsa_topk_gfx950._persistent_decode_groups(1, 90000, topk, device) is None
    assert (
        dsa_topk_gfx950._persistent_decode_groups(
            1,
            256 * 1024 + 1,
            topk,
            device,
        )
        is None
    )


def test_dsa_persistent_decode_handles_tail_ties_and_infinities() -> None:
    page_size = 64
    q_len_per_req = 4
    requests = 2
    rows = requests * q_len_per_req
    cols = 147493
    topk = 2048
    seq_lens = torch.tensor(
        [cols - 17, cols - 113],
        device="cuda",
        dtype=torch.int32,
    )
    q_offsets = torch.arange(q_len_per_req, device="cuda", dtype=torch.int32)
    row_ends = (seq_lens[:, None] - (q_len_per_req - 1) + q_offsets[None, :]).reshape(
        -1
    )
    logits = torch.zeros((rows, cols), device="cuda", dtype=torch.float32)
    for row, row_end in enumerate(row_ends.cpu().tolist()):
        if row == 0:
            logits[row, :row_end] = float("inf")
        elif row == 1:
            logits[row, :row_end] = -float("inf")
        else:
            logits[row, :37] = float("inf")
            logits[row, 37:row_end:257] = -float("inf")
    block_table = _make_reversed_decode_block_table(requests, cols, page_size)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    assert (
        dsa_topk_gfx950._persistent_decode_groups(
            rows,
            cols,
            topk,
            logits.device,
        )
        == 8
    )
    dsa_topk_gfx950._dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )

    _assert_decode_topk_slots(
        logits,
        out,
        lens_out,
        seq_lens,
        block_table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        topk=topk,
    )
    workspace = dsa_topk_gfx950._persistent_topk_workspace(rows, logits.device)
    assert all(int(torch.count_nonzero(t).item()) == 0 for t in workspace)


@pytest.mark.parametrize("warm_workspace", (False, True), ids=("cold-key", "warm-key"))
def test_dsa_persistent_decode_is_graph_capturable(warm_workspace: bool) -> None:
    page_size = 64
    q_len_per_req = 4
    requests = 2
    rows = requests * q_len_per_req
    cols = 131072
    topk = 2048
    seq_lens = torch.tensor(
        [cols - 17, cols - 71],
        device="cuda",
        dtype=torch.int32,
    )
    q_offsets = torch.arange(q_len_per_req, device="cuda", dtype=torch.int32)
    row_ends = (seq_lens[:, None] - (q_len_per_req - 1) + q_offsets[None, :]).reshape(
        -1
    )
    logits = _make_grouped_radix_logits(
        torch.zeros_like(row_ends),
        row_ends,
        cols=cols,
        topk=topk,
        seed=5907,
    )
    block_table = _make_reversed_decode_block_table(requests, cols, page_size)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    device_index = logits.device.index
    assert device_index is not None
    workspace_key = (
        device_index,
        int(capture_stream.cuda_stream),
        dsa_topk_gfx950._next_power_of_two(rows),
    )
    assert workspace_key not in dsa_topk_gfx950._persistent_topk_workspace_cache
    if warm_workspace:
        with torch.cuda.stream(capture_stream):
            dsa_topk_gfx950._dsa_decode_topk_slots(
                logits,
                block_table,
                seq_lens,
                page_size=page_size,
                topk=topk,
                q_len_per_req=q_len_per_req,
                out=out,
                lens_out=lens_out,
            )
        capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        dsa_topk_gfx950._dsa_decode_topk_slots(
            logits,
            block_table,
            seq_lens,
            page_size=page_size,
            topk=topk,
            q_len_per_req=q_len_per_req,
            out=out,
            lens_out=lens_out,
        )
    capture_stream.synchronize()
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()

    _assert_decode_topk_slots(
        logits,
        out,
        lens_out,
        seq_lens,
        block_table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        topk=topk,
    )


def test_dsa_persistent_decode_is_stream_local() -> None:
    page_size = 64
    q_len_per_req = 4
    requests = 2
    rows = requests * q_len_per_req
    cols = 131072
    topk = 2048
    seq_lens = torch.tensor(
        [cols - 17, cols - 71],
        device="cuda",
        dtype=torch.int32,
    )
    q_offsets = torch.arange(q_len_per_req, device="cuda", dtype=torch.int32)
    row_ends = (seq_lens[:, None] - (q_len_per_req - 1) + q_offsets[None, :]).reshape(
        -1
    )
    row_starts = torch.zeros_like(row_ends)
    logits_a = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=6907,
    )
    logits_b = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=7907,
    )
    block_table = _make_reversed_decode_block_table(requests, cols, page_size)
    out_a = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    out_b = torch.empty_like(out_a)
    lens_a = torch.empty((rows,), device="cuda", dtype=torch.int32)
    lens_b = torch.empty_like(lens_a)

    current_stream = torch.cuda.current_stream()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a.wait_stream(current_stream)
    stream_b.wait_stream(current_stream)
    with torch.cuda.stream(stream_a):
        workspace_a = dsa_topk_gfx950._persistent_topk_workspace(
            rows,
            logits_a.device,
        )
        for _ in range(3):
            dsa_topk_gfx950._dsa_decode_topk_slots(
                logits_a,
                block_table,
                seq_lens,
                page_size=page_size,
                topk=topk,
                q_len_per_req=q_len_per_req,
                out=out_a,
                lens_out=lens_a,
            )
    with torch.cuda.stream(stream_b):
        workspace_b = dsa_topk_gfx950._persistent_topk_workspace(
            rows,
            logits_b.device,
        )
        for _ in range(3):
            dsa_topk_gfx950._dsa_decode_topk_slots(
                logits_b,
                block_table,
                seq_lens,
                page_size=page_size,
                topk=topk,
                q_len_per_req=q_len_per_req,
                out=out_b,
                lens_out=lens_b,
            )
    assert all(
        tensor_a.data_ptr() != tensor_b.data_ptr()
        for tensor_a, tensor_b in zip(workspace_a, workspace_b, strict=True)
    )
    current_stream.wait_stream(stream_a)
    current_stream.wait_stream(stream_b)

    _assert_decode_topk_slots(
        logits_a,
        out_a,
        lens_a,
        seq_lens,
        block_table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        topk=topk,
    )
    _assert_decode_topk_slots(
        logits_b,
        out_b,
        lens_b,
        seq_lens,
        block_table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        topk=topk,
    )
    assert all(int(torch.count_nonzero(t).item()) == 0 for t in workspace_a)
    assert all(int(torch.count_nonzero(t).item()) == 0 for t in workspace_b)


def test_dsa_prefill_topk_hist_derived_handles_shifted_and_inf_rows() -> None:
    rows = 4
    cols = 524288
    topk = 2048
    row_starts = torch.tensor([17, 31, 47, 63], device="cuda", dtype=torch.int32)
    row_ends = torch.tensor(
        [1017, cols - 91, cols - 61, cols - 31],
        device="cuda",
        dtype=torch.int32,
    )
    generator = _generator("cuda", 4907)
    logits = torch.randn(
        (rows, cols),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    columns = torch.arange(cols, device="cuda")
    valid = (columns[None, :] >= row_starts[:, None]) & (
        columns[None, :] < row_ends[:, None]
    )
    logits.masked_fill_(~valid, -float("inf"))
    logits[1, row_starts[1] + 1000 : row_ends[1]] = -float("inf")
    logits[2, row_starts[2] : row_starts[2] + 37] = float("inf")
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )


@pytest.mark.parametrize("cols", [131072, 262144], ids=["runtime", "staged"])
def test_dsa_prefill_grouped_radix_topk_is_graph_capturable(cols: int) -> None:
    rows = 64
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 11
    row_ends = cols - (rows - 1 - row_ids) * 19
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        dsa_topk_gfx950._dsa_prefill_topk_indices(
            logits,
            row_starts,
            row_ends,
            topk=topk,
            out=out,
            lens_out=lens_out,
        )
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsa_topk_gfx950._dsa_prefill_topk_indices(
            logits,
            row_starts,
            row_ends,
            topk=topk,
            out=out,
            lens_out=lens_out,
        )
    out.fill_(-7)
    graph.replay()

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )


@pytest.mark.parametrize("cols", [131072, 262144], ids=["runtime", "staged"])
def test_dsa_prefill_grouped_radix_topk_is_stream_local(cols: int) -> None:
    rows = 64
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 13
    row_ends = cols - (rows - 1 - row_ids) * 23
    logits_a = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=2907,
    )
    logits_b = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=3907,
    )
    out_a = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    out_b = torch.empty_like(out_a)
    lens_a = torch.empty((rows,), device="cuda", dtype=torch.int32)
    lens_b = torch.empty_like(lens_a)

    current_stream = torch.cuda.current_stream()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a.wait_stream(current_stream)
    stream_b.wait_stream(current_stream)
    with torch.cuda.stream(stream_a):
        dsa_topk_gfx950._dsa_prefill_topk_indices(
            logits_a,
            row_starts,
            row_ends,
            topk=topk,
            out=out_a,
            lens_out=lens_a,
        )
    with torch.cuda.stream(stream_b):
        dsa_topk_gfx950._dsa_prefill_topk_indices(
            logits_b,
            row_starts,
            row_ends,
            topk=topk,
            out=out_b,
            lens_out=lens_b,
        )
    current_stream.wait_stream(stream_a)
    current_stream.wait_stream(stream_b)

    _assert_grouped_radix_topk(
        logits_a,
        out_a,
        lens_a,
        row_starts,
        row_ends,
        topk=topk,
    )
    _assert_grouped_radix_topk(
        logits_b,
        out_b,
        lens_b,
        row_starts,
        row_ends,
        topk=topk,
    )


@pytest.mark.parametrize("warm_workspace", (False, True), ids=("cold-key", "warm-key"))
def test_dsa_persistent_prefill_topk_is_graph_capturable(
    warm_workspace: bool,
) -> None:
    rows = 64 if warm_workspace else 96
    cols = 131072
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 11
    row_ends = cols - (rows - 1 - row_ids) * 19
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    # Compile before capture without populating the capture stream's cache key.
    dsa_topk_gfx950._dsa_prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    device_index = logits.device.index
    assert device_index is not None
    workspace_key = (
        device_index,
        int(capture_stream.cuda_stream),
        dsa_topk_gfx950._next_power_of_two(rows),
    )
    assert workspace_key not in dsa_topk_gfx950._persistent_topk_workspace_cache
    if warm_workspace:
        with torch.cuda.stream(capture_stream):
            dsa_topk_gfx950._dsa_prefill_topk_indices(
                logits,
                row_starts,
                row_ends,
                topk=topk,
                out=out,
                lens_out=lens_out,
            )
        capture_stream.synchronize()
        assert workspace_key in dsa_topk_gfx950._persistent_topk_workspace_cache

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        dsa_topk_gfx950._dsa_prefill_topk_indices(
            logits,
            row_starts,
            row_ends,
            topk=topk,
            out=out,
            lens_out=lens_out,
        )
    capture_stream.synchronize()
    assert workspace_key in dsa_topk_gfx950._persistent_topk_workspace_cache
    if warm_workspace:
        graph_workspace = dsa_topk_gfx950._persistent_topk_workspace_cache[
            workspace_key
        ]
        graph_workspace_ptrs = tuple(t.data_ptr() for t in graph_workspace)
        pressure_streams = [torch.cuda.Stream() for _ in range(17)]
        for pressure_stream in pressure_streams:
            with torch.cuda.stream(pressure_stream):
                dsa_topk_gfx950._persistent_topk_workspace(
                    rows,
                    logits.device,
                )
        torch.cuda.synchronize()
        assert len(dsa_topk_gfx950._persistent_topk_workspace_cache) <= (
            dsa_topk_gfx950._PERSISTENT_PREFILL_WORKSPACE_CACHE_MAXSIZE
        )
        assert workspace_key in dsa_topk_gfx950._persistent_topk_workspace_cache
        assert (
            tuple(
                t.data_ptr()
                for t in dsa_topk_gfx950._persistent_topk_workspace_cache[workspace_key]
            )
            == graph_workspace_ptrs
        )
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()

    _assert_grouped_radix_topk(
        logits,
        out,
        lens_out,
        row_starts,
        row_ends,
        topk=topk,
    )


def test_dsa_persistent_prefill_topk_is_stream_local() -> None:
    rows = 64
    cols = 131072
    topk = 2048
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = row_ids * 13
    row_ends = cols - (rows - 1 - row_ids) * 23
    logits_a = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=2907,
    )
    logits_b = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=3907,
    )
    out_a = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    out_b = torch.empty_like(out_a)
    lens_a = torch.empty((rows,), device="cuda", dtype=torch.int32)
    lens_b = torch.empty_like(lens_a)

    current_stream = torch.cuda.current_stream()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a.wait_stream(current_stream)
    stream_b.wait_stream(current_stream)
    with torch.cuda.stream(stream_a):
        workspace_a = dsa_topk_gfx950._persistent_topk_workspace(
            rows,
            logits_a.device,
        )
        for _ in range(3):
            dsa_topk_gfx950._dsa_prefill_topk_indices(
                logits_a,
                row_starts,
                row_ends,
                topk=topk,
                out=out_a,
                lens_out=lens_a,
            )
    with torch.cuda.stream(stream_b):
        workspace_b = dsa_topk_gfx950._persistent_topk_workspace(
            rows,
            logits_b.device,
        )
        for _ in range(3):
            dsa_topk_gfx950._dsa_prefill_topk_indices(
                logits_b,
                row_starts,
                row_ends,
                topk=topk,
                out=out_b,
                lens_out=lens_b,
            )
    assert all(
        tensor_a.data_ptr() != tensor_b.data_ptr()
        for tensor_a, tensor_b in zip(workspace_a, workspace_b, strict=True)
    )
    current_stream.wait_stream(stream_a)
    current_stream.wait_stream(stream_b)

    _assert_grouped_radix_topk(
        logits_a,
        out_a,
        lens_a,
        row_starts,
        row_ends,
        topk=topk,
    )
    _assert_grouped_radix_topk(
        logits_b,
        out_b,
        lens_b,
        row_starts,
        row_ends,
        topk=topk,
    )
    assert all(int(torch.count_nonzero(t).item()) == 0 for t in workspace_a)
    assert all(int(torch.count_nonzero(t).item()) == 0 for t in workspace_b)


def _pack_sparse_kv(
    latent: torch.Tensor,
    rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_lora_rank = latent.shape[1]
    qk_rope_head_dim = rope.shape[1]
    scale = latent.float().abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6) / 448.0
    latent_fp8 = (latent.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    row_bytes = kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2
    sparse = torch.empty(
        (latent.shape[0], row_bytes),
        dtype=torch.uint8,
        device=latent.device,
    )
    sparse[:, :kv_lora_rank].copy_(latent_fp8.view(torch.uint8))
    scale_start = kv_lora_rank
    scale_end = scale_start + kv_lora_rank // 128 * 4
    sparse[:, scale_start:scale_end].view(torch.float32).copy_(scale)
    sparse[:, scale_end:].view(torch.bfloat16).copy_(rope)
    return sparse, latent_fp8.float() * scale


def _dsa_reference(
    q: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    refs = []
    kv_lora_rank = latent.shape[1]
    for token in range(q.shape[0]):
        valid_slots = topk_slots[token, : int(topk_lens[token].item())].long()
        valid_slots = valid_slots[valid_slots >= 0]
        q_nope = q[token, :, :kv_lora_rank].float()
        q_rope = q[token, :, kv_lora_rank:].float()
        if valid_slots.numel() == 0:
            refs.append(torch.zeros_like(q_nope))
            continue
        k_nope = latent.index_select(0, valid_slots).float()
        k_rope = rope.index_select(0, valid_slots).float()
        scores = torch.einsum("hd,kd->hk", q_nope, k_nope)
        scores += torch.einsum("hd,kd->hk", q_rope, k_rope)
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        refs.append(torch.matmul(probs, k_nope))
    return torch.stack(refs, dim=0).to(torch.bfloat16)


def _dsa_visible_ranges(case: _DSACase, device: str) -> tuple[list[range], int]:
    if case.mode == "decode":
        assert case.visible_lens is not None
        ranges = [range(0, int(visible_len)) for visible_len in case.visible_lens]
        return ranges, _round_up_to_page(max(case.visible_lens), 64)

    assert case.prefix_lens is not None
    assert case.extend_lens is not None
    kv_workspace_slots, _, _, ranges = _make_prefill_workspace(
        case.prefix_lens,
        case.extend_lens,
        device=device,
    )
    return ranges, _round_up_to_page(int(kv_workspace_slots.numel()), 64)


def _make_selected_topk_slots(
    case: _DSACase,
    visible_ranges: Sequence[range],
    *,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert case.topk_lens is not None
    assert len(case.topk_lens) == len(visible_ranges)
    topk_slots = torch.full(
        (len(visible_ranges), case.topk), -1, device=device, dtype=torch.int32
    )
    lens: list[int] = []
    for token, visible_range in enumerate(visible_ranges):
        visible_count = len(visible_range)
        count = min(int(case.topk_lens[token]), visible_count, int(case.topk))
        lens.append(count)
        if count == 0:
            continue
        candidates = torch.arange(
            visible_range.start,
            visible_range.stop,
            device=device,
            dtype=torch.int32,
        )
        perm = torch.randperm(visible_count, device=device, generator=generator)[:count]
        topk_slots[token, :count] = candidates.index_select(0, perm)
    return topk_slots, torch.tensor(lens, device=device, dtype=torch.int32)


def _assert_slots_visible(
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    visible_ranges: Sequence[range],
) -> None:
    for token, visible_range in enumerate(visible_ranges):
        count = int(topk_lens[token].item())
        valid = topk_slots[token, :count]
        if count:
            assert (valid >= visible_range.start).all()
            assert (valid < visible_range.stop).all()
        assert (topk_slots[token, count:] == -1).all()


@pytest.mark.parametrize(
    "mode,api",
    [
        pytest.param("decode", gluon_dsa_decode_gfx950, id="decode"),
        pytest.param("prefill", gluon_dsa_prefill_gfx950, id="prefill"),
    ],
)
@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_fp8"),
    ],
)
def test_dsa_with_sparse_kvcache(mode: str, api, q_dtype: torch.dtype) -> None:
    device = "cuda"
    tokens = 3
    num_heads = 2
    num_slots = 16
    topk = 512
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q_bf16 = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_bf16.to(q_dtype)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    sparse_kv, dequant_latent = _pack_sparse_kv(latent, rope)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([5, 7, 4], device=device, dtype=torch.int32)
    for token in range(tokens):
        count = int(topk_lens[token].item())
        topk_slots[token, :count] = torch.randperm(num_slots, device=device)[:count]

    out = api(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
    )

    ref = _dsa_reference(
        q,
        dequant_latent,
        rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "mode,api",
    [
        pytest.param("decode", gluon_dsa_decode_gfx950, id="decode"),
        pytest.param("prefill", gluon_dsa_prefill_gfx950, id="prefill"),
    ],
)
@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_fp8"),
    ],
)
def test_dsa_dense_kvcache(mode: str, api, q_dtype: torch.dtype) -> None:
    device = "cuda"
    tokens = 3
    num_heads = 2
    num_slots = 16
    topk = 512
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q_bf16 = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_bf16.to(q_dtype)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    kv_cache = torch.cat([latent, rope], dim=-1).to(q_dtype)
    dequant_latent = kv_cache[:, :kv_lora_rank].float().to(torch.bfloat16)
    dequant_rope = kv_cache[:, kv_lora_rank:].float().to(torch.bfloat16)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([5, 7, 4], device=device, dtype=torch.int32)
    for token in range(tokens):
        count = int(topk_lens[token].item())
        topk_slots[token, :count] = torch.randperm(num_slots, device=device)[:count]

    out = api(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
    )

    ref = _dsa_reference(
        q,
        dequant_latent,
        dequant_rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


def test_dsa_decode_sparse_kvcache_trims_large_topk_for_tiny_lens() -> None:
    device = "cuda"
    tokens = 2
    num_heads = 2
    num_slots = 8
    topk = 2048
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    sparse_kv, dequant_latent = _pack_sparse_kv(latent, rope)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([1, 3], device=device, dtype=torch.int32)
    topk_slots[0, 0] = 2
    topk_slots[1, :3] = torch.tensor([0, 5, 7], device=device, dtype=torch.int32)

    out = gluon_dsa_decode_gfx950(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
    )

    ref = _dsa_reference(
        q,
        dequant_latent,
        rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "case",
    _GLM52_DSA_CASES,
    ids=lambda case: case.name,
)
def test_dsa_glm52_selected_attention_cases(case: _DSACase) -> None:
    device = "cuda"
    page_size = 64
    gen = _generator(device, case.seed)
    visible_ranges, num_slots = _dsa_visible_ranges(case, device)
    tokens = len(visible_ranges)
    q = _randn_bf16(
        (tokens, case.num_heads, case.kv_lora_rank + case.qk_rope_head_dim),
        device=device,
        generator=gen,
    )
    latent = _randn_bf16((num_slots, case.kv_lora_rank), device=device, generator=gen)
    rope = _randn_bf16((num_slots, case.qk_rope_head_dim), device=device, generator=gen)
    topk_slots, topk_lens = _make_selected_topk_slots(
        case, visible_ranges, device=device, generator=gen
    )
    _assert_slots_visible(topk_slots, topk_lens, visible_ranges)

    kv_cache = None
    sparse_kv_cache = None
    if case.kv_layout == "dense":
        kv_cache = torch.cat([latent, rope], dim=-1).contiguous()
        reference_latent = kv_cache[:, : case.kv_lora_rank]
        reference_rope = kv_cache[:, case.kv_lora_rank :]
    elif case.kv_layout == "sparse":
        sparse_kv_cache, reference_latent = _pack_sparse_kv(latent, rope)
        reference_rope = rope
    else:
        raise AssertionError(f"unknown DSA KV layout {case.kv_layout!r}")

    softmax_scale = 1.0 / math.sqrt(case.qk_nope_head_dim + case.qk_rope_head_dim)
    common_kwargs = {
        "q": q,
        "kv_cache": kv_cache,
        "sparse_kv_cache": sparse_kv_cache,
        "topk_slots": topk_slots,
        "topk_lens": topk_lens,
        "max_seqlen_k": max(len(visible_range) for visible_range in visible_ranges),
        "qk_nope_head_dim": case.qk_nope_head_dim,
        "kv_lora_rank": case.kv_lora_rank,
        "qk_rope_head_dim": case.qk_rope_head_dim,
        "softmax_scale": softmax_scale,
        "page_size": page_size,
    }
    if case.mode == "decode":
        out = gluon_dsa_decode_gfx950(q_len_per_req=case.q_len_per_req, **common_kwargs)
    else:
        out = gluon_dsa_prefill_gfx950(**common_kwargs)

    ref = _dsa_reference(
        q,
        reference_latent,
        reference_rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, case.num_heads, case.kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)
