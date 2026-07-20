# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
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


def test_dsa_topk_has_no_custom_compiled_runner_cache() -> None:
    removed_symbols = (
        "_CompiledRunnerPlan",
        "_compiled_runner_cache",
        "_launch_warmed_compiled_kernel",
        "_trivial_decode_runner_plans",
        "_runtime_decode_runner_plans",
        "_manual_decode_runner_plans",
        "_trivial_prefill_runner_plans",
        "_runtime_prefill_runner_plans",
        "_manual_prefill_runner_plans",
        "_persistent_runner_plans",
    )

    assert not any(hasattr(dsa_topk_gfx950, name) for name in removed_symbols)


def test_dsa_topk_has_no_dedicated_trivial_specializations() -> None:
    assert not hasattr(dsa_topk_gfx950, "_dsa_trivial_decode_topk2048_kernel")
    assert not hasattr(dsa_topk_gfx950, "_dsa_trivial_prefill_topk2048_kernel")


def test_dsa_topk_has_no_superseded_decode_paths() -> None:
    removed_symbols = (
        "_dsa_decode_staged_wide_radix_topk_slots",
        "_dsa_decode_radix_topk_slots",
        "_dsa_persistent_radix_topk_row_pool_kernel",
        "_dsa_persistent_radix_topk_heterogeneous_kernel",
        "_dsa_persistent_decode_topk_split_rows",
        "_dsa_persistent_decode_topk_heterogeneous",
        "_persistent_decode_split_plan",
        "_persistent_decode_heterogeneous_plan",
        "_persistent_decode_interleaved_plan",
        "_persistent_decode_interleaved_current_groups_plan",
        "_dsa_decode_select_topk_kernel",
        "_dsa_prefill_select_topk_kernel",
    )

    assert not any(hasattr(dsa_topk_gfx950, name) for name in removed_symbols)


def test_dsa_topk_has_no_superseded_grouped_prefill_path() -> None:
    removed_symbols = (
        "_PREFILL_RADIX_BLOCK_N",
        "_PREFILL_RADIX_HIST_TARGET_GROUPS_PER_CU",
        "_dsa_prefill_radix_scatter_kernel",
        "_run_prefill_wide_radix_prefix_passes",
        "_dsa_prefill_radix_topk",
    )

    assert not any(hasattr(dsa_topk_gfx950, name) for name in removed_symbols)


def test_dsa_manual_decode_config_source_contract() -> None:
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_LONG_RUNTIME_CONFIG == (8192, 8)
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_SHORT_MANUAL_CONFIG == (8192, 8)
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_LONG_MANUAL_CONFIG == (8192, 4)
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_SHORT_MANUAL_MAX_COLS == 65536
    assert dsa_topk_gfx950._ONEBLOCK_RADIX_SCHEDULE == (12, 12, 8)
    assert dsa_topk_gfx950._ONEBLOCK_PREFILL_RADIX_BLOCK_N == 4096
    assert dsa_topk_gfx950._ONEBLOCK_RADIX_MAX_COLS == 90000
    assert dsa_topk_gfx950._ONEBLOCK_DECODE_RUNTIME_MAX_COLS == 256 * 1024

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


@pytest.mark.parametrize(
    ("invalid_input", "invalid_kind"),
    (
        ("q", "strided"),
        ("weights", "strided"),
        ("index_k_cache", "strided"),
        ("seq_lens", "strided"),
        ("block_table", "strided"),
        ("seq_lens", "int64"),
        ("block_table", "int64"),
    ),
)
def test_dsa_decode_topk_fp8_rejects_noncanonical_inputs(
    invalid_input: str,
    invalid_kind: str,
) -> None:
    device = "cuda"
    page_size = 64
    topk = 512
    inputs = {
        "q": torch.empty((2, 2, 128), device=device, dtype=torch.bfloat16),
        "weights": torch.empty((2, 2), device=device, dtype=torch.float32),
        "index_k_cache": torch.empty((128, 132), device=device, dtype=torch.uint8),
        "seq_lens": torch.tensor([64, 64], device=device, dtype=torch.int32),
        "block_table": torch.zeros((2, 2), device=device, dtype=torch.int32),
    }
    tensor = inputs[invalid_input]
    if invalid_kind == "strided":
        inputs[invalid_input] = (
            _strided_1d(tensor) if tensor.dim() == 1 else _strided_last_dim(tensor)
        )
        error = ValueError
        match = "contiguous"
    else:
        inputs[invalid_input] = tensor.to(torch.int64)
        error = TypeError
        match = "int32"

    with pytest.raises(error, match=match):
        gluon_dsa_decode_topk_fp8_gfx950(
            inputs["q"],
            inputs["weights"],
            inputs["seq_lens"],
            inputs["block_table"],
            page_size=page_size,
            topk=topk,
            softmax_scale=128**-0.5,
            q_len_per_req=1,
            index_k_cache=inputs["index_k_cache"],
        )


@pytest.mark.parametrize(
    ("invalid_capability", "match"),
    (
        ("q_dtype", "BF16 q"),
        ("weights_dtype", "FP32 weights"),
        ("head_dim", "head_dim=128"),
        ("page_size", "page_size=64"),
        ("topk", "supports topk"),
        ("q_len_per_req", "q_len_per_req"),
    ),
)
def test_dsa_decode_topk_fp8_rejects_unsupported_capabilities(
    invalid_capability: str,
    match: str,
) -> None:
    q = torch.empty((2, 2, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.empty((2, 2), device="cuda", dtype=torch.float32)
    if invalid_capability == "q_dtype":
        q = q.to(torch.float16)
    elif invalid_capability == "weights_dtype":
        weights = weights.to(torch.bfloat16)
    elif invalid_capability == "head_dim":
        q = torch.empty((2, 2, 64), device="cuda", dtype=torch.bfloat16)

    with pytest.raises((TypeError, ValueError), match=match):
        gluon_dsa_decode_topk_fp8_gfx950(
            q,
            weights,
            torch.tensor([64, 64], device="cuda", dtype=torch.int32),
            torch.zeros((2, 2), device="cuda", dtype=torch.int32),
            page_size=32 if invalid_capability == "page_size" else 64,
            topk=256 if invalid_capability == "topk" else 512,
            softmax_scale=128**-0.5,
            q_len_per_req=7 if invalid_capability == "q_len_per_req" else 1,
            index_k_cache=torch.empty((128, 132), device="cuda", dtype=torch.uint8),
        )


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


@pytest.mark.parametrize(
    ("invalid_input", "invalid_kind"),
    (
        ("q", "strided"),
        ("weights", "strided"),
        ("index_k_cache", "strided"),
        ("kv_workspace_slots", "strided"),
        ("row_starts", "strided"),
        ("row_ends", "strided"),
        ("kv_workspace_slots", "int32"),
        ("row_starts", "int64"),
        ("row_ends", "int64"),
    ),
)
def test_dsa_prefill_topk_fp8_rejects_noncanonical_inputs(
    invalid_input: str,
    invalid_kind: str,
) -> None:
    device = "cuda"
    page_size = 64
    topk = 512
    inputs = {
        "q": torch.empty((2, 2, 128), device=device, dtype=torch.bfloat16),
        "weights": torch.empty((2, 2), device=device, dtype=torch.float32),
        "index_k_cache": torch.empty((128, 132), device=device, dtype=torch.uint8),
        "kv_workspace_slots": torch.arange(128, device=device, dtype=torch.int64),
        "row_starts": torch.tensor([0, 64], device=device, dtype=torch.int32),
        "row_ends": torch.tensor([64, 128], device=device, dtype=torch.int32),
    }
    tensor = inputs[invalid_input]
    if invalid_kind == "strided":
        inputs[invalid_input] = (
            _strided_1d(tensor) if tensor.dim() == 1 else _strided_last_dim(tensor)
        )
        error = ValueError
        match = "contiguous"
    else:
        dtype = torch.int32 if invalid_kind == "int32" else torch.int64
        inputs[invalid_input] = tensor.to(dtype)
        error = TypeError
        match = "must be int"

    with pytest.raises(error, match=match):
        gluon_dsa_prefill_topk_fp8_gfx950(
            inputs["q"],
            inputs["weights"],
            inputs["kv_workspace_slots"],
            inputs["row_starts"],
            inputs["row_ends"],
            topk=topk,
            softmax_scale=128**-0.5,
            index_k_cache=inputs["index_k_cache"],
            page_size=page_size,
        )


@pytest.mark.parametrize(
    ("invalid_capability", "match"),
    (
        ("q_dtype", "BF16 q"),
        ("weights_dtype", "FP32 weights"),
        ("head_dim", "head_dim=128"),
        ("page_size", "page_size=64"),
        ("topk", "supports topk"),
    ),
)
def test_dsa_prefill_topk_fp8_rejects_unsupported_capabilities(
    invalid_capability: str,
    match: str,
) -> None:
    q = torch.empty((2, 2, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.empty((2, 2), device="cuda", dtype=torch.float32)
    if invalid_capability == "q_dtype":
        q = q.to(torch.float16)
    elif invalid_capability == "weights_dtype":
        weights = weights.to(torch.bfloat16)
    elif invalid_capability == "head_dim":
        q = torch.empty((2, 2, 64), device="cuda", dtype=torch.bfloat16)

    with pytest.raises((TypeError, ValueError), match=match):
        gluon_dsa_prefill_topk_fp8_gfx950(
            q,
            weights,
            torch.arange(128, device="cuda", dtype=torch.int64),
            torch.tensor([0, 64], device="cuda", dtype=torch.int32),
            torch.tensor([64, 128], device="cuda", dtype=torch.int32),
            topk=256 if invalid_capability == "topk" else 512,
            softmax_scale=128**-0.5,
            index_k_cache=torch.empty((128, 132), device="cuda", dtype=torch.uint8),
            page_size=32 if invalid_capability == "page_size" else 64,
        )


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


def _assert_persistent_workspace_layout(
    workspace: tuple[torch.Tensor, ...],
    rows: int,
) -> None:
    row_bucket = dsa_topk_gfx950._next_power_of_two(rows)
    num_passes = dsa_topk_gfx950._PERSISTENT_PREFILL_NUM_PASSES.value
    num_buckets = dsa_topk_gfx950._PERSISTENT_PREFILL_NUM_BUCKETS.value
    counter_stride = dsa_topk_gfx950._PERSISTENT_PREFILL_COUNTER_STRIDE.value
    histograms, pass_arrivals, pass_done, reset_arrivals, output_counters = workspace

    assert histograms.shape == (row_bucket, num_passes, num_buckets)
    assert pass_arrivals.shape == (row_bucket, counter_stride)
    assert pass_done.shape == (row_bucket, counter_stride)
    assert reset_arrivals.shape == (row_bucket, counter_stride)
    assert output_counters.shape == (row_bucket, 2, counter_stride)


def _assert_persistent_workspace_reset(workspace: tuple[torch.Tensor, ...]) -> None:
    assert all(int(torch.count_nonzero(tensor).item()) == 0 for tensor in workspace)


def test_dsa_persistent_workspace_uses_one_monotonic_pass_arrival_per_row() -> None:
    rows = 33
    workspace = dsa_topk_gfx950._persistent_topk_workspace(
        rows,
        torch.device("cuda"),
    )

    _assert_persistent_workspace_layout(workspace, rows)
    _assert_persistent_workspace_reset(workspace)


def test_dsa_persistent_pass_arrival_uses_monotonic_generations() -> None:
    source = inspect.getsource(dsa_topk_gfx950._dsa_persistent_radix_topk_row.fn)

    assert "generation_last_arrival = (pass_index + 1) * GROUPS_PER_ROW - 1" in source
    assert "gl.atomic_xchg(" not in source
    assert source.count("pass_arrivals + row * _PERSISTENT_PREFILL_COUNTER_STRIDE") == 2


@pytest.mark.parametrize(
    ("rows", "cols", "expected_groups"),
    (
        (32, 131072, 2),
        (64, 131072, 2),
        (64, 147493, 2),
        (64, 262144, 4),
        (64, 524288, 4),
        (64, 1048576, 4),
        (32, 1048577, 4),
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
    _assert_persistent_workspace_layout(workspace, rows)
    _assert_persistent_workspace_reset(workspace)


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


def test_dsa_decode_topk_dispatches_persistent_radix_for_batched_queries() -> None:
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


@pytest.mark.parametrize("cols", (512, 1024, 2048))
@pytest.mark.parametrize("q_len_per_req", (1, 4))
def test_dsa_decode_topk_trivial_2048_maps_grouped_queries_to_physical_slots(
    cols: int,
    q_len_per_req: int,
) -> None:
    page_size = 64
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


@pytest.mark.parametrize("cols", (512, 1024, 2048))
def test_dsa_prefill_topk_trivial_2048_maps_candidate_ranges(cols: int) -> None:
    rows = 4
    topk = 2048
    row_starts = torch.tensor(
        [0, 3, 17, cols // 2],
        device="cuda",
        dtype=torch.int32,
    )
    row_ends = torch.tensor(
        [cols, cols - 5, cols - 1, cols // 2],
        device="cuda",
        dtype=torch.int32,
    )
    logits = torch.zeros((rows, cols), device="cuda", dtype=torch.float32)
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

    expected = torch.full_like(out, -1)
    expected_lens = row_ends - row_starts
    for row in range(rows):
        count = int(expected_lens[row].item())
        expected[row, :count] = torch.arange(
            int(row_starts[row].item()),
            int(row_ends[row].item()),
            device="cuda",
            dtype=torch.int32,
        )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(lens_out, expected_lens, rtol=0, atol=0)


@pytest.mark.parametrize(
    "cols",
    [8192, 16384, 32768, 65536, 65537, 90000, 131072, 524288],
)
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


@pytest.mark.parametrize(
    ("rows", "cols", "topk"),
    (
        pytest.param(5, 512 * 1024, 512, id="topk512-tail"),
        pytest.param(3, 512 * 1024, 1024, id="topk1024-tail"),
        pytest.param(129, 256 * 1024 + 64, 2048, id="high-row-tail"),
    ),
)
def test_dsa_decode_topk_persistent_replaces_staged_wide(
    rows: int,
    cols: int,
    topk: int,
) -> None:
    page_size = 64
    seq_lens = (
        cols - (torch.arange(rows, device="cuda", dtype=torch.int32) * 53 + 17) % 1024
    )
    row_starts = torch.zeros((rows,), device="cuda", dtype=torch.int32)
    logits = _make_grouped_radix_logits(
        row_starts,
        seq_lens,
        cols=cols,
        topk=topk,
        seed=3907 + topk,
    )
    block_table = _make_reversed_decode_block_table(rows, cols, page_size)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    assert dsa_topk_gfx950._use_persistent_decode(
        rows,
        cols,
        topk,
        logits.device,
    )
    dsa_topk_gfx950._dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=1,
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
        q_len_per_req=1,
        topk=topk,
    )
    workspace = dsa_topk_gfx950._persistent_topk_workspace(rows, logits.device)
    _assert_persistent_workspace_reset(workspace)


@pytest.mark.parametrize(
    ("rows", "cols", "topk", "expected"),
    (
        (1, 90000, 2048, False),
        (1, 90001, 2048, True),
        (128, 131072, 2048, True),
        (129, 131072, 2048, False),
        (1, 262144, 1024, False),
        (1, 262145, 512, True),
        (514, 1024 * 1024, 2048, True),
    ),
)
def test_dsa_persistent_decode_preserves_oneblock_shape_boundary(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    cols: int,
    topk: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(dsa_topk_gfx950, "_device_compute_units", lambda _: 256)

    assert (
        dsa_topk_gfx950._use_persistent_decode(
            rows,
            cols,
            topk,
            torch.device("cuda", 0),
        )
        is expected
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

    assert dsa_topk_gfx950._use_persistent_decode(
        rows,
        cols,
        topk,
        logits.device,
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
    _assert_persistent_workspace_layout(workspace, rows)
    _assert_persistent_workspace_reset(workspace)


@pytest.mark.parametrize(
    ("rows", "cols", "topk", "expected"),
    (
        (33, 90048, 2048, (32, 8, 32, 1, 8, 1)),
        (36, 90048, 2048, (32, 8, 32, 4, 8, 4)),
        (5, 1024 * 1024, 2048, (4, 64, 4, 1, 64, 1)),
        (33, 512 * 1024, 512, (32, 8, 32, 1, 32, 1)),
        (65, 512 * 1024, 1024, (64, 4, 64, 1, 32, 1)),
        (129, 262208, 2048, (128, 2, 128, 1, 32, 1)),
        (129, 512 * 1024, 2048, (128, 2, 128, 1, 32, 1)),
        (132, 512 * 1024, 2048, (128, 2, 128, 4, 32, 4)),
        (128, 512 * 1024, 2048, (128, 2, 128, 0, 2, 0)),
        (192, 262208, 2048, (128, 2, 128, 64, 4, 64)),
        (129, 1024 * 1024, 1024, (128, 2, 128, 1, 64, 1)),
        (514, 1024 * 1024, 2048, (512, 2, 128, 2, 64, 2)),
    ),
)
def test_dsa_persistent_interleaved_plan_reuses_resident_workgroups(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    cols: int,
    topk: int,
    expected: tuple[int, int, int, int, int, int],
) -> None:
    monkeypatch.setattr(dsa_topk_gfx950, "_device_compute_units", lambda _: 256)

    plan = dsa_topk_gfx950._persistent_interleaved_plan(
        rows,
        cols,
        topk,
        torch.device("cuda", 0),
    )

    assert plan == expected
    assert plan is not None
    (
        main_rows,
        main_groups,
        main_row_teams,
        tail_rows,
        tail_groups,
        tail_row_teams,
    ) = plan
    assert main_rows + tail_rows == rows
    assert main_groups & (main_groups - 1) == 0
    assert tail_groups & (tail_groups - 1) == 0
    assert main_groups * main_row_teams <= 256
    assert tail_groups * tail_row_teams <= main_groups * main_row_teams


@pytest.mark.parametrize(
    ("rows", "cols", "topk"),
    ((0, 90048, 2048), (1, 90000, 2048), (1, 90048, 256)),
)
def test_dsa_persistent_interleaved_plan_rejects_unsupported_inputs(
    rows: int,
    cols: int,
    topk: int,
) -> None:
    assert (
        dsa_topk_gfx950._persistent_interleaved_plan(
            rows,
            cols,
            topk,
            torch.device("cuda", 0),
        )
        is None
    )


def test_dsa_persistent_interleaved_mode_and_groups_are_constexpr() -> None:
    params = {
        param.name: param
        for param in (
            dsa_topk_gfx950._dsa_persistent_radix_topk_interleaved_kernel.params
        )
    }

    assert params["MAIN_GROUPS_PER_ROW"].is_constexpr
    assert params["TAIL_GROUPS_PER_ROW"].is_constexpr
    assert params["IS_DECODE"].is_constexpr


def test_dsa_persistent_interleaved_decode_uses_one_sided_tile_masks() -> None:
    helpers = (
        dsa_topk_gfx950._persistent_interleaved_publish_histogram,
        dsa_topk_gfx950._persistent_interleaved_emit_row,
    )

    for helper in helpers:
        source = inspect.getsource(helper.fn)
        assert source.count("valid = offsets < row_end") == 1
        assert source.count("valid = (offsets >= row_start) & (offsets < row_end)") == 1


@pytest.mark.parametrize(
    ("rows", "cols", "topk"),
    (
        (33, 90048, 512),
        (65, 90048, 1024),
        (129, 90048, 2048),
        (514, 90048, 2048),
        (129, 262208, 2048),
    ),
)
def test_dsa_persistent_prefill_interleaved_repeat_and_reset(
    rows: int,
    cols: int,
    topk: int,
) -> None:
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = 257 + row_ids % 113
    row_ends = cols - (rows - 1 - row_ids) % 1021
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=8977 + rows + topk,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)
    assert (
        dsa_topk_gfx950._persistent_prefill_groups(
            rows,
            cols,
            topk,
            logits.device,
        )
        is None
    )
    assert cols > dsa_topk_gfx950._ONEBLOCK_RADIX_MAX_COLS
    assert not (
        dsa_topk_gfx950._PREFILL_RUNTIME_RADIX_MIN_COLS
        <= cols
        <= dsa_topk_gfx950._PREFILL_RUNTIME_RADIX_MAX_COLS
    )
    assert cols < dsa_topk_gfx950._PREFILL_HIST_DERIVED_MIN_COLS
    assert (
        dsa_topk_gfx950._persistent_interleaved_plan(
            rows,
            cols,
            topk,
            logits.device,
        )
        is not None
    )

    for _ in range(2):
        out.fill_(-7)
        lens_out.fill_(-7)
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
        workspace = dsa_topk_gfx950._persistent_topk_workspace(
            rows,
            logits.device,
        )
        _assert_persistent_workspace_layout(workspace, rows)
        _assert_persistent_workspace_reset(workspace)


def test_dsa_persistent_prefill_interleaved_respects_causal_ranges() -> None:
    rows = 129
    cols = 90048
    topk = 512
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = 193 + row_ids % 97
    row_lens = torch.full_like(row_starts, topk + 65)
    row_lens[0] = 0
    row_lens[1] = topk - 1
    row_lens[2] = topk
    row_lens[64] = topk - 1
    row_lens[-1] = topk + 47
    row_ends = row_starts + row_lens
    logits = torch.full(
        (rows, cols),
        8.0,
        device="cuda",
        dtype=torch.float32,
    )
    for row, (start, end) in enumerate(
        zip(row_starts.cpu().tolist(), row_ends.cpu().tolist(), strict=True)
    ):
        if start != end:
            logits[row, start:end] = torch.linspace(
                -1.0,
                1.0,
                end - start,
                device="cuda",
            )
    out = torch.full((rows, topk), -7, device="cuda", dtype=torch.int32)
    lens_out = torch.full((rows,), -7, device="cuda", dtype=torch.int32)

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
    _assert_persistent_workspace_reset(workspace)


@pytest.mark.parametrize("warm_workspace", (False, True), ids=("cold-key", "warm-key"))
def test_dsa_persistent_prefill_interleaved_is_graph_capturable(
    warm_workspace: bool,
) -> None:
    rows = 33
    cols = 90048
    topk = 512
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    row_starts = 131 + row_ids * 7
    row_ends = cols - (rows - 1 - row_ids) * 19
    logits = _make_grouped_radix_logits(
        row_starts,
        row_ends,
        cols=cols,
        topk=topk,
        seed=8999,
    )
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    def invoke() -> None:
        dsa_topk_gfx950._dsa_prefill_topk_indices(
            logits,
            row_starts,
            row_ends,
            topk=topk,
            out=out,
            lens_out=lens_out,
        )

    invoke()
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
            invoke()
        capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        invoke()
    capture_stream.synchronize()
    graph_workspace = dsa_topk_gfx950._persistent_topk_workspace_cache[workspace_key]
    _assert_persistent_workspace_layout(graph_workspace, rows)

    for _ in range(2):
        out.fill_(-7)
        lens_out.fill_(-7)
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
        _assert_persistent_workspace_reset(graph_workspace)


@pytest.mark.parametrize(
    ("rows", "cols", "q_len_per_req", "final_seq_len"),
    (
        (33, 90048, 3, 2049),
        (128, 90048, 4, 2049),
        (132, 90048, 4, 90031),
        (514, 90048, 1, 90031),
        (5, 1024 * 1024, 1, 1024 * 1024 - 17),
        (129, 1024 * 1024, 1, 1024 * 1024 - 17),
    ),
)
def test_dsa_persistent_decode_interleaved_repeat_and_reset(
    rows: int,
    cols: int,
    q_len_per_req: int,
    final_seq_len: int,
) -> None:
    page_size = 64
    topk = 2048
    requests = rows // q_len_per_req
    request_ids = torch.arange(requests, device="cuda", dtype=torch.int32)
    seq_lens = cols - (request_ids * 53 + 17) % 1024
    seq_lens[-1] = final_seq_len
    generator = _generator("cuda", 8957 + rows)
    logits = torch.randn(
        (rows, cols),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    block_table = _make_reversed_decode_block_table(requests, cols, page_size)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)
    for _ in range(2):
        out.fill_(-7)
        lens_out.fill_(-7)
        dsa_topk_gfx950._dsa_persistent_decode_topk_slots(
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
        _assert_persistent_workspace_reset(workspace)


def test_dsa_persistent_decode_interleaved_refines_exact_first_pass_bucket() -> None:
    rows = 33
    cols = 90048
    topk = 2048
    page_size = 64
    logits = torch.zeros((rows, cols), device="cuda", dtype=torch.float32)
    logits[:, :topk] = 1.0
    seq_lens = torch.full((rows,), cols, device="cuda", dtype=torch.int32)
    block_table = _make_reversed_decode_block_table(rows, cols, page_size)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)
    dsa_topk_gfx950._dsa_persistent_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=1,
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
        q_len_per_req=1,
        topk=topk,
    )
    workspace = dsa_topk_gfx950._persistent_topk_workspace(rows, logits.device)
    _assert_persistent_workspace_reset(workspace)


@pytest.mark.parametrize("warm_workspace", (False, True), ids=("cold-key", "warm-key"))
def test_dsa_persistent_decode_interleaved_is_graph_capturable(
    warm_workspace: bool,
) -> None:
    page_size = 64
    q_len_per_req = 3
    requests = 11
    rows = requests * q_len_per_req
    cols = 90048
    topk = 2048
    request_ids = torch.arange(requests, device="cuda", dtype=torch.int32)
    seq_lens = cols - (request_ids * 53 + 17) % 1024
    seq_lens[-1] = 2049
    generator = _generator("cuda", 8967)
    logits = torch.randn(
        (rows, cols),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    block_table = _make_reversed_decode_block_table(requests, cols, page_size)
    out = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    lens_out = torch.empty((rows,), device="cuda", dtype=torch.int32)

    def invoke() -> None:
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

    invoke()
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
            invoke()
        capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        invoke()
    capture_stream.synchronize()
    graph_workspace = dsa_topk_gfx950._persistent_topk_workspace_cache[workspace_key]
    _assert_persistent_workspace_layout(graph_workspace, rows)
    if warm_workspace:
        _assert_persistent_workspace_reset(graph_workspace)

    for _ in range(2):
        out.fill_(-7)
        lens_out.fill_(-7)
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
        _assert_persistent_workspace_reset(graph_workspace)


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
    _assert_persistent_workspace_layout(workspace_a, rows)
    _assert_persistent_workspace_layout(workspace_b, rows)
    _assert_persistent_workspace_reset(workspace_a)
    _assert_persistent_workspace_reset(workspace_b)


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


@pytest.mark.parametrize("cols", [131072, 262144], ids=["two-groups", "four-groups"])
def test_dsa_prefill_homogeneous_persistent_topk_is_graph_capturable(
    cols: int,
) -> None:
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


@pytest.mark.parametrize("cols", [131072, 262144], ids=["two-groups", "four-groups"])
def test_dsa_prefill_homogeneous_persistent_topk_is_stream_local(cols: int) -> None:
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
    graph_workspace = dsa_topk_gfx950._persistent_topk_workspace_cache[workspace_key]
    _assert_persistent_workspace_layout(graph_workspace, rows)
    if warm_workspace:
        _assert_persistent_workspace_reset(graph_workspace)
    if warm_workspace:
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
    _assert_persistent_workspace_reset(graph_workspace)


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
    _assert_persistent_workspace_layout(workspace_a, rows)
    _assert_persistent_workspace_layout(workspace_b, rows)
    _assert_persistent_workspace_reset(workspace_a)
    _assert_persistent_workspace_reset(workspace_b)


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
