#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from dsa_topk_workloads import (
    CASES_BY_NAME,
    DECODE_CASES,
    PAGE_SIZE,
    PREFILL_CASES,
    TOPK,
    DecodeCase,
    PrefillCase,
    prefill_launch_rows,
    prefill_launch_slices,
    prefill_row_ranges,
)
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    local_topk_to_global_slots,
)
from tokenspeed_kernel_amd.ops.attention.gluon import dsa_topk_gfx950

BACKENDS = ("gluon", "aiter_raw", "aiter_contract")


@dataclass(frozen=True)
class AiterOps:
    decode: Callable[..., None]
    prefill: Callable[..., None]
    use_multiblock: Callable[[int, int], bool]


@dataclass
class DecodeState:
    case: DecodeCase
    logits: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    gluon_out: torch.Tensor
    gluon_lens: torch.Tensor
    aiter_raw_out: torch.Tensor
    aiter_contract_out: torch.Tensor
    aiter_contract_lens: torch.Tensor


@dataclass
class PrefillState:
    case: PrefillCase
    logits: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    launches: tuple[tuple[int, int], ...]
    gluon_out: torch.Tensor
    gluon_lens: torch.Tensor
    aiter_raw_out: torch.Tensor
    aiter_contract_out: torch.Tensor
    aiter_contract_lens: torch.Tensor


@triton.jit
def _write_prefill_lens_kernel(
    row_starts,
    row_ends,
    lens_out,
    rows,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < rows
    starts = tl.load(row_starts + offsets, mask=valid, other=0)
    ends = tl.load(row_ends + offsets, mask=valid, other=0)
    lengths = tl.maximum(ends - starts, 0)
    tl.store(lens_out + offsets, tl.minimum(lengths, topk), mask=valid)


def _load_aiter_ops() -> AiterOps:
    try:
        from aiter.ops.topk import (
            top_k_per_row_decode,
            top_k_per_row_prefill,
            topk_use_mulblocks,
        )
    except ImportError as error:
        raise RuntimeError(
            "AITER is required for this comparison. Add its checkout to PYTHONPATH."
        ) from error
    return AiterOps(
        decode=top_k_per_row_decode,
        prefill=top_k_per_row_prefill,
        use_multiblock=topk_use_mulblocks,
    )


def _stable_seed(name: str) -> int:
    return 2_718 + sum((index + 1) * ord(char) for index, char in enumerate(name))


def _empty_outputs(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    return (
        torch.empty((rows, TOPK), dtype=torch.int32, device=device),
        torch.empty((rows,), dtype=torch.int32, device=device),
    )


def _build_decode_state(case: DecodeCase) -> DecodeState:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(_stable_seed(case.name))
    logits = torch.empty(
        (case.rows, case.context_len), dtype=torch.float32, device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    seq_lens = torch.tensor(case.seq_lens, dtype=torch.int32, device=device)
    pages = case.context_len // PAGE_SIZE
    logical_pages = torch.arange(pages, dtype=torch.int32, device=device).flip(0)
    request_offsets = (
        torch.arange(case.requests, dtype=torch.int32, device=device) * pages
    )
    block_table = request_offsets[:, None] + logical_pages[None, :]
    gluon_out, gluon_lens = _empty_outputs(case.rows)
    aiter_raw_out, _ = _empty_outputs(case.rows)
    aiter_contract_out, aiter_contract_lens = _empty_outputs(case.rows)
    return DecodeState(
        case=case,
        logits=logits,
        seq_lens=seq_lens,
        block_table=block_table.contiguous(),
        gluon_out=gluon_out,
        gluon_lens=gluon_lens,
        aiter_raw_out=aiter_raw_out,
        aiter_contract_out=aiter_contract_out,
        aiter_contract_lens=aiter_contract_lens,
    )


def _build_prefill_state(case: PrefillCase) -> PrefillState:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(_stable_seed(case.name))
    launches = prefill_launch_slices(case)
    max_launch_rows = max(end - start for start, end in launches)
    logits = torch.empty(
        (max_launch_rows, case.context_len),
        dtype=torch.float32,
        device=device,
    ).uniform_(-1.0, 1.0, generator=generator)
    starts, ends = prefill_row_ranges(case)
    row_starts = torch.tensor(starts, dtype=torch.int32, device=device)
    row_ends = torch.tensor(ends, dtype=torch.int32, device=device)
    gluon_out, gluon_lens = _empty_outputs(case.rows)
    aiter_raw_out, _ = _empty_outputs(case.rows)
    aiter_contract_out, aiter_contract_lens = _empty_outputs(case.rows)
    return PrefillState(
        case=case,
        logits=logits,
        row_starts=row_starts,
        row_ends=row_ends,
        launches=launches,
        gluon_out=gluon_out,
        gluon_lens=gluon_lens,
        aiter_raw_out=aiter_raw_out,
        aiter_contract_out=aiter_contract_out,
        aiter_contract_lens=aiter_contract_lens,
    )


def _invoke_decode_gluon(state: DecodeState) -> None:
    case = state.case
    dsa_topk_gfx950._dsa_topk_indices(
        state.logits,
        state.seq_lens,
        state.seq_lens,
        block_table=state.block_table,
        page_size=PAGE_SIZE,
        topk=TOPK,
        q_len_per_req=case.q_len_per_req,
        out=state.gluon_out,
        lens_out=state.gluon_lens,
    )


def _invoke_decode_aiter(
    state: DecodeState,
    ops: AiterOps,
    out: torch.Tensor,
) -> None:
    case = state.case
    ops.decode(
        state.logits,
        case.q_len_per_req,
        state.seq_lens,
        out,
        case.rows,
        state.logits.stride(0),
        state.logits.stride(1),
        k=TOPK,
    )


def _invoke_decode_aiter_contract(state: DecodeState, ops: AiterOps) -> None:
    _invoke_decode_aiter(state, ops, state.aiter_raw_out)
    local_topk_to_global_slots(
        local_topk_offsets=state.aiter_raw_out,
        block_table=state.block_table,
        block_size=PAGE_SIZE,
        seq_lens=state.seq_lens,
        q_len_per_req=state.case.q_len_per_req,
        out=state.aiter_contract_out,
        lens_out=state.aiter_contract_lens,
    )


def _invoke_prefill_gluon(state: PrefillState) -> None:
    for start, end in state.launches:
        rows = end - start
        dsa_topk_gfx950._dsa_topk_indices(
            state.logits[:rows],
            state.row_starts[start:end],
            state.row_ends[start:end],
            topk=TOPK,
            out=state.gluon_out[start:end],
            lens_out=state.gluon_lens[start:end],
        )


def _invoke_prefill_aiter(
    state: PrefillState,
    ops: AiterOps,
    out: torch.Tensor,
) -> None:
    for start, end in state.launches:
        rows = end - start
        logits = state.logits[:rows]
        ops.prefill(
            logits,
            state.row_starts[start:end],
            state.row_ends[start:end],
            out[start:end],
            None,
            rows,
            logits.stride(0),
            logits.stride(1),
            k=TOPK,
        )


def _invoke_prefill_aiter_contract(state: PrefillState, ops: AiterOps) -> None:
    _invoke_prefill_aiter(state, ops, state.aiter_contract_out)
    block = 256
    _write_prefill_lens_kernel[(triton.cdiv(state.case.rows, block),)](
        state.row_starts,
        state.row_ends,
        state.aiter_contract_lens,
        state.case.rows,
        topk=TOPK,
        BLOCK=block,
        num_warps=4,
    )


def _decode_slots_to_logical(
    state: DecodeState,
    slots: torch.Tensor,
) -> torch.Tensor:
    case = state.case
    pages = case.context_len // PAGE_SIZE
    row_requests = (
        torch.arange(case.rows, dtype=torch.int32, device=slots.device)
        // case.q_len_per_req
    )
    valid = slots >= 0
    physical_page = torch.div(slots, PAGE_SIZE, rounding_mode="floor")
    page_offset = slots % PAGE_SIZE
    request_page_base = row_requests[:, None] * pages
    logical_page = pages - 1 - (physical_page - request_page_base)
    logical = logical_page * PAGE_SIZE + page_offset
    return torch.where(valid, logical, -1)


def _validate_index_matrix(
    indices: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    lens: torch.Tensor | None,
) -> torch.Tensor:
    expected_lens = torch.clamp(ends - starts, min=0, max=TOPK)
    if lens is not None:
        torch.testing.assert_close(lens, expected_lens, rtol=0, atol=0)

    columns = torch.arange(TOPK, device=indices.device)[None, :]
    valid = columns < expected_lens[:, None]
    outside = valid & ((indices < starts[:, None]) | (indices >= ends[:, None]))
    if bool(outside.any()):
        raise AssertionError("selected index is outside its candidate range")
    if bool(((~valid) & (indices != -1)).any()):
        raise AssertionError("output tail is not -1")

    sentinel = torch.iinfo(indices.dtype).max
    ordered = torch.sort(torch.where(valid, indices, sentinel), dim=1).values
    adjacent_valid = columns[:, 1:] < expected_lens[:, None]
    duplicates = adjacent_valid & (ordered[:, 1:] == ordered[:, :-1])
    if bool(duplicates.any()):
        raise AssertionError("selected indices are not unique")
    return expected_lens


def _decode_validation_rows(case: DecodeCase) -> tuple[int, ...]:
    requests = sorted({0, case.requests // 2, case.requests - 1})
    return tuple(
        sorted(
            {
                request * case.q_len_per_req + q_offset
                for request in requests
                for q_offset in (0, case.q_len_per_req - 1)
            }
        )
    )


def _validate_decode_values(
    state: DecodeState,
    logical_indices: torch.Tensor,
    expected_lens: torch.Tensor,
) -> None:
    for row in _decode_validation_rows(state.case):
        count = int(expected_lens[row].item())
        if count == 0:
            continue
        end = state.case.candidate_lens[row]
        selected = logical_indices[row, :count].long()
        actual = torch.sort(state.logits[row].index_select(0, selected)).values
        expected = torch.sort(
            torch.topk(state.logits[row, :end], count, sorted=False).values
        ).values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _validate_decode(state: DecodeState) -> None:
    starts = torch.zeros(state.case.rows, dtype=torch.int32, device="cuda")
    ends = torch.tensor(state.case.candidate_lens, dtype=torch.int32, device="cuda")
    outputs = (
        (
            _decode_slots_to_logical(state, state.gluon_out),
            state.gluon_lens,
        ),
        (state.aiter_raw_out, None),
        (
            _decode_slots_to_logical(state, state.aiter_contract_out),
            state.aiter_contract_lens,
        ),
    )
    for indices, lens in outputs:
        expected_lens = _validate_index_matrix(
            indices,
            starts,
            ends,
            lens=lens,
        )
        _validate_decode_values(state, indices, expected_lens)


def _prefill_validation_rows(case: PrefillCase) -> tuple[int, ...]:
    boundaries = {0, case.rows // 2, case.rows - 1}
    request_end = 0
    for extend_len in case.extend_lens:
        if request_end:
            boundaries.add(request_end - 1)
        request_end += extend_len
        boundaries.add(request_end - 1)
        if request_end < case.rows:
            boundaries.add(request_end)
    return tuple(sorted(boundaries))


def _validate_prefill_values(
    state: PrefillState,
    indices: torch.Tensor,
    expected_lens: torch.Tensor,
) -> None:
    max_launch_rows = state.logits.shape[0]
    for row in _prefill_validation_rows(state.case):
        count = int(expected_lens[row].item())
        if count == 0:
            continue
        local_row = row % max_launch_rows
        start = int(state.row_starts[row].item())
        end = int(state.row_ends[row].item())
        selected = indices[row, :count].long()
        actual = torch.sort(state.logits[local_row].index_select(0, selected)).values
        expected = torch.sort(
            torch.topk(state.logits[local_row, start:end], count, sorted=False).values
        ).values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _validate_prefill(state: PrefillState) -> None:
    outputs = (
        (state.gluon_out, state.gluon_lens),
        (state.aiter_raw_out, None),
        (state.aiter_contract_out, state.aiter_contract_lens),
    )
    for indices, lens in outputs:
        expected_lens = _validate_index_matrix(
            indices,
            state.row_starts,
            state.row_ends,
            lens=lens,
        )
        _validate_prefill_values(state, indices, expected_lens)


def _capture(
    invoke: Callable[[], None],
    stream: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    with torch.cuda.stream(stream):
        invoke()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        invoke()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    return graph


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summarize(samples: list[float]) -> dict[str, object]:
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "p10_us": _percentile(samples, 0.10),
        "p90_us": _percentile(samples, 0.90),
        "max_us": max(samples),
        "stdev_us": statistics.pstdev(samples),
        "samples_us": samples,
    }


def _benchmark_invocations(
    invocations: dict[str, Callable[[], None]],
    validate: Callable[[], None],
    reset_outputs: Callable[[], None],
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    repeats: int,
    batch: int,
) -> dict[str, dict[str, object]]:
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {name: _capture(invoke, stream) for name, invoke in invocations.items()}
    validate()

    with torch.cuda.stream(stream):
        for index in range(warmup):
            offset = index % len(BACKENDS)
            order = BACKENDS[offset:] + BACKENDS[:offset]
            for name in order:
                graphs[name].replay()
    stream.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in BACKENDS}
    for sample_index in range(repeats):
        offset = sample_index % len(BACKENDS)
        order = BACKENDS[offset:] + BACKENDS[:offset]
        events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        with torch.cuda.stream(stream):
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(batch):
                    graphs[name].replay()
                end.record()
                events[name] = (start, end)
        events[order[-1]][1].synchronize()
        for name, (start, end) in events.items():
            samples[name].append(float(start.elapsed_time(end)) * 1_000.0 / batch)

    with torch.cuda.stream(stream):
        reset_outputs()
        for name in BACKENDS:
            graphs[name].replay()
    stream.synchronize()
    validate()
    del graphs
    torch.cuda.synchronize()
    return {name: _summarize(values) for name, values in samples.items()}


def _reset_decode_outputs(state: DecodeState) -> None:
    state.gluon_out.fill_(-7)
    state.gluon_lens.fill_(-7)
    state.aiter_raw_out.fill_(-7)
    state.aiter_contract_out.fill_(-7)
    state.aiter_contract_lens.fill_(-7)


def _reset_prefill_outputs(state: PrefillState) -> None:
    state.gluon_out.fill_(-7)
    state.gluon_lens.fill_(-7)
    state.aiter_raw_out.fill_(-7)
    state.aiter_contract_out.fill_(-7)
    state.aiter_contract_lens.fill_(-7)


def _decode_gluon_dispatch(state: DecodeState) -> dict[str, object]:
    case = state.case
    launch = dsa_topk_gfx950._dsa_topk_plan(
        case.rows,
        case.context_len,
        TOPK,
        state.logits.device,
        is_decode=True,
    )
    if launch.kind == "trivial":
        return {"name": "trivial"}
    if launch.kind == "persistent-interleaved":
        plan = dsa_topk_gfx950._persistent_interleaved_plan(
            case.rows,
            case.context_len,
            TOPK,
            state.logits.device,
        )
        return {"name": "persistent-fill-po2", "plan": list(plan)}
    return {"name": "oneblock-manual"}


def _prefill_gluon_dispatch(
    rows: int,
    cols: int,
    device: torch.device,
) -> dict[str, object]:
    launch = dsa_topk_gfx950._dsa_topk_plan(
        rows,
        cols,
        TOPK,
        device,
        is_decode=False,
    )
    if launch.kind == "trivial":
        return {"name": "trivial"}
    if launch.kind == "persistent-homogeneous":
        return {
            "name": "persistent-prefill",
            "groups_per_row": launch.groups_per_row,
        }
    if launch.kind == "persistent-interleaved":
        return {"name": "persistent-interleaved-prefill"}
    if launch.use_compact_final and launch.block_n in (
        dsa_topk_gfx950._ONEBLOCK_PREFILL_WIDE_SHORT_BLOCK_N,
        dsa_topk_gfx950._ONEBLOCK_PREFILL_WIDE_LONG_BLOCK_N,
    ):
        return {
            "name": "oneblock-manual-compact-prefill",
            "block_n": launch.block_n,
        }
    if not launch.use_compact_final:
        return {"name": "oneblock-manual-early-stop"}
    return {"name": "oneblock-manual-compact"}


def _dispatch_groups(
    launch_rows: tuple[int, ...],
    dispatch: Callable[[int], dict[str, object]],
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for rows, count in Counter(launch_rows).items():
        groups.append({"rows": rows, "count": count, "dispatch": dispatch(rows)})
    return groups


def _result_ratios(timing: dict[str, dict[str, object]]) -> dict[str, float]:
    gluon = float(timing["gluon"]["median_us"])
    return {
        "aiter_raw_over_gluon": float(timing["aiter_raw"]["median_us"]) / gluon,
        "aiter_contract_over_gluon": (
            float(timing["aiter_contract"]["median_us"]) / gluon
        ),
    }


def _run_decode_case(
    case: DecodeCase,
    ops: AiterOps,
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    repeats: int,
    batch: int,
) -> dict[str, object]:
    state = _build_decode_state(case)
    invocations = {
        "gluon": lambda: _invoke_decode_gluon(state),
        "aiter_raw": lambda: _invoke_decode_aiter(state, ops, state.aiter_raw_out),
        "aiter_contract": lambda: _invoke_decode_aiter_contract(state, ops),
    }
    timing = _benchmark_invocations(
        invocations,
        lambda: _validate_decode(state),
        lambda: _reset_decode_outputs(state),
        stream,
        warmup=warmup,
        repeats=repeats,
        batch=batch,
    )
    result = {
        "kind": "decode",
        "case": case.name,
        "suite": case.suite,
        "requests": case.requests,
        "rows": case.rows,
        "context_len": case.context_len,
        "q_len_per_req": case.q_len_per_req,
        "seq_lens": list(case.seq_lens),
        "candidate_len_min": min(case.candidate_lens),
        "candidate_len_max": max(case.candidate_lens),
        "gluon_dispatch": _decode_gluon_dispatch(state),
        "aiter_dispatch": "oneblock-radix",
        "timing": timing,
        "validated": True,
    }
    result.update(_result_ratios(timing))
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_prefill_case(
    case: PrefillCase,
    ops: AiterOps,
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    repeats: int,
    batch: int,
) -> dict[str, object]:
    state = _build_prefill_state(case)
    invocations = {
        "gluon": lambda: _invoke_prefill_gluon(state),
        "aiter_raw": lambda: _invoke_prefill_aiter(state, ops, state.aiter_raw_out),
        "aiter_contract": lambda: _invoke_prefill_aiter_contract(state, ops),
    }
    timing = _benchmark_invocations(
        invocations,
        lambda: _validate_prefill(state),
        lambda: _reset_prefill_outputs(state),
        stream,
        warmup=warmup,
        repeats=repeats,
        batch=batch,
    )
    launch_rows = prefill_launch_rows(case)
    candidate_lens = tuple(
        end - start
        for start, end in zip(
            state.row_starts.tolist(), state.row_ends.tolist(), strict=True
        )
    )
    result = {
        "kind": "prefill",
        "case": case.name,
        "suite": case.suite,
        "requests": case.requests,
        "rows": case.rows,
        "context_len": case.context_len,
        "prefix_lens": list(case.prefix_lens),
        "extend_lens": list(case.extend_lens),
        "launch_rows": list(launch_rows),
        "launches": len(launch_rows),
        "candidate_len_min": min(candidate_lens),
        "candidate_len_max": max(candidate_lens),
        "candidate_elements": sum(candidate_lens),
        "gluon_dispatch": _dispatch_groups(
            launch_rows,
            lambda rows: _prefill_gluon_dispatch(
                rows, case.context_len, state.logits.device
            ),
        ),
        "aiter_dispatch": _dispatch_groups(
            launch_rows,
            lambda rows: {
                "name": (
                    "multiblock-radix"
                    if ops.use_multiblock(rows, case.context_len)
                    else "oneblock-radix"
                )
            },
        ),
        "timing": timing,
        "validated": True,
    }
    result.update(_result_ratios(timing))
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    raw = [float(result["aiter_raw_over_gluon"]) for result in results]
    contract = [float(result["aiter_contract_over_gluon"]) for result in results]
    return {
        "cases": len(results),
        "gluon_speedup_vs_aiter_raw_geomean": _geomean(raw),
        "gluon_speedup_vs_aiter_contract_geomean": _geomean(contract),
        "gluon_wins_vs_aiter_raw": sum(value > 1.0 for value in raw),
        "gluon_wins_vs_aiter_contract": sum(value > 1.0 for value in contract),
    }


def _aggregates(results: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {"all": results}
    for result in results:
        groups.setdefault(str(result["kind"]), []).append(result)
        groups.setdefault(str(result["suite"]), []).append(result)
    return {name: _aggregate(group) for name, group in groups.items()}


def _select_cases(args: argparse.Namespace) -> tuple[DecodeCase | PrefillCase, ...]:
    if args.case != "all":
        return (CASES_BY_NAME[args.case],)
    cases: tuple[DecodeCase | PrefillCase, ...]
    if args.kind == "decode":
        cases = DECODE_CASES
    elif args.kind == "prefill":
        cases = PREFILL_CASES
    else:
        cases = (*DECODE_CASES, *PREFILL_CASES)
    if args.suite != "all":
        cases = tuple(case for case in cases if case.suite == args.suite)
    return cases


def _run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "")
    if "gfx950" not in arch:
        raise RuntimeError(f"gfx950 is required, got {arch!r}")
    ops = _load_aiter_ops()
    cases = _select_cases(args)
    if not cases:
        raise ValueError("the case filters selected no workloads")

    stream = torch.cuda.Stream()
    results: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}", file=sys.stderr, flush=True)
        if isinstance(case, DecodeCase):
            result = _run_decode_case(
                case,
                ops,
                stream,
                warmup=args.warmup,
                repeats=args.repeats,
                batch=args.batch,
            )
        else:
            result = _run_prefill_case(
                case,
                ops,
                stream,
                warmup=args.warmup,
                repeats=args.repeats,
                batch=args.batch,
            )
        results.append(result)
        timing = result["timing"]
        print(
            f"  Gluon {timing['gluon']['median_us']:.3f} us; "
            f"AITER raw {timing['aiter_raw']['median_us']:.3f} us; "
            f"AITER contract {timing['aiter_contract']['median_us']:.3f} us",
            file=sys.stderr,
            flush=True,
        )

    payload = {
        "gluon_revision": args.gluon_revision,
        "aiter_revision": args.aiter_revision,
        "device": properties.name,
        "arch": arch,
        "compute_units": properties.multi_processor_count,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "topk": TOPK,
        "timing": "balanced CUDA events around batched graph replay",
        "warmup_replays_per_backend": args.warmup,
        "timing_repeats": args.repeats,
        "replays_per_sample": args.batch,
        "contract_note": (
            "Decode AITER contract includes logical-to-physical slot mapping and "
            "lens output. Prefill AITER contract includes lens output."
        ),
        "prefill_note": (
            "Each result is one scheduler forward. Its graph contains every "
            "selector slice imposed by the 512 MiB logits cap."
        ),
        "aggregates": _aggregates(results),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def _compact_launch_rows(rows: list[int]) -> str:
    parts: list[str] = []
    index = 0
    while index < len(rows):
        end = index + 1
        while end < len(rows) and rows[end] == rows[index]:
            end += 1
        count = end - index
        parts.append(f"{count}x{rows[index]}" if count > 1 else str(rows[index]))
        index = end
    return "+".join(parts)


def _dispatch_names(groups: list[dict[str, object]]) -> str:
    names: list[str] = []
    for group in groups:
        dispatch = group["dispatch"]
        name = str(dispatch["name"])
        if name not in names:
            names.append(name)
    return "+".join(names)


def _print_aggregate(name: str, aggregate: dict[str, object]) -> None:
    print(
        f"{name}: Gluon "
        f"{aggregate['gluon_speedup_vs_aiter_contract_geomean']:.3f}x vs "
        f"AITER contract; wins "
        f"{aggregate['gluon_wins_vs_aiter_contract']}/{aggregate['cases']}"
    )


def _report(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    aggregates = payload["aggregates"]
    for name in ("all", "decode", "prefill"):
        if name in aggregates:
            _print_aggregate(name.capitalize(), aggregates[name])

    decode_results = [
        result for result in payload["results"] if result["kind"] == "decode"
    ]
    if decode_results:
        print()
        print(
            "| Decode workload | Rows x N | q | Live range | Gluon path | "
            "Gluon | AITER+map | Speedup |"
        )
        print("|---|---:|---:|---:|---|---:|---:|---:|")
        for result in decode_results:
            timing = result["timing"]
            print(
                f"| {result['case']} | {result['rows']}x{result['context_len']} | "
                f"{result['q_len_per_req']} | {result['candidate_len_min']}-"
                f"{result['candidate_len_max']} | "
                f"{result['gluon_dispatch']['name']} | "
                f"{timing['gluon']['median_us']:.3f} us | "
                f"{timing['aiter_contract']['median_us']:.3f} us | "
                f"{result['aiter_contract_over_gluon']:.2f}x |"
            )

    prefill_results = [
        result for result in payload["results"] if result["kind"] == "prefill"
    ]
    if prefill_results:
        print()
        print(
            "| Prefill workload | R x N | Selector slices | Gluon path(s) | "
            "AITER path(s) | Gluon | AITER+lens | Speedup |"
        )
        print("|---|---:|---:|---|---|---:|---:|---:|")
        for result in prefill_results:
            timing = result["timing"]
            print(
                f"| {result['case']} | {result['rows']}x{result['context_len']} | "
                f"{_compact_launch_rows(result['launch_rows'])} | "
                f"{_dispatch_names(result['gluon_dispatch'])} | "
                f"{_dispatch_names(result['aiter_dispatch'])} | "
                f"{timing['gluon']['median_us']:.3f} us | "
                f"{timing['aiter_contract']['median_us']:.3f} us | "
                f"{result['aiter_contract_over_gluon']:.2f}x |"
            )


def _list_cases(_: argparse.Namespace) -> None:
    print("| Kind | Case | Suite | Runtime inputs |")
    print("|---|---|---|---|")
    for case in (*DECODE_CASES, *PREFILL_CASES):
        if isinstance(case, DecodeCase):
            inputs = (
                f"B={case.requests}, q={case.q_len_per_req}, "
                f"N={case.context_len}, live={min(case.seq_lens)}-"
                f"{max(case.seq_lens)}"
            )
            kind = "decode"
        else:
            inputs = (
                f"B={case.requests}, R={case.rows}, N={case.context_len}, "
                f"slices={_compact_launch_rows(list(prefill_launch_rows(case)))}"
            )
            kind = "prefill"
        print(f"| {kind} | {case.name} | {case.suite} | {inputs} |")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--kind", choices=("all", "decode", "prefill"), default="all")
    run.add_argument(
        "--suite",
        choices=("all", *(sorted({case.suite for case in CASES_BY_NAME.values()}))),
        default="all",
    )
    run.add_argument("--case", choices=("all", *CASES_BY_NAME), default="all")
    run.add_argument("--warmup", type=int, default=12)
    run.add_argument("--repeats", type=int, default=20)
    run.add_argument("--batch", type=int, default=10)
    run.add_argument("--gluon-revision", default="unknown")
    run.add_argument("--aiter-revision", default="unknown")
    run.add_argument("--json-out", type=Path)
    run.set_defaults(func=_run)
    report = subparsers.add_parser("report")
    report.add_argument("input", type=Path)
    report.set_defaults(func=_report)
    list_cases = subparsers.add_parser("list")
    list_cases.set_defaults(func=_list_cases)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
