from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

_WORKLOADS_PATH = Path(__file__).parents[1] / "benchmark" / "dsa_topk_workloads.py"
_SPEC = importlib.util.spec_from_file_location("dsa_topk_workloads", _WORKLOADS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
workloads = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = workloads
_SPEC.loader.exec_module(workloads)


def test_dsa_topk_benchmark_manifest_has_expected_workloads() -> None:
    assert len(workloads.DECODE_CASES) == 17
    assert len(workloads.PREFILL_CASES) == 17
    assert len(workloads.CASES_BY_NAME) == 34
    assert len(set(workloads.CASES_BY_NAME)) == 34

    agentic = [
        case for case in workloads.DECODE_CASES if case.suite == "decode-agentic"
    ]
    assert [case.requests for case in agentic] == [1, 2, 3, 4, 6, 8, 12, 16]
    assert all(case.q_len_per_req == 4 for case in agentic)
    assert all(case.context_len == 80_000 for case in agentic)


def test_decode_cases_follow_runtime_shape_contract() -> None:
    for case in workloads.DECODE_CASES:
        assert case.rows == case.requests * case.q_len_per_req
        assert case.context_len % workloads.PAGE_SIZE == 0
        assert case.context_len >= case.max_model_len
        assert case.context_len - case.max_model_len < workloads.PAGE_SIZE
        assert len(case.candidate_lens) == case.rows
        assert min(case.candidate_lens) >= 0
        assert max(case.candidate_lens) <= case.context_len

    mtp = workloads.DecodeCase(
        "mtp-candidate-ends",
        "test",
        80_000,
        4,
        (50_003,),
    )
    assert mtp.candidate_lens == (50_000, 50_001, 50_002, 50_003)


def test_prefill_row_ranges_match_packed_runtime_mapping() -> None:
    case = workloads.PrefillCase(
        "packed",
        "test",
        prefix_lens=(3, 5),
        extend_lens=(2, 3),
    )

    starts, ends = workloads.prefill_row_ranges(case)

    assert case.rows == 5
    assert case.context_len == 13
    assert starts == (0, 0, 5, 5, 5)
    assert ends == (4, 5, 11, 12, 13)


def test_prefill_launch_slices_match_runtime_logits_cap() -> None:
    exact_rows = {
        "cold50k-08k": (8_192,),
        "cold50k-16k": (8_192,),
        "cold50k-24k": (5_461, 2_731),
        "cold50k-32k": (4_096, 4_096),
        "cold50k-40k": (3_276, 3_276, 1_640),
        "cold50k-48k": (2_730, 2_730, 2_730, 2),
        "cold50k-final": (848,),
        "continuous-final-plus-fresh": (2_340, 2_340, 2_340, 1_172),
        "cached-b02": (1_123, 489),
    }
    run_counts = {
        "cached-b08": Counter({278: 23, 242: 1}),
        "cached-full-budget": Counter({234: 35, 2: 1}),
        "long1m-b01": Counter({134: 6, 8: 1}),
        "long1m-b02": Counter({70: 23, 34: 1}),
        "long1m-b04": Counter({35: 95, 7: 1}),
    }

    for name, expected in exact_rows.items():
        assert workloads.prefill_launch_rows(workloads.CASES_BY_NAME[name]) == expected
    for name, expected in run_counts.items():
        assert (
            Counter(workloads.prefill_launch_rows(workloads.CASES_BY_NAME[name]))
            == expected
        )


def test_prefill_slices_partition_rows_and_obey_memory_cap() -> None:
    for case in workloads.PREFILL_CASES:
        slices = workloads.prefill_launch_slices(case)
        assert slices[0][0] == 0
        assert slices[-1][1] == case.rows
        assert all(left[1] == right[0] for left, right in zip(slices, slices[1:]))
        assert sum(end - start for start, end in slices) == case.rows
        assert all(
            (end - start) * case.context_len * 4 <= workloads.PREFILL_MAX_LOGITS_BYTES
            for start, end in slices
        )


def test_cached_prefill_prefixes_are_page_aligned_and_replay_tail() -> None:
    cached = [
        case
        for case in workloads.PREFILL_CASES
        if case.suite in {"prefill-cached", "prefill-long"}
        and case.name != "cached-full-budget"
    ]
    for case in cached:
        assert all(prefix % workloads.PAGE_SIZE == 0 for prefix in case.prefix_lens)
        assert all(801 <= extend <= 864 for extend in case.extend_lens)
