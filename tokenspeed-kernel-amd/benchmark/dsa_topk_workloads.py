from __future__ import annotations

from dataclasses import dataclass

PAGE_SIZE = 64
TOPK = 2048
PREFILL_MAX_LOGITS_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class DecodeCase:
    name: str
    suite: str
    max_model_len: int
    q_len_per_req: int
    seq_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.seq_lens:
            raise ValueError("decode cases require at least one request")
        if self.q_len_per_req < 1:
            raise ValueError("q_len_per_req must be positive")
        if min(self.seq_lens) < 1:
            raise ValueError("decode sequence lengths must be positive")
        if max(self.seq_lens) > self.max_model_len:
            raise ValueError("decode sequence length exceeds max_model_len")

    @property
    def requests(self) -> int:
        return len(self.seq_lens)

    @property
    def rows(self) -> int:
        return self.requests * self.q_len_per_req

    @property
    def context_len(self) -> int:
        return round_up(self.max_model_len, PAGE_SIZE)

    @property
    def candidate_lens(self) -> tuple[int, ...]:
        q_len = self.q_len_per_req
        return tuple(
            max(0, seq_len - (q_len - 1) + q_offset)
            for seq_len in self.seq_lens
            for q_offset in range(q_len)
        )


@dataclass(frozen=True)
class PrefillCase:
    name: str
    suite: str
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.prefix_lens:
            raise ValueError("prefill cases require at least one request")
        if len(self.prefix_lens) != len(self.extend_lens):
            raise ValueError("prefix_lens and extend_lens must have equal length")
        if min(self.prefix_lens) < 0:
            raise ValueError("prefill prefix lengths must be nonnegative")
        if min(self.extend_lens) < 1:
            raise ValueError("prefill extension lengths must be positive")

    @property
    def requests(self) -> int:
        return len(self.prefix_lens)

    @property
    def rows(self) -> int:
        return sum(self.extend_lens)

    @property
    def context_len(self) -> int:
        return sum(
            prefix + extend
            for prefix, extend in zip(self.prefix_lens, self.extend_lens, strict=True)
        )


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def prefill_row_ranges(case: PrefillCase) -> tuple[tuple[int, ...], tuple[int, ...]]:
    starts: list[int] = []
    ends: list[int] = []
    request_start = 0
    for prefix_len, extend_len in zip(case.prefix_lens, case.extend_lens, strict=True):
        starts.extend([request_start] * extend_len)
        ends.extend(
            request_start + prefix_len + token_offset + 1
            for token_offset in range(extend_len)
        )
        request_start += prefix_len + extend_len
    return tuple(starts), tuple(ends)


def prefill_max_rows_per_launch(
    case: PrefillCase,
    max_logits_bytes: int = PREFILL_MAX_LOGITS_BYTES,
) -> int:
    return max(1, max_logits_bytes // (4 * case.context_len))


def prefill_launch_slices(
    case: PrefillCase,
    max_logits_bytes: int = PREFILL_MAX_LOGITS_BYTES,
) -> tuple[tuple[int, int], ...]:
    rows_per_launch = prefill_max_rows_per_launch(case, max_logits_bytes)
    return tuple(
        (start, min(start + rows_per_launch, case.rows))
        for start in range(0, case.rows, rows_per_launch)
    )


def prefill_launch_rows(
    case: PrefillCase,
    max_logits_bytes: int = PREFILL_MAX_LOGITS_BYTES,
) -> tuple[int, ...]:
    return tuple(
        end - start for start, end in prefill_launch_slices(case, max_logits_bytes)
    )


def _sample(values: tuple[int, ...], count: int) -> tuple[int, ...]:
    if count == 1:
        return (values[(len(values) - 1) // 2],)
    return tuple(
        values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)
    )


def _spread(
    count: int,
    low: int,
    high: int,
    *,
    single: int,
) -> tuple[int, ...]:
    if count == 1:
        return (single,)
    return tuple(low + (high - low) * index // (count - 1) for index in range(count))


_AGENTIC_DECODE_LENS = tuple(50_500 + index * 1_213 for index in range(16))

DECODE_CASES = (
    tuple(
        DecodeCase(
            name=f"agentic80k-b{requests:02d}-q4",
            suite="decode-agentic",
            max_model_len=80_000,
            q_len_per_req=4,
            seq_lens=_sample(_AGENTIC_DECODE_LENS, requests),
        )
        for requests in (1, 2, 3, 4, 6, 8, 12, 16)
    )
    + tuple(
        DecodeCase(
            name=f"ci256k-b{requests:02d}-q4",
            suite="decode-ci-long",
            max_model_len=262_144,
            q_len_per_req=4,
            seq_lens=_spread(
                requests,
                65_536,
                245_760,
                single=196_608,
            ),
        )
        for requests in (1, 4, 16)
    )
    + tuple(
        DecodeCase(
            name=f"nonspec256k-b{requests:03d}-q1",
            suite="decode-nonspec",
            max_model_len=262_144,
            q_len_per_req=1,
            seq_lens=_spread(
                requests,
                8_193,
                262_111,
                single=196_609,
            ),
        )
        for requests in (1, 24, 127, 128)
    )
    + tuple(
        DecodeCase(
            name=f"long1m-b{requests:02d}-q4",
            suite="decode-1m",
            max_model_len=1_048_576,
            q_len_per_req=4,
            seq_lens=_spread(
                requests,
                262_145,
                1_048_511,
                single=999_489,
            ),
        )
        for requests in (1, 4)
    )
)


PREFILL_CASES = (
    PrefillCase("cold50k-08k", "prefill-cold", (0,), (8_192,)),
    PrefillCase("cold50k-16k", "prefill-cold", (8_192,), (8_192,)),
    PrefillCase("cold50k-24k", "prefill-cold", (16_384,), (8_192,)),
    PrefillCase("cold50k-32k", "prefill-cold", (24_576,), (8_192,)),
    PrefillCase("cold50k-40k", "prefill-cold", (32_768,), (8_192,)),
    PrefillCase("cold50k-48k", "prefill-cold", (40_960,), (8_192,)),
    PrefillCase("cold50k-final", "prefill-cold", (49_152,), (848,)),
    PrefillCase(
        "continuous-final-plus-fresh",
        "prefill-continuous",
        (49_152, 0),
        (848, 7_344),
    ),
    PrefillCase("cached-turn02", "prefill-cached", (50_496,), (804,)),
    PrefillCase("cached-turn08", "prefill-cached", (58_240,), (860,)),
    PrefillCase("cached-turn15", "prefill-cached", (67_392,), (808,)),
    PrefillCase(
        "cached-b02",
        "prefill-cached",
        (50_496, 67_392),
        (804, 808),
    ),
    PrefillCase(
        "cached-b08",
        "prefill-cached",
        (50_496, 53_056, 55_680, 58_240, 60_864, 63_488, 66_048, 67_392),
        (804, 844, 820, 860, 836, 812, 852, 808),
    ),
    PrefillCase(
        "cached-full-budget",
        "prefill-cached",
        (
            50_496,
            51_776,
            53_056,
            54_336,
            55_680,
            56_960,
            58_240,
            59_584,
            60_864,
            62_144,
        ),
        (804, 824, 844, 864, 820, 840, 860, 816, 836, 684),
    ),
    PrefillCase("long1m-b01", "prefill-long", (999_488,), (812,)),
    PrefillCase(
        "long1m-b02",
        "prefill-long",
        (899_968, 999_488),
        (832, 812),
    ),
    PrefillCase(
        "long1m-b04",
        "prefill-long",
        (799_936, 899_968, 999_488, 1_046_976),
        (864, 832, 812, 824),
    ),
)


CASES_BY_NAME = {case.name: case for case in (*DECODE_CASES, *PREFILL_CASES)}
