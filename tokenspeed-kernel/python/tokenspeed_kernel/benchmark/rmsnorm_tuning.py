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

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenspeed_kernel.benchmark.throughput import ThroughputCalculator
from tokenspeed_kernel.ops.layernorm.gluon import (
    _rmsnorm_block_full_aiter,
    _rmsnorm_block_full,
    _rmsnorm_streaming_block,
    _rmsnorm_wave_row,
    rmsnorm as gluon_rmsnorm,
)
from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm


_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_DEFAULT_TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_F32_SIZE_PER_THREAD_SWEEP = (1, 2, 4, 8)
_HALF_SIZE_PER_THREAD_SWEEP = (1, 2, 4, 8, 16)
_TOLERANCE = 2e-2


@dataclass(frozen=True)
class Candidate:
    name: str
    strategy: str
    params: dict[str, int]
    launcher: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]

    def run(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.launcher(x, weight, eps, residual=residual)


def _make_block_full_launcher(
    *,
    num_warps: int,
    size_per_thread: int,
) -> Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    def launcher(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return _rmsnorm_block_full(
            x,
            weight,
            eps,
            residual=residual,
            num_warps=num_warps,
            size_per_thread=size_per_thread,
        )

    return launcher


def _make_block_full_aiter_launcher(
    *,
    num_warps: int,
    size_per_thread: int,
) -> Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    def launcher(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return _rmsnorm_block_full_aiter(
            x,
            weight,
            eps,
            residual=residual,
            num_warps=num_warps,
            size_per_thread=size_per_thread,
        )

    return launcher


def _make_wave_row_launcher(
    *,
    size_per_thread: int,
) -> Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    def launcher(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return _rmsnorm_wave_row(
            x,
            weight,
            eps,
            residual=residual,
            size_per_thread=size_per_thread,
        )

    return launcher


def _make_streaming_block_launcher(
    *,
    col_block: int,
    num_warps: int,
    size_per_thread: int,
) -> Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    def launcher(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return _rmsnorm_streaming_block(
            x,
            weight,
            eps,
            residual=residual,
            col_block=col_block,
            num_warps=num_warps,
            size_per_thread=size_per_thread,
        )

    return launcher


def _block_full_candidate_name(num_warps: int, size_per_thread: int) -> str:
    if size_per_thread == 1:
        return f"block_full_w{num_warps}"
    return f"block_full_spt{size_per_thread}_w{num_warps}"


def _streaming_block_candidate_name(
    col_block: int,
    num_warps: int,
    size_per_thread: int,
) -> str:
    if size_per_thread == 1:
        return f"stream_c{col_block}_w{num_warps}"
    return f"stream_c{col_block}_spt{size_per_thread}_w{num_warps}"


def default_size_per_thread_values(dtype: torch.dtype) -> tuple[int, ...]:
    if dtype == torch.float32:
        return _F32_SIZE_PER_THREAD_SWEEP
    return _HALF_SIZE_PER_THREAD_SWEEP


def build_default_candidates(
    *,
    size_per_thread_values: tuple[int, ...] = _HALF_SIZE_PER_THREAD_SWEEP,
) -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate("triton_rmsnorm", "baseline", {}, triton_rmsnorm),
        Candidate("gluon_rmsnorm", "baseline", {}, gluon_rmsnorm),
        Candidate(
            "block_full_aiter_spt16_w4",
            "block_full_aiter",
            {"num_warps": 4, "size_per_thread": 16},
            _make_block_full_aiter_launcher(num_warps=4, size_per_thread=16),
        ),
    ]

    for size_per_thread in size_per_thread_values:
        for num_warps in (1, 2, 4, 8):
            candidates.append(
                Candidate(
                    _block_full_candidate_name(num_warps, size_per_thread),
                    "block_full",
                    {
                        "num_warps": num_warps,
                        "size_per_thread": size_per_thread,
                    },
                    _make_block_full_launcher(
                        num_warps=num_warps,
                        size_per_thread=size_per_thread,
                    ),
                )
            )

    for size_per_thread in size_per_thread_values:
        candidates.append(
            Candidate(
                f"wave_row_spt{size_per_thread}",
                "wave_row",
                {"size_per_thread": size_per_thread},
                _make_wave_row_launcher(size_per_thread=size_per_thread),
            )
        )

    for col_block in (512, 1024, 2048):
        for size_per_thread in size_per_thread_values:
            for num_warps in (1, 2, 4, 8):
                candidates.append(
                    Candidate(
                        _streaming_block_candidate_name(
                            col_block,
                            num_warps,
                            size_per_thread,
                        ),
                        "streaming_block",
                        {
                            "col_block": col_block,
                            "num_warps": num_warps,
                            "size_per_thread": size_per_thread,
                        },
                        _make_streaming_block_launcher(
                            col_block=col_block,
                            num_warps=num_warps,
                            size_per_thread=size_per_thread,
                        ),
                    )
                )

    return candidates


def build_shapes(
    token_counts: list[int],
    hidden_size: int,
    *,
    include_odd: bool,
    odd_hidden_size: int,
) -> list[dict[str, Any]]:
    shapes = [
        {"num_tokens": num_tokens, "hidden_size": hidden_size, "residual": residual}
        for residual in (False, True)
        for num_tokens in token_counts
    ]
    if include_odd:
        shapes.extend(
            {
                "num_tokens": 11,
                "hidden_size": odd_hidden_size,
                "residual": residual,
            }
            for residual in (False, True)
        )
    return shapes


def _parse_token_counts(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one token count is required")
    if any(value <= 0 for value in values):
        raise ValueError("token counts must be positive")
    return values


def _parse_size_per_thread_values(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("at least one size_per_thread value is required")
    if any(value <= 0 for value in values):
        raise ValueError("size_per_thread values must be positive")
    return values


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    weight = rank - float(low)
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _regime(shape: dict[str, Any]) -> str:
    if int(shape["hidden_size"]) != 2880:
        return "odd"
    return "decode" if int(shape["num_tokens"]) <= 32 else "prefill"


def _make_inputs(
    shape: dict[str, Any],
    dtype: torch.dtype,
    *,
    weight_dtype: torch.dtype,
    seed: int,
    device: str = "cuda",
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    num_tokens = int(shape["num_tokens"])
    hidden_size = int(shape["hidden_size"])
    x = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(dtype)
    residual = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).to(dtype)
        if bool(shape.get("residual", False))
        else None
    )
    weight = torch.randn(
        hidden_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(weight_dtype)
    return {"x": x, "weight": weight, "eps": 1e-6, "residual": residual}


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    x_float = x.to(torch.float32)
    weight_float = weight.to(torch.float32)
    if residual is not None:
        x_float = x_float + residual.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    out = (x_float * torch.rsqrt(variance + eps) * weight_float).to(x.dtype)
    if residual is None:
        return out
    return out, x_float.to(x.dtype)


def _diffs(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, bool]:
    actual_f = actual.to(torch.float32)
    expected_f = expected.to(torch.float32)
    abs_diff = (actual_f - expected_f).abs()
    rel_diff = abs_diff / expected_f.abs().clamp_min(1e-12)
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    max_rel = float(rel_diff.max().item()) if rel_diff.numel() else 0.0
    passed = bool(
        torch.allclose(
            actual,
            expected,
            atol=_TOLERANCE,
            rtol=_TOLERANCE,
        )
    )
    return max_abs, max_rel, passed


def _verify(
    candidate: Candidate,
    inputs: dict[str, Any],
) -> tuple[bool, float, float]:
    expected = _reference(**inputs)
    actual = candidate.run(**inputs)
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple):
            return False, float("inf"), float("inf")
        pairs = zip(actual, expected, strict=True)
    else:
        if not isinstance(actual, torch.Tensor):
            return False, float("inf"), float("inf")
        pairs = [(actual, expected)]

    passed = True
    max_abs = 0.0
    max_rel = 0.0
    for actual_tensor, expected_tensor in pairs:
        abs_diff, rel_diff, tensors_passed = _diffs(actual_tensor, expected_tensor)
        passed = passed and tensors_passed
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
    return passed, max_abs, max_rel


def _time_candidate(
    candidate: Candidate,
    inputs: dict[str, Any],
    *,
    warmup_iters: int,
    bench_iters: int,
) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup_iters):
            candidate.run(**inputs)
    torch.cuda.synchronize()

    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    with torch.no_grad():
        for i in range(bench_iters):
            start_events[i].record()
            candidate.run(**inputs)
            end_events[i].record()
    torch.cuda.synchronize()
    times = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(start_events, end_events, strict=False)
    ]
    times.sort()
    return times


def run_tuning(
    candidates: list[Candidate],
    shapes: list[dict[str, Any]],
    *,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    warmup_iters: int,
    bench_iters: int,
    verify: bool,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for shape in shapes:
        inputs = _make_inputs(shape, dtype, weight_dtype=weight_dtype, seed=seed)
        for candidate in candidates:
            result: dict[str, Any] = {
                "candidate": candidate.name,
                "strategy": candidate.strategy,
                "params": dict(candidate.params),
                "shape": dict(shape),
                "regime": _regime(shape),
                "dtype": str(dtype),
                "weight_dtype": str(weight_dtype),
                "num_iters": bench_iters,
                "numerics_passed": None,
                "max_abs_diff": None,
                "max_rel_diff": None,
                "p50_latency_us": None,
                "p90_latency_us": None,
                "p99_latency_us": None,
                "min_latency_us": None,
                "max_latency_us": None,
                "bandwidth_gb_s": None,
                "error": None,
            }
            try:
                if verify:
                    passed, max_abs, max_rel = _verify(candidate, inputs)
                    result["numerics_passed"] = passed
                    result["max_abs_diff"] = max_abs
                    result["max_rel_diff"] = max_rel
                    if not passed:
                        results.append(result)
                        continue
                times = _time_candidate(
                    candidate,
                    inputs,
                    warmup_iters=warmup_iters,
                    bench_iters=bench_iters,
                )
                p50 = _percentile(times, 50.0)
                p90 = _percentile(times, 90.0)
                p99 = _percentile(times, 99.0)
                _, bandwidth = ThroughputCalculator.compute(
                    "norm",
                    "rmsnorm",
                    shape,
                    p50,
                    dtype=dtype,
                    weight_dtype=weight_dtype,
                )
                result.update(
                    {
                        "p50_latency_us": p50,
                        "p90_latency_us": p90,
                        "p99_latency_us": p99,
                        "min_latency_us": float(times[0]),
                        "max_latency_us": float(times[-1]),
                        "bandwidth_gb_s": bandwidth,
                    }
                )
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
    return results


def _format_value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def format_tuning_report(results: list[dict[str, Any]], *, top_k: int = 5) -> str:
    grouped: dict[tuple[int, int, bool], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        shape = result["shape"]
        grouped[
            (
                int(shape["num_tokens"]),
                int(shape["hidden_size"]),
                bool(shape.get("residual", False)),
            )
        ].append(result)

    lines: list[str] = []
    for key in sorted(grouped):
        num_tokens, hidden_size, residual = key
        group = [item for item in grouped[key] if item["p50_latency_us"] is not None]
        group.sort(key=lambda item: item["p50_latency_us"])
        lines.append("")
        lines.append(
            f"shape num_tokens={num_tokens}, hidden_size={hidden_size}, "
            f"residual={residual}"
        )
        lines.append("candidate,strategy,p50_us,p90_us,p99_us,gb_s,numerics")
        for item in group[:top_k]:
            lines.append(
                ",".join(
                    [
                        item["candidate"],
                        item["strategy"],
                        _format_value(item["p50_latency_us"]),
                        _format_value(item["p90_latency_us"]),
                        _format_value(item["p99_latency_us"]),
                        _format_value(item["bandwidth_gb_s"]),
                        _format_value(item["numerics_passed"]),
                    ]
                )
            )

        errors = [item for item in grouped[key] if item["error"]]
        for item in errors[:top_k]:
            lines.append(f"{item['candidate']},{item['strategy']},ERROR,{item['error']}")

    return "\n".join(lines).lstrip()


def _filter_candidates(
    candidates: list[Candidate],
    raw_names: str | None,
) -> list[Candidate]:
    if raw_names is None:
        return candidates
    requested = {name.strip() for name in raw_names.split(",") if name.strip()}
    selected = [candidate for candidate in candidates if candidate.name in requested]
    missing = requested - {candidate.name for candidate in selected}
    if missing:
        raise ValueError(f"unknown candidate(s): {', '.join(sorted(missing))}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune GPT-OSS RMSNorm kernels")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument(
        "--weight-dtype",
        choices=sorted(_DTYPES),
        default="bf16",
        help="RMSNorm weight dtype. GPT-OSS norm scales are bf16.",
    )
    parser.add_argument(
        "--token-counts",
        default=",".join(str(value) for value in _DEFAULT_TOKEN_COUNTS),
        help="Comma-separated num_tokens values for hidden_size.",
    )
    parser.add_argument("--hidden-size", type=int, default=2880)
    parser.add_argument("--include-odd", action="store_true")
    parser.add_argument("--odd-hidden-size", type=int, default=2897)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", help="Comma-separated candidate name filter")
    parser.add_argument(
        "--size-per-thread-values",
        help=(
            "Comma-separated SPT sweep override. Defaults to 1,2,4,8,16 for "
            "half dtypes and 1,2,4,8 for fp32."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--export", help="Write full tuning results as JSON")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numerical verification before timing.",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("RMSNorm tuning requires CUDA")
    if args.hidden_size <= 0 or args.odd_hidden_size <= 0:
        raise ValueError("hidden sizes must be positive")
    if args.warmup_iters < 0 or args.bench_iters <= 0:
        raise ValueError("warmup-iters must be >= 0 and bench-iters must be > 0")

    dtype = _DTYPES[args.dtype]
    weight_dtype = _DTYPES[args.weight_dtype]
    token_counts = _parse_token_counts(args.token_counts)
    size_per_thread_values = (
        _parse_size_per_thread_values(args.size_per_thread_values)
        if args.size_per_thread_values is not None
        else default_size_per_thread_values(dtype)
    )
    shapes = build_shapes(
        token_counts,
        args.hidden_size,
        include_odd=args.include_odd,
        odd_hidden_size=args.odd_hidden_size,
    )
    candidates = _filter_candidates(
        build_default_candidates(size_per_thread_values=size_per_thread_values),
        args.candidates,
    )
    results = run_tuning(
        candidates,
        shapes,
        dtype=dtype,
        weight_dtype=weight_dtype,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        verify=not args.no_verify,
        seed=args.seed,
    )

    print(format_tuning_report(results, top_k=args.top_k))
    if args.export:
        output_path = Path(args.export)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
