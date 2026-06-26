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

"""Attention request metadata input generators for numerical correctness tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from tokenspeed_kernel.numerics.input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    _resolve_device,
)

__all__ = [
    "CacheLayout",
    "LengthMode",
    "MHARequestMetadataInputConfig",
    "MHARequestMetadataInput",
    "MHARequestMetadataValues",
    "PageTableIndexing",
]

CacheLayout = Literal["none", "dense", "paged"]
LengthMode = Literal["ragged", "regular", "fixed_per_request"]
PageTableIndexing = Literal["identity", "random"]


def _check_nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_cache_layout(value: str) -> CacheLayout:
    if value not in ("none", "dense", "paged"):
        raise ValueError(
            f"cache_layout must be 'none', 'dense', or 'paged', got {value!r}"
        )
    return cast(CacheLayout, value)


def _check_length_mode(name: str, value: str) -> LengthMode:
    if value not in ("ragged", "regular", "fixed_per_request"):
        raise ValueError(
            f"{name} must be 'ragged', 'regular', or 'fixed_per_request', got {value!r}"
        )
    return cast(LengthMode, value)


def _check_page_table_indexing(value: str) -> PageTableIndexing:
    if value not in ("identity", "random"):
        raise ValueError(
            f"page_table_indexing must be 'identity' or 'random', got {value!r}"
        )
    return cast(PageTableIndexing, value)


def _cumsum(lengths: list[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return offsets


def _split_lengths(
    *,
    total: int,
    batch_size: int,
    mode: LengthMode,
    generator: torch.Generator,
    min_per_request: int = 0,
    max_per_request: int | None = None,
) -> list[int]:
    """Split a total token count into per-request lengths."""

    if min_per_request < 0:
        raise ValueError("min_per_request must be non-negative")
    if max_per_request is not None:
        max_per_request = int(max_per_request)
        if max_per_request < min_per_request:
            raise ValueError(
                f"max_per_request={max_per_request} must be >= "
                f"min_per_request={min_per_request}"
            )
    if total < batch_size * min_per_request:
        raise ValueError(
            f"cannot split total={total} across batch_size={batch_size} with "
            f"min_per_request={min_per_request}"
        )
    if max_per_request is not None and total > batch_size * max_per_request:
        raise ValueError(
            f"cannot split total={total} across batch_size={batch_size} with "
            f"max_per_request={max_per_request}"
        )

    if mode == "fixed_per_request":
        if total % batch_size != 0:
            raise ValueError(
                f"fixed_per_request requires total={total} to be divisible by "
                f"batch_size={batch_size}"
            )
        per_request = total // batch_size
        if per_request < min_per_request:
            raise ValueError(
                f"fixed_per_request produced {per_request}, below "
                f"min_per_request={min_per_request}"
            )
        if max_per_request is not None and per_request > max_per_request:
            raise ValueError(
                f"fixed_per_request produced {per_request}, above "
                f"max_per_request={max_per_request}"
            )
        return [per_request] * batch_size

    if mode == "regular":
        base = total // batch_size
        remainder = total % batch_size
        lengths = [base] * batch_size
        if remainder:
            order = torch.randperm(batch_size, generator=generator).tolist()
            for idx in order[:remainder]:
                lengths[int(idx)] += 1
        if any(length < min_per_request for length in lengths):
            raise ValueError(
                f"regular split produced lengths below {min_per_request}: {lengths}"
            )
        if max_per_request is not None and any(
            length > max_per_request for length in lengths
        ):
            raise ValueError(
                f"regular split produced lengths above {max_per_request}: {lengths}"
            )
        return lengths

    lengths = [min_per_request] * batch_size
    remaining = total - batch_size * min_per_request
    if remaining == 0:
        return lengths

    if max_per_request is not None:
        capacities = [max_per_request - min_per_request] * batch_size
        order = torch.randperm(batch_size, generator=generator).tolist()
        for pos, idx in enumerate(order):
            rest_capacity = sum(capacities[int(rest)] for rest in order[pos + 1 :])
            min_add = max(0, remaining - rest_capacity)
            max_add = min(capacities[int(idx)], remaining)
            if min_add == max_add:
                add = min_add
            else:
                add = int(
                    torch.randint(
                        min_add,
                        max_add + 1,
                        (1,),
                        generator=generator,
                    ).item()
                )
            lengths[int(idx)] += add
            remaining -= add
        return lengths

    weights = torch.rand(batch_size, generator=generator, dtype=torch.float64)
    weights = weights / weights.sum()
    raw = weights * remaining
    extras = torch.floor(raw).to(torch.int64)
    remainder = int(remaining - int(extras.sum().item()))
    if remainder:
        # Break exact fractional ties with metadata-seeded noise.
        fractions = raw - extras.to(raw.dtype)
        fractions = (
            fractions
            + torch.rand(batch_size, generator=generator, dtype=torch.float64) * 1e-9
        )
        for idx in torch.argsort(fractions, descending=True)[:remainder].tolist():
            extras[int(idx)] += 1
    return [length + int(extra) for length, extra in zip(lengths, extras.tolist())]


def _make_cumulative_offsets(
    lengths: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    offsets = _cumsum(lengths)
    return torch.tensor(offsets, dtype=torch.int32, device=device), offsets


@dataclass
class MHARequestMetadataValues:
    """Generated request-length and cache metadata."""

    cached_lens_cpu: list[int]
    new_q_lens_cpu: list[int]
    new_kv_lens_cpu: list[int]
    visible_kv_lens_cpu: list[int]
    cu_seqlens_q: torch.Tensor
    cu_seqlens_q_cpu: list[int]
    cu_seqlens_kv: torch.Tensor
    cu_seqlens_kv_cpu: list[int]
    cache_seqlens: torch.Tensor
    max_seqlen_q: int
    resolved_max_seqlen_k: int


@dataclass
class MHARequestMetadataInputConfig:
    """Initialization parameters for ``MHARequestMetadataInput``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of request sequences represented by generated metadata.
    batch_size: int

    # Required: total already-resident KV tokens across all requests.
    total_cached_tokens: int

    # Required: total query tokens that produce outputs in this call.
    total_new_q_tokens: int

    # ------------------------------------------------------------------
    # Optional token-count and generation-mode configuration.
    # ------------------------------------------------------------------

    # Optional: total new K/V tokens inserted for this call. Defaults to Q.
    total_new_kv_tokens: int | None = None

    # Optional: split policy for total_cached_tokens.
    cached_length_mode: Literal["ragged", "regular"] = "ragged"

    # Optional: upper bound for generated cached tokens in one request.
    max_cached_tokens_per_request: int | None = None

    # Optional: split policy for total_new_q_tokens.
    new_q_length_mode: LengthMode = "ragged"

    # Optional: upper bound for generated query tokens in one request.
    max_new_q_tokens_per_request: int | None = None

    # Optional: split policy for total_new_kv_tokens when it is independent.
    new_kv_length_mode: LengthMode | None = None

    # Optional: upper bound for generated new K/V tokens in one request.
    max_new_kv_tokens_per_request: int | None = None

    # Optional: keep per-request new K/V lengths identical to new Q lengths.
    tie_new_kv_to_query: bool = True

    # ------------------------------------------------------------------
    # Optional cache metadata configuration.
    # ------------------------------------------------------------------

    # Optional: K/V storage surface represented by this metadata.
    cache_layout: CacheLayout = "none"

    # Optional: explicit upper bound passed as max_seqlen_k.
    max_seqlen_k: int | None = None

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: default generation device for tensor metadata.
    device: DeviceLike = None


@dataclass(init=False)
class MHARequestMetadataInput(NumericsInputGenerator):
    """Generator for MHA request-length and cache metadata.

    This generator owns the non-value inputs derived from request shape:
    per-request cached/new token lengths, packed cumulative offsets, visible KV
    sequence lengths, and cache sequence lengths. It is intended to be composed
    by higher-level MHA input generators, while still being directly usable in
    metadata-focused tests.
    """

    config: MHARequestMetadataInputConfig

    def __init__(
        self,
        config: MHARequestMetadataInputConfig | None = None,
        **kwargs: object,
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword parameters, not both")
        self.config = config or MHARequestMetadataInputConfig(**kwargs)  # type: ignore[arg-type]
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> MHARequestMetadataValues:
        self._normalize_config()
        target_device = _resolve_device(self.config.device, device)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        cached_lens = _split_lengths(
            total=self.config.total_cached_tokens,
            batch_size=self.config.batch_size,
            mode=self.config.cached_length_mode,
            generator=generator,
            min_per_request=0,
            max_per_request=self.config.max_cached_tokens_per_request,
        )
        new_q_lens = _split_lengths(
            total=self.config.total_new_q_tokens,
            batch_size=self.config.batch_size,
            mode=self.config.new_q_length_mode,
            generator=generator,
            min_per_request=1,
            max_per_request=self.config.max_new_q_tokens_per_request,
        )
        if self.config.tie_new_kv_to_query:
            new_kv_lens = list(new_q_lens)
        else:
            new_kv_mode = (
                self.config.new_kv_length_mode or self.config.new_q_length_mode
            )
            new_kv_lens = _split_lengths(
                total=self.config.total_new_kv_tokens,
                batch_size=self.config.batch_size,
                mode=new_kv_mode,
                generator=generator,
                min_per_request=0,
                max_per_request=self.config.max_new_kv_tokens_per_request,
            )

        visible_kv_lens = [
            cached + new_kv
            for cached, new_kv in zip(cached_lens, new_kv_lens, strict=True)
        ]
        max_visible_kv = max(visible_kv_lens) if visible_kv_lens else 0
        if (
            self.config.max_seqlen_k is not None
            and self.config.max_seqlen_k < max_visible_kv
        ):
            raise ValueError(
                f"max_seqlen_k={self.config.max_seqlen_k} is smaller than generated "
                f"visible KV length {max_visible_kv}"
            )

        max_seqlen_q = max(new_q_lens)
        resolved_max_seqlen_k = (
            int(self.config.max_seqlen_k)
            if self.config.max_seqlen_k is not None
            else max_visible_kv
        )
        cu_seqlens_q, cu_seqlens_q_cpu = _make_cumulative_offsets(
            new_q_lens,
            device=target_device,
        )
        cu_seqlens_kv, cu_seqlens_kv_cpu = _make_cumulative_offsets(
            visible_kv_lens,
            device=target_device,
        )
        return MHARequestMetadataValues(
            cached_lens_cpu=cached_lens,
            new_q_lens_cpu=new_q_lens,
            new_kv_lens_cpu=new_kv_lens,
            visible_kv_lens_cpu=visible_kv_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_kv_cpu=cu_seqlens_kv_cpu,
            cache_seqlens=torch.tensor(
                visible_kv_lens,
                dtype=torch.int32,
                device=target_device,
            ),
            max_seqlen_q=max_seqlen_q,
            resolved_max_seqlen_k=resolved_max_seqlen_k,
        )

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.total_cached_tokens = _check_nonnegative(
            "total_cached_tokens",
            self.config.total_cached_tokens,
        )
        self.config.total_new_q_tokens = _check_positive(
            "total_new_q_tokens",
            self.config.total_new_q_tokens,
        )
        if self.config.total_new_kv_tokens is None:
            self.config.total_new_kv_tokens = self.config.total_new_q_tokens
        self.config.total_new_kv_tokens = _check_nonnegative(
            "total_new_kv_tokens",
            self.config.total_new_kv_tokens,
        )
        if self.config.tie_new_kv_to_query and (
            self.config.total_new_kv_tokens != self.config.total_new_q_tokens
        ):
            raise ValueError(
                "tie_new_kv_to_query requires total_new_kv_tokens to match "
                "total_new_q_tokens"
            )
        self.config.cache_layout = _check_cache_layout(self.config.cache_layout)
        cached_length_mode = _check_length_mode(
            "cached_length_mode",
            self.config.cached_length_mode,
        )
        if cached_length_mode == "fixed_per_request":
            raise ValueError("cached_length_mode does not support fixed_per_request")
        self.config.cached_length_mode = cast(
            Literal["ragged", "regular"],
            cached_length_mode,
        )
        self.config.new_q_length_mode = _check_length_mode(
            "new_q_length_mode",
            self.config.new_q_length_mode,
        )
        if self.config.new_kv_length_mode is not None:
            self.config.new_kv_length_mode = _check_length_mode(
                "new_kv_length_mode",
                self.config.new_kv_length_mode,
            )
        if self.config.max_cached_tokens_per_request is not None:
            self.config.max_cached_tokens_per_request = _check_nonnegative(
                "max_cached_tokens_per_request",
                self.config.max_cached_tokens_per_request,
            )
        if self.config.max_new_q_tokens_per_request is not None:
            self.config.max_new_q_tokens_per_request = _check_positive(
                "max_new_q_tokens_per_request",
                self.config.max_new_q_tokens_per_request,
            )
        if self.config.max_new_kv_tokens_per_request is not None:
            self.config.max_new_kv_tokens_per_request = _check_nonnegative(
                "max_new_kv_tokens_per_request",
                self.config.max_new_kv_tokens_per_request,
            )
        if self.config.max_seqlen_k is not None:
            self.config.max_seqlen_k = _check_nonnegative(
                "max_seqlen_k", self.config.max_seqlen_k
            )
        if self.config.cache_layout == "none" and self.config.total_cached_tokens != 0:
            raise ValueError("cache_layout='none' requires total_cached_tokens == 0")
        if self.config.cache_layout == "none" and not self.config.tie_new_kv_to_query:
            raise ValueError("cache_layout='none' requires tie_new_kv_to_query=True")
