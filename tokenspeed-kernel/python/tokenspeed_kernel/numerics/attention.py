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

"""Numerics framework hooks for attention-family ops."""

from __future__ import annotations

import math
from typing import Any

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance
from tokenspeed_numerics_input_generators import (
    AttentionMergeStateInputConfig,
    AttentionMergeStateInputs,
)


def tolerance(dtype: torch.dtype, *, mode: str | None = None, **_: Any) -> Tolerance:
    if mode == "attn_merge_state":
        return Tolerance(atol=5.0e-2, rtol=5.0e-2)
    return Tolerance(atol=1.0e-1, rtol=5.0e-2)


set_family_tolerance("attention", tolerance)


class AttentionMergeStateInputGenerator(InputGenerator):
    """Adapter for attention merge-state input generation.

    Shape kwargs:
        total_q:   number of query rows being merged
        num_heads: number of attention heads
        head_dim:  per-head output dimension
    """

    def generate(
        self,
        *,
        total_q: int,
        num_heads: int,
        head_dim: int,
        lse_scale_log2: float = math.log2(math.e),
        lse_bound: float = 6.0,
    ) -> dict[str, Any]:
        values = AttentionMergeStateInputs(
            AttentionMergeStateInputConfig(
                total_q=total_q,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                lse_scale_log2=lse_scale_log2,
                lse_bound=lse_bound,
                device=self.device,
            )
        ).generate(seed=self.seed, device=self.device)
        return {
            "out_a": values.out_a,
            "lse_a": values.lse_a,
            "out_b": values.out_b,
            "lse_b": values.lse_b,
            "lse_scale_log2": values.lse_scale_log2,
        }


set_input_generator(
    "attention",
    "attn_merge_state",
    AttentionMergeStateInputGenerator,
)

_ATTN_MERGE_STATE_STANDARD_SHAPES: list[dict[str, int | float]] = [
    {"total_q": 1, "num_heads": 2, "head_dim": 64},
    {"total_q": 8, "num_heads": 4, "head_dim": 128},
]

set_standard_shapes(
    "attention",
    "attn_merge_state",
    _ATTN_MERGE_STATE_STANDARD_SHAPES,
)
set_benchmark_shapes(
    "attention",
    "attn_merge_state",
    _ATTN_MERGE_STATE_STANDARD_SHAPES,
)
