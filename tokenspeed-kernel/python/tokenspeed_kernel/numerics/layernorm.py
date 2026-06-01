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

from typing import Any

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance


def tolerance(
    dtype: torch.dtype,
    **_: Any,
) -> Tolerance:
    if dtype == torch.float16:
        return Tolerance(atol=2e-2, rtol=2e-2)
    if dtype == torch.bfloat16:
        return Tolerance(atol=2e-2, rtol=2e-2)
    if dtype == torch.float32:
        return Tolerance(atol=1e-5, rtol=1e-5)
    raise KeyError(f"No RMSNorm tolerance baseline for dtype={dtype}")


set_family_tolerance("norm", tolerance)


class RMSNormInputGenerator(InputGenerator):
    def _generate_value(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        values = torch.randn(
            *shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.rng,
        )
        return values.to(dtype)

    def generate(
        self,
        num_tokens: int,
        hidden_size: int,
        eps: float = 1e-6,
        residual: bool = False,
    ) -> dict[str, Any]:
        x = self._generate_value((num_tokens, hidden_size), self.dtype)
        residual_tensor = (
            self._generate_value((num_tokens, hidden_size), self.dtype)
            if residual
            else None
        )
        weight = self._generate_value((hidden_size,), torch.float32)

        return {
            "x": x,
            "weight": weight,
            "eps": eps,
            "residual": residual_tensor,
            "out": None,
        }


set_input_generator("norm", "rmsnorm", RMSNormInputGenerator)


RMSNORM_STANDARD_SHAPES: list[dict[str, int]] = [
    {"num_tokens": 1, "hidden_size": 128},
    {"num_tokens": 3, "hidden_size": 129},
    {"num_tokens": 7, "hidden_size": 2880},
    {"num_tokens": 11, "hidden_size": 2897},
]


RMSNORM_BENCHMARK_SHAPES: list[dict[str, int]] = [
    {"num_tokens": 1, "hidden_size": 4096},
    {"num_tokens": 8, "hidden_size": 4096},
    {"num_tokens": 16, "hidden_size": 4096},
    {"num_tokens": 1, "hidden_size": 7168},
    {"num_tokens": 8, "hidden_size": 7168},
    {"num_tokens": 16, "hidden_size": 7168},
    {"num_tokens": 64, "hidden_size": 7168},
]


set_standard_shapes("norm", "rmsnorm", RMSNORM_STANDARD_SHAPES)
set_benchmark_shapes("norm", "rmsnorm", RMSNORM_BENCHMARK_SHAPES)
