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
from tokenspeed_kernel import quantize_fp8
from tokenspeed_numerics_input_generators import (
    FP8QuantizationInputConfig,
    FP8QuantizationInputs,
    fp8_quantization_reference,
)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("solution", ["triton"])
def test_fp8_quantization_generator_runs_pure_cast_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(33, 333),
            dtype=dtype,
            granularity="none",
        )
    ).generate(seed=51, device=device)

    out = quantize_fp8(values.x, solution=solution)
    ref = fp8_quantization_reference(values.x, granularity="none").to(out.dtype)
    torch.cuda.synchronize()

    assert out.shape == values.x.shape
    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["triton"])
def test_fp8_quantization_generator_runs_scaled_cast_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(17, 512),
            dtype=dtype,
            granularity="tensor",
        )
    ).generate(seed=52, device=device)
    assert values.scale is not None

    out = quantize_fp8(values.x, scale=values.scale, solution=solution)
    ref = fp8_quantization_reference(
        values.x,
        granularity="tensor",
        scale=values.scale,
    ).to(out.dtype)
    torch.cuda.synchronize()

    assert out.shape == values.x.shape
    assert _bitwise_equal(out, ref)
