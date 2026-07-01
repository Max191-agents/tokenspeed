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
from tokenspeed_kernel import (
    quantize_fp8,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tokenspeed_numerics_input_generators import (
    FP8QuantizationInputConfig,
    FP8QuantizationInputs,
    MXFP4QuantizationInputConfig,
    MXFP4QuantizationInputs,
    MXFP8QuantizationInputConfig,
    MXFP8QuantizationInputs,
    NVFP4QuantizationInputConfig,
    NVFP4QuantizationInputs,
    fp8_quantization_reference,
    mxfp4_quantization_reference,
    mxfp8_quantization_reference,
    nvfp4_quantization_reference,
)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _uint8_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.to(torch.uint8), b.to(torch.uint8))


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
def test_mxfp4_quantization_generator_runs_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "mxfp4", solution, dtype, "x")
    values = MXFP4QuantizationInputs(
        MXFP4QuantizationInputConfig(
            shape=(5, 64),
            dtype=dtype,
        )
    ).generate(seed=53, device=device)
    expected_out, expected_scales = mxfp4_quantization_reference(values.x)

    out, scales = quantize_mxfp4(
        values.x,
        scale_size=values.scale_size,
        scale_layout=values.scale_layout,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == expected_out.shape
    assert scales.shape == expected_scales.shape
    assert _uint8_equal(out, expected_out)
    assert _uint8_equal(scales, expected_scales)


@pytest.mark.parametrize("solution", ["flashinfer"])
def test_mxfp8_quantization_generator_runs_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "mxfp8", solution, dtype, "x")
    values = MXFP8QuantizationInputs(
        MXFP8QuantizationInputConfig(
            shape=(5, 64),
            dtype=dtype,
        )
    ).generate(seed=54, device=device)
    expected_out, _expected_scales = mxfp8_quantization_reference(values.x)

    out, scales = quantize_mxfp8(values.x, solution=solution)
    torch.cuda.synchronize()

    assert out.shape == expected_out.shape
    assert _bitwise_equal(out, expected_out)
    assert scales.numel() > 0


@pytest.mark.parametrize("solution", ["flashinfer"])
def test_nvfp4_quantization_generator_runs_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "nvfp4", solution, dtype, "x")
    values = NVFP4QuantizationInputs(
        NVFP4QuantizationInputConfig(
            shape=(5, 64),
            dtype=dtype,
            scale=0.125,
        )
    ).generate(seed=55, device=device)
    expected_out, expected_scales = nvfp4_quantization_reference(
        values.x,
        scale=values.scale,
    )

    out, scales = quantize_nvfp4(
        values.x,
        scale=values.scale,
        scale_layout=values.scale_layout,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == expected_out.shape
    assert scales.shape == expected_scales.shape
    assert _uint8_equal(out, expected_out)
    assert _bitwise_equal(scales, expected_scales)


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
