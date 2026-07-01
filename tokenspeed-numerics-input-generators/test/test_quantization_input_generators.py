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
from tokenspeed_numerics_input_generators import (
    FP8QuantizationInputConfig,
    FP8QuantizationInputs,
    fp8_quantization_reference,
    fp8_scale_shape,
)


def test_fp8_scale_shape_matches_granularity() -> None:
    shape = (5, 256)

    assert fp8_scale_shape(shape, granularity="none") is None
    assert fp8_scale_shape(shape, granularity="tensor") == (1,)
    assert fp8_scale_shape(shape, granularity="token") == (5, 1)
    assert fp8_scale_shape(shape, granularity="token_group", group_size=128) == (5, 2)


@pytest.mark.parametrize("granularity", ["none", "tensor", "token", "token_group"])
def test_fp8_quantization_inputs_generate_values_and_scales(
    granularity: str,
) -> None:
    group_size = 128 if granularity == "token_group" else None
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(7, 256),
            dtype=torch.float16,
            granularity=granularity,  # type: ignore[arg-type]
            group_size=group_size,
        )
    ).generate(seed=41, device="cpu")

    assert values.x.shape == (7, 256)
    assert values.x.dtype == torch.float16
    assert (
        None if values.scale is None else tuple(values.scale.shape)
    ) == fp8_scale_shape((7, 256), granularity=granularity, group_size=group_size)

    ref = fp8_quantization_reference(
        values.x,
        granularity=granularity,  # type: ignore[arg-type]
        group_size=group_size,
        scale=values.scale,
    )
    assert ref.shape == values.x.shape
    assert ref.dtype == torch.float32
    assert torch.isfinite(ref).all()


def test_fp8_token_group_scale_values_match_generated_input() -> None:
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(3, 256),
            dtype=torch.float32,
            granularity="token_group",
            group_size=128,
        )
    ).generate(seed=42, device="cpu")

    assert values.scale is not None
    grouped = values.x.float().view(3, 2, 128)
    expected = (
        grouped.abs().amax(dim=-1) / torch.finfo(torch.float8_e4m3fn).max
    ).clamp(min=1e-10)
    torch.testing.assert_close(values.scale, expected)


def test_fp8_quantization_reference_matches_manual_token_quantization() -> None:
    x = torch.tensor(
        [[0.0, 1.0, -2.0], [4.0, -8.0, 16.0]],
        dtype=torch.float32,
    )
    scale = x.abs().amax(dim=-1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max
    ref = fp8_quantization_reference(x, granularity="token", scale=scale)

    manual = (x / scale).clamp(
        torch.finfo(torch.float8_e4m3fn).min,
        torch.finfo(torch.float8_e4m3fn).max,
    )
    manual = manual.to(torch.float8_e4m3fn).float()

    torch.testing.assert_close(ref, manual)


def test_fp8_quantization_rejects_token_group_without_group_size() -> None:
    with pytest.raises(ValueError, match="requires group_size"):
        FP8QuantizationInputs(
            FP8QuantizationInputConfig(
                shape=(4, 256),
                dtype=torch.float16,
                granularity="token_group",
            )
        )


def test_fp8_quantization_rejects_incompatible_group_size() -> None:
    with pytest.raises(ValueError, match="last dimension must be divisible"):
        FP8QuantizationInputs(
            FP8QuantizationInputConfig(
                shape=(4, 255),
                dtype=torch.float16,
                granularity="token_group",
                group_size=128,
            )
        )


def test_fp8_quantization_rejects_fp8_input_dtype() -> None:
    with pytest.raises(ValueError, match="before FP8 cast"):
        FP8QuantizationInputs(
            FP8QuantizationInputConfig(
                shape=(4, 128),
                dtype=torch.float8_e4m3fn,
            )
        )


def test_fp8_quantization_rejects_non_float_encoding_for_tensor_scale() -> None:
    with pytest.raises(ValueError, match="only meaningful for token_group"):
        FP8QuantizationInputs(
            FP8QuantizationInputConfig(
                shape=(4, 128),
                dtype=torch.float16,
                granularity="tensor",
                scale_encoding="ue8m0",
            )
        )
