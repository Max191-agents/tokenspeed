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
    MXFP4QuantizationInputConfig,
    MXFP4QuantizationInputs,
    fp8_quantization_reference,
    fp8_scale_shape,
    mxfp4_quantization_reference,
    mxfp4_scale_shape,
)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    magnitude_bits = nibbles & 0x7
    exponent = (magnitude_bits >> 1).to(torch.float32)
    mantissa = (magnitude_bits & 0x1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent - 1.0)
    subnormal = 0.5 * mantissa
    magnitude = torch.where(exponent == 0, subnormal, normal)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
    return magnitude * sign


def _dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = packed.new_empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.float32,
    )
    out[..., 0::2] = _e2m1_values(packed & 0xF)
    out[..., 1::2] = _e2m1_values(packed >> 4)
    scale_values = torch.pow(2.0, scale.to(torch.int32) - 127).to(torch.float32)
    return out * scale_values.repeat_interleave(32, dim=-1)


def test_fp8_scale_shape_matches_granularity() -> None:
    shape = (5, 256)

    assert fp8_scale_shape(shape, granularity="none") is None
    assert fp8_scale_shape(shape, granularity="tensor") == (1,)
    assert fp8_scale_shape(shape, granularity="token") == (5, 1)
    assert fp8_scale_shape(shape, granularity="token_group", group_size=128) == (5, 2)


def test_mxfp4_scale_shape_matches_scale_groups() -> None:
    assert mxfp4_scale_shape((5, 256)) == (5, 8)


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


def test_mxfp4_quantization_inputs_generate_representable_values() -> None:
    values = MXFP4QuantizationInputs(
        MXFP4QuantizationInputConfig(
            shape=(3, 64),
            dtype=torch.bfloat16,
        )
    ).generate(seed=43, device="cpu")

    assert values.x.shape == (3, 64)
    assert values.x.dtype == torch.bfloat16
    assert values.scale_size == 32
    assert values.scale_layout == "linear"

    packed, scales = mxfp4_quantization_reference(
        values.x,
        scale_size=values.scale_size,
        scale_layout=values.scale_layout,
    )
    assert packed.shape == (3, 32)
    assert packed.dtype == torch.uint8
    assert scales.shape == (3, 2)
    assert scales.dtype == torch.uint8
    dequant = _dequantize_mxfp4(packed, scales)
    torch.testing.assert_close(dequant, values.x.float(), atol=0, rtol=0)


def test_mxfp4_quantization_reference_matches_known_values() -> None:
    base = torch.tensor(
        [
            0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.0,
            -2.0,
            3.0,
            -3.0,
            4.0,
            -4.0,
            6.0,
            -6.0,
            0.0,
        ],
        dtype=torch.bfloat16,
    )
    row = torch.cat([base, base, base * 0.25, base * 0.25], dim=0)
    x = torch.stack([row, row], dim=0)

    packed, scales = mxfp4_quantization_reference(x)

    assert packed.shape == (2, 32)
    assert scales.shape == (2, 2)
    torch.testing.assert_close(
        scales,
        torch.tensor([[127, 125], [127, 125]], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        _dequantize_mxfp4(packed, scales), x.float(), atol=0, rtol=0
    )


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


def test_mxfp4_quantization_rejects_incompatible_last_dim() -> None:
    with pytest.raises(ValueError, match="last dimension must be divisible"):
        MXFP4QuantizationInputs(
            MXFP4QuantizationInputConfig(
                shape=(4, 63),
                dtype=torch.bfloat16,
            )
        )


def test_mxfp4_quantization_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="bf16 or fp16"):
        MXFP4QuantizationInputs(
            MXFP4QuantizationInputConfig(
                shape=(4, 64),
                dtype=torch.float32,
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
