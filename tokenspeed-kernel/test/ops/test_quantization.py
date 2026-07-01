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
    quantize_fp8_with_scale,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize
from tokenspeed_kernel.platform import current_platform
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
)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2880),
        (8, 2880),
        (33, 2880),
        (4, 4096),
        (2, 1),
        (3, 513),
    ],
)
def test_quantize_fp8_pure_cast_bf16(
    device: str,
    solution: str,
    shape: tuple[int, ...],
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=shape,
            dtype=dtype,
            output_dtype=current_platform().fp8e4m3fn.dtype,
            granularity="none",
        )
    ).generate(seed=sum(shape), device=device)

    ref = fp8_quantization_reference(
        values.x,
        granularity="none",
        output_dtype=current_platform().fp8e4m3fn.dtype,
    ).to(current_platform().fp8e4m3fn.dtype)

    out = quantize_fp8(values.x, solution=solution)
    torch.cuda.synchronize()

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["triton"])
def test_quantize_mxfp4_dynamic_scales(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "mxfp4", solution, dtype, "x")
    values = MXFP4QuantizationInputs(
        MXFP4QuantizationInputConfig(
            shape=(2, 64),
            dtype=dtype,
        )
    ).generate(seed=11, device=device)
    ref_out, ref_scale = mxfp4_quantization_reference(values.x)

    out, scale = quantize_mxfp4(
        values.x,
        scale_size=values.scale_size,
        scale_layout=values.scale_layout,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == ref_out.shape
    assert out.dtype == torch.uint8
    assert scale.shape == ref_scale.shape
    assert scale.dtype == torch.uint8
    assert torch.equal(out.cpu(), ref_out.cpu())
    assert torch.equal(scale.cpu(), ref_scale.cpu())


@pytest.mark.parametrize("solution", ["triton"])
def test_quantize_fp8_strided_slice(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")

    s, h, qk_nope, v_head = 4096, 16, 128, 128
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(s, h, qk_nope + v_head),
            dtype=dtype,
            output_dtype=current_platform().fp8e4m3fn.dtype,
            granularity="none",
        )
    ).generate(seed=12, device=device)
    v = values.x[..., qk_nope:]
    assert not v.is_contiguous()

    ref = fp8_quantization_reference(
        v,
        granularity="none",
        output_dtype=current_platform().fp8e4m3fn.dtype,
    ).to(current_platform().fp8e4m3fn.dtype)

    out = quantize_fp8(v, solution=solution)
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize("scale", [2.0, 0.5, 7.5])
def test_quantize_fp8_scale_float(
    device: str,
    solution: str,
    scale: float,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(2048, 512),
            dtype=dtype,
            output_dtype=current_platform().fp8e4m3fn.dtype,
            granularity="none",
        )
    ).generate(seed=int(scale * 100), device=device)

    ref = fp8_quantization_reference(
        values.x,
        granularity="tensor",
        scale=scale,
        output_dtype=current_platform().fp8e4m3fn.dtype,
    ).to(current_platform().fp8e4m3fn.dtype)

    out = quantize_fp8(values.x, scale=scale, solution=solution)
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["triton"])
def test_quantize_fp8_scale_tensor(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(8, 2880),
            dtype=dtype,
            output_dtype=current_platform().fp8e4m3fn.dtype,
            granularity="tensor",
        )
    ).generate(seed=13, device=device)
    assert values.scale is not None
    ref = fp8_quantization_reference(
        values.x,
        granularity="tensor",
        scale=values.scale,
        output_dtype=current_platform().fp8e4m3fn.dtype,
    ).to(current_platform().fp8e4m3fn.dtype)

    out = quantize_fp8(values.x, scale=values.scale, solution=solution)
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize(
    "n",
    [
        # gpt-oss-120b: H = 2880 (hidden), I/tp = 2880/2 = 1440 (per-rank
        # ispp). Both are non-power-of-2, so the n-axis must be masked
        # both on load and on store for the W4A8 MoE forward path.
        2880,
        1440,
        # ``M`` not divisible by ``BLOCK_M`` exercises the m-axis tail mask
        # while ``N`` is non-pow2, ruling out a simple "round both up" bug.
        7,
        333,
    ],
)
def test_pure_cast_non_pow2_n(device: str, n: int) -> None:
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(33, n),
            dtype=torch.bfloat16,
            output_dtype=torch.float8_e4m3fn,
            granularity="none",
        )
    ).generate(seed=n, device=device)
    ref = fp8_quantization_reference(
        values.x,
        granularity="none",
        output_dtype=torch.float8_e4m3fn,
    ).to(torch.float8_e4m3fn)
    out = fp8_quantize(values.x)
    torch.cuda.synchronize()
    assert out.shape == ref.shape
    assert _bitwise_equal(out, ref)


@pytest.mark.skipif(
    not current_platform().is_cdna3,
    reason="float8_e4m3fnuz (tl.float8e4b8) is only supported on AMD CDNA3",
)
def test_pure_cast_e4m3fnuz(device: str) -> None:
    """CDNA3-specific fp8 dtype (bias=8). The Triton cast must saturate to
    ``±240`` to match ``x.to(torch.float8_e4m3fnuz)``."""
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(2048, 512),
            dtype=torch.bfloat16,
            output_dtype=torch.float8_e4m3fnuz,
            granularity="none",
        )
    ).generate(seed=14, device=device)
    ref = fp8_quantization_reference(
        values.x,
        granularity="none",
        output_dtype=torch.float8_e4m3fnuz,
    ).to(torch.float8_e4m3fnuz)
    out = fp8_quantize(values.x, fp8_dtype=torch.float8_e4m3fnuz)
    torch.cuda.synchronize()
    assert out.dtype == torch.float8_e4m3fnuz
    assert _bitwise_equal(out, ref)


@pytest.mark.skipif(
    not current_platform().is_cdna3,
    reason="float8_e4m3fnuz (tl.float8e4b8) is only supported on AMD CDNA3",
)
@pytest.mark.parametrize("scale", [2.0, 0.5, 7.5])
def test_scaled_cast_e4m3fnuz_matches_reference(device: str, scale: float) -> None:
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(2048, 512),
            dtype=torch.bfloat16,
            output_dtype=torch.float8_e4m3fnuz,
            granularity="none",
        )
    ).generate(seed=int(scale * 100) + 14, device=device)
    ref = fp8_quantization_reference(
        values.x,
        granularity="tensor",
        scale=torch.tensor([scale], dtype=torch.float32, device=device),
        output_dtype=torch.float8_e4m3fnuz,
    ).to(torch.float8_e4m3fnuz)
    out = fp8_quantize(values.x, scale=scale, fp8_dtype=torch.float8_e4m3fnuz)
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["trtllm"])
@pytest.mark.parametrize("granularity", ["tensor", "token"])
def test_quantize_fp8_with_scale_tensor_and_token(
    device: str,
    solution: str,
    granularity: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", solution, dtype, "x")
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(16, 128),
            dtype=dtype,
            output_dtype=current_platform().fp8e4m3fn.dtype,
            granularity=granularity,
        )
    ).generate(seed=15, device=device)
    assert values.scale is not None
    fp8 = current_platform().fp8e4m3fn

    out, scale = quantize_fp8_with_scale(
        values.x,
        granularity=granularity,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == values.x.shape
    assert out.dtype == fp8.dtype
    assert scale.dtype == torch.float32
    if granularity == "tensor":
        assert scale.shape == (1,)
    else:
        assert scale.shape == (values.x.shape[0], 1)


@pytest.mark.parametrize("solution", ["trtllm"])
def test_quantize_fp8_with_scale_token_group(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", solution, dtype, "x")
    values = FP8QuantizationInputs(
        FP8QuantizationInputConfig(
            shape=(16, 256),
            dtype=dtype,
            output_dtype=current_platform().fp8e4m3fn.dtype,
            granularity="token_group",
            group_size=128,
        )
    ).generate(seed=16, device=device)
    assert values.scale is not None
    fp8 = current_platform().fp8e4m3fn

    out, scale = quantize_fp8_with_scale(
        values.x,
        granularity="token_group",
        group_size=128,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == values.x.shape
    assert out.dtype == fp8.dtype
    assert scale.dtype == torch.float32
    assert scale.numel() > 0


@pytest.mark.parametrize("solution", ["flashinfer"])
def test_quantize_mxfp8_shape_and_scale(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "mxfp8", solution, dtype, "x")
    values = MXFP8QuantizationInputs(
        MXFP8QuantizationInputConfig(
            shape=(17, 2880),
            dtype=dtype,
        )
    ).generate(seed=17, device=device)

    out, scale = quantize_mxfp8(values.x, solution=solution)
    torch.cuda.synchronize()

    assert out.shape[:-1] == values.x.shape[:-1]
    assert out.shape[-1] >= values.x.shape[-1]
    assert scale.numel() > 0


@pytest.mark.parametrize("solution", ["flashinfer"])
def test_quantize_nvfp4_shape_and_scale(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "nvfp4", solution, dtype, "x")
    values = NVFP4QuantizationInputs(
        NVFP4QuantizationInputConfig(
            shape=(16, 256),
            dtype=dtype,
            scale=0.125,
        )
    ).generate(seed=18, device=device)

    out, scale = quantize_nvfp4(
        values.x,
        scale=values.scale,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape[:-1] == values.x.shape[:-1]
    assert out.shape[-1] == values.x.shape[-1] // 2
    assert scale.numel() > 0
