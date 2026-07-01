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
import tokenspeed_kernel
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    GemmInputConfig,
    GemmInputs,
    gemm_reference,
    mxfp4_gemm_input_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GEMM generator kernel compatibility tests require a CUDA/ROCm GPU.",
)


def test_mxfp4_gemm_generator_runs_triton_kernel(device: str, require) -> None:
    require("gemm", "mm", "triton", torch.uint8, "a")
    values = GemmInputs(
        mxfp4_gemm_input_config(
            M=5,
            N=33,
            K=64,
            c_dtype=torch.float32,
        )
    ).generate(seed=31, device=device)
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is not None
    assert values.B_scales is not None

    actual = tokenspeed_kernel.mm(
        values.A,
        values.B,
        A_scales=values.A_scales,
        B_scales=values.B_scales,
        C=values.C,
        out_dtype=values.C.dtype,
        quant="mxfp4",
        expected_kernel_name="triton_mm_mxfp4",
    )
    expected = gemm_reference(values).to(device=device)
    torch.cuda.synchronize()

    assert actual.data_ptr() == values.C.data_ptr()
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_fp8_scaled_gemm_generator_runs_triton_kernel(
    device: str,
    require,
) -> None:
    if not current_platform().is_blackwell_plus:
        pytest.skip("triton_mm_fp8_scaled requires NVIDIA Blackwell or newer")
    require("gemm", "mm", "triton", torch.float8_e4m3fn, "a")

    values = GemmInputs(
        GemmInputConfig(
            M=8,
            N=64,
            K=64,
            a_dtype=torch.float8_e4m3fn,
            b_dtype=torch.float8_e4m3fn,
            c_dtype=torch.float16,
            a_scale_shape=(8,),
            b_scale_shape=(64,),
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
        )
    ).generate(seed=37, device=device)
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is not None
    assert values.B_scales is not None

    actual = tokenspeed_kernel.mm(
        values.A,
        values.B.transpose(0, 1).contiguous(),
        A_scales=values.A_scales,
        B_scales=values.B_scales,
        C=values.C,
        out_dtype=values.C.dtype,
        quant="fp8",
        expected_kernel_name="triton_mm_fp8_scaled",
    )
    expected = gemm_reference(values).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == values.C.shape
    assert actual.dtype == values.C.dtype
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.25, rtol=0.25)
