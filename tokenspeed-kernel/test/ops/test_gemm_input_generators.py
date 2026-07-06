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

import math

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    CustomDType,
    GemmInputConfig,
    GemmInputs,
    gemm_reference,
    gemm_scale_shape,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GEMM generator kernel compatibility tests require a CUDA/ROCm GPU.",
)


def _skip_if_cuda_extension_gemm_unavailable(exc: BaseException) -> None:
    message = str(exc)
    skip_fragments = (
        "library not found",
        "No module named 'tvm_ffi'",
        "requires SM90",
        "required CUDA ARCH",
        "unsupported (hd_in",
    )
    if any(fragment in message for fragment in skip_fragments):
        pytest.skip(message)
    raise exc


def test_dense_gemm_generator_runs_reference_kernel(device: str, require) -> None:
    require("gemm", "mm", "reference", torch.float32, "a")
    values = GemmInputs(
        GemmInputConfig(
            M=7,
            N=11,
            K=13,
            a_dtype=torch.float32,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
        )
    ).generate(seed=29, device=device)
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is None
    assert values.B_scales is None

    actual = tokenspeed_kernel.mm(
        values.A,
        values.B,
        C=values.C,
        out_dtype=values.C.dtype,
        expected_kernel_name="torch_mm",
    )
    expected = gemm_reference(values).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == values.C.shape
    assert actual.dtype == values.C.dtype
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_mxfp4_gemm_generator_runs_triton_kernel(device: str, require) -> None:
    require("gemm", "mm", "triton", torch.uint8, "a")
    values = GemmInputs(
        GemmInputConfig(
            M=5,
            N=33,
            K=64,
            a_dtype=CustomDType.MXFP4,
            b_dtype=CustomDType.MXFP4,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=5,
                N=33,
                K=64,
                block_shape=(32,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=5,
                N=33,
                K=64,
                block_shape=(32,),
            ),
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


def test_router_projection_generator_runs_fp32_router_gemm(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("fp32_router_gemm requires NVIDIA SM90+")

    from tokenspeed_kernel.thirdparty.cuda import fp32_router_gemm

    hidden_dim = 3072
    values = GemmInputs(
        GemmInputConfig(
            M=8,
            N=256,
            K=hidden_dim,
            a_dtype=torch.bfloat16,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
        )
    ).generate(seed=43, device=device)
    assert values.A is not None
    assert values.B is not None
    values.B = (values.B.float() / math.sqrt(hidden_dim)).contiguous()

    try:
        actual = fp32_router_gemm(values.A, values.B)
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)

    expected = gemm_reference(values, out_dtype=torch.float32).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)


def test_router_projection_generator_runs_dsv3_router_gemm(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("dsv3_router_gemm requires NVIDIA SM90+")

    from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm

    hidden_dim = 7168
    values = GemmInputs(
        GemmInputConfig(
            M=8,
            N=256,
            K=hidden_dim,
            a_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            c_dtype=torch.float32,
        )
    ).generate(seed=47, device=device)
    assert values.A is not None
    assert values.B is not None
    values.B = (values.B.float() / math.sqrt(hidden_dim)).to(torch.bfloat16).contiguous()

    try:
        actual = dsv3_router_gemm(
            values.A,
            values.B,
            out_dtype=torch.float32,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)

    expected = gemm_reference(values, out_dtype=torch.float32).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)


def test_lm_head_projection_generator_runs_fused_lm_head_gemm(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("lm_head_gemm requires NVIDIA SM90+")

    try:
        from tokenspeed_kernel.thirdparty.cuda.lm_head_gemm import (
            is_supported,
            lm_head_gemm,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)

    hidden_dim = 7168
    values = GemmInputs(
        GemmInputConfig(
            M=1,
            N=16160,
            K=hidden_dim,
            a_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            c_dtype=torch.bfloat16,
        )
    ).generate(seed=49, device=device)
    assert values.A is not None
    assert values.B is not None
    values.B = (values.B.float() / math.sqrt(hidden_dim)).to(torch.bfloat16).contiguous()

    if not is_supported(values.A, values.B):
        pytest.skip("lm_head_gemm reports generated shape as unsupported")

    try:
        actual = lm_head_gemm(values.A, values.B)
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)

    expected = gemm_reference(values).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.1, rtol=0.1)


def test_mxfp8_blockscale_gemm_generator_runs_triton_kernel(
    device: str,
    require,
) -> None:
    if not current_platform().is_blackwell_plus:
        pytest.skip("triton_mm_fp8_blockscale requires NVIDIA Blackwell or newer")
    require("gemm", "mm", "triton", torch.float8_e4m3fn, "a")

    block_size = [128, 128]
    M, N, K = 8, 256, 256
    values = GemmInputs(
        GemmInputConfig(
            M=M,
            N=N,
            K=K,
            a_dtype=torch.float8_e4m3fn,
            b_dtype=torch.float8_e4m3fn,
            c_dtype=torch.float16,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=M,
                N=N,
                K=K,
                block_shape=tuple(block_size),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=M,
                N=N,
                K=K,
                block_shape=tuple(block_size),
            ),
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
        )
    ).generate(seed=41, device=device)
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
        block_size=block_size,
        quant="mxfp8",
        expected_kernel_name="triton_mm_fp8_blockscale",
    )
    expected = gemm_reference(values).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == values.C.shape
    assert actual.dtype == values.C.dtype
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.25, rtol=0.25)


def test_mxfp8_gemm_generator_runs_deep_gemm_kernel(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("deep_gemm_mm_fp8_blockscale requires NVIDIA SM90+")

    from tokenspeed_kernel.ops.gemm import deep_gemm as deep_gemm_ops

    kernel = getattr(deep_gemm_ops, "deep_gemm_mm_fp8_blockscale", None)
    if kernel is None:
        pytest.skip("DeepGEMM kernel is not available")

    block_size = [128, 128]
    values = GemmInputs(
        GemmInputConfig(
            M=128,
            N=128,
            K=256,
            a_dtype=torch.float8_e4m3fn,
            b_dtype=torch.float8_e4m3fn,
            c_dtype=torch.bfloat16,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=128,
                N=128,
                K=256,
                block_shape=tuple(block_size),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=128,
                N=128,
                K=256,
                block_shape=tuple(block_size),
            ),
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
        )
    ).generate(seed=53, device=device)
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is not None
    assert values.B_scales is not None

    try:
        actual = kernel(
            values.A,
            values.B,
            values.A_scales,
            values.B_scales,
            values.C.dtype,
            block_size=block_size,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)
    expected = gemm_reference(values).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == values.C.shape
    assert actual.dtype == values.C.dtype
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.25, rtol=0.25)
