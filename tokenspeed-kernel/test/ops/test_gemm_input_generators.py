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
    LMHeadProjectionInputConfig,
    LMHeadProjectionInputs,
    RouterProjectionInputConfig,
    RouterProjectionInputs,
    gemm_reference,
    lm_head_projection_reference,
    mxfp4_gemm_input_config,
    mxfp8_gemm_input_config,
    router_projection_reference,
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


def test_router_projection_generator_runs_fp32_router_gemm(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("fp32_router_gemm requires NVIDIA SM90+")

    from tokenspeed_kernel.thirdparty.cuda import fp32_router_gemm

    values = RouterProjectionInputs(
        RouterProjectionInputConfig(
            num_tokens=8,
            hidden_dim=3072,
            num_experts=256,
            hidden_dtype=torch.bfloat16,
            router_weight_dtype=torch.float32,
        )
    ).generate(seed=43, device=device)

    try:
        actual = fp32_router_gemm(values.hidden_states, values.router_weights)
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)

    expected = router_projection_reference(values).to(device=device)
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)


def test_router_projection_generator_runs_dsv3_router_gemm(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("dsv3_router_gemm requires NVIDIA SM90+")

    from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm

    values = RouterProjectionInputs(
        RouterProjectionInputConfig(
            num_tokens=8,
            hidden_dim=7168,
            num_experts=256,
            hidden_dtype=torch.bfloat16,
            router_weight_dtype=torch.bfloat16,
        )
    ).generate(seed=47, device=device)

    try:
        actual = dsv3_router_gemm(
            values.hidden_states,
            values.router_weights,
            out_dtype=torch.float32,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)

    expected = router_projection_reference(values).to(device=device)
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

    values = LMHeadProjectionInputs(
        LMHeadProjectionInputConfig(
            num_tokens=1,
            hidden_dim=7168,
            vocab_size=16160,
            hidden_dtype=torch.bfloat16,
            weight_dtype=torch.bfloat16,
        )
    ).generate(seed=49, device=device)

    if not is_supported(values.hidden_states, values.weight):
        pytest.skip("lm_head_gemm reports generated shape as unsupported")

    try:
        actual = lm_head_gemm(values.hidden_states, values.weight)
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_cuda_extension_gemm_unavailable(exc)

    expected = lm_head_projection_reference(values).to(device=device)
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
        mxfp8_gemm_input_config(
            M=M,
            N=N,
            K=K,
            c_dtype=torch.float16,
            block_shape=tuple(block_size),
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
        mxfp8_gemm_input_config(
            M=128,
            N=128,
            K=256,
            c_dtype=torch.bfloat16,
            block_shape=tuple(block_size),
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
