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
import torch
from tokenspeed_numerics_input_generators import (
    CustomDType,
    GemmInputConfig,
    GemmInputs,
    GemmInputValues,
    MoeAlignBlockSizeInputConfig,
    MoeAlignBlockSizeInputs,
    MoeAlignBlockSizeInputValues,
    MoEBiasedGroupedTopKInputConfig,
    MoEBiasedGroupedTopKInputs,
    MoEDeepSeekV4MegaMoEStagingInputConfig,
    MoEDeepSeekV4MegaMoEStagingInputs,
    MoEFinalizeFuseSharedInputConfig,
    MoEFinalizeFuseSharedInputs,
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
    MoESoftmaxTopKRoutingInputConfig,
    MoESoftmaxTopKRoutingInputs,
    MoESoftplusSqrtTopKRoutingInputConfig,
    MoESoftplusSqrtTopKRoutingInputs,
    TensorInput,
    gemm_reference,
    gemm_scale_shape,
    nvfp4_dequantization_reference,
    nvfp4_quantization_reference,
)

_fp8_dtype = torch.float8_e4m3fn


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64, _fp8_dtype],
)
def test_tensor_input_generates_standard_fp_dtypes(dtype: torch.dtype) -> None:
    first = TensorInput((2, 3), dtype).generate(seed=123, device="cpu")
    second = TensorInput((2, 3), dtype).generate(seed=123, device="cpu")

    assert first.values is not None
    assert second.values is not None
    assert first.scales is None
    assert second.scales is None
    assert first.values.shape == (2, 3)
    assert first.values.dtype == dtype
    assert torch.equal(first.values, second.values)


def test_tensor_input_dtype_none_skips_generation() -> None:
    tensor = TensorInput((2, 3), None).generate(seed=123, device="cpu")

    assert tensor.values is None
    assert tensor.scales is None


def test_tensor_input_rejects_non_floating_torch_dtype() -> None:
    with pytest.raises(ValueError, match="TensorInput only supports floating"):
        TensorInput((2, 3), torch.int32).generate(seed=123, device="cpu")


def test_tensor_input_uses_named_arguments() -> None:
    tensor = TensorInput((2, 3), torch.float32)

    values = tensor.generate(seed=123, device="cpu")

    assert values is not None
    assert values.values is not None
    assert values.scales is None
    assert values.values.shape == (2, 3)
    assert values.values.dtype == torch.float32


def test_tensor_input_generates_torch_values_and_scales() -> None:
    tensor = TensorInput(
        (4, 5),
        torch.float16,
        scale_shape=(4, 1),
        scale_dtype=torch.float32,
    ).generate(seed=125, device="cpu")

    assert tensor.values is not None
    assert tensor.scales is not None
    assert tensor.values.shape == (4, 5)
    assert tensor.scales.shape == (4, 1)
    assert tensor.values.dtype == torch.float16
    assert tensor.scales.dtype == torch.float32
    assert torch.all(tensor.values.float().abs() <= 1.0)
    assert torch.all(tensor.scales > 0.0)
    assert torch.all(tensor.scales <= 1.0)


def test_tensor_input_reuses_mutable_dtype() -> None:
    tensor = TensorInput(
        (4, 5),
        torch.float16,
        scale_shape=(4, 1),
        scale_dtype=torch.float32,
    )
    tensor.dtype = torch.float64

    values = tensor.generate(seed=125, device="cpu")

    assert values.values is not None
    assert values.values.dtype == torch.float64


def test_tensor_input_generates_mxfp4_values_and_scales() -> None:
    tensor = TensorInput(
        (4, 8),
        CustomDType.MXFP4,
        scale_shape=(4, 1),
    ).generate(seed=126, device="cpu")

    assert tensor.values is not None
    assert tensor.scales is not None
    assert tensor.values.shape == (4, 8)
    assert tensor.values.dtype == torch.uint8
    assert tensor.scales.dtype == torch.uint8
    assert torch.all(tensor.scales >= 121)
    assert torch.all(tensor.scales <= 124)


def test_tensor_input_requires_mxfp4_scales() -> None:
    with pytest.raises(ValueError, match="mxfp4 tensors require scale_shape"):
        TensorInput((4, 8), CustomDType.MXFP4)


def test_tensor_input_requires_ue8m0_scales_for_mxfp4() -> None:
    with pytest.raises(ValueError, match="mxfp4 scale_dtype"):
        TensorInput(
            (4, 8),
            CustomDType.MXFP4,
            scale_shape=(4, 1),
            scale_dtype=torch.float32,
        )

    values = TensorInput(
        (4, 8),
        CustomDType.MXFP4,
        scale_shape=(4, 1),
        scale_dtype=CustomDType.UE8M0,
    ).generate(seed=127, device="cpu")

    assert values.scales is not None
    assert values.scales.dtype == torch.uint8


def test_tensor_input_generates_nvfp4_values_and_scales() -> None:
    tensor = TensorInput(
        (4, 8),
        CustomDType.NVFP4,
        scale_shape=(4, 1),
    ).generate(seed=130, device="cpu")

    assert tensor.values is not None
    assert tensor.scales is not None
    assert tensor.values.shape == (4, 8)
    assert tensor.values.dtype == torch.uint8
    assert tensor.scales.shape == (4, 1)
    assert tensor.scales.dtype == torch.float8_e4m3fn
    assert torch.all(tensor.scales.float() > 0.0)


def test_tensor_input_requires_fp8_scales_for_nvfp4() -> None:
    with pytest.raises(ValueError, match="nvfp4 tensors require scale_shape"):
        TensorInput((4, 8), CustomDType.NVFP4)

    with pytest.raises(ValueError, match="nvfp4 scale_dtype"):
        TensorInput(
            (4, 8),
            CustomDType.NVFP4,
            scale_shape=(4, 1),
            scale_dtype=torch.bfloat16,
        )

    for scale_dtype in (torch.float8_e4m3fn, torch.float32, torch.uint8):
        values = TensorInput(
            (4, 8),
            CustomDType.NVFP4,
            scale_shape=(4, 1),
            scale_dtype=scale_dtype,
        ).generate(seed=131, device="cpu")

        assert values.scales is not None
        assert values.scales.dtype == scale_dtype


def test_tensor_input_generates_float4_nvfp4_storage() -> None:
    values = TensorInput(
        (4, 8),
        torch.float4_e2m1fn_x2,
        scale_shape=(4, 1),
        scale_dtype=torch.uint8,
    ).generate(seed=132, device="cpu")

    assert values.values is not None
    assert values.scales is not None
    assert values.values.shape == (4, 8)
    assert values.values.dtype == torch.float4_e2m1fn_x2
    assert values.values.view(torch.uint8).dtype == torch.uint8
    assert values.scales.dtype == torch.uint8
    assert torch.all(values.scales.view(torch.float8_e4m3fn).float() > 0.0)


def test_tensor_input_generates_mxint4_values_and_scales() -> None:
    tensor = TensorInput(
        (4, 8),
        CustomDType.MXINT4,
        scale_shape=(4, 1),
    ).generate(seed=128, device="cpu")

    assert tensor.values is not None
    assert tensor.scales is not None
    assert tensor.values.shape == (4, 8)
    assert tensor.values.dtype == torch.uint8
    assert tensor.scales.shape == (4, 1)
    assert tensor.scales.dtype == torch.bfloat16
    assert torch.all(tensor.scales.float() > 0.0)
    assert torch.all(tensor.scales.float() <= 0.25)


def test_tensor_input_requires_bf16_scales_for_mxint4() -> None:
    with pytest.raises(ValueError, match="mxint4 tensors require scale_shape"):
        TensorInput((4, 8), CustomDType.MXINT4)

    with pytest.raises(ValueError, match="mxint4 scale_dtype"):
        TensorInput(
            (4, 8),
            CustomDType.MXINT4,
            scale_shape=(4, 1),
            scale_dtype=torch.float32,
        )

    values = TensorInput(
        (4, 8),
        CustomDType.MXINT4,
        scale_shape=(4, 1),
        scale_dtype=torch.bfloat16,
    ).generate(seed=129, device="cpu")

    assert values.scales is not None
    assert values.scales.dtype == torch.bfloat16


def test_tensor_input_requires_scale_shape_and_dtype_together() -> None:
    with pytest.raises(ValueError, match="scale_shape and scale_dtype"):
        TensorInput((4, 5), torch.float16, scale_shape=(4, 1))
    with pytest.raises(ValueError, match="scale_shape and scale_dtype"):
        TensorInput((4, 5), torch.float16, scale_dtype=torch.float32)


def test_tensor_input_rejects_incompatible_scale_shape() -> None:
    with pytest.raises(ValueError, match="scale_shape must be prefix-compatible"):
        TensorInput(
            (4, 5),
            torch.float16,
            scale_shape=(4, 6),
            scale_dtype=torch.float32,
        )


def test_gemm_inputs_generate_operands_and_layouts() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=2,
            N=3,
            K=4,
            a_dtype=torch.float16,
            b_dtype=torch.float32,
            c_dtype=torch.float64,
            a_layout="KM",
            b_layout="KN",
        )
    ).generate(seed=5, device="cpu")

    assert inputs.A is not None
    assert inputs.B is not None
    assert inputs.C is not None
    assert inputs.A.shape == (4, 2)
    assert inputs.B.shape == (4, 3)
    assert inputs.C.shape == (2, 3)
    assert inputs.A.dtype == torch.float16
    assert inputs.B.dtype == torch.float32
    assert inputs.C.dtype == torch.float64


def test_gemm_inputs_use_mutable_config_fields() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=2,
            N=3,
            K=4,
            a_dtype=torch.float16,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
        )
    )

    inputs.config.b_dtype = torch.float64
    values = inputs.generate(seed=7, device="cpu")

    assert values.B is not None
    assert values.B.dtype == torch.float64
    assert values.C.shape == (2, 3)
    assert values.C.dtype == torch.float32


def test_gemm_inputs_accept_config_objects() -> None:
    config = GemmInputConfig(
        M=2,
        N=3,
        K=4,
        a_dtype=torch.float16,
        b_dtype=torch.float64,
        c_dtype=torch.float32,
    )
    inputs = GemmInputs(config)

    values = inputs.generate(seed=8, device="cpu")

    assert inputs.config is config
    assert values.B is not None
    assert values.B.dtype == torch.float64
    assert values.C.shape == (2, 3)


def test_gemm_reference_handles_dense_layouts() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=2,
            N=3,
            K=4,
            a_dtype=torch.float32,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
            a_layout="KM",
            b_layout="KN",
        )
    ).generate(seed=10, device="cpu")
    assert values.A is not None
    assert values.B is not None

    ref = gemm_reference(values, a_layout="KM", b_layout="KN")
    manual = values.A.T.float() @ values.B.float()

    assert ref.shape == (2, 3)
    torch.testing.assert_close(ref, manual)


def test_gemm_inputs_require_c_dtype() -> None:
    with pytest.raises(TypeError, match="c_dtype"):
        GemmInputConfig(
            M=2,
            N=3,
            K=4,
            a_dtype=torch.float16,
            b_dtype=torch.float32,
        )

    inputs = GemmInputs(
        GemmInputConfig(
            M=2,
            N=3,
            K=4,
            a_dtype=torch.float16,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
        )
    )
    inputs.config.c_dtype = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="c_dtype is required"):
        inputs.generate(seed=8, device="cpu")


def test_gemm_inputs_reject_invalid_layout() -> None:
    with pytest.raises(ValueError, match="a_layout"):
        GemmInputs(
            GemmInputConfig(
                M=2,
                N=3,
                K=4,
                a_dtype=torch.float16,
                b_dtype=torch.float16,
                c_dtype=torch.float32,
                a_layout="bad",  # type: ignore[arg-type]
            )
        )


def test_gemm_inputs_reject_custom_mxfp4_without_scales() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=CustomDType.MXFP4,
            b_dtype=CustomDType.MXFP4,
            c_dtype=torch.float32,
        )
    )

    with pytest.raises(ValueError, match="mxfp4 tensors require scale_shape"):
        inputs.generate(seed=9, device="cpu")


def test_gemm_inputs_reject_mxfp4_non_kernel_layouts() -> None:
    with pytest.raises(ValueError, match="a_layout='MK'"):
        GemmInputs(
            GemmInputConfig(
                M=4,
                N=8,
                K=64,
                a_dtype=CustomDType.MXFP4,
                b_dtype=CustomDType.MXFP4,
                c_dtype=torch.float32,
                a_layout="KM",
                a_scale_shape=(4, 2),
                b_scale_shape=(8, 2),
            )
        )


def test_gemm_inputs_generate_scaled_operands() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=4,
            N=6,
            K=8,
            a_dtype=_fp8_dtype,
            b_dtype=_fp8_dtype,
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape("channel", "a", M=4, N=6, K=8),
            b_scale_shape=gemm_scale_shape("channel", "b", M=4, N=6, K=8),
        )
    ).generate(seed=17, device="cpu")

    assert inputs.A is not None
    assert inputs.B is not None
    assert inputs.A_scales is not None
    assert inputs.B_scales is not None
    assert inputs.C is not None
    assert inputs.A.shape == (4, 8)
    assert inputs.B.shape == (6, 8)
    assert inputs.A_scales.shape == (4,)
    assert inputs.B_scales.shape == (6,)
    assert inputs.A.dtype == _fp8_dtype
    assert inputs.B.dtype == _fp8_dtype
    assert inputs.C.shape == (4, 6)


def test_gemm_inputs_use_mutable_scale_config_fields() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=4,
            N=6,
            K=8,
            a_dtype=_fp8_dtype,
            b_dtype=_fp8_dtype,
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape("channel", "a", M=4, N=6, K=8),
            b_scale_shape=gemm_scale_shape("channel", "b", M=4, N=6, K=8),
        )
    )

    inputs.config.a_scale_dtype = torch.float64
    inputs.config.c_dtype = torch.float64
    values = inputs.generate(seed=18, device="cpu")

    assert values.A_scales is not None
    assert values.C is not None
    assert values.A_scales.dtype == torch.float64
    assert values.C.dtype == torch.float64


def test_gemm_inputs_accept_scaled_config_objects() -> None:
    config = GemmInputConfig(
        M=2,
        N=3,
        K=4,
        a_dtype=torch.float16,
        b_dtype=torch.float64,
        a_scale_dtype=torch.float32,
        b_scale_dtype=torch.float64,
        c_dtype=torch.float32,
        a_scale_shape=(1,),
        b_scale_shape=(3,),
    )
    inputs = GemmInputs(config)

    values = inputs.generate(seed=20, device="cpu")

    assert inputs.config is config
    assert values.B is not None
    assert values.B_scales is not None
    assert values.B.dtype == torch.float64
    assert values.B_scales.dtype == torch.float64
    assert values.C.shape == (2, 3)
    assert values.C.dtype == torch.float32


def test_gemm_inputs_support_mxfp4_ue8m0_scales() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=CustomDType.MXFP4,
            b_dtype=CustomDType.MXFP4,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=4,
                N=8,
                K=64,
                block_shape=(32,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=4,
                N=8,
                K=64,
                block_shape=(32,),
            ),
        )
    ).generate(seed=19, device="cpu")

    assert inputs.A is not None
    assert inputs.B is not None
    assert inputs.A_scales is not None
    assert inputs.B_scales is not None
    assert inputs.A.shape == (4, 32)
    assert inputs.B.shape == (8, 32)
    assert inputs.A.dtype == torch.uint8
    assert inputs.B.dtype == torch.uint8
    assert inputs.A_scales.shape == (4, 2)
    assert inputs.B_scales.shape == (8, 2)
    assert inputs.A_scales.dtype == torch.uint8
    assert inputs.B_scales.dtype == torch.uint8
    assert inputs.C is not None
    assert inputs.C.shape == (4, 8)


def test_gemm_reference_dequantizes_mxfp4_inputs() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=CustomDType.MXFP4,
            b_dtype=CustomDType.MXFP4,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=4,
                N=8,
                K=64,
                block_shape=(32,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=4,
                N=8,
                K=64,
                block_shape=(32,),
            ),
        )
    ).generate(seed=21, device="cpu")

    ref = gemm_reference(values)

    assert ref.shape == (4, 8)
    assert ref.dtype == torch.float32
    assert torch.isfinite(ref).all()


def test_gemm_inputs_support_nvfp4_fp8_scales() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=CustomDType.NVFP4,
            b_dtype=CustomDType.NVFP4,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=4,
                N=8,
                K=64,
                block_shape=(16,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=4,
                N=8,
                K=64,
                block_shape=(16,),
            ),
        )
    ).generate(seed=96, device="cpu")

    assert inputs.A is not None
    assert inputs.B is not None
    assert inputs.A_scales is not None
    assert inputs.B_scales is not None
    assert inputs.A.shape == (4, 32)
    assert inputs.B.shape == (8, 32)
    assert inputs.A.dtype == torch.uint8
    assert inputs.B.dtype == torch.uint8
    assert inputs.A_scales.shape == (4, 4)
    assert inputs.B_scales.shape == (8, 4)
    assert inputs.A_scales.dtype == torch.float8_e4m3fn
    assert inputs.B_scales.dtype == torch.float8_e4m3fn
    assert inputs.C.shape == (4, 8)


@pytest.mark.parametrize("storage_dtype", [CustomDType.NVFP4, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize(
    "scale_dtype",
    [torch.float8_e4m3fn, torch.float32, torch.uint8],
)
def test_gemm_inputs_support_nvfp4_storage_and_scale_dtypes(
    storage_dtype: object,
    scale_dtype: torch.dtype,
) -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=storage_dtype,  # type: ignore[arg-type]
            b_dtype=storage_dtype,  # type: ignore[arg-type]
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=4,
                N=8,
                K=64,
                block_shape=(16,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=4,
                N=8,
                K=64,
                block_shape=(16,),
            ),
            a_scale_dtype=scale_dtype,
            b_scale_dtype=scale_dtype,
        )
    ).generate(seed=98, device="cpu")

    assert inputs.A is not None
    assert inputs.B is not None
    assert inputs.A_scales is not None
    assert inputs.B_scales is not None
    expected_storage_dtype = (
        torch.uint8 if storage_dtype == CustomDType.NVFP4 else torch.float4_e2m1fn_x2
    )
    assert inputs.A.dtype == expected_storage_dtype
    assert inputs.B.dtype == expected_storage_dtype
    assert inputs.A_scales.dtype == scale_dtype
    assert inputs.B_scales.dtype == scale_dtype

    ref = gemm_reference(inputs, alpha=torch.tensor([1.0], dtype=torch.float32))
    assert ref.shape == (4, 8)
    assert torch.isfinite(ref).all()


def test_gemm_reference_dequantizes_nvfp4_inputs() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=CustomDType.NVFP4,
            b_dtype=CustomDType.NVFP4,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=4,
                N=8,
                K=64,
                block_shape=(16,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=4,
                N=8,
                K=64,
                block_shape=(16,),
            ),
        )
    ).generate(seed=97, device="cpu")

    ref = gemm_reference(values, alpha=torch.tensor([0.5], dtype=torch.float32))

    assert ref.shape == (4, 8)
    assert ref.dtype == torch.float32
    assert torch.isfinite(ref).all()


def test_gemm_inputs_reject_invalid_nvfp4_config() -> None:
    with pytest.raises(ValueError, match="prefix-compatible"):
        GemmInputs(
            GemmInputConfig(
                M=4,
                N=8,
                K=68,
                a_dtype=CustomDType.NVFP4,
                b_dtype=CustomDType.NVFP4,
                c_dtype=torch.float32,
                a_scale_shape=gemm_scale_shape(
                    "block",
                    "a",
                    M=4,
                    N=8,
                    K=68,
                    block_shape=(16,),
                ),
                b_scale_shape=gemm_scale_shape(
                    "block",
                    "b",
                    M=4,
                    N=8,
                    K=68,
                    block_shape=(16,),
                ),
            )
        ).generate(seed=99, device="cpu")

    with pytest.raises(ValueError, match="a_layout='MK'"):
        GemmInputs(
            GemmInputConfig(
                M=4,
                N=8,
                K=64,
                a_dtype=CustomDType.NVFP4,
                b_dtype=CustomDType.NVFP4,
                c_dtype=torch.float32,
                a_layout="KM",
                a_scale_shape=(4, 4),
                b_scale_shape=(8, 4),
            )
        )


def test_gemm_inputs_support_mxint4_bf16_scales() -> None:
    inputs = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=CustomDType.MXINT4,
            b_dtype=CustomDType.MXINT4,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=4,
                N=8,
                K=64,
                block_shape=(32,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=4,
                N=8,
                K=64,
                block_shape=(32,),
            ),
        )
    ).generate(seed=94, device="cpu")

    assert inputs.A is not None
    assert inputs.B is not None
    assert inputs.A_scales is not None
    assert inputs.B_scales is not None
    assert inputs.A.shape == (4, 32)
    assert inputs.B.shape == (8, 32)
    assert inputs.A.dtype == torch.uint8
    assert inputs.B.dtype == torch.uint8
    assert inputs.A_scales.shape == (4, 2)
    assert inputs.B_scales.shape == (8, 2)
    assert inputs.A_scales.dtype == torch.bfloat16
    assert inputs.B_scales.dtype == torch.bfloat16
    assert inputs.C.shape == (4, 8)


def test_gemm_reference_dequantizes_mxint4_inputs() -> None:
    values = GemmInputValues(
        A=torch.ones((1, 4), dtype=torch.float32),
        B=torch.tensor([[0x21, 0xF0]], dtype=torch.uint8),
        C=torch.zeros((1, 1), dtype=torch.float32),
        B_scales=torch.tensor([[1.0]], dtype=torch.bfloat16),
    )

    ref = gemm_reference(values)

    torch.testing.assert_close(ref, torch.tensor([[2.0]], dtype=torch.float32))


def test_gemm_inputs_reject_mxint4_non_kernel_layouts() -> None:
    with pytest.raises(ValueError, match="a_layout='MK'"):
        GemmInputs(
            GemmInputConfig(
                M=4,
                N=8,
                K=64,
                a_dtype=CustomDType.MXINT4,
                b_dtype=CustomDType.MXINT4,
                c_dtype=torch.float32,
                a_layout="KM",
                a_scale_shape=(4, 2),
                b_scale_shape=(8, 2),
            )
        )


def test_gemm_reference_applies_scaled_operands() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=3,
            N=5,
            K=7,
            a_dtype=torch.float32,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
            a_scale_shape=(3,),
            b_scale_shape=(5,),
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
        )
    ).generate(seed=22, device="cpu")
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is not None
    assert values.B_scales is not None

    ref = gemm_reference(values)
    manual = (values.A.float() * values.A_scales.float().view(3, 1)) @ (
        values.B.float() * values.B_scales.float().view(5, 1)
    ).T

    torch.testing.assert_close(ref, manual)


def test_gemm_reference_applies_channel_scales_to_transposed_layouts() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=3,
            N=5,
            K=7,
            a_dtype=torch.float32,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
            a_layout="KM",
            b_layout="KN",
            a_scale_shape=(3,),
            b_scale_shape=(5,),
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
        )
    ).generate(seed=98, device="cpu")
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is not None
    assert values.B_scales is not None

    ref = gemm_reference(values, a_layout="KM", b_layout="KN")
    manual = (values.A.float() * values.A_scales.float().view(1, 3)).T @ (
        values.B.float() * values.B_scales.float().view(1, 5)
    )

    torch.testing.assert_close(ref, manual)


def test_gemm_reference_applies_2d_block_scales() -> None:
    block_shape = (128, 128)
    values = GemmInputs(
        GemmInputConfig(
            M=4,
            N=256,
            K=256,
            a_dtype=torch.float8_e4m3fn,
            b_dtype=torch.float8_e4m3fn,
            c_dtype=torch.float32,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=4,
                N=256,
                K=256,
                block_shape=block_shape,
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=4,
                N=256,
                K=256,
                block_shape=block_shape,
            ),
            a_scale_dtype=torch.float32,
            b_scale_dtype=torch.float32,
        )
    ).generate(seed=93, device="cpu")
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is not None
    assert values.B_scales is not None
    assert values.A_scales.shape == (4, 2)
    assert values.B_scales.shape == (2, 2)

    a_scales = values.A_scales.repeat_interleave(block_shape[1], dim=-1)
    b_scales = values.B_scales.repeat_interleave(
        block_shape[0],
        dim=-2,
    ).repeat_interleave(block_shape[1], dim=-1)
    expected = (values.A.float() * a_scales) @ (values.B.float() * b_scales).transpose(
        -1, -2
    )

    torch.testing.assert_close(gemm_reference(values), expected)


def test_gemm_inputs_cover_router_projection_shape() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=5,
            N=7,
            K=64,
            a_dtype=torch.bfloat16,
            b_dtype=torch.float32,
            c_dtype=torch.float32,
        )
    ).generate(seed=95, device="cpu")
    assert values.A is not None
    assert values.B is not None
    values.B = (values.B.float() / math.sqrt(64)).contiguous()

    logits = gemm_reference(values, out_dtype=torch.float32)

    assert values.A.shape == (5, 64)
    assert values.B.shape == (7, 64)
    assert logits.shape == (5, 7)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert logits.float().std() > 0.0


def test_gemm_inputs_cover_lm_head_projection_shape() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=4,
            N=11,
            K=64,
            a_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            c_dtype=torch.bfloat16,
        )
    ).generate(seed=96, device="cpu")
    assert values.A is not None
    assert values.B is not None
    values.B = (values.B.float() / math.sqrt(64)).to(torch.bfloat16).contiguous()

    logits = gemm_reference(values)

    assert values.A.shape == (4, 64)
    assert values.B.shape == (11, 64)
    assert logits.shape == (4, 11)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits.float()).all()
    assert logits.float().std() > 0.0


def test_composes_nvfp4_gemm_with_swiglu_quant_reference() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=3,
            N=64,
            K=64,
            a_dtype=CustomDType.NVFP4,
            b_dtype=CustomDType.NVFP4,
            c_dtype=torch.bfloat16,
            a_scale_shape=gemm_scale_shape(
                "block",
                "a",
                M=3,
                N=64,
                K=64,
                block_shape=(16,),
            ),
            b_scale_shape=gemm_scale_shape(
                "block",
                "b",
                M=3,
                N=64,
                K=64,
                block_shape=(16,),
            ),
        )
    ).generate(seed=23, device="cpu")
    assert values.A is not None
    assert values.B is not None
    assert values.A_scales is not None
    assert values.B_scales is not None
    output_global_scale = torch.tensor([0.01], dtype=torch.float32)

    gate_up = gemm_reference(
        values,
        alpha=torch.tensor([1.0e-3], dtype=torch.float32),
        out_dtype=torch.bfloat16,
    )
    gate, up = gate_up.float().chunk(2, dim=-1)
    swiglu = torch.nn.functional.silu(gate) * up
    packed, scales = nvfp4_quantization_reference(
        swiglu,
        scale=output_global_scale,
        scale_size=16,
    )

    assert gate_up.shape == (3, 64)
    assert packed.shape == (3, 16)
    assert scales.shape == (3, 2)
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.float8_e4m3fn
    dequant = nvfp4_dequantization_reference(
        packed,
        scales,
        scale=output_global_scale,
    )
    assert dequant.shape == (3, 32)
    assert torch.isfinite(dequant).all()


def test_moe_inputs_compose_dense_weight_gemms() -> None:
    inputs = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=16,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
        )
    ).generate(seed=23, device="cpu")

    assert inputs.hidden_states is not None
    assert inputs.router_logits is not None
    assert inputs.topk_ids is not None
    assert isinstance(inputs, MoeInputValues)
    assert isinstance(inputs.w13, GemmInputValues)
    assert isinstance(inputs.w2, GemmInputValues)
    assert inputs.hidden_states.shape == (5, 16)
    assert inputs.router_logits.shape == (5, 4)
    assert inputs.topk_ids.shape == (5, 2)
    assert inputs.topk_weights.shape == (5, 2)
    assert torch.all(inputs.topk_ids >= 0)
    assert torch.all(inputs.topk_ids < 4)
    torch.testing.assert_close(
        inputs.topk_weights.float().sum(dim=-1),
        torch.ones(5),
        atol=1.0e-3,
        rtol=1.0e-3,
    )
    scores = torch.softmax(inputs.router_logits.float(), dim=-1)
    expected_weights, expected_ids = torch.topk(scores, k=2, dim=-1, sorted=False)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(inputs.topk_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(
        inputs.topk_weights,
        expected_weights.to(inputs.topk_weights.dtype),
        atol=0,
        rtol=0,
    )
    assert inputs.w13.A is None
    assert inputs.w2.A is None
    assert inputs.w13.B is not None
    assert inputs.w2.B is not None
    assert inputs.w13.C is not None
    assert inputs.w2.C is not None
    assert inputs.w13.B.shape == (4, 64, 16)
    assert inputs.w2.B.shape == (4, 16, 32)
    assert inputs.w13.C.shape == (4, 5, 64)
    assert inputs.w2.C.shape == (4, 5, 16)
    assert inputs.w13.C.dtype == torch.float16
    assert inputs.w2.C.dtype == torch.float16
    assert inputs.w13_bias is None
    assert inputs.w2_bias is None


def test_moe_inputs_reuse_mutable_child_generators() -> None:
    inputs = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=16,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
        )
    )

    assert inputs.hidden_states_input is not None
    assert isinstance(inputs.w13, GemmInputs)
    hidden_states_input = inputs.hidden_states_input
    w13 = inputs.w13
    hidden_states_input.dtype = torch.float32
    inputs.w13.config.b_dtype = torch.float64
    values = inputs.generate(seed=24, device="cpu")

    assert inputs.hidden_states_input is hidden_states_input
    assert inputs.w13 is w13
    assert values.hidden_states is not None
    assert values.w13.B is not None
    assert values.hidden_states.dtype == torch.float32
    assert values.w13.B.dtype == torch.float64


def test_moe_inputs_accept_config_objects() -> None:
    config = MoeInputConfig(
        num_tokens=5,
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
        hidden_dtype=torch.float16,
        w13=GemmInputConfig(
            M=5,
            N=64,
            K=16,
            a_dtype=None,
            b_dtype=torch.float64,
            c_dtype=torch.float32,
            batch_shape=(4,),
        ),
    )
    inputs = MoeInputs(config)

    values = inputs.generate(seed=25, device="cpu")

    assert inputs.config is config
    assert isinstance(inputs.w13, GemmInputs)
    assert inputs.w13.config is config.w13
    assert isinstance(values.w13, GemmInputValues)
    assert values.w13.B is not None
    assert values.w13.B.dtype == torch.float64
    assert values.w13.C.shape == (4, 5, 64)
    assert values.w13.C.dtype == torch.float32


def test_moe_inputs_compose_mxfp4_weight_gemms() -> None:
    inputs = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
            weight_format="mxfp4",
        )
    ).generate(seed=29, device="cpu")

    assert isinstance(inputs.w13, GemmInputValues)
    assert isinstance(inputs.w2, GemmInputValues)
    assert inputs.w13.A is None
    assert inputs.w13.B is not None
    assert inputs.w2.A is None
    assert inputs.w2.B is not None
    assert inputs.w13.A_scales is None
    assert inputs.w2.A_scales is None
    assert inputs.w13.B_scales is not None
    assert inputs.w2.B_scales is not None
    assert inputs.w13.B.shape == (4, 64, 32)
    assert inputs.w13.B_scales.shape == (4, 64, 2)
    assert inputs.w2.B.shape == (4, 64, 16)
    assert inputs.w2.B_scales.shape == (4, 64, 1)
    assert inputs.w13.B.dtype == torch.uint8
    assert inputs.w13.B_scales.dtype == torch.uint8


def test_moe_inputs_compose_mxint4_weight_gemms() -> None:
    inputs = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
            weight_format="mxint4",
        )
    ).generate(seed=34, device="cpu")

    assert inputs.w13.A is None
    assert inputs.w2.A is None
    assert inputs.w13.B is not None
    assert inputs.w2.B is not None
    assert inputs.w13.B_scales is not None
    assert inputs.w2.B_scales is not None
    assert inputs.w13.B.shape == (4, 64, 32)
    assert inputs.w13.B_scales.shape == (4, 64, 2)
    assert inputs.w2.B.shape == (4, 64, 16)
    assert inputs.w2.B_scales.shape == (4, 64, 1)
    assert inputs.w13.B.dtype == torch.uint8
    assert inputs.w2.B.dtype == torch.uint8
    assert inputs.w13.B_scales.dtype == torch.bfloat16
    assert inputs.w2.B_scales.dtype == torch.bfloat16


def test_moe_inputs_generate_optional_activation_scales() -> None:
    inputs = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
            weight_format="mxfp4",
            activation_scale_dtype=torch.float32,
            w13_activation_scale=0.25,
            w2_activation_scale=0.5,
        )
    ).generate(seed=31, device="cpu")

    assert inputs.w13_activation_scale is not None
    assert inputs.w2_activation_scale is not None
    assert inputs.w13_activation_scale.shape == (4,)
    assert inputs.w2_activation_scale.shape == (4,)
    assert inputs.w13_activation_scale.dtype == torch.float32
    assert inputs.w2_activation_scale.dtype == torch.float32
    torch.testing.assert_close(
        inputs.w13_activation_scale,
        torch.full((4,), 0.25, dtype=torch.float32),
    )
    torch.testing.assert_close(
        inputs.w2_activation_scale,
        torch.full((4,), 0.5, dtype=torch.float32),
    )


def test_moe_inputs_verify_topk_config() -> None:
    with pytest.raises(ValueError, match="top_k must be <= num_experts"):
        MoeInputs(
            MoeInputConfig(
                num_tokens=4,
                hidden_size=8,
                intermediate_size=12,
                num_experts=3,
                top_k=4,
                hidden_dtype=torch.float32,
            )
        )


def test_moe_inputs_verify_activation_scale_config() -> None:
    with pytest.raises(ValueError, match="activation_scale_dtype"):
        MoeInputs(
            MoeInputConfig(
                num_tokens=4,
                hidden_size=8,
                intermediate_size=12,
                num_experts=3,
                top_k=2,
                hidden_dtype=torch.float32,
                activation_scale_dtype=torch.int32,
            )
        )
    with pytest.raises(ValueError, match="w13_activation_scale"):
        MoeInputs(
            MoeInputConfig(
                num_tokens=4,
                hidden_size=8,
                intermediate_size=12,
                num_experts=3,
                top_k=2,
                hidden_dtype=torch.float32,
                activation_scale_dtype=torch.float32,
                w13_activation_scale=0.0,
            )
        )


def test_moe_inputs_verify_mxint4_weight_config() -> None:
    with pytest.raises(ValueError, match="mxint4 MoE weights"):
        MoeInputs(
            MoeInputConfig(
                num_tokens=4,
                hidden_size=64,
                intermediate_size=32,
                num_experts=3,
                top_k=2,
                hidden_dtype=torch.float32,
                weight_format="mxint4",
                weight_dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="mxint4 MoE weight_scale_dtype"):
        MoeInputs(
            MoeInputConfig(
                num_tokens=4,
                hidden_size=64,
                intermediate_size=32,
                num_experts=3,
                top_k=2,
                hidden_dtype=torch.float32,
                weight_format="mxint4",
                weight_scale_dtype=torch.float32,
            )
        )


def test_moe_inputs_verify_child_gemm_shapes() -> None:
    with pytest.raises(ValueError, match="w13 GEMM config must use M/N/K"):
        MoeInputs(
            MoeInputConfig(
                num_tokens=4,
                hidden_size=8,
                intermediate_size=12,
                num_experts=3,
                top_k=2,
                hidden_dtype=torch.float32,
                w13=GemmInputConfig(
                    M=4,
                    N=25,
                    K=8,
                    a_dtype=None,
                    b_dtype=torch.float32,
                    c_dtype=torch.float32,
                    batch_shape=(3,),
                ),
            )
        )


def test_moe_softmax_topk_routing_inputs_generate_values() -> None:
    values = MoESoftmaxTopKRoutingInputs(
        MoESoftmaxTopKRoutingInputConfig(
            num_tokens=3,
            num_experts=8,
            num_experts_real=6,
            top_k=4,
            scaling_factor=2.0,
        )
    ).generate(seed=51, device="cpu")

    assert values.logits.shape == (3, 8)
    assert values.logits.dtype == torch.float32
    assert values.correction_bias.shape == (8,)
    assert values.topk_indices.shape == (3, 4)
    assert values.topk_indices.dtype == torch.int32
    assert values.topk_weights.shape == (3, 4)
    assert values.topk_weights.dtype == torch.float32


def test_moe_softmax_topk_routing_verifies_config() -> None:
    with pytest.raises(ValueError, match="num_experts_real must be < num_experts"):
        MoESoftmaxTopKRoutingInputs(
            MoESoftmaxTopKRoutingInputConfig(
                num_tokens=3,
                num_experts=8,
                num_experts_real=8,
                top_k=2,
            )
        )

    with pytest.raises(ValueError, match="topk_indices_dtype"):
        MoESoftmaxTopKRoutingInputs(
            MoESoftmaxTopKRoutingInputConfig(
                num_tokens=3,
                num_experts=8,
                num_experts_real=6,
                top_k=2,
                topk_indices_dtype=torch.float32,
            )
        )


def test_moe_biased_grouped_topk_inputs_generate_values() -> None:
    values = MoEBiasedGroupedTopKInputs(
        MoEBiasedGroupedTopKInputConfig(
            num_tokens=4,
            hidden_size=6,
            num_experts=8,
            top_k=3,
            num_expert_groups=2,
            top_k_groups=1,
            renormalize=True,
            routed_scaling_factor=2.5,
            use_logical_to_physical_map=True,
            num_token_non_padded=3,
        )
    ).generate(seed=52, device="cpu")

    assert values.hidden_states.shape == (4, 6)
    assert values.gating_output.shape == (4, 8)
    assert values.correction_bias.shape == (8,)
    assert values.logical_to_physical_map is not None
    torch.testing.assert_close(
        values.logical_to_physical_map.sort().values,
        torch.arange(8, dtype=torch.int32),
    )
    assert values.num_token_non_padded is not None
    assert int(values.num_token_non_padded.item()) == 3


def test_moe_biased_grouped_topk_verifies_config() -> None:
    with pytest.raises(ValueError, match="divisible by num_expert_groups"):
        MoEBiasedGroupedTopKInputs(
            MoEBiasedGroupedTopKInputConfig(
                num_tokens=3,
                hidden_size=4,
                num_experts=7,
                top_k=2,
                num_expert_groups=2,
            )
        )

    with pytest.raises(ValueError, match="selected groups"):
        MoEBiasedGroupedTopKInputs(
            MoEBiasedGroupedTopKInputConfig(
                num_tokens=3,
                hidden_size=4,
                num_experts=8,
                top_k=5,
                num_expert_groups=2,
                top_k_groups=1,
            )
        )


def test_moe_softplus_sqrt_topk_routing_inputs_generate_non_hash_values() -> None:
    values = MoESoftplusSqrtTopKRoutingInputs(
        MoESoftplusSqrtTopKRoutingInputConfig(
            num_tokens=3,
            num_experts=8,
            top_k=4,
            routed_scaling_factor=2.0,
        )
    ).generate(seed=54, device="cpu")

    assert values.logits.shape == (3, 8)
    assert values.logits.dtype == torch.float32
    assert values.correction_bias is not None
    assert values.correction_bias.shape == (8,)
    assert values.input_ids is None
    assert values.hash_indices_table is None
    assert values.topk_indices.shape == (3, 4)
    assert values.topk_indices.dtype == torch.int32
    assert values.topk_weights.shape == (3, 4)


def test_moe_softplus_sqrt_topk_routing_verifies_config() -> None:
    with pytest.raises(ValueError, match="renormalize=True"):
        MoESoftplusSqrtTopKRoutingInputs(
            MoESoftplusSqrtTopKRoutingInputConfig(
                num_tokens=2,
                num_experts=8,
                renormalize=False,
            )
        )


def test_moe_deepseek_v4_mega_moe_staging_inputs_generate_values() -> None:
    values = MoEDeepSeekV4MegaMoEStagingInputs(
        MoEDeepSeekV4MegaMoEStagingInputConfig(
            num_tokens=3,
            hidden_size=256,
            num_experts=8,
            top_k=3,
            hidden_dtype=torch.bfloat16,
        )
    ).generate(seed=55, device="cpu")

    assert values.hidden_states.shape == (3, 256)
    assert values.hidden_states.dtype == torch.bfloat16
    assert values.topk_ids.shape == (3, 3)
    assert values.topk_ids.dtype == torch.int32
    assert torch.all(values.topk_ids >= 0)
    assert torch.all(values.topk_ids < 8)
    assert values.topk_weights.shape == (3, 3)
    torch.testing.assert_close(
        values.topk_weights.sum(dim=-1),
        torch.ones(3),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert values.x_fp8.shape == (3, 256)
    assert values.x_fp8.dtype == torch.float8_e4m3fn
    assert values.x_sf.shape == (3, 2)
    assert values.x_sf.dtype == torch.int32


def test_moe_deepseek_v4_mega_moe_staging_verifies_config() -> None:
    with pytest.raises(ValueError, match="multiple of 128"):
        MoEDeepSeekV4MegaMoEStagingInputs(
            MoEDeepSeekV4MegaMoEStagingInputConfig(
                num_tokens=2,
                hidden_size=192,
                num_experts=4,
                top_k=2,
            )
        )


def test_moe_finalize_fuse_shared_inputs_generate_values() -> None:
    values = MoEFinalizeFuseSharedInputs(
        MoEFinalizeFuseSharedInputConfig(
            num_tokens=3,
            hidden_size=8,
            top_k=2,
            hidden_size_padded=12,
            total_num_padded_tokens=8,
            num_dropped_slots=1,
            include_shared_output=True,
            expert_weights_dtype=torch.float32,
        )
    ).generate(seed=57, device="cpu")

    assert values.gemm2_out.shape == (8, 12)
    assert values.gemm2_out.dtype == torch.bfloat16
    assert values.expanded_idx_to_permuted_idx.shape == (6,)
    assert values.expanded_idx_to_permuted_idx.dtype == torch.int32
    assert int((values.expanded_idx_to_permuted_idx == -1).sum().item()) == 1
    assert values.expert_weights.shape == (3, 2)
    assert values.expert_weights.dtype == torch.float32
    torch.testing.assert_close(
        values.expert_weights.sum(dim=-1),
        torch.ones(3),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert values.shared_output is not None
    assert values.shared_output.shape == (3, 8)
    assert values.shared_output.dtype == torch.bfloat16


def test_moe_finalize_fuse_shared_inputs_support_no_shared_output() -> None:
    values = MoEFinalizeFuseSharedInputs(
        MoEFinalizeFuseSharedInputConfig(
            num_tokens=2,
            hidden_size=8,
            top_k=2,
            include_shared_output=False,
            expert_weights_dtype=torch.bfloat16,
        )
    ).generate(seed=58, device="cpu")

    assert values.shared_output is None
    assert values.gemm2_out.shape == (4, 8)
    assert values.expert_weights.dtype == torch.bfloat16


def test_moe_finalize_fuse_shared_verifies_config() -> None:
    with pytest.raises(ValueError, match="top_k"):
        MoEFinalizeFuseSharedInputs(
            MoEFinalizeFuseSharedInputConfig(
                num_tokens=2,
                hidden_size=8,
                top_k=65,
            )
        )
    with pytest.raises(ValueError, match="total_num_padded_tokens"):
        MoEFinalizeFuseSharedInputs(
            MoEFinalizeFuseSharedInputConfig(
                num_tokens=2,
                hidden_size=8,
                top_k=2,
                total_num_padded_tokens=3,
            )
        )


def test_moe_align_block_size_inputs_generate_topk_ids() -> None:
    first = MoeAlignBlockSizeInputs(
        MoeAlignBlockSizeInputConfig(
            total_tokens=6,
            top_k=2,
            num_experts=4,
            block_size=8,
        )
    ).generate(seed=31, device="cpu")
    second = MoeAlignBlockSizeInputs(
        MoeAlignBlockSizeInputConfig(
            total_tokens=6,
            top_k=2,
            num_experts=4,
            block_size=8,
        )
    ).generate(seed=31, device="cpu")

    assert isinstance(first, MoeAlignBlockSizeInputValues)
    assert first.topk_ids.shape == (6, 2)
    assert first.topk_ids.dtype == torch.int32
    assert torch.all(first.topk_ids >= 0)
    assert torch.all(first.topk_ids < 4)
    assert torch.equal(first.topk_ids, second.topk_ids)
    assert first.block_size == 8
    assert first.num_experts == 4


def test_moe_align_block_size_inputs_verify_config() -> None:
    with pytest.raises(ValueError, match="top_k must be <= num_experts"):
        MoeAlignBlockSizeInputs(
            MoeAlignBlockSizeInputConfig(
                total_tokens=4,
                top_k=5,
                num_experts=4,
                block_size=8,
            )
        )

    with pytest.raises(ValueError, match="topk_ids_dtype"):
        MoeAlignBlockSizeInputs(
            MoeAlignBlockSizeInputConfig(
                total_tokens=4,
                top_k=2,
                num_experts=4,
                block_size=8,
                topk_ids_dtype=torch.float32,
            )
        )
