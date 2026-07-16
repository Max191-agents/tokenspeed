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

import inspect
import math

import pytest
import tokenspeed_numerics_input_generators.moe as moe_generators
import torch
from tokenspeed_numerics_input_generators import (
    CustomDType,
    GemmInputConfig,
    GemmInputs,
    GemmInputValues,
    MoeDispatchInputConfig,
    MoeDispatchInputs,
    MoeDispatchInputValues,
    MoeDownCombineInputConfig,
    MoeDownCombineInputs,
    MoeDownCombineInputValues,
    MoeGateUpInputConfig,
    MoeGateUpInputs,
    MoeGateUpInputValues,
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
    MoeRoutingInputConfig,
    MoeRoutingInputs,
    MoeRoutingInputValues,
    TensorInput,
    gemm_reference,
    gemm_scale_shape,
    nvfp4_dequantization_reference,
    nvfp4_quantization_reference,
)

_fp8_dtype = torch.float8_e4m3fn


def test_moe_module_defines_only_fundamental_generators() -> None:
    generator_names = {
        name
        for name, value in vars(moe_generators).items()
        if name.endswith("Inputs")
        and inspect.isclass(value)
        and value.__module__ == moe_generators.__name__
    }

    assert generator_names == {
        "MoeDispatchInputs",
        "MoeDownCombineInputs",
        "MoeGateUpInputs",
        "MoeInputs",
        "MoeRoutingInputs",
    }


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


def test_moe_routing_inputs_generate_only_router_operands() -> None:
    config = MoeRoutingInputConfig(
        num_tokens=5,
        hidden_size=16,
        num_experts=4,
        hidden_dtype=torch.float16,
        router_dtype=torch.float32,
        router_bias_dtype=torch.float32,
        correction_bias_dtype=torch.float32,
    )
    first = MoeRoutingInputs(config).generate(seed=23, device="cpu")
    second = MoeRoutingInputs(config).generate(seed=23, device="cpu")

    assert isinstance(first, MoeRoutingInputValues)
    assert set(vars(first)) == {
        "hidden_states",
        "router_weight",
        "router_bias",
        "correction_bias",
    }
    assert first.hidden_states.shape == (5, 16)
    assert first.router_weight.shape == (4, 16)
    assert first.router_bias is not None
    assert first.router_bias.shape == (4,)
    assert first.correction_bias is not None
    assert first.correction_bias.shape == (4,)
    assert torch.equal(first.hidden_states, second.hidden_states)
    assert torch.equal(first.router_weight, second.router_weight)
    assert torch.equal(first.router_bias, second.router_bias)
    assert torch.equal(first.correction_bias, second.correction_bias)

    router_logits = (
        first.hidden_states.float() @ first.router_weight.float().transpose(0, 1)
        + first.router_bias.float()
    )
    assert router_logits.shape == (5, 4)
    assert torch.isfinite(router_logits).all()


def test_moe_routing_inputs_verify_shape_and_dtype_config() -> None:
    with pytest.raises(ValueError, match="num_experts must be positive"):
        MoeRoutingInputs(
            MoeRoutingInputConfig(
                num_tokens=3,
                hidden_size=8,
                num_experts=0,
                hidden_dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="router_dtype"):
        MoeRoutingInputs(
            MoeRoutingInputConfig(
                num_tokens=3,
                hidden_size=8,
                num_experts=4,
                hidden_dtype=torch.float32,
                router_dtype=torch.int32,
            )
        )


def test_moe_dispatch_inputs_generate_deterministic_metadata() -> None:
    config = MoeDispatchInputConfig(
        num_tokens=6,
        hidden_size=12,
        num_experts=4,
        top_k=2,
        hidden_dtype=torch.bfloat16,
    )
    first = MoeDispatchInputs(config).generate(seed=31, device="cpu")
    second = MoeDispatchInputs(config).generate(seed=31, device="cpu")

    assert isinstance(first, MoeDispatchInputValues)
    assert first.hidden_states.shape == (6, 12)
    assert first.topk_ids.shape == (6, 2)
    assert first.topk_weights.shape == (6, 2)
    assert torch.equal(first.hidden_states, second.hidden_states)
    assert torch.equal(first.topk_ids, second.topk_ids)
    assert torch.equal(first.topk_weights, second.topk_weights)
    assert torch.all(first.topk_ids >= 0)
    assert torch.all(first.topk_ids < 4)
    torch.testing.assert_close(first.topk_weights.sum(dim=-1), torch.ones(6))


@pytest.mark.parametrize(
    ("weight_dtype", "scale_dtype"),
    [
        (torch.float16, None),
        (CustomDType.MXFP4, None),
        (CustomDType.MXINT4, None),
    ],
)
def test_moe_gate_up_inputs_generate_projection_operands(
    weight_dtype: torch.dtype | CustomDType,
    scale_dtype: torch.dtype | None,
) -> None:
    values = MoeGateUpInputs(
        MoeGateUpInputConfig(
            num_routed_tokens=10,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            activation_dtype=torch.float16,
            weight_dtype=weight_dtype,
            weight_scale_dtype=scale_dtype,
            bias_dtype=torch.float32,
            activation_scale_dtype=torch.float32,
            activation_scale=0.25,
        )
    ).generate(seed=32, device="cpu")

    assert isinstance(values, MoeGateUpInputValues)
    assert values.routed_hidden_states.shape == (10, 64)
    assert values.expert_ids.shape == (10,)
    assert values.w13.B is not None
    assert values.w13.B.shape == (
        (4, 64, 64) if weight_dtype == torch.float16 else (4, 64, 32)
    )
    assert values.w13_bias is not None
    assert values.w13_bias.shape == (4, 64)
    assert values.activation_scale is not None
    torch.testing.assert_close(
        values.activation_scale,
        torch.full((4,), 0.25, dtype=torch.float32),
    )
    if weight_dtype == torch.float16:
        assert values.w13.B_scales is None
    else:
        assert values.w13.B_scales is not None
        assert values.w13.B_scales.shape == (4, 64, 2)


def test_moe_down_combine_inputs_generate_projection_and_reduction_inputs() -> None:
    values = MoeDownCombineInputs(
        MoeDownCombineInputConfig(
            num_tokens=5,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            activation_dtype=torch.bfloat16,
            weight_dtype=CustomDType.MXFP4,
            bias_dtype=torch.float32,
            activation_scale_dtype=torch.float32,
            include_shared_output=True,
        )
    ).generate(seed=33, device="cpu")

    assert isinstance(values, MoeDownCombineInputValues)
    assert values.activations.shape == (10, 32)
    assert values.topk_ids.shape == (5, 2)
    assert values.topk_weights.shape == (5, 2)
    assert values.w2.B is not None
    assert values.w2.B.shape == (4, 64, 16)
    assert values.w2.B_scales is not None
    assert values.w2.B_scales.shape == (4, 64, 1)
    assert values.w2_bias is not None
    assert values.w2_bias.shape == (4, 64)
    assert values.activation_scale is not None
    assert values.shared_output is not None
    assert values.shared_output.shape == (5, 64)


def test_moe_inputs_compose_full_fused_operands() -> None:
    values = MoeInputs(
        MoeInputConfig(
            routing=MoeRoutingInputConfig(
                num_tokens=5,
                hidden_size=16,
                num_experts=4,
                hidden_dtype=torch.float16,
                router_bias_dtype=torch.float32,
            ),
            intermediate_size=32,
            bias_dtype=torch.float32,
        )
    ).generate(seed=34, device="cpu")

    assert isinstance(values, MoeInputValues)
    assert isinstance(values.routing, MoeRoutingInputValues)
    assert set(vars(values.routing)) == {
        "hidden_states",
        "router_weight",
        "router_bias",
        "correction_bias",
    }
    assert values.routing.hidden_states.shape == (5, 16)
    assert values.routing.router_weight.shape == (4, 16)
    assert isinstance(values.w13, GemmInputValues)
    assert isinstance(values.w2, GemmInputValues)
    assert values.w13.A is None
    assert values.w2.A is None
    assert values.w13.B is not None
    assert values.w2.B is not None
    assert values.w13.B.shape == (4, 64, 16)
    assert values.w2.B.shape == (4, 16, 32)
    assert values.w13_bias is not None
    assert values.w2_bias is not None


@pytest.mark.parametrize(
    ("weight_dtype", "expected_scale_dtype"),
    [
        (CustomDType.MXFP4, torch.uint8),
        (CustomDType.MXINT4, torch.bfloat16),
    ],
)
def test_moe_inputs_infer_quantized_weight_scales(
    weight_dtype: CustomDType,
    expected_scale_dtype: torch.dtype,
) -> None:
    values = MoeInputs(
        MoeInputConfig(
            routing=MoeRoutingInputConfig(
                num_tokens=5,
                hidden_size=64,
                num_experts=4,
                hidden_dtype=torch.float16,
            ),
            intermediate_size=32,
            weight_dtype=weight_dtype,
        )
    ).generate(seed=35, device="cpu")

    assert values.w13.B is not None
    assert values.w2.B is not None
    assert values.w13.B_scales is not None
    assert values.w2.B_scales is not None
    assert values.w13.B.shape == (4, 64, 32)
    assert values.w2.B.shape == (4, 64, 16)
    assert values.w13.B_scales.shape == (4, 64, 2)
    assert values.w2.B_scales.shape == (4, 64, 1)
    assert values.w13.B_scales.dtype == expected_scale_dtype
    assert values.w2.B_scales.dtype == expected_scale_dtype


def test_moe_inputs_reuse_mutable_children() -> None:
    generator = MoeInputs(
        MoeInputConfig(
            routing=MoeRoutingInputConfig(
                num_tokens=4,
                hidden_size=16,
                num_experts=4,
                hidden_dtype=torch.float16,
            ),
            intermediate_size=32,
        )
    )
    assert generator.routing is not None
    assert generator.routing.hidden_states_input is not None
    assert generator.w13 is not None

    generator.routing.hidden_states_input.dtype = torch.float32
    generator.w13.config.b_dtype = torch.float64
    values = generator.generate(seed=36, device="cpu")

    assert values.routing.hidden_states.dtype == torch.float32
    assert values.w13.B is not None
    assert values.w13.B.dtype == torch.float64


def test_moe_inputs_verify_projection_config() -> None:
    with pytest.raises(ValueError, match="intermediate_size must be positive"):
        MoeInputs(
            MoeInputConfig(
                routing=MoeRoutingInputConfig(
                    num_tokens=4,
                    hidden_size=64,
                    num_experts=4,
                    hidden_dtype=torch.float32,
                ),
                intermediate_size=0,
            )
        )

    with pytest.raises(ValueError, match="divisible by the weight scale block size"):
        MoeInputs(
            MoeInputConfig(
                routing=MoeRoutingInputConfig(
                    num_tokens=4,
                    hidden_size=48,
                    num_experts=4,
                    hidden_dtype=torch.float32,
                ),
                intermediate_size=32,
                weight_dtype=CustomDType.MXFP4,
            )
        )

    with pytest.raises(ValueError, match="activation_scale_dtype"):
        MoeInputs(
            MoeInputConfig(
                routing=MoeRoutingInputConfig(
                    num_tokens=4,
                    hidden_size=64,
                    num_experts=4,
                    hidden_dtype=torch.float32,
                ),
                intermediate_size=32,
                activation_scale_dtype=torch.int32,
            )
        )


def test_moe_stage_generation_rejects_mutated_child_shape() -> None:
    generator = MoeDispatchInputs(
        MoeDispatchInputConfig(
            num_tokens=4,
            hidden_size=8,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float32,
        )
    )
    assert generator.hidden_states_input is not None
    generator.hidden_states_input.shape = (4, 7)

    with pytest.raises(ValueError, match="hidden_states shape"):
        generator.generate(seed=37, device="cpu")
