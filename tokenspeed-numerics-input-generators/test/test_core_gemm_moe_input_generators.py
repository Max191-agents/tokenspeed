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
    CustomDType,
    GemmInputConfig,
    GemmInputs,
    GemmInputValues,
    LMHeadProjectionInputConfig,
    LMHeadProjectionInputs,
    LMHeadProjectionInputValues,
    MoeAlignBlockSizeInputConfig,
    MoeAlignBlockSizeInputs,
    MoeAlignBlockSizeInputValues,
    MoEBiasedGroupedTopKInputConfig,
    MoEBiasedGroupedTopKInputs,
    MoEBiasedGroupedTopKInputValues,
    MoEDeepSeekV4MegaMoEStagingInputConfig,
    MoEDeepSeekV4MegaMoEStagingInputs,
    MoEDeepSeekV4MegaMoEStagingInputValues,
    MoEFinalizeFuseSharedInputConfig,
    MoEFinalizeFuseSharedInputs,
    MoEFinalizeFuseSharedInputValues,
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
    MoESoftmaxTopKRoutingInputConfig,
    MoESoftmaxTopKRoutingInputs,
    MoESoftmaxTopKRoutingInputValues,
    MoESoftplusSqrtTopKRoutingInputConfig,
    MoESoftplusSqrtTopKRoutingInputs,
    MoESoftplusSqrtTopKRoutingInputValues,
    NVFP4GemmSwiGLUNVFP4QuantInputConfig,
    NVFP4GemmSwiGLUNVFP4QuantInputs,
    RouterProjectionInputConfig,
    RouterProjectionInputs,
    RouterProjectionInputValues,
    TensorInput,
    canonicalize_moe_align_block_size,
    gemm_reference,
    gemm_scale_shape,
    lm_head_projection_reference,
    moe_align_block_size_buffer_dims,
    moe_align_block_size_reference,
    moe_biased_grouped_topk_reference,
    moe_deepseek_v4_mega_moe_staging_reference,
    moe_finalize_fuse_shared_reference,
    moe_reference,
    moe_softmax_topk_routing_reference,
    moe_softplus_sqrt_topk_routing_reference,
    mxfp4_gemm_input_config,
    mxfp8_gemm_input_config,
    mxint4_gemm_input_config,
    nvfp4_dequantization_reference,
    nvfp4_gemm_input_config,
    nvfp4_gemm_swiglu_nvfp4_quant_reference,
    router_projection_reference,
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
            scale_dtype=torch.float32,
        )

    values = TensorInput(
        (4, 8),
        CustomDType.NVFP4,
        scale_shape=(4, 1),
        scale_dtype=torch.float8_e4m3fn,
    ).generate(seed=131, device="cpu")

    assert values.scales is not None
    assert values.scales.dtype == torch.float8_e4m3fn


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
            mxfp4_gemm_input_config(
                M=4,
                N=8,
                K=64,
                c_dtype=torch.float32,
                a_layout="KM",
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
        mxfp4_gemm_input_config(
            M=4,
            N=8,
            K=64,
            c_dtype=torch.float32,
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
        mxfp4_gemm_input_config(
            M=4,
            N=8,
            K=64,
            c_dtype=torch.float32,
        )
    ).generate(seed=21, device="cpu")

    ref = gemm_reference(values)

    assert ref.shape == (4, 8)
    assert ref.dtype == torch.float32
    assert torch.isfinite(ref).all()


def test_gemm_inputs_support_nvfp4_fp8_scales() -> None:
    inputs = GemmInputs(
        nvfp4_gemm_input_config(
            M=4,
            N=8,
            K=64,
            c_dtype=torch.float32,
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


def test_gemm_reference_dequantizes_nvfp4_inputs() -> None:
    values = GemmInputs(
        nvfp4_gemm_input_config(
            M=4,
            N=8,
            K=64,
            c_dtype=torch.float32,
        )
    ).generate(seed=97, device="cpu")

    ref = gemm_reference(values, alpha=torch.tensor([0.5], dtype=torch.float32))

    assert ref.shape == (4, 8)
    assert ref.dtype == torch.float32
    assert torch.isfinite(ref).all()


def test_nvfp4_gemm_input_config_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="divisible by block_size"):
        nvfp4_gemm_input_config(M=4, N=8, K=68, c_dtype=torch.float32)

    with pytest.raises(ValueError, match="a_layout='MK'"):
        GemmInputs(
            nvfp4_gemm_input_config(
                M=4,
                N=8,
                K=64,
                c_dtype=torch.float32,
                a_layout="KM",
            )
        )


def test_gemm_inputs_support_mxint4_bf16_scales() -> None:
    inputs = GemmInputs(
        mxint4_gemm_input_config(
            M=4,
            N=8,
            K=64,
            c_dtype=torch.float32,
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
            mxint4_gemm_input_config(
                M=4,
                N=8,
                K=64,
                c_dtype=torch.float32,
                a_layout="KM",
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
        mxfp8_gemm_input_config(
            M=4,
            N=256,
            K=256,
            c_dtype=torch.float32,
            block_shape=block_shape,
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


def test_mxfp8_gemm_input_config_rejects_irregular_block_grid() -> None:
    with pytest.raises(ValueError, match="K to be divisible"):
        mxfp8_gemm_input_config(
            M=4,
            N=256,
            K=192,
            c_dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="N to be divisible"):
        mxfp8_gemm_input_config(
            M=4,
            N=192,
            K=256,
            c_dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="a_layout='MK'"):
        mxfp8_gemm_input_config(
            M=4,
            N=256,
            K=256,
            c_dtype=torch.float32,
            a_layout="KM",
        )


def test_router_projection_inputs_generate_useful_logits() -> None:
    values = RouterProjectionInputs(
        RouterProjectionInputConfig(
            num_tokens=5,
            hidden_dim=64,
            num_experts=7,
            hidden_dtype=torch.bfloat16,
            router_weight_dtype=torch.float32,
        )
    ).generate(seed=95, device="cpu")

    assert values.hidden_states.shape == (5, 64)
    assert values.router_weights.shape == (7, 64)
    assert values.hidden_states.dtype == torch.bfloat16
    assert values.router_weights.dtype == torch.float32

    logits = router_projection_reference(values)

    assert logits.shape == (5, 7)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert logits.float().std() > 0.0
    assert logits.float().abs().max() < 10.0


def test_router_projection_reference_matches_matmul() -> None:
    values = RouterProjectionInputValues(
        hidden_states=torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -0.25, 2.0]],
            dtype=torch.float32,
        ),
        router_weights=torch.tensor(
            [[0.5, 1.0, 2.0], [1.5, -1.0, 0.25]],
            dtype=torch.float32,
        ),
    )

    logits = router_projection_reference(values)

    torch.testing.assert_close(logits, values.hidden_states @ values.router_weights.T)


def test_router_projection_inputs_reject_invalid_configs() -> None:
    with pytest.raises(ValueError, match="num_tokens"):
        RouterProjectionInputs(
            RouterProjectionInputConfig(
                num_tokens=0,
                hidden_dim=64,
                num_experts=7,
            )
        )
    with pytest.raises(ValueError, match="hidden_dtype"):
        RouterProjectionInputs(
            RouterProjectionInputConfig(
                num_tokens=5,
                hidden_dim=64,
                num_experts=7,
                hidden_dtype=torch.int32,  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="router_weight_scale"):
        RouterProjectionInputs(
            RouterProjectionInputConfig(
                num_tokens=5,
                hidden_dim=64,
                num_experts=7,
                router_weight_scale=0.0,
            )
        )


def test_router_projection_reference_rejects_invalid_values() -> None:
    values = RouterProjectionInputValues(
        hidden_states=torch.randn(2, 4),
        router_weights=torch.randn(3, 5),
    )

    with pytest.raises(ValueError, match="hidden dimensions"):
        router_projection_reference(values)


def test_lm_head_projection_inputs_generate_useful_logits() -> None:
    values = LMHeadProjectionInputs(
        LMHeadProjectionInputConfig(
            num_tokens=4,
            hidden_dim=64,
            vocab_size=11,
            hidden_dtype=torch.bfloat16,
            weight_dtype=torch.bfloat16,
        )
    ).generate(seed=96, device="cpu")

    assert values.hidden_states.shape == (4, 64)
    assert values.weight.shape == (11, 64)
    assert values.hidden_states.dtype == torch.bfloat16
    assert values.weight.dtype == torch.bfloat16

    logits = lm_head_projection_reference(values)

    assert logits.shape == (4, 11)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits.float()).all()
    assert logits.float().std() > 0.0
    assert logits.float().abs().max() < 10.0


def test_lm_head_projection_reference_matches_matmul() -> None:
    values = LMHeadProjectionInputValues(
        hidden_states=torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -0.25, 2.0]],
            dtype=torch.float32,
        ),
        weight=torch.tensor(
            [[0.5, 1.0, 2.0], [1.5, -1.0, 0.25]],
            dtype=torch.float32,
        ),
    )

    logits = lm_head_projection_reference(values, out_dtype=torch.float32)

    torch.testing.assert_close(logits, values.hidden_states @ values.weight.T)


def test_lm_head_projection_inputs_reject_invalid_configs() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        LMHeadProjectionInputs(
            LMHeadProjectionInputConfig(
                num_tokens=4,
                hidden_dim=64,
                vocab_size=0,
            )
        )
    with pytest.raises(ValueError, match="weight_dtype"):
        LMHeadProjectionInputs(
            LMHeadProjectionInputConfig(
                num_tokens=4,
                hidden_dim=64,
                vocab_size=11,
                weight_dtype=torch.int32,  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="weight_scale"):
        LMHeadProjectionInputs(
            LMHeadProjectionInputConfig(
                num_tokens=4,
                hidden_dim=64,
                vocab_size=11,
                weight_scale=0.0,
            )
        )


def test_lm_head_projection_reference_rejects_invalid_values() -> None:
    values = LMHeadProjectionInputValues(
        hidden_states=torch.randn(2, 4),
        weight=torch.randn(3, 5),
    )

    with pytest.raises(ValueError, match="hidden dimensions"):
        lm_head_projection_reference(values)


def test_nvfp4_gemm_swiglu_inputs_generate_values_and_reference() -> None:
    values = NVFP4GemmSwiGLUNVFP4QuantInputs(
        NVFP4GemmSwiGLUNVFP4QuantInputConfig(
            M=3,
            K=64,
            intermediate_size=32,
            dtype=torch.bfloat16,
        )
    ).generate(seed=23, device="cpu")

    assert values.x.shape == (3, 64)
    assert values.w1.shape == (64, 64)
    assert values.x_fp4.shape == (3, 32)
    assert values.w1_fp4.shape == (64, 32)
    assert values.x_scale.shape == (3, 4)
    assert values.w1_scale.shape == (64, 4)
    assert values.fc1_alpha.shape == (1,)
    torch.testing.assert_close(
        values.fc1_alpha,
        values.x_global_scale * values.w1_global_scale,
    )
    assert values.output_global_scale.shape == (1,)
    torch.testing.assert_close(
        values.output_global_scale * values.output_global_scale_inv,
        torch.ones_like(values.output_global_scale),
    )

    packed, scales = nvfp4_gemm_swiglu_nvfp4_quant_reference(values)

    assert packed.shape == (3, 16)
    assert scales.shape == (3, 2)
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.float8_e4m3fn
    dequant = nvfp4_dequantization_reference(
        packed,
        scales,
        scale=values.output_global_scale,
    )
    assert dequant.shape == (3, 32)
    assert torch.isfinite(dequant).all()


def test_nvfp4_gemm_swiglu_reference_rejects_bad_shapes() -> None:
    values = NVFP4GemmSwiGLUNVFP4QuantInputs(
        NVFP4GemmSwiGLUNVFP4QuantInputConfig(
            M=3,
            K=64,
            intermediate_size=32,
            dtype=torch.bfloat16,
        )
    ).generate(seed=24, device="cpu")
    values.w1_fp4 = values.w1_fp4[:-1]

    with pytest.raises(ValueError, match="even gate/up"):
        nvfp4_gemm_swiglu_nvfp4_quant_reference(values)


def test_nvfp4_gemm_swiglu_inputs_verify_dimensions() -> None:
    with pytest.raises(ValueError, match="K must be divisible"):
        NVFP4GemmSwiGLUNVFP4QuantInputs(
            NVFP4GemmSwiGLUNVFP4QuantInputConfig(
                M=3,
                K=65,
                intermediate_size=32,
                dtype=torch.bfloat16,
            )
        )

    with pytest.raises(ValueError, match="intermediate_size must be divisible"):
        NVFP4GemmSwiGLUNVFP4QuantInputs(
            NVFP4GemmSwiGLUNVFP4QuantInputConfig(
                M=3,
                K=64,
                intermediate_size=31,
                dtype=torch.bfloat16,
            )
        )

    with pytest.raises(ValueError, match="output_global_scale must be positive"):
        NVFP4GemmSwiGLUNVFP4QuantInputs(
            NVFP4GemmSwiGLUNVFP4QuantInputConfig(
                M=3,
                K=64,
                intermediate_size=32,
                dtype=torch.bfloat16,
                output_global_scale=0.0,
            )
        )


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


def test_moe_reference_matches_manual_dense_silu() -> None:
    values = MoeInputs(
        MoeInputConfig(
            num_tokens=4,
            hidden_size=8,
            intermediate_size=12,
            num_experts=3,
            top_k=2,
            hidden_dtype=torch.float32,
            bias_dtype=torch.float32,
        )
    ).generate(seed=30, device="cpu")

    ref = moe_reference(values)
    assert values.hidden_states is not None
    assert values.w13.B is not None
    assert values.w2.B is not None
    assert values.w13_bias is not None
    assert values.w2_bias is not None
    manual = torch.zeros(4, 8, dtype=torch.float32)
    for token in range(values.hidden_states.shape[0]):
        hidden = values.hidden_states[token].float()
        for slot in range(values.topk_ids.shape[1]):
            expert = int(values.topk_ids[token, slot])
            route_weight = values.topk_weights[token, slot].float()
            gate_up = hidden @ values.w13.B[expert].float().T
            gate_up = gate_up + values.w13_bias[expert].float()
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            expert_output = activated @ values.w2.B[expert].float().T
            expert_output = expert_output + values.w2_bias[expert].float()
            manual[token] += route_weight * expert_output

    torch.testing.assert_close(ref, manual, atol=1.0e-5, rtol=1.0e-5)


def test_moe_reference_handles_mxfp4_weight_values() -> None:
    values = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
            weight_format="mxfp4",
        )
    ).generate(seed=32, device="cpu")

    ref = moe_reference(values)

    assert ref.shape == (5, 64)
    assert ref.dtype == torch.float16


def test_moe_reference_handles_mxint4_weight_values() -> None:
    values = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
            weight_format="mxint4",
        )
    ).generate(seed=35, device="cpu")

    ref = moe_reference(values)

    assert ref.shape == (5, 64)
    assert ref.dtype == torch.float16
    assert torch.isfinite(ref).all()


def test_moe_reference_rejects_invalid_topk_weights() -> None:
    values = MoeInputs(
        MoeInputConfig(
            num_tokens=4,
            hidden_size=8,
            intermediate_size=12,
            num_experts=3,
            top_k=2,
            hidden_dtype=torch.float32,
        )
    ).generate(seed=33, device="cpu")
    bad_values = MoeInputValues(
        hidden_states=values.hidden_states,
        router_logits=values.router_logits,
        topk_ids=values.topk_ids,
        topk_weights=values.topk_weights * 0.5,
        w13=values.w13,
        w2=values.w2,
        w13_bias=values.w13_bias,
        w2_bias=values.w2_bias,
        w13_activation_scale=values.w13_activation_scale,
        w2_activation_scale=values.w2_activation_scale,
    )

    with pytest.raises(ValueError, match="topk_weights rows must sum to 1"):
        moe_reference(bad_values)


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


def test_moe_softmax_topk_routing_inputs_generate_values_and_reference() -> None:
    values = MoESoftmaxTopKRoutingInputs(
        MoESoftmaxTopKRoutingInputConfig(
            num_tokens=3,
            num_experts=8,
            num_experts_real=6,
            top_k=4,
            scaling_factor=2.0,
        )
    ).generate(seed=51, device="cpu")

    ref = moe_softmax_topk_routing_reference(values)

    assert values.logits.shape == (3, 8)
    assert values.logits.dtype == torch.float32
    assert values.correction_bias.shape == (8,)
    assert values.topk_indices.shape == (3, 4)
    assert values.topk_indices.dtype == torch.int32
    assert values.topk_weights.shape == (3, 4)
    assert values.topk_weights.dtype == torch.float32
    assert ref.topk_indices.shape == (3, 4)
    assert ref.topk_weights.shape == (3, 4)
    assert torch.all(ref.topk_indices[:, 0] == -1)
    assert torch.all(ref.topk_weights >= 0.0)
    assert torch.isfinite(ref.topk_weights).all()


def test_moe_softmax_topk_routing_reference_handles_renormalization() -> None:
    values = MoESoftmaxTopKRoutingInputValues(
        logits=torch.tensor([[0.0, 1.0, -1.0, 0.5]], dtype=torch.float32),
        correction_bias=torch.tensor([0.0, 1.0, 0.0, 3.0], dtype=torch.float32),
        topk_indices=torch.empty((1, 2), dtype=torch.int64),
        topk_weights=torch.empty((1, 2), dtype=torch.float32),
        num_experts_real=3,
        scaling_factor=6.0,
        renormalize=True,
    )

    ref = moe_softmax_topk_routing_reference(values)
    probs = torch.softmax(values.logits, dim=-1)
    selected = probs[:, [3, 1]]
    expected_weights = selected / selected.sum(dim=-1, keepdim=True) * 6.0

    torch.testing.assert_close(
        ref.topk_indices,
        torch.tensor([[-1, 1]], dtype=torch.int64),
    )
    torch.testing.assert_close(ref.topk_weights, expected_weights)


def test_moe_softmax_topk_routing_verifies_config_and_values() -> None:
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

    values = MoESoftmaxTopKRoutingInputValues(
        logits=torch.zeros((2, 4), dtype=torch.float32),
        correction_bias=torch.zeros((3,), dtype=torch.float32),
        topk_indices=torch.empty((2, 2), dtype=torch.int32),
        topk_weights=torch.empty((2, 2), dtype=torch.float32),
        num_experts_real=3,
        scaling_factor=1.0,
        renormalize=False,
    )
    with pytest.raises(ValueError, match="one value per expert"):
        moe_softmax_topk_routing_reference(values)


def test_moe_biased_grouped_topk_inputs_generate_values_and_reference() -> None:
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

    ref = moe_biased_grouped_topk_reference(values)

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
    assert ref.topk_weights.shape == (4, 3)
    assert ref.topk_ids.shape == (4, 3)
    assert ref.topk_ids.dtype == torch.int32
    assert torch.all(ref.topk_ids[:3] >= 0)
    assert torch.all(ref.topk_ids[:3] < 8)
    assert torch.all(ref.topk_ids[3] == -1)
    torch.testing.assert_close(
        ref.topk_weights[:3].sum(dim=-1),
        torch.full((3,), 2.5),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_moe_biased_grouped_topk_reference_matches_manual_group_filter() -> None:
    values = MoEBiasedGroupedTopKInputValues(
        hidden_states=torch.zeros((1, 4), dtype=torch.float32),
        gating_output=torch.tensor(
            [[0.0, 1.0, 3.0, -1.0, 2.0, -2.0]],
            dtype=torch.float32,
        ),
        correction_bias=torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=torch.float32,
        ),
        top_k=3,
        renormalize=False,
        num_expert_groups=3,
        top_k_groups=2,
        routed_scaling_factor=1.0,
        logical_to_physical_map=None,
        num_token_non_padded=None,
    )

    ref = moe_biased_grouped_topk_reference(values)
    scores = values.gating_output.sigmoid()
    selection_scores = scores + values.correction_bias.reshape(1, -1)
    group_scores = selection_scores.reshape(1, 3, 2).topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = torch.topk(group_scores, k=2, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand(1, 3, 2).reshape(1, 6)
    expected_ids = torch.topk(
        selection_scores.masked_fill(~expert_mask, float("-inf")),
        k=3,
        dim=-1,
        sorted=False,
    ).indices.to(torch.int32)
    expected_weights = scores.gather(1, expected_ids.to(torch.long)).to(torch.float32)

    torch.testing.assert_close(ref.topk_ids, expected_ids)
    torch.testing.assert_close(ref.topk_weights, expected_weights)


def test_moe_biased_grouped_topk_verifies_config_and_values() -> None:
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

    values = MoEBiasedGroupedTopKInputs(
        MoEBiasedGroupedTopKInputConfig(
            num_tokens=2,
            hidden_size=4,
            num_experts=4,
            top_k=2,
            use_logical_to_physical_map=True,
        )
    ).generate(seed=53, device="cpu")
    assert values.logical_to_physical_map is not None
    values.logical_to_physical_map[0] = values.logical_to_physical_map[1]

    with pytest.raises(ValueError, match="must be a permutation"):
        moe_biased_grouped_topk_reference(values)


def test_moe_softplus_sqrt_topk_routing_inputs_generate_non_hash_values() -> None:
    values = MoESoftplusSqrtTopKRoutingInputs(
        MoESoftplusSqrtTopKRoutingInputConfig(
            num_tokens=3,
            num_experts=8,
            top_k=4,
            routed_scaling_factor=2.0,
        )
    ).generate(seed=54, device="cpu")

    ref = moe_softplus_sqrt_topk_routing_reference(values)

    assert values.logits.shape == (3, 8)
    assert values.logits.dtype == torch.float32
    assert values.correction_bias is not None
    assert values.correction_bias.shape == (8,)
    assert values.input_ids is None
    assert values.hash_indices_table is None
    assert values.topk_indices.shape == (3, 4)
    assert values.topk_indices.dtype == torch.int32
    assert values.topk_weights.shape == (3, 4)
    assert ref.topk_indices.shape == (3, 4)
    assert ref.topk_weights.shape == (3, 4)
    torch.testing.assert_close(
        ref.topk_weights.sum(dim=-1),
        torch.full((3,), 2.0),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_moe_softplus_sqrt_topk_routing_reference_handles_hash_table() -> None:
    values = MoESoftplusSqrtTopKRoutingInputValues(
        logits=torch.tensor(
            [
                [0.0, 1.0, -1.0, 0.5],
                [2.0, -2.0, 0.25, 1.5],
            ],
            dtype=torch.float32,
        ),
        correction_bias=None,
        input_ids=torch.tensor([1, 0], dtype=torch.int64),
        hash_indices_table=torch.tensor(
            [
                [3, 1],
                [0, 2],
            ],
            dtype=torch.int32,
        ),
        topk_indices=torch.empty((2, 2), dtype=torch.int32),
        topk_weights=torch.empty((2, 2), dtype=torch.float32),
        renormalize=True,
        routed_scaling_factor=3.0,
    )

    ref = moe_softplus_sqrt_topk_routing_reference(values)
    transformed = torch.sqrt(torch.nn.functional.softplus(values.logits))
    expected_ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32)
    expected_weights = transformed.gather(1, expected_ids.to(torch.long))
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * 3.0

    torch.testing.assert_close(ref.topk_indices, expected_ids)
    torch.testing.assert_close(ref.topk_weights, expected_weights)


def test_moe_softplus_sqrt_topk_routing_verifies_config_and_values() -> None:
    with pytest.raises(ValueError, match="renormalize=True"):
        MoESoftplusSqrtTopKRoutingInputs(
            MoESoftplusSqrtTopKRoutingInputConfig(
                num_tokens=2,
                num_experts=8,
                renormalize=False,
            )
        )

    values = MoESoftplusSqrtTopKRoutingInputValues(
        logits=torch.zeros((1, 4), dtype=torch.float32),
        correction_bias=None,
        input_ids=torch.tensor([0], dtype=torch.int32),
        hash_indices_table=torch.tensor([[1, 1]], dtype=torch.int32),
        topk_indices=torch.empty((1, 2), dtype=torch.int32),
        topk_weights=torch.empty((1, 2), dtype=torch.float32),
        renormalize=True,
        routed_scaling_factor=1.0,
    )

    with pytest.raises(ValueError, match="must not repeat experts"):
        moe_softplus_sqrt_topk_routing_reference(values)


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

    ref = moe_deepseek_v4_mega_moe_staging_reference(values)

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
    assert ref.x_fp8.shape == values.x_fp8.shape
    assert ref.x_sf.shape == values.x_sf.shape
    torch.testing.assert_close(ref.topk_idx_out, values.topk_ids)
    torch.testing.assert_close(ref.topk_weights_out, values.topk_weights)


def test_moe_deepseek_v4_mega_moe_staging_reference_packs_scale_exponents() -> None:
    values = MoEDeepSeekV4MegaMoEStagingInputValues(
        hidden_states=torch.ones((1, 128), dtype=torch.float32),
        topk_ids=torch.tensor([[2, 0]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        x_fp8=torch.empty((1, 128), dtype=torch.float8_e4m3fn),
        x_sf=torch.empty((1, 1), dtype=torch.int32),
        topk_idx_out=torch.empty((1, 2), dtype=torch.int32),
        topk_weights_out=torch.empty((1, 2), dtype=torch.float32),
        num_experts=4,
    )

    ref = moe_deepseek_v4_mega_moe_staging_reference(values)

    # For amax=1, the exact scale is 1/448. The staging op rounds that up to
    # the next power of two, 2^-8, whose biased exponent byte is 119.
    assert int(ref.x_sf[0, 0].item()) == 0x77777777
    torch.testing.assert_close(ref.topk_idx_out, values.topk_ids)
    torch.testing.assert_close(ref.topk_weights_out, values.topk_weights)
    assert torch.isfinite(ref.x_fp8.float()).all()


def test_moe_deepseek_v4_mega_moe_staging_verifies_config_and_values() -> None:
    with pytest.raises(ValueError, match="multiple of 128"):
        MoEDeepSeekV4MegaMoEStagingInputs(
            MoEDeepSeekV4MegaMoEStagingInputConfig(
                num_tokens=2,
                hidden_size=192,
                num_experts=4,
                top_k=2,
            )
        )

    values = MoEDeepSeekV4MegaMoEStagingInputs(
        MoEDeepSeekV4MegaMoEStagingInputConfig(
            num_tokens=2,
            hidden_size=128,
            num_experts=4,
            top_k=2,
        )
    ).generate(seed=56, device="cpu")
    values.topk_ids[0, 0] = 4

    with pytest.raises(ValueError, match="less than num_experts"):
        moe_deepseek_v4_mega_moe_staging_reference(values)


def test_moe_finalize_fuse_shared_inputs_generate_values_and_reference() -> None:
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

    ref = moe_finalize_fuse_shared_reference(values)

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
    assert ref.shape == (3, 8)
    assert ref.dtype == torch.bfloat16
    assert torch.isfinite(ref.float()).all()


def test_moe_finalize_fuse_shared_reference_matches_manual_sum() -> None:
    values = MoEFinalizeFuseSharedInputValues(
        gemm2_out=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.bfloat16,
        ),
        expanded_idx_to_permuted_idx=torch.tensor([0, 2, -1, 1], dtype=torch.int32),
        expert_weights=torch.tensor(
            [[0.25, 0.75], [0.5, 0.5]],
            dtype=torch.float32,
        ),
        shared_output=torch.tensor(
            [[1.0, -1.0], [0.5, 0.25]],
            dtype=torch.bfloat16,
        ),
    )

    ref = moe_finalize_fuse_shared_reference(values)
    expected = torch.stack(
        [
            0.25 * values.gemm2_out[0, :2].float()
            + 0.75 * values.gemm2_out[2, :2].float()
            + values.shared_output[0].float(),
            0.5 * values.gemm2_out[1, :2].float() + values.shared_output[1].float(),
        ]
    ).to(torch.bfloat16)

    torch.testing.assert_close(ref, expected)


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

    ref = moe_finalize_fuse_shared_reference(values)

    assert values.shared_output is None
    assert values.gemm2_out.shape == (4, 8)
    assert values.expert_weights.dtype == torch.bfloat16
    assert ref.shape == (2, 8)
    assert ref.dtype == torch.bfloat16


def test_moe_finalize_fuse_shared_verifies_config_and_values() -> None:
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

    values = MoEFinalizeFuseSharedInputs(
        MoEFinalizeFuseSharedInputConfig(
            num_tokens=2,
            hidden_size=8,
            top_k=2,
        )
    ).generate(seed=59, device="cpu")
    values.expanded_idx_to_permuted_idx[0] = values.gemm2_out.shape[0]

    with pytest.raises(ValueError, match="less than gemm2_out rows"):
        moe_finalize_fuse_shared_reference(values)


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


def test_moe_align_block_size_reference_pads_each_expert() -> None:
    values = MoeAlignBlockSizeInputValues(
        topk_ids=torch.tensor(
            [
                [1, 2, 3],
                [0, 1, 3],
                [0, 2, 3],
                [0, 1, 2],
            ],
            dtype=torch.int32,
        ),
        block_size=4,
        num_experts=4,
    )

    assert moe_align_block_size_buffer_dims(values) == (4, 16)
    ref = moe_align_block_size_reference(values)

    torch.testing.assert_close(
        ref.sorted_token_ids,
        torch.tensor(
            [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12],
            dtype=torch.int32,
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        ref.expert_ids,
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        ref.num_tokens_post_pad,
        torch.tensor([16], dtype=torch.int32),
        atol=0,
        rtol=0,
    )

    canonical = canonicalize_moe_align_block_size(ref, block_size=4)
    torch.testing.assert_close(
        canonical,
        torch.tensor(
            [16, 0, 1, 2, 3, 3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12],
            dtype=torch.int32,
        ),
        atol=0,
        rtol=0,
    )


def test_moe_align_block_size_reference_rejects_invalid_ids() -> None:
    values = MoeAlignBlockSizeInputValues(
        topk_ids=torch.tensor([[0, 4]], dtype=torch.int32),
        block_size=4,
        num_experts=4,
    )

    with pytest.raises(ValueError, match="less than num_experts"):
        moe_align_block_size_reference(values)
