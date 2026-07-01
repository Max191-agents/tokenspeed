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
    MoeAlignBlockSizeInputConfig,
    MoeAlignBlockSizeInputs,
    MoeAlignBlockSizeInputValues,
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
    NVFP4GemmSwiGLUNVFP4QuantInputConfig,
    NVFP4GemmSwiGLUNVFP4QuantInputs,
    TensorInput,
    canonicalize_moe_align_block_size,
    gemm_reference,
    gemm_scale_shape,
    moe_align_block_size_buffer_dims,
    moe_align_block_size_reference,
    moe_reference,
    mxfp4_gemm_input_config,
    nvfp4_dequantization_reference,
    nvfp4_gemm_swiglu_nvfp4_quant_reference,
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


def test_gemm_reference_applies_2d_block_scales() -> None:
    block_shape = (128, 128)
    values = GemmInputs(
        GemmInputConfig(
            M=4,
            N=256,
            K=256,
            a_dtype=_fp8_dtype,
            b_dtype=_fp8_dtype,
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
