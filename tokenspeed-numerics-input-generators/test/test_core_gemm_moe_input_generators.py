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
    GemmInputValues,
    GemmInputs,
    MoeInputConfig,
    MoeInputValues,
    MoeInputs,
    TensorInput,
    gemm_scale_shape,
    mxfp4_gemm_input_config,
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
    assert torch.all(inputs.topk_ids >= 0)
    assert torch.all(inputs.topk_ids < 4)
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
