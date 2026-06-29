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
from tokenspeed_kernel.numerics.comparison import compare_outputs, format_comparison
from tokenspeed_kernel.numerics.input_generators import (
    CustomDType,
    GemmInputConfig,
    GemmInputValues,
    GemmInputs,
    MoeInputConfig,
    MoeInputValues,
    MoeInputs,
    ScaledGemmInputConfig,
    ScaledGemmInputValues,
    ScaledGemmInputs,
    ScaledTensorInputConfig,
    ScaledTensorInput,
    TensorInput,
    gemm_scale_shape,
    mxfp4_scaled_gemm_input_config,
)
from tokenspeed_kernel.numerics.inputs import get_input_generator
from tokenspeed_kernel.numerics.tolerance import Tolerance
from tokenspeed_kernel.numerics.verify import (
    _verification_signature_and_reference,
    verify_kernel,
)
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.signature import (
    ScaleFormat,
    format_signature,
    format_signatures,
    tensor_format,
)

_fp8_dtype = Platform.get().fp8e4m3fn.dtype


class TestCompareOutputs:
    def test_treats_nan_as_mismatch(self) -> None:
        actual = torch.tensor([1.0, float("nan")], dtype=torch.float32)
        expected = torch.tensor([1.0, 1.0], dtype=torch.float32)

        result = compare_outputs(
            actual,
            expected,
            tolerance=Tolerance(atol=1e6, rtol=1e6),
        )

        assert not result.passed
        assert result.num_mismatches == 1

    def test_treats_inf_as_mismatch(self) -> None:
        actual = torch.tensor([float("inf"), 1.0], dtype=torch.float32)
        expected = torch.tensor([float("inf"), 1.0], dtype=torch.float32)

        result = compare_outputs(
            actual,
            expected,
            tolerance=Tolerance(atol=1e6, rtol=1e6),
        )

        assert not result.passed
        assert result.num_mismatches == 1


def test_gemm_input_generator_uses_signature_scale_metadata() -> None:
    scale = ScaleFormat(
        storage_dtype=torch.float32,
        granularity="block",
        block_shape=(128, 128),
    )
    signature = next(
        iter(format_signatures(("a", "b"), "mxfp8", {_fp8_dtype}, scale=scale))
    )
    generator = get_input_generator(
        "gemm",
        "mm",
        dtype=_fp8_dtype,
        traits={},
        format_signature=signature,
        device="cpu",
    )

    inputs = generator.generate(M=4, N=256, K=128)

    assert inputs["A"].dtype == _fp8_dtype
    assert inputs["B"].dtype == _fp8_dtype
    assert inputs["C"].shape == (4, 256)
    assert inputs["C"].dtype == torch.bfloat16
    assert inputs["A_scales"].shape == (4, 1)
    assert inputs["B_scales"].shape == (2, 1)
    assert inputs["A_scales"].dtype == torch.float32
    assert inputs["B_scales"].dtype == torch.float32
    assert inputs["block_size"] == [128, 128]


def test_gemm_input_generator_supports_mxfp4_fp8_scales() -> None:
    scale = ScaleFormat(
        storage_dtype=_fp8_dtype,
        granularity="block",
        block_shape=(32,),
    )
    signature = format_signature(
        a=tensor_format("mxfp4", torch.uint8, scale=scale),
        b=tensor_format("mxfp4", torch.uint8, scale=scale),
    )
    generator = get_input_generator(
        "gemm",
        "mm",
        dtype=torch.uint8,
        traits={},
        format_signature=signature,
        device="cpu",
    )

    inputs = generator.generate(M=4, N=8, K=64)

    assert inputs["A"].shape == (4, 32)
    assert inputs["B"].shape == (8, 32)
    assert inputs["C"].shape == (4, 8)
    assert inputs["A"].dtype == torch.uint8
    assert inputs["B"].dtype == torch.uint8
    assert inputs["C"].dtype == torch.bfloat16
    assert inputs["A_scales"].shape == (4, 2)
    assert inputs["B_scales"].shape == (8, 2)
    assert inputs["A_scales"].dtype == _fp8_dtype
    assert inputs["B_scales"].dtype == _fp8_dtype
    assert inputs["block_size"] == [32]


def test_gemm_input_generator_output_dict_accepts_generated_c() -> None:
    from tokenspeed_kernel.numerics.reference.gemm import torch_mm

    generator = get_input_generator(
        "gemm",
        "mm",
        dtype=torch.float32,
        traits={},
        device="cpu",
    )

    inputs = generator.generate(M=2, N=3, K=4)
    output = torch_mm(**inputs)

    assert output.shape == inputs["C"].shape
    assert output.dtype == inputs["C"].dtype


def test_gemm_input_generator_requires_mxfp8_block_shape() -> None:
    scale = ScaleFormat(
        storage_dtype=torch.float32,
        granularity="block",
        dynamic_block_shape=True,
    )
    signature = next(
        iter(format_signatures(("a", "b"), "mxfp8", {_fp8_dtype}, scale=scale))
    )
    generator = get_input_generator(
        "gemm",
        "mm",
        dtype=_fp8_dtype,
        traits={},
        format_signature=signature,
        device="cpu",
    )

    with pytest.raises(ValueError, match="requires concrete block_shape"):
        generator.generate(M=4, N=256, K=128)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64, _fp8_dtype],
)
def test_tensor_input_generates_standard_fp_dtypes(dtype: torch.dtype) -> None:
    first = TensorInput((2, 3), dtype).generate(seed=123, device="cpu")
    second = TensorInput((2, 3), dtype).generate(seed=123, device="cpu")

    assert first is not None
    assert second is not None
    assert first.shape == (2, 3)
    assert first.dtype == dtype
    assert torch.equal(first, second)


def test_tensor_input_dtype_none_skips_generation() -> None:
    tensor = TensorInput((2, 3), None).generate(seed=123, device="cpu")

    assert tensor is None


def test_tensor_input_rejects_non_floating_torch_dtype() -> None:
    with pytest.raises(ValueError, match="TensorInput only supports floating"):
        TensorInput((2, 3), torch.int32).generate(seed=123, device="cpu")


def test_tensor_input_uses_named_arguments() -> None:
    tensor = TensorInput((2, 3), torch.float32)

    values = tensor.generate(seed=123, device="cpu")

    assert values is not None
    assert values.shape == (2, 3)
    assert values.dtype == torch.float32


def test_scaled_tensor_inputs_accept_config_objects() -> None:
    scaled_config = ScaledTensorInputConfig(
        value_shape=(2, 4),
        value_dtype=torch.float16,
        scale_shape=(2, 1),
        scale_dtype=torch.float32,
    )
    scaled = ScaledTensorInput(scaled_config)
    scaled_values = scaled.generate(seed=124, device="cpu")

    assert scaled.config is scaled_config
    assert scaled.values_input is not None
    assert scaled_values.values is not None
    assert scaled_values.values.dtype == torch.float16


def test_scaled_tensor_input_generates_torch_values_and_scales() -> None:
    tensor = ScaledTensorInput(
        ScaledTensorInputConfig(
            value_shape=(4, 5),
            value_dtype=torch.float16,
            scale_shape=(4, 1),
            scale_dtype=torch.float32,
        )
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


def test_scaled_tensor_input_reuses_mutable_value_generator_dtype() -> None:
    tensor = ScaledTensorInput(
        ScaledTensorInputConfig(
            value_shape=(4, 5),
            value_dtype=torch.float16,
            scale_shape=(4, 1),
            scale_dtype=torch.float32,
        )
    )
    assert tensor.values_input is not None
    values_input = tensor.values_input
    values_input.dtype = torch.float64

    values = tensor.generate(seed=125, device="cpu")

    assert tensor.values_input is values_input
    assert values.values is not None
    assert values.values.dtype == torch.float64


def test_scaled_tensor_input_generates_mxfp4_values_and_scales() -> None:
    tensor = ScaledTensorInput(
        ScaledTensorInputConfig(
            value_shape=(4, 8),
            value_dtype=CustomDType.MXFP4,
            scale_shape=(4, 1),
            scale_dtype=torch.float32,
        )
    ).generate(seed=126, device="cpu")

    assert tensor.values is not None
    assert tensor.scales is not None
    assert tensor.values.shape == (4, 8)
    assert tensor.values.dtype == torch.uint8
    assert tensor.scales.dtype == torch.float32
    assert torch.all(tensor.scales > 0.0)
    assert torch.all(tensor.scales <= 0.125)


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


def test_gemm_inputs_support_custom_mxfp4_dtype() -> None:
    values = GemmInputs(
        GemmInputConfig(
            M=4,
            N=8,
            K=64,
            a_dtype=CustomDType.MXFP4,
            b_dtype=CustomDType.MXFP4,
            c_dtype=torch.float32,
        )
    ).generate(seed=9, device="cpu")

    assert values.A is not None
    assert values.B is not None
    assert values.C is not None
    assert values.A.shape == (4, 32)
    assert values.B.shape == (8, 32)
    assert values.A.dtype == torch.uint8
    assert values.B.dtype == torch.uint8
    assert values.C.shape == (4, 8)


def test_scaled_gemm_inputs_generate_scaled_operands() -> None:
    inputs = ScaledGemmInputs(
        ScaledGemmInputConfig(
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
    assert inputs.A.values is not None
    assert inputs.B.values is not None
    assert inputs.A.scales is not None
    assert inputs.B.scales is not None
    assert inputs.C is not None
    assert inputs.A.values.shape == (4, 8)
    assert inputs.B.values.shape == (6, 8)
    assert inputs.A.scales.shape == (4,)
    assert inputs.B.scales.shape == (6,)
    assert inputs.A.values.dtype == _fp8_dtype
    assert inputs.B.values.dtype == _fp8_dtype
    assert inputs.C.shape == (4, 6)


def test_scaled_gemm_inputs_use_mutable_config_fields() -> None:
    inputs = ScaledGemmInputs(
        ScaledGemmInputConfig(
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

    assert values.A.scales is not None
    assert values.C is not None
    assert values.A.scales.dtype == torch.float64
    assert values.C.dtype == torch.float64


def test_scaled_gemm_inputs_accept_config_objects() -> None:
    config = ScaledGemmInputConfig(
        M=2,
        N=3,
        K=4,
        a_dtype=torch.float16,
        b_dtype=torch.float64,
        a_scale_dtype=torch.float32,
        b_scale_dtype=torch.float64,
        c_dtype=torch.float32,
        b_scale_shape=(3,),
    )
    inputs = ScaledGemmInputs(config)

    values = inputs.generate(seed=20, device="cpu")

    assert inputs.config is config
    assert values.B.values is not None
    assert values.B.scales is not None
    assert values.B.values.dtype == torch.float64
    assert values.B.scales.dtype == torch.float64
    assert values.C.shape == (2, 3)
    assert values.C.dtype == torch.float32


def test_scaled_gemm_inputs_support_mxfp4_fp8_scales() -> None:
    inputs = ScaledGemmInputs(
        mxfp4_scaled_gemm_input_config(
            M=4,
            N=8,
            K=64,
            scale_dtype=_fp8_dtype,
            c_dtype=torch.float32,
        )
    ).generate(seed=19, device="cpu")

    assert inputs.A is not None
    assert inputs.B is not None
    assert inputs.A.values is not None
    assert inputs.B.values is not None
    assert inputs.A.scales is not None
    assert inputs.B.scales is not None
    assert inputs.A.values.shape == (4, 32)
    assert inputs.B.values.shape == (8, 32)
    assert inputs.A.values.dtype == torch.uint8
    assert inputs.B.values.dtype == torch.uint8
    assert inputs.A.scales.shape == (4, 2)
    assert inputs.B.scales.shape == (8, 2)
    assert inputs.A.scales.dtype == _fp8_dtype
    assert inputs.B.scales.dtype == _fp8_dtype
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


def test_moe_inputs_compose_mxfp4_scaled_weight_gemms() -> None:
    inputs = MoeInputs(
        MoeInputConfig(
            num_tokens=5,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.float16,
            weight_format="mxfp4",
            weight_scale_dtype=_fp8_dtype,
        )
    ).generate(seed=29, device="cpu")

    assert isinstance(inputs.w13, ScaledGemmInputValues)
    assert isinstance(inputs.w2, ScaledGemmInputValues)
    assert inputs.w13.A is not None
    assert inputs.w13.B is not None
    assert inputs.w2.A is not None
    assert inputs.w2.B is not None
    assert inputs.w13.A.values is None
    assert inputs.w13.A.scales is None
    assert inputs.w2.A.values is None
    assert inputs.w2.A.scales is None
    assert inputs.w13.B.values is not None
    assert inputs.w13.B.scales is not None
    assert inputs.w2.B.values is not None
    assert inputs.w2.B.scales is not None
    assert inputs.w13.B.values.shape == (4, 64, 32)
    assert inputs.w13.B.scales.shape == (4, 64, 2)
    assert inputs.w2.B.values.shape == (4, 64, 16)
    assert inputs.w2.B.scales.shape == (4, 64, 1)
    assert inputs.w13.B.values.dtype == torch.uint8
    assert inputs.w13.B.scales.dtype == _fp8_dtype


def test_moe_align_block_size_generator_uses_typed_tensor_input() -> None:
    first = get_input_generator(
        "moe",
        "align_block_size",
        dtype=torch.int32,
        traits={},
        device="cpu",
        seed=31,
    ).generate(total_tokens=6, top_k=2, num_experts=4, block_size=64)
    second = get_input_generator(
        "moe",
        "align_block_size",
        dtype=torch.int32,
        traits={},
        device="cpu",
        seed=31,
    ).generate(total_tokens=6, top_k=2, num_experts=4, block_size=64)

    assert first["topk_ids"].shape == (6, 2)
    assert first["topk_ids"].dtype == torch.int32
    assert torch.all(first["topk_ids"] >= 0)
    assert torch.all(first["topk_ids"] < 4)
    assert torch.equal(first["topk_ids"], second["topk_ids"])
    assert first["block_size"] == 64
    assert first["num_experts"] == 4


def test_verification_uses_signature_with_compatible_reference(fresh_registry) -> None:
    tensor_scale = ScaleFormat(storage_dtype=torch.float32, granularity="tensor")
    channel_scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
    tensor_signature = next(
        iter(
            format_signatures(
                ("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=tensor_scale
            )
        )
    )
    channel_signature = next(
        iter(
            format_signatures(
                ("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=channel_scale
            )
        )
    )
    ref_spec = KernelSpec(
        name="test_tensor_scale_reference",
        family="gemm",
        mode="mm",
        solution="reference",
        format_signatures=frozenset({tensor_signature}),
        traits={"b_layout": frozenset({"KN"})},
    )
    test_spec = KernelSpec(
        name="test_fp8_scaled",
        family="gemm",
        mode="mm",
        solution="triton",
        format_signatures=frozenset({channel_signature, tensor_signature}),
        traits={"b_layout": frozenset({"KN"})},
    )
    registry = KernelRegistry.get()
    registry.register(ref_spec, lambda **_kwargs: None)
    registry.register(test_spec, lambda **_kwargs: None)

    signature, reference = _verification_signature_and_reference(
        registry, test_spec, _fp8_dtype, "a"
    )

    assert signature == tensor_signature
    assert reference is ref_spec


class TestNumericsVerification:
    def _get_verifiable_specs(
        dtype: torch.dtype, dtype_role: str, family: str | None = None
    ) -> list[KernelSpec]:
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        specs: list[KernelSpec] = []
        for family_name, mode in registry.list_operators():
            if family and family_name != family:
                continue
            # Only run kernels that have a paired reference for this dtype;
            # otherwise verify_kernel raises ValueError and the test errors.
            op_specs = registry.get_for_operator(family_name, mode)
            dtype_specs = [
                s
                for s in op_specs
                if s.format_signatures_for_storage_dtype(dtype, dtype_role)
            ]
            has_reference = any(s.solution == "reference" for s in dtype_specs)
            if not has_reference:
                continue
            for spec in dtype_specs:
                if spec.solution == "reference":
                    continue
                if spec.solution == "deep_gemm":
                    continue
                if not spec.capability.satisfied_by(platform):
                    continue
                specs.append(spec)
        specs.sort(key=lambda s: (s.family, s.mode, s.name))
        return specs

    def _verify(self, spec: KernelSpec, dtype: torch.dtype, dtype_role: str) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for numerics verification")

        try:
            results = verify_kernel(
                spec.name, dtype=dtype, dtype_role=dtype_role, verbose=False
            )
        except Exception as exc:
            pytest.fail(
                f"Kernel {spec.name} raised an exception during verification: {exc}"
            )

        for i, result in enumerate(results):
            if not result.passed:
                pytest.fail(
                    f"Kernel {spec.name} failed numerics verification for shape set {i}:\n"
                    f"{format_comparison(result, kernel_name=spec.name)}"
                )

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(_fp8_dtype, "a", family="gemm"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_gemm_fp8(self, spec: KernelSpec):
        self._verify(spec, _fp8_dtype, "a")

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.bfloat16, "a", family="gemm"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_gemm_bf16(self, spec: KernelSpec):
        self._verify(spec, torch.bfloat16, "a")

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.bfloat16, "x", family="quantize"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_quantize_bf16(self, spec: KernelSpec):
        self._verify(spec, torch.bfloat16, "x")

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.int32, "indices", family="moe"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_moe_int32(self, spec: KernelSpec):
        self._verify(spec, torch.int32, "indices")
