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
from tokenspeed_kernel.numerics.inputs import get_input_generator, get_standard_shapes
from tokenspeed_kernel.numerics.outputs import get_output_extractor
from tokenspeed_kernel.numerics.tolerance import Tolerance
from tokenspeed_kernel.numerics.verify import (
    _compatible_reference_for_signature,
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
from tokenspeed_numerics_input_generators import (
    AttentionMergeStateInputValues,
    argmax_reference,
    attention_merge_state_reference,
    moe_reference,
    mxfp4_quantization_reference,
    rope_reference,
)

_fp8_dtype = Platform.get().fp8e4m3fn.dtype
_mla_fp8_dtypes = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
)


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


def test_gemm_input_generator_supports_mxfp4_ue8m0_scales() -> None:
    scale = ScaleFormat(
        storage_dtype=torch.uint8,
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
    assert inputs["A_scales"].dtype == torch.uint8
    assert inputs["B_scales"].dtype == torch.uint8
    assert inputs["block_size"] == [32]


def test_gemm_input_generator_supports_nvfp4_fp8_scales() -> None:
    scale = ScaleFormat(
        storage_dtype=torch.float8_e4m3fn,
        granularity="block",
        block_shape=(16,),
    )
    signature = format_signature(
        a=tensor_format("nvfp4", torch.uint8, scale=scale),
        b=tensor_format("nvfp4", torch.uint8, scale=scale),
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
    assert inputs["A_scales"].shape == (4, 4)
    assert inputs["B_scales"].shape == (8, 4)
    assert inputs["A_scales"].dtype == torch.float8_e4m3fn
    assert inputs["B_scales"].dtype == torch.float8_e4m3fn
    assert inputs["alpha"].shape == (1,)
    assert inputs["alpha"].dtype == torch.float32
    assert inputs["block_size"] == [16]


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


def test_gemm_mxfp4_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("gemm", "mm")
    assert any(spec.name == "torch_mm_mxfp4" for spec in specs)
    assert any(spec.name == "triton_mm_mxfp4" for spec in specs)


def test_gemm_mxfp4_reference_matches_package_reference() -> None:
    from tokenspeed_kernel.numerics.reference.gemm import torch_mm_mxfp4
    from tokenspeed_numerics_input_generators import GemmInputValues, gemm_reference

    scale = ScaleFormat(
        storage_dtype=torch.uint8,
        granularity="block",
        block_shape=(32,),
    )
    signature = format_signature(
        a=tensor_format("mxfp4", torch.uint8, scale=scale),
        b=tensor_format("mxfp4", torch.uint8, scale=scale),
    )
    inputs = get_input_generator(
        "gemm",
        "mm",
        dtype=torch.uint8,
        traits={},
        format_signature=signature,
        device="cpu",
    ).generate(M=4, N=8, K=64)

    actual = torch_mm_mxfp4(**inputs)
    expected = gemm_reference(
        GemmInputValues(
            A=inputs["A"],
            B=inputs["B"],
            C=inputs["C"],
            A_scales=inputs["A_scales"],
            B_scales=inputs["B_scales"],
        ),
        out_dtype=inputs["out_dtype"],
        alpha=inputs["alpha"],
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gemm_nvfp4_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("gemm", "mm")
    assert any(spec.name == "torch_mm_nvfp4" for spec in specs)
    assert any(spec.name == "cublaslt_mm_nvfp4" for spec in specs)


def test_gemm_nvfp4_reference_matches_package_reference() -> None:
    from tokenspeed_kernel.numerics.reference.gemm import torch_mm_nvfp4
    from tokenspeed_numerics_input_generators import GemmInputValues, gemm_reference

    scale = ScaleFormat(
        storage_dtype=torch.float8_e4m3fn,
        granularity="block",
        block_shape=(16,),
    )
    signature = format_signature(
        a=tensor_format("nvfp4", torch.uint8, scale=scale),
        b=tensor_format("nvfp4", torch.uint8, scale=scale),
    )
    inputs = get_input_generator(
        "gemm",
        "mm",
        dtype=torch.uint8,
        traits={},
        format_signature=signature,
        device="cpu",
    ).generate(M=4, N=8, K=64)

    actual = torch_mm_nvfp4(**inputs)
    expected = gemm_reference(
        GemmInputValues(
            A=inputs["A"],
            B=inputs["B"],
            C=inputs["C"],
            A_scales=inputs["A_scales"],
            B_scales=inputs["B_scales"],
        ),
        out_dtype=inputs["out_dtype"],
        alpha=inputs["alpha"],
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_gemm_fp8_channel_scaled_reference_matches_package_reference() -> None:
    from tokenspeed_kernel.numerics.reference.gemm import torch_mm_fp8_scaled_nkm
    from tokenspeed_numerics_input_generators import GemmInputValues, gemm_reference

    scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
    signature = next(
        iter(format_signatures(("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=scale))
    )
    inputs = get_input_generator(
        "gemm",
        "mm",
        dtype=_fp8_dtype,
        traits={"b_layout": frozenset({"KN"})},
        format_signature=signature,
        device="cpu",
    ).generate(M=4, N=6, K=8)

    assert inputs["A_scales"].shape == (4,)
    assert inputs["B_scales"].shape == (6,)
    actual = torch_mm_fp8_scaled_nkm(**inputs)
    expected = gemm_reference(
        GemmInputValues(
            A=inputs["A"],
            B=inputs["B"],
            C=inputs["C"],
            A_scales=inputs["A_scales"],
            B_scales=inputs["B_scales"],
        ),
        b_layout="KN",
        out_dtype=inputs["out_dtype"],
        alpha=inputs["alpha"],
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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


def test_quantization_fp8_generator_uses_package_inputs() -> None:
    unscaled = get_input_generator(
        "quantization",
        "fp8",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=71,
    ).generate(M=3, K=256, has_scale=False)
    scaled = get_input_generator(
        "quantization",
        "fp8",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=71,
    ).generate(M=3, K=256, has_scale=True)

    assert set(unscaled) == {"x"}
    assert unscaled["x"].shape == (3, 256)
    assert unscaled["x"].dtype == torch.bfloat16
    assert set(scaled) == {"x", "scale"}
    assert scaled["x"].shape == (3, 256)
    assert scaled["x"].dtype == torch.bfloat16
    assert scaled["scale"].shape == (1,)
    assert scaled["scale"].dtype == torch.float32
    assert torch.equal(unscaled["x"], scaled["x"])


def test_quantization_fp8_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("quantization", "fp8")
    assert any(spec.name == "torch_quantization_fp8" for spec in specs)
    assert any(spec.name == "triton_quantize_fp8" for spec in specs)
    assert get_standard_shapes("quantization", "fp8")
    generator = get_input_generator(
        "quantization",
        "fp8",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
    )
    assert generator.generate(M=2, K=128)["x"].shape == (2, 128)


def test_quantization_mxfp4_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "quantization",
        "mxfp4",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=83,
    ).generate(M=3, K=64)

    assert set(inputs) == {"x", "global_scale", "scale_size", "scale_layout"}
    assert inputs["x"].shape == (3, 64)
    assert inputs["x"].dtype == torch.bfloat16
    assert inputs["global_scale"] is None
    assert inputs["scale_size"] == 32
    assert inputs["scale_layout"] == "linear"

    packed, scales = mxfp4_quantization_reference(
        inputs["x"],
        scale_size=inputs["scale_size"],
        scale_layout=inputs["scale_layout"],
    )
    assert packed.shape == (3, 32)
    assert packed.dtype == torch.uint8
    assert scales.shape == (3, 2)
    assert scales.dtype == torch.uint8


def test_quantization_mxfp4_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("quantization", "mxfp4")
    assert any(spec.name == "torch_quantization_mxfp4" for spec in specs)
    assert any(spec.name == "triton_quantize_mxfp4" for spec in specs)
    assert get_standard_shapes("quantization", "mxfp4")


def test_quantization_mxfp4_reference_matches_package_reference() -> None:
    from tokenspeed_kernel.numerics.reference.quantize import torch_quantization_mxfp4

    inputs = get_input_generator(
        "quantization",
        "mxfp4",
        dtype=torch.float16,
        traits={},
        device="cpu",
        seed=84,
    ).generate(M=2, K=64)

    actual = torch_quantization_mxfp4(**inputs)
    expected = mxfp4_quantization_reference(
        inputs["x"],
        scale_size=inputs["scale_size"],
        scale_layout=inputs["scale_layout"],
    )

    assert len(actual) == len(expected) == 2
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


def test_quantization_input_generator_rejects_unsupported_mode() -> None:
    generator = get_input_generator(
        "quantization",
        "fp8",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
    )
    generator.op_mode = "unsupported"

    with pytest.raises(ValueError, match="unsupported quantize input mode"):
        generator.generate(M=2, K=128)


def test_sampling_argmax_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "sampling",
        "argmax",
        dtype=torch.float32,
        traits={},
        device="cpu",
        seed=81,
    ).generate(M=4, N=17, max_pattern="tied")

    assert set(inputs) == {"logits"}
    assert inputs["logits"].shape == (4, 17)
    assert inputs["logits"].dtype == torch.float32
    expected = argmax_reference(inputs["logits"])
    assert expected.shape == (4,)
    assert torch.all(expected >= 0)

    nan_inputs = get_input_generator(
        "sampling",
        "argmax",
        dtype=torch.float32,
        traits={},
        device="cpu",
        seed=82,
    ).generate(M=4, N=17, nan_pattern="mixed")
    nan_expected = argmax_reference(nan_inputs["logits"])
    assert (nan_expected == -1).any()


def test_sampling_argmax_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("sampling", "argmax")
    assert any(spec.name == "torch_sampling_argmax" for spec in specs)
    assert get_standard_shapes("sampling", "argmax")


def test_sampling_argmax_reference_honors_out_buffer() -> None:
    from tokenspeed_kernel.numerics.reference.sampling import torch_sampling_argmax

    logits = torch.tensor(
        [
            [0.0, 2.0, 2.0],
            [float("nan"), -1.0, -2.0],
            [float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    out = torch.empty((3,), dtype=torch.int32)

    returned = torch_sampling_argmax(logits, out=out)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.tensor([1, 1, -1], dtype=torch.int32))


def test_embedding_rope_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "embedding",
        "rope",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=91,
    ).generate(
        num_tokens=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=64,
        rotary_dim=32,
        with_q_output=True,
        with_k_output=True,
    )

    assert inputs["query"].shape == (4, 128)
    assert inputs["key"].shape == (4, 64)
    assert inputs["query"].dtype == torch.bfloat16
    assert inputs["key"].dtype == torch.bfloat16
    assert inputs["output_q_rope"].shape == inputs["query"].shape
    assert inputs["output_k_rope"].shape == inputs["key"].shape
    assert inputs["positions"].shape == (4,)
    assert int(inputs["positions"].min().item()) >= 0
    assert int(inputs["positions"].max().item()) < inputs["cos_sin_cache"].shape[0]

    q_ref, k_ref = rope_reference(
        inputs["query"],
        inputs["key"],
        inputs["positions"],
        head_size=inputs["head_size"],
        cos_sin_cache=inputs["cos_sin_cache"],
        is_neox=inputs["is_neox"],
        rotary_dim=inputs["rotary_dim"],
    )
    assert q_ref.shape == inputs["query"].shape
    assert k_ref.shape == inputs["key"].shape


def test_embedding_rope_output_extractor_collects_mutated_outputs() -> None:
    extractor = get_output_extractor("embedding", "rope")
    assert extractor is not None
    inputs = get_input_generator(
        "embedding",
        "rope",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=92,
    ).generate(
        num_tokens=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=64,
        rotary_dim=64,
        with_fused_kv=True,
        cache_size=8,
        with_q_output=True,
    )
    assert inputs["fused_set_kv_buffer_arg"] is not None

    outputs = extractor(inputs, None)

    assert len(outputs) == 4
    assert outputs[0].shape == inputs["query"].shape
    assert outputs[1].shape == inputs["key"].shape
    assert outputs[2].shape == inputs["key"].shape
    assert outputs[3].shape == inputs["key"].shape


def test_embedding_rope_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("embedding", "rope")
    assert any(spec.name == "torch_embedding_rope" for spec in specs)
    assert any(spec.name == "triton_embedding_rope" for spec in specs)
    assert get_standard_shapes("embedding", "rope")


def test_attention_merge_state_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "attention",
        "attn_merge_state",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=101,
    ).generate(total_q=5, num_heads=2, head_dim=16, lse_bound=2.0)

    assert set(inputs) == {"out_a", "lse_a", "out_b", "lse_b", "lse_scale_log2"}
    assert inputs["out_a"].shape == (5, 2, 16)
    assert inputs["out_b"].shape == (5, 2, 16)
    assert inputs["out_a"].dtype == torch.bfloat16
    assert inputs["out_b"].dtype == torch.bfloat16
    assert inputs["lse_a"].shape == (5, 2)
    assert inputs["lse_b"].shape == (5, 2)
    assert inputs["lse_a"].dtype == torch.float32
    assert inputs["lse_b"].dtype == torch.float32

    from tokenspeed_kernel.numerics.reference.attention import torch_attn_merge_state

    actual = torch_attn_merge_state(**inputs)
    expected = attention_merge_state_reference(
        AttentionMergeStateInputValues(
            out_a=inputs["out_a"],
            lse_a=inputs["lse_a"],
            out_b=inputs["out_b"],
            lse_b=inputs["lse_b"],
            lse_scale_log2=inputs["lse_scale_log2"],
        )
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


def test_attention_merge_state_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("attention", "attn_merge_state")
    assert any(spec.name == "torch_attn_merge_state" for spec in specs)
    assert any(spec.name == "triton_attn_merge_state" for spec in specs)
    assert get_standard_shapes("attention", "attn_merge_state")


def test_attention_mha_prefill_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "attention",
        "mha_prefill",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=102,
    ).generate(
        batch_size=2,
        total_new_q_tokens=6,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=64,
    )

    assert inputs["q"].shape == (6, 2, 64)
    assert inputs["k"].shape == (6, 1, 64)
    assert inputs["v"].shape == (6, 1, 64)
    assert inputs["cu_seqlens_q_cpu"] == [0, 3, 6]
    assert inputs["return_lse"] is False

    from tokenspeed_kernel.numerics.reference.attention import torch_mha_prefill

    out = torch_mha_prefill(**inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == inputs["q"].shape


def test_attention_mha_decode_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "attention",
        "mha_decode_with_kvcache",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=103,
    ).generate(
        batch_size=2,
        total_cached_tokens=8,
        total_new_q_tokens=2,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=64,
        page_size=64,
    )

    assert inputs["q"].shape == (2, 2, 64)
    assert inputs["k_cache"].shape[1:] == (64, 1, 64)
    assert inputs["v_cache"].shape == inputs["k_cache"].shape
    assert inputs["page_table"].shape[0] == 2
    assert inputs["cache_seqlens"].shape == (2,)
    assert inputs["return_lse"] is False

    from tokenspeed_kernel.numerics.reference.attention import (
        torch_mha_decode_with_kvcache,
    )

    out = torch_mha_decode_with_kvcache(**inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == inputs["q"].shape


def test_attention_mha_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    for mode, reference_name, triton_name in (
        ("mha_prefill", "torch_mha_prefill", "triton_mha_prefill"),
        (
            "mha_extend_with_kvcache",
            "torch_mha_extend_with_kvcache",
            "triton_mha_extend_with_kvcache",
        ),
        (
            "mha_decode_with_kvcache",
            "torch_mha_decode_with_kvcache",
            "triton_mha_decode_with_kvcache_cached",
        ),
    ):
        specs = registry.get_for_operator("attention", mode)
        assert any(spec.name == reference_name for spec in specs)
        assert any(spec.name == triton_name for spec in specs)
        assert get_standard_shapes("attention", mode)


def test_attention_mla_prefill_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "attention",
        "mla_prefill",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=104,
    ).generate(
        batch_size=2,
        total_new_q_tokens=6,
        num_q_heads=4,
        num_kv_heads=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        kv_lora_rank=16,
        v_head_dim=8,
    )

    assert inputs["q"].shape == (6, 4, 12)
    assert inputs["k"].shape == (6, 2, 12)
    assert inputs["v"].shape == (6, 2, 8)
    assert inputs["cu_seqlens_q"].shape == (3,)
    assert inputs["cu_seqlens_kv"].shape == (3,)
    assert inputs["return_lse"] is False

    from tokenspeed_kernel.numerics.reference.attention import torch_mla_prefill

    out = torch_mla_prefill(**inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (6, 4, 8)


def test_attention_mla_decode_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "attention",
        "mla_decode_with_kvcache",
        dtype=torch.bfloat16,
        traits={},
        device="cpu",
        seed=105,
    ).generate(
        batch_size=2,
        total_cached_tokens=8,
        total_new_q_tokens=2,
        num_q_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        kv_lora_rank=16,
        v_head_dim=8,
        page_size=64,
    )

    assert inputs["q"].shape == (2, 1, 4, 20)
    assert inputs["kv_cache"].shape[1:] == (64, 1, 20)
    assert inputs["page_table"].shape[0] == 2
    assert inputs["cache_seqlens"].shape == (2,)
    assert inputs["return_lse"] is False

    from tokenspeed_kernel.numerics.reference.attention import (
        torch_mla_decode_with_kvcache,
    )

    out = torch_mla_decode_with_kvcache(**inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1, 4, 16)


@pytest.mark.parametrize("dtype", _mla_fp8_dtypes)
def test_attention_mla_fp8_generators_use_package_inputs(
    dtype: torch.dtype,
) -> None:
    prefill_inputs = get_input_generator(
        "attention",
        "mla_prefill",
        dtype=dtype,
        traits={},
        device="cpu",
        seed=106,
    ).generate(
        batch_size=2,
        total_new_q_tokens=6,
        num_q_heads=4,
        num_kv_heads=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        kv_lora_rank=16,
        v_head_dim=8,
    )
    decode_inputs = get_input_generator(
        "attention",
        "mla_decode_with_kvcache",
        dtype=dtype,
        traits={},
        device="cpu",
        seed=107,
    ).generate(
        batch_size=2,
        total_cached_tokens=8,
        total_new_q_tokens=2,
        num_q_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        kv_lora_rank=16,
        v_head_dim=8,
        page_size=64,
    )

    assert prefill_inputs["q"].dtype == dtype
    assert prefill_inputs["k"].dtype == dtype
    assert prefill_inputs["v"].dtype == dtype
    assert decode_inputs["q"].dtype == dtype
    assert decode_inputs["kv_cache"].dtype == dtype

    from tokenspeed_kernel.numerics.reference.attention import (
        torch_mla_decode_with_kvcache,
        torch_mla_prefill,
    )

    prefill_out = torch_mla_prefill(**prefill_inputs)
    decode_out = torch_mla_decode_with_kvcache(**decode_inputs)
    assert isinstance(prefill_out, torch.Tensor)
    assert isinstance(decode_out, torch.Tensor)
    assert prefill_out.shape == (6, 4, 8)
    assert decode_out.shape == (2, 1, 4, 16)
    assert prefill_out.dtype == torch.bfloat16
    assert decode_out.dtype == torch.bfloat16


def test_attention_mla_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    for mode, reference_name, triton_name in (
        ("mla_prefill", "torch_mla_prefill", "triton_mla_prefill"),
        (
            "mla_decode_with_kvcache",
            "torch_mla_decode_with_kvcache",
            "triton_mla_decode_with_kvcache",
        ),
    ):
        specs = registry.get_for_operator("attention", mode)
        assert any(spec.name == reference_name for spec in specs)
        triton_spec = next(spec for spec in specs if spec.name == triton_name)
        for signature in triton_spec.format_signatures:
            assert (
                _compatible_reference_for_signature(registry, triton_spec, signature)
                is not None
            )
        assert get_standard_shapes("attention", mode)


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


def test_moe_apply_generator_uses_package_inputs() -> None:
    inputs = get_input_generator(
        "moe",
        "apply",
        dtype=torch.bfloat16,
        traits={"weight_dtype": frozenset({"mxfp4"})},
        device="cpu",
        seed=93,
    ).generate(
        num_tokens=3,
        hidden_size=64,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        weight_dtype="mxfp4",
        activation="silu",
    )
    values = inputs["w"]._tokenspeed_numerics_values

    assert inputs["x"].shape == (3, 64)
    assert inputs["router_logits"].shape == (3, 4)
    assert inputs["topk_ids"].shape == (3, 2)
    assert inputs["topk_weights"].shape == (3, 2)
    assert values.w13.B.shape == (4, 128, 32)
    assert values.w2.B.shape == (4, 64, 32)
    expected = moe_reference(values, output_dtype=torch.bfloat16)
    assert expected.shape == (3, 64)

    from tokenspeed_kernel.numerics.reference.moe import torch_moe_apply

    actual = torch_moe_apply(**inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_moe_apply_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("moe", "apply")
    assert any(spec.name == "torch_moe_apply" for spec in specs)
    assert any(spec.name == "triton_mxfp4_moe_apply" for spec in specs)
    assert get_standard_shapes("moe", "apply")


def test_moe_process_weights_generator_observes_moe_reference() -> None:
    inputs = get_input_generator(
        "moe",
        "process_weights",
        dtype=torch.bfloat16,
        traits={"weight_dtype": frozenset({"mxfp4"})},
        device="cpu",
        seed=94,
    ).generate(
        num_tokens=3,
        hidden_size=64,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        weight_dtype="mxfp4",
        activation="silu",
    )
    values = inputs["w"]._tokenspeed_numerics_values

    assert set(inputs) == {"plan", "w"}
    assert inputs["plan"]["weight_dtype"] == "mxfp4"
    assert inputs["plan"]["apply_kernel_name"] == "reference"
    assert inputs["w"].w13_weight.shape == (4, 128, 32)
    assert inputs["w"].w2_weight.shape == (4, 64, 32)

    from tokenspeed_kernel.numerics.reference.moe import torch_moe_process_weights

    actual = torch_moe_process_weights(**inputs)
    expected = moe_reference(values, output_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_moe_process_weights_numerics_registration_matches_kernel_family() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()

    specs = registry.get_for_operator("moe", "process_weights")
    assert any(spec.name == "torch_moe_process_weights" for spec in specs)
    assert any(spec.name == "triton_mxfp4_moe_process_weights" for spec in specs)
    assert get_standard_shapes("moe", "process_weights")
    assert get_output_extractor("moe", "process_weights") is not None


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


def test_verification_uses_empty_signature_with_compatible_reference(
    fresh_registry,
) -> None:
    signature = format_signature()
    ref_spec = KernelSpec(
        name="test_empty_signature_reference",
        family="moe",
        mode="process_weights",
        solution="reference",
        format_signatures=frozenset({signature}),
        traits={"weight_dtype": frozenset({"mxfp4"})},
    )
    test_spec = KernelSpec(
        name="test_empty_signature_process_weights",
        family="moe",
        mode="process_weights",
        solution="triton",
        format_signatures=frozenset({signature}),
        traits={"weight_dtype": frozenset({"mxfp4"})},
    )
    registry = KernelRegistry.get()
    registry.register(ref_spec, lambda **_kwargs: None)
    registry.register(test_spec, lambda **_kwargs: None)

    selected_signature, reference = _verification_signature_and_reference(
        registry,
        test_spec,
        torch.bfloat16,
        "x",
    )

    assert selected_signature == signature
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

    def test_gemm_mxfp4_triton_uint8(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for numerics verification")
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        spec = registry.get_by_name("triton_mm_mxfp4")
        if spec is None or not spec.capability.satisfied_by(platform):
            pytest.skip("triton_mm_mxfp4 is not available on this platform")

        try:
            results = verify_kernel(
                spec.name,
                dtype=torch.uint8,
                dtype_role="a",
                shapes=[
                    {"M": 4, "N": 32, "K": 64},
                    {"M": 17, "N": 64, "K": 128},
                ],
                tolerance=Tolerance(atol=1.0e-1, rtol=1.0e-1),
                verbose=False,
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

    def test_gemm_nvfp4_cublaslt_uint8(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for numerics verification")
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        spec = registry.get_by_name("cublaslt_mm_nvfp4")
        if spec is None or not spec.capability.satisfied_by(platform):
            pytest.skip("cublaslt_mm_nvfp4 is not available on this platform")

        try:
            results = verify_kernel(
                spec.name,
                dtype=torch.uint8,
                dtype_role="a",
                shapes=[
                    {"M": 4, "N": 32, "K": 64},
                    {"M": 17, "N": 64, "K": 128},
                ],
                tolerance=Tolerance(atol=1.0e-1, rtol=1.0e-1),
                verbose=False,
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
        _get_verifiable_specs(torch.bfloat16, "x", family="quantization"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_quantization_bf16(self, spec: KernelSpec):
        self._verify(spec, torch.bfloat16, "x")

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.int32, "indices", family="moe"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_moe_int32(self, spec: KernelSpec):
        self._verify(spec, torch.int32, "indices")

    @pytest.mark.parametrize(
        "kernel_name",
        [
            "triton_mxfp4_precomputed_moe_apply",
            "triton_mxfp4_ep_precomputed_moe_apply",
            "triton_mxfp4_moe_apply",
        ],
    )
    def test_moe_apply_mxfp4_triton_bf16(self, kernel_name: str):
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        spec = registry.get_by_name(kernel_name)
        if spec is None or not spec.capability.satisfied_by(platform):
            pytest.skip(f"{kernel_name} is not available on this platform")
        self._verify(spec, torch.bfloat16, "x")

    def test_moe_process_weights_mxfp4_triton_bf16(self):
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        spec = registry.get_by_name("triton_mxfp4_moe_process_weights")
        if spec is None or not spec.capability.satisfied_by(platform):
            pytest.skip("triton_mxfp4_moe_process_weights is not available")
        self._verify(spec, torch.bfloat16, "x")

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.bfloat16, "out_a", family="attention"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_attention_merge_state_bf16(self, spec: KernelSpec):
        self._verify(spec, torch.bfloat16, "out_a")

    @pytest.mark.parametrize(
        "kernel_name",
        [
            "triton_mha_prefill",
            "triton_mha_extend_with_kvcache",
            "triton_mha_decode_with_kvcache_cached",
        ],
    )
    def test_attention_mha_triton_bf16(self, kernel_name: str):
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        spec = registry.get_by_name(kernel_name)
        if spec is None or not spec.capability.satisfied_by(platform):
            pytest.skip(f"{kernel_name} is not available on this platform")
        self._verify(spec, torch.bfloat16, "q")

    @pytest.mark.parametrize(
        "kernel_name",
        [
            "triton_mla_prefill",
            "triton_mla_decode_with_kvcache",
        ],
    )
    def test_attention_mla_triton_bf16(self, kernel_name: str):
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        spec = registry.get_by_name(kernel_name)
        if spec is None or not spec.capability.satisfied_by(platform):
            pytest.skip(f"{kernel_name} is not available on this platform")
        self._verify(spec, torch.bfloat16, "q")

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.float32, "logits", family="sampling"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_sampling_argmax_float32(self, spec: KernelSpec):
        self._verify(spec, torch.float32, "logits")

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.bfloat16, "query", family="embedding"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_embedding_rope_bf16(self, spec: KernelSpec):
        self._verify(spec, torch.bfloat16, "query")
