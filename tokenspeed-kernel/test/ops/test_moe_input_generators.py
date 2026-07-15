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
import tokenspeed_kernel.numerics.moe  # noqa: F401
import torch
from moe_references import (
    moe_align_block_size_reference,
    moe_biased_grouped_topk_reference,
    moe_deepseek_v4_mega_moe_staging_reference,
    moe_finalize_fuse_shared_reference,
    moe_softmax_topk_routing_reference,
    moe_softplus_sqrt_topk_routing_reference,
)
from tokenspeed_kernel import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.numerics.inputs import get_input_generator
from tokenspeed_kernel.numerics.moe import canonicalize_align_block_size
from tokenspeed_kernel.numerics.reference.moe import moe_reference
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import load_builtin_kernels
from tokenspeed_numerics_input_generators import (
    CustomDType,
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
)


def _make_mxfp4_moe_weight_module(values: MoeInputValues) -> torch.nn.Module:
    if values.hidden_states is None or values.router_logits is None:
        raise ValueError("MoE values must include hidden states and router logits")
    if values.w13.B is None or values.w13.B_scales is None:
        raise ValueError("MoE values must include W13 MXFP4 weights and scales")
    if values.w2.B is None or values.w2.B_scales is None:
        raise ValueError("MoE values must include W2 MXFP4 weights and scales")

    num_experts = values.w13.B.shape[0]
    layer = torch.nn.Module()
    layer.num_experts = num_experts
    layer.num_local_experts = num_experts
    layer.ep_size = 1
    layer.ep_rank = 0
    layer.top_k = values.topk_ids.shape[1]
    layer.activation = "silu"
    layer.swiglu_arg = None
    layer.w13_weight = torch.nn.Parameter(values.w13.B.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        values.w13.B_scales.clone(),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(values.w2.B.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(
        values.w2.B_scales.clone(),
        requires_grad=False,
    )
    return layer


def _make_dense_moe_weight_module(values: MoeInputValues) -> torch.nn.Module:
    if values.hidden_states is None or values.router_logits is None:
        raise ValueError("MoE values must include hidden states and router logits")
    if values.w13.B is None or values.w2.B is None:
        raise ValueError("MoE values must include dense W13 and W2 weights")

    num_experts = values.w13.B.shape[0]
    layer = torch.nn.Module()
    layer.num_experts = num_experts
    layer.num_local_experts = num_experts
    layer.ep_size = 1
    layer.ep_rank = 0
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.top_k = values.topk_ids.shape[1]
    layer.activation = "silu"
    layer.swiglu_arg = None
    layer.w13_weight = torch.nn.Parameter(values.w13.B.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(values.w2.B.clone(), requires_grad=False)
    return layer


def _mxint4_checkpoint_words_from_generator(packed: torch.Tensor) -> torch.Tensor:
    """Convert generated signed INT4 bytes to checkpoint-style int32 words."""

    low = packed & 0xF
    high = packed >> 4
    nibbles = packed.new_empty((*packed.shape[:-1], packed.shape[-1] * 2))
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high
    signed = nibbles.to(torch.int8)
    signed = torch.where(signed >= 8, signed - 16, signed)
    unsigned = (signed.to(torch.int32) + 8).reshape(*signed.shape[:-1], -1, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    return ((unsigned << shifts).sum(dim=-1)).to(torch.int32)


def test_moe_align_block_size_numerics_adapter_uses_generator() -> None:
    generated = get_input_generator(
        "moe",
        "align_block_size",
        dtype=torch.int32,
        traits={},
        device="cpu",
        seed=41,
    ).generate(total_tokens=7, top_k=2, num_experts=5, block_size=4)

    values = MoeAlignBlockSizeInputValues(
        topk_ids=generated["topk_ids"],
        block_size=generated["block_size"],
        num_experts=generated["num_experts"],
    )
    ref = moe_align_block_size_reference(values)

    assert values.topk_ids.shape == (7, 2)
    assert values.topk_ids.dtype == torch.int32
    assert torch.all(values.topk_ids >= 0)
    assert torch.all(values.topk_ids < 5)
    assert ref.sorted_token_ids.numel() % values.block_size == 0
    assert ref.expert_ids.numel() * values.block_size == ref.sorted_token_ids.numel()
    canonical = canonicalize_align_block_size(
        ref.sorted_token_ids,
        ref.expert_ids,
        ref.num_tokens_post_pad,
        values.block_size,
    )
    assert canonical.dtype == torch.int32
    assert (
        canonical.numel() == 1 + ref.expert_ids.numel() + ref.sorted_token_ids.numel()
    )


def test_moe_softmax_topk_routing_generator_runs_cuda_helper() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_nvidia:
        pytest.skip("routing_flash compatibility test requires an NVIDIA CUDA GPU")

    from tokenspeed_kernel.thirdparty.cuda import routing_flash

    values = MoESoftmaxTopKRoutingInputs(
        MoESoftmaxTopKRoutingInputConfig(
            num_tokens=8,
            num_experts=384,
            num_experts_real=256,
            top_k=12,
            scaling_factor=6.0,
            renormalize=False,
        )
    ).generate(seed=45, device="cuda")
    expected = moe_softmax_topk_routing_reference(values)

    try:
        routing_flash(
            values.logits,
            values.correction_bias,
            values.topk_indices,
            values.topk_weights,
            values.num_experts_real,
            values.scaling_factor,
            values.renormalize,
        )
    except RuntimeError as exc:
        pytest.skip(f"routing_flash extension unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(values.topk_indices, expected.topk_indices)
    torch.testing.assert_close(
        values.topk_weights,
        expected.topk_weights,
        rtol=1.0e-3,
        atol=8.0e-2,
    )


def test_moe_biased_grouped_topk_generator_matches_triton_fallback() -> None:
    from tokenspeed_kernel.thirdparty.triton import minimax_biased_grouped_topk

    values = MoEBiasedGroupedTopKInputs(
        MoEBiasedGroupedTopKInputConfig(
            num_tokens=5,
            hidden_size=8,
            num_experts=8,
            top_k=3,
            num_expert_groups=2,
            top_k_groups=1,
            renormalize=True,
            routed_scaling_factor=2.0,
            use_logical_to_physical_map=True,
            num_token_non_padded=4,
        )
    ).generate(seed=46, device="cpu")
    expected = moe_biased_grouped_topk_reference(values)

    actual_weights, actual_ids = minimax_biased_grouped_topk(
        values.hidden_states,
        values.gating_output,
        values.correction_bias,
        topk=values.top_k,
        renormalize=values.renormalize,
        num_expert_group=values.num_expert_groups,
        topk_group=values.top_k_groups,
        routed_scaling_factor=values.routed_scaling_factor,
        num_token_non_padded=values.num_token_non_padded,
        logical_to_physical_map=values.logical_to_physical_map,
    )

    torch.testing.assert_close(actual_ids, expected.topk_ids)
    torch.testing.assert_close(actual_weights, expected.topk_weights)


def test_moe_softplus_sqrt_topk_routing_generator_runs_cuda_helpers() -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_nvidia:
        pytest.skip("softplus-sqrt routing compatibility test requires NVIDIA CUDA")

    from tokenspeed_kernel.thirdparty.cuda import (
        hash_softplus_sqrt_topk_flash,
        softplus_sqrt_topk_flash,
    )

    non_hash = MoESoftplusSqrtTopKRoutingInputs(
        MoESoftplusSqrtTopKRoutingInputConfig(
            num_tokens=8,
            num_experts=256,
            top_k=6,
            routed_scaling_factor=1.75,
        )
    ).generate(seed=47, device="cuda")
    non_hash_expected = moe_softplus_sqrt_topk_routing_reference(non_hash)
    assert non_hash.correction_bias is not None
    try:
        softplus_sqrt_topk_flash(
            non_hash.logits,
            non_hash.correction_bias,
            non_hash.topk_indices,
            non_hash.topk_weights,
            non_hash.routed_scaling_factor,
            non_hash.renormalize,
        )
    except RuntimeError as exc:
        pytest.skip(f"softplus_sqrt_topk_flash extension unavailable: {exc}")

    hashed = MoESoftplusSqrtTopKRoutingInputs(
        MoESoftplusSqrtTopKRoutingInputConfig(
            num_tokens=8,
            num_experts=256,
            top_k=6,
            use_hash_table=True,
            hash_table_size=16,
            routed_scaling_factor=2.25,
        )
    ).generate(seed=48, device="cuda")
    hashed_expected = moe_softplus_sqrt_topk_routing_reference(hashed)
    assert hashed.input_ids is not None
    assert hashed.hash_indices_table is not None
    try:
        hash_softplus_sqrt_topk_flash(
            hashed.logits,
            hashed.input_ids,
            hashed.hash_indices_table,
            hashed.topk_indices,
            hashed.topk_weights,
            hashed.routed_scaling_factor,
            hashed.renormalize,
        )
    except RuntimeError as exc:
        pytest.skip(f"hash_softplus_sqrt_topk_flash extension unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(non_hash.topk_indices, non_hash_expected.topk_indices)
    torch.testing.assert_close(
        non_hash.topk_weights,
        non_hash_expected.topk_weights,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    torch.testing.assert_close(hashed.topk_indices, hashed_expected.topk_indices)
    torch.testing.assert_close(
        hashed.topk_weights,
        hashed_expected.topk_weights,
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_moe_deepseek_v4_mega_moe_staging_generator_runs_triton_helper() -> None:
    if not torch.cuda.is_available():
        pytest.skip("DeepSeek V4 MegaMoE staging compatibility test requires CUDA")

    from tokenspeed_kernel.thirdparty.triton import stage_deepseek_v4_mega_moe_inputs

    values = MoEDeepSeekV4MegaMoEStagingInputs(
        MoEDeepSeekV4MegaMoEStagingInputConfig(
            num_tokens=4,
            hidden_size=128,
            num_experts=8,
            top_k=3,
            hidden_dtype=torch.bfloat16,
        )
    ).generate(seed=49, device="cuda")
    expected = moe_deepseek_v4_mega_moe_staging_reference(values)

    try:
        stage_deepseek_v4_mega_moe_inputs(
            values.hidden_states,
            values.topk_weights,
            values.topk_ids,
            values.x_fp8,
            values.x_sf,
            values.topk_idx_out,
            values.topk_weights_out,
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"stage_deepseek_v4_mega_moe_inputs unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(
        values.x_fp8.float(),
        expected.x_fp8.float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(values.x_sf, expected.x_sf)
    torch.testing.assert_close(values.topk_idx_out, expected.topk_idx_out)
    torch.testing.assert_close(values.topk_weights_out, expected.topk_weights_out)


def test_moe_finalize_fuse_shared_generator_runs_cuda_helper() -> None:
    platform = current_platform()
    if (
        not torch.cuda.is_available()
        or not platform.is_nvidia
        or not platform.is_hopper_plus
    ):
        pytest.skip("moe_finalize_fuse_shared compatibility test requires NVIDIA SM90+")

    from tokenspeed_kernel.thirdparty.cuda import moe_finalize_fuse_shared

    values = MoEFinalizeFuseSharedInputs(
        MoEFinalizeFuseSharedInputConfig(
            num_tokens=4,
            hidden_size=64,
            top_k=2,
            include_shared_output=True,
            expert_weights_dtype=torch.float32,
        )
    ).generate(seed=50, device="cuda")
    expected = moe_finalize_fuse_shared_reference(values)

    try:
        actual = moe_finalize_fuse_shared(
            values.gemm2_out,
            values.expanded_idx_to_permuted_idx,
            values.expert_weights,
            values.shared_output,
            values.expert_weights.shape[1],
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        pytest.skip(f"moe_finalize_fuse_shared extension unavailable: {exc}")
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5.0e-2, atol=0.1)


def test_mxfp4_moe_generator_runs_triton_precomputed_kernel(device: str) -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_amd:
        pytest.skip("Triton MXFP4 MoE compatibility test requires an AMD GPU")

    load_builtin_kernels()
    config = MoeInputConfig(
        num_tokens=4,
        hidden_size=64,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        hidden_dtype=torch.bfloat16,
        router_dtype=torch.bfloat16,
        weight_format="mxfp4",
        weight_dtype=CustomDType.MXFP4,
        weight_scale_dtype=None,
        bias_dtype=None,
    )
    values = MoeInputs(config).generate(seed=42, device=device)
    layer = _make_mxfp4_moe_weight_module(values)
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        internal_activation_dtype="input",
        with_bias=False,
        solution="triton",
    )
    assert plan["solution"] == "triton"
    assert plan["process_weights_kernel_name"] == "triton_mxfp4_moe_process_weights"

    moe_process_weights(plan, layer)
    actual = moe_apply(
        plan,
        values.hidden_states,
        layer,
        values.router_logits,
        topk_weights=values.topk_weights,
        topk_ids=values.topk_ids,
    )
    torch.cuda.synchronize()

    expected = moe_reference(values, output_dtype=torch.bfloat16).to(device=device)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5.0e-2, atol=0.1)


def test_mxint4_moe_generator_matches_flashinfer_trtllm_contract() -> None:
    config = MoeInputConfig(
        num_tokens=4,
        hidden_size=64,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        hidden_dtype=torch.bfloat16,
        router_dtype=torch.float32,
        weight_format="mxint4",
        weight_dtype=CustomDType.MXINT4,
        weight_scale_dtype=None,
        bias_dtype=None,
    )
    values = MoeInputs(config).generate(seed=44, device="cpu")
    assert values.w13.B is not None
    assert values.w13.B_scales is not None
    assert values.w2.B is not None
    assert values.w2.B_scales is not None

    w13_weight_packed = _mxint4_checkpoint_words_from_generator(values.w13.B)
    w2_weight_packed = _mxint4_checkpoint_words_from_generator(values.w2.B)
    layer = torch.nn.Module()
    layer.num_experts = config.num_experts
    layer.num_local_experts = config.num_experts
    layer.ep_size = 1
    layer.ep_rank = 0
    layer.tp_size = 1
    layer.top_k = config.top_k
    layer.intermediate_size = config.intermediate_size
    layer.w13_weight_packed = torch.nn.Parameter(
        w13_weight_packed,
        requires_grad=False,
    )
    layer.w2_weight_packed = torch.nn.Parameter(
        w2_weight_packed,
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        values.w13.B_scales.clone(),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        values.w2.B_scales.clone(),
        requires_grad=False,
    )

    assert layer.w13_weight_packed.shape == (4, 128, 8)
    assert layer.w2_weight_packed.shape == (4, 64, 8)
    assert layer.w13_weight_packed.dtype == torch.int32
    assert layer.w2_weight_packed.dtype == torch.int32
    assert layer.w13_weight_scale.shape == (4, 128, 2)
    assert layer.w2_weight_scale.shape == (4, 64, 2)
    assert layer.w13_weight_scale.dtype == torch.bfloat16
    assert layer.w2_weight_scale.dtype == torch.bfloat16
    assert values.hidden_states is not None
    assert values.hidden_states.shape == (4, 64)
    assert values.hidden_states.dtype == torch.bfloat16
    assert values.router_logits is not None
    assert values.router_logits.shape == (4, 4)
    assert values.topk_ids.shape == (4, 2)
    assert values.topk_ids.dtype == torch.int32
    assert moe_reference(values, output_dtype=torch.bfloat16).shape == (4, 64)


def test_dense_moe_generator_runs_flashinfer_cutlass_kernel(device: str) -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_hopper_plus:
        pytest.skip("dense FlashInfer MoE compatibility test requires NVIDIA Hopper+")

    load_builtin_kernels()
    config = MoeInputConfig(
        num_tokens=4,
        hidden_size=64,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        hidden_dtype=torch.bfloat16,
        router_dtype=torch.bfloat16,
        weight_format="dense",
        weight_dtype=torch.bfloat16,
        bias_dtype=None,
    )
    values = MoeInputs(config).generate(seed=43, device=device)
    layer = _make_dense_moe_weight_module(values)
    plan = moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="silu",
        internal_activation_dtype="input",
        with_bias=False,
        solution="flashinfer_cutlass",
    )
    assert plan["solution"] == "flashinfer_cutlass"
    assert plan["process_weights_kernel_name"] == (
        "flashinfer_cutlass_unquant_moe_process_weights"
    )

    moe_process_weights(plan, layer)
    actual = moe_apply(
        plan,
        values.hidden_states,
        layer,
        values.router_logits,
        topk_weights=values.topk_weights,
        topk_ids=values.topk_ids,
    )
    torch.cuda.synchronize()

    expected = moe_reference(values, output_dtype=torch.bfloat16).to(device=device)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5.0e-2, atol=0.1)
