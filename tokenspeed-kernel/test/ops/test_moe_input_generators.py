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
    MoeDispatchInputConfig,
    MoeDispatchInputs,
    MoeDownCombineInputConfig,
    MoeDownCombineInputs,
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
    MoeRoutingInputConfig,
    MoeRoutingInputs,
)


def _make_mxfp4_moe_weight_module(values: MoeInputValues) -> torch.nn.Module:
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
    layer.top_k = values.routing.topk_ids.shape[1]
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
    layer.top_k = values.routing.topk_ids.shape[1]
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

    topk_ids = generated["topk_ids"]
    block_size = generated["block_size"]
    num_experts = generated["num_experts"]
    ref = moe_align_block_size_reference(
        topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )

    assert topk_ids.shape == (7, 2)
    assert topk_ids.dtype == torch.int32
    assert torch.all(topk_ids >= 0)
    assert torch.all(topk_ids < 5)
    assert ref.sorted_token_ids.numel() % block_size == 0
    assert ref.expert_ids.numel() * block_size == ref.sorted_token_ids.numel()
    canonical = canonicalize_align_block_size(
        ref.sorted_token_ids,
        ref.expert_ids,
        ref.num_tokens_post_pad,
        block_size,
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

    num_experts_real = 256
    scaling_factor = 6.0
    renormalize = False
    values = MoeRoutingInputs(
        MoeRoutingInputConfig(
            num_tokens=8,
            hidden_size=16,
            num_experts=384,
            top_k=12,
            hidden_dtype=torch.float32,
            correction_bias_dtype=torch.float32,
            renormalize=renormalize,
            routed_scaling_factor=scaling_factor,
        )
    ).generate(seed=45, device="cuda")
    assert values.correction_bias is not None
    expected = moe_softmax_topk_routing_reference(
        values.router_logits,
        values.correction_bias,
        top_k=12,
        num_experts_real=num_experts_real,
        scaling_factor=scaling_factor,
        renormalize=renormalize,
    )

    try:
        routing_flash(
            values.router_logits,
            values.correction_bias,
            values.topk_ids,
            values.topk_weights,
            num_experts_real,
            scaling_factor,
            renormalize,
        )
    except RuntimeError as exc:
        pytest.skip(f"routing_flash extension unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(values.topk_ids, expected.topk_indices)
    torch.testing.assert_close(
        values.topk_weights,
        expected.topk_weights,
        rtol=1.0e-3,
        atol=8.0e-2,
    )


def test_moe_biased_grouped_topk_generator_matches_triton_fallback() -> None:
    from tokenspeed_kernel.thirdparty.triton import minimax_biased_grouped_topk

    values = MoeRoutingInputs(
        MoeRoutingInputConfig(
            num_tokens=5,
            hidden_size=8,
            num_experts=8,
            top_k=3,
            hidden_dtype=torch.float32,
            score_function="sigmoid",
            correction_bias_dtype=torch.float32,
            num_expert_groups=2,
            top_k_groups=1,
            renormalize=True,
            routed_scaling_factor=2.0,
        )
    ).generate(seed=46, device="cpu")
    assert values.correction_bias is not None
    logical_to_physical_map = torch.randperm(8, dtype=torch.int32)
    num_token_non_padded = torch.tensor(4, dtype=torch.int32)
    expected = moe_biased_grouped_topk_reference(
        values.router_logits,
        values.correction_bias,
        top_k=3,
        renormalize=True,
        num_expert_groups=2,
        top_k_groups=1,
        routed_scaling_factor=2.0,
        logical_to_physical_map=logical_to_physical_map,
        num_token_non_padded=num_token_non_padded,
    )

    actual_weights, actual_ids = minimax_biased_grouped_topk(
        values.hidden_states,
        values.router_logits,
        values.correction_bias,
        topk=3,
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=2.0,
        num_token_non_padded=num_token_non_padded,
        logical_to_physical_map=logical_to_physical_map,
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

    non_hash = MoeRoutingInputs(
        MoeRoutingInputConfig(
            num_tokens=8,
            hidden_size=16,
            num_experts=256,
            top_k=6,
            hidden_dtype=torch.float32,
            score_function="softplus_sqrt",
            correction_bias_dtype=torch.float32,
            routed_scaling_factor=1.75,
        )
    ).generate(seed=47, device="cuda")
    assert non_hash.correction_bias is not None
    non_hash_expected = moe_softplus_sqrt_topk_routing_reference(
        non_hash.router_logits,
        top_k=6,
        routed_scaling_factor=1.75,
        correction_bias=non_hash.correction_bias,
    )
    try:
        softplus_sqrt_topk_flash(
            non_hash.router_logits,
            non_hash.correction_bias,
            non_hash.topk_ids,
            non_hash.topk_weights,
            1.75,
            True,
        )
    except RuntimeError as exc:
        pytest.skip(f"softplus_sqrt_topk_flash extension unavailable: {exc}")

    hashed = MoeRoutingInputs(
        MoeRoutingInputConfig(
            num_tokens=8,
            hidden_size=16,
            num_experts=256,
            top_k=6,
            hidden_dtype=torch.float32,
            score_function="softplus_sqrt",
            routed_scaling_factor=2.25,
        )
    ).generate(seed=48, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(48)
    input_ids = torch.randint(
        0,
        16,
        (8,),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    hash_indices_table = torch.stack(
        [
            torch.randperm(256, dtype=torch.int32, device="cuda", generator=generator)[
                :6
            ]
            for _ in range(16)
        ]
    )
    hashed_expected = moe_softplus_sqrt_topk_routing_reference(
        hashed.router_logits,
        top_k=6,
        routed_scaling_factor=2.25,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )
    try:
        hash_softplus_sqrt_topk_flash(
            hashed.router_logits,
            input_ids,
            hash_indices_table,
            hashed.topk_ids,
            hashed.topk_weights,
            2.25,
            True,
        )
    except RuntimeError as exc:
        pytest.skip(f"hash_softplus_sqrt_topk_flash extension unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(non_hash.topk_ids, non_hash_expected.topk_indices)
    torch.testing.assert_close(
        non_hash.topk_weights,
        non_hash_expected.topk_weights,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    torch.testing.assert_close(hashed.topk_ids, hashed_expected.topk_indices)
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

    values = MoeDispatchInputs(
        MoeDispatchInputConfig(
            num_tokens=4,
            hidden_size=128,
            num_experts=8,
            top_k=3,
            hidden_dtype=torch.bfloat16,
        )
    ).generate(seed=49, device="cuda")
    x_fp8 = torch.empty_like(values.hidden_states, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty((4, 1), dtype=torch.int32, device="cuda")
    topk_idx_out = torch.empty_like(values.topk_ids)
    topk_weights_out = torch.empty_like(values.topk_weights)
    expected = moe_deepseek_v4_mega_moe_staging_reference(
        values.hidden_states,
        values.topk_ids,
        values.topk_weights,
    )

    try:
        stage_deepseek_v4_mega_moe_inputs(
            values.hidden_states,
            values.topk_weights,
            values.topk_ids,
            x_fp8,
            x_sf,
            topk_idx_out,
            topk_weights_out,
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"stage_deepseek_v4_mega_moe_inputs unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(
        x_fp8.float(),
        expected.x_fp8.float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(x_sf, expected.x_sf)
    torch.testing.assert_close(topk_idx_out, expected.topk_idx_out)
    torch.testing.assert_close(topk_weights_out, expected.topk_weights_out)


def test_moe_finalize_fuse_shared_generator_runs_cuda_helper() -> None:
    platform = current_platform()
    if (
        not torch.cuda.is_available()
        or not platform.is_nvidia
        or not platform.is_hopper_plus
    ):
        pytest.skip("moe_finalize_fuse_shared compatibility test requires NVIDIA SM90+")

    from tokenspeed_kernel.thirdparty.cuda import moe_finalize_fuse_shared

    values = MoeDownCombineInputs(
        MoeDownCombineInputConfig(
            num_tokens=4,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            activation_dtype=torch.bfloat16,
            include_shared_output=True,
        )
    ).generate(seed=50, device="cuda")
    assert values.w2.B is not None
    expert_ids = values.topk_ids.reshape(-1).to(torch.long)
    selected_weights = values.w2.B[expert_ids].float()
    gemm2_out = (
        torch.bmm(
            selected_weights,
            values.activations.float().unsqueeze(-1),
        )
        .squeeze(-1)
        .to(torch.bfloat16)
    )
    expanded_idx_to_permuted_idx = torch.arange(
        gemm2_out.shape[0],
        dtype=torch.int32,
        device="cuda",
    )
    expected = moe_finalize_fuse_shared_reference(
        gemm2_out,
        expanded_idx_to_permuted_idx,
        values.topk_weights,
        values.shared_output,
    )

    try:
        actual = moe_finalize_fuse_shared(
            gemm2_out,
            expanded_idx_to_permuted_idx,
            values.topk_weights,
            values.shared_output,
            values.topk_weights.shape[1],
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
        routing=MoeRoutingInputConfig(
            num_tokens=4,
            hidden_size=64,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.bfloat16,
            router_dtype=torch.bfloat16,
            topk_weights_dtype=torch.bfloat16,
        ),
        intermediate_size=64,
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
    routing = values.routing
    actual = moe_apply(
        plan,
        routing.hidden_states,
        layer,
        routing.router_logits,
        topk_weights=routing.topk_weights,
        topk_ids=routing.topk_ids,
    )
    torch.cuda.synchronize()

    expected = moe_reference(values, output_dtype=torch.bfloat16).to(device=device)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5.0e-2, atol=0.1)


def test_mxint4_moe_generator_matches_flashinfer_trtllm_contract() -> None:
    config = MoeInputConfig(
        routing=MoeRoutingInputConfig(
            num_tokens=4,
            hidden_size=64,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.bfloat16,
            router_dtype=torch.float32,
            topk_weights_dtype=torch.bfloat16,
        ),
        intermediate_size=64,
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
    layer.num_experts = config.routing.num_experts
    layer.num_local_experts = config.routing.num_experts
    layer.ep_size = 1
    layer.ep_rank = 0
    layer.tp_size = 1
    layer.top_k = config.routing.top_k
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
    assert values.routing.hidden_states.shape == (4, 64)
    assert values.routing.hidden_states.dtype == torch.bfloat16
    assert values.routing.router_logits.shape == (4, 4)
    assert values.routing.topk_ids.shape == (4, 2)
    assert values.routing.topk_ids.dtype == torch.int32
    assert moe_reference(values, output_dtype=torch.bfloat16).shape == (4, 64)


def test_dense_moe_generator_runs_flashinfer_cutlass_kernel(device: str) -> None:
    platform = current_platform()
    if not torch.cuda.is_available() or not platform.is_hopper_plus:
        pytest.skip("dense FlashInfer MoE compatibility test requires NVIDIA Hopper+")

    load_builtin_kernels()
    config = MoeInputConfig(
        routing=MoeRoutingInputConfig(
            num_tokens=4,
            hidden_size=64,
            num_experts=4,
            top_k=2,
            hidden_dtype=torch.bfloat16,
            router_dtype=torch.bfloat16,
            topk_weights_dtype=torch.bfloat16,
        ),
        intermediate_size=64,
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
    routing = values.routing
    actual = moe_apply(
        plan,
        routing.hidden_states,
        layer,
        routing.router_logits,
        topk_weights=routing.topk_weights,
        topk_ids=routing.topk_ids,
    )
    torch.cuda.synchronize()

    expected = moe_reference(values, output_dtype=torch.bfloat16).to(device=device)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5.0e-2, atol=0.1)
