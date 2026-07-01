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
from tokenspeed_kernel import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.numerics.inputs import get_input_generator
from tokenspeed_kernel.numerics.moe import canonicalize_align_block_size
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import load_builtin_kernels
from tokenspeed_numerics_input_generators import (
    CustomDType,
    MoeAlignBlockSizeInputValues,
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
    canonicalize_moe_align_block_size,
    moe_align_block_size_reference,
    moe_reference,
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
    torch.testing.assert_close(
        canonicalize_align_block_size(
            ref.sorted_token_ids,
            ref.expert_ids,
            ref.num_tokens_post_pad,
            values.block_size,
        ),
        canonicalize_moe_align_block_size(ref, block_size=values.block_size),
        atol=0,
        rtol=0,
    )


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
