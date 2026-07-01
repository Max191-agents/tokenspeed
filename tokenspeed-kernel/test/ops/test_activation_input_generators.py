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
from tokenspeed_kernel.ops.activation.triton import (
    fused_gate_sigmoid_mul_add,
    fused_swiglu_fp8_ue8m0,
    sigmoid_mul,
    silu_and_mul,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    FusedGateSigmoidMulAddInputConfig,
    FusedGateSigmoidMulAddInputs,
    FusedSwiGLUFP8UE8M0InputConfig,
    FusedSwiGLUFP8UE8M0Inputs,
    GatedActivationInputConfig,
    GatedActivationInputs,
    SigmoidMulInputConfig,
    SigmoidMulInputs,
    fused_gate_sigmoid_mul_add_reference,
    fused_swiglu_fp8_ue8m0_reference,
    gated_activation_reference,
    sigmoid_mul_reference,
)

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton activation generator smoke tests require an NVIDIA or AMD GPU.",
)


def test_sigmoid_mul_generator_runs_kernel_dense(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=19,
            hidden_dim=1024,
            dtype=torch.bfloat16,
        )
    ).generate(seed=101, device=device)

    out = sigmoid_mul(values.x.clone(), values.gate)
    ref = sigmoid_mul_reference(values.x, values.gate)

    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_sigmoid_mul_generator_runs_kernel_qkv_split(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=17,
            hidden_dim=16 * 128,
            dtype=torch.bfloat16,
            gate_layout="qkv_split",
            num_heads=16,
            num_kv_heads=2,
            head_dim=128,
        )
    ).generate(seed=102, device=device)

    out = sigmoid_mul(values.x.clone(), values.gate)
    ref = sigmoid_mul_reference(values.x, values.gate)

    assert not values.gate.is_contiguous()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_silu_and_mul_generator_runs_kernel(device: str) -> None:
    values = GatedActivationInputs(
        GatedActivationInputConfig(
            num_tokens=23,
            hidden_dim=2048,
            dtype=torch.bfloat16,
            activation="silu",
        )
    ).generate(seed=103, device=device)

    out = silu_and_mul(values.x)
    ref = gated_activation_reference(values.x, activation="silu")

    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_fused_gate_sigmoid_mul_add_generator_runs_kernel(device: str) -> None:
    values = FusedGateSigmoidMulAddInputs(
        FusedGateSigmoidMulAddInputConfig(
            num_tokens=13,
            hidden_dim=512,
            dtype=torch.bfloat16,
        )
    ).generate(seed=104, device=device)
    final = values.final_hidden_states.clone()

    out = fused_gate_sigmoid_mul_add(
        values.hidden_states,
        values.gate_weight,
        values.shared_output,
        final,
    )
    ref = fused_gate_sigmoid_mul_add_reference(values)

    assert out.data_ptr() == final.data_ptr()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_fused_swiglu_fp8_ue8m0_generator_runs_kernel(device: str) -> None:
    values = FusedSwiGLUFP8UE8M0Inputs(
        FusedSwiGLUFP8UE8M0InputConfig(
            num_tokens=5,
            hidden_dim=256,
            dtype=torch.bfloat16,
            swiglu_limit=7.0,
        )
    ).generate(seed=105, device=device)
    expected_q, expected_scales = fused_swiglu_fp8_ue8m0_reference(values)

    actual_q, actual_scales = fused_swiglu_fp8_ue8m0(
        values.gate_up,
        values.swiglu_limit,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_scales, expected_scales)
