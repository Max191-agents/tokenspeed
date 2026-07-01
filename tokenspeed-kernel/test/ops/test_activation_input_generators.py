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
import torch.nn.functional as F
from tokenspeed_kernel.ops.activation.triton import (
    fused_gate_sigmoid_mul_add,
    sigmoid_mul,
    silu_and_mul,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    FusedGateSigmoidMulAddInputConfig,
    FusedGateSigmoidMulAddInputs,
    GatedActivationInputConfig,
    GatedActivationInputs,
    SigmoidMulInputConfig,
    SigmoidMulInputs,
)

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton activation generator smoke tests require an NVIDIA or AMD GPU.",
)


def _sigmoid_mul_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x.float() * gate.reshape_as(x).float().sigmoid()


def _silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    return F.silu(gate) * up


def _fused_gate_reference(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    gate = (hidden_states.float() @ gate_weight.float().unsqueeze(1)).sigmoid()
    return final_hidden_states.float() + gate * shared_output.float()


def test_sigmoid_mul_generator_runs_kernel_dense(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=19,
            hidden_dim=1024,
            dtype=torch.bfloat16,
        )
    ).generate(seed=101, device=device)

    out = sigmoid_mul(values.x.clone(), values.gate)
    ref = _sigmoid_mul_reference(values.x, values.gate)

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
    ref = _sigmoid_mul_reference(values.x, values.gate)

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
    ref = _silu_and_mul_reference(values.x)

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
    ref = _fused_gate_reference(
        values.hidden_states,
        values.gate_weight,
        values.shared_output,
        values.final_hidden_states,
    )

    assert out.data_ptr() == final.data_ptr()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
