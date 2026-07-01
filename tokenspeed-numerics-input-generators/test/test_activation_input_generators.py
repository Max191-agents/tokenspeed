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
from tokenspeed_numerics_input_generators import (
    FusedGateSigmoidMulAddInputConfig,
    FusedGateSigmoidMulAddInputs,
    FusedSwiGLUFP8UE8M0InputConfig,
    FusedSwiGLUFP8UE8M0Inputs,
    GatedActivationInputConfig,
    GatedActivationInputs,
    SigmoidMulInputConfig,
    SigmoidMulInputs,
)


def _sigmoid_mul_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    gate_dense = gate.reshape_as(x)
    return x.float() * gate_dense.float().sigmoid()


def _gated_activation_reference(
    x: torch.Tensor,
    *,
    activation: str,
) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    if activation == "silu":
        activated = F.silu(gate)
    elif activation == "gelu":
        activated = F.gelu(gate, approximate="none")
    elif activation == "gelu_tanh":
        activated = F.gelu(gate, approximate="tanh")
    else:
        raise AssertionError(f"unexpected activation={activation!r}")
    return activated * up


def _fused_gate_reference(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    gate = (hidden_states.float() @ gate_weight.float().unsqueeze(1)).sigmoid()
    return final_hidden_states.float() + gate * shared_output.float()


def test_sigmoid_mul_inputs_generate_dense_gate() -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=5,
            hidden_dim=16,
            dtype=torch.float32,
        )
    ).generate(seed=11, device="cpu")

    assert values.x.shape == (5, 16)
    assert values.gate.shape == (5, 16)
    assert values.gate_storage is None
    assert values.x.is_contiguous()
    assert values.gate.is_contiguous()
    ref = _sigmoid_mul_reference(values.x, values.gate)
    assert ref.shape == values.x.shape
    assert torch.isfinite(ref).all()


def test_sigmoid_mul_inputs_generate_qkv_split_gate_view() -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=3,
            hidden_dim=32,
            dtype=torch.float32,
            gate_layout="qkv_split",
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
        )
    ).generate(seed=12, device="cpu")

    assert values.x.shape == (3, 32)
    assert values.gate.shape == (3, 4, 8)
    assert values.gate_storage is not None
    assert values.gate.stride(-1) == 1
    assert not values.gate.is_contiguous()
    assert values.gate.stride(0) == values.gate_storage.shape[-1]
    ref = _sigmoid_mul_reference(values.x, values.gate)
    assert ref.shape == values.x.shape


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_gated_activation_inputs_generate_split_tensor(activation: str) -> None:
    values = GatedActivationInputs(
        GatedActivationInputConfig(
            num_tokens=7,
            hidden_dim=24,
            dtype=torch.float32,
            activation=activation,  # type: ignore[arg-type]
        )
    ).generate(seed=13, device="cpu")

    assert values.x.shape == (7, 48)
    ref = _gated_activation_reference(values.x, activation=activation)
    assert ref.shape == (7, 24)
    assert torch.isfinite(ref).all()


def test_fused_gate_sigmoid_mul_add_inputs_scale_gate_weight() -> None:
    config = FusedGateSigmoidMulAddInputConfig(
        num_tokens=11,
        hidden_dim=128,
        dtype=torch.float32,
    )
    values = FusedGateSigmoidMulAddInputs(config).generate(seed=14, device="cpu")

    assert values.hidden_states.shape == (11, 128)
    assert values.gate_weight.shape == (128,)
    assert values.shared_output.shape == (11, 128)
    assert values.final_hidden_states.shape == (11, 128)
    ref = _fused_gate_reference(
        values.hidden_states,
        values.gate_weight,
        values.shared_output,
        values.final_hidden_states,
    )
    assert ref.shape == (11, 128)

    gate = (values.hidden_states.float() @ values.gate_weight.float()).sigmoid()
    assert gate.min() > 0.01
    assert gate.max() < 0.99


def test_fused_swiglu_fp8_ue8m0_inputs_generate_grouped_gate_up() -> None:
    values = FusedSwiGLUFP8UE8M0Inputs(
        FusedSwiGLUFP8UE8M0InputConfig(
            num_tokens=4,
            hidden_dim=256,
            dtype=torch.bfloat16,
            swiglu_limit=7.0,
        )
    ).generate(seed=15, device="cpu")

    assert values.gate_up.shape == (4, 512)
    assert values.gate_up.dtype == torch.bfloat16
    assert values.swiglu_limit == 7.0
    gate, up = values.gate_up.float().chunk(2, dim=-1)
    y = F.silu(gate.clamp(max=7.0)) * up.clamp(min=-7.0, max=7.0)
    assert y.shape == (4, 256)
    assert torch.isfinite(y).all()


def test_sigmoid_mul_inputs_reject_invalid_qkv_split_config() -> None:
    with pytest.raises(ValueError, match="num_heads \\* head_dim == hidden_dim"):
        SigmoidMulInputs(
            SigmoidMulInputConfig(
                num_tokens=3,
                hidden_dim=31,
                dtype=torch.float32,
                gate_layout="qkv_split",
                num_heads=4,
                head_dim=8,
                num_kv_heads=2,
            )
        )


def test_activation_inputs_reject_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match="floating torch dtype"):
        GatedActivationInputs(
            GatedActivationInputConfig(
                num_tokens=3,
                hidden_dim=16,
                dtype=torch.int32,
            )
        )


def test_fused_swiglu_fp8_ue8m0_rejects_incompatible_group_size() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be divisible"):
        FusedSwiGLUFP8UE8M0Inputs(
            FusedSwiGLUFP8UE8M0InputConfig(
                num_tokens=3,
                hidden_dim=192,
                dtype=torch.bfloat16,
                group_size=128,
            )
        )
