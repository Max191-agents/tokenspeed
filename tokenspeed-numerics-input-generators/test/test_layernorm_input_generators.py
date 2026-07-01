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
    FusedQKRMSNormRopeGateInputConfig,
    FusedQKRMSNormRopeGateInputs,
    ParallelRMSNormInputConfig,
    ParallelRMSNormInputs,
    QKRMSNormInputConfig,
    QKRMSNormInputs,
    RMSNormInputConfig,
    RMSNormInputs,
    build_rope_cos_sin_cache,
    fused_qk_rmsnorm_rope_gate_reference,
    gemma_rmsnorm_reference,
    qk_rmsnorm_reference,
    rmsnorm_reference,
)


def test_rmsnorm_inputs_generate_values_and_reference() -> None:
    values = RMSNormInputs(
        RMSNormInputConfig(
            num_tokens=5,
            hidden_dim=16,
            dtype=torch.bfloat16,
            with_residual=True,
        )
    ).generate(seed=11, device="cpu")

    assert values.x.shape == (5, 16)
    assert values.x.dtype == torch.bfloat16
    assert values.weight.shape == (16,)
    assert values.weight.dtype == torch.float32
    assert values.residual is not None
    assert values.residual.shape == values.x.shape

    out, residual_out = rmsnorm_reference(
        values.x,
        values.weight,
        1e-6,
        residual=values.residual,
    )
    assert out.shape == values.x.shape
    assert out.dtype == values.x.dtype
    torch.testing.assert_close(residual_out, (values.x + values.residual))


def test_gemma_rmsnorm_reference_uses_unit_offset_weight() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    weight = torch.zeros(4, dtype=torch.float32)

    ordinary = rmsnorm_reference(x, weight, 1e-6)
    gemma = gemma_rmsnorm_reference(x, weight, 1e-6)

    assert torch.count_nonzero(ordinary) == 0
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(gemma, expected)


def test_gemma_rmsnorm_reference_residual_returns_residual_sum() -> None:
    values = RMSNormInputs(
        RMSNormInputConfig(
            num_tokens=3,
            hidden_dim=8,
            dtype=torch.float16,
            with_residual=True,
        )
    ).generate(seed=15, device="cpu")
    assert values.residual is not None

    out, residual_out = gemma_rmsnorm_reference(
        values.x,
        values.weight,
        1e-6,
        residual=values.residual,
    )

    assert out.shape == values.x.shape
    assert out.dtype == values.x.dtype
    torch.testing.assert_close(residual_out, values.x + values.residual)


@pytest.mark.parametrize("input_layout", ["dense", "qkv_split"])
def test_qk_rmsnorm_inputs_generate_layouts(input_layout: str) -> None:
    values = QKRMSNormInputs(
        QKRMSNormInputConfig(
            num_tokens=7,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float16,
            input_layout=input_layout,  # type: ignore[arg-type]
        )
    ).generate(seed=12, device="cpu")

    assert values.q.shape == (7, 32)
    assert values.k.shape == (7, 16)
    assert values.q_weight.shape == (8,)
    assert values.k_weight.shape == (8,)
    assert values.q.stride(-1) == 1
    assert values.k.stride(-1) == 1
    if input_layout == "qkv_split":
        assert values.qkv_storage is not None
        assert not values.k.is_contiguous()

    q_ref, k_ref = qk_rmsnorm_reference(
        values.q,
        values.k,
        values.q_weight,
        values.k_weight,
        1e-6,
        head_dim=8,
    )
    assert q_ref.shape == values.q.shape
    assert k_ref.shape == values.k.shape
    assert q_ref.dtype == values.q.dtype
    assert k_ref.dtype == values.k.dtype


def test_fused_qk_rmsnorm_rope_gate_inputs_generate_metadata_and_reference() -> None:
    values = FusedQKRMSNormRopeGateInputs(
        FusedQKRMSNormRopeGateInputConfig(
            num_tokens=6,
            num_q_heads=3,
            num_kv_heads=1,
            head_dim=16,
            rotary_dim=8,
            max_position=64,
            dtype=torch.bfloat16,
        )
    ).generate(seed=13, device="cpu")

    assert values.q_gate.shape == (6, 96)
    assert values.k.shape == (6, 16)
    assert values.q_weight.shape == (16,)
    assert values.k_weight.shape == (16,)
    assert values.cos_sin_cache.shape == (64, 8)
    assert values.positions.shape == (6,)
    assert int(values.positions.min()) >= 0
    assert int(values.positions.max()) < 64

    q_ref, k_ref, gate_ref = fused_qk_rmsnorm_rope_gate_reference(
        values.q_gate,
        values.k,
        values.q_weight,
        values.k_weight,
        values.cos_sin_cache,
        values.positions,
        1e-6,
        num_q_heads=3,
        num_kv_heads=1,
        head_dim=16,
        rotary_dim=8,
    )
    assert q_ref.shape == (6, 48)
    assert k_ref.shape == (6, 16)
    assert gate_ref.shape == (6, 48)
    expected_gate = values.q_gate.view(6, 3, 32)[..., 16:].reshape(6, 48)
    torch.testing.assert_close(gate_ref, expected_gate)


def test_fused_qk_metadata_seed_controls_positions_independently() -> None:
    generator = FusedQKRMSNormRopeGateInputs(
        FusedQKRMSNormRopeGateInputConfig(
            num_tokens=6,
            num_q_heads=3,
            num_kv_heads=1,
            head_dim=16,
            rotary_dim=8,
            max_position=64,
            dtype=torch.bfloat16,
        )
    )

    values1 = generator.generate(seed=13, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=14, metadata_seed=99, device="cpu")

    torch.testing.assert_close(values1.positions, values2.positions)
    assert not torch.equal(values1.q_gate, values2.q_gate)


def test_parallel_rmsnorm_inputs_generate_two_independent_contracts() -> None:
    values = ParallelRMSNormInputs(
        ParallelRMSNormInputConfig(
            num_tokens=4,
            hidden_dim1=8,
            hidden_dim2=12,
            dtype=torch.float32,
        )
    ).generate(seed=14, device="cpu")

    assert values.input1.shape == values.output1.shape == (4, 8)
    assert values.input2.shape == values.output2.shape == (4, 12)
    ref1 = rmsnorm_reference(values.input1, values.weight1, 1e-6)
    ref2 = rmsnorm_reference(values.input2, values.weight2, 1e-6)
    assert ref1.shape == values.output1.shape
    assert ref2.shape == values.output2.shape


def test_build_rope_cos_sin_cache_shape_and_values() -> None:
    cache = build_rope_cos_sin_cache(
        rotary_dim=4,
        max_position=3,
        base=10000.0,
        device="cpu",
    )
    assert cache.shape == (3, 4)
    torch.testing.assert_close(cache[0], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_rmsnorm_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="regular floating"):
        RMSNormInputs(
            RMSNormInputConfig(
                num_tokens=1,
                hidden_dim=8,
                dtype=torch.float8_e4m3fn,
            )
        )


def test_qk_rmsnorm_rejects_bad_layout() -> None:
    with pytest.raises(ValueError, match="unsupported input_layout"):
        QKRMSNormInputs(
            QKRMSNormInputConfig(
                num_tokens=1,
                num_q_heads=1,
                num_kv_heads=1,
                head_dim=8,
                dtype=torch.float16,
                input_layout="packed",  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("bad_rotary_dim", [0, 3, 32])
def test_fused_qk_rejects_invalid_rotary_dim(bad_rotary_dim: int) -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        FusedQKRMSNormRopeGateInputs(
            FusedQKRMSNormRopeGateInputConfig(
                num_tokens=1,
                num_q_heads=1,
                num_kv_heads=1,
                head_dim=16,
                rotary_dim=bad_rotary_dim,
                dtype=torch.float16,
            )
        )


def test_qk_reference_rejects_incompatible_head_dim() -> None:
    x = torch.randn(2, 7)
    weight = torch.ones(4)
    with pytest.raises(ValueError, match="divisible"):
        qk_rmsnorm_reference(x, x, weight, weight, 1e-6, head_dim=4)
