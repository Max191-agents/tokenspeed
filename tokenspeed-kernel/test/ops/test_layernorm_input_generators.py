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
from tokenspeed_kernel.ops.layernorm.flashinfer import (
    fused_add_rmsnorm as flashinfer_fused_add_rmsnorm,
)
from tokenspeed_kernel.ops.layernorm.flashinfer import rmsnorm as flashinfer_rmsnorm
from tokenspeed_kernel.ops.layernorm.triton import (
    fused_qk_rmsnorm_rope_gate,
    qk_rmsnorm,
    rmsnorm,
    rmsnorm_fused_parallel,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    FusedQKRMSNormRopeGateInputConfig,
    FusedQKRMSNormRopeGateInputs,
    ParallelRMSNormInputConfig,
    ParallelRMSNormInputs,
    QKRMSNormInputConfig,
    QKRMSNormInputs,
    RMSNormInputConfig,
    RMSNormInputs,
    fused_qk_rmsnorm_rope_gate_reference,
    qk_rmsnorm_reference,
    rmsnorm_reference,
)

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton layernorm generator smoke tests require an NVIDIA or AMD GPU.",
)


def test_rmsnorm_generator_runs_triton_residual_kernel(device: str) -> None:
    config = RMSNormInputConfig(
        num_tokens=9,
        hidden_dim=128,
        dtype=torch.bfloat16,
        with_residual=True,
    )
    values = RMSNormInputs(config).generate(seed=61, device=device)

    out, residual_out = rmsnorm(
        values.x,
        values.weight,
        config.eps,
        residual=values.residual,
    )
    ref_out, ref_residual = rmsnorm_reference(
        values.x,
        values.weight,
        config.eps,
        residual=values.residual,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual_out, ref_residual, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not platform.is_nvidia,
    reason="FlashInfer layernorm kernels require NVIDIA CUDA.",
)
def test_rmsnorm_generator_runs_flashinfer_kernel(device: str) -> None:
    config = RMSNormInputConfig(
        num_tokens=9,
        hidden_dim=128,
        dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
    )
    values = RMSNormInputs(config).generate(seed=65, device=device)

    try:
        out = flashinfer_rmsnorm(
            values.x,
            values.weight,
            config.eps,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer RMSNorm kernel unavailable: {exc}")
    ref = rmsnorm_reference(values.x, values.weight, config.eps)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not platform.is_nvidia,
    reason="FlashInfer layernorm kernels require NVIDIA CUDA.",
)
def test_rmsnorm_generator_runs_flashinfer_fused_add_kernel(device: str) -> None:
    config = RMSNormInputConfig(
        num_tokens=9,
        hidden_dim=128,
        dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        with_residual=True,
    )
    values = RMSNormInputs(config).generate(seed=66, device=device)
    assert values.residual is not None

    x = values.x.clone()
    residual = values.residual.clone()
    try:
        flashinfer_fused_add_rmsnorm(
            x,
            residual,
            values.weight,
            config.eps,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer fused add RMSNorm kernel unavailable: {exc}")
    ref_out, ref_residual = rmsnorm_reference(
        values.x,
        values.weight,
        config.eps,
        residual=values.residual,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(x, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual, ref_residual, atol=2e-2, rtol=2e-2)


def test_qk_rmsnorm_generator_runs_triton_strided_qkv_kernel(device: str) -> None:
    config = QKRMSNormInputConfig(
        num_tokens=11,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float16,
        input_layout="qkv_split",
    )
    values = QKRMSNormInputs(config).generate(seed=62, device=device)
    assert values.qkv_storage is not None
    assert not values.k.is_contiguous()

    q_out, k_out = qk_rmsnorm(
        values.q,
        values.k,
        values.q_weight,
        values.k_weight,
        config.eps,
    )
    q_ref, k_ref = qk_rmsnorm_reference(
        values.q,
        values.k,
        values.q_weight,
        values.k_weight,
        config.eps,
        head_dim=config.head_dim,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


def test_fused_qk_rmsnorm_rope_gate_generator_runs_triton_kernel(
    device: str,
) -> None:
    config = FusedQKRMSNormRopeGateInputConfig(
        num_tokens=13,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        rotary_dim=32,
        max_position=256,
        dtype=torch.bfloat16,
    )
    values = FusedQKRMSNormRopeGateInputs(config).generate(seed=63, device=device)

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        values.q_gate,
        values.k,
        values.q_weight,
        values.k_weight,
        values.cos_sin_cache,
        values.positions,
        config.eps,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rotary_dim,
    )
    q_ref, k_ref, gate_ref = fused_qk_rmsnorm_rope_gate_reference(
        values.q_gate,
        values.k,
        values.q_weight,
        values.k_weight,
        values.cos_sin_cache,
        values.positions,
        config.eps,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        rotary_dim=config.rotary_dim,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


def test_parallel_rmsnorm_generator_runs_triton_kernel(device: str) -> None:
    config = ParallelRMSNormInputConfig(
        num_tokens=7,
        hidden_dim1=128,
        hidden_dim2=256,
        dtype=torch.bfloat16,
    )
    values = ParallelRMSNormInputs(config).generate(seed=64, device=device)

    rmsnorm_fused_parallel(
        values.input1,
        values.weight1,
        values.output1,
        values.input2,
        values.weight2,
        values.output2,
        config.eps,
    )
    ref1 = rmsnorm_reference(values.input1, values.weight1, config.eps)
    ref2 = rmsnorm_reference(values.input2, values.weight2, config.eps)
    torch.cuda.synchronize()

    torch.testing.assert_close(values.output1, ref1, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(values.output2, ref2, atol=2e-2, rtol=2e-2)
