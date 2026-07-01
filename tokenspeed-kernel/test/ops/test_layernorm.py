from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.triton import (
    fused_qk_rmsnorm_rope_gate,
    qk_rmsnorm,
    rmsnorm,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    FusedQKRMSNormRopeGateInputConfig,
    FusedQKRMSNormRopeGateInputs,
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
    reason="Triton layernorm tests require an NVIDIA or AMD GPU.",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 2880])
def test_rmsnorm(dtype: torch.dtype, hidden_size: int, device: str) -> None:
    eps = 1e-6
    values = RMSNormInputs(
        RMSNormInputConfig(
            num_tokens=7,
            hidden_dim=hidden_size,
            dtype=dtype,
            eps=eps,
        )
    ).generate(seed=41, device=device)

    out = rmsnorm(values.x, values.weight, eps)
    ref = rmsnorm_reference(values.x, values.weight, eps)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 2880])
def test_rmsnorm_with_residual(
    dtype: torch.dtype, hidden_size: int, device: str
) -> None:
    eps = 1e-6
    values = RMSNormInputs(
        RMSNormInputConfig(
            num_tokens=7,
            hidden_dim=hidden_size,
            dtype=dtype,
            eps=eps,
            with_residual=True,
        )
    ).generate(seed=42, device=device)
    assert values.residual is not None

    out, residual_out = rmsnorm(
        values.x,
        values.weight,
        eps,
        residual=values.residual,
    )
    ref, ref_residual = rmsnorm_reference(
        values.x,
        values.weight,
        eps,
        residual=values.residual,
    )

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual_out, ref_residual, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim",
    # qwen3_5_text_base_config defaults: q=16/kv=2/d=256.
    # Variants cover wider q/kv ratios and the head_dim=128 fall-back.
    [(16, 2, 256), (32, 8, 128), (28, 4, 128), (40, 8, 128)],
)
def test_qk_rmsnorm_gemma_weight_matches_two_calls(
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
) -> None:
    eps = 1e-6
    values = QKRMSNormInputs(
        QKRMSNormInputConfig(
            num_tokens=17,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            eps=eps,
        )
    ).generate(seed=num_q_heads * 1000 + num_kv_heads * 100 + head_dim, device=device)

    q_out, k_out = qk_rmsnorm(
        values.q,
        values.k,
        values.q_weight,
        values.k_weight,
        eps,
    )
    q_ref, k_ref = qk_rmsnorm_reference(
        values.q,
        values.k,
        values.q_weight,
        values.k_weight,
        eps,
        head_dim=head_dim,
    )

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


def test_qk_rmsnorm_gemma_weight_strided_qkv_split(device: str) -> None:
    """Runtime path: q and k arrive as strided views from a packed qkv split.
    The kernel's stride-aware addressing must handle the non-contiguous
    leading-axis case without needing a ``.contiguous()`` copy."""
    num_q_heads, num_kv_heads, head_dim = 16, 2, 256
    dtype = torch.bfloat16
    eps = 1e-6
    values = QKRMSNormInputs(
        QKRMSNormInputConfig(
            num_tokens=19,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            eps=eps,
            input_layout="qkv_split",
        )
    ).generate(seed=43, device=device)
    assert values.qkv_storage is not None
    # Sanity: the views must share storage with qkv and be non-contiguous so we
    # actually exercise the strided path.
    assert values.q.data_ptr() == values.qkv_storage.data_ptr()
    assert values.q.stride(0) == values.qkv_storage.stride(0)
    assert not values.q.is_contiguous()
    assert not values.k.is_contiguous()

    q_out, k_out = qk_rmsnorm(
        values.q,
        values.k,
        values.q_weight,
        values.k_weight,
        eps,
    )
    q_ref, k_ref = qk_rmsnorm_reference(
        values.q,
        values.k,
        values.q_weight,
        values.k_weight,
        eps,
        head_dim=head_dim,
    )

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,rotary_dim",
    # Qwen3.5 production: head_dim=256, partial_rotary_factor=0.25 → rotary_dim=64.
    # Other rows pin down full RoPE and an aggressive partial setting.
    [
        (16, 2, 256, 64),
        (16, 2, 256, 256),
        (32, 8, 128, 128),
        (28, 4, 128, 32),
    ],
)
def test_fused_qk_rmsnorm_rope_gate_matches_reference(
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    device: str,
) -> None:
    """Fused kernel must match the unfused (qk_rmsnorm + apply_rope + gate split) path.

    Regression test: an earlier version of the fused kernel hard-coded
    ``rotary_dim == head_dim`` and silently corrupted both q/k and the
    cos/sin cache reads on Qwen3.5 (rotary_dim=64, head_dim=256), tanking
    the MTP speculative-decode acceptance rate.
    """
    eps = 1e-6
    config = FusedQKRMSNormRopeGateInputConfig(
        num_tokens=23,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=1024,
        dtype=dtype,
        eps=eps,
    )
    values = FusedQKRMSNormRopeGateInputs(config).generate(
        seed=num_q_heads * 1000 + num_kv_heads * 100 + head_dim + rotary_dim,
        metadata_seed=45,
        device=device,
    )

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        values.q_gate,
        values.k,
        values.q_weight,
        values.k_weight,
        values.cos_sin_cache,
        values.positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )

    q_ref, k_ref, gate_ref = fused_qk_rmsnorm_rope_gate_reference(
        values.q_gate,
        values.k,
        values.q_weight,
        values.k_weight,
        values.cos_sin_cache,
        values.positions,
        eps,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
    )

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)
    # Gate is a verbatim copy of the second-half slice of q_gate.
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


def test_fused_qk_rmsnorm_rope_gate_empty(device: str) -> None:
    """Zero-token batch returns correctly shaped empty tensors without launching."""
    head_dim, rotary_dim = 256, 64
    num_q_heads, num_kv_heads = 4, 1
    values = FusedQKRMSNormRopeGateInputs(
        FusedQKRMSNormRopeGateInputConfig(
            num_tokens=0,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=16,
            dtype=torch.bfloat16,
        )
    ).generate(seed=46, metadata_seed=47, device=device)

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        values.q_gate,
        values.k,
        values.q_weight,
        values.k_weight,
        values.cos_sin_cache,
        values.positions,
        1e-6,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
    assert q_out.shape == (0, num_q_heads * head_dim)
    assert k_out.shape == (0, num_kv_heads * head_dim)
    assert gate_out.shape == (0, num_q_heads * head_dim)


@pytest.mark.parametrize("bad_rotary_dim", [0, -2, 65, 33])
def test_fused_qk_rmsnorm_rope_gate_rejects_invalid_rotary_dim(
    bad_rotary_dim: int, device: str
) -> None:
    """rotary_dim must be a positive even integer <= head_dim."""
    head_dim = 64
    num_q_heads, num_kv_heads, n = 2, 1, 3
    values = FusedQKRMSNormRopeGateInputs(
        FusedQKRMSNormRopeGateInputConfig(
            num_tokens=n,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=16,
            dtype=torch.bfloat16,
        )
    ).generate(seed=49, metadata_seed=50, device=device)
    with pytest.raises(ValueError, match="rotary_dim"):
        fused_qk_rmsnorm_rope_gate(
            values.q_gate,
            values.k,
            values.q_weight,
            values.k_weight,
            values.cos_sin_cache,
            values.positions,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
            bad_rotary_dim,
        )


def test_rmsnorm_inplace(device: str) -> None:
    eps = 1e-6
    values = RMSNormInputs(
        RMSNormInputConfig(
            num_tokens=7,
            hidden_dim=128,
            dtype=torch.bfloat16,
            eps=eps,
        )
    ).generate(seed=48, device=device)
    x_ref = values.x.clone()

    out = rmsnorm(values.x, values.weight, eps, out=values.x)
    ref = rmsnorm_reference(x_ref, values.weight, eps)

    assert out.data_ptr() == values.x.data_ptr()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
