from __future__ import annotations

import pytest
import torch
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
    fused_gate_sigmoid_mul_add_reference,
    gated_activation_reference,
    sigmoid_mul_reference,
)

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton activation tests require an NVIDIA or AMD GPU.",
)


def _activation_tol(dtype: torch.dtype) -> float:
    return 1e-2 if dtype == torch.bfloat16 else 5e-3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    # Qwen3.5 attn_output_gate decode shapes (num_tokens, num_heads * head_dim).
    [(1, 4096), (17, 6144), (128, 4096), (256, 8192)],
)
def test_sigmoid_mul_matches_eager(
    dtype: torch.dtype, shape: tuple[int, int], device: str
) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=shape[0],
            hidden_dim=shape[1],
            dtype=dtype,
        )
    ).generate(seed=shape[0] * 101 + shape[1], device=device)
    ref = sigmoid_mul_reference(values.x, values.gate)

    out = sigmoid_mul(values.x.clone(), values.gate)

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_sigmoid_mul_is_inplace(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=8,
            hidden_dim=256,
            dtype=torch.bfloat16,
        )
    ).generate(seed=11, device=device)
    same = sigmoid_mul(values.x, values.gate)
    assert same.data_ptr() == values.x.data_ptr()


def test_sigmoid_mul_empty(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=0,
            hidden_dim=256,
            dtype=torch.bfloat16,
        )
    ).generate(seed=12, device=device)
    out = sigmoid_mul(values.x, values.gate)
    assert out.shape == values.x.shape


def test_sigmoid_mul_rejects_shape_mismatch(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=4,
            hidden_dim=32,
            dtype=torch.bfloat16,
        )
    ).generate(seed=13, device=device)
    gate = values.gate[:, :16].contiguous()
    with pytest.raises(ValueError, match="shape mismatch"):
        sigmoid_mul(values.x, gate)


def test_sigmoid_mul_rejects_dtype_mismatch(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=4,
            hidden_dim=32,
            dtype=torch.bfloat16,
        )
    ).generate(seed=14, device=device)
    gate = values.gate.to(torch.float16)
    with pytest.raises(ValueError, match="dtype mismatch"):
        sigmoid_mul(values.x, gate)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim",
    # qwen3.5 attn_output_gate variants: q=16/kv=2/d=256 (base default) plus
    # head_dim=128 fall-backs.
    [(16, 2, 256), (32, 8, 128), (40, 8, 128), (48, 8, 128)],
)
def test_sigmoid_mul_strided_gate_from_qkv_split(
    dtype: torch.dtype,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
) -> None:
    """Runtime path: gate is the [T, H, D] strided view obtained via
    ``qkv.split`` → ``.view(T, H, 2*D)`` → ``torch.chunk(q_gate, 2, dim=-1)``.
    ``gate.stride(0)`` is the full qkv row width (q_size*2 + 2*kv_size),
    not just H*2*D. The kernel must read this strided view directly without
    a contiguous copy."""
    num_tokens = 19
    q_size = num_heads * head_dim
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=num_tokens,
            hidden_dim=q_size,
            dtype=dtype,
            gate_layout="qkv_split",
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
    ).generate(seed=num_heads * 1000 + num_kv_heads * 100 + head_dim, device=device)
    # Lock in the production-shape stride: row stride is the full qkv width.
    assert values.gate_storage is not None
    assert not values.gate.is_contiguous()
    assert values.gate.stride(0) == values.gate_storage.stride(0)
    assert values.gate.stride(-1) == 1

    ref = sigmoid_mul_reference(values.x, values.gate)

    out = sigmoid_mul(values.x.clone(), values.gate)

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_sigmoid_mul_rejects_4d_gate(device: str) -> None:
    values = SigmoidMulInputs(
        SigmoidMulInputConfig(
            num_tokens=4,
            hidden_dim=32,
            dtype=torch.bfloat16,
        )
    ).generate(seed=15, device=device)
    gate = values.gate.reshape(4, 2, 4, 4)
    with pytest.raises(ValueError, match="gate must be 2D or 3D"):
        sigmoid_mul(values.x, gate)


# --- silu_and_mul tests ---


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 7168 * 2), (17, 1024), (128, 9216 * 2)])
def test_silu_and_mul_matches_eager(
    dtype: torch.dtype, shape: tuple[int, int], device: str
) -> None:
    values = GatedActivationInputs(
        GatedActivationInputConfig(
            num_tokens=shape[0],
            hidden_dim=shape[1] // 2,
            dtype=dtype,
            activation="silu",
        )
    ).generate(seed=shape[0] * 103 + shape[1], device=device)
    ref = gated_activation_reference(values.x, activation="silu")

    out = silu_and_mul(values.x)

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_silu_and_mul_writes_provided_output(device: str) -> None:
    values = GatedActivationInputs(
        GatedActivationInputConfig(
            num_tokens=8,
            hidden_dim=256,
            dtype=torch.bfloat16,
            activation="silu",
        )
    ).generate(seed=21, device=device)
    out = torch.empty(8, 256, device=device, dtype=torch.bfloat16)
    same = silu_and_mul(values.x, out)
    assert same.data_ptr() == out.data_ptr()


def test_silu_and_mul_empty(device: str) -> None:
    values = GatedActivationInputs(
        GatedActivationInputConfig(
            num_tokens=0,
            hidden_dim=256,
            dtype=torch.bfloat16,
            activation="silu",
        )
    ).generate(seed=22, device=device)
    out = silu_and_mul(values.x)
    assert out.shape == (0, 256)


def test_silu_and_mul_rejects_bad_output_shape(device: str) -> None:
    values = GatedActivationInputs(
        GatedActivationInputConfig(
            num_tokens=4,
            hidden_dim=256,
            dtype=torch.bfloat16,
            activation="silu",
        )
    ).generate(seed=23, device=device)
    out = torch.empty(4, 128, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out shape"):
        silu_and_mul(values.x, out)


# --- fused_gate_sigmoid_mul_add tests ---


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    [(1, 3584), (1, 5120), (17, 3584), (128, 5120), (256, 3584)],
)
def test_fused_gate_sigmoid_mul_add_matches_eager(
    dtype: torch.dtype, num_tokens: int, hidden_dim: int, device: str
) -> None:
    values = FusedGateSigmoidMulAddInputs(
        FusedGateSigmoidMulAddInputConfig(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )
    ).generate(seed=num_tokens * 107 + hidden_dim, device=device)
    ref = fused_gate_sigmoid_mul_add_reference(values)

    out = fused_gate_sigmoid_mul_add(
        values.hidden_states,
        values.gate_weight,
        values.shared_output.clone(),
        values.final_hidden_states.clone(),
    )

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_fused_gate_sigmoid_mul_add_is_inplace(device: str) -> None:
    values = FusedGateSigmoidMulAddInputs(
        FusedGateSigmoidMulAddInputConfig(
            num_tokens=8,
            hidden_dim=256,
            dtype=torch.bfloat16,
        )
    ).generate(seed=31, device=device)

    result = fused_gate_sigmoid_mul_add(
        values.hidden_states,
        values.gate_weight,
        values.shared_output,
        values.final_hidden_states,
    )
    assert result.data_ptr() == values.final_hidden_states.data_ptr()


def test_fused_gate_sigmoid_mul_add_empty(device: str) -> None:
    values = FusedGateSigmoidMulAddInputs(
        FusedGateSigmoidMulAddInputConfig(
            num_tokens=0,
            hidden_dim=256,
            dtype=torch.bfloat16,
        )
    ).generate(seed=32, device=device)

    out = fused_gate_sigmoid_mul_add(
        values.hidden_states,
        values.gate_weight,
        values.shared_output,
        values.final_hidden_states,
    )
    assert out.shape == (0, 256)
