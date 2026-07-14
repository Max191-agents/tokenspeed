from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.ops.activation.cuda import (
    silu_and_mul_fuse_block_quant,
    silu_and_mul_fuse_nvfp4_quant,
)
from tokenspeed_kernel.ops.activation.flashinfer import (
    gelu_and_mul as flashinfer_gelu_and_mul,
)
from tokenspeed_kernel.ops.activation.flashinfer import (
    gelu_tanh_and_mul as flashinfer_gelu_tanh_and_mul,
)
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.activation.triton import (
    fused_gate_sigmoid_mul_add,
    fused_swiglu_fp8_ue8m0,
    sigmoid_mul,
    silu_and_mul,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators.quantization import (
    nvfp4_quantization_reference,
)

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton activation tests require an NVIDIA or AMD GPU.",
)


@dataclass
class _SigmoidMulValues:
    x: torch.Tensor
    gate: torch.Tensor
    gate_storage: torch.Tensor | None = None


@dataclass
class _GatedActivationValues:
    x: torch.Tensor


@dataclass
class _FusedGateSigmoidMulAddValues:
    hidden_states: torch.Tensor
    gate_weight: torch.Tensor
    shared_output: torch.Tensor
    final_hidden_states: torch.Tensor


@dataclass
class _FusedSwiGLUFP8BlockQuantValues:
    gate_up: torch.Tensor
    scale_out: torch.Tensor
    group_size: int
    num_tokens_per_expert: torch.Tensor | None = None
    num_tokens_hint: int | None = None
    num_experts: int | None = None


@dataclass
class _FusedSwiGLUNVFP4QuantValues:
    gate_up: torch.Tensor
    global_scale: torch.Tensor
    scale_size: int


def _randn(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> torch.Tensor:
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    values = torch.randn(
        shape,
        dtype=torch.float32,
        device=target_device,
        generator=generator,
    ).clamp(-3.0, 3.0)
    return values.to(dtype).contiguous()


def _activation_tol(dtype: torch.dtype) -> float:
    return 1e-2 if dtype == torch.bfloat16 else 5e-3


def _sigmoid_mul_values(
    *,
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    gate_layout: str = "dense",
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> _SigmoidMulValues:
    x = _randn((num_tokens, hidden_dim), dtype=dtype, device=device, seed=seed + 1)
    if gate_layout == "dense":
        gate = _randn(
            (num_tokens, hidden_dim),
            dtype=dtype,
            device=device,
            seed=seed + 2,
        )
        return _SigmoidMulValues(x=x, gate=gate)

    if gate_layout != "qkv_split":
        raise ValueError(f"unsupported gate_layout={gate_layout!r}")
    if num_heads is None or num_kv_heads is None or head_dim is None:
        raise ValueError("qkv_split layout requires num_heads, num_kv_heads, head_dim")
    if num_heads * head_dim != hidden_dim:
        raise ValueError("num_heads * head_dim must equal hidden_dim")

    q_size = hidden_dim
    kv_size = num_kv_heads * head_dim
    qkv = _randn(
        (num_tokens, 2 * q_size + 2 * kv_size),
        dtype=dtype,
        device=device,
        seed=seed + 2,
    )
    q_gate = qkv[:, : 2 * q_size].view(num_tokens, num_heads, 2 * head_dim)
    _q, gate = torch.chunk(q_gate, 2, dim=-1)
    return _SigmoidMulValues(x=x, gate=gate, gate_storage=qkv)


def _sigmoid_mul_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if gate.ndim == 3:
        gate = gate.reshape_as(x)
    return x.float() * gate.float().sigmoid()


def _gated_activation_values(
    *,
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> _GatedActivationValues:
    x = _randn(
        (num_tokens, 2 * hidden_dim),
        dtype=dtype,
        device=device,
        seed=seed + 1,
    )
    return _GatedActivationValues(x=x)


def _gated_activation_reference(
    x: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    if activation == "silu":
        activated = torch.nn.functional.silu(gate)
    elif activation == "gelu":
        activated = torch.nn.functional.gelu(gate)
    elif activation == "gelu_tanh":
        activated = torch.nn.functional.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"unsupported activation={activation!r}")
    return activated * up


def _fused_gate_sigmoid_mul_add_values(
    *,
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> _FusedGateSigmoidMulAddValues:
    hidden_states = _randn(
        (num_tokens, hidden_dim),
        dtype=dtype,
        device=device,
        seed=seed + 1,
    )
    gate_weight = _randn(
        (hidden_dim,),
        dtype=dtype,
        device=device,
        seed=seed + 2,
    )
    gate_weight = (gate_weight.float() / math.sqrt(hidden_dim)).to(dtype)
    shared_output = _randn(
        (num_tokens, hidden_dim),
        dtype=dtype,
        device=device,
        seed=seed + 3,
    )
    final_hidden_states = _randn(
        (num_tokens, hidden_dim),
        dtype=dtype,
        device=device,
        seed=seed + 4,
    )
    return _FusedGateSigmoidMulAddValues(
        hidden_states=hidden_states,
        gate_weight=gate_weight,
        shared_output=shared_output,
        final_hidden_states=final_hidden_states,
    )


def _fused_gate_sigmoid_mul_add_reference(
    values: _FusedGateSigmoidMulAddValues,
) -> torch.Tensor:
    gate = (
        values.hidden_states.float() @ values.gate_weight.float().unsqueeze(1)
    ).sigmoid()
    return values.final_hidden_states.float() + gate * values.shared_output.float()


def _fused_swiglu_fp8_ue8m0_reference(
    gate_up: torch.Tensor,
    *,
    swiglu_limit: float,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = gate_up.float().chunk(2, dim=-1)
    if swiglu_limit > 0.0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = torch.nn.functional.silu(gate) * up

    num_tokens, hidden_dim = activated.shape
    groups_per_row = hidden_dim // group_size
    grouped = activated.reshape(num_tokens, groups_per_row, group_size)
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    scale_raw = grouped.abs().amax(dim=-1) / fp8_info.max
    exponent = torch.ceil(torch.log2(scale_raw.clamp(min=1.0e-10)))
    scale = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=gate_up.device),
        exponent,
    )
    quantized = torch.clamp(
        grouped / scale.unsqueeze(-1),
        min=fp8_info.min,
        max=fp8_info.max,
    ).to(fp8_dtype)

    packed_scale_cols = (groups_per_row + 3) // 4
    packed_scales = torch.zeros(
        (num_tokens, packed_scale_cols),
        dtype=torch.int32,
        device=gate_up.device,
    )
    biased_exponent = (exponent + 127.0).clamp(min=0.0, max=255.0).to(torch.int32)
    for group_idx in range(groups_per_row):
        packed_col = group_idx // 4
        packed_pos = group_idx % 4
        packed_scales[:, packed_col] |= biased_exponent[:, group_idx] << (
            packed_pos * 8
        )

    return quantized.reshape(num_tokens, hidden_dim).contiguous(), packed_scales


def _fused_swiglu_fp8_block_quant_values(
    *,
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    group_size: int = 128,
    num_experts: int | None = None,
) -> _FusedSwiGLUFP8BlockQuantValues:
    gate_up_shape = (
        (num_experts, num_tokens, 2 * hidden_dim)
        if num_experts is not None
        else (num_tokens, 2 * hidden_dim)
    )
    gate_up = _randn(gate_up_shape, dtype=dtype, device=device, seed=seed + 1)
    scale_out = torch.zeros(
        (*gate_up_shape[:-1], hidden_dim // group_size),
        dtype=torch.float32,
        device=device,
    )
    num_tokens_per_expert = None
    num_tokens_hint = None
    if num_experts is not None:
        num_tokens_per_expert = torch.full(
            (num_experts,),
            num_tokens,
            dtype=torch.int32,
            device=device,
        )
        num_tokens_hint = num_tokens
    return _FusedSwiGLUFP8BlockQuantValues(
        gate_up=gate_up,
        scale_out=scale_out,
        group_size=group_size,
        num_tokens_per_expert=num_tokens_per_expert,
        num_tokens_hint=num_tokens_hint,
        num_experts=num_experts,
    )


def _fused_swiglu_fp8_block_quant_reference(
    values: _FusedSwiGLUFP8BlockQuantValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_dim = values.gate_up.shape[-1] // 2
    activated = _gated_activation_reference(values.gate_up, activation="silu")
    groups = activated.reshape(
        *activated.shape[:-1],
        hidden_dim // values.group_size,
        values.group_size,
    )
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    scales = (groups.abs().amax(dim=-1) / fp8_info.max).clamp(min=1.0e-10)
    quantized = torch.clamp(
        groups / scales.unsqueeze(-1),
        min=fp8_info.min,
        max=fp8_info.max,
    ).to(fp8_dtype)
    quantized = quantized.reshape(*activated.shape).contiguous()
    scales = scales.to(torch.float32).contiguous()

    if values.num_tokens_per_expert is not None:
        if values.num_experts is None:
            raise ValueError("num_experts is required with num_tokens_per_expert")
        counts = values.num_tokens_per_expert.to(device=quantized.device)
        token_positions = torch.arange(values.gate_up.shape[1], device=quantized.device)
        invalid = token_positions.unsqueeze(0) >= counts.unsqueeze(1)
        if torch.any(invalid):
            quantized = quantized.clone()
            scales = scales.clone()
            quantized.view(torch.uint8)[invalid] = 0
            scales[invalid] = 0.0
    return quantized, scales


def _fused_swiglu_nvfp4_quant_values(
    *,
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    input_scale: float = 0.125,
    scale_size: int = 16,
) -> _FusedSwiGLUNVFP4QuantValues:
    gate_up = _randn(
        (num_tokens, 2 * hidden_dim),
        dtype=dtype,
        device=device,
        seed=seed + 1,
    )
    global_scale = torch.tensor([1.0 / input_scale], dtype=torch.float32, device=device)
    return _FusedSwiGLUNVFP4QuantValues(
        gate_up=gate_up,
        global_scale=global_scale,
        scale_size=scale_size,
    )


def _fused_swiglu_nvfp4_quant_reference(
    values: _FusedSwiGLUNVFP4QuantValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    activated = _gated_activation_reference(values.gate_up, activation="silu")
    return nvfp4_quantization_reference(
        activated,
        scale=torch.reciprocal(values.global_scale).reshape(1),
        scale_size=values.scale_size,
        scale_layout="linear",
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    # Qwen3.5 attn_output_gate decode shapes (num_tokens, num_heads * head_dim).
    [(1, 4096), (17, 6144), (128, 4096), (256, 8192)],
)
def test_sigmoid_mul_matches_eager(
    dtype: torch.dtype, shape: tuple[int, int], device: str
) -> None:
    values = _sigmoid_mul_values(
        num_tokens=shape[0],
        hidden_dim=shape[1],
        dtype=dtype,
        seed=shape[0] * 101 + shape[1],
        device=device,
    )
    ref = _sigmoid_mul_reference(values.x, values.gate)

    out = sigmoid_mul(values.x.clone(), values.gate)

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_sigmoid_mul_is_inplace(device: str) -> None:
    values = _sigmoid_mul_values(
        num_tokens=8,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=11,
        device=device,
    )
    same = sigmoid_mul(values.x, values.gate)
    assert same.data_ptr() == values.x.data_ptr()


def test_sigmoid_mul_empty(device: str) -> None:
    values = _sigmoid_mul_values(
        num_tokens=0,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=12,
        device=device,
    )
    out = sigmoid_mul(values.x, values.gate)
    assert out.shape == values.x.shape


def test_sigmoid_mul_rejects_shape_mismatch(device: str) -> None:
    values = _sigmoid_mul_values(
        num_tokens=4,
        hidden_dim=32,
        dtype=torch.bfloat16,
        seed=13,
        device=device,
    )
    gate = values.gate[:, :16].contiguous()
    with pytest.raises(ValueError, match="shape mismatch"):
        sigmoid_mul(values.x, gate)


def test_sigmoid_mul_rejects_dtype_mismatch(device: str) -> None:
    values = _sigmoid_mul_values(
        num_tokens=4,
        hidden_dim=32,
        dtype=torch.bfloat16,
        seed=14,
        device=device,
    )
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
    ``qkv.split`` -> ``.view(T, H, 2*D)`` -> ``torch.chunk(q_gate, 2, dim=-1)``.
    ``gate.stride(0)`` is the full qkv row width (q_size*2 + 2*kv_size),
    not just H*2*D. The kernel must read this strided view directly without
    a contiguous copy."""
    num_tokens = 19
    q_size = num_heads * head_dim
    values = _sigmoid_mul_values(
        num_tokens=num_tokens,
        hidden_dim=q_size,
        dtype=dtype,
        gate_layout="qkv_split",
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=num_heads * 1000 + num_kv_heads * 100 + head_dim,
        device=device,
    )
    assert values.gate_storage is not None
    assert not values.gate.is_contiguous()
    assert values.gate.stride(0) == values.gate_storage.stride(0)
    assert values.gate.stride(-1) == 1

    ref = _sigmoid_mul_reference(values.x, values.gate)

    out = sigmoid_mul(values.x.clone(), values.gate)

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_sigmoid_mul_rejects_4d_gate(device: str) -> None:
    values = _sigmoid_mul_values(
        num_tokens=4,
        hidden_dim=32,
        dtype=torch.bfloat16,
        seed=15,
        device=device,
    )
    gate = values.gate.reshape(4, 2, 4, 4)
    with pytest.raises(ValueError, match="gate must be 2D or 3D"):
        sigmoid_mul(values.x, gate)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 7168 * 2), (17, 1024), (128, 9216 * 2)])
def test_silu_and_mul_matches_eager(
    dtype: torch.dtype, shape: tuple[int, int], device: str
) -> None:
    values = _gated_activation_values(
        num_tokens=shape[0],
        hidden_dim=shape[1] // 2,
        dtype=dtype,
        seed=shape[0] * 103 + shape[1],
        device=device,
    )
    ref = _gated_activation_reference(values.x, activation="silu")

    out = silu_and_mul(values.x)

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_silu_and_mul_writes_provided_output(device: str) -> None:
    values = _gated_activation_values(
        num_tokens=8,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=21,
        device=device,
    )
    out = torch.empty(8, 256, device=device, dtype=torch.bfloat16)
    same = silu_and_mul(values.x, out)
    assert same.data_ptr() == out.data_ptr()


def test_silu_and_mul_empty(device: str) -> None:
    values = _gated_activation_values(
        num_tokens=0,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=22,
        device=device,
    )
    out = silu_and_mul(values.x)
    assert out.shape == (0, 256)


def test_silu_and_mul_rejects_bad_output_shape(device: str) -> None:
    values = _gated_activation_values(
        num_tokens=4,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=23,
        device=device,
    )
    out = torch.empty(4, 128, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out shape"):
        silu_and_mul(values.x, out)


@pytest.mark.skipif(
    not platform.is_nvidia,
    reason="FlashInfer gated activation kernels require NVIDIA CUDA.",
)
@pytest.mark.parametrize(
    "activation,kernel",
    [
        ("silu", flashinfer_silu_and_mul),
        ("gelu", flashinfer_gelu_and_mul),
        ("gelu_tanh", flashinfer_gelu_tanh_and_mul),
    ],
)
def test_gated_activation_flashinfer_kernel(
    device: str,
    activation: str,
    kernel,
) -> None:
    values = _gated_activation_values(
        num_tokens=11,
        hidden_dim=1024,
        dtype=torch.bfloat16,
        seed=111,
        device=device,
    )

    try:
        out = kernel(values.x)
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer gated activation kernel unavailable: {exc}")
    torch.cuda.synchronize()

    ref = _gated_activation_reference(values.x, activation=activation)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    [(1, 3584), (1, 5120), (17, 3584), (128, 5120), (256, 3584)],
)
def test_fused_gate_sigmoid_mul_add_matches_eager(
    dtype: torch.dtype, num_tokens: int, hidden_dim: int, device: str
) -> None:
    values = _fused_gate_sigmoid_mul_add_values(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        dtype=dtype,
        seed=num_tokens * 107 + hidden_dim,
        device=device,
    )
    ref = _fused_gate_sigmoid_mul_add_reference(values)

    out = fused_gate_sigmoid_mul_add(
        values.hidden_states,
        values.gate_weight,
        values.shared_output.clone(),
        values.final_hidden_states.clone(),
    )

    tol = _activation_tol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


def test_fused_gate_sigmoid_mul_add_is_inplace(device: str) -> None:
    values = _fused_gate_sigmoid_mul_add_values(
        num_tokens=8,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=31,
        device=device,
    )

    result = fused_gate_sigmoid_mul_add(
        values.hidden_states,
        values.gate_weight,
        values.shared_output,
        values.final_hidden_states,
    )
    assert result.data_ptr() == values.final_hidden_states.data_ptr()


def test_fused_gate_sigmoid_mul_add_empty(device: str) -> None:
    values = _fused_gate_sigmoid_mul_add_values(
        num_tokens=0,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=32,
        device=device,
    )

    out = fused_gate_sigmoid_mul_add(
        values.hidden_states,
        values.gate_weight,
        values.shared_output,
        values.final_hidden_states,
    )
    assert out.shape == (0, 256)


def test_fused_swiglu_fp8_ue8m0_matches_reference(device: str) -> None:
    values = _gated_activation_values(
        num_tokens=5,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=105,
        device=device,
    )
    expected_q, expected_scales = _fused_swiglu_fp8_ue8m0_reference(
        values.x,
        swiglu_limit=7.0,
    )

    actual_q, actual_scales = fused_swiglu_fp8_ue8m0(values.x, 7.0)
    torch.cuda.synchronize()

    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_scales, expected_scales)


@pytest.mark.skipif(
    not platform.is_nvidia,
    reason="CUDA fused block quantization kernels require NVIDIA CUDA.",
)
def test_fused_swiglu_fp8_block_quant_cuda_kernel() -> None:
    values = _fused_swiglu_fp8_block_quant_values(
        num_tokens=4,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=108,
        device="cuda",
    )
    try:
        actual_q, actual_scales = silu_and_mul_fuse_block_quant(
            values.gate_up,
            values.scale_out,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"CUDA fused block quantization extension unavailable: {exc}")
    torch.cuda.synchronize()

    expected_q, expected_scales = _fused_swiglu_fp8_block_quant_reference(values)
    assert actual_q.shape == expected_q.shape
    assert actual_q.dtype == expected_q.dtype
    assert actual_scales.shape == expected_scales.shape

    gate, up = values.gate_up.float().chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate) * up
    dequant = (
        actual_q.float().reshape(4, 2, 128) * actual_scales.unsqueeze(-1)
    ).reshape(4, 256)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0),
        dequant.flatten().unsqueeze(0),
    )
    assert cos_sim.item() > 0.99


@pytest.mark.skipif(
    not platform.is_nvidia,
    reason="CUDA fused block quantization kernels require NVIDIA CUDA.",
)
def test_fused_swiglu_fp8_block_quant_ep_cuda_kernel() -> None:
    values = _fused_swiglu_fp8_block_quant_values(
        num_tokens=4,
        hidden_dim=256,
        dtype=torch.bfloat16,
        seed=110,
        device="cuda",
        num_experts=3,
    )
    try:
        actual_q, actual_scales = silu_and_mul_fuse_block_quant(
            values.gate_up,
            values.scale_out,
            enable_pdl=False,
            num_tokens_per_expert=values.num_tokens_per_expert,
            num_tokens_hint=values.num_tokens_hint,
            num_experts=values.num_experts,
        )
    except RuntimeError as exc:
        pytest.skip(f"CUDA fused block quantization extension unavailable: {exc}")
    torch.cuda.synchronize()

    expected_q, expected_scales = _fused_swiglu_fp8_block_quant_reference(values)
    assert actual_q.shape == expected_q.shape
    assert actual_q.dtype == expected_q.dtype
    assert actual_scales.shape == expected_scales.shape

    gate, up = values.gate_up.float().chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate) * up
    dequant = (
        actual_q.float().reshape(3, 4, 2, 128) * actual_scales.unsqueeze(-1)
    ).reshape(3, 4, 256)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0),
        dequant.flatten().unsqueeze(0),
    )
    assert cos_sim.item() > 0.99


@pytest.mark.skipif(
    not platform.is_nvidia,
    reason="CUDA fused NVFP4 quantization kernels require NVIDIA CUDA.",
)
def test_fused_swiglu_nvfp4_quant_cuda_kernel() -> None:
    values = _fused_swiglu_nvfp4_quant_values(
        num_tokens=4,
        hidden_dim=64,
        dtype=torch.bfloat16,
        seed=109,
        device="cuda",
    )
    try:
        actual_q, actual_scales = silu_and_mul_fuse_nvfp4_quant(
            values.gate_up,
            values.global_scale,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"CUDA fused NVFP4 quantization extension unavailable: {exc}")
    torch.cuda.synchronize()

    expected_q, _expected_scales = _fused_swiglu_nvfp4_quant_reference(values)
    assert actual_q.shape == expected_q.shape
    assert actual_q.dtype == expected_q.dtype
    assert actual_scales.dtype == torch.float8_e4m3fn
