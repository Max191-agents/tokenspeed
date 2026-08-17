from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenspeed_kernel.ops.layernorm.triton as layernorm_triton
import torch
from tokenspeed_kernel.ops.layernorm.triton import (
    fused_qk_rmsnorm_rope,
    fused_qk_rmsnorm_rope_gate,
    qk_rmsnorm,
    rmsnorm,
    rmsnorm_fused_parallel,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton layernorm tests require an NVIDIA or AMD GPU.",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 2880])
def test_rmsnorm(dtype: torch.dtype, hidden_size: int, device: str) -> None:
    num_tokens = 7
    eps = 1e-6
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    out = rmsnorm(x, weight, eps)

    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_float * torch.rsqrt(variance + eps) * weight).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 2880])
def test_rmsnorm_with_residual(
    dtype: torch.dtype, hidden_size: int, device: str
) -> None:
    num_tokens = 7
    eps = 1e-6
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    out, residual_out = rmsnorm(x, weight, eps, residual=residual)

    x_float = x.to(torch.float32) + residual.to(torch.float32)
    ref_residual = x_float.to(dtype)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_float * torch.rsqrt(variance + eps) * weight).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual_out, ref_residual, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_parallel_rmsnorm_preserves_same_dtype_behavior(
    dtype: torch.dtype, device: str
) -> None:
    torch.manual_seed(48)
    tokens = 11
    eps = 1e-6
    q = torch.randn(tokens, 384, device=device, dtype=dtype)
    kv = torch.randn(tokens, 160, device=device, dtype=dtype)
    q_weight = torch.randn(384, device=device, dtype=torch.float32)
    kv_weight = torch.randn(160, device=device, dtype=torch.float32)
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)

    rmsnorm_fused_parallel(q, q_weight, q_out, kv, kv_weight, kv_out, eps)

    q_variance = q.float().pow(2).mean(dim=-1, keepdim=True)
    kv_variance = kv.float().pow(2).mean(dim=-1, keepdim=True)
    q_ref = (q.float() * torch.rsqrt(q_variance + eps) * q_weight).to(dtype)
    kv_ref = (kv.float() * torch.rsqrt(kv_variance + eps) * kv_weight).to(dtype)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(kv_out, kv_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not platform.is_amd, reason="FP8 output is an AMD-only extension")
@pytest.mark.parametrize("placement", ["prefix", "suffix"])
def test_fused_parallel_rmsnorm_writes_strided_fp8_output(
    placement: str, device: str
) -> None:
    torch.manual_seed(49)
    tokens = 11
    eps = 1e-6
    q = torch.randn(tokens, 1536, device=device, dtype=torch.bfloat16)
    kv = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(1536, device=device, dtype=torch.float32)
    kv_weight = torch.randn(512, device=device, dtype=torch.float32)
    q_out = torch.empty_like(q)
    combined_key = torch.full(
        (tokens, 576), -6.0, device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    if placement == "prefix":
        fp8_output = combined_key[:, :512]
        untouched = combined_key[:, 512:]
    else:
        fp8_output = combined_key[:, 64:]
        untouched = combined_key[:, :64]
    untouched_before = untouched.view(torch.uint8).clone()

    rmsnorm_fused_parallel(
        q,
        q_weight,
        q_out,
        kv,
        kv_weight,
        fp8_output,
        eps,
    )

    q_variance = q.float().pow(2).mean(dim=-1, keepdim=True)
    kv_variance = kv.float().pow(2).mean(dim=-1, keepdim=True)
    q_ref = (q.float() * torch.rsqrt(q_variance + eps) * q_weight).to(q.dtype)
    normalized_kv = kv.float() * torch.rsqrt(kv_variance + eps) * kv_weight
    fp8_ref = normalized_kv.to(kv.dtype).to(torch.float8_e4m3fn)

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fp8_output.float(), fp8_ref.float(), atol=0.5, rtol=0)
    assert torch.equal(
        untouched.view(torch.uint8),
        untouched_before,
    )


@pytest.mark.skipif(not platform.is_amd, reason="FP8 output is an AMD-only extension")
def test_fused_parallel_rmsnorm_fp8_rounds_through_bf16_and_saturates(
    device: str,
) -> None:
    hidden_size = 64
    q = torch.ones((1, hidden_size), device=device, dtype=torch.bfloat16)
    kv = torch.ones_like(q)
    q_weight = torch.ones(hidden_size, device=device, dtype=torch.float32)
    signs = torch.where(torch.arange(hidden_size, device=device) % 2 == 0, 1.0, -1.0)
    kv_weight = signs * 465.0
    q_out = torch.empty_like(q)
    kv_fp8 = torch.empty_like(kv, dtype=torch.float8_e4m3fn)

    rmsnorm_fused_parallel(q, q_weight, q_out, kv, kv_weight, kv_fp8, 0.0)

    normalized = kv.float() * kv_weight
    staged = normalized.to(torch.bfloat16).to(torch.float8_e4m3fn)
    direct = normalized.to(torch.float8_e4m3fn)
    assert torch.equal(kv_fp8.view(torch.uint8), staged.view(torch.uint8))
    assert torch.all(kv_fp8.float().abs() == 448.0)
    assert torch.isnan(direct.float()).all()


@pytest.mark.skipif(not platform.is_amd, reason="FP8 output is an AMD-only extension")
def test_fused_parallel_rmsnorm_fp8_cuda_graph_replay(device: str) -> None:
    torch.manual_seed(50)
    eps = 1e-6
    q = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    kv = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(128, device=device, dtype=torch.float32)
    kv_weight = torch.randn(64, device=device, dtype=torch.float32)
    q_out = torch.empty_like(q)
    kv_fp8 = torch.empty_like(kv, dtype=torch.float8_e4m3fn)

    rmsnorm_fused_parallel(q, q_weight, q_out, kv, kv_weight, kv_fp8, eps)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm_fused_parallel(q, q_weight, q_out, kv, kv_weight, kv_fp8, eps)

    next_q = torch.randn_like(q)
    next_kv = torch.randn_like(kv)
    q.copy_(next_q)
    kv.copy_(next_kv)
    graph.replay()
    torch.cuda.synchronize()

    q_variance = next_q.float().pow(2).mean(dim=-1, keepdim=True)
    kv_variance = next_kv.float().pow(2).mean(dim=-1, keepdim=True)
    q_ref = (next_q.float() * torch.rsqrt(q_variance + eps) * q_weight).to(
        torch.bfloat16
    )
    kv_ref = (
        (next_kv.float() * torch.rsqrt(kv_variance + eps) * kv_weight)
        .to(torch.bfloat16)
        .to(torch.float8_e4m3fn)
    )
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(kv_fp8.float(), kv_ref.float(), atol=0.5, rtol=0)


@pytest.mark.parametrize(
    "input2_dtype,output2_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
    ],
)
def test_fused_parallel_rmsnorm_rejects_invalid_output2_dtype(
    input2_dtype: torch.dtype, output2_dtype: torch.dtype, device: str
) -> None:
    input1 = torch.ones((2, 32), device=device, dtype=torch.bfloat16)
    input2 = torch.ones((2, 32), device=device, dtype=input2_dtype)
    weight = torch.ones(32, device=device, dtype=torch.float32)
    output1 = torch.empty_like(input1)
    output2 = torch.empty_like(input2, dtype=output2_dtype)

    with pytest.raises(TypeError, match="output2 dtype|E4M3 output2 requires BF16"):
        rmsnorm_fused_parallel(input1, weight, output1, input2, weight, output2, 1e-6)


def test_fused_parallel_rmsnorm_rejects_fp8_output_on_non_amd(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    monkeypatch.setattr(
        layernorm_triton,
        "current_platform",
        lambda: SimpleNamespace(is_amd=False),
    )
    input1 = torch.ones((2, 32), device=device, dtype=torch.bfloat16)
    input2 = torch.ones_like(input1)
    weight = torch.ones(32, device=device, dtype=torch.float32)
    output1 = torch.empty_like(input1)
    output2 = torch.empty_like(input2, dtype=torch.float8_e4m3fn)

    with pytest.raises(
        RuntimeError, match="BF16-to-E4M3 output2.*only supported on AMD"
    ):
        rmsnorm_fused_parallel(input1, weight, output1, input2, weight, output2, 1e-6)


@pytest.mark.skipif(not platform.is_amd, reason="FP8 output is an AMD-only extension")
def test_fused_parallel_rmsnorm_rejects_fp8_output_device_mismatch(
    device: str,
) -> None:
    input1 = torch.ones((2, 32), device=device, dtype=torch.bfloat16)
    input2 = torch.ones_like(input1)
    weight = torch.ones(32, device=device, dtype=torch.float32)
    output1 = torch.empty_like(input1)
    output2 = torch.empty(input2.shape, device="cpu", dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="output2 must be on"):
        rmsnorm_fused_parallel(input1, weight, output1, input2, weight, output2, 1e-6)


def _gemma_ref(
    x: torch.Tensor, w: torch.Tensor, head_dim: int, eps: float, dtype: torch.dtype
) -> torch.Tensor:
    x_by_head = x.reshape(-1, head_dim).to(torch.float32)
    variance = x_by_head.pow(2).mean(dim=-1, keepdim=True)
    out = x_by_head * torch.rsqrt(variance + eps) * (1.0 + w)
    return out.to(dtype).view(x.shape)


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
    num_tokens = 17
    eps = 1e-6
    q = torch.randn(num_tokens, num_q_heads * head_dim, device=device, dtype=dtype)
    k = torch.randn(num_tokens, num_kv_heads * head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    q_gemma_weight = q_weight + 1.0
    k_gemma_weight = k_weight + 1.0

    q_out, k_out = qk_rmsnorm(q, k, q_gemma_weight, k_gemma_weight, eps)

    torch.testing.assert_close(
        q_out, _gemma_ref(q, q_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_out, _gemma_ref(k, k_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )


def test_qk_rmsnorm_gemma_weight_strided_qkv_split(device: str) -> None:
    """Runtime path: q and k arrive as strided views from a packed qkv split.
    The kernel's stride-aware addressing must handle the non-contiguous
    leading-axis case without needing a ``.contiguous()`` copy."""
    num_tokens = 19
    num_q_heads, num_kv_heads, head_dim = 16, 2, 256
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    dtype = torch.bfloat16
    eps = 1e-6

    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, device=device, dtype=dtype)
    q, k, _v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    # Sanity: the views must share storage with qkv and be non-contiguous so we
    # actually exercise the strided path.
    assert q.data_ptr() == qkv.data_ptr()
    assert q.stride(0) == qkv.stride(0)
    assert not q.is_contiguous()
    assert not k.is_contiguous()

    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    q_gemma_weight = q_weight + 1.0
    k_gemma_weight = k_weight + 1.0

    q_out, k_out = qk_rmsnorm(q, k, q_gemma_weight, k_gemma_weight, eps)

    torch.testing.assert_close(
        q_out, _gemma_ref(q, q_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_out, _gemma_ref(k, k_weight, head_dim, eps, dtype), atol=2e-2, rtol=2e-2
    )


def _build_rope_cache(
    rotary_dim: int, max_pos: int, base: float, device: str
) -> torch.Tensor:
    """Mirror tokenspeed.runtime.layers.rotary_embedding._compute_cos_sin_cache."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    # Per-position layout: [cos(rotary_dim/2), sin(rotary_dim/2)] — total rotary_dim.
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def _ref_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching unfused (split → qk_rmsnorm → apply_rope).

    Mirrors the bf16 round-trips at the same boundaries as the production path:
      1. GemmaRMSNorm computes fp32 then stores bf16  (qk_rmsnorm output)
      2. RoPE reads bf16, computes fp32, stores bf16  (apply_rope_with_cos_sin_cache_inplace)
    """
    dtype = q_gate.dtype
    n_tokens = q_gate.shape[0]

    # 1. Split q_gate into q and gate per head.
    q_gate_3d = q_gate.view(n_tokens, num_q_heads, 2 * head_dim)
    q, gate = torch.chunk(q_gate_3d, 2, dim=-1)
    q = q.reshape(n_tokens, num_q_heads * head_dim).contiguous()
    gate = gate.reshape(n_tokens, num_q_heads * head_dim).contiguous()

    # 2. GemmaRMSNorm over the full head_dim, with weight already +1.
    def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_h = x.reshape(-1, head_dim).to(torch.float32)
        var = x_h.pow(2).mean(dim=-1, keepdim=True)
        return (x_h * torch.rsqrt(var + eps) * w).to(dtype).view(x.shape)

    q_normed = _rmsnorm(q, q_weight)
    k_normed = _rmsnorm(k, k_weight)

    # 3. Partial RoPE on the first rotary_dim elements of each head.
    half_rotary = rotary_dim // 2
    cos = cos_sin_cache[positions, :half_rotary].unsqueeze(1).to(torch.float32)
    sin = (
        cos_sin_cache[positions, half_rotary:rotary_dim].unsqueeze(1).to(torch.float32)
    )

    def _apply_partial_rope(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        x_h = x.reshape(n_tokens, num_heads, head_dim).to(torch.float32)
        x1 = x_h[..., :half_rotary]
        x2 = x_h[..., half_rotary:rotary_dim]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out = x_h.clone()
        out[..., :half_rotary] = o1
        out[..., half_rotary:rotary_dim] = o2
        return out.reshape(n_tokens, num_heads * head_dim).to(dtype)

    q_out = _apply_partial_rope(q_normed, num_q_heads)
    k_out = _apply_partial_rope(k_normed, num_kv_heads)
    return q_out, k_out, gate


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
    num_tokens = 23
    eps = 1e-6
    max_pos = 1024

    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * head_dim, device=device, dtype=dtype
    )
    k = torch.randn(num_tokens, num_kv_heads * head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1
    q_gemma_weight = q_weight + 1.0
    k_gemma_weight = k_weight + 1.0
    cos_sin_cache = _build_rope_cache(rotary_dim, max_pos, base=10000.0, device=device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), device=device, dtype=torch.int64
    )

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_gemma_weight,
        k_gemma_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )

    q_ref, k_ref, gate_ref = _ref_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_gemma_weight,
        k_gemma_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )

    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)
    # Gate is a verbatim copy of the second-half slice of q_gate.
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


def test_fused_qk_rmsnorm_rope_gate_empty(device: str) -> None:
    """Zero-token batch returns correctly shaped empty tensors without launching."""
    head_dim, rotary_dim = 256, 64
    num_q_heads, num_kv_heads = 4, 1
    q_gate = torch.empty(
        0, num_q_heads * 2 * head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.empty(0, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.ones(head_dim, device=device, dtype=torch.float32)
    cos_sin_cache = _build_rope_cache(rotary_dim, 16, base=10000.0, device=device)
    positions = torch.empty(0, device=device, dtype=torch.int64)

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        weight,
        weight,
        cos_sin_cache,
        positions,
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
    q_gate = torch.randn(
        n, num_q_heads * 2 * head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(n, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.ones(head_dim, device=device, dtype=torch.float32)
    cos_sin_cache = _build_rope_cache(
        max(2, bad_rotary_dim if bad_rotary_dim > 0 else 2),
        16,
        base=10000.0,
        device=device,
    )
    positions = torch.zeros(n, device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="rotary_dim"):
        fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            weight,
            weight,
            cos_sin_cache,
            positions,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
            bad_rotary_dim,
        )


def test_rmsnorm_inplace(device: str) -> None:
    num_tokens = 7
    hidden_size = 128
    eps = 1e-6
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    x_ref = x.clone()
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    out = rmsnorm(x, weight, eps, out=x)

    x_float = x_ref.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_float * torch.rsqrt(variance + eps) * weight).to(torch.bfloat16)
    assert out.data_ptr() == x.data_ptr()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# fused_qk_rmsnorm_rope (no gate, full RoPE) — used by DFLASH draft model
# ---------------------------------------------------------------------------


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm reference (standard, NOT Gemma)."""
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _ref_rope_full(
    x: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Full-dim RoPE reference (rotary_dim == head_dim)."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    # Gather cos/sin for each token's position
    cos = cos_sin_cache[positions.long(), :half].float()  # [n_tokens, half]
    sin = cos_sin_cache[positions.long(), half:].float()
    x_f = x.float()
    x1, x2 = x_f[..., :half], x_f[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim",
    [(8, 8, 128), (8, 2, 128), (4, 4, 64)],
)
def test_fused_qk_rmsnorm_rope(
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
) -> None:
    """Verify fused_qk_rmsnorm_rope matches separate qk_norm + RoPE."""
    n_tokens = 13
    eps = 1e-6
    max_pos = 4096

    q = torch.randn(n_tokens, num_q_heads * head_dim, device=device, dtype=dtype)
    k = torch.randn(n_tokens, num_kv_heads * head_dim, device=device, dtype=dtype)
    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    positions = torch.randint(0, max_pos, (n_tokens,), device=device, dtype=torch.int32)

    # Build cos_sin_cache: (max_pos, head_dim) = [cos(half) | sin(half)]
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device=device)

    # --- Fused kernel ---
    q_fused, k_fused = fused_qk_rmsnorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )

    # --- Reference: per-head norm then full RoPE ---
    # Reshape to per-head, apply norm per head, reshape back
    q_heads = q.view(n_tokens, num_q_heads, head_dim)
    k_heads = k.view(n_tokens, num_kv_heads, head_dim)
    q_normed = torch.stack(
        [_ref_rmsnorm(q_heads[:, h], q_weight, eps) for h in range(num_q_heads)], dim=1
    ).view(n_tokens, num_q_heads * head_dim)
    k_normed = torch.stack(
        [_ref_rmsnorm(k_heads[:, h], k_weight, eps) for h in range(num_kv_heads)], dim=1
    ).view(n_tokens, num_kv_heads * head_dim)

    # Apply RoPE per head
    q_ref_heads = q_normed.view(n_tokens, num_q_heads, head_dim)
    k_ref_heads = k_normed.view(n_tokens, num_kv_heads, head_dim)
    q_ref = torch.stack(
        [
            _ref_rope_full(q_ref_heads[:, h], cos_sin_cache, positions)
            for h in range(num_q_heads)
        ],
        dim=1,
    ).view(n_tokens, num_q_heads * head_dim)
    k_ref = torch.stack(
        [
            _ref_rope_full(k_ref_heads[:, h], cos_sin_cache, positions)
            for h in range(num_kv_heads)
        ],
        dim=1,
    ).view(n_tokens, num_kv_heads * head_dim)

    torch.testing.assert_close(q_fused, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_fused, k_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_rmsnorm_rope_non_contiguous_input(
    dtype: torch.dtype,
    device: str,
) -> None:
    """Verify the kernel handles non-contiguous q/k views from qkv.split()."""
    n_tokens = 5
    num_q_heads = 4
    num_kv_heads = 4
    head_dim = 128
    eps = 1e-6
    max_pos = 2048
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # Simulate qkv.split() which produces non-contiguous views
    qkv = torch.randn(n_tokens, q_size + 2 * kv_size, device=device, dtype=dtype)
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    assert not q.is_contiguous() or not k.is_contiguous()  # at least one non-contig

    q_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    positions = torch.randint(0, max_pos, (n_tokens,), device=device, dtype=torch.int32)

    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device=device)

    # Should not crash — non-contiguous inputs handled via stride
    q_fused, k_fused = fused_qk_rmsnorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )

    # Compare with contiguous version
    q_contig_fused, k_contig_fused = fused_qk_rmsnorm_rope(
        q.contiguous(),
        k.contiguous(),
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )

    torch.testing.assert_close(q_fused, q_contig_fused, atol=0, rtol=0)
    torch.testing.assert_close(k_fused, k_contig_fused, atol=0, rtol=0)
