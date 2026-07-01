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

import platform
from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.ops.attention import tokenspeed_mla as kernel_mla
from tokenspeed_kernel.ops.attention.tokenspeed_mla import (
    mla_kv_pack_quantize_fp8,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    MLAKVPackKVLayout,
    MLAKVPackQuantizeFP8InputConfig,
    MLAKVPackQuantizeFP8Inputs,
    MLAKVPackQuantizeFP8InputValues,
    MLAPrefillFP8InputConfig,
    MLAPrefillFP8Inputs,
    mla_kv_pack_quantize_fp8_reference,
    mla_prefill_fp8_reference,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="tokenspeed_mla kernels are NVIDIA-only",
)

# K2.5 / DSv3 chunked-prefill shape with TP=4.
S = 256
H = 16
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128

# (is_causal, return_lse) variants exposed by tokenspeed_mla_prefill.
PREFILL_BINARY_VARIANT_FLAGS = [
    (causal, lse) for causal in (False, True) for lse in (False, True)
]

# (seq_lens_q, seq_lens_k, h_q, h_k) — varlen problem layouts.
PREFILL_BINARY_SHAPE_CASES = [
    ((64, 128, 32), (64, 128, 32), 8, 8),
    ((128, 256, 96), (128, 256, 96), 128, 128),
    ((990,), (990,), 128, 128),
    ((1024, 1139), (1024, 1139), 8, 8),
]


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _make_mla_kv_pack_values(
    device: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    k_scale_inv: float = 1.0,
    v_scale_inv: float = 1.0,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    k_pe_rank: int = 3,
    kv_layout: MLAKVPackKVLayout = "separate",
    seed: int = 0,
) -> MLAKVPackQuantizeFP8InputValues:
    return MLAKVPackQuantizeFP8Inputs(
        MLAKVPackQuantizeFP8InputConfig(
            num_tokens=S,
            num_kv_heads=H,
            qk_nope_head_dim=QK_NOPE,
            qk_rope_head_dim=QK_ROPE,
            v_head_dim=V_HEAD,
            input_dtype=dtype,
            k_scale_inv=k_scale_inv,
            v_scale_inv=v_scale_inv,
            fp8_dtype=fp8_dtype,
            k_pe_rank=k_pe_rank,
            kv_layout=kv_layout,
        )
    ).generate(seed=seed, device=device)


def _make_kv_slice_inputs(
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> MLAKVPackQuantizeFP8InputValues:
    """Mirror the deepseek_v3.py call site: k_nope and v are slice views of
    a packed kv tensor produced by kv_b_proj."""
    return _make_mla_kv_pack_values(
        device,
        dtype=dtype,
        kv_layout="packed_slices",
        seed=0,
    )


def _host_arch() -> str:
    return {
        "amd64": "x86_64",
        "arm64": "aarch64",
        "x64": "x86_64",
    }.get(platform.machine().lower(), platform.machine().lower())


def _prefill_shape_id(case) -> str:
    seq_lens_q, seq_lens_k, h_q, h_k = case
    return f"sQ{sum(seq_lens_q)}_sK{sum(seq_lens_k)}_hq{h_q}_hk{h_k}"


def _prefill_variant_id(flags: tuple[bool, bool]) -> str:
    causal, lse = flags
    return ("causal" if causal else "nocausal") + ("_lse" if lse else "")


def _require_mla_binary_prefill():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for tokenspeed-mla binary prefill")

    import tokenspeed_mla.fmha_binary as fmha_binary

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    expected_suffix = f"sm_{props.major}{props.minor}a_{_host_arch()}.so"
    so_path = fmha_binary._resolve_so_path()
    if not so_path.exists():
        pytest.skip(f"tokenspeed-mla binary prefill SO not found: {so_path}")
    assert so_path.name.endswith(expected_suffix)
    return fmha_binary, so_path


def _run_mla_kv_pack_quantize_fp8(
    values: MLAKVPackQuantizeFP8InputValues,
    *,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {}
    if k_out is not None:
        kwargs["k_out"] = k_out
    if v_out is not None:
        kwargs["v_out"] = v_out
    if enable_pdl is not None:
        kwargs["enable_pdl"] = enable_pdl
    return mla_kv_pack_quantize_fp8(
        values.k_nope,
        values.k_pe,
        values.v,
        k_scale_inv=values.k_scale_inv,
        v_scale_inv=values.v_scale_inv,
        fp8_dtype=values.fp8_dtype,
        **kwargs,
    )


def test_binary_prefill_so_loads() -> None:
    fmha_binary, so_path = _require_mla_binary_prefill()

    module = fmha_binary._load_module(str(so_path))

    assert getattr(module, fmha_binary._FUNC_NAMES[(False, False)], None) is not None
    assert fmha_binary.has_binary_prefill()


@pytest.mark.parametrize(
    "shape_case",
    PREFILL_BINARY_SHAPE_CASES,
    ids=[_prefill_shape_id(case) for case in PREFILL_BINARY_SHAPE_CASES],
)
@pytest.mark.parametrize(
    "variant_flags",
    PREFILL_BINARY_VARIANT_FLAGS,
    ids=[_prefill_variant_id(flags) for flags in PREFILL_BINARY_VARIANT_FLAGS],
)
def test_kernel_tokenspeed_mla_prefill_binary_e2e(
    device: str, monkeypatch, shape_case, variant_flags: tuple[bool, bool]
) -> None:
    is_causal, return_lse = variant_flags

    _require_mla_binary_prefill()

    import tokenspeed_mla.mla_prefill as mla_prefill

    monkeypatch.setattr(mla_prefill, "_PREFILL_BACKEND_ENV", "binary")
    mla_prefill._resolve_backend.cache_clear()

    seq_lens_q, seq_lens_k, h_q, h_k = shape_case
    assert seq_lens_q == seq_lens_k
    assert h_q == h_k
    values = MLAPrefillFP8Inputs(
        MLAPrefillFP8InputConfig(
            batch_size=len(seq_lens_q),
            total_tokens=sum(seq_lens_q),
            num_heads=h_q,
            qk_head_dim=QK_NOPE + QK_ROPE,
            v_head_dim=V_HEAD,
            source_dtype=torch.bfloat16,
            length_mode="ragged",
            max_tokens_per_request=max(seq_lens_q),
        )
    ).generate(seed=3 + sum(seq_lens_q) + h_q, metadata_seed=4 + h_q, device=device)
    expected = mla_prefill_fp8_reference(values, is_causal=is_causal)

    try:
        actual = kernel_mla.tokenspeed_mla_prefill(
            values.query,
            values.key,
            values.value,
            values.metadata.cache_seqlens,
            values.metadata.cu_seqlens_kv,
            values.metadata.resolved_max_seqlen_k,
            batch_size=len(values.metadata.visible_kv_lens_cpu),
            softmax_scale=values.softmax_scale,
            is_causal=is_causal,
            return_lse=return_lse,
            cum_seq_lens_q=values.metadata.cu_seqlens_q,
            max_seq_len_q=values.metadata.max_seqlen_q,
        )
    finally:
        mla_prefill._resolve_backend.cache_clear()
    torch.cuda.synchronize()

    if return_lse:
        actual, actual_lse = actual
        assert actual_lse.shape == expected.lse.shape
        assert actual_lse.dtype == torch.float32
    tolerance = 0.25 if is_causal else 0.1
    assert actual.shape == expected.out.shape
    assert actual.dtype == expected.out.dtype
    torch.testing.assert_close(
        actual.float(), expected.out.float(), atol=tolerance, rtol=1e-5
    )
    if return_lse:
        torch.testing.assert_close(
            actual_lse.float(), expected.lse.float(), atol=tolerance, rtol=1e-5
        )


def test_kernel_tokenspeed_mla_prefill_binary_uses_generator(
    device: str,
    monkeypatch,
) -> None:
    _require_mla_binary_prefill()

    import tokenspeed_mla.mla_prefill as mla_prefill

    monkeypatch.setattr(mla_prefill, "_PREFILL_BACKEND_ENV", "binary")
    mla_prefill._resolve_backend.cache_clear()

    values = MLAPrefillFP8Inputs(
        MLAPrefillFP8InputConfig(
            batch_size=3,
            total_tokens=192,
            num_heads=8,
            qk_head_dim=QK_NOPE + QK_ROPE,
            v_head_dim=V_HEAD,
            source_dtype=torch.bfloat16,
            length_mode="fixed_per_request",
        )
    ).generate(seed=30, metadata_seed=31, device=device)
    expected = mla_prefill_fp8_reference(values, is_causal=True)

    try:
        actual, actual_lse = kernel_mla.tokenspeed_mla_prefill(
            values.query,
            values.key,
            values.value,
            values.metadata.cache_seqlens,
            values.metadata.cu_seqlens_kv,
            values.metadata.resolved_max_seqlen_k,
            batch_size=len(values.metadata.visible_kv_lens_cpu),
            softmax_scale=values.softmax_scale,
            is_causal=True,
            return_lse=True,
            cum_seq_lens_q=values.metadata.cu_seqlens_q,
            max_seq_len_q=values.metadata.max_seqlen_q,
        )
    finally:
        mla_prefill._resolve_backend.cache_clear()
    torch.cuda.synchronize()

    assert actual.shape == expected.out.shape
    assert actual.dtype == expected.out.dtype
    torch.testing.assert_close(
        actual.float(), expected.out.float(), atol=0.25, rtol=1e-5
    )
    torch.testing.assert_close(
        actual_lse.float(), expected.lse.float(), atol=0.25, rtol=1e-5
    )


def test_pure_cast_strided_inputs(device: str) -> None:
    """k_nope/v are non-contiguous slices, scale=1.0 — the prefill call site."""
    values = _make_kv_slice_inputs(device)
    assert not values.k_nope.is_contiguous()
    assert not values.v.is_contiguous()

    k_ref, v_ref = mla_kv_pack_quantize_fp8_reference(values)
    k_out, v_out = _run_mla_kv_pack_quantize_fp8(values)
    torch.cuda.synchronize()

    assert k_out.shape == (S, H, QK_NOPE + QK_ROPE)
    assert v_out.shape == (S, H, V_HEAD)
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_scaled_independent_k_v(device: str) -> None:
    """k and v use different scales; output reflects each independently."""
    values = _make_mla_kv_pack_values(
        device,
        k_scale_inv=0.5,
        v_scale_inv=1.7,
        seed=1,
    )

    k_ref, v_ref = mla_kv_pack_quantize_fp8_reference(values)
    k_out, v_out = _run_mla_kv_pack_quantize_fp8(values)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_k_pe_2d_and_3d_equivalent(device: str) -> None:
    """k_pe is accepted as both [s, 1, rope] and [s, rope]; same output."""
    values_3d = _make_mla_kv_pack_values(device, seed=2)
    values_2d = replace(values_3d, k_pe=values_3d.k_pe.squeeze(1))

    k_3d, v_3d = _run_mla_kv_pack_quantize_fp8(values_3d)
    k_2d, v_2d = _run_mla_kv_pack_quantize_fp8(values_2d)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_3d, k_2d)
    assert _bitwise_equal(v_3d, v_2d)


def test_contiguous_inputs(device: str) -> None:
    """Standalone (non-slice) k_nope, v inputs also work."""
    values = _make_mla_kv_pack_values(device, seed=3)
    assert values.k_nope.is_contiguous()
    assert values.v.is_contiguous()

    k_ref, v_ref = mla_kv_pack_quantize_fp8_reference(values)
    k_out, v_out = _run_mla_kv_pack_quantize_fp8(values)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_fp16_input(device: str) -> None:
    values = _make_kv_slice_inputs(device, dtype=torch.float16)

    k_ref, v_ref = mla_kv_pack_quantize_fp8_reference(values)
    k_out, v_out = _run_mla_kv_pack_quantize_fp8(values)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_e5m2_output(device: str) -> None:
    values = _make_mla_kv_pack_values(
        device,
        fp8_dtype=torch.float8_e5m2,
        seed=4,
    )
    k_ref, v_ref = mla_kv_pack_quantize_fp8_reference(values)
    k_out, v_out = _run_mla_kv_pack_quantize_fp8(values)
    torch.cuda.synchronize()

    assert k_out.dtype == torch.float8_e5m2
    assert v_out.dtype == torch.float8_e5m2
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_preallocated_outputs(device: str) -> None:
    values = _make_mla_kv_pack_values(device, seed=5)
    k_out = torch.empty(
        (S, H, QK_NOPE + QK_ROPE), dtype=torch.float8_e4m3fn, device=device
    )
    v_out = torch.empty((S, H, V_HEAD), dtype=torch.float8_e4m3fn, device=device)

    k_ret, v_ret = _run_mla_kv_pack_quantize_fp8(
        values,
        k_out=k_out,
        v_out=v_out,
    )
    torch.cuda.synchronize()

    assert k_ret.data_ptr() == k_out.data_ptr()
    assert v_ret.data_ptr() == v_out.data_ptr()

    k_ref, v_ref = mla_kv_pack_quantize_fp8_reference(values)
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


@pytest.mark.parametrize("k_scale_inv,v_scale_inv", [(1.0, 1.0), (0.5, 1.7)])
def test_pdl_off_matches_pdl_on(
    device: str, k_scale_inv: float, v_scale_inv: float
) -> None:
    """PDL is a scheduling hint; output must be bitwise-identical regardless."""
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper+ (SM≥90)")
    values = _make_mla_kv_pack_values(
        device,
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        seed=6,
    )

    k_off, v_off = _run_mla_kv_pack_quantize_fp8(
        values,
        enable_pdl=False,
    )
    k_on, v_on = _run_mla_kv_pack_quantize_fp8(
        values,
        enable_pdl=True,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(k_off, k_on)
    assert _bitwise_equal(v_off, v_on)
