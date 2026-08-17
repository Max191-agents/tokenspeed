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

"""Tests for backend-selected MLA query preparation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import tokenspeed_kernel.ops.attention.mla_prepare as mla_prepare_ops
import tokenspeed_kernel.ops.embedding as embedding_ops
import tokenspeed_kernel.ops.gemm as gemm_ops
import torch
from tokenspeed_kernel.ops.attention import mla_prepare_fp8_query
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    select_kernel,
)


def _cpu_inputs(
    *,
    tokens: int = 2,
    heads: int = 3,
    kv_heads: int = 1,
    q_dim: int = 4,
    latent_dim: int = 5,
    rope_dim: int = 6,
) -> dict[str, torch.Tensor]:
    projection_storage = torch.randn(
        heads,
        q_dim,
        latent_dim,
        dtype=torch.bfloat16,
    )
    return {
        "q_nope": torch.randn(tokens, heads, q_dim, dtype=torch.bfloat16),
        "projection_weight": projection_storage.transpose(1, 2),
        "positions": torch.arange(tokens, dtype=torch.int32),
        "q_rope": torch.randn(tokens, heads, rope_dim, dtype=torch.bfloat16),
        "k_nope": torch.randn(tokens, kv_heads, latent_dim, dtype=torch.bfloat16),
        "k_rope": torch.randn(tokens, kv_heads, rope_dim, dtype=torch.bfloat16),
        "cos_sin_cache": torch.randn(16, rope_dim),
    }


def _gpu_inputs(
    device: str,
    *,
    tokens: int,
    heads: int,
    kv_heads: int,
    q_dim: int,
    latent_dim: int,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    projection_storage = torch.randn(
        heads,
        q_dim,
        latent_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    frequencies = torch.randn(128, rope_dim // 2, device=device)
    return {
        "q_nope": torch.randn(
            tokens, heads, q_dim, device=device, dtype=torch.bfloat16
        ),
        "projection_weight": projection_storage.transpose(1, 2),
        "positions": torch.arange(tokens, device=device, dtype=torch.int32),
        "q_rope": torch.randn(
            tokens, heads, rope_dim, device=device, dtype=torch.bfloat16
        ),
        "k_nope": torch.randn(
            tokens, kv_heads, latent_dim, device=device, dtype=torch.bfloat16
        ),
        "k_rope": torch.randn(
            tokens, kv_heads, rope_dim, device=device, dtype=torch.bfloat16
        ),
        "cos_sin_cache": torch.cat(
            (frequencies.cos(), frequencies.sin()), dim=-1
        ).contiguous(),
    }


def test_mla_prepare_accepts_transposed_production_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _cpu_inputs(
        tokens=2,
        heads=16,
        q_dim=192,
        latent_dim=512,
        rope_dim=64,
    )
    weight = inputs["projection_weight"]
    assert weight.shape == (16, 512, 192)
    assert weight.stride(1) == 1
    assert weight.stride(2) == 512

    query_out = torch.empty(2, 16, 576, dtype=torch.float8_e4m3fn)
    key_out = torch.empty(2, 1, 576, dtype=torch.float8_e4m3fn)

    def implementation(*args, **kwargs):
        return query_out, key_out

    monkeypatch.setattr(
        mla_prepare_ops,
        "select_kernel",
        lambda *args, **kwargs: SelectedKernel("test_mla_prepare", implementation),
    )

    returned = mla_prepare_fp8_query(**inputs)

    assert returned == (query_out, key_out)


def test_composite_mla_prepare_accepts_general_weight_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _cpu_inputs()
    inputs["projection_weight"] = inputs["projection_weight"].contiguous()
    weight = inputs["projection_weight"]
    assert weight.stride(1) != 1
    bmm = Mock(side_effect=lambda A, B, *, out, enable_pdl: out.zero_())
    query_result = torch.empty(2, 3, 11, dtype=torch.float8_e4m3fn)
    key_result = torch.empty(2, 1, 11, dtype=torch.float8_e4m3fn)
    apply_rope = Mock(return_value=(query_result, key_result))
    monkeypatch.setattr(gemm_ops, "bmm", bmm)
    monkeypatch.setattr(embedding_ops, "apply_rope_mla", apply_rope)

    returned = mla_prepare_ops.composite_mla_prepare_fp8_query(**inputs)

    assert returned == (query_result, key_result)
    bmm.assert_called_once()
    assert bmm.call_args.args[1] is weight
    apply_rope.assert_called_once()


def test_composite_mla_prepare_uses_raw_prefixes_without_prequantized_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _cpu_inputs()
    tokens, heads, _ = inputs["q_nope"].shape
    latent_dim = inputs["projection_weight"].shape[1]
    rope_dim = inputs["q_rope"].shape[2]
    absorbed_query = torch.full(
        (tokens, heads, latent_dim + rope_dim),
        -1.0,
        dtype=torch.bfloat16,
    )
    apply_calls: list[dict[str, object]] = []

    def fake_bmm(A, B, *, out, enable_pdl):
        assert A is not inputs["q_nope"]
        assert A.shape == (heads, tokens, inputs["q_nope"].shape[2])
        assert B is inputs["projection_weight"]
        assert enable_pdl
        out.fill_(3.0)
        return out

    query_result = torch.empty(
        tokens,
        heads,
        latent_dim + rope_dim,
        dtype=torch.float8_e4m3fn,
    )
    key_result = torch.empty(
        tokens,
        1,
        latent_dim + rope_dim,
        dtype=torch.float8_e4m3fn,
    )

    def fake_apply_rope_mla(**kwargs):
        apply_calls.append(kwargs)
        return query_result, key_result

    monkeypatch.setattr(gemm_ops, "bmm", fake_bmm)
    monkeypatch.setattr(embedding_ops, "apply_rope_mla", fake_apply_rope_mla)

    actual_query, actual_key = mla_prepare_ops.composite_mla_prepare_fp8_query(
        **inputs,
        absorbed_query=absorbed_query,
        enable_pdl=True,
    )

    assert actual_query is query_result
    assert actual_key is key_result
    assert torch.all(absorbed_query[..., :latent_dim] == 3.0)
    assert torch.all(absorbed_query[..., latent_dim:] == -1.0)
    assert len(apply_calls) == 1
    call = apply_calls[0]
    assert call["q_nope"].data_ptr() == absorbed_query[..., :latent_dim].data_ptr()
    assert call["k_nope"] is inputs["k_nope"]
    assert "q_nope_prequantized" not in call
    assert "k_nope_prequantized" not in call
    assert "key_out" not in call


def test_composite_mla_prepare_rejects_prequantized_key() -> None:
    inputs = _cpu_inputs()
    tokens = inputs["q_nope"].shape[0]
    latent_dim = inputs["projection_weight"].shape[1]
    rope_dim = inputs["q_rope"].shape[2]
    prequantized_key = torch.empty(
        tokens,
        1,
        latent_dim + rope_dim,
        dtype=torch.float8_e4m3fn,
    )

    with pytest.raises(ValueError, match="raw normalized key"):
        mla_prepare_ops.composite_mla_prepare_fp8_query(
            **inputs,
            prequantized_key=prequantized_key,
        )


def test_producer_mla_prepare_uses_prequantized_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _cpu_inputs()
    tokens, heads, _ = inputs["q_nope"].shape
    latent_dim = inputs["projection_weight"].shape[1]
    rope_dim = inputs["q_rope"].shape[2]
    prequantized_key = torch.full(
        (tokens, 1, latent_dim + rope_dim),
        7.0,
        dtype=torch.float8_e4m3fn,
    )
    apply_calls: list[dict[str, object]] = []

    def fake_bmm_fp8_output(A, B, *, out, enable_pdl):
        assert A.shape[:2] == (heads, tokens)
        assert B is inputs["projection_weight"]
        assert enable_pdl
        out.fill_(2.0)
        return out

    def fake_rope_mla_fp8_producer(**kwargs):
        apply_calls.append(kwargs)
        return kwargs["query_out"], kwargs["key_out"]

    monkeypatch.setattr(gemm_ops, "bmm_fp8_output", fake_bmm_fp8_output)
    monkeypatch.setattr(
        embedding_ops,
        "rope_mla_fp8_producer",
        fake_rope_mla_fp8_producer,
        raising=False,
    )

    actual_query, actual_key = mla_prepare_ops.producer_mla_prepare_fp8_query_amd(
        **inputs,
        prequantized_key=prequantized_key,
        enable_pdl=True,
    )

    assert actual_key is prequantized_key
    assert len(apply_calls) == 1
    call = apply_calls[0]
    assert call["k_nope"] is inputs["k_nope"]
    assert call["key_out"] is prequantized_key
    assert torch.all(actual_query[..., :latent_dim] == 2.0)


@pytest.mark.parametrize(
    "traits_update",
    [
        pytest.param({}, id="eligible"),
        pytest.param({"has_prequantized_key": True}, id="prequantized-key"),
        pytest.param({"has_absorbed_query": True}, id="caller-output"),
        pytest.param({"q_scale_is_one": False}, id="scaled-query"),
        pytest.param({"has_rope": False}, id="no-rope"),
    ],
)
def test_mla_prepare_strategy_selection(
    mi350_platform,
    traits_update: dict[str, object],
) -> None:
    traits = {
        "num_tokens": 16,
        "num_heads": 16,
        "q_nope_dim": 192,
        "kv_lora_rank": 512,
        "has_rope": True,
        "has_absorbed_query": False,
        "has_prequantized_key": False,
        "q_scale_is_one": True,
        **traits_update,
    }
    registry = KernelRegistry.get()
    registry.clear_cache()
    try:
        selected = select_kernel(
            "attention",
            "mla_prepare_fp8_query",
            mla_prepare_ops._MLA_PREPARE_FP8_SIGNATURE,
            platform=mi350_platform,
            traits=traits,
        )
    finally:
        registry.clear_cache()

    expected = (
        "composite_mla_prepare_fp8_query"
        if any(
            key in traits_update
            for key in ("has_absorbed_query", "q_scale_is_one", "has_rope")
        )
        else "producer_mla_prepare_fp8_query_amd"
    )
    assert selected.name == expected


def test_mla_prepare_producer_strategy_is_available_on_amd(mi450_platform) -> None:
    registry = KernelRegistry.get()
    registry.clear_cache()
    try:
        selected = select_kernel(
            "attention",
            "mla_prepare_fp8_query",
            mla_prepare_ops._MLA_PREPARE_FP8_SIGNATURE,
            platform=mi450_platform,
            traits={
                "has_rope": True,
                "has_absorbed_query": False,
                "has_prequantized_key": True,
                "q_scale_is_one": True,
            },
        )
    finally:
        registry.clear_cache()

    assert selected.name == "producer_mla_prepare_fp8_query_amd"


def test_mla_prepare_has_no_nvidia_kernel(h100_platform) -> None:
    registry = KernelRegistry.get()
    registry.clear_cache()
    try:
        with pytest.raises(NoKernelFoundError):
            select_kernel(
                "attention",
                "mla_prepare_fp8_query",
                mla_prepare_ops._MLA_PREPARE_FP8_SIGNATURE,
                platform=h100_platform,
                traits={
                    "has_rope": True,
                    "has_absorbed_query": False,
                    "has_prequantized_key": True,
                    "q_scale_is_one": True,
                },
            )
    finally:
        registry.clear_cache()


@pytest.mark.parametrize(
    ("update", "message"),
    [
        pytest.param(
            {"q_nope": torch.empty(0, 3, 4, dtype=torch.bfloat16)},
            "token count",
            id="empty",
        ),
        pytest.param(
            {"positions": torch.arange(2, dtype=torch.float32)},
            "positions must use",
            id="positions-dtype",
        ),
        pytest.param(
            {"cos_sin_cache": torch.empty(16, 8)},
            "cache width",
            id="cache-width",
        ),
        pytest.param(
            {"quant_scale_q": torch.ones(2)},
            "must contain one value",
            id="scale-shape",
        ),
    ],
)
def test_mla_prepare_validates_before_projection(
    monkeypatch: pytest.MonkeyPatch,
    update: dict[str, object],
    message: str,
) -> None:
    inputs: dict[str, object] = _cpu_inputs()
    inputs.update(update)
    bmm = Mock()
    monkeypatch.setattr(gemm_ops, "bmm", bmm)

    with pytest.raises((TypeError, ValueError), match=message):
        mla_prepare_fp8_query(**inputs, solution="composite")

    bmm.assert_not_called()


@pytest.mark.parametrize("unsupported", ["no-rope", "absorbed-query", "scaled-query"])
def test_prequantized_key_rejects_raw_composition_conditions(
    unsupported: str,
) -> None:
    inputs = _cpu_inputs()
    tokens, heads, _ = inputs["q_nope"].shape
    latent_dim = inputs["projection_weight"].shape[1]
    rope_dim = inputs["q_rope"].shape[2]
    prequantized_key = torch.empty(
        tokens,
        1,
        latent_dim + rope_dim,
        dtype=torch.float8_e4m3fn,
    )
    kwargs: dict[str, object] = {"prequantized_key": prequantized_key}
    if unsupported == "no-rope":
        inputs["cos_sin_cache"] = None
    elif unsupported == "absorbed-query":
        kwargs["absorbed_query"] = torch.empty(
            tokens,
            heads,
            latent_dim + rope_dim,
            dtype=torch.bfloat16,
        )
    else:
        kwargs["quant_scale_q"] = 0.5

    with pytest.raises(ValueError, match="requires RoPE"):
        mla_prepare_fp8_query(**inputs, **kwargs)


def test_composite_mla_prepare_gpu_matches_explicit_composition(
    device: str,
    require,
) -> None:
    require(
        "attention",
        "mla_prepare_fp8_query",
        "composite",
        torch.bfloat16,
        "q_nope",
    )
    from tokenspeed_kernel.ops.embedding import apply_rope_mla
    from tokenspeed_kernel.ops.gemm import bmm

    inputs = _gpu_inputs(
        device,
        tokens=5,
        heads=4,
        kv_heads=1,
        q_dim=24,
        latent_dim=32,
        rope_dim=16,
    )
    tokens, heads, _ = inputs["q_nope"].shape
    latent_dim = inputs["projection_weight"].shape[1]
    projected_nope = torch.empty(
        tokens, heads, latent_dim, device=device, dtype=torch.bfloat16
    )
    bmm(
        inputs["q_nope"].transpose(0, 1),
        inputs["projection_weight"],
        out=projected_nope.transpose(0, 1),
    )
    expected_query, expected_key = apply_rope_mla(
        positions=inputs["positions"],
        q_rope=inputs["q_rope"],
        k_rope=inputs["k_rope"],
        q_nope=projected_nope,
        k_nope=inputs["k_nope"],
        cos_sin_cache=inputs["cos_sin_cache"],
        quant_scale_q=1.25,
        quant_scale_kv=0.75,
    )
    actual_query, actual_key = mla_prepare_fp8_query(
        **inputs,
        quant_scale_q=1.25,
        quant_scale_kv=0.75,
        solution="composite",
    )

    assert torch.equal(actual_query.view(torch.uint8), expected_query.view(torch.uint8))
    assert torch.equal(actual_key.view(torch.uint8), expected_key.view(torch.uint8))


def test_producer_mla_prepare_gpu_matches_portable_reference(
    device: str,
    require,
) -> None:
    from tokenspeed_kernel.ops.embedding import apply_rope_mla
    from tokenspeed_kernel.ops.gemm import bmm

    require(
        "attention",
        "mla_prepare_fp8_query",
        "producer",
        torch.bfloat16,
        "q_nope",
    )
    inputs = _gpu_inputs(
        device,
        tokens=3,
        heads=16,
        kv_heads=1,
        q_dim=192,
        latent_dim=512,
        rope_dim=64,
    )
    tokens, heads, _ = inputs["q_nope"].shape
    latent_dim = inputs["projection_weight"].shape[1]
    rope_dim = inputs["q_rope"].shape[2]
    projected_nope = torch.empty(
        tokens, heads, latent_dim, device=device, dtype=torch.bfloat16
    )
    bmm(
        inputs["q_nope"].transpose(0, 1),
        inputs["projection_weight"],
        out=projected_nope.transpose(0, 1),
    )
    expected_query, expected_key = apply_rope_mla(
        positions=inputs["positions"],
        q_rope=inputs["q_rope"],
        k_rope=inputs["k_rope"],
        q_nope=projected_nope,
        k_nope=inputs["k_nope"],
        cos_sin_cache=inputs["cos_sin_cache"],
        solution="triton",
    )
    prequantized_key = torch.empty(
        tokens,
        1,
        latent_dim + rope_dim,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    prequantized_key[..., :latent_dim].copy_(inputs["k_nope"].to(torch.float8_e4m3fn))

    actual_query, actual_key = mla_prepare_fp8_query(
        **inputs,
        prequantized_key=prequantized_key,
        enable_pdl=True,
        solution="producer",
    )

    assert actual_key is prequantized_key
    torch.testing.assert_close(
        actual_query.float(), expected_query.float(), rtol=0, atol=0.5
    )
    torch.testing.assert_close(
        actual_key.float(), expected_key.float(), rtol=0, atol=0.5
    )
    assert torch.equal(
        actual_key[..., :latent_dim].view(torch.uint8),
        inputs["k_nope"].to(torch.float8_e4m3fn).view(torch.uint8),
    )


def test_mla_prepare_auto_selects_native_producer(
    device: str,
    require,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require(
        "attention",
        "mla_prepare_fp8_query",
        "producer",
        torch.bfloat16,
        "q_nope",
    )
    inputs = _gpu_inputs(
        device,
        tokens=3,
        heads=16,
        kv_heads=1,
        q_dim=192,
        latent_dim=512,
        rope_dim=64,
    )
    latent_dim = inputs["projection_weight"].shape[1]
    prequantized_key = torch.empty(
        3,
        1,
        576,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    prequantized_key[..., :latent_dim].copy_(inputs["k_nope"].to(torch.float8_e4m3fn))

    def fail_portable(*args, **kwargs):
        raise AssertionError("eligible gfx950 input unexpectedly used portable path")

    monkeypatch.setattr(
        mla_prepare_ops,
        "_portable_mla_prepare_fp8_query",
        fail_portable,
    )

    query, key = mla_prepare_fp8_query(
        **inputs,
        prequantized_key=prequantized_key,
    )

    assert query.shape == (3, 16, 576)
    assert query.dtype == torch.float8_e4m3fn
    assert key is prequantized_key


def test_mla_prepare_preserves_prequantized_key_with_composite_bmm(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    require,
) -> None:
    require(
        "attention",
        "mla_prepare_fp8_query",
        "producer",
        torch.bfloat16,
        "q_nope",
    )
    require(
        "gemm",
        "bmm_fp8_output",
        "composite",
        torch.bfloat16,
        "a",
    )
    inputs = _gpu_inputs(
        device,
        tokens=33,
        heads=16,
        kv_heads=1,
        q_dim=192,
        latent_dim=512,
        rope_dim=64,
    )

    latent_dim = inputs["projection_weight"].shape[1]
    prequantized_key = torch.empty(
        33,
        1,
        576,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    prequantized_key[..., :latent_dim].fill_(3.0)
    expected_prefix = prequantized_key[..., :latent_dim].clone()
    inputs["k_nope"].zero_()

    original_bmm_fp8_output = gemm_ops.bmm_fp8_output

    def force_composite_bmm(*args, **kwargs):
        return original_bmm_fp8_output(*args, **kwargs, solution="composite")

    monkeypatch.setattr(gemm_ops, "bmm_fp8_output", force_composite_bmm)

    query, key = mla_prepare_fp8_query(
        **inputs,
        prequantized_key=prequantized_key,
    )

    assert query.shape == (33, 16, 576)
    assert key is prequantized_key
    assert query.dtype == key.dtype == torch.float8_e4m3fn
    assert torch.equal(
        key[..., :latent_dim].view(torch.uint8),
        expected_prefix.view(torch.uint8),
    )


def test_mla_prepare_producer_path_captures_and_replays(
    device: str,
    require,
) -> None:
    require(
        "attention",
        "mla_prepare_fp8_query",
        "producer",
        torch.bfloat16,
        "q_nope",
    )
    inputs = _gpu_inputs(
        device,
        tokens=3,
        heads=16,
        kv_heads=1,
        q_dim=192,
        latent_dim=512,
        rope_dim=64,
    )
    prequantized_key = torch.empty(
        3,
        1,
        576,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    prequantized_key[..., :512].copy_(inputs["k_nope"].to(torch.float8_e4m3fn))
    expected_query, _ = mla_prepare_fp8_query(
        **inputs,
        prequantized_key=prequantized_key,
    )
    expected_query = expected_query.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_query, captured_key = mla_prepare_fp8_query(
            **inputs,
            prequantized_key=prequantized_key,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert captured_key is prequantized_key
    assert torch.equal(
        captured_query.view(torch.uint8),
        expected_query.view(torch.uint8),
    )

    inputs["q_nope"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(captured_query[..., :512]) == 0
