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

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models import deepseek_v3, glm5


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


class _RecordingCachePool:
    def __init__(self) -> None:
        self.mla_writes: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def set_mla_kv_buffer(self, *args, **kwargs) -> None:
        self.mla_writes.append((args, kwargs))


@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "qk_nope_head_dim", "kv_lora_rank"),
    (
        (1, 16, 192, 512),
        (4, 16, 192, 512),
        (17, 16, 192, 512),
        (32, 16, 192, 512),
        (33, 16, 192, 512),
        (64, 16, 192, 512),
        (7, 4, 96, 160),
    ),
)
def test_amd_absorbed_projection_uses_fp8_producer(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    num_heads: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
) -> None:
    calls: dict[str, object] = {}
    rope_head_dim = 64
    qk_head_dim = qk_nope_head_dim + rope_head_dim
    combined_width = kv_lora_rank + rope_head_dim
    selected_cache_locations = torch.arange(num_tokens, dtype=torch.int32) + 100

    class _AttentionBackend:
        full_attn_backend = SimpleNamespace(data_type=torch.float8_e4m3fn)

        def select_out_cache_loc(self, layer, locations, forward_mode):
            calls["cache_location_selection"] = (layer, locations, forward_mode)
            return selected_cache_locations

    def fake_bmm(*args, **kwargs):
        raise AssertionError("the BF16 projection path must not be used")

    def fake_mla_prepare_fp8_query(**kwargs):
        calls["prepare"] = kwargs
        query_output = torch.empty(
            (num_tokens, num_heads, combined_width),
            dtype=torch.float8_e4m3fn,
        )
        return query_output, kwargs["prequantized_key"]

    monkeypatch.setattr(deepseek_v3, "bmm", fake_bmm)
    monkeypatch.setattr(
        deepseek_v3,
        "mla_prepare_fp8_query",
        fake_mla_prepare_fp8_query,
        raising=False,
    )
    monkeypatch.setattr(deepseek_v3, "_is_amd", True)
    monkeypatch.setattr(deepseek_v3, "pdl_enabled", lambda: True)

    layer = SimpleNamespace(k_scale_float=1.0)
    rotary = SimpleNamespace(
        cos_sin_cache=torch.empty((32, 64), dtype=torch.bfloat16),
        is_neox_style=False,
    )
    model = SimpleNamespace(
        attention_backend="dsa",
        _MLA_KERNEL_BACKENDS=("dsa",),
        attn_mqa=layer,
        rotary_emb=rotary,
        num_local_heads=num_heads,
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        w_kc=torch.empty(
            (num_heads, qk_nope_head_dim, kv_lora_rank),
            dtype=torch.bfloat16,
        ),
    )
    model._mla_kv_is_fp8 = lambda context, scale: (
        deepseek_v3.DeepseekV3AttentionMLA._mla_kv_is_fp8(model, context, scale)
    )
    cache_pool = _RecordingCachePool()
    backend = _AttentionBackend()
    ctx = SimpleNamespace(
        attn_backend=backend,
        token_to_kv_pool=cache_pool,
        forward_mode=ForwardMode.DECODE,
    )

    query = torch.empty((num_tokens, num_heads * qk_head_dim), dtype=torch.bfloat16)
    latent_cache = torch.empty((num_tokens, combined_width), dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    requested_cache_locations = torch.arange(num_tokens, dtype=torch.int32)
    key_backing = torch.empty(
        (num_tokens + 2, 1, combined_width),
        dtype=torch.float8_e4m3fn,
    )
    key_output = key_backing[1 : num_tokens + 1]
    assert key_output.is_contiguous()
    assert key_output.storage_offset() == combined_width

    query_output, returned_key = (
        deepseek_v3.DeepseekV3AttentionMLA.forward_absorb_qkv_proj(
            model,
            query,
            latent_cache,
            positions,
            ctx,
            requested_cache_locations,
            key_fp8=key_output,
        )
    )

    assert query_output.shape == (num_tokens, num_heads, combined_width)
    assert query_output.dtype is torch.float8_e4m3fn
    assert returned_key is key_output

    prepare_call = calls["prepare"]
    assert prepare_call["q_nope"].shape == (
        num_tokens,
        num_heads,
        qk_nope_head_dim,
    )
    assert prepare_call["projection_weight"].shape == (
        num_heads,
        kv_lora_rank,
        qk_nope_head_dim,
    )
    assert prepare_call["positions"] is positions
    assert prepare_call["q_rope"].shape == (
        num_tokens,
        num_heads,
        rope_head_dim,
    )
    assert prepare_call["q_rope"].dtype is torch.bfloat16
    assert prepare_call["k_nope"].shape == (num_tokens, 1, kv_lora_rank)
    assert prepare_call["k_rope"].shape == (num_tokens, 1, rope_head_dim)
    assert prepare_call["cos_sin_cache"] is rotary.cos_sin_cache
    assert prepare_call["is_neox"] is False
    assert prepare_call["absorbed_query"] is None
    assert prepare_call["prequantized_key"] is key_output
    assert prepare_call["quant_scale_q"] == 1.0
    assert prepare_call["quant_scale_kv"] == 1.0
    assert prepare_call["enable_pdl"] is True

    selected_layer, original_locations, forward_mode = calls["cache_location_selection"]
    assert selected_layer is layer
    assert original_locations is requested_cache_locations
    assert forward_mode is ForwardMode.DECODE

    assert len(cache_pool.mla_writes) == 1
    cache_args, cache_kwargs = cache_pool.mla_writes[0]
    assert cache_args[0] is layer
    assert cache_args[1] is selected_cache_locations
    cache_nope = cache_kwargs["cache_k_nope"]
    cache_rope = cache_kwargs["cache_k_rope"]
    assert cache_nope.shape == (num_tokens, 1, kv_lora_rank)
    assert cache_rope.shape == (num_tokens, 1, rope_head_dim)
    assert _storage_ptr(cache_nope) == _storage_ptr(key_output)
    assert _storage_ptr(cache_rope) == _storage_ptr(key_output)
    assert cache_nope.storage_offset() == key_output.storage_offset()
    assert cache_rope.storage_offset() == key_output.storage_offset() + kv_lora_rank


def test_caller_owned_absorbed_query_is_delegated_to_fp8_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_tokens = 2
    calls: dict[str, object] = {}

    class _AttentionBackend:
        full_attn_backend = SimpleNamespace(data_type=torch.float8_e4m3fn)

        @staticmethod
        def select_out_cache_loc(layer, locations, forward_mode):
            return locations

    def fake_bmm(*args, **kwargs):
        raise AssertionError("the model must delegate FP8 MLA preparation")

    def fake_mla_prepare_fp8_query(**kwargs):
        calls["prepare"] = kwargs
        query = torch.empty((num_tokens, 16, 576), dtype=torch.float8_e4m3fn)
        key = torch.empty((num_tokens, 1, 576), dtype=torch.float8_e4m3fn)
        return query, key

    monkeypatch.setattr(deepseek_v3, "bmm", fake_bmm)
    monkeypatch.setattr(
        deepseek_v3,
        "mla_prepare_fp8_query",
        fake_mla_prepare_fp8_query,
        raising=False,
    )
    monkeypatch.setattr(deepseek_v3, "_is_amd", True)
    monkeypatch.setattr(deepseek_v3, "pdl_enabled", lambda: False)

    layer = SimpleNamespace(k_scale_float=1.0)
    model = SimpleNamespace(
        attention_backend="dsa",
        _MLA_KERNEL_BACKENDS=("dsa",),
        attn_mqa=layer,
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.empty((32, 64), dtype=torch.bfloat16),
            is_neox_style=True,
        ),
        num_local_heads=16,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        kv_lora_rank=512,
        w_kc=torch.empty((16, 192, 512), dtype=torch.bfloat16),
    )
    model._mla_kv_is_fp8 = lambda context, scale: (
        deepseek_v3.DeepseekV3AttentionMLA._mla_kv_is_fp8(model, context, scale)
    )
    cache_pool = _RecordingCachePool()
    ctx = SimpleNamespace(
        attn_backend=_AttentionBackend(),
        token_to_kv_pool=cache_pool,
        forward_mode=ForwardMode.DECODE,
    )
    absorbed_query = torch.full(
        (num_tokens, 16, 576),
        7.0,
        dtype=torch.bfloat16,
    )
    query, key = deepseek_v3.DeepseekV3AttentionMLA.forward_absorb_qkv_proj(
        model,
        torch.empty((num_tokens, 16, 192), dtype=torch.bfloat16),
        torch.empty((num_tokens, 576), dtype=torch.bfloat16),
        torch.arange(num_tokens, dtype=torch.int64),
        ctx,
        torch.arange(num_tokens, dtype=torch.int32),
        absorbed_query=absorbed_query,
    )

    prepare_call = calls["prepare"]
    assert prepare_call["q_nope"].shape == (num_tokens, 16, 192)
    assert prepare_call["projection_weight"].shape == (16, 512, 192)
    assert prepare_call["absorbed_query"] is absorbed_query
    assert prepare_call["prequantized_key"] is None
    assert _storage_ptr(prepare_call["q_rope"]) == _storage_ptr(absorbed_query)
    assert prepare_call["q_rope"].storage_offset() == 512
    assert query.dtype is torch.float8_e4m3fn
    assert key.dtype is torch.float8_e4m3fn


def test_non_amd_fp8_projection_keeps_legacy_bmm_rope_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_tokens, num_heads = 2, 3
    q_nope_dim, rope_dim, latent_dim = 4, 2, 5
    events: list[str] = []
    calls: dict[str, object] = {}
    selected_cache_locations = torch.tensor([7, 9], dtype=torch.int32)

    class _AttentionBackend:
        full_attn_backend = SimpleNamespace(data_type=torch.float8_e4m3fn)

        @staticmethod
        def select_out_cache_loc(layer, locations, forward_mode):
            events.append("select_cache_location")
            calls["requested_cache_locations"] = locations
            return selected_cache_locations

    class _CachePool:
        @staticmethod
        def set_mla_kv_buffer(*args, **kwargs) -> None:
            events.append("cache_write")
            calls["cache_write"] = (args, kwargs)

    def fake_bmm(a, b, *, out):
        events.append("bmm")
        calls["bmm"] = (a, b, out)
        out.fill_(2.0)
        return out

    def fail_fp8_producer(**kwargs):
        raise AssertionError("non-AMD execution must not call the producer kernel")

    def fake_apply_rope_mla(**kwargs):
        events.append("apply_rope_mla")
        calls["apply_rope_mla"] = kwargs
        torch.testing.assert_close(
            kwargs["q_nope"],
            torch.full_like(kwargs["q_nope"], 2.0),
            rtol=0,
            atol=0,
        )
        return (
            torch.empty(
                (num_tokens, num_heads, latent_dim + rope_dim),
                dtype=torch.float8_e4m3fn,
            ),
            torch.empty(
                (num_tokens, 1, latent_dim + rope_dim),
                dtype=torch.float8_e4m3fn,
            ),
        )

    monkeypatch.setattr(deepseek_v3, "_is_amd", False)
    monkeypatch.setattr(deepseek_v3, "bmm", fake_bmm)
    monkeypatch.setattr(
        deepseek_v3,
        "mla_prepare_fp8_query",
        fail_fp8_producer,
        raising=False,
    )
    monkeypatch.setattr(deepseek_v3, "apply_rope_mla", fake_apply_rope_mla)
    monkeypatch.setattr(deepseek_v3, "pdl_enabled", lambda: False)

    layer = SimpleNamespace(k_scale_float=1.0)
    rotary = SimpleNamespace(
        cos_sin_cache=torch.empty((32, rope_dim), dtype=torch.bfloat16),
        is_neox_style=False,
    )
    model = SimpleNamespace(
        attention_backend="dsa",
        _MLA_KERNEL_BACKENDS=("dsa",),
        attn_mqa=layer,
        rotary_emb=rotary,
        num_local_heads=num_heads,
        qk_head_dim=q_nope_dim + rope_dim,
        qk_nope_head_dim=q_nope_dim,
        qk_rope_head_dim=rope_dim,
        kv_lora_rank=latent_dim,
        w_kc=torch.empty(
            (num_heads, q_nope_dim, latent_dim),
            dtype=torch.bfloat16,
        ),
    )

    def is_fp8(context, scale):
        events.append("fp8_gate")
        assert scale == 1.0
        return True

    model._mla_kv_is_fp8 = is_fp8
    ctx = SimpleNamespace(
        attn_backend=_AttentionBackend(),
        token_to_kv_pool=_CachePool(),
        forward_mode=ForwardMode.DECODE,
    )
    positions = torch.arange(num_tokens, dtype=torch.int64)
    requested_cache_locations = torch.arange(num_tokens, dtype=torch.int32)

    query_output, key_output = (
        deepseek_v3.DeepseekV3AttentionMLA.forward_absorb_qkv_proj(
            model,
            torch.empty(
                (num_tokens, num_heads * (q_nope_dim + rope_dim)),
                dtype=torch.bfloat16,
            ),
            torch.empty(
                (num_tokens, latent_dim + rope_dim),
                dtype=torch.bfloat16,
            ),
            positions,
            ctx,
            requested_cache_locations,
        )
    )

    assert events == [
        "select_cache_location",
        "bmm",
        "fp8_gate",
        "apply_rope_mla",
        "cache_write",
    ]
    assert calls["requested_cache_locations"] is requested_cache_locations
    bmm_a, bmm_b, bmm_out = calls["bmm"]
    assert bmm_a.shape == (num_heads, num_tokens, q_nope_dim)
    assert bmm_b.shape == (num_heads, latent_dim, q_nope_dim)
    assert bmm_out.shape == (num_heads, num_tokens, latent_dim)

    rope_call = calls["apply_rope_mla"]
    assert rope_call["positions"] is positions
    assert rope_call["q_rope"].shape == (num_tokens, num_heads, rope_dim)
    assert rope_call["k_rope"].shape == (num_tokens, 1, rope_dim)
    assert rope_call["q_nope"].shape == (num_tokens, num_heads, latent_dim)
    assert rope_call["k_nope"].shape == (num_tokens, 1, latent_dim)
    assert rope_call["cos_sin_cache"] is rotary.cos_sin_cache
    assert rope_call["is_neox"] is False
    assert rope_call["quant_scale_q"] == 1.0
    assert rope_call["quant_scale_kv"] == 1.0
    assert rope_call["enable_pdl"] is False
    assert query_output.dtype is torch.float8_e4m3fn
    assert key_output.dtype is torch.float8_e4m3fn

    cache_args, cache_kwargs = calls["cache_write"]
    assert cache_args[0] is layer
    assert cache_args[1] is selected_cache_locations
    assert cache_kwargs["cache_k_nope"].shape == (num_tokens, 1, latent_dim)
    assert cache_kwargs["cache_k_rope"].shape == (num_tokens, 1, rope_dim)


def test_absorbed_projection_keeps_bf16_path_for_bf16_mla_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_tokens, num_heads = 2, 3
    q_nope_dim, rope_dim, latent_dim = 4, 2, 5
    calls: dict[str, object] = {}

    class _AttentionBackend:
        full_attn_backend = SimpleNamespace(data_type=torch.bfloat16)

        @staticmethod
        def select_out_cache_loc(layer, locations, forward_mode):
            return locations

    def fake_bmm(a, b, *, out):
        calls["bmm"] = (a, b, out)
        out.fill_(1.0)
        return out

    def fail_fp8_prepare(**kwargs):
        raise AssertionError("BF16 MLA cache must not request FP8 preparation")

    monkeypatch.setattr(deepseek_v3, "bmm", fake_bmm)
    monkeypatch.setattr(
        deepseek_v3,
        "mla_prepare_fp8_query",
        fail_fp8_prepare,
        raising=False,
    )
    monkeypatch.setattr(deepseek_v3, "_is_amd", True)

    layer = SimpleNamespace(k_scale_float=1.0)
    model = SimpleNamespace(
        attention_backend="dsa",
        _MLA_KERNEL_BACKENDS=("dsa",),
        attn_mqa=layer,
        rotary_emb=None,
        num_local_heads=num_heads,
        qk_head_dim=q_nope_dim + rope_dim,
        qk_nope_head_dim=q_nope_dim,
        qk_rope_head_dim=rope_dim,
        kv_lora_rank=latent_dim,
        w_kc=torch.empty(
            (num_heads, q_nope_dim, latent_dim),
            dtype=torch.bfloat16,
        ),
    )
    model._mla_kv_is_fp8 = lambda context, scale: (
        deepseek_v3.DeepseekV3AttentionMLA._mla_kv_is_fp8(model, context, scale)
    )
    cache_pool = _RecordingCachePool()
    ctx = SimpleNamespace(
        attn_backend=_AttentionBackend(),
        token_to_kv_pool=cache_pool,
        forward_mode=ForwardMode.DECODE,
    )
    query = torch.arange(
        num_tokens * num_heads * (q_nope_dim + rope_dim),
        dtype=torch.bfloat16,
    ).view(num_tokens, -1)
    latent_cache = torch.empty(
        (num_tokens, latent_dim + rope_dim),
        dtype=torch.bfloat16,
    )

    query_output, key_output = (
        deepseek_v3.DeepseekV3AttentionMLA.forward_absorb_qkv_proj(
            model,
            query,
            latent_cache,
            torch.arange(num_tokens),
            ctx,
            torch.arange(num_tokens, dtype=torch.int32),
        )
    )

    bmm_a, bmm_b, bmm_out = calls["bmm"]
    assert bmm_a.shape == (num_heads, num_tokens, q_nope_dim)
    assert bmm_b.shape == (num_heads, latent_dim, q_nope_dim)
    assert bmm_out.shape == (num_heads, num_tokens, latent_dim)
    assert query_output.shape == (num_tokens, num_heads, latent_dim + rope_dim)
    assert query_output.dtype is torch.bfloat16
    assert _storage_ptr(bmm_out) == _storage_ptr(query_output)
    assert key_output.shape == (num_tokens, 1, latent_dim + rope_dim)
    assert key_output.dtype is torch.bfloat16
    assert len(cache_pool.mla_writes) == 1


@pytest.mark.parametrize(
    ("is_amd", "k_scale", "expects_prequantized_key"),
    [
        pytest.param(False, 1.0, False, id="non-amd"),
        pytest.param(True, 1.0, True, id="amd-unit-scale"),
        pytest.param(True, 0.5, False, id="amd-nonunit-scale"),
    ],
)
def test_mixed_forward_prepares_normalized_fp8_key_only_when_eligible(
    monkeypatch: pytest.MonkeyPatch,
    is_amd: bool,
    k_scale: float,
    expects_prequantized_key: bool,
) -> None:
    num_prefill_tokens = 3
    num_decode_tokens = 1
    num_tokens = num_prefill_tokens + num_decode_tokens
    calls: dict[str, object] = {}

    monkeypatch.setattr(glm5, "_is_amd", is_amd)
    monkeypatch.setattr(glm5, "current_forward_ctx", lambda: None)

    def fused_projection(hidden_states, block_scale, output_dtype):
        assert block_scale is None
        assert output_dtype is torch.bfloat16
        return torch.empty((hidden_states.shape[0], 8 + 576), dtype=output_dtype)

    def fused_norm(**kwargs):
        calls["norm_kwargs"] = kwargs
        input_q_a = kwargs["input_q_a"]
        input_kv_a = kwargs["input_kv_a"]
        output_q_a = kwargs["output_q_a"]
        assert input_q_a.shape == output_q_a.shape == (num_tokens, 8)
        assert input_kv_a.shape == (num_tokens, 512)
        if expects_prequantized_key:
            output_kv_a = kwargs["output_kv_a"]
            assert output_kv_a.shape == (num_tokens, 512)
            assert output_kv_a.dtype is torch.float8_e4m3fn
        else:
            assert "output_kv_a" not in kwargs

    def q_projection(q_norm):
        return torch.empty((q_norm.shape[0], 16 * 256), dtype=q_norm.dtype), None

    def sparse_prefill(
        positions,
        query,
        latent_cache,
        ctx,
        out_cache_loc,
        output,
        **kwargs,
    ):
        calls["prefill_kwargs"] = kwargs
        assert positions.shape[0] == query.shape[0] == latent_cache.shape[0]
        assert ctx.forward_mode is ForwardMode.EXTEND
        assert kwargs["prefill_topk"] is prefill_metadata
        return output

    def decode(
        positions,
        query,
        latent_cache,
        ctx,
        out_cache_loc,
        output,
        **kwargs,
    ):
        calls["decode_kwargs"] = kwargs
        assert positions.shape[0] == query.shape[0] == latent_cache.shape[0]
        assert ctx.forward_mode is ForwardMode.DECODE
        topk_indices = kwargs["topk_indices"]
        topk_lens = kwargs["topk_lens"]
        assert topk_indices.shape == topk_lens.shape == (num_decode_tokens,)
        return output

    decode_metadata = SimpleNamespace(
        num_extends=1,
        seq_lens_k=torch.zeros(2, dtype=torch.int32),
        block_kv_indices=torch.zeros((2, 1), dtype=torch.int32),
    )
    backend = SimpleNamespace(
        full_attn_backend=SimpleNamespace(data_type=torch.float8_e4m3fn),
        forward_metadata=None,
        forward_decode_metadata=decode_metadata,
        spec_num_tokens=1,
    )
    prefill_metadata = object()
    decode_topk = glm5.GlmDsaDecodeTopK(
        topk_indices=torch.arange(num_tokens, dtype=torch.int32),
        topk_lens=torch.ones(num_tokens, dtype=torch.int32),
    )
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(),
        bs=2,
        num_extends=1,
        input_num_tokens=num_tokens,
        forward_mode=ForwardMode.MIXED,
        dsa_prefill_topk=prefill_metadata,
        dsa_decode_topk=decode_topk,
    )
    model = SimpleNamespace(
        q_lora_rank=8,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fused_qkv_a_proj_with_mqa=fused_projection,
        fused_qk_layernorm=fused_norm,
        attention_backend="dsa",
        _MLA_KERNEL_BACKENDS=("dsa",),
        rotary_emb=object(),
        attn_mqa=SimpleNamespace(k_scale_float=k_scale),
        skip_indexer_topk=True,
        is_nextn=False,
        q_b_proj=q_projection,
        num_local_heads=16,
        v_head_dim=128,
        _resolve_decode_window=glm5.GlmMoeDsaAttention._resolve_decode_window,
        _slice_decode_topk=glm5.GlmMoeDsaAttention._slice_decode_topk,
        forward_dsa_sparse_prefill=sparse_prefill,
        forward_absorb=decode,
        o_proj=lambda output: (output, None),
    )
    model._mla_kv_is_fp8 = lambda context, scale: (
        deepseek_v3.DeepseekV3AttentionMLA._mla_kv_is_fp8(model, context, scale)
    )
    comm_manager = SimpleNamespace(pre_attn_comm=lambda value, _ctx: value)

    output = glm5.GlmMoeDsaAttention.forward(
        model,
        torch.arange(num_tokens, dtype=torch.int64),
        torch.empty((num_tokens, 32), dtype=torch.bfloat16),
        ctx,
        torch.arange(num_tokens, dtype=torch.int32),
        comm_manager,
    )

    assert output.shape == (num_tokens, 16 * 128)
    norm_kwargs = calls["norm_kwargs"]
    prefill_kwargs = calls["prefill_kwargs"]
    decode_kwargs = calls["decode_kwargs"]
    if not expects_prequantized_key:
        assert "output_kv_a" not in norm_kwargs
        assert "key_fp8" not in prefill_kwargs
        assert "key_fp8" not in decode_kwargs
        return

    norm_key_prefix = norm_kwargs["output_kv_a"]
    prefill_key = prefill_kwargs["key_fp8"]
    decode_key = decode_kwargs["key_fp8"]
    assert norm_key_prefix.shape == (num_tokens, 512)
    assert prefill_key.shape == (num_prefill_tokens, 1, 576)
    assert decode_key.shape == (num_decode_tokens, 1, 576)
    assert prefill_key.is_contiguous()
    assert decode_key.is_contiguous()
    assert _storage_ptr(norm_key_prefix) == _storage_ptr(prefill_key)
    assert _storage_ptr(prefill_key) == _storage_ptr(decode_key)
    assert norm_key_prefix.storage_offset() == prefill_key.storage_offset() == 0
    assert decode_key.storage_offset() == num_prefill_tokens * 576
