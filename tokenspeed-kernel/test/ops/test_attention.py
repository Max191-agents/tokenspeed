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
from tokenspeed_kernel import (
    attn_merge_state,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
    mla_decode_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.numerics.attention_kernel_kwargs import (
    mha_decode_with_kvcache_kwargs,
    mha_extend_with_kvcache_kwargs,
    mha_prefill_kwargs,
    mla_decode_with_kvcache_kwargs,
    mla_prefill_kwargs,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators.attention import _AttentionMergeStateGenerator
from tokenspeed_numerics_input_generators import (
    AttentionMergeStateInputConfig,
    MHAInputConfig,
    MHAInputs,
    MHARequestMetadataInputConfig,
    MLAInputConfig,
    MLAInputs,
    attention_merge_state_reference,
    mla_reference,
)

platform = current_platform()
torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})


def _mha_config(
    *,
    batch_size: int,
    total_cached_tokens: int,
    total_new_q_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    cache_layout: str = "none",
    page_size: int | None = None,
    indexing: str | None = None,
    include_sinks: bool = False,
    metadata_kwargs: dict[str, object] | None = None,
) -> MHAInputConfig:
    return MHAInputConfig(
        batch_size=batch_size,
        total_cached_tokens=total_cached_tokens,
        total_new_q_tokens=total_new_q_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        cache_layout=cache_layout,  # type: ignore[arg-type]
        page_size=page_size,
        indexing=indexing,  # type: ignore[arg-type]
        include_sinks=include_sinks,
        metadata_input=MHARequestMetadataInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            cache_layout=cache_layout,  # type: ignore[arg-type]
            **(metadata_kwargs or {}),
        ),
    )


def _mla_config(
    *,
    batch_size: int,
    total_cached_tokens: int,
    total_new_q_tokens: int,
    num_q_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    q_dtype: torch.dtype,
    cache_layout: str = "none",
    page_size: int | None = None,
    indexing: str | None = None,
    metadata_kwargs: dict[str, object] | None = None,
) -> MLAInputConfig:
    metadata_fields = {
        "allow_untied_non_cached_kv": True,
        "new_q_length_mode": (
            "fixed_per_request" if cache_layout != "none" else "ragged"
        ),
        **(metadata_kwargs or {}),
    }
    return MLAInputConfig(
        batch_size=batch_size,
        total_cached_tokens=total_cached_tokens,
        total_new_q_tokens=total_new_q_tokens,
        num_q_heads=num_q_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        q_dtype=q_dtype,
        cache_layout=cache_layout,  # type: ignore[arg-type]
        page_size=page_size,
        indexing=indexing,  # type: ignore[arg-type]
        metadata_input=MHARequestMetadataInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            cache_layout=cache_layout,  # type: ignore[arg-type]
            **metadata_fields,
        ),
    )


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 64, 8, 2)],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "gluon"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("is_sliding", [False, True], ids=["full", "sliding"])
def test_mha_prefill(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    has_sink: bool,
    is_sliding: bool,
    require,
) -> None:
    require("attention", "mha_prefill", solution, dtype, "q")
    if solution == "fa4" and (has_sink or is_sliding):
        pytest.skip("FA4 MHA prefill does not support sinks or sliding window")

    inputs = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=0,
            total_new_q_tokens=2818,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=dtype,
            include_sinks=has_sink,
            metadata_kwargs={
                "new_q_length_mode": "ragged",
                "max_new_q_tokens_per_request": 1100,
            },
        )
    ).generate(metadata_seed=31, value_seed=41, device=device)
    window_left = 127 if is_sliding else -1

    out = mha_prefill(
        **mha_prefill_kwargs(inputs, window_left=window_left),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer"])
def test_mha_extend_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_extend_with_kvcache", solution, dtype, "q")

    page_size = 64
    max_cache_seqlen = 256
    inputs = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=208,
            total_new_q_tokens=10,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=page_size,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": max_cache_seqlen,
                "cached_length_mode": "ragged",
                "max_cached_tokens_per_request": 128,
                "new_q_length_mode": "ragged",
                "max_new_q_tokens_per_request": 4,
            },
        )
    ).generate(metadata_seed=32, value_seed=42, device=device)
    kwargs = mha_extend_with_kvcache_kwargs(inputs)

    out = mha_extend_with_kvcache(
        **kwargs,
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape

    if solution == "triton":
        prefix_seqlens = torch.tensor(
            inputs.metadata.cached_lens_cpu,
            device=device,
            dtype=torch.int32,
        )
        triton_kwargs = dict(kwargs)
        triton_kwargs.update(
            cache_seqlens=prefix_seqlens,
            max_seqlen_k=int(prefix_seqlens.max().item()),
            return_lse=True,
        )
        triton_out, triton_lse = mha_extend_with_kvcache(
            **triton_kwargs,
            solution=solution,
        )

        assert triton_out.shape == inputs.q.shape
        assert triton_lse.shape == (inputs.q.shape[0], inputs.q.shape[1])


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer", "gluon"])
@pytest.mark.parametrize("seqlen_q", [1, 4], ids=["q1", "q4"])
def test_mha_decode_with_kvcache(
    device: str,
    solution: str,
    seqlen_q: int,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_decode_with_kvcache", solution, dtype, "q")

    page_size = 64
    max_cache_seqlen = 256
    inputs = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=400,
            total_new_q_tokens=4 * seqlen_q,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=page_size,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": max_cache_seqlen,
                "cached_length_mode": "ragged",
                "max_cached_tokens_per_request": max_cache_seqlen - seqlen_q,
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=33 + seqlen_q, value_seed=43, device=device)

    out = mha_decode_with_kvcache(
        **mha_decode_with_kvcache_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "dtype,num_heads,qk_head_dim,v_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 192, 128, id="bf16"),
        pytest.param(platform.fp8e4m3fn.dtype, 128, 192, 128, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_mla_prefill(
    device: str,
    solution: str,
    is_causal: bool,
    dtype: torch.dtype,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    require,
) -> None:
    require("attention", "mla_prefill", solution, dtype, "q")

    inputs = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=0,
            total_new_q_tokens=1898,
            num_q_heads=num_heads,
            qk_nope_head_dim=qk_head_dim - 64,
            qk_rope_head_dim=64,
            kv_lora_rank=128,
            v_head_dim=v_head_dim,
            q_dtype=dtype,
            cache_layout="none",
            metadata_kwargs={
                "new_q_length_mode": "regular",
                "tie_new_kv_to_query": True,
            },
        )
    ).generate(metadata_seed=301, value_seed=401, device=device)
    ref = mla_reference(inputs, is_causal=is_causal)

    out, lse = mla_prefill(
        **mla_prefill_kwargs(inputs, is_causal=is_causal, return_lse=True),
        solution=solution,
    )

    assert inputs.q is not None
    assert inputs.v is not None
    assert out.shape == (inputs.q.shape[0], inputs.q.shape[1], inputs.v.shape[-1])
    assert lse.shape == (inputs.q.shape[0], inputs.q.shape[1])
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), ref.out.float(), rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, ref.lse, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,num_heads,kv_lora_rank,qk_rope_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 512, 64, id="bf16"),
        pytest.param(platform.fp8e4m3fn.dtype, 128, 512, 64, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
def test_mla_decode_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    require,
) -> None:
    require("attention", "mla_decode_with_kvcache", solution, dtype, "q")

    batch_size = 2
    q_len = 1
    page_size = 4
    max_seqlen_k = 7
    qk_nope_head_dim = 128

    inputs = MLAInputs(
        _mla_config(
            batch_size=batch_size,
            total_cached_tokens=10,
            total_new_q_tokens=batch_size * q_len,
            num_q_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=128,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=page_size,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": max_seqlen_k,
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=302, value_seed=402, device=device)
    ref = mla_reference(inputs)

    out, lse = mla_decode_with_kvcache(
        **mla_decode_with_kvcache_kwargs(inputs, return_lse=True),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == (batch_size, q_len, num_heads, kv_lora_rank)
    assert lse.shape == (batch_size, q_len, num_heads)
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), ref.out.float(), rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, ref.lse, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,head_dim,num_heads",
    [(torch.bfloat16, 64, 8)],
)
@pytest.mark.parametrize(
    "solution",
    ["triton", "cuda"],
)
def test_attn_merge_state(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    require,
) -> None:
    require("attention", "attn_merge_state", solution, dtype, "out_a")

    values = _AttentionMergeStateGenerator(
        AttentionMergeStateInputConfig(
            total_q=31,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
    ).generate(seed=503, device=device)
    out_ref, lse_ref = attention_merge_state_reference(values)

    out, lse = attn_merge_state(
        values.out_a,
        values.lse_a,
        values.out_b,
        values.lse_b,
        lse_scale_log2=values.lse_scale_log2,
        solution=solution,
    )

    assert out.shape == values.out_a.shape
    assert lse.shape == values.lse_a.shape
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)
