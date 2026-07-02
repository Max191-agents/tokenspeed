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

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    MHAInputConfig,
    MHAInputs,
    MHAInputValues,
    MHARequestMetadataInputConfig,
)

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not platform.is_hopper,
    reason="FA3 smoke tests require Hopper GPU.",
)


def _mha_values(
    *,
    batch_size: int,
    total_cached_tokens: int,
    total_new_q_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    cache_layout: str = "none",
    page_size: int | None = None,
    indexing: str | None = None,
    metadata_kwargs: dict[str, object] | None = None,
    metadata_seed: int = 42,
    value_seed: int = 43,
    device: str,
) -> MHAInputValues:
    return MHAInputs(
        MHAInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=dtype,
            cache_layout=cache_layout,  # type: ignore[arg-type]
            page_size=page_size,
            indexing=indexing,  # type: ignore[arg-type]
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                cache_layout=cache_layout,  # type: ignore[arg-type]
                **(metadata_kwargs or {}),
            ),
        )
    ).generate(metadata_seed=metadata_seed, value_seed=value_seed, device=device)


def _fixed_batch_q(values: MHAInputValues) -> torch.Tensor:
    assert values.q is not None
    q_lens = values.metadata.new_q_lens_cpu
    assert len(set(q_lens)) == 1
    return values.q.view(len(q_lens), q_lens[0], values.q.shape[-2], values.q.shape[-1])


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 2
    seqlen = 128

    values = _mha_values(
        batch_size=batch_size,
        total_cached_tokens=0,
        total_new_q_tokens=batch_size * seqlen,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        metadata_kwargs={"new_q_length_mode": "fixed_per_request"},
        device=device,
    )
    q, k, v = values.dense_qkv()

    out = flash_attn_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_ragged(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    values = _mha_values(
        batch_size=3,
        total_cached_tokens=0,
        total_new_q_tokens=38,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        metadata_kwargs={
            "new_q_length_mode": "ragged",
            "max_new_q_tokens_per_request": 17,
        },
        device=device,
    )
    assert values.q is not None
    assert values.k is not None
    assert values.v is not None

    out = flash_attn_varlen_func(
        q=values.q,
        k=values.k,
        v=values.v,
        cu_seqlens_q=values.metadata.cu_seqlens_q,
        cu_seqlens_k=values.metadata.cu_seqlens_q,
        max_seqlen_q=values.metadata.max_seqlen_q,
        max_seqlen_k=values.metadata.max_seqlen_q,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == values.q.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    decode_tokens = 1
    max_cache_seqlen = 256
    values = _mha_values(
        batch_size=batch_size,
        total_cached_tokens=208,
        total_new_q_tokens=batch_size * decode_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        cache_layout="dense",
        metadata_kwargs={
            "cached_length_mode": "ragged",
            "new_q_length_mode": "fixed_per_request",
            "max_seqlen_k": max_cache_seqlen,
        },
        device=device,
    )
    assert values.cache is not None
    assert values.cache.k_cache is not None
    assert values.cache.v_cache is not None
    q = _fixed_batch_q(values)

    out = flash_attn_with_kvcache(
        q=q,
        k_cache=values.cache.k_cache,
        v_cache=values.cache.v_cache,
        cache_seqlens=values.metadata.cache_seqlens,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_ragged_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    max_cache_seqlen = 256
    values = _mha_values(
        batch_size=batch_size,
        total_cached_tokens=208,
        total_new_q_tokens=10,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        cache_layout="dense",
        metadata_kwargs={
            "cached_length_mode": "ragged",
            "new_q_length_mode": "ragged",
            "max_new_q_tokens_per_request": 4,
            "max_seqlen_k": max_cache_seqlen,
        },
        device=device,
    )
    assert values.q is not None
    assert values.cache is not None
    assert values.cache.k_cache is not None
    assert values.cache.v_cache is not None

    out = flash_attn_with_kvcache(
        q=values.q,
        k_cache=values.cache.k_cache,
        v_cache=values.cache.v_cache,
        cache_seqlens=values.metadata.cache_seqlens,
        cu_seqlens_q=values.metadata.cu_seqlens_q,
        cu_seqlens_k_new=values.metadata.cu_seqlens_kv,
        max_seqlen_q=values.metadata.max_seqlen_q,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == values.q.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_ragged_with_paged_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    decode_tokens = 1
    page_size = 128
    max_cache_seqlen = 256
    values = _mha_values(
        batch_size=batch_size,
        total_cached_tokens=400,
        total_new_q_tokens=batch_size * decode_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        cache_layout="paged",
        page_size=page_size,
        indexing="identity",
        metadata_kwargs={
            "cached_length_mode": "ragged",
            "max_cached_tokens_per_request": max_cache_seqlen - decode_tokens,
            "new_q_length_mode": "fixed_per_request",
            "max_seqlen_k": max_cache_seqlen,
        },
        device=device,
    )
    assert values.cache is not None
    assert values.cache.k_cache is not None
    assert values.cache.v_cache is not None
    assert values.cache.page_table is not None
    q = _fixed_batch_q(values)

    out = flash_attn_with_kvcache(
        q=q,
        k_cache=values.cache.k_cache,
        v_cache=values.cache.v_cache,
        page_table=values.cache.page_table,
        cache_seqlens=values.metadata.cache_seqlens,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape
