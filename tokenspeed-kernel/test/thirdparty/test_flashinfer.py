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
from tokenspeed_kernel.ops.attention.flashinfer import (
    trtllm_batch_context_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
    trtllm_batch_decode_with_kv_cache_mla,
    trtllm_ragged_attention_deepseek,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    CacheLayout,
    KVCacheValues,
    MHAInputConfig,
    MHAInputs,
    MHAInputValues,
    MHARequestMetadataInputConfig,
    MLAInputConfig,
    MLAInputs,
    MLAInputValues,
    MLAKVCacheValues,
    PageTableIndexing,
)

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_blackwell),
    reason="FlashInfer TRTLLM tests require Blackwell GPU.",
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
    cache_layout: CacheLayout = "none",
    page_size: int | None = None,
    indexing: PageTableIndexing | None = None,
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
            cache_layout=cache_layout,
            page_size=page_size,
            indexing=indexing,
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                cache_layout=cache_layout,
                **(metadata_kwargs or {}),
            ),
        )
    ).generate(metadata_seed=metadata_seed, value_seed=value_seed, device=device)


def _mla_values(
    *,
    batch_size: int,
    total_cached_tokens: int,
    total_new_q_tokens: int,
    num_q_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    dtype: torch.dtype,
    page_size: int,
    metadata_kwargs: dict[str, object],
    device: str,
) -> MLAInputValues:
    return MLAInputs(
        MLAInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            num_q_heads=num_q_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=v_head_dim,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=page_size,
            indexing="identity",
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                cache_layout="paged",
                **metadata_kwargs,
            ),
        )
    ).generate(metadata_seed=42, value_seed=43, device=device)


def _flashinfer_mha_cache(cache: KVCacheValues) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.k_cache is None or cache.v_cache is None:
        raise ValueError("MHA cache values must include K and V cache tensors")
    return (
        cache.k_cache.transpose(1, 2).contiguous(),
        cache.v_cache.transpose(1, 2).contiguous(),
    )


def _flashinfer_mla_cache(cache: MLAKVCacheValues) -> torch.Tensor:
    return cache.kv_cache.transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        (torch.bfloat16, 128, 8, 8),
        (torch.bfloat16, 128, 16, 2),
    ],
)
def test_mha_prefill(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 3
    workspace_buffer = torch.empty(150 * 1024 * 1024, device=device, dtype=torch.uint8)
    values = _mha_values(
        batch_size=batch_size,
        total_cached_tokens=0,
        total_new_q_tokens=1880,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        metadata_kwargs={
            "new_q_length_mode": "ragged",
            "max_new_q_tokens_per_request": 834,
        },
        device=device,
    )
    assert values.q is not None
    assert values.k is not None
    assert values.v is not None

    out = trtllm_ragged_attention_deepseek(
        query=values.q,
        key=values.k,
        value=values.v,
        workspace_buffer=workspace_buffer,
        seq_lens=values.metadata.cache_seqlens,
        max_q_len=values.metadata.max_seqlen_q,
        max_kv_len=values.metadata.max_seqlen_q,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        o_sf_scale=-1.0,
        batch_size=batch_size,
        window_left=-1,
        cum_seq_lens_q=values.metadata.cu_seqlens_q,
        cum_seq_lens_kv=values.metadata.cu_seqlens_kv,
        enable_pdl=False,
        is_causal=True,
        return_lse=False,
    )

    assert out.shape == values.q.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        (torch.bfloat16, 128, 8, 8),
        (torch.bfloat16, 128, 16, 2),
    ],
)
def test_mha_prefill_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 3
    page_size = 64
    max_kv_len = 1024
    workspace_buffer = torch.empty(512 * 1024 * 1024, device=device, dtype=torch.uint8)
    values = _mha_values(
        batch_size=batch_size,
        total_cached_tokens=0,
        total_new_q_tokens=1880,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        cache_layout="paged",
        page_size=page_size,
        indexing="identity",
        metadata_kwargs={
            "new_q_length_mode": "ragged",
            "max_new_q_tokens_per_request": 834,
            "max_seqlen_k": max_kv_len,
        },
        device=device,
    )
    assert values.q is not None
    assert values.cache is not None
    assert values.cache.page_table is not None

    out = trtllm_batch_context_with_kv_cache(
        query=values.q,
        kv_cache=_flashinfer_mha_cache(values.cache),
        workspace_buffer=workspace_buffer,
        block_tables=values.cache.page_table,
        seq_lens=values.metadata.cache_seqlens,
        max_q_len=values.metadata.max_seqlen_q,
        max_kv_len=max_kv_len,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        batch_size=batch_size,
        cum_seq_lens_q=values.metadata.cu_seqlens_q,
        cum_seq_lens_kv=values.metadata.cu_seqlens_kv,
        out_dtype=dtype,
    )

    assert out.shape == values.q.shape


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 8), (torch.bfloat16, 128, 16, 2)],
)
def test_mha_decode_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    page_size = 64
    max_seq_len = 1024
    workspace_buffer = torch.empty(512 * 1024 * 1024, device=device, dtype=torch.uint8)
    values = _mha_values(
        batch_size=batch_size,
        total_cached_tokens=2789,
        total_new_q_tokens=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        cache_layout="paged",
        page_size=page_size,
        indexing="identity",
        metadata_kwargs={
            "cached_length_mode": "ragged",
            "max_cached_tokens_per_request": max_seq_len - 1,
            "new_q_length_mode": "fixed_per_request",
            "max_seqlen_k": max_seq_len,
        },
        device=device,
    )
    assert values.q is not None
    assert values.cache is not None
    assert values.cache.page_table is not None

    out = trtllm_batch_decode_with_kv_cache(
        query=values.q,
        kv_cache=_flashinfer_mha_cache(values.cache),
        workspace_buffer=workspace_buffer,
        block_tables=values.cache.page_table,
        seq_lens=values.metadata.cache_seqlens,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / math.sqrt(head_dim),
        bmm2_scale=1.0,
        out_dtype=dtype,
    )

    assert out.shape == values.q.shape


@pytest.mark.parametrize(
    "dtype,num_q_heads,qk_head_dim,kv_lora_rank",
    [(torch.bfloat16, 16, 64, 256)],
)
def test_mla_decode_with_kvcache(
    device: str,
    dtype: torch.dtype,
    num_q_heads: int,
    qk_head_dim: int,
    kv_lora_rank: int,
) -> None:
    batch_size = 4
    q_len_per_req = 1
    page_size = 64
    max_seq_len = 1024
    qk_nope_head_dim = qk_head_dim
    qk_rope_head_dim = qk_head_dim
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim
    query_head_dim = kv_lora_rank + qk_rope_head_dim
    output_head_dim = kv_lora_rank
    workspace_buffer = torch.empty(150 * 1024 * 1024, device=device, dtype=torch.uint8)
    values = _mla_values(
        batch_size=batch_size,
        total_cached_tokens=2789,
        total_new_q_tokens=batch_size * q_len_per_req,
        num_q_heads=num_q_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=output_head_dim,
        dtype=dtype,
        page_size=page_size,
        metadata_kwargs={
            "cached_length_mode": "ragged",
            "max_cached_tokens_per_request": max_seq_len - q_len_per_req,
            "new_q_length_mode": "fixed_per_request",
            "max_seqlen_k": max_seq_len,
        },
        device=device,
    )
    assert values.q is not None
    assert values.cache is not None
    assert values.cache.page_table is not None
    assert values.q.shape[-1] == query_head_dim
    assert values.cache.kv_cache.shape[-1] == kv_cache_dim

    out = trtllm_batch_decode_with_kv_cache_mla(
        query=values.q,
        kv_cache=_flashinfer_mla_cache(values.cache),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=values.cache.page_table,
        seq_lens=values.metadata.cache_seqlens,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / math.sqrt(query_head_dim),
    )

    assert out.shape == (batch_size, q_len_per_req, num_q_heads, output_head_dim)
