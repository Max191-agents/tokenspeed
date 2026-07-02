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
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    MHARequestMetadataInputConfig,
    MLAInputConfig,
    MLAInputs,
)

platform = current_platform()


@pytest.mark.skipif(not platform.is_hopper, reason="Requires Hopper GPU")
@pytest.mark.parametrize(
    "dtype,num_q_heads,head_dim_v,qk_rope_head_dim",
    [(torch.bfloat16, 16, 512, 64)],
)
def test_mla_decode_with_paged_kvcache(
    device: str,
    dtype: torch.dtype,
    num_q_heads: int,
    head_dim_v: int,
    qk_rope_head_dim: int,
) -> None:
    batch_size = 4
    q_len_per_req = 1
    page_size = 64
    max_seq_len = 1024
    kv_cache_dim = head_dim_v + qk_rope_head_dim
    values = MLAInputs(
        MLAInputConfig(
            batch_size=batch_size,
            total_cached_tokens=2789,
            total_new_q_tokens=batch_size * q_len_per_req,
            num_q_heads=num_q_heads,
            qk_nope_head_dim=head_dim_v,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=head_dim_v,
            v_head_dim=head_dim_v,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=page_size,
            indexing="identity",
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=batch_size,
                total_cached_tokens=2789,
                total_new_q_tokens=batch_size * q_len_per_req,
                cached_length_mode="ragged",
                max_cached_tokens_per_request=max_seq_len - q_len_per_req,
                new_q_length_mode="fixed_per_request",
                cache_layout="paged",
                max_seqlen_k=max_seq_len,
            ),
        )
    ).generate(metadata_seed=42, value_seed=43, device=device)
    assert values.q is not None
    assert values.cache is not None
    assert values.cache.page_table is not None
    assert values.q.shape == (batch_size, q_len_per_req, num_q_heads, kv_cache_dim)
    assert values.cache.kv_cache.shape[1:] == (page_size, 1, kv_cache_dim)
    assert values.cache.page_table.shape == (batch_size, max_seq_len // page_size)

    tile_scheduler_metadata, _ = get_mla_metadata()

    out, lse = flash_mla_with_kvcache(
        q=values.q,
        k_cache=values.cache.kv_cache,
        block_table=values.cache.page_table,
        cache_seqlens=values.metadata.cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        softmax_scale=1.0 / math.sqrt(kv_cache_dim),
        causal=True,
    )

    assert out.shape == (batch_size, q_len_per_req, num_q_heads, head_dim_v)
    assert lse.shape == (batch_size, num_q_heads, q_len_per_req)
