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

"""Reference attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_numerics_input_generators import (
    AttentionMergeStateInputValues,
    KVCacheValues,
    MHAInputValues,
    MHAReferenceValues,
    MHARequestMetadataValues,
    attention_merge_state_reference,
    mha_reference,
)


def _mha_result(
    values: MHAReferenceValues,
    *,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if return_lse:
        return values.out, values.lse
    return values.out


def _metadata_from_prefill_kwargs(
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_cpu: list[int],
    max_seqlen: int,
) -> MHARequestMetadataValues:
    lengths = [
        int(end) - int(start)
        for start, end in zip(cu_seqlens_q_cpu[:-1], cu_seqlens_q_cpu[1:], strict=True)
    ]
    return MHARequestMetadataValues(
        cached_lens_cpu=[0] * len(lengths),
        new_q_lens_cpu=lengths,
        new_kv_lens_cpu=lengths,
        visible_kv_lens_cpu=lengths,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_cpu=list(cu_seqlens_q_cpu),
        cu_seqlens_kv=cu_seqlens_q,
        cu_seqlens_kv_cpu=list(cu_seqlens_q_cpu),
        cache_seqlens=torch.tensor(
            lengths,
            dtype=torch.int32,
            device=cu_seqlens_q.device,
        ),
        max_seqlen_q=max_seqlen,
        resolved_max_seqlen_k=max_seqlen,
    )


def _metadata_from_cached_kwargs(
    *,
    q: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: torch.Tensor | None = None,
) -> MHARequestMetadataValues:
    batch_size = int(cache_seqlens.shape[0])
    if cu_seqlens_q is None:
        cu_seqlens_q_cpu = [idx * max_seqlen_q for idx in range(batch_size + 1)]
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q_cpu,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        cu_seqlens_q_cpu = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    new_q_lens = [
        int(end) - int(start)
        for start, end in zip(cu_seqlens_q_cpu[:-1], cu_seqlens_q_cpu[1:], strict=True)
    ]
    visible_lens = [int(v) for v in cache_seqlens.detach().cpu().tolist()]
    cu_seqlens_kv_cpu = [0]
    for length in visible_lens:
        cu_seqlens_kv_cpu.append(cu_seqlens_kv_cpu[-1] + length)
    return MHARequestMetadataValues(
        cached_lens_cpu=[0] * batch_size,
        new_q_lens_cpu=new_q_lens,
        new_kv_lens_cpu=[0] * batch_size,
        visible_kv_lens_cpu=visible_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_cpu=cu_seqlens_q_cpu,
        cu_seqlens_kv=torch.tensor(
            cu_seqlens_kv_cpu,
            dtype=torch.int32,
            device=q.device,
        ),
        cu_seqlens_kv_cpu=cu_seqlens_kv_cpu,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        resolved_max_seqlen_k=max_seqlen_k,
    )


def _mha_cache_values(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> KVCacheValues:
    return KVCacheValues(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        page_table_cpu=[
            [int(page) for page in row] for row in page_table.detach().cpu().tolist()
        ],
        cache_shape=tuple(k_cache.shape),
        max_pages_per_request=int(page_table.shape[1]),
        num_pages=int(k_cache.shape[0]),
    )


@register_kernel(
    "attention",
    "attn_merge_state",
    name="torch_attn_merge_state",
    solution="reference",
    signatures=format_signatures(
        ("out_a", "out_b"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference merge of two partial attention output/LSE states."""

    return attention_merge_state_reference(
        AttentionMergeStateInputValues(
            out_a=out_a,
            lse_a=lse_a,
            out_b=out_b,
            lse_b=lse_b,
            lse_scale_log2=lse_scale_log2,
        )
    )


@register_kernel(
    "attention",
    "mha_prefill",
    name="torch_mha_prefill",
    solution="reference",
    signatures=format_signatures(
        ("q", "k", "v"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference causal MHA prefill over explicit Q/K/V tensors."""

    values = MHAInputValues(
        metadata=_metadata_from_prefill_kwargs(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            max_seqlen=max_seqlen,
        ),
        q=q,
        k=k,
        v=v,
        sinks=sinks,
        cache=None,
    )
    return _mha_result(
        mha_reference(values, window_left=window_left, logit_cap=logit_cap),
        return_lse=return_lse,
    )


@register_kernel(
    "attention",
    "mha_extend_with_kvcache",
    name="torch_mha_extend_with_kvcache",
    solution="reference",
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference MHA extend over visible paged KV-cache rows."""

    del cu_seqlens_kv
    values = MHAInputValues(
        metadata=_metadata_from_cached_kwargs(
            q=q,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
        ),
        q=q,
        k=None,
        v=None,
        sinks=sinks,
        cache=_mha_cache_values(
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
        ),
    )
    return _mha_result(
        mha_reference(
            values,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
        ),
        return_lse=return_lse,
    )


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="torch_mha_decode_with_kvcache",
    solution="reference",
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"),
        "dense",
        {torch.float16, torch.bfloat16},
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference MHA decode over visible paged KV-cache rows."""

    values = MHAInputValues(
        metadata=_metadata_from_cached_kwargs(
            q=q,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        ),
        q=q,
        k=None,
        v=None,
        sinks=sinks,
        cache=_mha_cache_values(
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
        ),
    )
    return _mha_result(
        mha_reference(
            values,
            is_causal=max_seqlen_q > 1,
            window_left=window_left,
            logit_cap=logit_cap,
        ),
        return_lse=return_lse,
    )
