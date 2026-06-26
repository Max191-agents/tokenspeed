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

"""Adapters from generated MHA values to TokenSpeed attention kernel kwargs."""

from __future__ import annotations

from typing import Any

from tokenspeed_kernel.numerics.input_generators.attention import MHAInputValues

__all__ = [
    "mha_decode_with_kvcache_kwargs",
    "mha_extend_with_kvcache_kwargs",
    "mha_prefill_kwargs",
]


def mha_prefill_kwargs(
    inputs: MHAInputValues,
    *,
    window_left: int = -1,
    logit_cap: float = 0.0,
) -> dict[str, Any]:
    """Return keyword args for ``tokenspeed_kernel.mha_prefill``."""

    if inputs.cache is not None:
        raise ValueError("mha_prefill kwargs require non-cached values")
    if inputs.metadata.new_q_lens_cpu != inputs.metadata.new_kv_lens_cpu:
        raise ValueError("mha_prefill requires matching Q and KV lengths")
    if (
        inputs.q is None
        or inputs.k is None
        or inputs.v is None
        or inputs.metadata.cu_seqlens_q is None
        or inputs.metadata.cu_seqlens_q_cpu is None
    ):
        raise ValueError("generated values are incomplete")
    return {
        "q": inputs.q,
        "k": inputs.k,
        "v": inputs.v,
        "cu_seqlens_q": inputs.metadata.cu_seqlens_q,
        "cu_seqlens_q_cpu": inputs.metadata.cu_seqlens_q_cpu,
        "max_seqlen": inputs.metadata.max_seqlen_q,
        "window_left": window_left,
        "logit_cap": logit_cap,
        "sinks": inputs.sinks,
    }


def mha_extend_with_kvcache_kwargs(
    inputs: MHAInputValues,
    *,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
) -> dict[str, Any]:
    """Return keyword args for ``tokenspeed_kernel.mha_extend_with_kvcache``."""

    cache = inputs.cache
    if (
        inputs.q is None
        or cache is None
        or cache.k_cache is None
        or cache.v_cache is None
        or cache.page_table is None
        or inputs.metadata.cache_seqlens is None
        or inputs.metadata.cu_seqlens_q is None
        or inputs.metadata.cu_seqlens_kv is None
    ):
        raise ValueError("paged extend generated values are incomplete")
    return {
        "q": inputs.q,
        "cu_seqlens_q": inputs.metadata.cu_seqlens_q,
        "cu_seqlens_kv": inputs.metadata.cu_seqlens_kv,
        "k_cache": cache.k_cache,
        "v_cache": cache.v_cache,
        "page_table": cache.page_table,
        "cache_seqlens": inputs.metadata.cache_seqlens,
        "max_seqlen_q": inputs.metadata.max_seqlen_q,
        "max_seqlen_k": inputs.metadata.resolved_max_seqlen_k,
        "is_causal": is_causal,
        "window_left": window_left,
        "logit_cap": logit_cap,
        "sinks": inputs.sinks,
    }


def mha_decode_with_kvcache_kwargs(
    inputs: MHAInputValues,
    *,
    window_left: int = -1,
    logit_cap: float = 0.0,
) -> dict[str, Any]:
    """Return keyword args for ``tokenspeed_kernel.mha_decode_with_kvcache``."""

    if len(set(inputs.metadata.new_q_lens_cpu)) != 1:
        raise ValueError(
            "mha_decode_with_kvcache requires a fixed query length per request"
        )
    cache = inputs.cache
    if (
        inputs.q is None
        or cache is None
        or cache.k_cache is None
        or cache.v_cache is None
        or cache.page_table is None
        or inputs.metadata.cache_seqlens is None
    ):
        raise ValueError("paged decode generated values are incomplete")
    return {
        "q": inputs.q,
        "k_cache": cache.k_cache,
        "v_cache": cache.v_cache,
        "page_table": cache.page_table,
        "cache_seqlens": inputs.metadata.cache_seqlens,
        "max_seqlen_k": inputs.metadata.resolved_max_seqlen_k,
        "max_seqlen_q": inputs.metadata.max_seqlen_q,
        "window_left": window_left,
        "logit_cap": logit_cap,
        "sinks": inputs.sinks,
    }
