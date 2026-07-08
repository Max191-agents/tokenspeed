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

"""Numerics framework hooks for attention-family ops."""

from __future__ import annotations

import math
from typing import Any

import torch
from tokenspeed_kernel.numerics.attention_kernel_kwargs import (
    mha_decode_with_kvcache_kwargs,
    mha_extend_with_kvcache_kwargs,
    mha_prefill_kwargs,
    mla_decode_with_kvcache_kwargs,
    mla_prefill_kwargs,
)
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance
from tokenspeed_numerics_input_generators import (
    MHAInputConfig,
    MHAInputs,
    MHARequestMetadataInputConfig,
    MLAInputConfig,
    MLAInputs,
)


def tolerance(dtype: torch.dtype, *, mode: str | None = None, **_: Any) -> Tolerance:
    if mode == "attn_merge_state":
        return Tolerance(atol=5.0e-2, rtol=5.0e-2)
    return Tolerance(atol=1.0e-1, rtol=5.0e-2)


set_family_tolerance("attention", tolerance)


class MHAInputGenerator(InputGenerator):
    """Adapter from operation-level ``MHAInputs`` to TokenSpeed MHA kwargs."""

    def _values(
        self,
        *,
        batch_size: int,
        total_cached_tokens: int,
        total_new_q_tokens: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cache_layout: str,
        page_size: int | None,
        include_sinks: bool,
        cached_length_mode: str = "regular",
        new_q_length_mode: str = "fixed_per_request",
        max_cached_tokens_per_request: int | None = None,
        max_new_q_tokens_per_request: int | None = None,
    ):
        metadata_input = MHARequestMetadataInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            cached_length_mode=cached_length_mode,  # type: ignore[arg-type]
            new_q_length_mode=new_q_length_mode,  # type: ignore[arg-type]
            max_cached_tokens_per_request=max_cached_tokens_per_request,
            max_new_q_tokens_per_request=max_new_q_tokens_per_request,
            cache_layout=cache_layout,  # type: ignore[arg-type]
        )
        return MHAInputs(
            MHAInputConfig(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_dtype=self.dtype,
                cache_layout=cache_layout,  # type: ignore[arg-type]
                page_size=page_size,
                include_sinks=include_sinks,
                metadata_input=metadata_input,
            )
        ).generate(
            metadata_seed=self.seed,
            value_seed=self.seed + 1,
            device=self.device,
        )

    def generate(
        self,
        *,
        batch_size: int,
        total_cached_tokens: int = 0,
        total_new_q_tokens: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int | None = None,
        include_sinks: bool = False,
        window_left: int = -1,
        logit_cap: float = 0.0,
        return_lse: bool = False,
        is_causal: bool = True,
        cached_length_mode: str = "regular",
        new_q_length_mode: str = "fixed_per_request",
        max_cached_tokens_per_request: int | None = None,
        max_new_q_tokens_per_request: int | None = None,
    ) -> dict[str, Any]:
        if self.op_mode == "mha_prefill":
            values = self._values(
                batch_size=batch_size,
                total_cached_tokens=0,
                total_new_q_tokens=total_new_q_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                cache_layout="none",
                page_size=None,
                include_sinks=include_sinks,
                cached_length_mode="regular",
                new_q_length_mode=new_q_length_mode,
                max_new_q_tokens_per_request=max_new_q_tokens_per_request,
            )
            kwargs = mha_prefill_kwargs(
                values,
                window_left=window_left,
                logit_cap=logit_cap,
            )
        elif self.op_mode == "mha_extend_with_kvcache":
            values = self._values(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                cache_layout="paged",
                page_size=page_size,
                include_sinks=include_sinks,
                cached_length_mode=cached_length_mode,
                new_q_length_mode=new_q_length_mode,
                max_cached_tokens_per_request=max_cached_tokens_per_request,
                max_new_q_tokens_per_request=max_new_q_tokens_per_request,
            )
            kwargs = mha_extend_with_kvcache_kwargs(
                values,
                is_causal=is_causal,
                window_left=window_left,
                logit_cap=logit_cap,
            )
        elif self.op_mode == "mha_decode_with_kvcache":
            values = self._values(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                cache_layout="paged",
                page_size=page_size,
                include_sinks=include_sinks,
                cached_length_mode=cached_length_mode,
                new_q_length_mode=new_q_length_mode,
                max_cached_tokens_per_request=max_cached_tokens_per_request,
                max_new_q_tokens_per_request=max_new_q_tokens_per_request,
            )
            kwargs = mha_decode_with_kvcache_kwargs(
                values,
                window_left=window_left,
                logit_cap=logit_cap,
            )
        else:
            raise ValueError(f"unsupported MHA input mode {self.op_mode!r}")
        kwargs["return_lse"] = return_lse
        return kwargs


set_input_generator("attention", "mha_prefill", MHAInputGenerator)
set_input_generator("attention", "mha_extend_with_kvcache", MHAInputGenerator)
set_input_generator("attention", "mha_decode_with_kvcache", MHAInputGenerator)

_MHA_PREFILL_STANDARD_SHAPES: list[dict[str, int | bool | float]] = [
    {
        "batch_size": 2,
        "total_new_q_tokens": 8,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
        "return_lse": False,
    },
    {
        "batch_size": 1,
        "total_new_q_tokens": 4,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 64,
        "return_lse": False,
    },
]

_MHA_EXTEND_STANDARD_SHAPES: list[dict[str, int | bool | float]] = [
    {
        "batch_size": 2,
        "total_cached_tokens": 8,
        "total_new_q_tokens": 4,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
        "page_size": 64,
        "is_causal": True,
        "return_lse": False,
    },
    {
        "batch_size": 1,
        "total_cached_tokens": 4,
        "total_new_q_tokens": 2,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 64,
        "page_size": 64,
        "is_causal": True,
        "return_lse": False,
    },
]

_MHA_DECODE_STANDARD_SHAPES: list[dict[str, int | bool | float]] = [
    {
        "batch_size": 2,
        "total_cached_tokens": 8,
        "total_new_q_tokens": 2,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
        "page_size": 64,
        "return_lse": False,
    },
    {
        "batch_size": 1,
        "total_cached_tokens": 4,
        "total_new_q_tokens": 1,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 64,
        "page_size": 64,
        "return_lse": False,
    },
]

for _mha_mode, _mha_shapes in (
    ("mha_prefill", _MHA_PREFILL_STANDARD_SHAPES),
    ("mha_extend_with_kvcache", _MHA_EXTEND_STANDARD_SHAPES),
    ("mha_decode_with_kvcache", _MHA_DECODE_STANDARD_SHAPES),
):
    set_standard_shapes("attention", _mha_mode, _mha_shapes)
    set_benchmark_shapes("attention", _mha_mode, _mha_shapes)


class MLAInputGenerator(InputGenerator):
    """Adapter from operation-level ``MLAInputs`` to TokenSpeed MLA kwargs."""

    def _values(
        self,
        *,
        batch_size: int,
        total_cached_tokens: int,
        total_new_q_tokens: int,
        num_q_heads: int,
        num_kv_heads: int | None,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        cache_layout: str,
        page_size: int | None,
        cached_length_mode: str = "regular",
        new_q_length_mode: str = "fixed_per_request",
        max_seqlen_k: int | None = None,
    ):
        metadata_input = MHARequestMetadataInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            cached_length_mode=cached_length_mode,  # type: ignore[arg-type]
            new_q_length_mode=new_q_length_mode,  # type: ignore[arg-type]
            cache_layout=cache_layout,  # type: ignore[arg-type]
            max_seqlen_k=max_seqlen_k,
            allow_untied_non_cached_kv=True,
        )
        return MLAInputs(
            MLAInputConfig(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
                v_head_dim=v_head_dim,
                q_dtype=self.dtype,
                cache_layout=cache_layout,  # type: ignore[arg-type]
                page_size=page_size,
                metadata_input=metadata_input,
            )
        ).generate(
            metadata_seed=self.seed,
            value_seed=self.seed + 1,
            device=self.device,
        )

    def generate(
        self,
        *,
        batch_size: int,
        total_cached_tokens: int = 0,
        total_new_q_tokens: int,
        num_q_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        num_kv_heads: int | None = None,
        page_size: int | None = None,
        is_causal: bool = True,
        logit_cap: float = 0.0,
        return_lse: bool = False,
        cached_length_mode: str = "regular",
        new_q_length_mode: str = "fixed_per_request",
        max_seqlen_k: int | None = None,
    ) -> dict[str, Any]:
        if self.op_mode == "mla_prefill":
            values = self._values(
                batch_size=batch_size,
                total_cached_tokens=0,
                total_new_q_tokens=total_new_q_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
                v_head_dim=v_head_dim,
                cache_layout="none",
                page_size=None,
                cached_length_mode="regular",
                new_q_length_mode=new_q_length_mode,
            )
            return mla_prefill_kwargs(
                values,
                is_causal=is_causal,
                logit_cap=logit_cap,
                return_lse=return_lse,
            )
        if self.op_mode == "mla_decode_with_kvcache":
            values = self._values(
                batch_size=batch_size,
                total_cached_tokens=total_cached_tokens,
                total_new_q_tokens=total_new_q_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
                v_head_dim=v_head_dim,
                cache_layout="paged",
                page_size=page_size,
                cached_length_mode=cached_length_mode,
                new_q_length_mode=new_q_length_mode,
                max_seqlen_k=max_seqlen_k,
            )
            return mla_decode_with_kvcache_kwargs(
                values,
                logit_cap=logit_cap,
                return_lse=return_lse,
            )
        raise ValueError(f"unsupported MLA input mode {self.op_mode!r}")


set_input_generator("attention", "mla_prefill", MLAInputGenerator)
set_input_generator("attention", "mla_decode_with_kvcache", MLAInputGenerator)

_MLA_PREFILL_STANDARD_SHAPES: list[dict[str, int | bool | float]] = [
    {
        "batch_size": 2,
        "total_new_q_tokens": 6,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "kv_lora_rank": 16,
        "v_head_dim": 8,
        "is_causal": True,
        "return_lse": False,
    },
    {
        "batch_size": 1,
        "total_new_q_tokens": 3,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "kv_lora_rank": 16,
        "v_head_dim": 8,
        "is_causal": True,
        "return_lse": False,
    },
]

_MLA_DECODE_STANDARD_SHAPES: list[dict[str, int | bool | float]] = [
    {
        "batch_size": 2,
        "total_cached_tokens": 8,
        "total_new_q_tokens": 2,
        "num_q_heads": 4,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "kv_lora_rank": 16,
        "v_head_dim": 8,
        "page_size": 64,
        "return_lse": False,
    },
    {
        "batch_size": 1,
        "total_cached_tokens": 4,
        "total_new_q_tokens": 1,
        "num_q_heads": 2,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "kv_lora_rank": 16,
        "v_head_dim": 8,
        "page_size": 64,
        "return_lse": False,
    },
]

for _mla_mode, _mla_shapes in (
    ("mla_prefill", _MLA_PREFILL_STANDARD_SHAPES),
    ("mla_decode_with_kvcache", _MLA_DECODE_STANDARD_SHAPES),
):
    set_standard_shapes("attention", _mla_mode, _mla_shapes)
    set_benchmark_shapes("attention", _mla_mode, _mla_shapes)


class AttentionMergeStateInputGenerator(InputGenerator):
    """Adapter for attention merge-state input generation.

    Shape kwargs:
        total_q:   number of query rows being merged
        num_heads: number of attention heads
        head_dim:  per-head output dimension
    """

    def generate(
        self,
        *,
        total_q: int,
        num_heads: int,
        head_dim: int,
        lse_scale_log2: float = math.log2(math.e),
        lse_bound: float = 6.0,
    ) -> dict[str, Any]:
        target_device = torch.device("cpu" if self.device is None else self.device)
        rng_device = "cuda" if target_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(self.seed)
        out_shape = (total_q, num_heads, head_dim)
        lse_shape = (total_q, num_heads)
        out_a = torch.randn(
            out_shape,
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        ).to(self.dtype)
        out_b = torch.randn(
            out_shape,
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        ).to(self.dtype)
        lse_a = (
            torch.rand(
                lse_shape,
                dtype=torch.float32,
                device=target_device,
                generator=generator,
            )
            * (2.0 * lse_bound)
            - lse_bound
        )
        lse_b = (
            torch.rand(
                lse_shape,
                dtype=torch.float32,
                device=target_device,
                generator=generator,
            )
            * (2.0 * lse_bound)
            - lse_bound
        )
        return {
            "out_a": out_a,
            "lse_a": lse_a,
            "out_b": out_b,
            "lse_b": lse_b,
            "lse_scale_log2": float(lse_scale_log2),
        }


set_input_generator(
    "attention",
    "attn_merge_state",
    AttentionMergeStateInputGenerator,
)

_ATTN_MERGE_STATE_STANDARD_SHAPES: list[dict[str, int | float]] = [
    {"total_q": 1, "num_heads": 2, "head_dim": 64},
    {"total_q": 8, "num_heads": 4, "head_dim": 128},
]

set_standard_shapes(
    "attention",
    "attn_merge_state",
    _ATTN_MERGE_STATE_STANDARD_SHAPES,
)
set_benchmark_shapes(
    "attention",
    "attn_merge_state",
    _ATTN_MERGE_STATE_STANDARD_SHAPES,
)
