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
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
    mla_decode_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.numerics.attention_kernel_kwargs import (
    mla_decode_with_kvcache_kwargs,
    mla_prefill_kwargs,
    mha_decode_with_kvcache_kwargs,
    mha_extend_with_kvcache_kwargs,
    mha_prefill_kwargs,
)
from tokenspeed_numerics_input_generators import (
    MHAInputConfig,
    MHAInputs,
    MLAInputConfig,
    MLAInputs,
    MHARequestMetadataInputConfig,
)


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
    num_kv_heads: int | None = None,
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
        num_kv_heads=num_kv_heads,
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


@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_mha_prefill_generator_runs_attention_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("attention", "mha_prefill", solution, dtype, "q")

    inputs = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=0,
            total_new_q_tokens=33,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            q_dtype=dtype,
            cache_layout="none",
            include_sinks=True,
            metadata_kwargs={"new_q_length_mode": "ragged"},
        )
    ).generate(metadata_seed=101, value_seed=201, device=device)

    out = mha_prefill(**mha_prefill_kwargs(inputs), solution=solution)

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()


def test_mha_paged_extend_generator_runs_triton_attention_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    solution = "triton"
    require("attention", "mha_extend_with_kvcache", solution, dtype, "q")

    inputs = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=47,
            total_new_q_tokens=11,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=64,
            include_sinks=True,
            metadata_kwargs={
                "cached_length_mode": "ragged",
                "new_q_length_mode": "ragged",
            },
        )
    ).generate(metadata_seed=102, value_seed=202, device=device)

    out = mha_extend_with_kvcache(
        **mha_extend_with_kvcache_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()


def test_mla_prefill_generator_runs_attention_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    solution = "triton"
    require("attention", "mla_prefill", solution, dtype, "q")

    inputs = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=0,
            total_new_q_tokens=33,
            num_q_heads=8,
            num_kv_heads=8,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=128,
            v_head_dim=128,
            q_dtype=dtype,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "ragged"},
        )
    ).generate(metadata_seed=104, value_seed=204, device=device)

    out = mla_prefill(
        **mla_prefill_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert inputs.v is not None
    assert out.shape == (inputs.q.shape[0], inputs.q.shape[1], inputs.v.shape[-1])
    assert not torch.isnan(out.float()).any()


def test_mla_paged_decode_generator_runs_attention_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    solution = "triton"
    require("attention", "mla_decode_with_kvcache", solution, dtype, "q")

    inputs = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=12,
            total_new_q_tokens=2,
            num_q_heads=8,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=128,
            v_head_dim=128,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=4,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": 7,
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=105, value_seed=205, device=device)

    out = mla_decode_with_kvcache(
        **mla_decode_with_kvcache_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == (inputs.q.shape[0], inputs.q.shape[1], inputs.q.shape[2], 128)
    assert not torch.isnan(out.float()).any()


@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_mha_paged_decode_generator_runs_attention_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("attention", "mha_decode_with_kvcache", solution, dtype, "q")

    inputs = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=51,
            total_new_q_tokens=4,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=64,
            include_sinks=True,
            metadata_kwargs={
                "cached_length_mode": "ragged",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=103, value_seed=203, device=device)

    out = mha_decode_with_kvcache(
        **mha_decode_with_kvcache_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()
