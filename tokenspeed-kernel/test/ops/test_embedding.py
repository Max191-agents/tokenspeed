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
from tokenspeed_kernel.numerics.reference.embedding import rope_reference
from tokenspeed_kernel.ops.embedding import FusedSetKVBufferArg, apply_rope
from tokenspeed_numerics_input_generators import (
    RopeInputConfig,
    RopeInputs,
    RopeInputValues,
)


def _apply_rope_values(
    values: RopeInputValues,
    config: RopeInputConfig,
    *,
    solution: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_arg = None
    if values.fused_kv is not None:
        fused_arg = FusedSetKVBufferArg(
            value=values.fused_kv.value,
            k_buffer=values.fused_kv.k_buffer,
            v_buffer=values.fused_kv.v_buffer,
            k_scale=values.fused_kv.k_scale,
            v_scale=values.fused_kv.v_scale,
            cache_loc=values.fused_kv.cache_loc,
        )
    return apply_rope(
        positions=values.positions,
        query=values.query,
        key=values.key,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
        fused_set_kv_buffer_arg=fused_arg,
        output_q_rope=values.output_q_rope,
        output_k_rope=values.output_k_rope,
        solution=solution,
    )


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_neox_full_bf16(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "query")
    config = RopeInputConfig(
        num_tokens=17,
        num_q_heads=8,
        num_kv_heads=2,
        head_size=128,
        rotary_dim=128,
        max_position=1024,
        dtype=dtype,
        is_neox=True,
    )
    values = RopeInputs(config).generate(seed=1, metadata_seed=2, device=device)
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )
    q_out, k_out = _apply_rope_values(values, config, solution=solution)

    assert q_out.data_ptr() == values.query.data_ptr()
    assert k_out.data_ptr() == values.key.data_ptr()
    torch.testing.assert_close(values.query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(values.key, k_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_gptj_full_bf16(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "query")
    config = RopeInputConfig(
        num_tokens=9,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=64,
        rotary_dim=64,
        max_position=512,
        dtype=dtype,
        is_neox=False,
    )
    values = RopeInputs(config).generate(seed=3, metadata_seed=4, device=device)
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )
    _apply_rope_values(values, config, solution=solution)

    torch.testing.assert_close(values.query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(values.key, k_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_neox_partial_bf16(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "query")
    config = RopeInputConfig(
        num_tokens=5,
        num_q_heads=4,
        num_kv_heads=1,
        head_size=128,
        rotary_dim=64,
        max_position=256,
        dtype=dtype,
        is_neox=True,
    )
    values = RopeInputs(config).generate(seed=5, metadata_seed=6, device=device)
    query_orig = values.query.clone()
    key_orig = values.key.clone()
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )
    _apply_rope_values(values, config, solution=solution)

    torch.testing.assert_close(values.query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(values.key, k_ref, rtol=2e-2, atol=2e-2)

    q_view = values.query.view(config.num_tokens, config.num_q_heads, config.head_size)
    q_orig_view = query_orig.view(
        config.num_tokens,
        config.num_q_heads,
        config.head_size,
    )
    assert torch.equal(
        q_view[..., config.rotary_dim :], q_orig_view[..., config.rotary_dim :]
    )
    k_view = values.key.view(config.num_tokens, config.num_kv_heads, config.head_size)
    k_orig_view = key_orig.view(
        config.num_tokens,
        config.num_kv_heads,
        config.head_size,
    )
    assert torch.equal(
        k_view[..., config.rotary_dim :], k_orig_view[..., config.rotary_dim :]
    )


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_single_token(
    device: str,
    solution: str,
    require,
) -> None:
    """Edge case: num_tokens == 1 (decode step)."""
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "query")
    config = RopeInputConfig(
        num_tokens=1,
        num_q_heads=8,
        num_kv_heads=1,
        head_size=128,
        rotary_dim=128,
        max_position=64,
        dtype=dtype,
        is_neox=True,
    )
    values = RopeInputs(config).generate(seed=7, metadata_seed=8, device=device)
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )
    _apply_rope_values(values, config, solution=solution)

    torch.testing.assert_close(values.query, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(values.key, k_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("solution", ["triton", "cuda"])
def test_rope_fused_set_kv_buffer(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("embedding", "rope", solution, dtype, "query")
    config = RopeInputConfig(
        num_tokens=13,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=128,
        rotary_dim=128,
        max_position=512,
        cache_size=32,
        dtype=dtype,
        is_neox=True,
        with_fused_kv=True,
        with_q_output=True,
    )
    values = RopeInputs(config).generate(seed=9, metadata_seed=10, device=device)
    assert values.fused_kv is not None
    assert values.output_q_rope is not None
    query_orig = values.query.clone()
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )
    q_out, k_out = _apply_rope_values(values, config, solution=solution)

    assert q_out.data_ptr() == values.output_q_rope.data_ptr()
    assert k_out.data_ptr() == values.key.data_ptr()
    torch.testing.assert_close(values.query, query_orig, rtol=0, atol=0)
    torch.testing.assert_close(values.output_q_rope, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(values.key, k_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        values.fused_kv.k_buffer.index_select(0, values.fused_kv.cache_loc),
        k_ref,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        values.fused_kv.v_buffer.index_select(0, values.fused_kv.cache_loc),
        values.fused_kv.value.reshape(config.num_tokens, -1),
        rtol=0,
        atol=0,
    )
