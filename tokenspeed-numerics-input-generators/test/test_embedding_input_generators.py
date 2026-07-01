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
from tokenspeed_numerics_input_generators import (
    RopeInputConfig,
    RopeInputs,
    rope_reference,
)


def test_rope_inputs_generate_neox_full_head_values_and_reference() -> None:
    config = RopeInputConfig(
        num_tokens=5,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.bfloat16,
    )
    values = RopeInputs(config).generate(seed=21, metadata_seed=22, device="cpu")

    assert values.query.shape == (5, 64)
    assert values.key.shape == (5, 32)
    assert values.positions.shape == (5,)
    assert values.positions.dtype == torch.int64
    assert values.cos_sin_cache.shape == (1024, 16)
    assert values.fused_kv is None

    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )
    assert q_ref.shape == values.query.shape
    assert k_ref.shape == values.key.shape
    assert q_ref.dtype == values.query.dtype
    assert k_ref.dtype == values.key.dtype


def test_rope_reference_preserves_partial_rotary_tail() -> None:
    config = RopeInputConfig(
        num_tokens=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=16,
        rotary_dim=8,
        dtype=torch.float32,
        is_neox=True,
    )
    values = RopeInputs(config).generate(seed=23, metadata_seed=24, device="cpu")

    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=True,
        rotary_dim=config.rotary_dim,
    )

    q_tail = q_ref.view(3, 2, 16)[..., 8:]
    q_orig_tail = values.query.view(3, 2, 16)[..., 8:]
    k_tail = k_ref.view(3, 1, 16)[..., 8:]
    k_orig_tail = values.key.view(3, 1, 16)[..., 8:]
    torch.testing.assert_close(q_tail, q_orig_tail)
    torch.testing.assert_close(k_tail, k_orig_tail)


def test_rope_inputs_generate_gptj_layout_reference() -> None:
    config = RopeInputConfig(
        num_tokens=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        is_neox=False,
    )
    values = RopeInputs(config).generate(seed=25, metadata_seed=26, device="cpu")
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=False,
        rotary_dim=config.rotary_dim,
    )

    assert q_ref.shape == values.query.shape
    assert k_ref.shape == values.key.shape
    assert not torch.equal(q_ref, values.query)
    assert not torch.equal(k_ref, values.key)


def test_rope_inputs_generate_fused_kv_and_output_buffers() -> None:
    config = RopeInputConfig(
        num_tokens=6,
        num_q_heads=3,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        max_position=64,
        with_fused_kv=True,
        cache_size=10,
        with_q_output=True,
        with_k_output=True,
    )
    values = RopeInputs(config).generate(seed=27, metadata_seed=28, device="cpu")

    assert values.fused_kv is not None
    assert values.fused_kv.value.shape == (6, 2, 8)
    assert values.fused_kv.k_buffer.shape == (10, 16)
    assert values.fused_kv.v_buffer.shape == (10, 16)
    assert values.fused_kv.cache_loc.shape == (6,)
    assert values.fused_kv.cache_loc.dtype == torch.int32
    assert values.fused_kv.cache_loc.unique().numel() == 6
    assert int(values.fused_kv.cache_loc.min()) >= 0
    assert int(values.fused_kv.cache_loc.max()) < 10
    assert values.output_q_rope is not None
    assert values.output_q_rope.shape == values.query.shape
    assert values.output_k_rope is not None
    assert values.output_k_rope.shape == values.key.shape


def test_rope_metadata_seed_controls_positions_and_cache_locs() -> None:
    generator = RopeInputs(
        RopeInputConfig(
            num_tokens=6,
            num_q_heads=3,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.float32,
            with_fused_kv=True,
            cache_size=10,
        )
    )

    values1 = generator.generate(seed=29, metadata_seed=101, device="cpu")
    values2 = generator.generate(seed=30, metadata_seed=101, device="cpu")

    assert values1.fused_kv is not None
    assert values2.fused_kv is not None
    torch.testing.assert_close(values1.positions, values2.positions)
    torch.testing.assert_close(values1.fused_kv.cache_loc, values2.fused_kv.cache_loc)
    assert not torch.equal(values1.query, values2.query)
    assert not torch.equal(values1.fused_kv.value, values2.fused_kv.value)


@pytest.mark.parametrize("bad_rotary_dim", [0, 3, 16])
def test_rope_rejects_invalid_rotary_dim(bad_rotary_dim: int) -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        RopeInputs(
            RopeInputConfig(
                num_tokens=1,
                num_q_heads=1,
                num_kv_heads=1,
                head_size=8,
                rotary_dim=bad_rotary_dim,
                dtype=torch.float16,
            )
        )


def test_rope_rejects_invalid_position_dtype() -> None:
    with pytest.raises(ValueError, match="position_dtype"):
        RopeInputs(
            RopeInputConfig(
                num_tokens=1,
                num_q_heads=1,
                num_kv_heads=1,
                head_size=8,
                dtype=torch.float16,
                position_dtype=torch.float32,
            )
        )


def test_rope_rejects_too_small_cache_for_fused_kv() -> None:
    with pytest.raises(ValueError, match="cache_size"):
        RopeInputs(
            RopeInputConfig(
                num_tokens=4,
                num_q_heads=1,
                num_kv_heads=1,
                head_size=8,
                dtype=torch.float16,
                with_fused_kv=True,
                cache_size=3,
            )
        )
