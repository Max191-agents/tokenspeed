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

from dataclasses import fields

import pytest
import torch
from tokenspeed_numerics_input_generators import (
    MLARopeInputConfig,
    MLARopeInputs,
    MLARopeInputValues,
    RopeInputConfig,
    RopeInputs,
)


def test_mla_rope_values_contain_only_operation_inputs() -> None:
    assert tuple(field.name for field in fields(MLARopeInputValues)) == (
        "q_rope",
        "k_rope",
        "q_nope",
        "k_nope",
        "cos_sin_cache",
        "positions",
    )


def test_rope_inputs_generate_neox_full_head_values() -> None:
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
    assert values.query.dtype == torch.bfloat16
    assert values.key.dtype == torch.bfloat16


def test_rope_inputs_generate_gptj_layout_values() -> None:
    config = RopeInputConfig(
        num_tokens=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        is_neox=False,
    )
    values = RopeInputs(config).generate(seed=25, metadata_seed=26, device="cpu")

    assert values.query.shape == (4, 16)
    assert values.key.shape == (4, 8)
    assert values.cos_sin_cache.shape == (1024, 8)


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


def test_mla_rope_inputs_generate_rank2_values() -> None:
    config = MLARopeInputConfig(
        num_tokens=4,
        num_q_heads=3,
        qk_nope_head_dim=5,
        qk_rope_head_dim=8,
        dtype=torch.bfloat16,
        max_position=32,
    )
    values = MLARopeInputs(config).generate(
        seed=31,
        metadata_seed=32,
        device="cpu",
    )

    assert values.q_rope.shape == (4, 3, 8)
    assert values.q_nope.shape == (4, 3, 5)
    assert values.k_rope.shape == (4, 8)
    assert values.k_nope.shape == (4, 5)
    assert values.q_rope.dtype == torch.bfloat16
    assert values.k_nope.dtype == torch.bfloat16
    assert values.cos_sin_cache.shape == (32, 8)
    assert values.positions.shape == (4,)


def test_mla_rope_inputs_generate_rank3_values() -> None:
    config = MLARopeInputConfig(
        num_tokens=3,
        num_q_heads=4,
        qk_nope_head_dim=6,
        qk_rope_head_dim=10,
        dtype=torch.float64,
        k_rank=3,
        num_kv_heads=2,
        is_neox=False,
        max_position=64,
    )
    values = MLARopeInputs(config).generate(
        seed=33,
        metadata_seed=34,
        device="cpu",
    )

    assert values.k_rope.shape == (3, 2, 10)
    assert values.k_nope.shape == (3, 2, 6)
    assert values.q_rope.shape == (3, 4, 10)
    assert values.q_nope.shape == (3, 4, 6)
    assert values.q_rope.dtype == torch.float64
    assert values.k_nope.dtype == torch.float64


def test_mla_rope_metadata_seed_controls_positions_only() -> None:
    generator = MLARopeInputs(
        MLARopeInputConfig(
            num_tokens=5,
            num_q_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=8,
            dtype=torch.float16,
        )
    )

    values1 = generator.generate(seed=35, metadata_seed=201, device="cpu")
    values2 = generator.generate(seed=36, metadata_seed=201, device="cpu")

    torch.testing.assert_close(values1.positions, values2.positions)
    assert not torch.equal(values1.q_rope, values2.q_rope)
    assert not torch.equal(values1.k_nope, values2.k_nope)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    ],
)
def test_mla_rope_supports_float_dtypes(dtype: torch.dtype) -> None:
    values = MLARopeInputs(
        MLARopeInputConfig(
            num_tokens=2,
            num_q_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=8,
            dtype=dtype,
        )
    ).generate(seed=37, device="cpu")

    assert values.q_rope.dtype == dtype
    assert values.q_nope.dtype == dtype
    assert values.k_rope.dtype == dtype
    assert values.k_nope.dtype == dtype


def test_mla_rope_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="implicit shared KV head"):
        MLARopeInputs(
            MLARopeInputConfig(
                num_tokens=1,
                num_q_heads=1,
                qk_nope_head_dim=4,
                qk_rope_head_dim=8,
                dtype=torch.float16,
                k_rank=2,
                num_kv_heads=2,
            )
        )
    with pytest.raises(ValueError, match="dtype"):
        MLARopeInputs(
            MLARopeInputConfig(
                num_tokens=1,
                num_q_heads=1,
                qk_nope_head_dim=4,
                qk_rope_head_dim=8,
                dtype=torch.int32,
            )
        )
