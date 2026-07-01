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
    AllGatherInputConfig,
    AllGatherInputs,
    AllReduceInputConfig,
    AllReduceInputs,
    ReduceScatterInputConfig,
    ReduceScatterInputs,
    all_gather_reference,
    all_reduce_sum_reference,
    reduce_scatter_sum_reference,
)


def test_all_reduce_inputs_generate_values_and_reference() -> None:
    values = AllReduceInputs(
        AllReduceInputConfig(
            world_size=3,
            shape=(4, 8),
            dtype=torch.float32,
        )
    ).generate(seed=11, device="cpu")

    assert len(values.rank_inputs) == 3
    assert all(rank_input.shape == (4, 8) for rank_input in values.rank_inputs)

    expected = all_reduce_sum_reference(values)
    manual = sum(rank_input for rank_input in values.rank_inputs)
    torch.testing.assert_close(expected, manual)


def test_all_gather_inputs_generate_values_and_reference() -> None:
    config = AllGatherInputConfig(
        world_size=4,
        total_tokens=13,
        hidden_size=8,
        max_tokens_per_rank=6,
        dtype=torch.float32,
    )
    values = AllGatherInputs(config).generate(
        seed=12,
        metadata_seed=13,
        device="cpu",
    )

    assert len(values.rank_inputs) == 4
    assert sum(values.tokens_per_rank) == 13
    assert max(values.tokens_per_rank) <= 6
    for rank, rank_input in enumerate(values.rank_inputs):
        assert rank_input.shape == (values.tokens_per_rank[rank], 8)

    expected = all_gather_reference(values)
    torch.testing.assert_close(expected, torch.cat(values.rank_inputs, dim=0))


def test_reduce_scatter_inputs_generate_values_and_reference() -> None:
    config = ReduceScatterInputConfig(
        world_size=4,
        total_tokens=13,
        hidden_size=8,
        max_tokens_per_rank=6,
        dtype=torch.float32,
    )
    values = ReduceScatterInputs(config).generate(
        seed=14,
        metadata_seed=15,
        device="cpu",
    )

    assert len(values.rank_inputs) == 4
    assert sum(values.tokens_per_rank) == 13
    assert max(values.tokens_per_rank) <= 6
    assert all(rank_input.shape == (13, 8) for rank_input in values.rank_inputs)

    expected = reduce_scatter_sum_reference(values)
    reduced = sum(values.rank_inputs)
    offset = 0
    for rank, num_tokens in enumerate(values.tokens_per_rank):
        torch.testing.assert_close(
            expected[rank],
            reduced[offset : offset + num_tokens],
        )
        offset += num_tokens


def test_collective_metadata_seed_controls_token_distribution_only() -> None:
    generator = AllGatherInputs(
        AllGatherInputConfig(
            world_size=4,
            total_tokens=17,
            hidden_size=8,
            max_tokens_per_rank=7,
            dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=16, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=17, metadata_seed=99, device="cpu")

    assert values1.tokens_per_rank == values2.tokens_per_rank
    assert not torch.equal(values1.rank_inputs[0], values2.rank_inputs[0])


def test_zero_token_all_gather_and_reduce_scatter_are_valid() -> None:
    all_gather_values = AllGatherInputs(
        AllGatherInputConfig(
            world_size=3,
            total_tokens=0,
            hidden_size=8,
            max_tokens_per_rank=0,
            dtype=torch.float32,
        )
    ).generate(seed=18, metadata_seed=19, device="cpu")
    reduce_scatter_values = ReduceScatterInputs(
        ReduceScatterInputConfig(
            world_size=3,
            total_tokens=0,
            hidden_size=8,
            max_tokens_per_rank=0,
            dtype=torch.float32,
        )
    ).generate(seed=20, metadata_seed=21, device="cpu")

    assert all_gather_reference(all_gather_values).shape == (0, 8)
    assert [
        output.shape for output in reduce_scatter_sum_reference(reduce_scatter_values)
    ] == [
        (0, 8),
        (0, 8),
        (0, 8),
    ]


@pytest.mark.parametrize("world_size", [0, -1])
def test_collectives_reject_invalid_world_size(world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size"):
        AllReduceInputs(
            AllReduceInputConfig(
                world_size=world_size,
                shape=(8,),
            )
        )


def test_all_reduce_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        AllReduceInputs(
            AllReduceInputConfig(
                world_size=2,
                shape=(),
            )
        )


def test_collectives_reject_impossible_token_cap() -> None:
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        AllGatherInputs(
            AllGatherInputConfig(
                world_size=2,
                total_tokens=5,
                hidden_size=8,
                max_tokens_per_rank=2,
            )
        )


def test_collectives_reject_nonfloating_dtype() -> None:
    with pytest.raises(ValueError, match="regular floating"):
        ReduceScatterInputs(
            ReduceScatterInputConfig(
                world_size=2,
                total_tokens=5,
                hidden_size=8,
                dtype=torch.int32,
            )
        )
