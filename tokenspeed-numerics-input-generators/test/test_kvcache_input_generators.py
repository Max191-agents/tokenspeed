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
    KVCacheTransferInputConfig,
    KVCacheTransferInputs,
    MLAKVCacheTransferInputConfig,
    MLAKVCacheTransferInputs,
    kv_cache_transfer_reference,
    mla_kv_cache_transfer_reference,
)


def test_kv_cache_transfer_inputs_generate_values_and_reference() -> None:
    values = KVCacheTransferInputs(
        KVCacheTransferInputConfig(
            num_layers=3,
            num_slots=8,
            num_transfers=4,
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.float16,
        )
    ).generate(seed=31, metadata_seed=32, device="cpu")

    assert len(values.src_k_layers) == 3
    assert len(values.dst_k_layers) == 3
    assert values.src_k_layers[0].shape == (8, 2, 16)
    assert values.src_v_layers[0].shape == (8, 2, 16)
    assert values.dst_indices.unique().numel() == 4
    assert values.src_indices.unique().numel() == 4

    expected_k, expected_v = kv_cache_transfer_reference(values)
    for layer_idx in range(3):
        dst = values.dst_indices.to(torch.int64)
        src = values.src_indices.to(torch.int64)
        torch.testing.assert_close(
            expected_k[layer_idx][dst],
            values.src_k_layers[layer_idx][src],
        )
        torch.testing.assert_close(
            expected_v[layer_idx][dst],
            values.src_v_layers[layer_idx][src],
        )


def test_kv_cache_transfer_metadata_seed_controls_indices_only() -> None:
    generator = KVCacheTransferInputs(
        KVCacheTransferInputConfig(
            num_layers=2,
            num_slots=8,
            num_transfers=4,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=33, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=34, metadata_seed=99, device="cpu")

    torch.testing.assert_close(values1.src_indices, values2.src_indices)
    torch.testing.assert_close(values1.dst_indices, values2.dst_indices)
    assert not torch.equal(values1.src_k_layers[0], values2.src_k_layers[0])


def test_mla_kv_cache_transfer_inputs_generate_values_and_reference() -> None:
    values = MLAKVCacheTransferInputs(
        MLAKVCacheTransferInputConfig(
            num_layers=2,
            num_slots=7,
            num_transfers=3,
            kv_cache_dim=24,
            dtype=torch.bfloat16,
        )
    ).generate(seed=35, metadata_seed=36, device="cpu")

    assert len(values.src_layers) == 2
    assert values.src_layers[0].shape == (7, 1, 24)
    assert values.dst_layers[0].shape == (7, 1, 24)
    assert values.dst_indices.unique().numel() == 3

    expected = mla_kv_cache_transfer_reference(values)
    dst = values.dst_indices.to(torch.int64)
    src = values.src_indices.to(torch.int64)
    for layer_idx in range(2):
        torch.testing.assert_close(
            expected[layer_idx][dst],
            values.src_layers[layer_idx][src],
        )


@pytest.mark.parametrize("num_transfers", [-1, 9])
def test_kv_cache_transfer_rejects_invalid_transfer_count(
    num_transfers: int,
) -> None:
    with pytest.raises(ValueError, match="num_transfers"):
        KVCacheTransferInputs(
            KVCacheTransferInputConfig(
                num_layers=1,
                num_slots=8,
                num_transfers=num_transfers,
                num_kv_heads=2,
                head_dim=8,
                dtype=torch.float16,
            )
        )


def test_kv_cache_transfer_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="regular floating"):
        KVCacheTransferInputs(
            KVCacheTransferInputConfig(
                num_layers=1,
                num_slots=8,
                num_transfers=2,
                num_kv_heads=2,
                head_dim=8,
                dtype=torch.float8_e4m3fn,
            )
        )


def test_mla_kv_cache_transfer_rejects_invalid_index_dtype() -> None:
    with pytest.raises(ValueError, match="index_dtype"):
        MLAKVCacheTransferInputs(
            MLAKVCacheTransferInputConfig(
                num_layers=1,
                num_slots=8,
                num_transfers=2,
                kv_cache_dim=16,
                dtype=torch.float16,
                index_dtype=torch.float32,
            )
        )
