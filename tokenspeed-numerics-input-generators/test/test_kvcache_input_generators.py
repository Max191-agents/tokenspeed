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
    KVCacheStoreInputConfig,
    KVCacheStoreInputs,
    KVCacheTransferInputConfig,
    KVCacheTransferInputs,
    MLAKVCacheTransferInputConfig,
    MLAKVCacheTransferInputs,
    PageTableGatherInputConfig,
    PageTableGatherInputs,
    kv_cache_store_reference,
    kv_cache_transfer_reference,
    mla_kv_cache_transfer_reference,
    page_table_gather_reference,
)


def test_kv_cache_store_inputs_generate_values_and_reference() -> None:
    values = KVCacheStoreInputs(
        KVCacheStoreInputConfig(
            num_tokens=4,
            num_slots=8,
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.float16,
        )
    ).generate(seed=21, metadata_seed=22, device="cpu")

    assert values.k_src.shape == (4, 2, 16)
    assert values.v_src.shape == (4, 2, 16)
    assert values.k_dst.shape == (8, 2, 16)
    assert values.v_dst.shape == (8, 2, 16)
    assert values.loc.unique().numel() == 4

    expected_k, expected_v = kv_cache_store_reference(values)
    loc = values.loc.to(torch.int64)
    torch.testing.assert_close(expected_k[loc], values.k_src)
    torch.testing.assert_close(expected_v[loc], values.v_src)

    untouched = torch.ones(values.k_dst.shape[0], dtype=torch.bool)
    untouched[loc.cpu()] = False
    torch.testing.assert_close(expected_k[untouched], values.k_dst[untouched])
    torch.testing.assert_close(expected_v[untouched], values.v_dst[untouched])


def test_kv_cache_store_metadata_seed_controls_locations_only() -> None:
    generator = KVCacheStoreInputs(
        KVCacheStoreInputConfig(
            num_tokens=4,
            num_slots=8,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=23, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=24, metadata_seed=99, device="cpu")

    torch.testing.assert_close(values1.loc, values2.loc)
    assert not torch.equal(values1.k_src, values2.k_src)


def test_page_table_gather_inputs_generate_values_and_reference() -> None:
    config = PageTableGatherInputConfig(
        source_rows=8,
        batch_size=4,
        max_num_pages=5,
        page_size=16,
        dummy_slot=999,
    )
    values = PageTableGatherInputs(config).generate(
        seed=25,
        metadata_seed=26,
        device="cpu",
    )

    assert values.req_to_page.shape == (8, 5)
    assert values.req_pool_indices.shape == (4,)
    assert values.seq_lens.shape == (4,)
    assert values.out.shape == (4, 5)
    assert values.req_pool_indices.unique().numel() == 4
    assert torch.all(values.seq_lens >= 0)
    assert torch.all(values.seq_lens <= config.max_num_pages * config.page_size)

    expected = page_table_gather_reference(
        values,
        page_size=config.page_size,
        dummy_slot=config.dummy_slot,
    )
    for row_idx in range(config.batch_size):
        seq_len = int(values.seq_lens[row_idx].item())
        n_pages = (seq_len + config.page_size - 1) // config.page_size
        req_idx = int(values.req_pool_indices[row_idx].item())
        torch.testing.assert_close(
            expected[row_idx, :n_pages],
            values.req_to_page[req_idx, :n_pages],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            expected[row_idx, n_pages:],
            torch.full_like(expected[row_idx, n_pages:], config.dummy_slot),
            atol=0,
            rtol=0,
        )


def test_page_table_gather_metadata_seed_controls_request_layout_only() -> None:
    generator = PageTableGatherInputs(
        PageTableGatherInputConfig(
            source_rows=8,
            batch_size=4,
            max_num_pages=5,
            page_size=16,
        )
    )

    values1 = generator.generate(seed=27, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=28, metadata_seed=99, device="cpu")

    torch.testing.assert_close(values1.req_pool_indices, values2.req_pool_indices)
    torch.testing.assert_close(values1.seq_lens, values2.seq_lens)
    assert not torch.equal(values1.req_to_page, values2.req_to_page)


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


@pytest.mark.parametrize("num_tokens", [-1, 9])
def test_kv_cache_store_rejects_invalid_token_count(num_tokens: int) -> None:
    with pytest.raises(ValueError, match="num_tokens"):
        KVCacheStoreInputs(
            KVCacheStoreInputConfig(
                num_tokens=num_tokens,
                num_slots=8,
                num_kv_heads=2,
                head_dim=8,
                dtype=torch.float16,
            )
        )


def test_page_table_gather_rejects_batch_larger_than_source_table() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        PageTableGatherInputs(
            PageTableGatherInputConfig(
                source_rows=4,
                batch_size=5,
                max_num_pages=3,
                page_size=16,
            )
        )


def test_page_table_gather_rejects_negative_dummy_slot() -> None:
    with pytest.raises(ValueError, match="dummy_slot"):
        PageTableGatherInputs(
            PageTableGatherInputConfig(
                source_rows=4,
                batch_size=2,
                max_num_pages=3,
                page_size=16,
                dummy_slot=-1,
            )
        )
