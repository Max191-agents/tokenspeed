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

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    gather_page_table_with_padding,
    store_kv_cache,
    transfer_kv_all_layer,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)
from tokenspeed_numerics_input_generators import (
    FP8KVCacheWriteInputConfig,
    FP8KVCacheWriteInputs,
    KVCacheStoreInputConfig,
    KVCacheStoreInputs,
    KVCacheTransferInputConfig,
    KVCacheTransferInputs,
    MLAKVCacheTransferInputConfig,
    MLAKVCacheTransferInputs,
    PageTableGatherInputConfig,
    PageTableGatherInputs,
    fp8_kv_cache_write_reference,
    kv_cache_store_reference,
    kv_cache_transfer_reference,
    mla_kv_cache_transfer_reference,
    page_table_gather_reference,
)


def _ptr_tensor(layers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [layer.data_ptr() for layer in layers],
        device=layers[0].device,
        dtype=torch.uint64,
    )


def test_fp8_kv_cache_write_generator_runs_paged_scaled_kernel(device: str) -> None:
    values = FP8KVCacheWriteInputs(
        FP8KVCacheWriteInputConfig(
            num_tokens=5,
            num_slots=16,
            num_kv_heads=2,
            head_dim=32,
            page_size=4,
            input_dtype=torch.bfloat16,
            cache_layout="paged",
            input_layout="heads",
            k_scale=1.25,
            v_scale=1.5,
        )
    ).generate(seed=71, metadata_seed=72, device=device)
    expected_k, expected_v = fp8_kv_cache_write_reference(values)

    fused_fp8_set_kv_buffer(
        values.k,
        values.v,
        values.k_cache,
        values.v_cache,
        values.cache_loc,
        k_scale=values.k_scale,
        v_scale=values.v_scale,
        page_size=values.page_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        values.k_cache.float(), expected_k.float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        values.v_cache.float(), expected_v.float(), atol=0, rtol=0
    )


def test_fp8_kv_cache_write_generator_runs_flat_unscaled_kernel(device: str) -> None:
    values = FP8KVCacheWriteInputs(
        FP8KVCacheWriteInputConfig(
            num_tokens=5,
            num_slots=16,
            num_kv_heads=2,
            head_dim=32,
            page_size=4,
            input_dtype=torch.bfloat16,
            cache_layout="flat",
            input_layout="flattened",
        )
    ).generate(seed=73, metadata_seed=74, device=device)
    expected_k, expected_v = fp8_kv_cache_write_reference(values)

    fused_fp8_set_kv_buffer(
        values.k,
        values.v,
        values.k_cache,
        values.v_cache,
        values.cache_loc,
        k_scale=values.k_scale,
        v_scale=values.v_scale,
        page_size=values.page_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        values.k_cache.float(), expected_k.float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        values.v_cache.float(), expected_v.float(), atol=0, rtol=0
    )


def test_kv_cache_store_generator_runs_kernel(device: str) -> None:
    values = KVCacheStoreInputs(
        KVCacheStoreInputConfig(
            num_tokens=4,
            num_slots=8,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.float16,
        )
    ).generate(seed=77, metadata_seed=78, device=device)
    expected_k, expected_v = kv_cache_store_reference(values)

    store_kv_cache(
        values.k_src,
        values.v_src,
        values.k_dst,
        values.v_dst,
        values.loc,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(values.k_dst, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(values.v_dst, expected_v, atol=0, rtol=0)


def test_page_table_gather_generator_runs_kernel(device: str) -> None:
    config = PageTableGatherInputConfig(
        source_rows=8,
        batch_size=4,
        max_num_pages=5,
        page_size=16,
        dummy_slot=999,
    )
    values = PageTableGatherInputs(config).generate(
        seed=79,
        metadata_seed=80,
        device=device,
    )
    expected = page_table_gather_reference(
        values,
        page_size=config.page_size,
        dummy_slot=config.dummy_slot,
    )

    gather_page_table_with_padding(
        values.req_to_page,
        values.req_pool_indices,
        values.seq_lens,
        values.out,
        bs=config.batch_size,
        max_num_pages=config.max_num_pages,
        page_size=config.page_size,
        dummy_slot=config.dummy_slot,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(values.out, expected, atol=0, rtol=0)


def test_kv_cache_transfer_generator_runs_per_layer_kernel(device: str) -> None:
    values = KVCacheTransferInputs(
        KVCacheTransferInputConfig(
            num_layers=1,
            num_slots=8,
            num_transfers=4,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.float16,
        )
    ).generate(seed=81, metadata_seed=82, device=device)
    expected_k, expected_v = kv_cache_transfer_reference(values)

    item_size = values.src_k_layers[0].stride(0) * values.src_k_layers[0].element_size()
    transfer_kv_per_layer(
        src_k=values.src_k_layers[0],
        dst_k=values.dst_k_layers[0],
        src_v=values.src_v_layers[0],
        dst_v=values.dst_v_layers[0],
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=item_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(values.dst_k_layers[0], expected_k[0], atol=0, rtol=0)
    torch.testing.assert_close(values.dst_v_layers[0], expected_v[0], atol=0, rtol=0)


def test_kv_cache_transfer_generator_runs_all_layer_kernel(device: str) -> None:
    values = KVCacheTransferInputs(
        KVCacheTransferInputConfig(
            num_layers=3,
            num_slots=8,
            num_transfers=4,
            num_kv_heads=2,
            head_dim=32,
            dtype=torch.float16,
        )
    ).generate(seed=83, metadata_seed=84, device=device)
    expected_k, expected_v = kv_cache_transfer_reference(values)

    item_size = values.src_k_layers[0].stride(0) * values.src_k_layers[0].element_size()
    transfer_kv_all_layer(
        src_k_layers=_ptr_tensor(values.src_k_layers),
        dst_k_layers=_ptr_tensor(values.dst_k_layers),
        src_v_layers=_ptr_tensor(values.src_v_layers),
        dst_v_layers=_ptr_tensor(values.dst_v_layers),
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=item_size,
        num_layers=len(values.src_k_layers),
    )
    torch.cuda.synchronize()

    for layer_idx in range(len(values.src_k_layers)):
        torch.testing.assert_close(
            values.dst_k_layers[layer_idx],
            expected_k[layer_idx],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            values.dst_v_layers[layer_idx],
            expected_v[layer_idx],
            atol=0,
            rtol=0,
        )


def test_mla_kv_cache_transfer_generator_runs_per_layer_kernel(device: str) -> None:
    values = MLAKVCacheTransferInputs(
        MLAKVCacheTransferInputConfig(
            num_layers=1,
            num_slots=8,
            num_transfers=4,
            kv_cache_dim=64,
            dtype=torch.float16,
        )
    ).generate(seed=85, metadata_seed=86, device=device)
    expected = mla_kv_cache_transfer_reference(values)

    item_size = values.src_layers[0].stride(0) * values.src_layers[0].element_size()
    transfer_kv_per_layer_mla(
        src=values.src_layers[0],
        dst=values.dst_layers[0],
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=item_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(values.dst_layers[0], expected[0], atol=0, rtol=0)


def test_mla_kv_cache_transfer_generator_runs_all_layer_kernel(device: str) -> None:
    values = MLAKVCacheTransferInputs(
        MLAKVCacheTransferInputConfig(
            num_layers=3,
            num_slots=8,
            num_transfers=4,
            kv_cache_dim=64,
            dtype=torch.float16,
        )
    ).generate(seed=87, metadata_seed=88, device=device)
    expected = mla_kv_cache_transfer_reference(values)

    item_size = values.src_layers[0].stride(0) * values.src_layers[0].element_size()
    transfer_kv_all_layer_mla(
        src_layers=_ptr_tensor(values.src_layers),
        dst_layers=_ptr_tensor(values.dst_layers),
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=item_size,
        num_layers=len(values.src_layers),
    )
    torch.cuda.synchronize()

    for layer_idx in range(len(values.src_layers)):
        torch.testing.assert_close(
            values.dst_layers[layer_idx],
            expected[layer_idx],
            atol=0,
            rtol=0,
        )
