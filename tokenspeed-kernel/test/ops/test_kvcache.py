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
    transfer_kv_all_layer,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)
from tokenspeed_numerics_input_generators import (
    KVCacheTransferInputConfig,
    KVCacheTransferInputs,
    MLAKVCacheTransferInputConfig,
    MLAKVCacheTransferInputs,
    kv_cache_transfer_reference,
    mla_kv_cache_transfer_reference,
)


def _ptr_tensor(layers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [layer.data_ptr() for layer in layers],
        device=layers[0].device,
        dtype=torch.uint64,
    )


def test_transfer_kv_per_layer(device: str) -> None:
    values = KVCacheTransferInputs(
        KVCacheTransferInputConfig(
            num_layers=1,
            num_slots=6,
            num_transfers=2,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.float16,
        )
    ).generate(seed=121, metadata_seed=122, device=device)
    expected_k, expected_v = kv_cache_transfer_reference(values)

    transfer_kv_per_layer(
        src_k=values.src_k_layers[0],
        dst_k=values.dst_k_layers[0],
        src_v=values.src_v_layers[0],
        dst_v=values.dst_v_layers[0],
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=values.src_k_layers[0].stride(0)
        * values.src_k_layers[0].element_size(),
    )

    torch.cuda.synchronize()

    assert torch.equal(values.dst_k_layers[0], expected_k[0])
    assert torch.equal(values.dst_v_layers[0], expected_v[0])


def test_transfer_kv_all_layer(device: str) -> None:
    values = KVCacheTransferInputs(
        KVCacheTransferInputConfig(
            num_layers=3,
            num_slots=6,
            num_transfers=2,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.float16,
        )
    ).generate(seed=123, metadata_seed=124, device=device)
    expected_k, expected_v = kv_cache_transfer_reference(values)
    slot_stride_bytes = (
        values.src_k_layers[0].stride(0) * values.src_k_layers[0].element_size()
    )

    transfer_kv_all_layer(
        src_k_layers=_ptr_tensor(values.src_k_layers),
        dst_k_layers=_ptr_tensor(values.dst_k_layers),
        src_v_layers=_ptr_tensor(values.src_v_layers),
        dst_v_layers=_ptr_tensor(values.dst_v_layers),
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=slot_stride_bytes,
        num_layers=len(values.src_k_layers),
    )

    torch.cuda.synchronize()

    for layer_idx in range(len(values.src_k_layers)):
        assert torch.equal(values.dst_k_layers[layer_idx], expected_k[layer_idx])
        assert torch.equal(values.dst_v_layers[layer_idx], expected_v[layer_idx])


def test_transfer_kv_per_layer_mla(device: str) -> None:
    values = MLAKVCacheTransferInputs(
        MLAKVCacheTransferInputConfig(
            num_layers=1,
            num_slots=6,
            num_transfers=2,
            kv_cache_dim=576,
            dtype=torch.float16,
        )
    ).generate(seed=125, metadata_seed=126, device=device)
    expected = mla_kv_cache_transfer_reference(values)

    transfer_kv_per_layer_mla(
        src=values.src_layers[0],
        dst=values.dst_layers[0],
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=values.src_layers[0].stride(0) * values.src_layers[0].element_size(),
    )

    torch.cuda.synchronize()

    assert torch.equal(values.dst_layers[0], expected[0])


def test_transfer_kv_all_layer_mla(device: str) -> None:
    values = MLAKVCacheTransferInputs(
        MLAKVCacheTransferInputConfig(
            num_layers=3,
            num_slots=6,
            num_transfers=2,
            kv_cache_dim=576,
            dtype=torch.float16,
        )
    ).generate(seed=127, metadata_seed=128, device=device)
    expected = mla_kv_cache_transfer_reference(values)
    slot_stride_bytes = (
        values.src_layers[0].stride(0) * values.src_layers[0].element_size()
    )

    transfer_kv_all_layer_mla(
        src_layers=_ptr_tensor(values.src_layers),
        dst_layers=_ptr_tensor(values.dst_layers),
        src_indices=values.src_indices,
        dst_indices=values.dst_indices,
        item_size=slot_stride_bytes,
        num_layers=len(values.src_layers),
    )

    torch.cuda.synchronize()

    for layer_idx in range(len(values.src_layers)):
        assert torch.equal(values.dst_layers[layer_idx], expected[layer_idx])
