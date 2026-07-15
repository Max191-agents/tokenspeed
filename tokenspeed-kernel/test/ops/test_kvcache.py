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

import math

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


def _bounded_values(
    shape: tuple[int, ...],
    *,
    device: str,
    dtype: torch.dtype,
    offset: int = 0,
) -> torch.Tensor:
    values = (torch.arange(math.prod(shape), device=device) + offset) % 31
    return ((values.to(torch.float32) - 15) / 16).reshape(shape).to(dtype)


def test_fp8_kv_cache_write_paged_scaled(device: str) -> None:
    page_size = 4
    input_shape = (5, 2, 32)
    cache_shape = (4, page_size, 2, 32)
    k = _bounded_values(input_shape, device=device, dtype=torch.bfloat16)
    v = _bounded_values(
        input_shape,
        device=device,
        dtype=torch.bfloat16,
        offset=7,
    )
    k_cache = torch.zeros(cache_shape, device=device, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    cache_loc = torch.tensor([0, 3, 5, 10, 15], device=device, dtype=torch.int32)
    k_scale = 1.25
    v_scale = 1.5

    expected_k = k_cache.clone()
    expected_v = v_cache.clone()
    pages = cache_loc.to(torch.int64) // page_size
    offsets = cache_loc.to(torch.int64) % page_size
    expected_k[pages, offsets] = (k.float() / k_scale).to(k_cache.dtype)
    expected_v[pages, offsets] = (v.float() / v_scale).to(v_cache.dtype)

    fused_fp8_set_kv_buffer(
        k,
        v,
        k_cache,
        v_cache,
        cache_loc,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(k_cache.float(), expected_k.float())
    assert torch.equal(v_cache.float(), expected_v.float())


def test_fp8_kv_cache_write_flat_unscaled(device: str) -> None:
    page_size = 4
    num_kv_heads = 2
    head_dim = 32
    input_shape = (5, num_kv_heads * head_dim)
    cache_shape = (16, num_kv_heads, head_dim)
    k = _bounded_values(input_shape, device=device, dtype=torch.bfloat16)
    v = _bounded_values(
        input_shape,
        device=device,
        dtype=torch.bfloat16,
        offset=7,
    )
    k_cache = torch.zeros(cache_shape, device=device, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    cache_loc = torch.tensor([0, 3, 5, 10, 15], device=device, dtype=torch.int32)

    expected_k = k_cache.clone()
    expected_v = v_cache.clone()
    expected_k[cache_loc.to(torch.int64)] = k.view(5, num_kv_heads, head_dim).to(
        k_cache.dtype
    )
    expected_v[cache_loc.to(torch.int64)] = v.view(5, num_kv_heads, head_dim).to(
        v_cache.dtype
    )

    fused_fp8_set_kv_buffer(
        k,
        v,
        k_cache,
        v_cache,
        cache_loc,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(k_cache.float(), expected_k.float())
    assert torch.equal(v_cache.float(), expected_v.float())


def test_store_kv_cache(device: str) -> None:
    source_shape = (4, 2, 32)
    destination_shape = (8, 2, 32)
    k_src = _bounded_values(source_shape, device=device, dtype=torch.float16)
    v_src = _bounded_values(
        source_shape,
        device=device,
        dtype=torch.float16,
        offset=7,
    )
    k_dst = torch.zeros(destination_shape, device=device, dtype=torch.float16)
    v_dst = torch.zeros_like(k_dst)
    locations = torch.tensor([1, 3, 4, 7], device=device, dtype=torch.int32)

    expected_k = k_dst.clone()
    expected_v = v_dst.clone()
    expected_k[locations.to(torch.int64)] = k_src
    expected_v[locations.to(torch.int64)] = v_src

    store_kv_cache(k_src, v_src, k_dst, v_dst, locations)
    torch.cuda.synchronize()

    assert torch.equal(k_dst, expected_k)
    assert torch.equal(v_dst, expected_v)


def test_gather_page_table_with_padding(device: str) -> None:
    batch_size = 4
    max_num_pages = 5
    page_size = 16
    dummy_slot = 999
    req_to_page = torch.arange(
        8 * max_num_pages,
        device=device,
        dtype=torch.int32,
    ).reshape(8, max_num_pages)
    req_pool_indices = torch.tensor([6, 1, 4, 0], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([0, 1, 17, 80], device=device, dtype=torch.int32)
    out = torch.full(
        (batch_size, max_num_pages),
        dummy_slot,
        device=device,
        dtype=torch.int32,
    )

    expected = torch.full_like(out, dummy_slot)
    for row, (request_index, seq_len) in enumerate(
        zip(req_pool_indices.tolist(), seq_lens.tolist(), strict=True)
    ):
        num_pages = (seq_len + page_size - 1) // page_size
        expected[row, :num_pages] = req_to_page[request_index, :num_pages]

    gather_page_table_with_padding(
        req_to_page,
        req_pool_indices,
        seq_lens,
        out,
        bs=batch_size,
        max_num_pages=max_num_pages,
        page_size=page_size,
        dummy_slot=dummy_slot,
    )
    torch.cuda.synchronize()

    assert torch.equal(out, expected)


def test_transfer_kv_per_layer(device: str) -> None:
    num_slots = 6
    num_heads = 8
    head_dim = 128
    element_dim = num_heads * head_dim

    k_cache_dst = torch.zeros(
        num_slots, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v_cache_dst = torch.zeros_like(k_cache_dst)

    k_cache_src = torch.arange(
        num_slots * num_heads * head_dim,
        device=device,
        dtype=torch.float16,
    ).reshape(num_slots, num_heads, head_dim)
    v_cache_src = torch.arange(
        10_000,
        10_000 + num_slots * num_heads * head_dim,
        device=device,
        dtype=torch.float16,
    ).reshape(num_slots, num_heads, head_dim)

    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)

    expected_k = k_cache_dst.clone()
    expected_v = v_cache_dst.clone()
    expected_k[indices_dst.to(torch.int64)] = k_cache_src[indices_src.to(torch.int64)]
    expected_v[indices_dst.to(torch.int64)] = v_cache_src[indices_src.to(torch.int64)]

    transfer_kv_per_layer(
        src_k=k_cache_src,
        dst_k=k_cache_dst,
        src_v=v_cache_src,
        dst_v=v_cache_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=element_dim * k_cache_src.element_size(),
    )

    torch.cuda.synchronize()

    assert torch.equal(k_cache_dst, expected_k)
    assert torch.equal(v_cache_dst, expected_v)


def test_transfer_kv_all_layer(device: str) -> None:
    num_layers = 3
    num_slots = 6
    num_heads = 8
    head_dim = 128

    k_layers_dst = [
        torch.zeros(num_slots, num_heads, head_dim, device=device, dtype=torch.float16)
        for _ in range(num_layers)
    ]
    v_layers_dst = [torch.zeros_like(k_layers_dst[0]) for _ in range(num_layers)]
    k_layers_src = [
        torch.arange(
            layer_idx * num_slots * num_heads * head_dim,
            (layer_idx + 1) * num_slots * num_heads * head_dim,
            device=device,
            dtype=torch.float16,
        ).reshape(num_slots, num_heads, head_dim)
        for layer_idx in range(num_layers)
    ]
    v_layers_src = [
        torch.arange(
            20_000 + layer_idx * num_slots * num_heads * head_dim,
            20_000 + (layer_idx + 1) * num_slots * num_heads * head_dim,
            device=device,
            dtype=torch.float16,
        ).reshape(num_slots, num_heads, head_dim)
        for layer_idx in range(num_layers)
    ]

    k_ptr_dst = torch.tensor(
        [layer.data_ptr() for layer in k_layers_dst], device=device, dtype=torch.uint64
    )
    v_ptr_dst = torch.tensor(
        [layer.data_ptr() for layer in v_layers_dst], device=device, dtype=torch.uint64
    )
    k_ptr_src = torch.tensor(
        [layer.data_ptr() for layer in k_layers_src], device=device, dtype=torch.uint64
    )
    v_ptr_src = torch.tensor(
        [layer.data_ptr() for layer in v_layers_src], device=device, dtype=torch.uint64
    )
    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)
    slot_stride_bytes = k_layers_dst[0].stride(0) * k_layers_dst[0].element_size()

    expected_k = [layer.clone() for layer in k_layers_dst]
    expected_v = [layer.clone() for layer in v_layers_dst]
    for layer_idx in range(num_layers):
        expected_k[layer_idx][indices_dst.to(torch.int64)] = k_layers_src[layer_idx][
            indices_src.to(torch.int64)
        ]
        expected_v[layer_idx][indices_dst.to(torch.int64)] = v_layers_src[layer_idx][
            indices_src.to(torch.int64)
        ]

    transfer_kv_all_layer(
        src_k_layers=k_ptr_src,
        dst_k_layers=k_ptr_dst,
        src_v_layers=v_ptr_src,
        dst_v_layers=v_ptr_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=slot_stride_bytes,
        num_layers=num_layers,
    )

    torch.cuda.synchronize()

    for layer_idx in range(num_layers):
        assert torch.equal(k_layers_dst[layer_idx], expected_k[layer_idx])
        assert torch.equal(v_layers_dst[layer_idx], expected_v[layer_idx])


def test_transfer_kv_per_layer_mla(device: str) -> None:
    num_slots = 6
    kv_cache_dim = 576

    cache_dst = torch.zeros(
        num_slots, 1, kv_cache_dim, device=device, dtype=torch.float16
    )
    cache_src = torch.arange(
        num_slots * kv_cache_dim,
        device=device,
        dtype=torch.float16,
    ).reshape(num_slots, 1, kv_cache_dim)
    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)

    expected = cache_dst.clone()
    expected[indices_dst.to(torch.int64)] = cache_src[indices_src.to(torch.int64)]

    transfer_kv_per_layer_mla(
        src=cache_src,
        dst=cache_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=kv_cache_dim * cache_src.element_size(),
    )

    torch.cuda.synchronize()

    assert torch.equal(cache_dst, expected)


def test_transfer_kv_all_layer_mla(device: str) -> None:
    num_layers = 3
    num_slots = 6
    kv_cache_dim = 576

    layers_dst = [
        torch.zeros(num_slots, 1, kv_cache_dim, device=device, dtype=torch.float16)
        for _ in range(num_layers)
    ]
    layers_src = [
        torch.arange(
            layer_idx * num_slots * kv_cache_dim,
            (layer_idx + 1) * num_slots * kv_cache_dim,
            device=device,
            dtype=torch.float16,
        ).reshape(num_slots, 1, kv_cache_dim)
        for layer_idx in range(num_layers)
    ]
    ptr_dst = torch.tensor(
        [layer.data_ptr() for layer in layers_dst], device=device, dtype=torch.uint64
    )
    ptr_src = torch.tensor(
        [layer.data_ptr() for layer in layers_src], device=device, dtype=torch.uint64
    )
    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)
    slot_stride_bytes = layers_dst[0].stride(0) * layers_dst[0].element_size()

    expected = [layer.clone() for layer in layers_dst]
    for layer_idx in range(num_layers):
        expected[layer_idx][indices_dst.to(torch.int64)] = layers_src[layer_idx][
            indices_src.to(torch.int64)
        ]

    transfer_kv_all_layer_mla(
        src_layers=ptr_src,
        dst_layers=ptr_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=slot_stride_bytes,
        num_layers=num_layers,
    )

    torch.cuda.synchronize()

    for layer_idx in range(num_layers):
        assert torch.equal(layers_dst[layer_idx], expected[layer_idx])
