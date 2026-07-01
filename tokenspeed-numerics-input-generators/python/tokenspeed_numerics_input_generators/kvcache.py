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

"""KV-cache input generators for cache-row transfer operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _resolve_device,
    _rng_for_device,
)

__all__ = [
    "KVCacheStoreInputConfig",
    "KVCacheStoreInputs",
    "KVCacheStoreInputValues",
    "KVCacheTransferInputConfig",
    "KVCacheTransferInputs",
    "KVCacheTransferInputValues",
    "MLAKVCacheTransferInputConfig",
    "MLAKVCacheTransferInputs",
    "MLAKVCacheTransferInputValues",
    "PageTableGatherInputConfig",
    "PageTableGatherInputs",
    "PageTableGatherInputValues",
    "kv_cache_store_reference",
    "kv_cache_transfer_reference",
    "mla_kv_cache_transfer_reference",
    "page_table_gather_reference",
]

_REGULAR_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _check_nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_float_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _REGULAR_FLOAT_DTYPES:
        raise ValueError(f"{name} must be a regular floating torch dtype, got {dtype}")
    return dtype


def _check_index_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be torch.int32 or torch.int64, got {dtype}")
    return dtype


def _require_tensor(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} generation unexpectedly returned None")
    return values


def _generate_unique_indices(
    *,
    num_slots: int,
    num_transfers: int,
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    if num_transfers == 0:
        return torch.empty(0, dtype=dtype, device=target_device)
    generator = _rng_for_device(target_device, seed)
    indices = torch.randperm(num_slots, device=target_device, generator=generator)
    return indices[:num_transfers].to(dtype)


def _generate_index_matrix(
    *,
    rows: int,
    cols: int,
    high: int,
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    generator = _rng_for_device(target_device, seed)
    return torch.randint(
        0,
        high,
        (rows, cols),
        dtype=dtype,
        device=target_device,
        generator=generator,
    )


def _randint_seq_lens(
    *,
    batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    if batch_size == 0:
        return torch.empty(0, dtype=dtype, device=target_device)
    generator = _rng_for_device(target_device, seed)
    return torch.randint(
        0,
        max_seq_len + 1,
        (batch_size,),
        dtype=dtype,
        device=target_device,
        generator=generator,
    )


def _generate_tensor_layers(
    inputs: list[TensorInput],
    *,
    seed: int,
    device: DeviceLike,
    name: str,
) -> list[torch.Tensor]:
    layers: list[torch.Tensor] = []
    for layer_idx, tensor_input in enumerate(inputs):
        layers.append(
            _require_tensor(
                tensor_input.generate(
                    seed=_child_seed(seed, layer_idx + 1),
                    device=device,
                ).values,
                f"{name}[{layer_idx}]",
            ).contiguous()
        )
    return layers


@dataclass
class KVCacheStoreInputValues:
    """Generated values for K/V cache store/scatter operations."""

    k_src: torch.Tensor
    v_src: torch.Tensor
    k_dst: torch.Tensor
    v_dst: torch.Tensor
    loc: torch.Tensor


@dataclass
class KVCacheStoreInputConfig:
    """Initialization parameters for per-token K/V cache store inputs.

    The represented operation scatters generated token K/V rows into cache
    slots: ``k_dst[loc[i]] = k_src[i]`` and ``v_dst[loc[i]] = v_src[i]``.
    Destination locations are generated without duplicates so the result is
    independent of write order.
    """

    # Required: number of token rows to store.
    num_tokens: int

    # Required: number of destination cache slots.
    num_slots: int

    # Required: number of KV heads in each token/cache slot.
    num_kv_heads: int

    # Required: per-head dimension.
    head_dim: int

    # Required: generated dtype for source and destination caches.
    dtype: torch.dtype

    # Optional: dtype for generated destination locations.
    index_dtype: torch.dtype = torch.int32

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class KVCacheStoreInputs(NumericsInputGenerator):
    """Generator for per-token K/V cache store/scatter inputs."""

    config: KVCacheStoreInputConfig
    k_src_input: TensorInput | None
    v_src_input: TensorInput | None
    k_dst_input: TensorInput | None
    v_dst_input: TensorInput | None

    def __init__(self, config: KVCacheStoreInputConfig) -> None:
        self.config = config
        self.k_src_input = None
        self.v_src_input = None
        self.k_dst_input = None
        self.v_dst_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_slots = _check_positive("num_slots", self.config.num_slots)
        if self.config.num_tokens > self.config.num_slots:
            raise ValueError(
                "num_tokens must be <= num_slots for unique cache locations; "
                f"got num_tokens={self.config.num_tokens}, "
                f"num_slots={self.config.num_slots}"
            )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.index_dtype = _check_index_dtype(
            "index_dtype", self.config.index_dtype
        )
        src_shape = (
            self.config.num_tokens,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        dst_shape = (
            self.config.num_slots,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        self.k_src_input = self.k_src_input or TensorInput(
            src_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.v_src_input = self.v_src_input or TensorInput(
            src_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.k_dst_input = self.k_dst_input or TensorInput(
            dst_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.v_dst_input = self.v_dst_input or TensorInput(
            dst_shape,
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> KVCacheStoreInputValues:
        self.__post_init__()
        if (
            self.k_src_input is None
            or self.v_src_input is None
            or self.k_dst_input is None
            or self.v_dst_input is None
        ):
            raise ValueError("KVCacheStoreInputs child generators must be initialized")
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        loc = _generate_unique_indices(
            num_slots=self.config.num_slots,
            num_transfers=self.config.num_tokens,
            dtype=self.config.index_dtype,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        return KVCacheStoreInputValues(
            k_src=_require_tensor(
                self.k_src_input.generate(
                    seed=_child_seed(seed, 1), device=device
                ).values,
                "k_src",
            ).contiguous(),
            v_src=_require_tensor(
                self.v_src_input.generate(
                    seed=_child_seed(seed, 2), device=device
                ).values,
                "v_src",
            ).contiguous(),
            k_dst=_require_tensor(
                self.k_dst_input.generate(
                    seed=_child_seed(seed, 3), device=device
                ).values,
                "k_dst",
            ).contiguous(),
            v_dst=_require_tensor(
                self.v_dst_input.generate(
                    seed=_child_seed(seed, 4), device=device
                ).values,
                "v_dst",
            ).contiguous(),
            loc=loc.contiguous(),
        )


@dataclass
class PageTableGatherInputValues:
    """Generated values for page-table gather-with-padding operations."""

    req_to_page: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out: torch.Tensor


@dataclass
class PageTableGatherInputConfig:
    """Initialization parameters for active page-table gather inputs.

    The represented operation gathers selected request rows from a source page
    table and writes dummy slots after the number of valid pages implied by
    each request length.
    """

    # Required: number of rows in the source request-to-page table.
    source_rows: int

    # Required: number of active request rows to gather.
    batch_size: int

    # Required: number of page-table columns in source and destination.
    max_num_pages: int

    # Required: number of cache tokens represented by one page.
    page_size: int

    # Optional: dummy page id written into padding columns.
    dummy_slot: int = 0

    # Optional: dtype for request/page metadata tensors.
    index_dtype: torch.dtype = torch.int32

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class PageTableGatherInputs(NumericsInputGenerator):
    """Generator for page-table gather-with-padding inputs."""

    config: PageTableGatherInputConfig

    def __init__(self, config: PageTableGatherInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.source_rows = _check_positive(
            "source_rows", self.config.source_rows
        )
        self.config.batch_size = _check_nonnegative(
            "batch_size", self.config.batch_size
        )
        if self.config.batch_size > self.config.source_rows:
            raise ValueError(
                "batch_size must be <= source_rows for unique request rows; "
                f"got batch_size={self.config.batch_size}, "
                f"source_rows={self.config.source_rows}"
            )
        self.config.max_num_pages = _check_positive(
            "max_num_pages", self.config.max_num_pages
        )
        self.config.page_size = _check_positive("page_size", self.config.page_size)
        self.config.index_dtype = _check_index_dtype(
            "index_dtype", self.config.index_dtype
        )
        if self.config.dummy_slot < 0:
            raise ValueError(
                f"dummy_slot must be non-negative, got {self.config.dummy_slot}"
            )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> PageTableGatherInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        req_to_page = _generate_index_matrix(
            rows=self.config.source_rows,
            cols=self.config.max_num_pages,
            high=max(self.config.source_rows * self.config.max_num_pages, 1),
            dtype=self.config.index_dtype,
            seed=_child_seed(seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        req_pool_indices = _generate_unique_indices(
            num_slots=self.config.source_rows,
            num_transfers=self.config.batch_size,
            dtype=self.config.index_dtype,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        max_seq_len = self.config.max_num_pages * self.config.page_size
        seq_lens = _randint_seq_lens(
            batch_size=self.config.batch_size,
            max_seq_len=max_seq_len,
            dtype=self.config.index_dtype,
            seed=_child_seed(metadata_base_seed, 2),
            device=device,
            configured_device=self.config.device,
        )
        out = torch.full(
            (self.config.batch_size, self.config.max_num_pages),
            self.config.dummy_slot,
            dtype=self.config.index_dtype,
            device=target_device,
        )
        return PageTableGatherInputValues(
            req_to_page=req_to_page.contiguous(),
            req_pool_indices=req_pool_indices.contiguous(),
            seq_lens=seq_lens.contiguous(),
            out=out,
        )


@dataclass
class KVCacheTransferInputValues:
    """Generated values for K/V cache row transfer operations."""

    src_k_layers: list[torch.Tensor]
    dst_k_layers: list[torch.Tensor]
    src_v_layers: list[torch.Tensor]
    dst_v_layers: list[torch.Tensor]
    src_indices: torch.Tensor
    dst_indices: torch.Tensor


@dataclass
class KVCacheTransferInputConfig:
    """Initialization parameters for K/V cache row transfer inputs.

    The represented operation copies selected source cache slots to selected
    destination cache slots for each K/V layer:
    ``dst_k[layer][dst_indices[i]] = src_k[layer][src_indices[i]]`` and the
    same for V. Destination indices are generated without duplicates so the
    result is mathematically well-defined.
    """

    # Required: number of independent cache layers.
    num_layers: int

    # Required: number of slot rows in each source and destination cache.
    num_slots: int

    # Required: number of row transfers to generate.
    num_transfers: int

    # Required: number of KV heads stored in each slot.
    num_kv_heads: int

    # Required: per-head dimension stored in each slot.
    head_dim: int

    # Required: generated dtype for source and destination caches.
    dtype: torch.dtype

    # Optional: dtype for generated source/destination indices.
    index_dtype: torch.dtype = torch.int32

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class KVCacheTransferInputs(NumericsInputGenerator):
    """Generator for K/V cache row transfer inputs."""

    config: KVCacheTransferInputConfig
    src_k_inputs: list[TensorInput]
    dst_k_inputs: list[TensorInput]
    src_v_inputs: list[TensorInput]
    dst_v_inputs: list[TensorInput]

    def __init__(self, config: KVCacheTransferInputConfig) -> None:
        self.config = config
        self.src_k_inputs = []
        self.dst_k_inputs = []
        self.src_v_inputs = []
        self.dst_v_inputs = []
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_layers = _check_positive("num_layers", self.config.num_layers)
        self.config.num_slots = _check_positive("num_slots", self.config.num_slots)
        self.config.num_transfers = _check_nonnegative(
            "num_transfers", self.config.num_transfers
        )
        if self.config.num_transfers > self.config.num_slots:
            raise ValueError(
                "num_transfers must be <= num_slots for unique transfer indices; "
                f"got num_transfers={self.config.num_transfers}, "
                f"num_slots={self.config.num_slots}"
            )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.index_dtype = _check_index_dtype(
            "index_dtype", self.config.index_dtype
        )
        shape = (
            self.config.num_slots,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        if not self.src_k_inputs:
            self.src_k_inputs = [
                TensorInput(shape, self.config.dtype, device=self.config.device)
                for _ in range(self.config.num_layers)
            ]
        if not self.dst_k_inputs:
            self.dst_k_inputs = [
                TensorInput(shape, self.config.dtype, device=self.config.device)
                for _ in range(self.config.num_layers)
            ]
        if not self.src_v_inputs:
            self.src_v_inputs = [
                TensorInput(shape, self.config.dtype, device=self.config.device)
                for _ in range(self.config.num_layers)
            ]
        if not self.dst_v_inputs:
            self.dst_v_inputs = [
                TensorInput(shape, self.config.dtype, device=self.config.device)
                for _ in range(self.config.num_layers)
            ]

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> KVCacheTransferInputValues:
        self.__post_init__()
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        src_indices = _generate_unique_indices(
            num_slots=self.config.num_slots,
            num_transfers=self.config.num_transfers,
            dtype=self.config.index_dtype,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        dst_indices = _generate_unique_indices(
            num_slots=self.config.num_slots,
            num_transfers=self.config.num_transfers,
            dtype=self.config.index_dtype,
            seed=_child_seed(metadata_base_seed, 2),
            device=device,
            configured_device=self.config.device,
        )
        return KVCacheTransferInputValues(
            src_k_layers=_generate_tensor_layers(
                self.src_k_inputs,
                seed=_child_seed(seed, 1),
                device=device,
                name="src_k_layers",
            ),
            dst_k_layers=_generate_tensor_layers(
                self.dst_k_inputs,
                seed=_child_seed(seed, 2),
                device=device,
                name="dst_k_layers",
            ),
            src_v_layers=_generate_tensor_layers(
                self.src_v_inputs,
                seed=_child_seed(seed, 3),
                device=device,
                name="src_v_layers",
            ),
            dst_v_layers=_generate_tensor_layers(
                self.dst_v_inputs,
                seed=_child_seed(seed, 4),
                device=device,
                name="dst_v_layers",
            ),
            src_indices=src_indices.contiguous(),
            dst_indices=dst_indices.contiguous(),
        )


@dataclass
class MLAKVCacheTransferInputValues:
    """Generated values for MLA cache row transfer operations."""

    src_layers: list[torch.Tensor]
    dst_layers: list[torch.Tensor]
    src_indices: torch.Tensor
    dst_indices: torch.Tensor


@dataclass
class MLAKVCacheTransferInputConfig:
    """Initialization parameters for MLA cache row transfer inputs.

    MLA stores one compressed cache tensor per layer rather than separate K and
    V tensors. The represented operation copies selected source rows to selected
    destination rows for every layer.
    """

    # Required: number of independent cache layers.
    num_layers: int

    # Required: number of slot rows in each source and destination cache.
    num_slots: int

    # Required: number of row transfers to generate.
    num_transfers: int

    # Required: flattened compressed cache width.
    kv_cache_dim: int

    # Required: generated dtype for source and destination caches.
    dtype: torch.dtype

    # Optional: dtype for generated source/destination indices.
    index_dtype: torch.dtype = torch.int32

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MLAKVCacheTransferInputs(NumericsInputGenerator):
    """Generator for MLA cache row transfer inputs."""

    config: MLAKVCacheTransferInputConfig
    src_inputs: list[TensorInput]
    dst_inputs: list[TensorInput]

    def __init__(self, config: MLAKVCacheTransferInputConfig) -> None:
        self.config = config
        self.src_inputs = []
        self.dst_inputs = []
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_layers = _check_positive("num_layers", self.config.num_layers)
        self.config.num_slots = _check_positive("num_slots", self.config.num_slots)
        self.config.num_transfers = _check_nonnegative(
            "num_transfers", self.config.num_transfers
        )
        if self.config.num_transfers > self.config.num_slots:
            raise ValueError(
                "num_transfers must be <= num_slots for unique transfer indices; "
                f"got num_transfers={self.config.num_transfers}, "
                f"num_slots={self.config.num_slots}"
            )
        self.config.kv_cache_dim = _check_positive(
            "kv_cache_dim", self.config.kv_cache_dim
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.index_dtype = _check_index_dtype(
            "index_dtype", self.config.index_dtype
        )
        shape = (self.config.num_slots, 1, self.config.kv_cache_dim)
        if not self.src_inputs:
            self.src_inputs = [
                TensorInput(shape, self.config.dtype, device=self.config.device)
                for _ in range(self.config.num_layers)
            ]
        if not self.dst_inputs:
            self.dst_inputs = [
                TensorInput(shape, self.config.dtype, device=self.config.device)
                for _ in range(self.config.num_layers)
            ]

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MLAKVCacheTransferInputValues:
        self.__post_init__()
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        src_indices = _generate_unique_indices(
            num_slots=self.config.num_slots,
            num_transfers=self.config.num_transfers,
            dtype=self.config.index_dtype,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        dst_indices = _generate_unique_indices(
            num_slots=self.config.num_slots,
            num_transfers=self.config.num_transfers,
            dtype=self.config.index_dtype,
            seed=_child_seed(metadata_base_seed, 2),
            device=device,
            configured_device=self.config.device,
        )
        return MLAKVCacheTransferInputValues(
            src_layers=_generate_tensor_layers(
                self.src_inputs,
                seed=_child_seed(seed, 1),
                device=device,
                name="src_layers",
            ),
            dst_layers=_generate_tensor_layers(
                self.dst_inputs,
                seed=_child_seed(seed, 2),
                device=device,
                name="dst_layers",
            ),
            src_indices=src_indices.contiguous(),
            dst_indices=dst_indices.contiguous(),
        )


def kv_cache_store_reference(
    values: KVCacheStoreInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expected destination K/V caches after per-token store."""

    expected_k = values.k_dst.clone()
    expected_v = values.v_dst.clone()
    loc = values.loc.to(torch.int64)
    expected_k[loc] = values.k_src
    expected_v[loc] = values.v_src
    return expected_k, expected_v


def page_table_gather_reference(
    values: PageTableGatherInputValues,
    *,
    page_size: int,
    dummy_slot: int = 0,
) -> torch.Tensor:
    """Return expected gathered page table with padding columns cleared."""

    page_size = _check_positive("page_size", page_size)
    if dummy_slot < 0:
        raise ValueError(f"dummy_slot must be non-negative, got {dummy_slot}")
    expected = torch.full_like(values.out, dummy_slot)
    max_num_pages = values.out.shape[1]
    for row_idx in range(values.req_pool_indices.numel()):
        seq_len = int(values.seq_lens[row_idx].item())
        n_pages = (seq_len + page_size - 1) // page_size
        if n_pages > max_num_pages:
            raise ValueError(
                "seq_lens imply more pages than out can hold; "
                f"row={row_idx}, seq_len={seq_len}, page_size={page_size}, "
                f"out_columns={max_num_pages}"
            )
        req_idx = int(values.req_pool_indices[row_idx].item())
        expected[row_idx, :n_pages] = values.req_to_page[req_idx, :n_pages]
    return expected


def kv_cache_transfer_reference(
    values: KVCacheTransferInputValues,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return expected destination K/V layers after cache-row transfer."""

    expected_k = [layer.clone() for layer in values.dst_k_layers]
    expected_v = [layer.clone() for layer in values.dst_v_layers]
    src_indices = values.src_indices.to(torch.int64)
    dst_indices = values.dst_indices.to(torch.int64)
    for layer_idx in range(len(expected_k)):
        expected_k[layer_idx][dst_indices] = values.src_k_layers[layer_idx][src_indices]
        expected_v[layer_idx][dst_indices] = values.src_v_layers[layer_idx][src_indices]
    return expected_k, expected_v


def mla_kv_cache_transfer_reference(
    values: MLAKVCacheTransferInputValues,
) -> list[torch.Tensor]:
    """Return expected destination MLA layers after cache-row transfer."""

    expected = [layer.clone() for layer in values.dst_layers]
    src_indices = values.src_indices.to(torch.int64)
    dst_indices = values.dst_indices.to(torch.int64)
    for layer_idx in range(len(expected)):
        expected[layer_idx][dst_indices] = values.src_layers[layer_idx][src_indices]
    return expected
