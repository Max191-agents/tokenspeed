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
    "KVCacheTransferInputConfig",
    "KVCacheTransferInputs",
    "KVCacheTransferInputValues",
    "MLAKVCacheTransferInputConfig",
    "MLAKVCacheTransferInputs",
    "MLAKVCacheTransferInputValues",
    "kv_cache_transfer_reference",
    "mla_kv_cache_transfer_reference",
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
