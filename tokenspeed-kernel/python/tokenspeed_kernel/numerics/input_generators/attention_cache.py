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

"""Attention KV-cache input generators for numerical correctness tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from tokenspeed_kernel.numerics.input_generators.attention_metadata import (
    PageTableIndexing,
)
from tokenspeed_kernel.numerics.input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
    _resolve_device,
)

__all__ = [
    "AttentionCacheInput",
    "KVCacheInput",
    "KVCacheInputConfig",
    "KVCacheLayout",
    "KVCacheValues",
    "MLAKVCacheInput",
    "MLAKVCacheInputConfig",
    "MLAKVCacheValues",
    "PageTableInput",
    "PageTableInputConfig",
    "PageTableValues",
]

KVCacheLayout = Literal["dense", "paged"]


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _check_cache_layout(value: str) -> KVCacheLayout:
    if value not in ("dense", "paged"):
        raise ValueError(f"cache_layout must be 'dense' or 'paged', got {value!r}")
    return cast(KVCacheLayout, value)


def _check_page_table_indexing(value: str) -> PageTableIndexing:
    if value not in ("identity", "random"):
        raise ValueError(
            f"page_table_indexing must be 'identity' or 'random', got {value!r}"
        )
    return cast(PageTableIndexing, value)


@dataclass
class PageTableValues:
    """Generated values for ``PageTableInput``."""

    page_table: torch.Tensor
    page_table_cpu: list[list[int]]
    num_pages: int

    def page_ids(self) -> list[int]:
        """Return all physical page ids in row-major table order."""

        return [page_id for row in self.page_table_cpu for page_id in row]


@dataclass
class PageTableInputConfig:
    """Initialization parameters for ``PageTableInput``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of request rows in the page table.
    batch_size: int

    # Required: padded page-table width, in logical pages per request.
    max_pages_per_request: int

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: physical-page assignment policy. Examples: random models
    # non-contiguous cache allocation; identity models row-major page ids.
    indexing: PageTableIndexing = "random"

    # Optional: default generation device.
    device: DeviceLike = None


@dataclass(init=False)
class PageTableInput(NumericsInputGenerator):
    """Generator for paged KV-cache page-table metadata.

    The generated table is fully initialized with valid physical page ids.
    Parent cache metadata determines which columns are active for each request
    from ``cache_seqlens``; inactive columns are valid poison entries that
    should be ignored by correct kernels.
    """

    config: PageTableInputConfig

    def __init__(
        self,
        config: PageTableInputConfig,
    ) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.max_pages_per_request = _check_positive(
            "max_pages_per_request",
            self.config.max_pages_per_request,
        )
        self.config.indexing = _check_page_table_indexing(self.config.indexing)

    def generate(self, *, seed: int, device: DeviceLike = None) -> PageTableValues:
        target_device = _resolve_device(self.config.device, device)
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.max_pages_per_request = _check_positive(
            "max_pages_per_request",
            self.config.max_pages_per_request,
        )
        self.config.indexing = _check_page_table_indexing(self.config.indexing)
        num_pages = self.config.batch_size * self.config.max_pages_per_request
        if self.config.indexing == "random":
            page_table_cpu = self._make_random_page_table(seed, num_pages)
        else:
            page_table_cpu = self._make_identity_page_table(num_pages)

        return PageTableValues(
            page_table=torch.tensor(
                page_table_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            page_table_cpu=page_table_cpu,
            num_pages=num_pages,
        )

    def _make_identity_page_table(self, num_pages: int) -> list[list[int]]:
        physical_pages = list(range(num_pages))
        return self._assign_physical_pages(physical_pages)

    def _make_random_page_table(self, seed: int, num_pages: int) -> list[list[int]]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        physical_pages = torch.randperm(num_pages, generator=generator).tolist()
        return self._assign_physical_pages([int(page) for page in physical_pages])

    def _assign_physical_pages(self, physical_pages: list[int]) -> list[list[int]]:
        page_table = [
            [0] * self.config.max_pages_per_request
            for _ in range(self.config.batch_size)
        ]
        cursor = 0
        for batch_idx in range(self.config.batch_size):
            assigned = physical_pages[
                cursor : cursor + self.config.max_pages_per_request
            ]
            cursor += self.config.max_pages_per_request
            page_table[batch_idx] = assigned
        return page_table


class AttentionCacheInput(NumericsInputGenerator):
    """Shared dense/paged cache layout logic for attention cache generators.

    Subclasses own the tensors stored inside each cache slot. This base owns
    the common cache-layout controls: dense vs paged shape construction, page
    table generation, and page-table metadata returned with generated values.
    """

    page_table_input: PageTableInput | None

    def _normalize_common_config(self) -> None:
        self.config.cache_layout = _check_cache_layout(self.config.cache_layout)
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.max_seqlen_k = _check_nonnegative(
            "max_seqlen_k", self.config.max_seqlen_k
        )
        if self.config.cache_layout == "paged":
            if self.config.page_size is None:
                raise ValueError("cache_layout='paged' requires page_size")
            self.config.page_size = _check_positive("page_size", self.config.page_size)
        elif self.config.page_size is not None:
            self.config.page_size = _check_positive("page_size", self.config.page_size)

    def _make_page_table_config(self) -> PageTableInputConfig:
        return PageTableInputConfig(
            batch_size=self.config.batch_size,
            max_pages_per_request=1,
            device=self.config.device,
        )

    def _make_cache_shape(
        self,
        seed: int,
        device: torch.device,
        *,
        inner_shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], PageTableValues | None, int, int]:
        if self.config.cache_layout == "dense":
            return (
                (
                    self.config.batch_size,
                    self.config.max_seqlen_k,
                    *inner_shape,
                ),
                None,
                0,
                0,
            )

        if self.config.page_size is None:
            raise ValueError("paged cache requires page_size")
        if self.page_table_input is None:
            self.page_table_input = PageTableInput(self._make_page_table_config())
        max_pages_per_request = max(
            1,
            math.ceil(self.config.max_seqlen_k / self.config.page_size),
        )
        self.page_table_input.config.batch_size = self.config.batch_size
        self.page_table_input.config.max_pages_per_request = max_pages_per_request
        page_table_values = self.page_table_input.generate(seed=seed, device=device)
        return (
            (
                page_table_values.num_pages,
                self.config.page_size,
                *inner_shape,
            ),
            page_table_values,
            max_pages_per_request,
            page_table_values.num_pages,
        )

    def _common_values(
        self,
        *,
        cache_shape: tuple[int, ...],
        page_table_values: PageTableValues | None,
        max_pages_per_request: int,
        num_pages: int,
    ) -> dict[str, object]:
        return {
            "page_table": (
                None if page_table_values is None else page_table_values.page_table
            ),
            "page_table_cpu": (
                None if page_table_values is None else page_table_values.page_table_cpu
            ),
            "cache_shape": cache_shape,
            "max_pages_per_request": max_pages_per_request,
            "num_pages": num_pages,
            "page_table_values": page_table_values,
        }


@dataclass
class KVCacheValues:
    """Generated values for ``KVCacheInput``."""

    k_cache: torch.Tensor | None
    v_cache: torch.Tensor | None
    page_table: torch.Tensor | None
    page_table_cpu: list[list[int]] | None
    cache_shape: tuple[int, ...]
    max_pages_per_request: int
    num_pages: int
    page_table_values: PageTableValues | None = None


@dataclass
class KVCacheInputConfig:
    """Initialization parameters for ``KVCacheInput``."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: cache representation to generate.
    cache_layout: KVCacheLayout

    # Required: number of request rows represented by the cache.
    batch_size: int

    # Required: maximum visible KV tokens per request.
    max_seqlen_k: int

    # Required: KV head count.
    num_kv_heads: int

    # Required: per-head K/V dimension.
    head_dim: int

    # Required: dtype for generated cache storage.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional cache/page configuration.
    # ------------------------------------------------------------------

    # Optional but required for paged cache: number of tokens per physical page.
    page_size: int | None = None

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: default generation device.
    device: DeviceLike = None

    # ------------------------------------------------------------------
    # Optional child generator configuration.
    # ------------------------------------------------------------------

    # Optional: child generator for paged-cache page tables. Defaults to None
    # for dense cache and to a nested PageTableInput for paged cache.
    page_table_input: PageTableInputConfig | None = None


@dataclass(init=False)
class KVCacheInput(AttentionCacheInput):
    """Generator for dense or paged MHA KV-cache storage.

    ``cache_layout`` controls the physical cache representation. Dense caches
    produce batch-major ``k_cache`` and ``v_cache`` tensors. Paged caches also
    generate a nested ``PageTableInput`` and use its full valid page-id
    permutation to size the physical cache.
    """

    config: KVCacheInputConfig
    k_cache_input: TensorInput | None
    v_cache_input: TensorInput | None
    page_table_input: PageTableInput | None

    def __init__(
        self,
        config: KVCacheInputConfig,
    ) -> None:
        self.config = config
        self.k_cache_input = None
        self.v_cache_input = None
        self.page_table_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.k_cache_input = self.k_cache_input or TensorInput(
            (0, self.config.num_kv_heads, self.config.head_dim),
            self.config.dtype,
            device=self.config.device,
        )
        self.v_cache_input = self.v_cache_input or TensorInput(
            (0, self.config.num_kv_heads, self.config.head_dim),
            self.config.dtype,
            device=self.config.device,
        )
        if self.config.cache_layout == "paged" and self.page_table_input is None:
            self.page_table_input = PageTableInput(
                self.config.page_table_input or self._make_page_table_config()
            )
        self.config.page_table_input = (
            None if self.page_table_input is None else self.page_table_input.config
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
        page_table_seed: int | None = None,
    ) -> KVCacheValues:
        self._normalize_config()
        target_device = _resolve_device(self.config.device, device)
        if page_table_seed is None:
            page_table_seed = seed
        (
            cache_shape,
            page_table_values,
            max_pages_per_request,
            num_pages,
        ) = self._make_cache_shape(
            page_table_seed,
            target_device,
            inner_shape=(self.config.num_kv_heads, self.config.head_dim),
        )
        if self.k_cache_input is None or self.v_cache_input is None:
            raise ValueError("KVCacheInput child tensor generators must be initialized")
        self.k_cache_input.shape = cache_shape
        self.v_cache_input.shape = cache_shape
        self.k_cache_input.dtype = self.config.dtype
        self.v_cache_input.dtype = self.config.dtype
        k_cache = self.k_cache_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        ).values
        v_cache = self.v_cache_input.generate(
            seed=_child_seed(seed, 2),
            device=target_device,
        ).values
        return KVCacheValues(
            k_cache=k_cache,
            v_cache=v_cache,
            **self._common_values(
                cache_shape=cache_shape,
                page_table_values=page_table_values,
                max_pages_per_request=max_pages_per_request,
                num_pages=num_pages,
            ),
        )

    def _normalize_config(self) -> None:
        self._normalize_common_config()
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)


@dataclass
class MLAKVCacheValues:
    """Generated values for ``MLAKVCacheInput``."""

    kv_cache: torch.Tensor
    page_table: torch.Tensor | None
    page_table_cpu: list[list[int]] | None
    cache_shape: tuple[int, ...]
    max_pages_per_request: int
    num_pages: int
    page_table_values: PageTableValues | None = None


@dataclass
class MLAKVCacheInputConfig:
    """Initialization parameters for compressed MLA KV-cache storage."""

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: cache representation to generate. TokenSpeed MLA decode uses
    # paged cache; dense is accepted for value-generation tests.
    cache_layout: KVCacheLayout

    # Required: number of request rows represented by the cache.
    batch_size: int

    # Required: maximum visible KV tokens per request.
    max_seqlen_k: int

    # Required: MLA latent KV rank stored in each compressed cache row.
    kv_lora_rank: int

    # Required: RoPE key dimension stored next to the latent KV row.
    qk_rope_head_dim: int

    # Required: dtype for generated compressed cache storage.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional cache/page configuration.
    # ------------------------------------------------------------------

    # Optional but required for paged cache: number of tokens per physical page.
    page_size: int | None = None

    # ------------------------------------------------------------------
    # Optional device configuration.
    # ------------------------------------------------------------------

    # Optional: default generation device.
    device: DeviceLike = None

    # ------------------------------------------------------------------
    # Optional child generator configuration.
    # ------------------------------------------------------------------

    # Optional: child generator for paged-cache page tables. Defaults to None
    # for dense cache and to a nested PageTableInput for paged cache.
    page_table_input: PageTableInputConfig | None = None


@dataclass(init=False)
class MLAKVCacheInput(AttentionCacheInput):
    """Generator for dense or paged compressed MLA KV-cache storage.

    MLA decode stores one compressed cache tensor instead of separate K and V
    caches. Each cache row has shape ``[1, kv_lora_rank + qk_rope_head_dim]``:
    one shared KV head containing latent KV values followed by the RoPE key
    part.
    """

    config: MLAKVCacheInputConfig
    kv_cache_input: TensorInput | None
    page_table_input: PageTableInput | None

    def __init__(
        self,
        config: MLAKVCacheInputConfig,
    ) -> None:
        self.config = config
        self.kv_cache_input = None
        self.page_table_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.kv_cache_input = self.kv_cache_input or TensorInput(
            (0, 1, self._cache_head_dim()),
            self.config.dtype,
            device=self.config.device,
        )
        if self.config.cache_layout == "paged" and self.page_table_input is None:
            self.page_table_input = PageTableInput(
                self.config.page_table_input or self._make_page_table_config()
            )
        self.config.page_table_input = (
            None if self.page_table_input is None else self.page_table_input.config
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
        page_table_seed: int | None = None,
    ) -> MLAKVCacheValues:
        self._normalize_config()
        target_device = _resolve_device(self.config.device, device)
        if page_table_seed is None:
            page_table_seed = seed
        (
            cache_shape,
            page_table_values,
            max_pages_per_request,
            num_pages,
        ) = self._make_cache_shape(
            page_table_seed,
            target_device,
            inner_shape=(1, self._cache_head_dim()),
        )
        if self.kv_cache_input is None:
            raise ValueError(
                "MLAKVCacheInput child tensor generator must be initialized"
            )
        self.kv_cache_input.shape = cache_shape
        self.kv_cache_input.dtype = self.config.dtype
        kv_cache = self.kv_cache_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        ).values
        if kv_cache is None:
            raise ValueError("MLAKVCacheInput dtype must generate a tensor")
        return MLAKVCacheValues(
            kv_cache=kv_cache,
            **self._common_values(
                cache_shape=cache_shape,
                page_table_values=page_table_values,
                max_pages_per_request=max_pages_per_request,
                num_pages=num_pages,
            ),
        )

    def _normalize_config(self) -> None:
        self._normalize_common_config()
        self.config.kv_lora_rank = _check_positive(
            "kv_lora_rank", self.config.kv_lora_rank
        )
        self.config.qk_rope_head_dim = _check_positive(
            "qk_rope_head_dim", self.config.qk_rope_head_dim
        )

    def _cache_head_dim(self) -> int:
        return self.config.kv_lora_rank + self.config.qk_rope_head_dim
