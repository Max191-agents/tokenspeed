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

"""Communication collective input generators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_numerics_input_generators.core import (
    DeviceLike,
    NumericsInputGenerator,
    TensorInput,
    _child_seed,
)

__all__ = [
    "AllGatherInputConfig",
    "AllGatherInputs",
    "AllGatherInputValues",
    "AllReduceInputConfig",
    "AllReduceInputs",
    "AllReduceInputValues",
    "ReduceScatterInputConfig",
    "ReduceScatterInputs",
    "ReduceScatterInputValues",
    "all_gather_reference",
    "all_reduce_sum_reference",
    "reduce_scatter_sum_reference",
]

_REGULAR_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


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


def _check_float_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in _REGULAR_FLOAT_DTYPES:
        raise ValueError(f"{name} must be a regular floating torch dtype, got {dtype}")
    return dtype


def _check_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    normalized = tuple(int(dim) for dim in shape)
    if not normalized:
        raise ValueError("shape must contain at least one dimension")
    if any(dim <= 0 for dim in normalized):
        raise ValueError(f"shape dimensions must be positive, got {shape}")
    return normalized


def _require_tensor(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} generation unexpectedly returned None")
    return values


def _check_max_tokens_per_rank(
    *,
    world_size: int,
    total_tokens: int,
    max_tokens_per_rank: int | None,
) -> int:
    if max_tokens_per_rank is None:
        return total_tokens
    max_tokens_per_rank = _check_nonnegative("max_tokens_per_rank", max_tokens_per_rank)
    if total_tokens > world_size * max_tokens_per_rank:
        raise ValueError(
            "total_tokens cannot fit within max_tokens_per_rank for each rank; "
            f"total_tokens={total_tokens}, world_size={world_size}, "
            f"max_tokens_per_rank={max_tokens_per_rank}"
        )
    return max_tokens_per_rank


def _generate_tokens_per_rank(
    *,
    world_size: int,
    total_tokens: int,
    max_tokens_per_rank: int,
    seed: int,
) -> list[int]:
    if total_tokens == 0:
        return [0 for _ in range(world_size)]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts = [0 for _ in range(world_size)]
    for _ in range(total_tokens):
        candidates = [
            rank for rank, count in enumerate(counts) if count < max_tokens_per_rank
        ]
        candidate_idx = int(
            torch.randint(
                0,
                len(candidates),
                (),
                generator=generator,
            ).item()
        )
        counts[candidates[candidate_idx]] += 1
    return counts


def _generate_tensor(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    return _require_tensor(
        TensorInput(shape, dtype, device=configured_device)
        .generate(seed=seed, device=device)
        .values,
        "communication tensor",
    ).contiguous()


def _sum_rank_inputs(rank_inputs: list[torch.Tensor]) -> torch.Tensor:
    if not rank_inputs:
        raise ValueError("rank_inputs must be non-empty")
    expected_shape = rank_inputs[0].shape
    expected_dtype = rank_inputs[0].dtype
    expected_device = rank_inputs[0].device
    acc_dtype = torch.float64 if expected_dtype == torch.float64 else torch.float32
    acc = torch.zeros_like(rank_inputs[0], dtype=acc_dtype)
    for rank, tensor in enumerate(rank_inputs):
        if tensor.shape != expected_shape:
            raise ValueError(
                "all rank inputs must have the same shape; "
                f"rank=0 shape={tuple(expected_shape)}, "
                f"rank={rank} shape={tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                "all rank inputs must have the same dtype; "
                f"rank=0 dtype={expected_dtype}, rank={rank} dtype={tensor.dtype}"
            )
        if tensor.device != expected_device:
            raise ValueError(
                "all rank inputs must be on the same device; "
                f"rank=0 device={expected_device}, rank={rank} device={tensor.device}"
            )
        acc = acc + tensor.to(acc_dtype)
    return acc.to(expected_dtype)


@dataclass
class AllReduceInputValues:
    """Generated per-rank inputs for a sum all-reduce collective."""

    rank_inputs: list[torch.Tensor]


@dataclass
class AllReduceInputConfig:
    """Initialization parameters for sum all-reduce inputs.

    All ranks contribute one tensor with the same shape and dtype. The reference
    result is the elementwise sum of every rank's tensor, returned to every
    rank.
    """

    # Required: number of ranks participating in the collective.
    world_size: int

    # Required: shape of each rank-local tensor.
    shape: tuple[int, ...] | list[int]

    # Optional: generated tensor dtype.
    dtype: torch.dtype = torch.bfloat16

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class AllReduceInputs(NumericsInputGenerator):
    """Generator for sum all-reduce collective inputs."""

    config: AllReduceInputConfig
    rank_inputs: list[TensorInput]

    def __init__(self, config: AllReduceInputConfig) -> None:
        self.config = config
        self.rank_inputs = []
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.world_size = _check_positive("world_size", self.config.world_size)
        self.config.shape = _check_shape(self.config.shape)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        if not self.rank_inputs:
            self.rank_inputs = [
                TensorInput(
                    self.config.shape,
                    self.config.dtype,
                    device=self.config.device,
                )
                for _ in range(self.config.world_size)
            ]

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> AllReduceInputValues:
        del metadata_seed
        self.__post_init__()
        if len(self.rank_inputs) != self.config.world_size:
            raise ValueError("rank_inputs must match world_size")
        return AllReduceInputValues(
            rank_inputs=[
                _require_tensor(
                    rank_input.generate(
                        seed=_child_seed(seed, rank + 1),
                        device=device,
                    ).values,
                    f"rank_inputs[{rank}]",
                ).contiguous()
                for rank, rank_input in enumerate(self.rank_inputs)
            ]
        )


@dataclass
class AllGatherInputValues:
    """Generated per-rank inputs for an all-gather collective."""

    rank_inputs: list[torch.Tensor]
    tokens_per_rank: list[int]


@dataclass
class AllGatherInputConfig:
    """Initialization parameters for all-gather inputs.

    Each rank contributes a local token shard with shape
    ``[tokens_per_rank[rank], hidden_size]``. The reference result concatenates
    rank shards in rank order.
    """

    # Required: number of ranks participating in the collective.
    world_size: int

    # Required: total number of tokens after gathering every rank shard.
    total_tokens: int

    # Required: hidden dimension for every token row.
    hidden_size: int

    # Optional: upper bound for generated token count on any one rank.
    max_tokens_per_rank: int | None = None

    # Optional: generated tensor dtype.
    dtype: torch.dtype = torch.bfloat16

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class AllGatherInputs(NumericsInputGenerator):
    """Generator for all-gather collective inputs."""

    config: AllGatherInputConfig

    def __init__(self, config: AllGatherInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.world_size = _check_positive("world_size", self.config.world_size)
        self.config.total_tokens = _check_nonnegative(
            "total_tokens", self.config.total_tokens
        )
        self.config.hidden_size = _check_positive(
            "hidden_size", self.config.hidden_size
        )
        self.config.max_tokens_per_rank = _check_max_tokens_per_rank(
            world_size=self.config.world_size,
            total_tokens=self.config.total_tokens,
            max_tokens_per_rank=self.config.max_tokens_per_rank,
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> AllGatherInputValues:
        self.__post_init__()
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        assert self.config.max_tokens_per_rank is not None
        tokens_per_rank = _generate_tokens_per_rank(
            world_size=self.config.world_size,
            total_tokens=self.config.total_tokens,
            max_tokens_per_rank=self.config.max_tokens_per_rank,
            seed=_child_seed(metadata_base_seed, 1),
        )
        return AllGatherInputValues(
            rank_inputs=[
                _generate_tensor(
                    shape=(num_tokens, self.config.hidden_size),
                    dtype=self.config.dtype,
                    seed=_child_seed(seed, rank + 1),
                    device=device,
                    configured_device=self.config.device,
                )
                for rank, num_tokens in enumerate(tokens_per_rank)
            ],
            tokens_per_rank=tokens_per_rank,
        )


@dataclass
class ReduceScatterInputValues:
    """Generated per-rank inputs for a sum reduce-scatter collective."""

    rank_inputs: list[torch.Tensor]
    tokens_per_rank: list[int]


@dataclass
class ReduceScatterInputConfig:
    """Initialization parameters for sum reduce-scatter inputs.

    Each rank contributes a full tensor with shape
    ``[sum(tokens_per_rank), hidden_size]``. The reference first performs an
    elementwise sum across ranks, then returns each rank's contiguous token
    shard according to ``tokens_per_rank``.
    """

    # Required: number of ranks participating in the collective.
    world_size: int

    # Required: total number of token rows before scattering.
    total_tokens: int

    # Required: hidden dimension for every token row.
    hidden_size: int

    # Optional: upper bound for generated token count on any one rank.
    max_tokens_per_rank: int | None = None

    # Optional: generated tensor dtype.
    dtype: torch.dtype = torch.bfloat16

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class ReduceScatterInputs(NumericsInputGenerator):
    """Generator for sum reduce-scatter collective inputs."""

    config: ReduceScatterInputConfig

    def __init__(self, config: ReduceScatterInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.world_size = _check_positive("world_size", self.config.world_size)
        self.config.total_tokens = _check_nonnegative(
            "total_tokens", self.config.total_tokens
        )
        self.config.hidden_size = _check_positive(
            "hidden_size", self.config.hidden_size
        )
        self.config.max_tokens_per_rank = _check_max_tokens_per_rank(
            world_size=self.config.world_size,
            total_tokens=self.config.total_tokens,
            max_tokens_per_rank=self.config.max_tokens_per_rank,
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> ReduceScatterInputValues:
        self.__post_init__()
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        assert self.config.max_tokens_per_rank is not None
        tokens_per_rank = _generate_tokens_per_rank(
            world_size=self.config.world_size,
            total_tokens=self.config.total_tokens,
            max_tokens_per_rank=self.config.max_tokens_per_rank,
            seed=_child_seed(metadata_base_seed, 1),
        )
        return ReduceScatterInputValues(
            rank_inputs=[
                _generate_tensor(
                    shape=(self.config.total_tokens, self.config.hidden_size),
                    dtype=self.config.dtype,
                    seed=_child_seed(seed, rank + 1),
                    device=device,
                    configured_device=self.config.device,
                )
                for rank in range(self.config.world_size)
            ],
            tokens_per_rank=tokens_per_rank,
        )


def all_reduce_sum_reference(values: AllReduceInputValues) -> torch.Tensor:
    """Return the sum all-reduce result shared by every rank."""

    return _sum_rank_inputs(values.rank_inputs)


def all_gather_reference(values: AllGatherInputValues) -> torch.Tensor:
    """Return the all-gather result formed by concatenating rank shards."""

    if len(values.rank_inputs) != len(values.tokens_per_rank):
        raise ValueError("rank_inputs must match tokens_per_rank")
    for rank, (rank_input, num_tokens) in enumerate(
        zip(values.rank_inputs, values.tokens_per_rank, strict=True)
    ):
        if rank_input.shape[0] != num_tokens:
            raise ValueError(
                "rank input token dimension must match tokens_per_rank; "
                f"rank={rank}, shape={tuple(rank_input.shape)}, "
                f"tokens_per_rank={num_tokens}"
            )
    return torch.cat(values.rank_inputs, dim=0)


def reduce_scatter_sum_reference(
    values: ReduceScatterInputValues,
) -> list[torch.Tensor]:
    """Return the per-rank reduce-scatter outputs after summing inputs."""

    if not values.rank_inputs:
        raise ValueError("rank_inputs must be non-empty")
    total_tokens = sum(values.tokens_per_rank)
    for rank, rank_input in enumerate(values.rank_inputs):
        if rank_input.shape[0] != total_tokens:
            raise ValueError(
                "rank input token dimension must equal sum(tokens_per_rank); "
                f"rank={rank}, shape={tuple(rank_input.shape)}, "
                f"total_tokens={total_tokens}"
            )
    reduced = _sum_rank_inputs(values.rank_inputs)
    outputs: list[torch.Tensor] = []
    offset = 0
    for num_tokens in values.tokens_per_rank:
        outputs.append(reduced[offset : offset + num_tokens].contiguous())
        offset += num_tokens
    return outputs
