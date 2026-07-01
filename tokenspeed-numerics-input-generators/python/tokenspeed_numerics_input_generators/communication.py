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
    _resolve_device,
)

__all__ = [
    "AllGatherInputConfig",
    "AllGatherInputs",
    "AllGatherInputValues",
    "AllReduceResidualRMSNormInputConfig",
    "AllReduceResidualRMSNormInputs",
    "AllReduceResidualRMSNormInputValues",
    "AllReduceResidualRMSNormReferenceValues",
    "AllReduceInputConfig",
    "AllReduceInputs",
    "AllReduceInputValues",
    "ExpertParallelDispatchRecord",
    "ExpertParallelRoutingInputConfig",
    "ExpertParallelRoutingInputs",
    "ExpertParallelRoutingInputValues",
    "ExpertParallelRoutingReferenceValues",
    "ReduceScatterInputConfig",
    "ReduceScatterInputs",
    "ReduceScatterInputValues",
    "all_gather_reference",
    "all_reduce_residual_rmsnorm_reference",
    "all_reduce_sum_reference",
    "expert_parallel_routing_reference",
    "reduce_scatter_sum_reference",
]

_REGULAR_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_TOPK_ID_DTYPES = {torch.int32, torch.int64}


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


def _check_topk_id_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if dtype not in _TOPK_ID_DTYPES:
        raise ValueError(f"{name} must be an integer routing dtype, got {dtype}")
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


def _generate_unique_topk_ids(
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    topk_ids = torch.empty(
        (num_tokens, top_k),
        dtype=dtype,
        device=device,
    )
    for token in range(num_tokens):
        topk_ids[token] = torch.randperm(
            num_experts,
            device=device,
            generator=generator,
        )[:top_k].to(dtype)
    return topk_ids.contiguous()


def _generate_topk_weights(
    *,
    num_tokens: int,
    top_k: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    if num_tokens == 0:
        return torch.empty((0, top_k), dtype=torch.float32, device=device)
    weights = torch.rand(
        (num_tokens, top_k),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    return (weights / weights.sum(dim=-1, keepdim=True)).contiguous()


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


def _sum_rank_inputs_for_rmsnorm(rank_inputs: list[torch.Tensor]) -> torch.Tensor:
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
    return acc


@dataclass
class AllReduceInputValues:
    """Generated per-rank inputs for a sum all-reduce collective."""

    rank_inputs: list[torch.Tensor]


@dataclass
class AllReduceResidualRMSNormInputValues:
    """Generated values for fused all-reduce, residual add, and RMSNorm."""

    rank_inputs: list[torch.Tensor]
    residuals: list[torch.Tensor]
    weight: torch.Tensor
    eps: float


@dataclass
class AllReduceResidualRMSNormReferenceValues:
    """Reference outputs for ``AllReduceResidualRMSNormInputs``."""

    norm_outputs: list[torch.Tensor]
    residual_outputs: list[torch.Tensor]


@dataclass(frozen=True)
class ExpertParallelDispatchRecord:
    """One token sent to one expert-owning rank during EP dispatch."""

    source_rank: int
    source_token: int
    target_rank: int
    topk_slots: tuple[int, ...]
    expert_ids: tuple[int, ...]


@dataclass
class ExpertParallelRoutingInputValues:
    """Generated values for expert-parallel MoE dispatch/combine routing."""

    rank_hidden_states: list[torch.Tensor]
    topk_ids: list[torch.Tensor]
    topk_weights: list[torch.Tensor]
    expert_outputs: list[torch.Tensor]
    tokens_per_rank: list[int]
    num_experts: int


@dataclass
class ExpertParallelRoutingReferenceValues:
    """Reference dispatch and combine values for expert-parallel routing."""

    dispatch_records: list[list[ExpertParallelDispatchRecord]]
    recv_hidden_states: list[torch.Tensor]
    recv_topk_ids: list[torch.Tensor]
    recv_topk_weights: list[torch.Tensor]
    recv_expert_outputs: list[torch.Tensor]
    num_recv_tokens_per_expert: list[torch.Tensor]
    combined_outputs: list[torch.Tensor]


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


@dataclass
class AllReduceResidualRMSNormInputConfig:
    """Initialization parameters for fused all-reduce + residual RMSNorm.

    Each rank contributes an input tensor. The inputs are summed across ranks,
    then each rank adds its local residual tensor and applies RMSNorm with a
    shared weight vector:

    ``residual_out[rank] = sum(input[peer]) + residual[rank]``
    ``norm_out[rank] = rmsnorm(residual_out[rank], weight, eps)``
    """

    # Required: number of ranks participating in the all-reduce.
    world_size: int

    # Required: number of token rows in each rank-local tensor.
    num_tokens: int

    # Required: hidden dimension normalized independently for each token row.
    hidden_size: int

    # Optional: generated all-reduce input dtype.
    dtype: torch.dtype = torch.bfloat16

    # Optional: generated residual dtype. ``None`` uses ``dtype``.
    residual_dtype: torch.dtype | None = None

    # Optional: generated RMSNorm weight dtype.
    weight_dtype: torch.dtype = torch.float32

    # Optional: RMSNorm epsilon.
    eps: float = 1e-6

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class ExpertParallelRoutingInputConfig:
    """Initialization parameters for expert-parallel MoE routing inputs.

    The represented operation routes each rank-local token to the rank(s) that
    own its selected top-k experts. Combine then returns weighted expert
    outputs to the token's original rank.
    """

    # Required: number of expert-parallel ranks.
    world_size: int

    # Required: total tokens across all ranks before dispatch.
    total_tokens: int

    # Required: hidden-state width for token activations and expert outputs.
    hidden_size: int

    # Required: total number of routed experts. Experts are partitioned
    # contiguously and evenly across ranks.
    num_experts: int

    # Required: number of selected experts per token.
    top_k: int

    # Optional: upper bound for generated token count on any one rank.
    max_tokens_per_rank: int | None = None

    # Optional: generated hidden-state and expert-output dtype.
    dtype: torch.dtype = torch.bfloat16

    # Optional: generated top-k id dtype.
    topk_ids_dtype: torch.dtype = torch.int64

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


@dataclass(init=False)
class AllReduceResidualRMSNormInputs(NumericsInputGenerator):
    """Generator for fused all-reduce, residual add, and RMSNorm inputs."""

    config: AllReduceResidualRMSNormInputConfig

    def __init__(self, config: AllReduceResidualRMSNormInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.world_size = _check_positive("world_size", self.config.world_size)
        self.config.num_tokens = _check_positive("num_tokens", self.config.num_tokens)
        self.config.hidden_size = _check_positive(
            "hidden_size", self.config.hidden_size
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        if self.config.residual_dtype is None:
            self.config.residual_dtype = self.config.dtype
        self.config.residual_dtype = _check_float_dtype(
            "residual_dtype", self.config.residual_dtype
        )
        self.config.weight_dtype = _check_float_dtype(
            "weight_dtype", self.config.weight_dtype
        )
        self.config.eps = float(self.config.eps)
        if self.config.eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {self.config.eps}")

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> AllReduceResidualRMSNormInputValues:
        del metadata_seed
        self.__post_init__()
        tensor_shape = (self.config.num_tokens, self.config.hidden_size)
        target_device = _resolve_device(self.config.device, device)
        return AllReduceResidualRMSNormInputValues(
            rank_inputs=[
                _generate_tensor(
                    shape=tensor_shape,
                    dtype=self.config.dtype,
                    seed=_child_seed(seed, rank + 1),
                    device=device,
                    configured_device=self.config.device,
                )
                for rank in range(self.config.world_size)
            ],
            residuals=[
                _generate_tensor(
                    shape=tensor_shape,
                    dtype=self.config.residual_dtype,
                    seed=_child_seed(seed, self.config.world_size + rank + 1),
                    device=device,
                    configured_device=self.config.device,
                )
                for rank in range(self.config.world_size)
            ],
            weight=_generate_tensor(
                shape=(self.config.hidden_size,),
                dtype=self.config.weight_dtype,
                seed=_child_seed(seed, 2 * self.config.world_size + 1),
                device=target_device,
                configured_device=self.config.device,
            ),
            eps=self.config.eps,
        )


@dataclass(init=False)
class ExpertParallelRoutingInputs(NumericsInputGenerator):
    """Generator for expert-parallel MoE dispatch/combine routing inputs."""

    config: ExpertParallelRoutingInputConfig

    def __init__(self, config: ExpertParallelRoutingInputConfig) -> None:
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
        self.config.num_experts = _check_positive(
            "num_experts", self.config.num_experts
        )
        self.config.top_k = _check_positive("top_k", self.config.top_k)
        if self.config.num_experts % self.config.world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")
        if self.config.top_k > self.config.num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.config.max_tokens_per_rank = _check_max_tokens_per_rank(
            world_size=self.config.world_size,
            total_tokens=self.config.total_tokens,
            max_tokens_per_rank=self.config.max_tokens_per_rank,
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.topk_ids_dtype = _check_topk_id_dtype(
            "topk_ids_dtype", self.config.topk_ids_dtype
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> ExpertParallelRoutingInputValues:
        self.__post_init__()
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        assert self.config.max_tokens_per_rank is not None
        target_device = _resolve_device(self.config.device, device)
        tokens_per_rank = _generate_tokens_per_rank(
            world_size=self.config.world_size,
            total_tokens=self.config.total_tokens,
            max_tokens_per_rank=self.config.max_tokens_per_rank,
            seed=_child_seed(metadata_base_seed, 1),
        )
        rank_hidden_states: list[torch.Tensor] = []
        topk_ids: list[torch.Tensor] = []
        topk_weights: list[torch.Tensor] = []
        expert_outputs: list[torch.Tensor] = []
        for rank, num_tokens in enumerate(tokens_per_rank):
            rank_hidden_states.append(
                _generate_tensor(
                    shape=(num_tokens, self.config.hidden_size),
                    dtype=self.config.dtype,
                    seed=_child_seed(seed, rank + 1),
                    device=target_device,
                    configured_device=self.config.device,
                )
            )
            topk_ids.append(
                _generate_unique_topk_ids(
                    num_tokens=num_tokens,
                    top_k=self.config.top_k,
                    num_experts=self.config.num_experts,
                    dtype=self.config.topk_ids_dtype,
                    seed=_child_seed(metadata_base_seed, rank + 2),
                    device=target_device,
                )
            )
            topk_weights.append(
                _generate_topk_weights(
                    num_tokens=num_tokens,
                    top_k=self.config.top_k,
                    seed=_child_seed(seed, self.config.world_size + rank + 1),
                    device=target_device,
                )
            )
            expert_outputs.append(
                _generate_tensor(
                    shape=(num_tokens, self.config.top_k, self.config.hidden_size),
                    dtype=self.config.dtype,
                    seed=_child_seed(seed, 2 * self.config.world_size + rank + 1),
                    device=target_device,
                    configured_device=self.config.device,
                )
            )
        return ExpertParallelRoutingInputValues(
            rank_hidden_states=rank_hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_outputs=expert_outputs,
            tokens_per_rank=tokens_per_rank,
            num_experts=self.config.num_experts,
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


def all_reduce_residual_rmsnorm_reference(
    values: AllReduceResidualRMSNormInputValues,
) -> AllReduceResidualRMSNormReferenceValues:
    """Return semantic fused all-reduce + residual RMSNorm outputs per rank."""

    if values.eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {values.eps}")
    if len(values.rank_inputs) != len(values.residuals):
        raise ValueError("rank_inputs must match residuals")
    reduced = _sum_rank_inputs_for_rmsnorm(values.rank_inputs)
    if values.weight.dim() != 1:
        raise ValueError(
            f"weight must be rank-1, got shape={tuple(values.weight.shape)}"
        )
    if values.weight.shape[0] != reduced.shape[-1]:
        raise ValueError(
            "weight length must match hidden dimension; "
            f"weight={tuple(values.weight.shape)}, reduced={tuple(reduced.shape)}"
        )
    if values.weight.device != reduced.device:
        raise ValueError(
            "weight must be on the same device as rank inputs; "
            f"weight={values.weight.device}, rank_inputs={reduced.device}"
        )
    norm_outputs: list[torch.Tensor] = []
    residual_outputs: list[torch.Tensor] = []
    for rank, residual in enumerate(values.residuals):
        if residual.shape != reduced.shape:
            raise ValueError(
                "residual shape must match reduced input shape; "
                f"rank={rank}, residual={tuple(residual.shape)}, "
                f"reduced={tuple(reduced.shape)}"
            )
        if residual.device != reduced.device:
            raise ValueError(
                "residual must be on the same device as rank inputs; "
                f"rank={rank}, residual={residual.device}, "
                f"rank_inputs={reduced.device}"
            )
        residual_out = reduced + residual.to(reduced.dtype)
        variance = residual_out.pow(2).mean(dim=-1, keepdim=True)
        norm = residual_out * torch.rsqrt(variance + values.eps)
        norm = norm * values.weight.to(reduced.dtype)
        residual_outputs.append(residual_out.contiguous())
        norm_outputs.append(norm.contiguous())
    return AllReduceResidualRMSNormReferenceValues(
        norm_outputs=norm_outputs,
        residual_outputs=residual_outputs,
    )


def _validate_expert_parallel_values(
    values: ExpertParallelRoutingInputValues,
) -> tuple[int, int, int, int]:
    world_size = len(values.rank_hidden_states)
    if world_size == 0:
        raise ValueError("rank_hidden_states must be non-empty")
    if (
        len(values.topk_ids) != world_size
        or len(values.topk_weights) != world_size
        or len(values.expert_outputs) != world_size
        or len(values.tokens_per_rank) != world_size
    ):
        raise ValueError("per-rank EP inputs must all match world_size")
    num_experts = _check_positive("num_experts", values.num_experts)
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    hidden_size: int | None = None
    top_k: int | None = None
    device: torch.device | None = None
    for rank in range(world_size):
        hidden = values.rank_hidden_states[rank]
        ids = values.topk_ids[rank]
        weights = values.topk_weights[rank]
        outputs = values.expert_outputs[rank]
        expected_tokens = int(values.tokens_per_rank[rank])
        if hidden.ndim != 2:
            raise ValueError(f"rank_hidden_states[{rank}] must be rank-2")
        if ids.ndim != 2 or weights.ndim != 2:
            raise ValueError(f"topk ids/weights for rank {rank} must be rank-2")
        if outputs.ndim != 3:
            raise ValueError(f"expert_outputs[{rank}] must be rank-3")
        if hidden.shape[0] != expected_tokens:
            raise ValueError(
                "rank hidden token dimension must match tokens_per_rank; "
                f"rank={rank}, hidden={tuple(hidden.shape)}, "
                f"tokens_per_rank={expected_tokens}"
            )
        if ids.shape != weights.shape:
            raise ValueError(f"topk ids/weights shape mismatch on rank {rank}")
        if ids.shape[0] != expected_tokens:
            raise ValueError(f"topk token dimension mismatch on rank {rank}")
        if outputs.shape[:2] != ids.shape or outputs.shape[0] != expected_tokens:
            raise ValueError(f"expert_outputs shape mismatch on rank {rank}")
        if hidden_size is None:
            hidden_size = int(hidden.shape[1])
        elif hidden.shape[1] != hidden_size:
            raise ValueError("all rank hidden sizes must match")
        if outputs.shape[2] != hidden_size:
            raise ValueError(f"expert_outputs hidden size mismatch on rank {rank}")
        if top_k is None:
            top_k = int(ids.shape[1])
        elif ids.shape[1] != top_k:
            raise ValueError("all ranks must use the same top_k")
        if top_k == 0:
            raise ValueError("top_k must be positive")
        if ids.dtype not in _TOPK_ID_DTYPES:
            raise ValueError("topk_ids must use an integer routing dtype")
        _check_float_dtype("topk_weights dtype", weights.dtype)
        _check_float_dtype("rank_hidden_states dtype", hidden.dtype)
        _check_float_dtype("expert_outputs dtype", outputs.dtype)
        if device is None:
            device = hidden.device
        if hidden.device != device or ids.device != device or weights.device != device:
            raise ValueError("all EP tensors must be on the same device")
        if outputs.device != device:
            raise ValueError("all EP tensors must be on the same device")
        if ids.numel() > 0:
            if ids.min().item() < 0:
                raise ValueError("topk_ids must be non-negative")
            if ids.max().item() >= num_experts:
                raise ValueError("topk_ids must be less than num_experts")
            sorted_ids = ids.sort(dim=-1).values
            if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
                raise ValueError("topk_ids must be unique for each token")
            weight_sums = weights.sum(dim=-1)
            if not torch.allclose(
                weight_sums,
                torch.ones_like(weight_sums),
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError("topk_weights must sum to 1 for each token")
    assert hidden_size is not None
    assert top_k is not None
    return world_size, num_experts, num_experts // world_size, hidden_size


def expert_parallel_routing_reference(
    values: ExpertParallelRoutingInputValues,
) -> ExpertParallelRoutingReferenceValues:
    """Return semantic EP dispatch records and weighted combine outputs."""

    world_size, num_experts, experts_per_rank, hidden_size = (
        _validate_expert_parallel_values(values)
    )
    del num_experts
    device = values.rank_hidden_states[0].device
    acc_dtype = (
        torch.float64
        if values.expert_outputs[0].dtype == torch.float64
        else torch.float32
    )
    records: list[list[ExpertParallelDispatchRecord]] = [[] for _ in range(world_size)]
    recv_hidden_rows: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    recv_topk_id_rows: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    recv_topk_weight_rows: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    recv_expert_output_rows: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    per_expert_counts = [
        torch.zeros((experts_per_rank,), dtype=torch.int32, device=device)
        for _ in range(world_size)
    ]
    combined_outputs = [
        torch.zeros(
            (tokens, hidden_size),
            dtype=acc_dtype,
            device=device,
        )
        for tokens in values.tokens_per_rank
    ]

    for source_rank in range(world_size):
        hidden = values.rank_hidden_states[source_rank]
        ids = values.topk_ids[source_rank]
        weights = values.topk_weights[source_rank]
        expert_outputs = values.expert_outputs[source_rank]
        for token in range(hidden.shape[0]):
            target_to_slots: dict[int, list[int]] = {}
            for slot in range(ids.shape[1]):
                expert = int(ids[token, slot].item())
                target_rank = expert // experts_per_rank
                target_to_slots.setdefault(target_rank, []).append(slot)
                combined_outputs[source_rank][token] += expert_outputs[token, slot].to(
                    acc_dtype
                ) * weights[token, slot].to(acc_dtype)
                per_expert_counts[target_rank][expert % experts_per_rank] += 1

            for target_rank in sorted(target_to_slots):
                slots = tuple(target_to_slots[target_rank])
                expert_ids = tuple(int(ids[token, slot].item()) for slot in slots)
                records[target_rank].append(
                    ExpertParallelDispatchRecord(
                        source_rank=source_rank,
                        source_token=token,
                        target_rank=target_rank,
                        topk_slots=slots,
                        expert_ids=expert_ids,
                    )
                )
                recv_hidden_rows[target_rank].append(hidden[token])
                recv_topk_id_rows[target_rank].append(ids[token])
                recv_topk_weight_rows[target_rank].append(weights[token])
                row_outputs = torch.zeros_like(expert_outputs[token])
                for slot in slots:
                    row_outputs[slot].copy_(expert_outputs[token, slot])
                recv_expert_output_rows[target_rank].append(row_outputs)

    recv_hidden_states: list[torch.Tensor] = []
    recv_topk_ids: list[torch.Tensor] = []
    recv_topk_weights: list[torch.Tensor] = []
    recv_expert_outputs: list[torch.Tensor] = []
    top_k = values.topk_ids[0].shape[1]
    for rank in range(world_size):
        if recv_hidden_rows[rank]:
            recv_hidden_states.append(torch.stack(recv_hidden_rows[rank]).contiguous())
            recv_topk_ids.append(torch.stack(recv_topk_id_rows[rank]).contiguous())
            recv_topk_weights.append(
                torch.stack(recv_topk_weight_rows[rank]).contiguous()
            )
            recv_expert_outputs.append(
                torch.stack(recv_expert_output_rows[rank]).contiguous()
            )
        else:
            recv_hidden_states.append(
                torch.empty(
                    (0, hidden_size),
                    dtype=values.rank_hidden_states[rank].dtype,
                    device=device,
                )
            )
            recv_topk_ids.append(
                torch.empty(
                    (0, top_k),
                    dtype=values.topk_ids[rank].dtype,
                    device=device,
                )
            )
            recv_topk_weights.append(
                torch.empty(
                    (0, top_k),
                    dtype=values.topk_weights[rank].dtype,
                    device=device,
                )
            )
            recv_expert_outputs.append(
                torch.empty(
                    (0, top_k, hidden_size),
                    dtype=values.expert_outputs[rank].dtype,
                    device=device,
                )
            )

    return ExpertParallelRoutingReferenceValues(
        dispatch_records=records,
        recv_hidden_states=recv_hidden_states,
        recv_topk_ids=recv_topk_ids,
        recv_topk_weights=recv_topk_weights,
        recv_expert_outputs=recv_expert_outputs,
        num_recv_tokens_per_expert=per_expert_counts,
        combined_outputs=[output.contiguous() for output in combined_outputs],
    )


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
