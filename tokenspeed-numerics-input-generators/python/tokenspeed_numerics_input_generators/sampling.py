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

"""Sampling-family input generators for numerical correctness tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    "ArgmaxInputConfig",
    "ArgmaxInputs",
    "ArgmaxInputValues",
    "ArgmaxPairInputConfig",
    "ArgmaxPairInputs",
    "ArgmaxPairInputValues",
    "GatherExpandScalarsInputConfig",
    "GatherExpandScalarsInputs",
    "GatherExpandScalarsInputValues",
    "MinPRenormInputConfig",
    "MinPRenormInputs",
    "MinPRenormInputValues",
    "MinPSamplingInputConfig",
    "MinPSamplingInputs",
    "MinPSamplingInputValues",
    "MinPSamplingReferenceValues",
    "TopPRenormInputConfig",
    "TopPRenormInputs",
    "TopPRenormInputValues",
    "SoftmaxInputConfig",
    "SoftmaxInputs",
    "SoftmaxInputValues",
    "SpeculativeChainSamplingInputConfig",
    "SpeculativeChainSamplingInputs",
    "SpeculativeChainSamplingInputValues",
    "SpeculativeChainSamplingReferenceValues",
    "SpeculativeGreedyVerifyInputConfig",
    "SpeculativeGreedyVerifyInputs",
    "SpeculativeGreedyVerifyInputValues",
    "SpeculativeGreedyVerifyReferenceValues",
    "TopKTopPRenormInputConfig",
    "TopKTopPRenormInputs",
    "TopKTopPRenormInputValues",
    "argmax_pair_reference",
    "argmax_reference",
    "gather_expand_scalars_reference",
    "min_p_sampling_reference",
    "min_p_renorm_reference",
    "softmax_reference",
    "speculative_chain_sampling_reference",
    "speculative_greedy_verify_reference",
    "top_p_renorm_reference",
    "top_k_top_p_renorm_reference",
]

SoftmaxTemperatureMode = Literal["none", "scalar", "per_row"]

_LOGIT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}
_PROB_DTYPES = {
    torch.float32,
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


def _check_minimum(name: str, value: int, minimum: int) -> int:
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _check_float_dtype(
    name: str,
    dtype: torch.dtype,
    *,
    allowed: set[torch.dtype],
) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if dtype not in allowed:
        raise ValueError(f"{name} must be one of {sorted(str(d) for d in allowed)}")
    return dtype


def _check_index_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be torch.int32 or torch.int64, got {dtype}")
    return dtype


def _require_tensor(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} generation unexpectedly returned None")
    return values


def _randint(
    *,
    low: int,
    high: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    generator = _rng_for_device(target_device, seed)
    return torch.randint(
        low,
        high,
        shape,
        dtype=dtype,
        device=target_device,
        generator=generator,
    )


@dataclass
class ArgmaxInputValues:
    """Generated values for ``ArgmaxInputs``."""

    logits: torch.Tensor
    out: torch.Tensor | None
    expected_indices: torch.Tensor


@dataclass
class ArgmaxPairInputValues:
    """Generated values for ``ArgmaxPairInputs``."""

    logits: torch.Tensor
    out: torch.Tensor | None
    expected_pair: torch.Tensor


@dataclass
class SoftmaxInputValues:
    """Generated values for ``SoftmaxInputs``."""

    logits: torch.Tensor
    temperature: float | torch.Tensor | None


@dataclass
class SpeculativeGreedyVerifyInputValues:
    """Generated values for ``SpeculativeGreedyVerifyInputs``.

    ``predicts``, ``accept_index``, and ``accept_token_num`` are output buffers.
    ``candidates`` and ``target_predict`` are the proposed draft-token chain
    and target greedy predictions that determine the accepted prefix.
    """

    predicts: torch.Tensor
    accept_index: torch.Tensor
    accept_token_num: torch.Tensor
    candidates: torch.Tensor
    target_predict: torch.Tensor


@dataclass
class SpeculativeGreedyVerifyReferenceValues:
    """Reference outputs for speculative greedy chain verification."""

    predicts: torch.Tensor
    accept_index: torch.Tensor
    accept_token_num: torch.Tensor


@dataclass
class SpeculativeChainSamplingInputValues:
    """Generated values for ``SpeculativeChainSamplingInputs``.

    ``predicts``, ``accept_index``, and ``accept_token_num`` are output buffers.
    ``draft_probs`` is optional mutable input/output state; when present, the
    operation writes back the rejected draft token's target probability.
    """

    predicts: torch.Tensor
    accept_index: torch.Tensor
    accept_token_num: torch.Tensor
    candidates: torch.Tensor
    uniform_samples: torch.Tensor
    uniform_samples_for_final_sampling: torch.Tensor
    target_probs: torch.Tensor
    draft_probs: torch.Tensor | None


@dataclass
class SpeculativeChainSamplingReferenceValues:
    """Reference outputs for target-only chain speculative sampling."""

    predicts: torch.Tensor
    accept_index: torch.Tensor
    accept_token_num: torch.Tensor
    draft_probs: torch.Tensor | None


@dataclass
class ArgmaxInputConfig:
    """Initialization parameters for row-wise argmax inputs.

    The represented operation returns the lowest index whose logit is maximal
    for each row. The default generation plants a dominant maximum in every row
    so correctness tests are not accidentally dominated by random ties.
    """

    # Required: number of independent rows. Zero is valid.
    num_rows: int

    # Required: number of logits per row.
    vocab_size: int

    # Required: generated dtype for logits.
    dtype: torch.dtype

    # Optional: generate an output tensor for APIs that support out=.
    out_dtype: torch.dtype | None = None

    # Optional: ensure each row has a known unique maximum.
    plant_unique_max: bool = True

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class ArgmaxPairInputConfig:
    """Initialization parameters for row-wise max/index pair inputs.

    The represented operation returns a float32 tensor with shape
    ``[num_rows, 2]``. Column 0 stores the row maximum and column 1 stores the
    lowest column index whose logit equals that maximum.
    """

    # Required: number of independent rows. Zero is valid.
    num_rows: int

    # Required: number of logits per row.
    vocab_size: int

    # Required: generated dtype for logits.
    dtype: torch.dtype

    # Optional: generate an output tensor for APIs that support out=.
    include_out: bool = False

    # Optional: ensure each row has a known unique maximum.
    plant_unique_max: bool = True

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class SoftmaxInputConfig:
    """Initialization parameters for row-wise softmax inputs.

    The represented operation is ``softmax(logits / temperature)`` over each
    row. Temperature can be omitted, shared by all rows, or generated per row.
    """

    # Required: number of independent probability rows. Zero is valid.
    num_rows: int

    # Required: number of logits per row.
    vocab_size: int

    # Required: generated dtype for logits.
    dtype: torch.dtype

    # Optional: no temperature, one scalar temperature, or one fp32
    # temperature per row.
    temperature_mode: SoftmaxTemperatureMode = "none"

    # Optional: lower bound for generated positive temperatures.
    min_temperature: float = 0.5

    # Optional: upper bound for generated positive temperatures.
    max_temperature: float = 1.5

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class SpeculativeGreedyVerifyInputConfig:
    """Initialization parameters for greedy speculative-chain verification.

    The represented operation verifies a chain of proposed draft token IDs
    against target-model greedy predictions. For row ``b``, draft token
    ``candidates[b, i + 1]`` is accepted while it matches
    ``target_predict[b, i]``. The accepted prefix stops at the first mismatch.
    The verifier writes all target predictions into ``predicts``, writes flat
    output indices for accepted positions plus the final bonus token, and
    writes the number of accepted draft tokens per row.
    """

    # Required: number of independent speculative requests.
    batch_size: int

    # Required: number of target predictions/draft-chain slots per request.
    # The maximum accepted draft-token count is num_draft_tokens - 1 because
    # the final slot is the target-side bonus token.
    num_draft_tokens: int

    # Required: number of possible token IDs. Must be at least 2 so generated
    # rows can force both matching and mismatching prefixes.
    vocab_size: int

    # Optional: minimum generated accepted draft-token count per row.
    min_accepted_tokens: int = 0

    # Optional: maximum generated accepted draft-token count per row. ``None``
    # means ``num_draft_tokens - 1``.
    max_accepted_tokens: int | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class SpeculativeChainSamplingInputConfig:
    """Initialization parameters for target-only chain speculative sampling.

    The represented operation verifies a chain of draft token IDs against
    target-model probabilities. Position 0 is accepted by construction. For
    later draft positions, ``candidates[row, i]`` is accepted when its target
    probability at the current target position exceeds ``threshold_single`` or
    passes the ``uniform <= target_prob / threshold_acc`` coin test. The first
    rejected slot, or the final bonus slot when all draft tokens are accepted,
    is sampled from ``relu(target_probs - draft_probs)`` with the rejected
    token removed.
    """

    # Required: number of independent speculative requests.
    batch_size: int

    # Required: number of candidate/output slots per request.
    num_draft_tokens: int

    # Required: number of possible token IDs. Must be at least 2 so generated
    # rows can force both matching and sampled replacement tokens.
    vocab_size: int

    # Optional: accept immediately when the draft token's target probability is
    # at least this value.
    threshold_single: float = 0.9

    # Optional: denominator for the stochastic accept test.
    threshold_acc: float = 1.0

    # Optional: generate mutable draft probabilities. ``None`` models the
    # target-only runtime path that skips draft-probability traffic.
    include_draft_probs: bool = False

    # Optional: minimum generated accepted draft-token count per row.
    min_accepted_tokens: int = 0

    # Optional: maximum generated accepted draft-token count per row. ``None``
    # means ``num_draft_tokens - 1``.
    max_accepted_tokens: int | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class ArgmaxInputs(NumericsInputGenerator):
    """Generator for row-wise argmax logits."""

    config: ArgmaxInputConfig
    logits_input: TensorInput | None

    def __init__(self, config: ArgmaxInputConfig) -> None:
        self.config = config
        self.logits_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.dtype = _check_float_dtype(
            "dtype", self.config.dtype, allowed=_LOGIT_DTYPES
        )
        if self.config.out_dtype is not None:
            self.config.out_dtype = _check_index_dtype(
                "out_dtype", self.config.out_dtype
            )
        self.logits_input = self.logits_input or TensorInput(
            (self.config.num_rows, self.config.vocab_size),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> ArgmaxInputValues:
        self.__post_init__()
        if self.logits_input is None:
            raise ValueError("ArgmaxInputs child generators must be initialized")
        logits = _require_tensor(
            self.logits_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "logits",
        ).contiguous()
        if self.config.plant_unique_max:
            metadata_base_seed = seed if metadata_seed is None else metadata_seed
            expected = _randint(
                low=0,
                high=self.config.vocab_size,
                shape=(self.config.num_rows,),
                dtype=torch.int64,
                seed=_child_seed(metadata_base_seed, 1),
                device=device,
                configured_device=self.config.device,
            )
            logits = (logits.float() * 0.25).to(self.config.dtype)
            if self.config.num_rows > 0:
                rows = torch.arange(
                    self.config.num_rows,
                    dtype=torch.int64,
                    device=logits.device,
                )
                logits[rows, expected] = torch.tensor(
                    8.0,
                    dtype=logits.dtype,
                    device=logits.device,
                )
        else:
            expected = argmax_reference(logits).to(torch.int64)
        out = None
        if self.config.out_dtype is not None:
            out = torch.empty(
                (self.config.num_rows,),
                dtype=self.config.out_dtype,
                device=logits.device,
            )
        return ArgmaxInputValues(logits=logits, out=out, expected_indices=expected)


@dataclass(init=False)
class ArgmaxPairInputs(NumericsInputGenerator):
    """Generator for row-wise ``(max_value, argmax_index)`` packed outputs."""

    config: ArgmaxPairInputConfig
    logits_input: TensorInput | None

    def __init__(self, config: ArgmaxPairInputConfig) -> None:
        self.config = config
        self.logits_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.dtype = _check_float_dtype(
            "dtype", self.config.dtype, allowed=_LOGIT_DTYPES
        )
        self.logits_input = self.logits_input or TensorInput(
            (self.config.num_rows, self.config.vocab_size),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> ArgmaxPairInputValues:
        self.__post_init__()
        if self.logits_input is None:
            raise ValueError("ArgmaxPairInputs child generators must be initialized")
        logits = _require_tensor(
            self.logits_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "logits",
        ).contiguous()
        if self.config.plant_unique_max:
            metadata_base_seed = seed if metadata_seed is None else metadata_seed
            expected = _randint(
                low=0,
                high=self.config.vocab_size,
                shape=(self.config.num_rows,),
                dtype=torch.int64,
                seed=_child_seed(metadata_base_seed, 1),
                device=device,
                configured_device=self.config.device,
            )
            logits = (logits.float() * 0.25).to(self.config.dtype)
            if self.config.num_rows > 0:
                rows = torch.arange(
                    self.config.num_rows,
                    dtype=torch.int64,
                    device=logits.device,
                )
                logits[rows, expected] = torch.tensor(
                    8.0,
                    dtype=logits.dtype,
                    device=logits.device,
                )
        out = None
        if self.config.include_out:
            out = torch.empty(
                (self.config.num_rows, 2),
                dtype=torch.float32,
                device=logits.device,
            )
        return ArgmaxPairInputValues(
            logits=logits,
            out=out,
            expected_pair=argmax_pair_reference(logits),
        )


@dataclass(init=False)
class SoftmaxInputs(NumericsInputGenerator):
    """Generator for row-wise softmax logits and optional temperatures."""

    config: SoftmaxInputConfig
    logits_input: TensorInput | None

    def __init__(self, config: SoftmaxInputConfig) -> None:
        self.config = config
        self.logits_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.dtype = _check_float_dtype(
            "dtype", self.config.dtype, allowed=_LOGIT_DTYPES
        )
        if self.config.temperature_mode not in ("none", "scalar", "per_row"):
            raise ValueError(
                "temperature_mode must be 'none', 'scalar', or 'per_row'; "
                f"got {self.config.temperature_mode!r}"
            )
        self.config.min_temperature = float(self.config.min_temperature)
        self.config.max_temperature = float(self.config.max_temperature)
        if self.config.min_temperature <= 0.0:
            raise ValueError(
                f"min_temperature must be positive, got {self.config.min_temperature}"
            )
        if self.config.max_temperature < self.config.min_temperature:
            raise ValueError(
                "max_temperature must be >= min_temperature; got "
                f"{self.config.max_temperature} < {self.config.min_temperature}"
            )
        self.logits_input = self.logits_input or TensorInput(
            (self.config.num_rows, self.config.vocab_size),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> SoftmaxInputValues:
        self.__post_init__()
        if self.logits_input is None:
            raise ValueError("SoftmaxInputs child generator must be initialized")
        target_device = _resolve_device(self.config.device, device)
        logits = _require_tensor(
            self.logits_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "logits",
        ).contiguous()
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        temperature: float | torch.Tensor | None
        if self.config.temperature_mode == "none":
            temperature = None
        else:
            generator = _rng_for_device(
                target_device, _child_seed(metadata_base_seed, 1)
            )
            temp_count = (
                1 if self.config.temperature_mode == "scalar" else self.config.num_rows
            )
            temp = torch.rand(
                (temp_count,),
                dtype=torch.float32,
                device=target_device,
                generator=generator,
            )
            temp = (
                temp * (self.config.max_temperature - self.config.min_temperature)
                + self.config.min_temperature
            )
            if self.config.temperature_mode == "scalar":
                temperature = float(temp.item())
            else:
                temperature = temp.view(self.config.num_rows, 1).contiguous()
        return SoftmaxInputValues(logits=logits, temperature=temperature)


@dataclass(init=False)
class SpeculativeGreedyVerifyInputs(NumericsInputGenerator):
    """Generator for greedy verification of chain speculative decoding."""

    config: SpeculativeGreedyVerifyInputConfig

    def __init__(self, config: SpeculativeGreedyVerifyInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.batch_size = _check_nonnegative(
            "batch_size", self.config.batch_size
        )
        self.config.num_draft_tokens = _check_positive(
            "num_draft_tokens", self.config.num_draft_tokens
        )
        self.config.vocab_size = _check_minimum("vocab_size", self.config.vocab_size, 2)
        self.config.min_accepted_tokens = _check_nonnegative(
            "min_accepted_tokens", self.config.min_accepted_tokens
        )
        max_allowed = self.config.num_draft_tokens - 1
        if self.config.max_accepted_tokens is None:
            self.config.max_accepted_tokens = max_allowed
        else:
            self.config.max_accepted_tokens = _check_nonnegative(
                "max_accepted_tokens", self.config.max_accepted_tokens
            )
        if self.config.min_accepted_tokens > self.config.max_accepted_tokens:
            raise ValueError(
                "min_accepted_tokens must be <= max_accepted_tokens; got "
                f"{self.config.min_accepted_tokens} > "
                f"{self.config.max_accepted_tokens}"
            )
        if self.config.max_accepted_tokens > max_allowed:
            raise ValueError(
                "max_accepted_tokens must be <= num_draft_tokens - 1; got "
                f"{self.config.max_accepted_tokens} > {max_allowed}"
            )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> SpeculativeGreedyVerifyInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        shape = (self.config.batch_size, self.config.num_draft_tokens)
        candidates = _randint(
            low=0,
            high=self.config.vocab_size,
            shape=shape,
            dtype=torch.int32,
            seed=_child_seed(seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        target_predict = _randint(
            low=0,
            high=self.config.vocab_size,
            shape=shape,
            dtype=torch.int64,
            seed=_child_seed(seed, 2),
            device=device,
            configured_device=self.config.device,
        )
        accept_counts = _randint(
            low=self.config.min_accepted_tokens,
            high=self.config.max_accepted_tokens + 1,
            shape=(self.config.batch_size,),
            dtype=torch.int32,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )

        for row in range(self.config.batch_size):
            accepted = int(accept_counts[row].item())
            for position in range(accepted):
                target_predict[row, position] = candidates[row, position + 1].to(
                    torch.int64
                )
            if accepted < self.config.num_draft_tokens - 1:
                mismatch = (
                    candidates[row, accepted + 1].to(torch.int64) + 1
                ) % self.config.vocab_size
                target_predict[row, accepted] = mismatch

        return SpeculativeGreedyVerifyInputValues(
            predicts=torch.full(
                (self.config.batch_size * self.config.num_draft_tokens,),
                -1,
                dtype=torch.int32,
                device=target_device,
            ),
            accept_index=torch.full(
                shape,
                -1,
                dtype=torch.int32,
                device=target_device,
            ),
            accept_token_num=torch.empty(
                (self.config.batch_size,),
                dtype=torch.int32,
                device=target_device,
            ),
            candidates=candidates.contiguous(),
            target_predict=target_predict.contiguous(),
        )


@dataclass(init=False)
class SpeculativeChainSamplingInputs(NumericsInputGenerator):
    """Generator for target-only chain speculative sampling."""

    config: SpeculativeChainSamplingInputConfig

    def __init__(self, config: SpeculativeChainSamplingInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.batch_size = _check_nonnegative(
            "batch_size", self.config.batch_size
        )
        self.config.num_draft_tokens = _check_positive(
            "num_draft_tokens", self.config.num_draft_tokens
        )
        self.config.vocab_size = _check_minimum("vocab_size", self.config.vocab_size, 2)
        self.config.threshold_single = float(self.config.threshold_single)
        self.config.threshold_acc = float(self.config.threshold_acc)
        if not 0.0 <= self.config.threshold_single <= 1.0:
            raise ValueError(
                "threshold_single must be in [0, 1], got "
                f"{self.config.threshold_single}"
            )
        if not 0.0 < self.config.threshold_acc <= 1.0:
            raise ValueError(
                f"threshold_acc must be in (0, 1], got {self.config.threshold_acc}"
            )
        self.config.min_accepted_tokens = _check_nonnegative(
            "min_accepted_tokens", self.config.min_accepted_tokens
        )
        max_allowed = self.config.num_draft_tokens - 1
        if self.config.max_accepted_tokens is None:
            self.config.max_accepted_tokens = max_allowed
        else:
            self.config.max_accepted_tokens = _check_nonnegative(
                "max_accepted_tokens", self.config.max_accepted_tokens
            )
        if self.config.min_accepted_tokens > self.config.max_accepted_tokens:
            raise ValueError(
                "min_accepted_tokens must be <= max_accepted_tokens; got "
                f"{self.config.min_accepted_tokens} > "
                f"{self.config.max_accepted_tokens}"
            )
        if self.config.max_accepted_tokens > max_allowed:
            raise ValueError(
                "max_accepted_tokens must be <= num_draft_tokens - 1; got "
                f"{self.config.max_accepted_tokens} > {max_allowed}"
            )
        if (
            self.config.threshold_single == 0.0
            and self.config.min_accepted_tokens != max_allowed
        ):
            raise ValueError(
                "threshold_single=0 forces all draft tokens to be accepted; "
                "min_accepted_tokens must equal num_draft_tokens - 1"
            )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> SpeculativeChainSamplingInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        shape = (self.config.batch_size, self.config.num_draft_tokens)
        candidates = _randint(
            low=0,
            high=self.config.vocab_size,
            shape=shape,
            dtype=torch.int32,
            seed=_child_seed(seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        accept_counts = _randint(
            low=self.config.min_accepted_tokens,
            high=self.config.max_accepted_tokens + 1,
            shape=(self.config.batch_size,),
            dtype=torch.int32,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        target_probs = torch.zeros(
            self.config.batch_size,
            self.config.num_draft_tokens,
            self.config.vocab_size,
            dtype=torch.float32,
            device=target_device,
        )
        uniform_samples = torch.full(
            shape,
            0.25,
            dtype=torch.float32,
            device=target_device,
        )
        uniform_samples_for_final_sampling = torch.zeros(
            (self.config.batch_size,),
            dtype=torch.float32,
            device=target_device,
        )

        low_reject_prob = (
            min(self.config.threshold_single, self.config.threshold_acc) * 0.25
        )
        max_allowed = self.config.num_draft_tokens - 1
        for row in range(self.config.batch_size):
            accepted = int(accept_counts[row].item())
            for position in range(self.config.num_draft_tokens):
                next_candidate = candidates[row, min(position + 1, max_allowed)]
                sample_id = int(
                    (int(next_candidate.item()) + 1) % self.config.vocab_size
                )
                target_probs[row, position, sample_id] = 1.0

            for position in range(accepted):
                draft_id = int(candidates[row, position + 1].item())
                target_probs[row, position].zero_()
                target_probs[row, position, draft_id] = 1.0
                uniform_samples[row, position] = 0.25

            if accepted < max_allowed:
                draft_id = int(candidates[row, accepted + 1].item())
                sample_id = (draft_id + 1) % self.config.vocab_size
                target_probs[row, accepted].zero_()
                target_probs[row, accepted, draft_id] = low_reject_prob
                target_probs[row, accepted, sample_id] = 1.0 - low_reject_prob
                uniform_samples[row, accepted] = 0.99

        draft_probs = (
            torch.zeros_like(target_probs) if self.config.include_draft_probs else None
        )
        return SpeculativeChainSamplingInputValues(
            predicts=torch.full(
                (self.config.batch_size * self.config.num_draft_tokens,),
                -1,
                dtype=torch.int32,
                device=target_device,
            ),
            accept_index=torch.full(
                shape,
                -1,
                dtype=torch.int32,
                device=target_device,
            ),
            accept_token_num=torch.empty(
                (self.config.batch_size,),
                dtype=torch.int32,
                device=target_device,
            ),
            candidates=candidates.contiguous(),
            uniform_samples=uniform_samples.contiguous(),
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs.contiguous(),
            draft_probs=None if draft_probs is None else draft_probs.contiguous(),
        )


@dataclass
class GatherExpandScalarsInputValues:
    """Generated values for ``GatherExpandScalarsInputs``."""

    index: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    min_p: torch.Tensor | None
    seed: torch.Tensor | None
    offsets: torch.Tensor | None


@dataclass
class GatherExpandScalarsInputConfig:
    """Initialization parameters for sampling scalar gather/broadcast inputs."""

    # Required: number of rows in each scalar pool.
    pool_rows: int

    # Required: number of batch rows to gather.
    batch_size: int

    # Required: repeat count for each gathered scalar.
    n: int

    # Optional: include min-p pool/output.
    include_min_p: bool = True

    # Optional: include seed pool/output.
    include_seed: bool = True

    # Optional: include offsets pool/output.
    include_offsets: bool = True

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class GatherExpandScalarsInputs(NumericsInputGenerator):
    """Generator for sampling scalar pool gather and repeat-interleave inputs."""

    config: GatherExpandScalarsInputConfig

    def __init__(self, config: GatherExpandScalarsInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.pool_rows = _check_positive("pool_rows", self.config.pool_rows)
        self.config.batch_size = _check_nonnegative(
            "batch_size", self.config.batch_size
        )
        self.config.n = _check_positive("n", self.config.n)

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> GatherExpandScalarsInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        index = _randint(
            low=0,
            high=self.config.pool_rows,
            shape=(self.config.batch_size,),
            dtype=torch.int32,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        temperature = torch.linspace(
            0.5,
            1.5,
            self.config.pool_rows,
            dtype=torch.float32,
            device=target_device,
        )
        top_k = torch.arange(
            1,
            self.config.pool_rows + 1,
            dtype=torch.int32,
            device=target_device,
        )
        top_p = torch.linspace(
            0.5,
            1.0,
            self.config.pool_rows,
            dtype=torch.float32,
            device=target_device,
        )
        min_p = (
            torch.linspace(
                0.0,
                0.2,
                self.config.pool_rows,
                dtype=torch.float32,
                device=target_device,
            )
            if self.config.include_min_p
            else None
        )
        seed_values = (
            torch.arange(
                10_000,
                10_000 + self.config.pool_rows,
                dtype=torch.int64,
                device=target_device,
            )
            if self.config.include_seed
            else None
        )
        offsets = (
            torch.arange(
                0,
                self.config.pool_rows,
                dtype=torch.int32,
                device=target_device,
            )
            * 7
            if self.config.include_offsets
            else None
        )
        return GatherExpandScalarsInputValues(
            index=index.contiguous(),
            temperature=temperature.contiguous(),
            top_k=top_k.contiguous(),
            top_p=top_p.contiguous(),
            min_p=None if min_p is None else min_p.contiguous(),
            seed=None if seed_values is None else seed_values.contiguous(),
            offsets=None if offsets is None else offsets.contiguous(),
        )


@dataclass
class MinPRenormInputValues:
    """Generated values for ``MinPRenormInputs``."""

    probs: torch.Tensor
    min_p: torch.Tensor


@dataclass
class MinPSamplingInputValues:
    """Generated values for ``MinPSamplingInputs``."""

    probs: torch.Tensor
    min_p: torch.Tensor


@dataclass
class MinPSamplingReferenceValues:
    """Exact reference outputs for deterministic-support min-p sampling."""

    samples: torch.Tensor
    valid: torch.Tensor


@dataclass
class MinPRenormInputConfig:
    """Initialization parameters for min-p probability renormalization."""

    # Required: number of probability rows.
    num_rows: int

    # Required: vocabulary width.
    vocab_size: int

    # Optional: dtype for probabilities.
    dtype: torch.dtype = torch.float32

    # Optional: dtype for min-p values.
    min_p_dtype: torch.dtype = torch.float32

    # Optional: maximum generated min-p threshold ratio.
    max_min_p: float = 0.2

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MinPRenormInputs(NumericsInputGenerator):
    """Generator for min-p filtering and renormalization inputs."""

    config: MinPRenormInputConfig

    def __init__(self, config: MinPRenormInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.dtype = _check_float_dtype(
            "dtype", self.config.dtype, allowed=_PROB_DTYPES
        )
        self.config.min_p_dtype = _check_float_dtype(
            "min_p_dtype",
            self.config.min_p_dtype,
            allowed={torch.float32, torch.bfloat16},
        )
        if not 0.0 <= self.config.max_min_p <= 1.0:
            raise ValueError(
                f"max_min_p must be in [0, 1], got {self.config.max_min_p}"
            )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MinPRenormInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        generator = _rng_for_device(target_device, _child_seed(seed, 1))
        raw = torch.rand(
            self.config.num_rows,
            self.config.vocab_size,
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        )
        probs = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        min_generator = _rng_for_device(
            target_device, _child_seed(metadata_base_seed, 1)
        )
        min_p = torch.rand(
            self.config.num_rows,
            dtype=torch.float32,
            device=target_device,
            generator=min_generator,
        )
        min_p = (min_p * self.config.max_min_p).to(self.config.min_p_dtype)
        return MinPRenormInputValues(
            probs=probs.to(self.config.dtype).contiguous(),
            min_p=min_p.contiguous(),
        )


@dataclass
class MinPSamplingInputConfig:
    """Initialization parameters for exact-checkable min-p sampling inputs.

    Min-p sampling first filters a probability row with
    ``probs >= min_p * max(probs)`` and then samples categorically from the
    renormalized row. General categorical sampling needs statistical
    validation. This generator creates rows where exactly one token survives
    min-p filtering, making the sampled token deterministic and suitable for
    exact kernel smoke tests.
    """

    # Required: number of sample rows. Zero is valid.
    num_rows: int

    # Required: vocabulary width.
    vocab_size: int

    # Optional: dtype for probabilities. FlashInfer sampling expects float32.
    dtype: torch.dtype = torch.float32

    # Optional: dtype for generated min-p values.
    min_p_dtype: torch.dtype = torch.float32

    # Optional: lower bound for generated min-p threshold ratios.
    min_min_p: float = 0.05

    # Optional: upper bound for generated min-p threshold ratios.
    max_min_p: float = 0.5

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class MinPSamplingInputs(NumericsInputGenerator):
    """Generator for min-p sampling rows with deterministic filtered support."""

    config: MinPSamplingInputConfig

    def __init__(self, config: MinPSamplingInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.dtype = _check_float_dtype(
            "dtype", self.config.dtype, allowed=_PROB_DTYPES
        )
        self.config.min_p_dtype = _check_float_dtype(
            "min_p_dtype",
            self.config.min_p_dtype,
            allowed={torch.float32, torch.bfloat16},
        )
        self.config.min_min_p = float(self.config.min_min_p)
        self.config.max_min_p = float(self.config.max_min_p)
        if not 0.0 < self.config.min_min_p <= self.config.max_min_p <= 1.0:
            raise ValueError(
                "min-p bounds must satisfy 0 < min_min_p <= max_min_p <= 1; "
                f"got min_min_p={self.config.min_min_p}, "
                f"max_min_p={self.config.max_min_p}"
            )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> MinPSamplingInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        selected = _randint(
            low=0,
            high=self.config.vocab_size,
            shape=(self.config.num_rows,),
            dtype=torch.int64,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        probs = torch.zeros(
            self.config.num_rows,
            self.config.vocab_size,
            dtype=torch.float32,
            device=target_device,
        )
        if self.config.num_rows > 0:
            probs.scatter_(1, selected.view(-1, 1), 1.0)

        min_p_generator = _rng_for_device(
            target_device, _child_seed(metadata_base_seed, 2)
        )
        min_p = torch.rand(
            self.config.num_rows,
            dtype=torch.float32,
            device=target_device,
            generator=min_p_generator,
        )
        min_p = (
            min_p * (self.config.max_min_p - self.config.min_min_p)
            + self.config.min_min_p
        )
        return MinPSamplingInputValues(
            probs=probs.to(self.config.dtype).contiguous(),
            min_p=min_p.to(self.config.min_p_dtype).contiguous(),
        )


@dataclass
class TopKTopPRenormInputValues:
    """Generated values for ``TopKTopPRenormInputs``."""

    probs: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor


@dataclass
class TopPRenormInputValues:
    """Generated values for ``TopPRenormInputs``."""

    probs: torch.Tensor
    top_p: torch.Tensor


@dataclass
class TopPRenormInputConfig:
    """Initialization parameters for top-p probability renormalization.

    The represented operation is nucleus filtering followed by row-wise
    renormalization. Each row keeps the smallest set of highest-probability
    tokens whose cumulative probability reaches ``top_p[row]``.
    """

    # Required: number of probability rows.
    num_rows: int

    # Required: vocabulary width.
    vocab_size: int

    # Optional: dtype for generated probabilities.
    dtype: torch.dtype = torch.float32

    # Optional: lower bound for generated top-p thresholds.
    min_top_p: float = 0.5

    # Optional: upper bound for generated top-p thresholds.
    max_top_p: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class TopPRenormInputs(NumericsInputGenerator):
    """Generator for top-p probability filtering and renormalization."""

    config: TopPRenormInputConfig

    def __init__(self, config: TopPRenormInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.dtype = _check_float_dtype(
            "dtype", self.config.dtype, allowed=_PROB_DTYPES
        )
        self.config.min_top_p = float(self.config.min_top_p)
        self.config.max_top_p = float(self.config.max_top_p)
        if not 0.0 < self.config.min_top_p <= self.config.max_top_p <= 1.0:
            raise ValueError(
                "top-p bounds must satisfy 0 < min_top_p <= max_top_p <= 1; "
                f"got min_top_p={self.config.min_top_p}, "
                f"max_top_p={self.config.max_top_p}"
            )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> TopPRenormInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        generator = _rng_for_device(target_device, _child_seed(seed, 1))
        raw = torch.rand(
            self.config.num_rows,
            self.config.vocab_size,
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        )
        probs = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        top_p_generator = _rng_for_device(
            target_device, _child_seed(metadata_base_seed, 1)
        )
        top_p = torch.rand(
            self.config.num_rows,
            dtype=torch.float32,
            device=target_device,
            generator=top_p_generator,
        )
        top_p = (
            top_p * (self.config.max_top_p - self.config.min_top_p)
            + self.config.min_top_p
        )
        return TopPRenormInputValues(
            probs=probs.to(self.config.dtype).contiguous(),
            top_p=top_p.contiguous(),
        )


@dataclass
class TopKTopPRenormInputConfig:
    """Initialization parameters for top-k/top-p probability renormalization."""

    # Required: number of probability rows.
    num_rows: int

    # Required: vocabulary width.
    vocab_size: int

    # Optional: dtype for probabilities.
    dtype: torch.dtype = torch.float32

    # Optional: maximum top-k value sampled per row.
    max_top_k: int | None = None

    # Optional: include rows where top-k is disabled by setting k >= vocab.
    include_disabled_top_k: bool = True

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class TopKTopPRenormInputs(NumericsInputGenerator):
    """Generator for top-k followed by top-p probability renormalization."""

    config: TopKTopPRenormInputConfig

    def __init__(self, config: TopKTopPRenormInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_positive("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.dtype = _check_float_dtype(
            "dtype", self.config.dtype, allowed=_PROB_DTYPES
        )
        if self.config.max_top_k is None:
            self.config.max_top_k = self.config.vocab_size
        self.config.max_top_k = _check_positive("max_top_k", self.config.max_top_k)
        if self.config.max_top_k > self.config.vocab_size:
            raise ValueError(
                "max_top_k must be <= vocab_size; got "
                f"{self.config.max_top_k} > {self.config.vocab_size}"
            )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> TopKTopPRenormInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        generator = _rng_for_device(target_device, _child_seed(seed, 1))
        raw = torch.rand(
            self.config.num_rows,
            self.config.vocab_size,
            dtype=torch.float32,
            device=target_device,
            generator=generator,
        )
        probs = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        metadata_base_seed = seed if metadata_seed is None else metadata_seed
        top_k = _randint(
            low=1,
            high=int(self.config.max_top_k) + 1,
            shape=(self.config.num_rows,),
            dtype=torch.int32,
            seed=_child_seed(metadata_base_seed, 1),
            device=device,
            configured_device=self.config.device,
        )
        if self.config.include_disabled_top_k and self.config.num_rows > 0:
            top_k[1::2] = self.config.vocab_size
        top_p_generator = _rng_for_device(
            target_device, _child_seed(metadata_base_seed, 2)
        )
        top_p = torch.rand(
            self.config.num_rows,
            dtype=torch.float32,
            device=target_device,
            generator=top_p_generator,
        )
        top_p = top_p * 0.5 + 0.5
        return TopKTopPRenormInputValues(
            probs=probs.to(self.config.dtype).contiguous(),
            top_k=top_k.contiguous(),
            top_p=top_p.contiguous(),
        )


def argmax_reference(logits: torch.Tensor) -> torch.Tensor:
    """Reference row-wise argmax with NaNs ignored and all-NaN rows as -1."""

    if logits.dim() != 2:
        return torch.argmax(logits, dim=-1)
    valid = ~torch.isnan(logits)
    masked = torch.where(valid, logits, torch.full_like(logits, -float("inf")))
    out = torch.argmax(masked, dim=-1)
    all_invalid = ~valid.any(dim=-1)
    return torch.where(all_invalid, torch.full_like(out, -1), out)


def softmax_reference(
    logits: torch.Tensor,
    temperature: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference row-wise ``softmax(logits / temperature)`` in fp32."""

    if logits.dim() != 2:
        raise ValueError(f"softmax expects 2D logits, got {logits.dim()}D")
    _check_float_dtype("logits dtype", logits.dtype, allowed=_LOGIT_DTYPES)
    logits_fp32 = logits.float()
    if temperature is not None:
        if isinstance(temperature, torch.Tensor):
            if temperature.dtype != torch.float32:
                raise ValueError(
                    f"temperature tensor must be float32, got {temperature.dtype}"
                )
            if temperature.numel() != logits.shape[0]:
                raise ValueError(
                    "temperature tensor must have one entry per row; got "
                    f"{temperature.numel()} for {logits.shape[0]} rows"
                )
            if torch.any(temperature <= 0):
                raise ValueError("temperature tensor entries must be positive")
            temp = temperature.to(device=logits.device).view(-1, 1)
        else:
            temperature = float(temperature)
            if temperature <= 0.0:
                raise ValueError(f"temperature must be positive, got {temperature}")
            temp = torch.tensor(
                temperature,
                dtype=torch.float32,
                device=logits.device,
            )
        logits_fp32 = logits_fp32 / temp
    return torch.softmax(logits_fp32, dim=-1)


def argmax_pair_reference(logits: torch.Tensor) -> torch.Tensor:
    """Reference row-wise max/index pair packed into a float32 tensor."""

    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")
    max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
    return torch.cat((max_vals.to(torch.float32), max_indices.to(torch.float32)), dim=1)


def _sample_first_from_weights(
    weights: torch.Tensor, coin: torch.Tensor
) -> torch.Tensor:
    total = weights.sum()
    if total <= 0:
        return torch.tensor(
            weights.numel() - 1, dtype=torch.int32, device=weights.device
        )
    threshold = coin.to(torch.float32) * total
    cdf = torch.cumsum(weights.to(torch.float32), dim=0)
    selected = torch.nonzero(cdf > threshold, as_tuple=False)
    if selected.numel() == 0:
        return torch.tensor(
            weights.numel() - 1, dtype=torch.int32, device=weights.device
        )
    return selected[0, 0].to(torch.int32)


def speculative_greedy_verify_reference(
    values: SpeculativeGreedyVerifyInputValues,
) -> SpeculativeGreedyVerifyReferenceValues:
    """Reference for greedy chain speculative verification."""

    candidates = values.candidates
    target_predict = values.target_predict
    if candidates.dim() != 2:
        raise ValueError(f"candidates must be 2D, got {candidates.dim()}D")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates; got "
            f"{tuple(target_predict.shape)} vs {tuple(candidates.shape)}"
        )
    if candidates.dtype != torch.int32:
        raise ValueError(f"candidates must be int32, got {candidates.dtype}")
    if target_predict.dtype != torch.int64:
        raise ValueError(f"target_predict must be int64, got {target_predict.dtype}")
    if values.predicts.dtype != torch.int32:
        raise ValueError(f"predicts must be int32, got {values.predicts.dtype}")
    if values.accept_index.dtype != torch.int32:
        raise ValueError(f"accept_index must be int32, got {values.accept_index.dtype}")
    if values.accept_token_num.dtype != torch.int32:
        raise ValueError(
            f"accept_token_num must be int32, got {values.accept_token_num.dtype}"
        )

    batch_size, num_draft_tokens = candidates.shape
    if values.predicts.numel() != batch_size * num_draft_tokens:
        raise ValueError(
            "predicts must have batch_size * num_draft_tokens elements; got "
            f"{values.predicts.numel()} for {batch_size} * {num_draft_tokens}"
        )
    if values.accept_index.shape != candidates.shape:
        raise ValueError(
            "accept_index must have the same shape as candidates; got "
            f"{tuple(values.accept_index.shape)} vs {tuple(candidates.shape)}"
        )
    if values.accept_token_num.shape != (batch_size,):
        raise ValueError(
            "accept_token_num must have shape [batch_size]; got "
            f"{tuple(values.accept_token_num.shape)}"
        )

    if num_draft_tokens == 1:
        accepted = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=candidates.device,
        )
    else:
        match = candidates[:, 1:] == target_predict[:, :-1].to(candidates.dtype)
        leading = torch.cumprod(match.to(torch.int32), dim=1)
        accepted = leading.sum(dim=1).to(torch.int32)

    positions = torch.arange(
        num_draft_tokens,
        dtype=torch.int32,
        device=candidates.device,
    ).unsqueeze(0)
    row_offsets = (
        torch.arange(batch_size, dtype=torch.int32, device=candidates.device)
        .unsqueeze(1)
        .mul(num_draft_tokens)
    )
    flat_indices = row_offsets + positions
    valid = positions <= accepted.unsqueeze(1)
    accept_index = torch.where(
        valid,
        flat_indices,
        torch.full_like(flat_indices, -1),
    )
    predicts = values.predicts.clone()
    for row in range(batch_size):
        accepted_count = int(accepted[row].item())
        row_offset = row * num_draft_tokens
        for position in range(accepted_count + 1):
            predicts[row_offset + position] = target_predict[row, position].to(
                torch.int32
            )
    return SpeculativeGreedyVerifyReferenceValues(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accepted,
    )


def speculative_chain_sampling_reference(
    values: SpeculativeChainSamplingInputValues,
    *,
    threshold_single: float,
    threshold_acc: float,
) -> SpeculativeChainSamplingReferenceValues:
    """Reference for target-only chain speculative sampling."""

    threshold_single = float(threshold_single)
    threshold_acc = float(threshold_acc)
    if not 0.0 <= threshold_single <= 1.0:
        raise ValueError(f"threshold_single must be in [0, 1], got {threshold_single}")
    if not 0.0 < threshold_acc <= 1.0:
        raise ValueError(f"threshold_acc must be in (0, 1], got {threshold_acc}")

    candidates = values.candidates
    target_probs = values.target_probs
    if candidates.dim() != 2:
        raise ValueError(f"candidates must be 2D, got {candidates.dim()}D")
    if target_probs.dim() != 3:
        raise ValueError(f"target_probs must be 3D, got {target_probs.dim()}D")
    batch_size, num_draft_tokens = candidates.shape
    if target_probs.shape[:2] != candidates.shape:
        raise ValueError(
            "target_probs leading dimensions must match candidates; got "
            f"{tuple(target_probs.shape[:2])} vs {tuple(candidates.shape)}"
        )
    vocab_size = target_probs.shape[2]
    if candidates.dtype != torch.int32:
        raise ValueError(f"candidates must be int32, got {candidates.dtype}")
    if target_probs.dtype != torch.float32:
        raise ValueError(f"target_probs must be float32, got {target_probs.dtype}")
    if values.uniform_samples.shape != candidates.shape:
        raise ValueError(
            "uniform_samples must have the same shape as candidates; got "
            f"{tuple(values.uniform_samples.shape)} vs {tuple(candidates.shape)}"
        )
    if values.uniform_samples_for_final_sampling.shape != (batch_size,):
        raise ValueError(
            "uniform_samples_for_final_sampling must have shape [batch_size]; got "
            f"{tuple(values.uniform_samples_for_final_sampling.shape)}"
        )
    if values.predicts.numel() != batch_size * num_draft_tokens:
        raise ValueError(
            "predicts must have batch_size * num_draft_tokens elements; got "
            f"{values.predicts.numel()} for {batch_size} * {num_draft_tokens}"
        )
    if values.accept_index.shape != candidates.shape:
        raise ValueError(
            "accept_index must have the same shape as candidates; got "
            f"{tuple(values.accept_index.shape)} vs {tuple(candidates.shape)}"
        )
    if values.accept_token_num.shape != (batch_size,):
        raise ValueError(
            "accept_token_num must have shape [batch_size]; got "
            f"{tuple(values.accept_token_num.shape)}"
        )
    if values.draft_probs is not None:
        if values.draft_probs.shape != target_probs.shape:
            raise ValueError(
                "draft_probs must have the same shape as target_probs; got "
                f"{tuple(values.draft_probs.shape)} vs {tuple(target_probs.shape)}"
            )
        if values.draft_probs.dtype != torch.float32:
            raise ValueError(
                f"draft_probs must be float32, got {values.draft_probs.dtype}"
            )
    if torch.any(candidates < 0) or torch.any(candidates >= vocab_size):
        raise ValueError("candidate token IDs must be in [0, vocab_size)")
    if torch.any(values.uniform_samples < 0) or torch.any(values.uniform_samples >= 1):
        raise ValueError("uniform_samples entries must be in [0, 1)")
    final_samples = values.uniform_samples_for_final_sampling
    if torch.any(final_samples < 0) or torch.any(final_samples >= 1):
        raise ValueError("uniform_samples_for_final_sampling entries must be in [0, 1)")

    predicts = values.predicts.clone()
    accept_index = values.accept_index.clone()
    accept_token_num = values.accept_token_num.clone()
    draft_probs = None if values.draft_probs is None else values.draft_probs.clone()
    max_speculative_steps = num_draft_tokens - 1

    for row in range(batch_size):
        row_offset = row * num_draft_tokens
        accepted = 0
        current_prob_position = 0
        final_position = max_speculative_steps
        rejected_id = -1
        coin = values.uniform_samples[row, 0]
        accept_index[row, 0] = row_offset

        for position in range(1, num_draft_tokens):
            draft_id = int(candidates[row, position].item())
            target_prob = target_probs[row, current_prob_position, draft_id]
            if target_prob >= threshold_single or coin <= target_prob / threshold_acc:
                predicts[row_offset + position - 1] = draft_id
                coin = values.uniform_samples[row, position]
                current_prob_position = position
                accepted += 1
                accept_index[row, position] = row_offset + position
            else:
                rejected_id = draft_id
                final_position = position - 1
                break

        accept_token_num[row] = accepted
        use_draft_probs = draft_probs is not None and accepted != max_speculative_steps
        q = target_probs[row, current_prob_position].to(torch.float32)
        p = (
            draft_probs[row, current_prob_position].to(torch.float32)
            if use_draft_probs and draft_probs is not None
            else torch.zeros_like(q)
        )
        weights = torch.clamp(q - p, min=0.0)
        if rejected_id != -1:
            weights[rejected_id] = 0.0
        sampled_id = _sample_first_from_weights(
            weights,
            values.uniform_samples_for_final_sampling[row],
        )
        predicts[row_offset + final_position] = sampled_id
        if draft_probs is not None and rejected_id != -1:
            draft_probs[row, current_prob_position, rejected_id] = target_probs[
                row,
                current_prob_position,
                rejected_id,
            ]

    return SpeculativeChainSamplingReferenceValues(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        draft_probs=draft_probs,
    )


def gather_expand_scalars_reference(
    values: GatherExpandScalarsInputValues,
    *,
    n: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Reference for index_select followed by repeat_interleave."""

    n = _check_positive("n", n)
    index = values.index.to(torch.int64)

    def _select(pool: torch.Tensor) -> torch.Tensor:
        return pool.index_select(0, index).repeat_interleave(n, dim=0)

    min_p = None if values.min_p is None else _select(values.min_p)
    seed = None if values.seed is None else _select(values.seed)
    offsets = None
    if values.offsets is not None:
        offsets = _select(values.offsets).to(torch.int64)
    return (
        _select(values.temperature),
        _select(values.top_k),
        _select(values.top_p),
        min_p,
        seed,
        offsets,
    )


def min_p_renorm_reference(probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Reference min-p filter and renormalization."""

    max_probs = probs.max(dim=-1, keepdim=True).values
    keep = probs >= min_p.to(probs.dtype).view(-1, 1) * max_probs
    out = torch.where(keep, probs, torch.zeros_like(probs))
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def min_p_sampling_reference(
    probs: torch.Tensor,
    min_p: torch.Tensor,
) -> MinPSamplingReferenceValues:
    """Exact reference for min-p sampling rows with single-token support.

    General min-p sampling is categorical and should be validated
    statistically. This helper is intentionally exact only for rows where
    min-p filtering leaves exactly one valid token.
    """

    if probs.dim() != 2:
        raise ValueError(f"probs must be 2D, got {probs.dim()}D")
    if min_p.dim() != 1:
        raise ValueError(f"min_p must be 1D, got {min_p.dim()}D")
    if min_p.numel() != probs.shape[0]:
        raise ValueError(
            "min_p must have one threshold per probability row; got "
            f"{min_p.numel()} for {probs.shape[0]} rows"
        )
    if torch.any(min_p <= 0.0) or torch.any(min_p > 1.0):
        raise ValueError("min_p thresholds must be in (0, 1]")

    max_probs = probs.max(dim=-1, keepdim=True).values
    keep = probs >= min_p.to(probs.dtype).view(-1, 1) * max_probs
    support_size = keep.sum(dim=-1)
    if torch.any(support_size != 1):
        raise ValueError(
            "exact min-p sampling reference requires one surviving token per row"
        )
    samples = torch.argmax(keep.to(torch.int32), dim=-1).to(torch.int32)
    return MinPSamplingReferenceValues(
        samples=samples,
        valid=torch.ones(probs.shape[0], dtype=torch.bool, device=probs.device),
    )


def top_p_renorm_reference(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Reference top-p filter and renormalization."""

    if probs.dim() != 2:
        raise ValueError(f"probs must be 2D, got {probs.dim()}D")
    if top_p.dim() != 1:
        raise ValueError(f"top_p must be 1D, got {top_p.dim()}D")
    if top_p.numel() != probs.shape[0]:
        raise ValueError(
            "top_p must have one threshold per probability row; got "
            f"{top_p.numel()} for {probs.shape[0]} rows"
        )
    if torch.any(top_p <= 0.0) or torch.any(top_p > 1.0):
        raise ValueError("top_p thresholds must be in (0, 1]")

    out = probs.clone()
    batch_size, vocab_size = out.shape
    for row in range(batch_size):
        sorted_values, _indices = torch.sort(out[row], descending=True)
        cumsum = torch.cumsum(sorted_values, dim=0)
        p = float(top_p[row].item())
        keep = min(int((cumsum < p).sum().item()) + 1, vocab_size)
        threshold = sorted_values[keep - 1]
        out[row] = torch.where(
            out[row] >= threshold,
            out[row],
            torch.zeros_like(out[row]),
        )
        denom = out[row].sum()
        if denom > 0:
            out[row] = out[row] / denom
    return out


def top_k_top_p_renorm_reference(
    probs: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """Reference top-k renormalization followed by top-p renormalization."""

    out = probs.clone()
    batch_size, vocab_size = out.shape
    for row in range(batch_size):
        k = min(int(top_k[row].item()), vocab_size)
        if k < vocab_size:
            kth = torch.topk(out[row], k, sorted=False).values.min()
            out[row] = torch.where(
                out[row] >= kth, out[row], torch.zeros_like(out[row])
            )
            denom = out[row].sum()
            if denom > 0:
                out[row] = out[row] / denom

    return top_p_renorm_reference(out, top_p)
