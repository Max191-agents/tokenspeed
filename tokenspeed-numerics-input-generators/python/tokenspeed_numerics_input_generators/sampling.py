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
    "TopKTopPRenormInputConfig",
    "TopKTopPRenormInputs",
    "TopKTopPRenormInputValues",
    "argmax_pair_reference",
    "argmax_reference",
    "gather_expand_scalars_reference",
    "min_p_renorm_reference",
    "top_k_top_p_renorm_reference",
]

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
class TopKTopPRenormInputValues:
    """Generated values for ``TopKTopPRenormInputs``."""

    probs: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor


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


def argmax_pair_reference(logits: torch.Tensor) -> torch.Tensor:
    """Reference row-wise max/index pair packed into a float32 tensor."""

    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")
    max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
    return torch.cat((max_vals.to(torch.float32), max_indices.to(torch.float32)), dim=1)


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
