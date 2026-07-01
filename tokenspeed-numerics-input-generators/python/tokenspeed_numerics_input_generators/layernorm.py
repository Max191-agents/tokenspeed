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

"""Layernorm-family input generators for numerical correctness tests."""

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
    "FusedQKRMSNormRopeGateInputConfig",
    "FusedQKRMSNormRopeGateInputs",
    "FusedQKRMSNormRopeGateInputValues",
    "ParallelRMSNormInputConfig",
    "ParallelRMSNormInputs",
    "ParallelRMSNormInputValues",
    "QKRMSNormInputConfig",
    "QKRMSNormInputs",
    "QKRMSNormInputValues",
    "RMSNormInputConfig",
    "RMSNormInputs",
    "RMSNormInputValues",
    "build_rope_cos_sin_cache",
    "fused_qk_rmsnorm_rope_gate_reference",
    "qk_rmsnorm_reference",
    "rmsnorm_reference",
]

QKInputLayout = Literal["dense", "qkv_split"]

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


def _require_tensor(values: torch.Tensor | None, name: str) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} generation unexpectedly returned None")
    return values


def _generate_weight(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    generator = _rng_for_device(target_device, seed)
    values = torch.randn(
        *shape,
        dtype=torch.float32,
        device=target_device,
        generator=generator,
    )
    return (1.0 + 0.1 * values).to(dtype)


def _generate_positions(
    *,
    num_tokens: int,
    max_position: int,
    seed: int,
    device: DeviceLike,
    configured_device: DeviceLike,
) -> torch.Tensor:
    target_device = _resolve_device(configured_device, device)
    generator = _rng_for_device(target_device, seed)
    return torch.randint(
        0,
        max_position,
        (num_tokens,),
        dtype=torch.int64,
        device=target_device,
        generator=generator,
    )


@dataclass
class RMSNormInputValues:
    """Generated values for ``RMSNormInputs``."""

    x: torch.Tensor
    weight: torch.Tensor
    residual: torch.Tensor | None


@dataclass
class RMSNormInputConfig:
    """Initialization parameters for ordinary or residual RMSNorm inputs.

    The represented operation normalizes each row over the final dimension:
    ``y = x * rsqrt(mean(x * x) + eps) * weight``. When ``with_residual`` is
    set, the effective normalized input is ``x + residual`` and that sum is
    also an operation output.
    """

    # Required: number of independent rows/tokens. Zero is valid.
    num_tokens: int

    # Required: width normalized by each RMSNorm reduction.
    hidden_dim: int

    # Required: generated dtype for x and optional residual.
    dtype: torch.dtype

    # Optional: dtype for the multiplicative weight vector.
    weight_dtype: torch.dtype = torch.float32

    # Optional: epsilon added inside rsqrt.
    eps: float = 1e-6

    # Optional: generate a residual tensor and model fused add + RMSNorm.
    with_residual: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class RMSNormInputs(NumericsInputGenerator):
    """Generator for RMSNorm and fused residual-add RMSNorm inputs."""

    config: RMSNormInputConfig
    x_input: TensorInput | None
    residual_input: TensorInput | None

    def __init__(self, config: RMSNormInputConfig) -> None:
        self.config = config
        self.x_input = None
        self.residual_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_dim = _check_positive("hidden_dim", self.config.hidden_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.weight_dtype = _check_float_dtype(
            "weight_dtype", self.config.weight_dtype
        )
        if self.config.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.config.eps}")

        shape = (self.config.num_tokens, self.config.hidden_dim)
        self.x_input = self.x_input or TensorInput(
            shape,
            self.config.dtype,
            device=self.config.device,
        )
        if self.config.with_residual:
            self.residual_input = self.residual_input or TensorInput(
                shape,
                self.config.dtype,
                device=self.config.device,
            )
        else:
            self.residual_input = None

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> RMSNormInputValues:
        self.__post_init__()
        if self.x_input is None:
            raise ValueError("RMSNormInputs child generators must be initialized")
        x = _require_tensor(
            self.x_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "x",
        )
        residual = None
        if self.residual_input is not None:
            residual = _require_tensor(
                self.residual_input.generate(
                    seed=_child_seed(seed, 2), device=device
                ).values,
                "residual",
            )
        weight = _generate_weight(
            (self.config.hidden_dim,),
            dtype=self.config.weight_dtype,
            seed=_child_seed(seed, 3),
            device=device,
            configured_device=self.config.device,
        )
        return RMSNormInputValues(
            x=x.contiguous(),
            weight=weight.contiguous(),
            residual=None if residual is None else residual.contiguous(),
        )


@dataclass
class QKRMSNormInputValues:
    """Generated values for ``QKRMSNormInputs``."""

    q: torch.Tensor
    k: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    qkv_storage: torch.Tensor | None = None


@dataclass
class QKRMSNormInputConfig:
    """Initialization parameters for per-head Q/K RMSNorm inputs.

    The represented operation applies independent RMSNorm reductions to every
    query and key head. ``q`` has flattened shape
    ``[num_tokens, num_q_heads * head_dim]`` and ``k`` has flattened shape
    ``[num_tokens, num_kv_heads * head_dim]``. Each head is normalized over
    ``head_dim``.
    """

    # Required: number of token rows. Zero is valid.
    num_tokens: int

    # Required: number of query heads.
    num_q_heads: int

    # Required: number of key/value heads.
    num_kv_heads: int

    # Required: per-head RMSNorm reduction width.
    head_dim: int

    # Required: generated dtype for q and k.
    dtype: torch.dtype

    # Optional: dtype for q_weight and k_weight.
    weight_dtype: torch.dtype = torch.float32

    # Optional: epsilon added inside rsqrt.
    eps: float = 1e-6

    # Optional: generate dense q/k tensors or strided q/k views from packed
    # [q, k, v] storage.
    input_layout: QKInputLayout = "dense"

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class QKRMSNormInputs(NumericsInputGenerator):
    """Generator for independent per-head RMSNorm of query and key tensors."""

    config: QKRMSNormInputConfig
    q_input: TensorInput | None
    k_input: TensorInput | None
    qkv_input: TensorInput | None

    def __init__(self, config: QKRMSNormInputConfig) -> None:
        self.config = config
        self.q_input = None
        self.k_input = None
        self.qkv_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.weight_dtype = _check_float_dtype(
            "weight_dtype", self.config.weight_dtype
        )
        if self.config.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.config.eps}")
        if self.config.input_layout not in ("dense", "qkv_split"):
            raise ValueError(f"unsupported input_layout={self.config.input_layout!r}")

        q_size = self.config.num_q_heads * self.config.head_dim
        kv_size = self.config.num_kv_heads * self.config.head_dim
        if self.config.input_layout == "dense":
            self.q_input = self.q_input or TensorInput(
                (self.config.num_tokens, q_size),
                self.config.dtype,
                device=self.config.device,
            )
            self.k_input = self.k_input or TensorInput(
                (self.config.num_tokens, kv_size),
                self.config.dtype,
                device=self.config.device,
            )
            self.qkv_input = None
        else:
            self.qkv_input = self.qkv_input or TensorInput(
                (self.config.num_tokens, q_size + 2 * kv_size),
                self.config.dtype,
                device=self.config.device,
            )
            self.q_input = None
            self.k_input = None

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> QKRMSNormInputValues:
        self.__post_init__()
        q_size = self.config.num_q_heads * self.config.head_dim
        kv_size = self.config.num_kv_heads * self.config.head_dim
        qkv_storage = None
        if self.config.input_layout == "dense":
            if self.q_input is None or self.k_input is None:
                raise ValueError("dense q/k generators must be initialized")
            q = _require_tensor(
                self.q_input.generate(seed=_child_seed(seed, 1), device=device).values,
                "q",
            ).contiguous()
            k = _require_tensor(
                self.k_input.generate(seed=_child_seed(seed, 2), device=device).values,
                "k",
            ).contiguous()
        else:
            if self.qkv_input is None:
                raise ValueError("qkv_split backing generator must be initialized")
            qkv_storage = _require_tensor(
                self.qkv_input.generate(
                    seed=_child_seed(seed, 1), device=device
                ).values,
                "qkv_storage",
            )
            q, k, _v = qkv_storage.split([q_size, kv_size, kv_size], dim=-1)

        q_weight = _generate_weight(
            (self.config.head_dim,),
            dtype=self.config.weight_dtype,
            seed=_child_seed(seed, 3),
            device=device,
            configured_device=self.config.device,
        )
        k_weight = _generate_weight(
            (self.config.head_dim,),
            dtype=self.config.weight_dtype,
            seed=_child_seed(seed, 4),
            device=device,
            configured_device=self.config.device,
        )
        return QKRMSNormInputValues(
            q=q,
            k=k,
            q_weight=q_weight.contiguous(),
            k_weight=k_weight.contiguous(),
            qkv_storage=qkv_storage,
        )


@dataclass
class FusedQKRMSNormRopeGateInputValues:
    """Generated values for ``FusedQKRMSNormRopeGateInputs``."""

    q_gate: torch.Tensor
    k: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    cos_sin_cache: torch.Tensor
    positions: torch.Tensor


@dataclass
class FusedQKRMSNormRopeGateInputConfig:
    """Initialization parameters for fused QK RMSNorm + RoPE + gate inputs.

    ``q_gate`` stores each query head as ``[q | gate]``. The operation
    extracts ``q`` and ``gate``, applies per-head RMSNorm to ``q`` and ``k``,
    applies RoPE to the first ``rotary_dim`` values of each normalized head,
    and returns the normalized/rotated q, normalized/rotated k, and copied gate.
    """

    # Required: number of token rows. Zero is valid.
    num_tokens: int

    # Required: number of query heads.
    num_q_heads: int

    # Required: number of key/value heads.
    num_kv_heads: int

    # Required: per-head RMSNorm reduction width.
    head_dim: int

    # Required: even rotary width. Must be positive and <= head_dim.
    rotary_dim: int

    # Required: generated dtype for q_gate and k.
    dtype: torch.dtype

    # Optional: dtype for q_weight and k_weight.
    weight_dtype: torch.dtype = torch.float32

    # Optional: number of position rows in the generated RoPE cache.
    max_position: int = 1024

    # Optional: RoPE frequency base used to build the cache.
    rope_base: float = 10000.0

    # Optional: epsilon added inside rsqrt.
    eps: float = 1e-6

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class FusedQKRMSNormRopeGateInputs(NumericsInputGenerator):
    """Generator for fused Q/K RMSNorm, RoPE, and gate split inputs."""

    config: FusedQKRMSNormRopeGateInputConfig
    q_gate_input: TensorInput | None
    k_input: TensorInput | None

    def __init__(self, config: FusedQKRMSNormRopeGateInputConfig) -> None:
        self.config = config
        self.q_gate_input = None
        self.k_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.num_kv_heads = _check_positive(
            "num_kv_heads", self.config.num_kv_heads
        )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.rotary_dim = _check_positive("rotary_dim", self.config.rotary_dim)
        self.config.max_position = _check_positive(
            "max_position", self.config.max_position
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.weight_dtype = _check_float_dtype(
            "weight_dtype", self.config.weight_dtype
        )
        if self.config.rotary_dim > self.config.head_dim:
            raise ValueError(
                "rotary_dim must be <= head_dim; got "
                f"{self.config.rotary_dim} > {self.config.head_dim}"
            )
        if self.config.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {self.config.rotary_dim}")
        if self.config.rope_base <= 0.0:
            raise ValueError(f"rope_base must be positive, got {self.config.rope_base}")
        if self.config.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.config.eps}")

        q_gate_size = self.config.num_q_heads * 2 * self.config.head_dim
        k_size = self.config.num_kv_heads * self.config.head_dim
        self.q_gate_input = self.q_gate_input or TensorInput(
            (self.config.num_tokens, q_gate_size),
            self.config.dtype,
            device=self.config.device,
        )
        self.k_input = self.k_input or TensorInput(
            (self.config.num_tokens, k_size),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> FusedQKRMSNormRopeGateInputValues:
        self.__post_init__()
        if self.q_gate_input is None or self.k_input is None:
            raise ValueError("fused QK child generators must be initialized")
        q_gate = _require_tensor(
            self.q_gate_input.generate(seed=_child_seed(seed, 1), device=device).values,
            "q_gate",
        )
        k = _require_tensor(
            self.k_input.generate(seed=_child_seed(seed, 2), device=device).values,
            "k",
        )
        q_weight = _generate_weight(
            (self.config.head_dim,),
            dtype=self.config.weight_dtype,
            seed=_child_seed(seed, 3),
            device=device,
            configured_device=self.config.device,
        )
        k_weight = _generate_weight(
            (self.config.head_dim,),
            dtype=self.config.weight_dtype,
            seed=_child_seed(seed, 4),
            device=device,
            configured_device=self.config.device,
        )
        target_device = _resolve_device(self.config.device, device)
        cos_sin_cache = build_rope_cos_sin_cache(
            rotary_dim=self.config.rotary_dim,
            max_position=self.config.max_position,
            base=self.config.rope_base,
            device=target_device,
        )
        positions = _generate_positions(
            num_tokens=self.config.num_tokens,
            max_position=self.config.max_position,
            seed=_child_seed(seed if metadata_seed is None else metadata_seed, 5),
            device=device,
            configured_device=self.config.device,
        )
        return FusedQKRMSNormRopeGateInputValues(
            q_gate=q_gate.contiguous(),
            k=k.contiguous(),
            q_weight=q_weight.contiguous(),
            k_weight=k_weight.contiguous(),
            cos_sin_cache=cos_sin_cache.contiguous(),
            positions=positions.contiguous(),
        )


@dataclass
class ParallelRMSNormInputValues:
    """Generated values for ``ParallelRMSNormInputs``."""

    input1: torch.Tensor
    weight1: torch.Tensor
    output1: torch.Tensor
    input2: torch.Tensor
    weight2: torch.Tensor
    output2: torch.Tensor


@dataclass
class ParallelRMSNormInputConfig:
    """Initialization parameters for two independent RMSNorms sharing rows."""

    # Required: shared row count for both inputs.
    num_tokens: int

    # Required: hidden size for the first RMSNorm.
    hidden_dim1: int

    # Required: hidden size for the second RMSNorm.
    hidden_dim2: int

    # Required: generated dtype for both inputs and outputs.
    dtype: torch.dtype

    # Optional: dtype for both weight vectors.
    weight_dtype: torch.dtype = torch.float32

    # Optional: epsilon added inside rsqrt.
    eps: float = 1e-6

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class ParallelRMSNormInputs(NumericsInputGenerator):
    """Generator for the fused-parallel RMSNorm kernel input contract."""

    config: ParallelRMSNormInputConfig
    input1: RMSNormInputs | None
    input2: RMSNormInputs | None

    def __init__(self, config: ParallelRMSNormInputConfig) -> None:
        self.config = config
        self.input1 = None
        self.input2 = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.hidden_dim1 = _check_positive(
            "hidden_dim1", self.config.hidden_dim1
        )
        self.config.hidden_dim2 = _check_positive(
            "hidden_dim2", self.config.hidden_dim2
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.weight_dtype = _check_float_dtype(
            "weight_dtype", self.config.weight_dtype
        )
        if self.config.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.config.eps}")
        self.input1 = self.input1 or RMSNormInputs(
            RMSNormInputConfig(
                num_tokens=self.config.num_tokens,
                hidden_dim=self.config.hidden_dim1,
                dtype=self.config.dtype,
                weight_dtype=self.config.weight_dtype,
                eps=self.config.eps,
                device=self.config.device,
            )
        )
        self.input2 = self.input2 or RMSNormInputs(
            RMSNormInputConfig(
                num_tokens=self.config.num_tokens,
                hidden_dim=self.config.hidden_dim2,
                dtype=self.config.dtype,
                weight_dtype=self.config.weight_dtype,
                eps=self.config.eps,
                device=self.config.device,
            )
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> ParallelRMSNormInputValues:
        self.__post_init__()
        if self.input1 is None or self.input2 is None:
            raise ValueError("ParallelRMSNormInputs child generators must exist")
        values1 = self.input1.generate(seed=_child_seed(seed, 1), device=device)
        values2 = self.input2.generate(seed=_child_seed(seed, 2), device=device)
        output1 = torch.empty_like(values1.x)
        output2 = torch.empty_like(values2.x)
        return ParallelRMSNormInputValues(
            input1=values1.x,
            weight1=values1.weight,
            output1=output1,
            input2=values2.x,
            weight2=values2.weight,
            output2=output2,
        )


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute RMSNorm in PyTorch using float32 reduction semantics."""

    x_fp32 = x.to(torch.float32)
    if residual is not None:
        x_fp32 = x_fp32 + residual.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    out = (x_fp32 * torch.rsqrt(variance + eps) * weight.to(torch.float32)).to(x.dtype)
    if residual is None:
        return out
    return out, x_fp32.to(x.dtype)


def qk_rmsnorm_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent per-head RMSNorm for q and k."""

    head_dim = _check_positive("head_dim", head_dim)

    def _norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % head_dim != 0:
            raise ValueError(
                f"last dimension {x.shape[-1]} must be divisible by head_dim {head_dim}"
            )
        x_by_head = x.reshape(-1, head_dim).to(torch.float32)
        variance = x_by_head.pow(2).mean(dim=-1, keepdim=True)
        out = x_by_head * torch.rsqrt(variance + eps) * weight.to(torch.float32)
        return out.to(x.dtype).view(x.shape)

    return _norm(q, q_weight), _norm(k, k_weight)


def build_rope_cos_sin_cache(
    *,
    rotary_dim: int,
    max_position: int,
    base: float,
    device: DeviceLike,
) -> torch.Tensor:
    """Build a RoPE cache with per-position ``[cos | sin]`` layout."""

    rotary_dim = _check_positive("rotary_dim", rotary_dim)
    max_position = _check_positive("max_position", max_position)
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if base <= 0.0:
        raise ValueError(f"base must be positive, got {base}")
    target_device = _resolve_device(device, None)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=target_device)
            / rotary_dim
        )
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=target_device)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def fused_qk_rmsnorm_rope_gate_reference(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for split + QK RMSNorm + partial RoPE + gate copy."""

    num_q_heads = _check_positive("num_q_heads", num_q_heads)
    num_kv_heads = _check_positive("num_kv_heads", num_kv_heads)
    head_dim = _check_positive("head_dim", head_dim)
    rotary_dim = _check_positive("rotary_dim", rotary_dim)
    if rotary_dim > head_dim:
        raise ValueError(
            f"rotary_dim must be <= head_dim, got {rotary_dim} > {head_dim}"
        )
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if q_gate.shape[-1] != num_q_heads * 2 * head_dim:
        raise ValueError(
            "q_gate last dimension must be num_q_heads * 2 * head_dim; got "
            f"{q_gate.shape[-1]}"
        )
    if k.shape[-1] != num_kv_heads * head_dim:
        raise ValueError(
            f"k last dimension must be num_kv_heads * head_dim; got {k.shape[-1]}"
        )

    dtype = q_gate.dtype
    num_tokens = q_gate.shape[0]
    q_gate_3d = q_gate.view(num_tokens, num_q_heads, 2 * head_dim)
    q, gate = torch.chunk(q_gate_3d, 2, dim=-1)
    q = q.reshape(num_tokens, num_q_heads * head_dim).contiguous()
    gate = gate.reshape(num_tokens, num_q_heads * head_dim).contiguous()

    q_normed, k_normed = qk_rmsnorm_reference(
        q,
        k,
        q_weight,
        k_weight,
        eps,
        head_dim=head_dim,
    )

    half_rotary = rotary_dim // 2
    cos = cos_sin_cache[positions, :half_rotary].unsqueeze(1).to(torch.float32)
    sin = (
        cos_sin_cache[positions, half_rotary:rotary_dim].unsqueeze(1).to(torch.float32)
    )

    def _apply_partial_rope(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        x_h = x.reshape(num_tokens, num_heads, head_dim).to(torch.float32)
        x1 = x_h[..., :half_rotary]
        x2 = x_h[..., half_rotary:rotary_dim]
        out = x_h.clone()
        out[..., :half_rotary] = x1 * cos - x2 * sin
        out[..., half_rotary:rotary_dim] = x2 * cos + x1 * sin
        return out.reshape(num_tokens, num_heads * head_dim).to(dtype)

    return (
        _apply_partial_rope(q_normed, num_q_heads),
        _apply_partial_rope(k_normed, num_kv_heads),
        gate.to(dtype),
    )
