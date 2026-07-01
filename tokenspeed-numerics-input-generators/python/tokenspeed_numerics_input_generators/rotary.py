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

"""Shared rotary positional embedding helpers."""

from __future__ import annotations

import torch
from tokenspeed_numerics_input_generators.core import DeviceLike, _resolve_device

__all__ = ["build_rope_cos_sin_cache"]


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


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
