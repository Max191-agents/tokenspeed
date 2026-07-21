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

import torch


def _rotary_reference(
    value: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    tokens = positions.numel()
    rotary_dim = cos_sin_cache.shape[-1]
    if positions.ndim != 1 or head_size <= 0:
        raise ValueError("RoPE inputs do not match the packed token/head shape")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("RoPE positions must use int32 or int64")
    if positions.device != value.device or cos_sin_cache.device != value.device:
        raise ValueError("RoPE inputs must be on the same device")
    if tokens == 0:
        return value.clone()
    if value.numel() % (tokens * head_size) != 0:
        raise ValueError("RoPE inputs do not match the packed token/head shape")
    if rotary_dim % 2 or rotary_dim > head_size:
        raise ValueError("RoPE rotary_dim must be even and no larger than head_size")

    source = value.view(tokens, -1, head_size)
    output = source.clone()
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64)).float()
    half = rotary_dim // 2
    cos = cos_sin[:, None, :half]
    sin = cos_sin[:, None, half:]
    if is_neox:
        first = source[..., :half].float()
        second = source[..., half:rotary_dim].float()
        output[..., :half] = first * cos - second * sin
        output[..., half:rotary_dim] = second * cos + first * sin
    else:
        first = source[..., :rotary_dim:2].float()
        second = source[..., 1:rotary_dim:2].float()
        output[..., :rotary_dim:2] = first * cos - second * sin
        output[..., 1:rotary_dim:2] = second * cos + first * sin
    return output.reshape_as(value)
