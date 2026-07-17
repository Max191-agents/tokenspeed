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

"""Reference embedding kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_numerics_input_generators import MLARopeInputValues


def _check_rope_dims(*, head_size: int, rotary_dim: int) -> None:
    if head_size <= 0:
        raise ValueError(f"head_size must be positive, got {head_size}")
    if rotary_dim <= 0:
        raise ValueError(f"rotary_dim must be positive, got {rotary_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if rotary_dim > head_size:
        raise ValueError(
            f"rotary_dim must be <= head_size, got {rotary_dim} > {head_size}"
        )


def _apply_rope_to_pe_slice(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    is_neox: bool,
) -> torch.Tensor:
    """Apply RoPE to a PE-only tensor whose last dimension is rotary width."""

    rotary_dim = x.shape[-1]
    _check_rope_dims(head_size=rotary_dim, rotary_dim=rotary_dim)
    if positions.shape != (x.shape[0],):
        raise ValueError(
            f"positions must have shape {(x.shape[0],)}, got {tuple(positions.shape)}"
        )
    if cos_sin_cache.shape[-1] != rotary_dim:
        raise ValueError(
            "cos_sin_cache last dimension must equal rotary width; got "
            f"{cos_sin_cache.shape[-1]} and {rotary_dim}"
        )
    if positions.numel() > 0:
        min_pos = int(positions.min().item())
        max_pos = int(positions.max().item())
        if min_pos < 0 or max_pos >= cos_sin_cache.shape[0]:
            raise ValueError(
                "positions must index cos_sin_cache; got range "
                f"[{min_pos}, {max_pos}] for cache size {cos_sin_cache.shape[0]}"
            )

    original_shape = x.shape
    x_view = x.reshape(x.shape[0], -1, rotary_dim)
    half = rotary_dim // 2
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64))
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(torch.float32)
    sin = sin.unsqueeze(-2).to(torch.float32)
    x_rot = x_view.to(torch.float32)
    if is_neox:
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        rotated = rotated.flatten(-2)
    return rotated.to(x.dtype).reshape(original_shape)


def rope_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query/key after rotary positional embedding."""

    head_size = int(head_size)
    if rotary_dim is None:
        rotary_dim = int(cos_sin_cache.shape[-1])
    rotary_dim = int(rotary_dim)
    _check_rope_dims(head_size=head_size, rotary_dim=rotary_dim)
    if query.shape[0] != key.shape[0]:
        raise ValueError(
            f"query/key token count mismatch: {query.shape[0]} vs {key.shape[0]}"
        )
    if positions.shape != (query.shape[0],):
        raise ValueError(
            "positions must have one entry per token; got "
            f"shape {tuple(positions.shape)} for {query.shape[0]} tokens"
        )
    if query.shape[-1] % head_size != 0 or key.shape[-1] % head_size != 0:
        raise ValueError("query/key last dimensions must be divisible by head_size")
    if cos_sin_cache.shape[-1] != rotary_dim:
        raise ValueError(
            "cos_sin_cache last dimension must equal rotary_dim; got "
            f"{cos_sin_cache.shape[-1]} and {rotary_dim}"
        )

    num_tokens = query.shape[0]
    half = rotary_dim // 2
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64))
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(torch.float32)
    sin = sin.unsqueeze(-2).to(torch.float32)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x_view = x.view(num_tokens, -1, head_size)
        x_rot = x_view[..., :rotary_dim].to(torch.float32)
        x_tail = x_view[..., rotary_dim:]
        if is_neox:
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        else:
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
            rotated = rotated.flatten(-2)
        out = torch.cat((rotated.to(x.dtype), x_tail), dim=-1)
        return out.reshape_as(x)

    return _apply(query), _apply(key)


@dataclass
class MLARopeQuantizeFP8ReferenceValues:
    """Expected outputs for fused RoPE plus FP8 quantization."""

    query: torch.Tensor
    key: torch.Tensor
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor


def mla_rope_quantize_fp8_reference(
    values: MLARopeInputValues,
    *,
    fp8_dtype: torch.dtype,
    quant_scale_q: float,
    quant_scale_kv: float,
    is_neox: bool,
) -> MLARopeQuantizeFP8ReferenceValues:
    """Return expected FP8 query/key outputs for fused MLA RoPE quantization."""

    for name, tensor in (
        ("q_rope", values.q_rope),
        ("k_rope", values.k_rope),
        ("q_nope", values.q_nope),
        ("k_nope", values.k_nope),
    ):
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"{name} dtype must be torch.float16 or torch.bfloat16, "
                f"got {tensor.dtype}"
            )
    if (
        values.q_rope.dtype != values.k_rope.dtype
        or values.q_rope.dtype != values.q_nope.dtype
        or values.q_rope.dtype != values.k_nope.dtype
    ):
        raise ValueError("q/k rope and nope inputs must have the same dtype")
    if values.q_rope.ndim != 3 or values.q_nope.ndim != 3:
        raise ValueError("q_rope and q_nope must be rank-3")
    if values.k_rope.ndim not in (2, 3) or values.k_nope.ndim != values.k_rope.ndim:
        raise ValueError("k_rope and k_nope must both be rank-2 or both be rank-3")
    if values.q_rope.shape[:2] != values.q_nope.shape[:2]:
        raise ValueError(
            "q_rope and q_nope must have matching token/head dimensions; got "
            f"{tuple(values.q_rope.shape)} and {tuple(values.q_nope.shape)}"
        )
    if values.k_rope.shape[:-1] != values.k_nope.shape[:-1]:
        raise ValueError(
            "k_rope and k_nope must have matching leading dimensions; got "
            f"{tuple(values.k_rope.shape)} and {tuple(values.k_nope.shape)}"
        )
    if values.k_rope.shape[0] != values.q_rope.shape[0]:
        raise ValueError(
            "q/k token dimensions must match; got "
            f"{values.q_rope.shape[0]} and {values.k_rope.shape[0]}"
        )
    if values.k_rope.shape[-1] != values.q_rope.shape[-1]:
        raise ValueError(
            "q/k rope dimensions must match; got "
            f"{values.q_rope.shape[-1]} and {values.k_rope.shape[-1]}"
        )
    if values.k_nope.shape[-1] != values.q_nope.shape[-1]:
        raise ValueError(
            "q/k nope dimensions must match; got "
            f"{values.q_nope.shape[-1]} and {values.k_nope.shape[-1]}"
        )
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"fp8_dtype must be an FP8 torch dtype, got {fp8_dtype}")
    if values.positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"positions must be int32 or int64, got {values.positions.dtype}"
        )
    quant_scale_q = float(quant_scale_q)
    quant_scale_kv = float(quant_scale_kv)
    if quant_scale_q <= 0.0:
        raise ValueError(f"quant_scale_q must be positive, got {quant_scale_q}")
    if quant_scale_kv <= 0.0:
        raise ValueError(f"quant_scale_kv must be positive, got {quant_scale_kv}")

    q_rope = _apply_rope_to_pe_slice(
        values.q_rope,
        values.positions,
        values.cos_sin_cache,
        is_neox=is_neox,
    )
    k_rope = _apply_rope_to_pe_slice(
        values.k_rope,
        values.positions,
        values.cos_sin_cache,
        is_neox=is_neox,
    )
    q_nope = (values.q_nope.float() * quant_scale_q).to(fp8_dtype).contiguous()
    q_rope = (q_rope.float() * quant_scale_q).to(fp8_dtype).contiguous()
    k_nope = (values.k_nope.float() * quant_scale_kv).to(fp8_dtype).contiguous()
    k_rope = (k_rope.float() * quant_scale_kv).to(fp8_dtype).contiguous()
    return MLARopeQuantizeFP8ReferenceValues(
        query=torch.cat((q_nope, q_rope), dim=-1).contiguous(),
        key=torch.cat((k_nope, k_rope), dim=-1).contiguous(),
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=k_nope,
        k_rope=k_rope,
    )


@register_kernel(
    "embedding",
    "rope",
    name="torch_embedding_rope",
    solution="reference",
    signatures=format_signatures(
        ("query", "key"), "dense", {torch.float16, torch.bfloat16}
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_embedding_rope(
    *,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    rotary_dim: int | None = None,
    fused_set_kv_buffer_arg: Any = None,
    output_q_rope: torch.Tensor | None = None,
    output_k_rope: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Reference rotary embedding with the same mutating contract as kernels."""

    del enable_pdl
    q_ref, k_ref = rope_reference(
        query,
        key,
        positions,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        rotary_dim=rotary_dim,
    )
    q_target = output_q_rope if output_q_rope is not None else query
    k_target = output_k_rope if output_k_rope is not None else key
    q_target.copy_(q_ref)
    k_target.copy_(k_ref)

    if fused_set_kv_buffer_arg is not None:
        cache_loc = fused_set_kv_buffer_arg.cache_loc.to(torch.int64)
        fused_set_kv_buffer_arg.k_buffer.index_copy_(0, cache_loc, k_target)
        fused_set_kv_buffer_arg.v_buffer.index_copy_(
            0,
            cache_loc,
            fused_set_kv_buffer_arg.value.reshape_as(k_target),
        )
