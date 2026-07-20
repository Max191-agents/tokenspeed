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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.contracts.ops.embedding.utils import _rotary_reference
from tokenspeed_kernel.operation import OperationSchema
from tokenspeed_kernel.signature import FormatSignature

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.embedding import (
        FusedMLASetKVBufferArg,
        FusedSetKVBufferArg,
    )

__all__ = ["ROPE", "rope_reference"]


_ROPE_BOOL_TRAITS = frozenset(
    {
        "partial_rotary",
        "is_neox",
        "has_fused_kv",
        "has_fused_mla_kv",
        "has_q_out",
        "has_k_out",
    }
)


def _validate_rope_signatures(signatures: frozenset[FormatSignature]) -> None:
    if not signatures:
        raise ValueError("embedding.rope requires a format signature")
    for signature in signatures:
        if not isinstance(signature, FormatSignature):
            raise TypeError("embedding.rope signatures must be FormatSignature")
        if {name for name, _ in signature.roles} != {"q", "k"}:
            raise ValueError("embedding.rope signatures require roles 'q' and 'k'")
        formats = [signature.format_for(role) for role in ("q", "k")]
        if any(
            tensor_format is None
            or tensor_format.format != "dense"
            or tensor_format.scale is not None
            for tensor_format in formats
        ):
            raise ValueError("embedding.rope roles must use unscaled dense formats")


def _validate_rope_traits(traits: Mapping[str, frozenset[Any]]) -> None:
    unknown = set(traits) - ({"head_size"} | _ROPE_BOOL_TRAITS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown embedding.rope trait(s): {names}")
    for name, values in traits.items():
        if not isinstance(values, frozenset) or not values:
            raise TypeError(
                f"embedding.rope trait {name!r} must be a non-empty frozenset"
            )
        if name == "head_size":
            if any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(
                    "embedding.rope head_size values must be positive integers"
                )
        elif any(type(value) is not bool for value in values):
            raise TypeError(f"embedding.rope trait {name!r} values must be bool")


def rope_reference(
    *,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fused_set_kv_buffer_arg: FusedSetKVBufferArg | None = None,
    fused_mla_set_kv_buffer_arg: FusedMLASetKVBufferArg | None = None,
    q_rope_out: torch.Tensor | None = None,
    k_rope_out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Apply RoPE and its optional fused KV-cache side effects."""
    del enable_pdl
    if fused_set_kv_buffer_arg is not None and fused_mla_set_kv_buffer_arg is not None:
        raise ValueError("standard and MLA fused KV writes are mutually exclusive")
    if fused_mla_set_kv_buffer_arg is not None and k_rope_out is not None:
        raise ValueError("MLA fused KV write stores rotated K directly in the cache")

    rotated_q = _rotary_reference(q, positions, head_size, cos_sin_cache, is_neox)
    rotated_k = _rotary_reference(k, positions, head_size, cos_sin_cache, is_neox)
    q_target = q if q_rope_out is None else q_rope_out
    q_target.copy_(rotated_q)
    if positions.numel() == 0:
        if fused_mla_set_kv_buffer_arg is None:
            (k if k_rope_out is None else k_rope_out).copy_(rotated_k)
        return

    if fused_mla_set_kv_buffer_arg is not None:
        k_nope = fused_mla_set_kv_buffer_arg.k_nope
        kv_buffer = fused_mla_set_kv_buffer_arg.kv_buffer
        cache_loc = fused_mla_set_kv_buffer_arg.cache_loc
        if cache_loc.dtype not in (torch.int32, torch.int64):
            raise TypeError("MLA fused KV cache locations must use int32 or int64")
        cache_loc = cache_loc.to(torch.int64)
        combined = torch.cat(
            (k_nope, rotated_k.view(positions.numel(), -1, head_size)), dim=-1
        )
        if combined.shape[1] != 1:
            raise ValueError("MLA fused KV write requires one key head")
        kv_buffer.index_copy_(0, cache_loc, combined[:, 0])
        return

    k_target = k if k_rope_out is None else k_rope_out
    k_target.copy_(rotated_k)
    if fused_set_kv_buffer_arg is None:
        return
    if (
        fused_set_kv_buffer_arg.k_scale is not None
        or fused_set_kv_buffer_arg.v_scale is not None
    ):
        raise ValueError("fused RoPE KV writes do not support scales")
    tokens = positions.numel()
    num_k_heads = k.numel() // (tokens * head_size)
    cache_loc = fused_set_kv_buffer_arg.cache_loc
    if cache_loc.dtype not in (torch.int32, torch.int64):
        raise TypeError("fused KV cache locations must use int32 or int64")
    cache_loc = cache_loc.to(torch.int64)
    k_buffer = fused_set_kv_buffer_arg.k_buffer.view(-1, num_k_heads, head_size)
    v_buffer = fused_set_kv_buffer_arg.v_buffer.view(-1, num_k_heads, head_size)
    k_buffer.index_copy_(0, cache_loc, rotated_k.view(tokens, num_k_heads, head_size))
    v_buffer.index_copy_(
        0,
        cache_loc,
        fused_set_kv_buffer_arg.value.view(tokens, num_k_heads, head_size),
    )


ROPE = OperationSchema(
    family="embedding",
    mode="rope",
    reference=rope_reference,
    validate_signatures=_validate_rope_signatures,
    validate_traits=_validate_rope_traits,
).publish()
