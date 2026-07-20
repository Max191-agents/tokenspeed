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
from typing import Any

import torch
from tokenspeed_kernel.contracts.ops.embedding.utils import _rotary_reference
from tokenspeed_kernel.operation import OperationSchema
from tokenspeed_kernel.signature import FormatSignature

__all__ = ["ROPE_MLA", "rope_mla_reference"]


_BOOL_TRAITS = frozenset({"is_neox", "has_scale_q_tensor", "has_scale_kv_tensor"})


def _validate_signatures(signatures: frozenset[FormatSignature]) -> None:
    roles = {"q_rope", "k_rope", "q_nope", "k_nope"}
    if not signatures:
        raise ValueError("embedding.rope_mla requires a format signature")
    for signature in signatures:
        if not isinstance(signature, FormatSignature):
            raise TypeError("embedding.rope_mla signatures must be FormatSignature")
        if {name for name, _ in signature.roles} != roles:
            raise ValueError(
                "embedding.rope_mla signatures require q_rope, k_rope, "
                "q_nope, and k_nope roles"
            )
        formats = [signature.format_for(role) for role in roles]
        if any(
            tensor_format is None
            or tensor_format.format != "dense"
            or tensor_format.scale is not None
            for tensor_format in formats
        ):
            raise ValueError("embedding.rope_mla roles must use unscaled dense formats")


def _validate_traits(traits: Mapping[str, frozenset[Any]]) -> None:
    unknown = set(traits) - ({"quantize_dtype"} | _BOOL_TRAITS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown embedding.rope_mla trait(s): {names}")
    for name, values in traits.items():
        if not isinstance(values, frozenset) or not values:
            raise TypeError(
                f"embedding.rope_mla trait {name!r} must be a non-empty frozenset"
            )
        if name == "quantize_dtype":
            if any(
                not isinstance(value, torch.dtype)
                or not str(value).startswith("torch.float8_")
                for value in values
            ):
                raise TypeError(
                    "embedding.rope_mla quantize_dtype values must be FP8 dtypes"
                )
        elif any(type(value) is not bool for value in values):
            raise TypeError(f"embedding.rope_mla trait {name!r} values must be bool")


def rope_mla_reference(
    *,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    is_neox: bool = True,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
) -> None:
    """Apply MLA RoPE and scale/cast each query and key output component."""
    del enable_pdl
    if q_rope.ndim != 3 or k_rope.ndim != 3:
        raise ValueError("MLA RoPE inputs must have rank 3")
    if q_rope.shape[0] != positions.numel() or k_rope.shape[0] != positions.numel():
        raise ValueError("MLA RoPE inputs must match the position count")
    if q_rope.shape[-1] != k_rope.shape[-1]:
        raise ValueError("MLA query and key RoPE dimensions must match")
    for scale in (quant_scale_q, quant_scale_kv):
        if isinstance(scale, torch.Tensor) and scale.numel() != 1:
            raise ValueError("MLA RoPE tensor scales must contain one value")

    head_size = q_rope.shape[-1]
    rotated_q = _rotary_reference(q_rope, positions, head_size, cos_sin_cache, is_neox)
    rotated_k = _rotary_reference(k_rope, positions, head_size, cos_sin_cache, is_neox)
    q_rope_out.copy_(rotated_q.float() * quant_scale_q)
    k_rope_out.copy_(rotated_k.float() * quant_scale_kv)
    q_nope_out.copy_(q_nope.float() * quant_scale_q)
    k_nope_out.copy_(k_nope.float() * quant_scale_kv)


ROPE_MLA = OperationSchema(
    family="embedding",
    mode="rope_mla",
    reference=rope_mla_reference,
    validate_signatures=_validate_signatures,
    validate_traits=_validate_traits,
).publish()
