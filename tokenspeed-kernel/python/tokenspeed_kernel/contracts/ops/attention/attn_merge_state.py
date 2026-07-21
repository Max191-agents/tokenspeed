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

import math
from collections.abc import Mapping
from typing import Any

import torch
from tokenspeed_kernel.operation import OperationSchema
from tokenspeed_kernel.signature import FormatSignature

__all__ = ["ATTN_MERGE_STATE", "attn_merge_state_reference"]


def _validate_signatures(signatures: frozenset[FormatSignature]) -> None:
    if not signatures:
        raise ValueError("attention.attn_merge_state requires a format signature")
    for signature in signatures:
        if not isinstance(signature, FormatSignature):
            raise TypeError("attention state merge signatures must be FormatSignature")
        if {name for name, _ in signature.roles} != {"out_a", "out_b"}:
            raise ValueError(
                "attention state merge signatures require roles 'out_a' and 'out_b'"
            )
        formats = [signature.format_for(role) for role in ("out_a", "out_b")]
        if any(
            tensor_format is None
            or tensor_format.format != "dense"
            or tensor_format.scale is not None
            for tensor_format in formats
        ):
            raise ValueError(
                "attention state merge roles must use unscaled dense formats"
            )


def _validate_traits(traits: Mapping[str, frozenset[Any]]) -> None:
    unknown = set(traits) - {"head_dim"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown attention.attn_merge_state trait(s): {names}")
    if "head_dim" not in traits:
        return
    values = traits["head_dim"]
    if not isinstance(values, frozenset) or not values:
        raise TypeError("attention state merge head_dim must be a non-empty frozenset")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(
            "attention state merge head_dim values must be positive integers"
        )


def attn_merge_state_reference(
    *,
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float = math.log2(math.e),
    inplace: bool = False,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two normalized partial-attention states for the same queries.

    Each state represents attention over a separate KV partition. The outputs
    are weighted by the partition normalizers encoded by their log-sum-exp
    (LSE) tensors, producing the state for the union of both partitions. The
    merge arithmetic is evaluated in float32.

    Args:
        out_a: Contiguous first partial output shaped
            ``[total_q, num_heads, head_dim]``. Its device and dtype determine
            those of the merged output. It is modified only when ``inplace`` is
            true.
        lse_a: Contiguous float32 LSE tensor for ``out_a``, shaped
            ``[total_q, num_heads]`` on the same device. It is expressed in the
            logarithm base described by ``lse_scale_log2`` and is modified only
            when ``inplace`` is true. Each value must be finite or ``-inf``.
        out_b: Contiguous second partial output with the same shape and device
            as ``out_a``. Its dtype may differ; merged values are cast to
            ``out_a.dtype``. This tensor is never modified.
        lse_b: Contiguous float32 LSE tensor for ``out_b`` with the same shape,
            device, value domain, and logarithm base as ``lse_a``. It is never
            modified.
        lse_scale_log2: Multiplier that converts the input LSE values to base-2
            logarithms. The default ``log2(e)`` means the LSE tensors use
            natural logarithms; use ``1.0`` for base-2 LSE. The value must be
            finite and nonzero.
        inplace: If false, allocate merged tensors and leave all inputs
            unchanged. If true, write into ``out_a`` and ``lse_a`` and return
            those same objects.
        enable_pdl: Backend launch hint for Programmatic Dependent Launch. It
            has no numerical effect and is ignored by this reference.

    Returns:
        ``(merged_output, merged_lse)`` with shapes matching ``out_a`` and
        ``lse_a``. The output uses ``out_a.dtype`` and the LSE uses float32.
        The merged LSE uses the same logarithm base as both input LSE tensors.
        With ``inplace=True``, the returned tensors alias ``out_a`` and
        ``lse_a``.

    Raises:
        TypeError: If either LSE tensor is not float32.
        ValueError: If output or LSE shapes disagree, an input is
            noncontiguous, or the inputs are on different devices.

    The two states must describe identical query rows and compatible,
    non-overlapping KV partitions. For each row and head, at least one
    partition must have a finite LSE.
    """
    del enable_pdl
    if out_a.ndim != 3 or out_b.shape != out_a.shape:
        raise ValueError("attention state outputs must have the same rank-3 shape")
    if lse_a.shape != out_a.shape[:2] or lse_b.shape != out_a.shape[:2]:
        raise ValueError("attention state LSE tensors must match the output rows")
    if lse_a.dtype is not torch.float32 or lse_b.dtype is not torch.float32:
        raise TypeError("attention state LSE tensors must use float32")
    tensors = (out_a, lse_a, out_b, lse_b)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("attention state inputs must be contiguous")
    if any(tensor.device != out_a.device for tensor in tensors[1:]):
        raise ValueError("attention state inputs must be on the same device")

    scale = float(lse_scale_log2)
    lse_a_log2 = lse_a.float() * scale
    lse_b_log2 = lse_b.float() * scale
    lse_max_log2 = torch.maximum(lse_a_log2, lse_b_log2)
    weight_a = torch.exp2(lse_a_log2 - lse_max_log2)
    weight_b = torch.exp2(lse_b_log2 - lse_max_log2)
    denominator = weight_a + weight_b
    output = (
        out_a.float() * weight_a[..., None] + out_b.float() * weight_b[..., None]
    ) / denominator[..., None]
    lse = (lse_max_log2 + torch.log2(denominator)) / scale

    if inplace:
        out_a.copy_(output)
        lse_a.copy_(lse)
        return out_a, lse_a
    return output.to(out_a.dtype), lse.to(lse_a.dtype)


ATTN_MERGE_STATE = OperationSchema(
    family="attention",
    mode="attn_merge_state",
    reference=attn_merge_state_reference,
    validate_signatures=_validate_signatures,
    validate_traits=_validate_traits,
).publish()
