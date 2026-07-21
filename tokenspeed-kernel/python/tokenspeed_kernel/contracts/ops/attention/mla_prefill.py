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

__all__ = ["MLA_PREFILL", "mla_prefill_reference"]


_MLA_PREFILL_BOOL_TRAITS = frozenset({"is_causal", "support_logit_cap", "return_lse"})
_MLA_PREFILL_INT_TRAITS = frozenset({"qk_head_dim", "v_head_dim"})


def _validate_mla_prefill_signatures(
    signatures: frozenset[FormatSignature],
) -> None:
    if not signatures:
        raise ValueError("attention.mla_prefill requires a format signature")
    for signature in signatures:
        if not isinstance(signature, FormatSignature):
            raise TypeError("MLA prefill signatures must be FormatSignature values")
        if {name for name, _ in signature.roles} != {"q", "k", "v"}:
            raise ValueError("MLA prefill signatures require roles 'q', 'k', and 'v'")
        formats = [signature.format_for(role) for role in ("q", "k", "v")]
        if any(
            tensor_format is None
            or tensor_format.format != "dense"
            or tensor_format.scale is not None
            for tensor_format in formats
        ):
            raise ValueError("MLA prefill roles must use unscaled dense formats")


def _validate_mla_prefill_traits(
    traits: Mapping[str, frozenset[Any]],
) -> None:
    allowed = _MLA_PREFILL_BOOL_TRAITS | _MLA_PREFILL_INT_TRAITS
    unknown = set(traits) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown attention.mla_prefill trait(s): {names}")
    for name, values in traits.items():
        if not isinstance(values, frozenset) or not values:
            raise TypeError(f"MLA prefill trait {name!r} must be a non-empty frozenset")
        if name in _MLA_PREFILL_INT_TRAITS:
            if any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(
                    f"MLA prefill trait {name!r} values must be positive integers"
                )
        elif any(type(value) is not bool for value in values):
            raise TypeError(f"MLA prefill trait {name!r} values must be bool")


def _packed_boundaries(
    value: torch.Tensor,
    name: str,
    *,
    token_count: int,
    max_seqlen: int,
) -> tuple[list[int], list[int]]:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() < 2
        or value.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"MLA prefill {name} must be a rank-1 integer tensor")
    boundaries = value.detach().cpu().tolist()
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if (
        boundaries[0] != 0
        or any(length < 0 for length in lengths)
        or boundaries[-1] != token_count
        or max(lengths, default=0) > max_seqlen
    ):
        raise ValueError(f"MLA prefill {name} is inconsistent with packed inputs")
    return boundaries, lengths


def _mla_sequence_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float,
    is_causal: bool,
    logit_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k) * softmax_scale
    if logit_cap > 0.0:
        scores = logit_cap * torch.tanh(scores / logit_cap)
    if is_causal:
        query_positions = torch.arange(q.shape[0], device=q.device)
        query_positions += max(k.shape[0] - q.shape[0], 0)
        key_positions = torch.arange(k.shape[0], device=q.device)
        visible = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~visible[:, None, :], float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.where(
        torch.isfinite(lse)[..., None],
        torch.exp(scores - lse[..., None]),
        torch.zeros_like(scores),
    )
    return torch.einsum("qhk,khd->qhd", weights, v), lse


def mla_prefill_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    seq_lens_kv: torch.Tensor | None = None,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate packed ragged grouped-query MLA attention in PyTorch.

    Query and KV sequences are packed independently along their first
    dimensions. Corresponding entries in the two cumulative-length tensors
    form one request, and attention never crosses request boundaries.

    Args:
        q: Query tensor shaped
            ``[total_q, num_query_heads, qk_head_dim]``.
        k: Key tensor shaped
            ``[total_kv, num_kv_heads, qk_head_dim]``.
            ``num_query_heads`` must be a positive integer multiple of
            ``num_kv_heads`` for grouped-query attention.
        v: Value tensor shaped
            ``[total_kv, num_kv_heads, value_head_dim]``. Its token and head
            axes must match ``k``. Q, K, and V must be on the same device;
            their registered dtypes may differ.
        cu_seqlens_q: Rank-1 int32 or int64 cumulative query boundaries shaped
            ``[batch_size + 1]``. They must start at zero, be nondecreasing,
            and end at ``total_q``.
        cu_seqlens_kv: Rank-1 int32 or int64 cumulative KV boundaries shaped
            ``[batch_size + 1]``. They must start at zero, be nondecreasing,
            end at ``total_kv``, and describe the same batch as
            ``cu_seqlens_q``.
        max_seqlen_q: Nonnegative integer upper bound on every packed query
            length. It may be larger than the actual maximum.
        max_seqlen_kv: Nonnegative integer upper bound on every packed KV
            length. It may be larger than the actual maximum.
        softmax_scale: Finite scalar multiplied into query-key logits before
            the optional cap and softmax.
        seq_lens_kv: Optional rank-1 int32 or int64 tensor shaped
            ``[batch_size]``. When supplied, it must exactly equal the
            differences between adjacent ``cu_seqlens_kv`` entries.
        is_causal: Whether to apply a right-aligned causal mask. For a query
            sequence shorter than its KV sequence, query row ``i`` is aligned
            to KV position ``max(kv_len - q_len, 0) + i``. False allows every
            query to attend the full corresponding KV sequence.
        logit_cap: Finite nonnegative soft cap. A positive value ``c``
            transforms each scaled logit ``x`` to ``c * tanh(x / c)``; zero
            disables the cap.
        return_lse: Whether to also return float32 natural-log LSE values
            shaped ``[total_q, num_query_heads]``.
        out: Optional caller-owned output tensor shaped
            ``[total_q, num_query_heads, value_head_dim]`` on the Q device. It
            is overwritten and returned by identity; its dtype controls the
            final output cast. It must not overlap Q, K, or V storage.

    Returns:
        Attention output shaped
        ``[total_q, num_query_heads, value_head_dim]``. Without ``out``, its
        dtype matches Q except that any FP8 Q dtype produces BF16 output. If
        ``return_lse`` is true, returns ``(output, lse)`` with float32,
        natural-log LSE shaped ``[total_q, num_query_heads]``. An empty KV
        sequence produces zero output and ``-inf`` LSE. Only ``out`` is
        modified.

    Raises:
        TypeError: If Q, K, or V is not a tensor, ``softmax_scale`` is bool, or
            ``is_causal`` or ``return_lse`` is not bool.
        ValueError: If tensor ranks, shapes, heads, dimensions, devices, packed
            boundaries, maximum lengths, scalar values, redundant KV lengths,
            or the output buffer are inconsistent.
    """
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("MLA prefill q, k, and v must be torch.Tensor values")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("MLA prefill q, k, and v must have rank 3")
    if k.shape[:2] != v.shape[:2]:
        raise ValueError("MLA prefill k and v must have matching tokens and heads")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("MLA prefill q and k must have the same head dimension")
    if (
        q.shape[1] <= 0
        or k.shape[1] <= 0
        or q.shape[-1] <= 0
        or v.shape[-1] <= 0
        or q.shape[1] % k.shape[1] != 0
    ):
        raise ValueError("MLA prefill query heads must be a multiple of KV heads")
    if k.device != q.device or v.device != q.device:
        raise ValueError("MLA prefill q, k, and v must be on the same device")
    if type(max_seqlen_q) is not int or max_seqlen_q < 0:
        raise ValueError("MLA prefill max_seqlen_q must be a non-negative integer")
    if type(max_seqlen_kv) is not int or max_seqlen_kv < 0:
        raise ValueError("MLA prefill max_seqlen_kv must be a non-negative integer")
    if isinstance(softmax_scale, bool):
        raise TypeError("MLA prefill softmax_scale must be a real number")
    scale = float(softmax_scale)
    cap = float(logit_cap)
    if not math.isfinite(scale):
        raise ValueError("MLA prefill softmax_scale must be finite")
    if not math.isfinite(cap) or cap < 0.0:
        raise ValueError("MLA prefill logit_cap must be finite and non-negative")
    if type(is_causal) is not bool or type(return_lse) is not bool:
        raise TypeError("MLA prefill is_causal and return_lse must be bool")

    q_boundaries, _ = _packed_boundaries(
        cu_seqlens_q,
        "cu_seqlens_q",
        token_count=q.shape[0],
        max_seqlen=max_seqlen_q,
    )
    kv_boundaries, kv_lengths = _packed_boundaries(
        cu_seqlens_kv,
        "cu_seqlens_kv",
        token_count=k.shape[0],
        max_seqlen=max_seqlen_kv,
    )
    if len(q_boundaries) != len(kv_boundaries):
        raise ValueError("MLA prefill Q and KV boundaries must describe one batch")
    if seq_lens_kv is not None:
        if (
            not isinstance(seq_lens_kv, torch.Tensor)
            or seq_lens_kv.ndim != 1
            or seq_lens_kv.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("MLA prefill seq_lens_kv must be a rank-1 integer tensor")
        supplied_lengths = seq_lens_kv.detach().cpu().tolist()
        if supplied_lengths != kv_lengths:
            raise ValueError(
                "MLA prefill seq_lens_kv must match cu_seqlens_kv differences"
            )

    expected_shape = (q.shape[0], q.shape[1], v.shape[-1])
    if out is not None:
        if not isinstance(out, torch.Tensor) or out.shape != expected_shape:
            raise ValueError(f"MLA prefill out must have shape {expected_shape}")
        if out.device != q.device:
            raise ValueError("MLA prefill out must be on the input device")

    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for q_start, q_end, kv_start, kv_end in zip(
        q_boundaries,
        q_boundaries[1:],
        kv_boundaries,
        kv_boundaries[1:],
    ):
        sequence_output, sequence_lse = _mla_sequence_reference(
            q[q_start:q_end],
            k[kv_start:kv_end],
            v[kv_start:kv_end],
            softmax_scale=scale,
            is_causal=is_causal,
            logit_cap=cap,
        )
        outputs.append(sequence_output)
        lses.append(sequence_lse)

    computed = torch.cat(outputs, dim=0)
    if out is None:
        output_dtype = (
            torch.bfloat16 if str(q.dtype).startswith("torch.float8_") else q.dtype
        )
        out = torch.empty(expected_shape, dtype=output_dtype, device=q.device)
    out.copy_(computed)
    lse = torch.cat(lses, dim=0)
    return (out, lse) if return_lse else out


MLA_PREFILL = OperationSchema(
    family="attention",
    mode="mla_prefill",
    reference=mla_prefill_reference,
    validate_signatures=_validate_mla_prefill_signatures,
    validate_traits=_validate_mla_prefill_traits,
).publish()
