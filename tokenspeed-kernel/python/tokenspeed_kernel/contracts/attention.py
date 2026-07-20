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

__all__ = [
    "ATTN_MERGE_STATE",
    "MLA_PREFILL",
    "attn_merge_state_reference",
    "mla_prefill_reference",
]


_MLA_PREFILL_BOOL_TRAITS = frozenset({"is_causal", "support_logit_cap", "return_lse"})
_MLA_PREFILL_INT_TRAITS = frozenset({"qk_head_dim", "v_head_dim"})


def _validate_merge_signatures(signatures: frozenset[FormatSignature]) -> None:
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


def _validate_merge_traits(traits: Mapping[str, frozenset[Any]]) -> None:
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
    """Evaluate packed ragged grouped-query MLA prefill in PyTorch."""
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
    """Merge two partial attention outputs using their log-sum-exp weights."""
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
    validate_signatures=_validate_merge_signatures,
    validate_traits=_validate_merge_traits,
).publish()

MLA_PREFILL = OperationSchema(
    family="attention",
    mode="mla_prefill",
    reference=mla_prefill_reference,
    validate_signatures=_validate_mla_prefill_signatures,
    validate_traits=_validate_mla_prefill_traits,
).publish()
