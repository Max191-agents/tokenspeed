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

"""Expected-value helpers for MoE kernel tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_numerics_input_generators import (
    MoeAlignBlockSizeInputValues,
    MoEBiasedGroupedTopKInputValues,
    MoEDeepSeekV4MegaMoEStagingInputValues,
    MoEFinalizeFuseSharedInputValues,
    MoESoftmaxTopKRoutingInputValues,
    MoESoftplusSqrtTopKRoutingInputValues,
)


@dataclass
class MoeAlignBlockSizeReferenceValues:
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_pad: torch.Tensor


@dataclass
class MoERoutingReferenceValues:
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class MoEGroupedRoutingReferenceValues:
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


@dataclass
class MoEStagingReferenceValues:
    x_fp8: torch.Tensor
    x_sf: torch.Tensor
    topk_idx_out: torch.Tensor
    topk_weights_out: torch.Tensor


def moe_align_block_size_reference(
    values: MoeAlignBlockSizeInputValues,
) -> MoeAlignBlockSizeReferenceValues:
    device = values.topk_ids.device
    pad_id = values.topk_ids.numel()
    sorted_chunks: list[torch.Tensor] = []
    expert_blocks: list[int] = []
    flat_ids = values.topk_ids.reshape(-1)
    flat_positions = torch.arange(pad_id, dtype=torch.int32, device=device)

    for expert in range(values.num_experts):
        selected = flat_positions[flat_ids == expert]
        count = int(selected.numel())
        padded_count = (
            (count + values.block_size - 1) // values.block_size * values.block_size
        )
        if padded_count == 0:
            continue
        if padded_count > count:
            selected = torch.cat(
                (
                    selected.to(torch.int32),
                    torch.full(
                        (padded_count - count,),
                        pad_id,
                        dtype=torch.int32,
                        device=device,
                    ),
                )
            )
        else:
            selected = selected.to(torch.int32)
        sorted_chunks.append(selected)
        expert_blocks.extend([expert] * (padded_count // values.block_size))

    sorted_token_ids = (
        torch.cat(sorted_chunks)
        if sorted_chunks
        else torch.empty((0,), dtype=torch.int32, device=device)
    )
    return MoeAlignBlockSizeReferenceValues(
        sorted_token_ids=sorted_token_ids,
        expert_ids=torch.tensor(expert_blocks, dtype=torch.int32, device=device),
        num_tokens_post_pad=torch.tensor(
            [sorted_token_ids.numel()],
            dtype=torch.int32,
            device=device,
        ),
    )


def moe_softmax_topk_routing_reference(
    values: MoESoftmaxTopKRoutingInputValues,
) -> MoERoutingReferenceValues:
    probs = torch.softmax(values.logits.float(), dim=-1)
    selection_scores = probs + values.correction_bias.reshape(1, -1)
    top_k = values.topk_indices.shape[1]
    out_indices = torch.empty_like(values.topk_indices)
    out_weights = torch.empty_like(values.topk_weights)
    scores_cpu = selection_scores.detach().cpu()
    probs_cpu = probs.detach().cpu()
    for row_idx in range(scores_cpu.shape[0]):
        ordered = sorted(
            range(scores_cpu.shape[1]),
            key=lambda expert: (-float(scores_cpu[row_idx, expert]), expert),
        )[:top_k]
        selected_probs = torch.tensor(
            [float(probs_cpu[row_idx, expert]) for expert in ordered],
            dtype=torch.float32,
            device=values.logits.device,
        )
        if values.renormalize:
            selected_probs = selected_probs / (selected_probs.sum() + 1.0e-10)
        out_indices[row_idx] = torch.tensor(
            [-1 if expert >= values.num_experts_real else expert for expert in ordered],
            dtype=values.topk_indices.dtype,
            device=values.logits.device,
        )
        out_weights[row_idx] = selected_probs * float(values.scaling_factor)
    return MoERoutingReferenceValues(out_indices, out_weights)


def moe_biased_grouped_topk_reference(
    values: MoEBiasedGroupedTopKInputValues,
) -> MoEGroupedRoutingReferenceValues:
    scores = values.gating_output.float().sigmoid()
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // values.num_expert_groups
    selection_scores = scores + values.correction_bias.reshape(1, -1)
    grouped_scores = selection_scores.reshape(
        num_tokens,
        values.num_expert_groups,
        experts_per_group,
    )
    group_scores = grouped_scores.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = torch.topk(
        group_scores,
        k=values.top_k_groups,
        dim=-1,
        sorted=False,
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, values.num_expert_groups, experts_per_group)
        .reshape(num_tokens, num_experts)
    )
    candidate_scores = selection_scores.masked_fill(~expert_mask, float("-inf"))
    topk_ids = torch.topk(
        candidate_scores,
        k=values.top_k,
        dim=-1,
        sorted=False,
    ).indices.to(torch.int32)
    topk_weights = scores.gather(1, topk_ids.to(torch.long)).to(torch.float32)

    if values.renormalize:
        weight_sum = topk_weights.sum(dim=-1, keepdim=True)
        denom = torch.where(weight_sum != 0.0, weight_sum, torch.ones_like(weight_sum))
        topk_weights = topk_weights / denom * float(values.routed_scaling_factor)
    if values.logical_to_physical_map is not None:
        topk_ids = values.logical_to_physical_map[topk_ids.to(torch.long)].to(
            torch.int32
        )
    if values.num_token_non_padded is not None:
        valid_tokens = int(values.num_token_non_padded.detach().cpu().item())
        if valid_tokens < num_tokens:
            topk_ids = topk_ids.clone()
            topk_ids[valid_tokens:, :] = -1
    return MoEGroupedRoutingReferenceValues(topk_weights, topk_ids)


def moe_softplus_sqrt_topk_routing_reference(
    values: MoESoftplusSqrtTopKRoutingInputValues,
) -> MoERoutingReferenceValues:
    transformed = torch.sqrt(torch.nn.functional.softplus(values.logits.float()))
    if values.input_ids is not None:
        assert values.hash_indices_table is not None
        topk_indices = values.hash_indices_table[values.input_ids.to(torch.long)].to(
            torch.int32
        )
    else:
        assert values.correction_bias is not None
        selection_scores = transformed + values.correction_bias.reshape(1, -1)
        scores_cpu = selection_scores.detach().cpu()
        selected = [
            torch.tensor(
                sorted(
                    range(scores_cpu.shape[1]),
                    key=lambda expert: (-float(scores_cpu[row_idx, expert]), expert),
                )[: values.topk_indices.shape[1]],
                dtype=torch.int32,
                device=values.logits.device,
            )
            for row_idx in range(scores_cpu.shape[0])
        ]
        topk_indices = (
            torch.stack(selected) if selected else torch.empty_like(values.topk_indices)
        )

    topk_weights = transformed.gather(1, topk_indices.to(torch.long))
    denom = topk_weights.sum(dim=-1, keepdim=True)
    denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
    topk_weights = (topk_weights / denom * float(values.routed_scaling_factor)).to(
        torch.float32
    )
    return MoERoutingReferenceValues(topk_indices.to(torch.int32), topk_weights)


def moe_deepseek_v4_mega_moe_staging_reference(
    values: MoEDeepSeekV4MegaMoEStagingInputValues,
) -> MoEStagingReferenceValues:
    hidden = values.hidden_states.float()
    num_tokens, hidden_size = hidden.shape
    grouped = hidden.reshape(num_tokens, hidden_size // 128, 128)
    grouped_abs = grouped.abs().reshape(num_tokens, hidden_size // 128, 4, 32)
    scale = grouped_abs.amax(dim=-1).clamp_min(1.0e-4) / 448.0
    scale_bits = scale.contiguous().view(torch.int32)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        torch.int32
    )
    scale_exp = scale_exp.clamp(1, 254)
    rounded_scale = torch.pow(
        torch.full_like(scale, 2.0),
        scale_exp.float() - 127.0,
    )
    scaled = grouped.reshape(num_tokens, hidden_size // 128, 4, 32) / (
        rounded_scale.unsqueeze(-1)
    )
    shifts = torch.arange(
        0,
        32,
        8,
        dtype=torch.int32,
        device=values.hidden_states.device,
    )
    return MoEStagingReferenceValues(
        x_fp8=scaled.reshape_as(hidden).to(torch.float8_e4m3fn),
        x_sf=((scale_exp.to(torch.int32) << shifts).sum(dim=-1)).to(torch.int32),
        topk_idx_out=values.topk_ids.clone(),
        topk_weights_out=values.topk_weights.clone(),
    )


def moe_finalize_fuse_shared_reference(
    values: MoEFinalizeFuseSharedInputValues,
) -> torch.Tensor:
    num_tokens, top_k = values.expert_weights.shape
    hidden_size = (
        values.shared_output.shape[1]
        if values.shared_output is not None
        else values.gemm2_out.shape[1]
    )
    output = torch.zeros(
        (num_tokens, hidden_size),
        dtype=torch.float32,
        device=values.gemm2_out.device,
    )
    indices = values.expanded_idx_to_permuted_idx.reshape(num_tokens, top_k)
    for token_idx in range(num_tokens):
        for topk_idx in range(top_k):
            permuted_idx = int(indices[token_idx, topk_idx].item())
            if permuted_idx != -1:
                output[token_idx] += (
                    values.expert_weights[token_idx, topk_idx].float()
                    * values.gemm2_out[permuted_idx, :hidden_size].float()
                )
    if values.shared_output is not None:
        output += values.shared_output.float()
    return output.to(torch.bfloat16)
