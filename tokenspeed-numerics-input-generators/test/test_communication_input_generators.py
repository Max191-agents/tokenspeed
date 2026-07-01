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

import pytest
import torch
from tokenspeed_numerics_input_generators import (
    AllGatherDualRMSNormInputConfig,
    AllGatherDualRMSNormInputs,
    AllGatherInputConfig,
    AllGatherInputs,
    AllReduceInputConfig,
    AllReduceInputs,
    AllReduceResidualRMSNormInputConfig,
    AllReduceResidualRMSNormInputs,
    DPSamplingInputConfig,
    DPSamplingInputs,
    DPSamplingInputValues,
    ExpertParallelRoutingInputConfig,
    ExpertParallelRoutingInputs,
    ExpertParallelRoutingInputValues,
    MiniMaxAllReduceQKRMSNormInputConfig,
    MiniMaxAllReduceQKRMSNormInputs,
    ReduceScatterInputConfig,
    ReduceScatterInputs,
    ReduceScatterResidualRMSNormInputConfig,
    ReduceScatterResidualRMSNormInputs,
    all_gather_dual_rmsnorm_reference,
    all_gather_reference,
    all_reduce_residual_rmsnorm_reference,
    all_reduce_sum_reference,
    dp_sampling_gather_reference,
    dp_sampling_reference,
    dp_sampling_swap_reference,
    expert_parallel_routing_reference,
    minimax_allreduce_qk_rmsnorm_reference,
    reduce_scatter_residual_rmsnorm_reference,
    reduce_scatter_sum_reference,
)


def test_all_reduce_inputs_generate_values_and_reference() -> None:
    values = AllReduceInputs(
        AllReduceInputConfig(
            world_size=3,
            shape=(4, 8),
            dtype=torch.float32,
        )
    ).generate(seed=11, device="cpu")

    assert len(values.rank_inputs) == 3
    assert all(rank_input.shape == (4, 8) for rank_input in values.rank_inputs)

    expected = all_reduce_sum_reference(values)
    manual = sum(rank_input for rank_input in values.rank_inputs)
    torch.testing.assert_close(expected, manual)


def test_all_reduce_residual_rmsnorm_inputs_generate_values_and_reference() -> None:
    config = AllReduceResidualRMSNormInputConfig(
        world_size=3,
        num_tokens=4,
        hidden_size=8,
        dtype=torch.float32,
        residual_dtype=torch.float32,
        weight_dtype=torch.float32,
        eps=1e-5,
    )
    values = AllReduceResidualRMSNormInputs(config).generate(seed=111, device="cpu")

    assert len(values.rank_inputs) == 3
    assert len(values.residuals) == 3
    assert all(rank_input.shape == (4, 8) for rank_input in values.rank_inputs)
    assert all(residual.shape == (4, 8) for residual in values.residuals)
    assert values.weight.shape == (8,)
    assert values.eps == pytest.approx(1e-5)

    refs = all_reduce_residual_rmsnorm_reference(values)
    reduced = torch.zeros_like(values.rank_inputs[0], dtype=torch.float32)
    for rank_input in values.rank_inputs:
        reduced = reduced + rank_input.float()
    manual_residual = reduced + values.residuals[1].float()
    manual_norm = manual_residual * torch.rsqrt(
        manual_residual.pow(2).mean(dim=-1, keepdim=True) + values.eps
    )
    manual_norm = manual_norm * values.weight.float()

    torch.testing.assert_close(refs.residual_outputs[1], manual_residual)
    torch.testing.assert_close(refs.norm_outputs[1], manual_norm)


def test_all_reduce_residual_rmsnorm_supports_bfloat16_inputs() -> None:
    values = AllReduceResidualRMSNormInputs(
        AllReduceResidualRMSNormInputConfig(
            world_size=2,
            num_tokens=3,
            hidden_size=16,
            dtype=torch.bfloat16,
            residual_dtype=torch.bfloat16,
            weight_dtype=torch.float32,
        )
    ).generate(seed=112, device="cpu")

    refs = all_reduce_residual_rmsnorm_reference(values)

    assert all(rank_input.dtype == torch.bfloat16 for rank_input in values.rank_inputs)
    assert all(residual.dtype == torch.bfloat16 for residual in values.residuals)
    assert values.weight.dtype == torch.float32
    assert [output.shape for output in refs.norm_outputs] == [(3, 16), (3, 16)]
    assert [output.dtype for output in refs.norm_outputs] == [
        torch.float32,
        torch.float32,
    ]


def test_expert_parallel_routing_inputs_generate_values_and_reference() -> None:
    values = ExpertParallelRoutingInputs(
        ExpertParallelRoutingInputConfig(
            world_size=3,
            total_tokens=11,
            hidden_size=5,
            num_experts=6,
            top_k=3,
            max_tokens_per_rank=5,
            dtype=torch.float32,
        )
    ).generate(seed=121, metadata_seed=122, device="cpu")

    assert len(values.rank_hidden_states) == 3
    assert len(values.topk_ids) == 3
    assert sum(values.tokens_per_rank) == 11
    assert max(values.tokens_per_rank) <= 5
    for rank, tokens in enumerate(values.tokens_per_rank):
        assert values.rank_hidden_states[rank].shape == (tokens, 5)
        assert values.topk_ids[rank].shape == (tokens, 3)
        assert values.topk_weights[rank].shape == (tokens, 3)
        assert values.expert_outputs[rank].shape == (tokens, 3, 5)
        if tokens:
            assert torch.all(values.topk_ids[rank] >= 0)
            assert torch.all(values.topk_ids[rank] < 6)
            assert torch.all(
                values.topk_ids[rank].sort(dim=-1).values[:, 1:]
                != values.topk_ids[rank].sort(dim=-1).values[:, :-1]
            )
            torch.testing.assert_close(
                values.topk_weights[rank].sum(dim=-1),
                torch.ones(tokens),
            )

    refs = expert_parallel_routing_reference(values)
    experts_per_rank = values.num_experts // len(values.rank_hidden_states)

    for rank in range(3):
        expected_recv_rows = 0
        expected_counts = torch.zeros((experts_per_rank,), dtype=torch.int32)
        for source_rank, ids in enumerate(values.topk_ids):
            del source_rank
            for token in range(ids.shape[0]):
                target_ranks = set()
                for expert in ids[token].tolist():
                    target_rank = int(expert) // experts_per_rank
                    if target_rank == rank:
                        expected_counts[int(expert) % experts_per_rank] += 1
                    target_ranks.add(target_rank)
                expected_recv_rows += int(rank in target_ranks)
        assert refs.recv_hidden_states[rank].shape == (expected_recv_rows, 5)
        torch.testing.assert_close(
            refs.num_recv_tokens_per_expert[rank],
            expected_counts,
            atol=0,
            rtol=0,
        )

    for rank, tokens in enumerate(values.tokens_per_rank):
        manual = torch.zeros((tokens, 5), dtype=torch.float32)
        for token in range(tokens):
            for slot in range(values.topk_ids[rank].shape[1]):
                manual[token] += (
                    values.expert_outputs[rank][token, slot].float()
                    * values.topk_weights[rank][token, slot]
                )
        torch.testing.assert_close(refs.combined_outputs[rank], manual)


def test_expert_parallel_routing_metadata_seed_controls_routing_layout() -> None:
    generator = ExpertParallelRoutingInputs(
        ExpertParallelRoutingInputConfig(
            world_size=2,
            total_tokens=7,
            hidden_size=4,
            num_experts=4,
            top_k=2,
            max_tokens_per_rank=4,
            dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=123, metadata_seed=77, device="cpu")
    values2 = generator.generate(seed=124, metadata_seed=77, device="cpu")

    assert values1.tokens_per_rank == values2.tokens_per_rank
    assert all(
        torch.equal(ids1, ids2)
        for ids1, ids2 in zip(values1.topk_ids, values2.topk_ids, strict=True)
    )
    assert not torch.equal(values1.rank_hidden_states[0], values2.rank_hidden_states[0])
    assert not torch.equal(values1.topk_weights[0], values2.topk_weights[0])


def test_dp_sampling_inputs_generate_values_and_reference() -> None:
    config = DPSamplingInputConfig(
        world_size=3,
        pad_batch_size=6,
        num_tokens_per_request=4,
        vocab_size=12,
        logits_dtype=torch.float32,
    )
    values = DPSamplingInputs(config).generate(
        seed=131,
        metadata_seed=132,
        device="cpu",
    )

    assert len(values.local_logits) == 3
    assert len(values.predict_local) == 3
    assert all(logits.shape == (24, 4) for logits in values.local_logits)
    assert all(predict.shape == (2, 4) for predict in values.predict_local)
    assert all(index.shape == (2, 4) for index in values.accept_index_local)
    assert all(length.shape == (2,) for length in values.accept_length_local)
    for rank in range(3):
        assert values.local_logits[rank].dtype == torch.float32
        assert values.predict_local[rank].dtype == torch.int32
        assert values.accept_index_local[rank].dtype == torch.int32
        assert values.accept_length_local[rank].dtype == torch.int32
        assert torch.all(values.predict_local[rank] >= 0)
        assert torch.all(values.predict_local[rank] < 12)
        assert torch.all(values.accept_index_local[rank] >= -1)
        assert torch.all(values.accept_index_local[rank] < 8)
        torch.testing.assert_close(
            (values.accept_index_local[rank] >= 0).sum(dim=1).to(torch.int32),
            values.accept_length_local[rank],
            atol=0,
            rtol=0,
        )

    refs = dp_sampling_reference(values)

    assert [logits.shape for logits in refs.swapped_logits] == [
        (8, 12),
        (8, 12),
        (8, 12),
    ]
    assert refs.predict.shape == (6, 4)
    assert refs.accept_index.shape == (6, 4)
    assert refs.accept_length.shape == (6,)


def test_dp_sampling_metadata_seed_controls_verify_outputs_only() -> None:
    generator = DPSamplingInputs(
        DPSamplingInputConfig(
            world_size=2,
            pad_batch_size=4,
            num_tokens_per_request=3,
            vocab_size=10,
            logits_dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=133, metadata_seed=134, device="cpu")
    values2 = generator.generate(seed=135, metadata_seed=134, device="cpu")

    assert not torch.equal(values1.local_logits[0], values2.local_logits[0])
    for rank in range(2):
        torch.testing.assert_close(
            values1.predict_local[rank],
            values2.predict_local[rank],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            values1.accept_index_local[rank],
            values2.accept_index_local[rank],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            values1.accept_length_local[rank],
            values2.accept_length_local[rank],
            atol=0,
            rtol=0,
        )


def test_dp_sampling_swap_reference_matches_batch_vocab_layout() -> None:
    world_size = 2
    pad_batch_size = 4
    num_tokens_per_request = 2
    vocab_size = 6
    v_local = vocab_size // world_size
    full_logits = torch.arange(
        pad_batch_size * num_tokens_per_request * vocab_size,
        dtype=torch.float32,
    ).view(pad_batch_size * num_tokens_per_request, vocab_size)
    values = DPSamplingInputValues(
        local_logits=[
            full_logits[:, rank * v_local : (rank + 1) * v_local].contiguous()
            for rank in range(world_size)
        ],
        predict_local=[
            torch.zeros((2, 2), dtype=torch.int32),
            torch.zeros((2, 2), dtype=torch.int32),
        ],
        accept_index_local=[
            torch.full((2, 2), -1, dtype=torch.int32),
            torch.full((2, 2), -1, dtype=torch.int32),
        ],
        accept_length_local=[
            torch.zeros((2,), dtype=torch.int32),
            torch.zeros((2,), dtype=torch.int32),
        ],
    )

    swapped = dp_sampling_swap_reference(values)

    torch.testing.assert_close(swapped[0], full_logits[:4])
    torch.testing.assert_close(swapped[1], full_logits[4:])


def test_dp_sampling_gather_reference_preserves_rank_major_verify_outputs() -> None:
    values = DPSamplingInputValues(
        local_logits=[
            torch.zeros((8, 3), dtype=torch.float32),
            torch.zeros((8, 3), dtype=torch.float32),
        ],
        predict_local=[
            torch.arange(0, 4, dtype=torch.int32).view(2, 2),
            torch.tensor([[4, 5], [0, 1]], dtype=torch.int32),
        ],
        accept_index_local=[
            torch.tensor([[0, -1], [2, 3]], dtype=torch.int32),
            torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32),
        ],
        accept_length_local=[
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([2, 0], dtype=torch.int32),
        ],
    )

    predict, accept_index, accept_length = dp_sampling_gather_reference(values)

    torch.testing.assert_close(
        predict,
        torch.tensor([[0, 1], [2, 3], [4, 5], [0, 1]], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        accept_index,
        torch.tensor([[0, -1], [2, 3], [0, 1], [-1, -1]], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        accept_length,
        torch.tensor([1, 2, 2, 0], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_all_gather_inputs_generate_values_and_reference() -> None:
    config = AllGatherInputConfig(
        world_size=4,
        total_tokens=13,
        hidden_size=8,
        max_tokens_per_rank=6,
        dtype=torch.float32,
    )
    values = AllGatherInputs(config).generate(
        seed=12,
        metadata_seed=13,
        device="cpu",
    )

    assert len(values.rank_inputs) == 4
    assert sum(values.tokens_per_rank) == 13
    assert max(values.tokens_per_rank) <= 6
    for rank, rank_input in enumerate(values.rank_inputs):
        assert rank_input.shape == (values.tokens_per_rank[rank], 8)

    expected = all_gather_reference(values)
    torch.testing.assert_close(expected, torch.cat(values.rank_inputs, dim=0))


def test_reduce_scatter_inputs_generate_values_and_reference() -> None:
    config = ReduceScatterInputConfig(
        world_size=4,
        total_tokens=13,
        hidden_size=8,
        max_tokens_per_rank=6,
        dtype=torch.float32,
    )
    values = ReduceScatterInputs(config).generate(
        seed=14,
        metadata_seed=15,
        device="cpu",
    )

    assert len(values.rank_inputs) == 4
    assert sum(values.tokens_per_rank) == 13
    assert max(values.tokens_per_rank) <= 6
    assert all(rank_input.shape == (13, 8) for rank_input in values.rank_inputs)

    expected = reduce_scatter_sum_reference(values)
    reduced = sum(values.rank_inputs)
    offset = 0
    for rank, num_tokens in enumerate(values.tokens_per_rank):
        torch.testing.assert_close(
            expected[rank],
            reduced[offset : offset + num_tokens],
        )
        offset += num_tokens


def test_reduce_scatter_residual_rmsnorm_inputs_generate_values_and_reference() -> None:
    values = ReduceScatterResidualRMSNormInputs(
        ReduceScatterResidualRMSNormInputConfig(
            world_size=3,
            total_tokens=8,
            hidden_size=6,
            dtype=torch.float32,
            residual_dtype=torch.float32,
            include_add_in=True,
            add_in_dtype=torch.float32,
            weight_dtype=torch.float32,
            eps=1e-5,
        )
    ).generate(seed=151, device="cpu")

    assert len(values.rank_inputs) == 3
    assert values.tokens_per_rank == [3, 3, 2]
    assert all(rank_input.shape == (8, 6) for rank_input in values.rank_inputs)
    assert [residual.shape for residual in values.residuals] == [
        (3, 6),
        (3, 6),
        (2, 6),
    ]
    assert values.add_ins is not None
    assert [add_in.shape for add_in in values.add_ins] == [(3, 6), (3, 6), (2, 6)]

    refs = reduce_scatter_residual_rmsnorm_reference(values)
    reduced = sum(values.rank_inputs)
    rank = 1
    offset = sum(values.tokens_per_rank[:rank])
    shard = reduced[offset : offset + values.tokens_per_rank[rank]]
    manual_residual = shard + values.residuals[rank] + values.add_ins[rank]
    manual_norm = manual_residual * torch.rsqrt(
        manual_residual.pow(2).mean(dim=-1, keepdim=True) + values.eps
    )
    manual_norm = manual_norm * values.weight

    torch.testing.assert_close(refs.residual_outputs[rank], manual_residual)
    torch.testing.assert_close(refs.norm_outputs[rank], manual_norm)


def test_reduce_scatter_residual_rmsnorm_outputs_match_wrapper_dtypes() -> None:
    values = ReduceScatterResidualRMSNormInputs(
        ReduceScatterResidualRMSNormInputConfig(
            world_size=2,
            total_tokens=5,
            hidden_size=8,
            dtype=torch.bfloat16,
            residual_dtype=torch.float32,
            weight_dtype=torch.float32,
        )
    ).generate(seed=152, device="cpu")

    refs = reduce_scatter_residual_rmsnorm_reference(values)

    assert [output.shape for output in refs.norm_outputs] == [(3, 8), (2, 8)]
    assert [output.dtype for output in refs.norm_outputs] == [
        torch.bfloat16,
        torch.bfloat16,
    ]
    assert [output.dtype for output in refs.residual_outputs] == [
        torch.float32,
        torch.float32,
    ]


def test_all_gather_dual_rmsnorm_inputs_generate_values_and_reference() -> None:
    values = AllGatherDualRMSNormInputs(
        AllGatherDualRMSNormInputConfig(
            world_size=3,
            total_tokens=10,
            q_lora_rank=4,
            kv_lora_rank=3,
            qk_rope_head_dim=2,
            max_tokens_per_rank=5,
            dtype=torch.float32,
            q_weight_dtype=torch.float32,
            kv_weight_dtype=torch.float32,
            eps_q=1e-5,
            eps_kv=2e-5,
        )
    ).generate(seed=161, metadata_seed=162, device="cpu")

    assert len(values.rank_inputs) == 3
    assert sum(values.tokens_per_rank) == 10
    assert max(values.tokens_per_rank) <= 5
    assert all(
        rank_input.shape == (values.tokens_per_rank[rank], 9)
        for rank, rank_input in enumerate(values.rank_inputs)
    )
    assert values.q_weight.shape == (4,)
    assert values.kv_weight.shape == (3,)

    refs = all_gather_dual_rmsnorm_reference(values)
    gathered = torch.cat(values.rank_inputs, dim=0)
    q_slice = gathered[:, :4]
    kv_slice = gathered[:, 4:7]
    manual_q = q_slice * torch.rsqrt(
        q_slice.pow(2).mean(dim=-1, keepdim=True) + values.eps_q
    )
    manual_q = manual_q * values.q_weight
    manual_kv = kv_slice * torch.rsqrt(
        kv_slice.pow(2).mean(dim=-1, keepdim=True) + values.eps_kv
    )
    manual_kv = manual_kv * values.kv_weight
    manual_gathered = gathered.clone()
    manual_gathered[:, 4:7] = manual_kv

    torch.testing.assert_close(refs.q_norm_output, manual_q)
    torch.testing.assert_close(refs.kv_norm_output, manual_kv)
    torch.testing.assert_close(refs.gathered_output, manual_gathered)


def test_all_gather_dual_rmsnorm_metadata_seed_controls_token_distribution() -> None:
    generator = AllGatherDualRMSNormInputs(
        AllGatherDualRMSNormInputConfig(
            world_size=4,
            total_tokens=13,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            max_tokens_per_rank=5,
            dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=163, metadata_seed=164, device="cpu")
    values2 = generator.generate(seed=165, metadata_seed=164, device="cpu")

    assert values1.tokens_per_rank == values2.tokens_per_rank
    assert not torch.equal(values1.rank_inputs[0], values2.rank_inputs[0])
    assert not torch.equal(values1.q_weight, values2.q_weight)


def test_minimax_allreduce_qk_rmsnorm_inputs_generate_values_and_reference() -> None:
    values = MiniMaxAllReduceQKRMSNormInputs(
        MiniMaxAllReduceQKRMSNormInputConfig(
            world_size=2,
            num_tokens=3,
            dtype=torch.float32,
            eps=1e-5,
        )
    ).generate(seed=171, device="cpu")

    assert len(values.q_rank_inputs) == 2
    assert len(values.k_rank_inputs) == 2
    assert all(q.shape == (3, 3072) for q in values.q_rank_inputs)
    assert all(k.shape == (3, 512) for k in values.k_rank_inputs)
    assert values.q_weight.shape == (3072,)
    assert values.k_weight.shape == (512,)
    assert values.q_weight.dtype == torch.bfloat16
    assert values.k_weight.dtype == torch.bfloat16

    refs = minimax_allreduce_qk_rmsnorm_reference(values)
    reduced_q = sum(values.q_rank_inputs)
    reduced_k = sum(values.k_rank_inputs)
    manual_q = reduced_q * torch.rsqrt(
        reduced_q.pow(2).mean(dim=-1, keepdim=True) + values.eps
    )
    manual_q = manual_q * values.q_weight.float()
    manual_k = reduced_k * torch.rsqrt(
        reduced_k.pow(2).mean(dim=-1, keepdim=True) + values.eps
    )
    manual_k = manual_k * values.k_weight.float()

    torch.testing.assert_close(refs.q_norm_outputs[0], manual_q)
    torch.testing.assert_close(refs.q_norm_outputs[1], manual_q)
    torch.testing.assert_close(refs.k_norm_outputs[0], manual_k)
    torch.testing.assert_close(refs.k_norm_outputs[1], manual_k)


def test_minimax_allreduce_qk_rmsnorm_generates_strided_views() -> None:
    values = MiniMaxAllReduceQKRMSNormInputs(
        MiniMaxAllReduceQKRMSNormInputConfig(
            world_size=4,
            num_tokens=2,
            dtype=torch.bfloat16,
            stride_padding=8,
        )
    ).generate(seed=172, device="cpu")

    assert values.q_storage is not None
    assert values.k_storage is not None
    assert values.q_rank_inputs[0].shape == (2, 1536)
    assert values.k_rank_inputs[0].shape == (2, 256)
    assert values.q_storage[0].shape == (2, 1544)
    assert values.k_storage[0].shape == (2, 264)
    assert values.q_rank_inputs[0].stride() == (1544, 1)
    assert values.k_rank_inputs[0].stride() == (264, 1)
    assert not values.q_rank_inputs[0].is_contiguous()
    assert not values.k_rank_inputs[0].is_contiguous()

    refs = minimax_allreduce_qk_rmsnorm_reference(values)

    assert refs.q_norm_outputs[0].shape == (2, 1536)
    assert refs.k_norm_outputs[0].shape == (2, 256)
    assert refs.q_norm_outputs[0].dtype == torch.bfloat16
    assert refs.k_norm_outputs[0].dtype == torch.bfloat16


def test_collective_metadata_seed_controls_token_distribution_only() -> None:
    generator = AllGatherInputs(
        AllGatherInputConfig(
            world_size=4,
            total_tokens=17,
            hidden_size=8,
            max_tokens_per_rank=7,
            dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=16, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=17, metadata_seed=99, device="cpu")

    assert values1.tokens_per_rank == values2.tokens_per_rank
    assert not torch.equal(values1.rank_inputs[0], values2.rank_inputs[0])


def test_zero_token_all_gather_and_reduce_scatter_are_valid() -> None:
    all_gather_values = AllGatherInputs(
        AllGatherInputConfig(
            world_size=3,
            total_tokens=0,
            hidden_size=8,
            max_tokens_per_rank=0,
            dtype=torch.float32,
        )
    ).generate(seed=18, metadata_seed=19, device="cpu")
    reduce_scatter_values = ReduceScatterInputs(
        ReduceScatterInputConfig(
            world_size=3,
            total_tokens=0,
            hidden_size=8,
            max_tokens_per_rank=0,
            dtype=torch.float32,
        )
    ).generate(seed=20, metadata_seed=21, device="cpu")

    assert all_gather_reference(all_gather_values).shape == (0, 8)
    assert [
        output.shape for output in reduce_scatter_sum_reference(reduce_scatter_values)
    ] == [
        (0, 8),
        (0, 8),
        (0, 8),
    ]


@pytest.mark.parametrize("world_size", [0, -1])
def test_collectives_reject_invalid_world_size(world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size"):
        AllReduceInputs(
            AllReduceInputConfig(
                world_size=world_size,
                shape=(8,),
            )
        )


def test_all_reduce_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        AllReduceInputs(
            AllReduceInputConfig(
                world_size=2,
                shape=(),
            )
        )


def test_collectives_reject_impossible_token_cap() -> None:
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        AllGatherInputs(
            AllGatherInputConfig(
                world_size=2,
                total_tokens=5,
                hidden_size=8,
                max_tokens_per_rank=2,
            )
        )


def test_collectives_reject_nonfloating_dtype() -> None:
    with pytest.raises(ValueError, match="regular floating"):
        ReduceScatterInputs(
            ReduceScatterInputConfig(
                world_size=2,
                total_tokens=5,
                hidden_size=8,
                dtype=torch.int32,
            )
        )


def test_dp_sampling_rejects_nondivisible_batch_or_vocab() -> None:
    with pytest.raises(ValueError, match="pad_batch_size"):
        DPSamplingInputs(
            DPSamplingInputConfig(
                world_size=3,
                pad_batch_size=5,
                num_tokens_per_request=2,
                vocab_size=12,
            )
        )
    with pytest.raises(ValueError, match="vocab_size"):
        DPSamplingInputs(
            DPSamplingInputConfig(
                world_size=3,
                pad_batch_size=6,
                num_tokens_per_request=2,
                vocab_size=10,
            )
        )


def test_dp_sampling_rejects_unsupported_logits_dtype() -> None:
    with pytest.raises(ValueError, match="logits_dtype"):
        DPSamplingInputs(
            DPSamplingInputConfig(
                world_size=2,
                pad_batch_size=4,
                num_tokens_per_request=2,
                vocab_size=8,
                logits_dtype=torch.float64,
            )
        )


def test_dp_sampling_reference_rejects_inconsistent_accept_length() -> None:
    values = DPSamplingInputs(
        DPSamplingInputConfig(
            world_size=2,
            pad_batch_size=4,
            num_tokens_per_request=2,
            vocab_size=8,
            logits_dtype=torch.float32,
        )
    ).generate(seed=136, metadata_seed=137, device="cpu")
    values.accept_length_local[0][0] += 1

    with pytest.raises(ValueError, match="accept_length_local"):
        dp_sampling_reference(values)


def test_expert_parallel_routing_rejects_invalid_expert_partition() -> None:
    with pytest.raises(ValueError, match="num_experts"):
        ExpertParallelRoutingInputs(
            ExpertParallelRoutingInputConfig(
                world_size=3,
                total_tokens=5,
                hidden_size=8,
                num_experts=5,
                top_k=2,
            )
        )


def test_expert_parallel_routing_reference_rejects_duplicate_topk_ids() -> None:
    values = ExpertParallelRoutingInputs(
        ExpertParallelRoutingInputConfig(
            world_size=2,
            total_tokens=4,
            hidden_size=8,
            num_experts=4,
            top_k=2,
            dtype=torch.float32,
        )
    ).generate(seed=125, metadata_seed=126, device="cpu")
    if values.topk_ids[0].shape[0] == 0:
        values.topk_ids[1][0, 1] = values.topk_ids[1][0, 0]
    else:
        values.topk_ids[0][0, 1] = values.topk_ids[0][0, 0]

    with pytest.raises(ValueError, match="unique"):
        expert_parallel_routing_reference(values)


def test_expert_parallel_routing_reference_rejects_bad_weight_sums() -> None:
    values = ExpertParallelRoutingInputValues(
        rank_hidden_states=[torch.ones((1, 4), dtype=torch.float32)],
        topk_ids=[torch.tensor([[0, 1]], dtype=torch.int64)],
        topk_weights=[torch.tensor([[0.25, 0.25]], dtype=torch.float32)],
        expert_outputs=[torch.ones((1, 2, 4), dtype=torch.float32)],
        tokens_per_rank=[1],
        num_experts=2,
    )

    with pytest.raises(ValueError, match="sum to 1"):
        expert_parallel_routing_reference(values)


def test_all_reduce_residual_rmsnorm_rejects_negative_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        AllReduceResidualRMSNormInputs(
            AllReduceResidualRMSNormInputConfig(
                world_size=2,
                num_tokens=4,
                hidden_size=8,
                eps=-1e-6,
            )
        )


def test_all_reduce_residual_rmsnorm_reference_rejects_bad_residual_shape() -> None:
    values = AllReduceResidualRMSNormInputs(
        AllReduceResidualRMSNormInputConfig(
            world_size=2,
            num_tokens=4,
            hidden_size=8,
            dtype=torch.float32,
        )
    ).generate(seed=113, device="cpu")
    values.residuals[0] = values.residuals[0][:, :-1]

    with pytest.raises(ValueError, match="residual shape"):
        all_reduce_residual_rmsnorm_reference(values)


def test_reduce_scatter_residual_rmsnorm_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="eps"):
        ReduceScatterResidualRMSNormInputs(
            ReduceScatterResidualRMSNormInputConfig(
                world_size=2,
                total_tokens=4,
                hidden_size=8,
                eps=-1e-6,
            )
        )


def test_reduce_scatter_residual_rmsnorm_reference_rejects_bad_add_shape() -> None:
    values = ReduceScatterResidualRMSNormInputs(
        ReduceScatterResidualRMSNormInputConfig(
            world_size=2,
            total_tokens=4,
            hidden_size=8,
            include_add_in=True,
            dtype=torch.float32,
        )
    ).generate(seed=153, device="cpu")
    assert values.add_ins is not None
    values.add_ins[0] = values.add_ins[0][:, :-1]

    with pytest.raises(ValueError, match="add_ins"):
        reduce_scatter_residual_rmsnorm_reference(values)


def test_all_gather_dual_rmsnorm_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="q_lora_rank"):
        AllGatherDualRMSNormInputs(
            AllGatherDualRMSNormInputConfig(
                world_size=2,
                total_tokens=4,
                q_lora_rank=0,
                kv_lora_rank=4,
                qk_rope_head_dim=2,
            )
        )


def test_all_gather_dual_rmsnorm_reference_rejects_bad_weight_shape() -> None:
    values = AllGatherDualRMSNormInputs(
        AllGatherDualRMSNormInputConfig(
            world_size=2,
            total_tokens=4,
            q_lora_rank=4,
            kv_lora_rank=3,
            qk_rope_head_dim=2,
            dtype=torch.float32,
        )
    ).generate(seed=166, metadata_seed=167, device="cpu")
    values.kv_weight = values.kv_weight[:-1]

    with pytest.raises(ValueError, match="kv_weight"):
        all_gather_dual_rmsnorm_reference(values)


def test_minimax_allreduce_qk_rmsnorm_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="world_size"):
        MiniMaxAllReduceQKRMSNormInputs(
            MiniMaxAllReduceQKRMSNormInputConfig(
                world_size=3,
                num_tokens=2,
            )
        )
    with pytest.raises(ValueError, match="q_global_hidden_size"):
        MiniMaxAllReduceQKRMSNormInputs(
            MiniMaxAllReduceQKRMSNormInputConfig(
                world_size=2,
                num_tokens=2,
                q_global_hidden_size=4096,
            )
        )
    with pytest.raises(ValueError, match="weights"):
        MiniMaxAllReduceQKRMSNormInputs(
            MiniMaxAllReduceQKRMSNormInputConfig(
                world_size=2,
                num_tokens=2,
                weight_dtype=torch.float32,
            )
        )
    with pytest.raises(ValueError, match="row stride"):
        MiniMaxAllReduceQKRMSNormInputs(
            MiniMaxAllReduceQKRMSNormInputConfig(
                world_size=2,
                num_tokens=2,
                dtype=torch.bfloat16,
                stride_padding=1,
            )
        )


def test_minimax_allreduce_qk_rmsnorm_reference_rejects_bad_weight_shape() -> None:
    values = MiniMaxAllReduceQKRMSNormInputs(
        MiniMaxAllReduceQKRMSNormInputConfig(
            world_size=2,
            num_tokens=2,
            dtype=torch.float32,
        )
    ).generate(seed=173, device="cpu")
    values.q_weight = values.q_weight[:-1]

    with pytest.raises(ValueError, match="q_weight"):
        minimax_allreduce_qk_rmsnorm_reference(values)
