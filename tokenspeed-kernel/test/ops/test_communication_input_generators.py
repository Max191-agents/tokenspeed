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

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.triton import (
    all_gather,
    all_reduce,
    all_reduce_can_run,
    allreduce_residual_rmsnorm,
    create_state,
    reduce_scatter,
)
from tokenspeed_numerics_input_generators import (
    AllGatherInputConfig,
    AllGatherInputs,
    AllReduceInputConfig,
    AllReduceInputs,
    AllReduceResidualRMSNormInputConfig,
    AllReduceResidualRMSNormInputs,
    DPSamplingInputConfig,
    DPSamplingInputs,
    ExpertParallelRoutingInputConfig,
    ExpertParallelRoutingInputs,
    ReduceScatterInputConfig,
    ReduceScatterInputs,
    all_gather_reference,
    all_reduce_residual_rmsnorm_reference,
    all_reduce_sum_reference,
    dp_sampling_reference,
    expert_parallel_routing_reference,
    reduce_scatter_sum_reference,
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required for communication generator tests")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not torch.version.hip:
        pytest.skip("communication generator smoke tests currently target AMD ROCm")


def _worker_fn(rank: int, world_size: int, port: int, error_dict) -> None:
    try:
        _worker_main(rank, world_size, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _worker_main(rank: int, world_size: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _check_all_reduce(rank, world_size, device)
        _check_all_gather(rank, world_size, device)
        _check_reduce_scatter(rank, world_size, device)
    finally:
        dist.destroy_process_group()


def _ar_rmsnorm_worker_fn(rank: int, world_size: int, port: int, result_dict) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _check_allreduce_residual_rmsnorm(rank, world_size, device)
    except RuntimeError:
        trace = traceback.format_exc()
        if _is_iris_backend_unavailable(trace):
            result_dict[rank] = ("skip", trace)
        else:
            result_dict[rank] = ("error", trace)
    except Exception:
        result_dict[rank] = ("error", traceback.format_exc())
    finally:
        dist.destroy_process_group()


def _is_iris_backend_unavailable(trace: str) -> bool:
    return (
        "iris" in trace.lower()
        and "export_dmabuf_handle" in trace
        and "HIP error code 1: invalid argument" in trace
    )


def _check_all_reduce(rank: int, world_size: int, device: torch.device) -> None:
    values = AllReduceInputs(
        AllReduceInputConfig(
            world_size=world_size,
            shape=(2880,),
            dtype=torch.bfloat16,
        )
    ).generate(seed=91, device="cpu")
    local = values.rank_inputs[rank].to(device=device)
    expected = all_reduce_sum_reference(values).to(device=device)
    state = create_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        max_numel=local.numel(),
        device=device,
    )

    assert all_reduce_can_run(state, local)
    result = all_reduce(state, local)
    torch.cuda.synchronize()

    assert result is local
    torch.testing.assert_close(local.float(), expected.float(), atol=1e-2, rtol=1e-2)


def _check_allreduce_residual_rmsnorm(
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    config = AllReduceResidualRMSNormInputConfig(
        world_size=world_size,
        num_tokens=8,
        hidden_size=2880,
        dtype=torch.bfloat16,
        residual_dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        eps=1e-6,
    )
    values = AllReduceResidualRMSNormInputs(config).generate(seed=101, device="cpu")
    refs = all_reduce_residual_rmsnorm_reference(values)
    local = values.rank_inputs[rank].to(device=device)
    residual = values.residuals[rank].to(device=device)
    weight = values.weight.to(device=device)

    norm_out, residual_out, scale, partial = allreduce_residual_rmsnorm(
        input_tensor=local,
        residual=residual,
        weight=weight,
        rank=rank,
        group=dist.group.WORLD,
        eps=config.eps,
        max_token_num=16,
    )
    torch.cuda.synchronize()

    assert scale is None
    assert partial is None
    assert norm_out is not None
    assert residual_out is not None
    torch.testing.assert_close(
        residual_out.float(),
        refs.residual_outputs[rank].to(device=device),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        norm_out.float(),
        refs.norm_outputs[rank].to(device=device),
        atol=2e-2,
        rtol=2e-2,
    )


def _check_all_gather(rank: int, world_size: int, device: torch.device) -> None:
    config = AllGatherInputConfig(
        world_size=world_size,
        total_tokens=9,
        hidden_size=2880,
        max_tokens_per_rank=6,
        dtype=torch.bfloat16,
    )
    values = AllGatherInputs(config).generate(
        seed=92,
        metadata_seed=93,
        device="cpu",
    )
    rsag = create_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        max_tokens=config.total_tokens,
        hidden_size=config.hidden_size,
        device=device,
    )
    expected = all_gather_reference(values).to(device=device)

    result = all_gather(
        rsag,
        values.rank_inputs[rank].to(device=device),
        token_list_in_group=values.tokens_per_rank,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _check_reduce_scatter(rank: int, world_size: int, device: torch.device) -> None:
    config = ReduceScatterInputConfig(
        world_size=world_size,
        total_tokens=9,
        hidden_size=2880,
        max_tokens_per_rank=6,
        dtype=torch.bfloat16,
    )
    values = ReduceScatterInputs(config).generate(
        seed=94,
        metadata_seed=95,
        device="cpu",
    )
    rsag = create_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        max_tokens=config.total_tokens,
        hidden_size=config.hidden_size,
        device=device,
    )
    expected = reduce_scatter_sum_reference(values)[rank].to(device=device)

    result = reduce_scatter(
        rsag,
        values.rank_inputs[rank].to(device=device),
        token_list_in_group=values.tokens_per_rank,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        result.float(),
        expected.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_expert_parallel_generator_maps_to_deepep_dispatch_contract() -> None:
    values = ExpertParallelRoutingInputs(
        ExpertParallelRoutingInputConfig(
            world_size=2,
            total_tokens=9,
            hidden_size=16,
            num_experts=4,
            top_k=2,
            max_tokens_per_rank=5,
            dtype=torch.bfloat16,
        )
    ).generate(seed=96, metadata_seed=97, device="cpu")
    refs = expert_parallel_routing_reference(values)

    rank = 0
    local_x = values.rank_hidden_states[rank]
    local_topk_ids = values.topk_ids[rank].to(torch.int64)
    local_topk_weights = values.topk_weights[rank].to(torch.float32)

    assert local_x.shape == (values.tokens_per_rank[rank], 16)
    assert local_topk_ids.shape == (values.tokens_per_rank[rank], 2)
    assert local_topk_weights.shape == (values.tokens_per_rank[rank], 2)
    assert refs.recv_hidden_states[rank].shape[1] == 16
    assert refs.recv_topk_ids[rank].shape[1] == 2
    assert refs.num_recv_tokens_per_expert[rank].shape == (2,)
    assert refs.combined_outputs[rank].shape == local_x.shape


def test_dp_sampling_generator_matches_triton_kernel_contract() -> None:
    config = DPSamplingInputConfig(
        world_size=2,
        pad_batch_size=4,
        num_tokens_per_request=3,
        vocab_size=8,
        logits_dtype=torch.bfloat16,
    )
    values = DPSamplingInputs(config).generate(
        seed=98,
        metadata_seed=99,
        device="cpu",
    )
    refs = dp_sampling_reference(values)

    reqs_per_rank = config.pad_batch_size // config.world_size
    v_local = config.vocab_size // config.world_size
    for rank in range(config.world_size):
        assert values.local_logits[rank].shape == (
            config.pad_batch_size * config.num_tokens_per_request,
            v_local,
        )
        assert values.local_logits[rank].dtype == config.logits_dtype
        assert values.local_logits[rank].is_contiguous()

        assert refs.swapped_logits[rank].shape == (
            reqs_per_rank * config.num_tokens_per_request,
            config.vocab_size,
        )
        assert refs.swapped_logits[rank].dtype == config.logits_dtype
        assert refs.swapped_logits[rank].is_contiguous()

        assert values.predict_local[rank].shape == (
            reqs_per_rank,
            config.num_tokens_per_request,
        )
        assert values.accept_index_local[rank].shape == (
            reqs_per_rank,
            config.num_tokens_per_request,
        )
        assert values.accept_length_local[rank].shape == (reqs_per_rank,)
        assert values.predict_local[rank].dtype == torch.int32
        assert values.accept_index_local[rank].dtype == torch.int32
        assert values.accept_length_local[rank].dtype == torch.int32
        assert values.predict_local[rank].is_contiguous()
        assert values.accept_index_local[rank].is_contiguous()
        assert values.accept_length_local[rank].is_contiguous()

    assert refs.predict.shape == (
        config.pad_batch_size,
        config.num_tokens_per_request,
    )
    assert refs.accept_index.shape == refs.predict.shape
    assert refs.accept_length.shape == (config.pad_batch_size,)


def test_communication_generators_run_triton_collectives_world2() -> None:
    world_size = 2
    _skip_if_unsupported(world_size)
    port = _get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker_fn,
        args=(world_size, port, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_allreduce_residual_rmsnorm_generator_runs_tokenspeed_kernel() -> None:
    world_size = 2
    _skip_if_unsupported(world_size)
    port = _get_open_port()
    result_dict = mp.Manager().dict()
    mp.spawn(
        _ar_rmsnorm_worker_fn,
        args=(world_size, port, result_dict),
        nprocs=world_size,
        join=True,
    )

    results = dict(result_dict)
    errors = {
        rank: trace for rank, (status, trace) in results.items() if status == "error"
    }
    if errors:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in errors.items()))
    skips = {
        rank: trace for rank, (status, trace) in results.items() if status == "skip"
    }
    if skips:
        pytest.skip(
            "IRIS AR+RMSNorm backend could not create its symmetric heap in this "
            "environment"
        )
