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

import socket
import traceback
from typing import List

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
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    AllGatherInputConfig,
    AllGatherInputs,
    AllReduceInputConfig,
    AllReduceInputs,
    AllReduceResidualRMSNormInputConfig,
    AllReduceResidualRMSNormInputs,
    ReduceScatterInputConfig,
    ReduceScatterInputs,
    all_gather_reference,
    all_reduce_residual_rmsnorm_reference,
    all_reduce_sum_reference,
    reduce_scatter_sum_reference,
)


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def token_cases(world_size: int) -> List[List[int]]:
    cases = [
        [8] * world_size,
        [8 + rank for rank in range(world_size)],
    ]
    if world_size >= 4:
        cases.append([1, 20, 3] + [0] * (world_size - 3))
    else:
        cases.append([3] + [0] * (world_size - 1))
    return cases


def worker_fn(rank, world_size, port, hidden_size, error_dict):
    try:
        worker_main(rank, world_size, port, hidden_size)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def worker_main(rank: int, world_size: int, port: int, hidden_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cases = token_cases(world_size)
        max_tokens = max(sum(tokens) for tokens in cases)
        rsag = create_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
        )

        for tokens in cases:
            check_all_gather(rsag, rank, world_size, tokens, hidden_size, device)
            check_reduce_scatter(rsag, rank, world_size, tokens, hidden_size, device)

        if current_platform().is_amd:
            check_all_reduce(rank, world_size, device)
            check_allreduce_residual_rmsnorm(rank, world_size, device)
    finally:
        dist.destroy_process_group()


def check_all_gather(
    rsag, rank: int, world_size: int, tokens: List[int], hidden_size: int, device
) -> None:
    values = AllGatherInputs(
        AllGatherInputConfig(
            world_size=world_size,
            total_tokens=sum(tokens),
            hidden_size=hidden_size,
            max_tokens_per_rank=max(tokens),
            dtype=torch.bfloat16,
        )
    ).generate(
        seed=1_000 + sum(tokens) + hidden_size,
        metadata_seed=2_000 + sum(tokens) + max(tokens),
        device="cpu",
    )
    local = values.rank_inputs[rank].to(device=device)
    expected = all_gather_reference(values).to(device=device)

    result = all_gather(rsag, local, token_list_in_group=values.tokens_per_rank)

    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def check_all_reduce(rank: int, world_size: int, device) -> None:
    max_numel = 512 * 1024 // torch.empty((), dtype=torch.bfloat16).element_size()
    state = create_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        max_numel=max_numel,
        device=device,
    )

    for numel in [2880, 20160, 23040, 92160, 184320]:
        values = AllReduceInputs(
            AllReduceInputConfig(
                world_size=world_size,
                shape=(numel,),
                dtype=torch.bfloat16,
            )
        ).generate(seed=3_000 + numel, device="cpu")
        tensor = values.rank_inputs[rank].to(device=device)
        expected = all_reduce_sum_reference(values).to(device=device)
        assert all_reduce_can_run(state, tensor)
        result = all_reduce(state, tensor)
        assert result is tensor
        torch.testing.assert_close(
            result.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            tensor.float(), expected.float(), atol=2e-2, rtol=2e-2
        )

    large = torch.full((300000,), rank + 1, dtype=torch.bfloat16, device=device)
    assert not all_reduce_can_run(state, large)


def check_allreduce_residual_rmsnorm(rank: int, world_size: int, device) -> None:
    hidden = 2880
    eps = 1e-6

    for tokens in [1, 8, 32]:
        values = AllReduceResidualRMSNormInputs(
            AllReduceResidualRMSNormInputConfig(
                world_size=world_size,
                num_tokens=tokens,
                hidden_size=hidden,
                dtype=torch.bfloat16,
                residual_dtype=torch.bfloat16,
                weight_dtype=torch.float32,
                eps=eps,
            )
        ).generate(seed=700 + tokens, device="cpu")
        refs = all_reduce_residual_rmsnorm_reference(values)
        x = values.rank_inputs[rank].to(device=device)
        residual = values.residuals[rank].to(device=device)
        weight = values.weight.to(device=device)

        ref_residual = refs.residual_outputs[rank].to(device=device)
        ref_norm = refs.norm_outputs[rank].to(device=device)

        norm_out, residual_out, scale, partial = allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=residual,
            weight=weight,
            rank=rank,
            group=dist.group.WORLD,
            eps=eps,
            max_token_num=64,
        )
        assert scale is None
        assert partial is None

        torch.testing.assert_close(
            residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)


def check_reduce_scatter(
    rsag, rank: int, world_size: int, tokens: List[int], hidden_size: int, device
) -> None:
    values = ReduceScatterInputs(
        ReduceScatterInputConfig(
            world_size=world_size,
            total_tokens=sum(tokens),
            hidden_size=hidden_size,
            max_tokens_per_rank=max(tokens),
            dtype=torch.bfloat16,
        )
    ).generate(
        seed=4_000 + sum(tokens) + hidden_size,
        metadata_seed=5_000 + sum(tokens) + max(tokens),
        device="cpu",
    )
    full = values.rank_inputs[rank].to(device=device)
    expected = reduce_scatter_sum_reference(values)[rank].to(device=device)

    result = reduce_scatter(rsag, full, token_list_in_group=values.tokens_per_rank)

    assert result.shape == expected.shape
    torch.testing.assert_close(result.float(), expected.float(), atol=2e-2, rtol=2e-2)


def run_rsag_test(world_size: int, hidden_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required for TritonRSAG tests")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=(world_size, port, hidden_size, error_dict),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_triton_communication_correctness_world4():
    run_rsag_test(world_size=4, hidden_size=2880)
