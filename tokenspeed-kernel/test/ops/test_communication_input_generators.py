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
    create_state,
    reduce_scatter,
)
from tokenspeed_numerics_input_generators import (
    AllGatherInputConfig,
    AllGatherInputs,
    AllReduceInputConfig,
    AllReduceInputs,
    ReduceScatterInputConfig,
    ReduceScatterInputs,
    all_gather_reference,
    all_reduce_sum_reference,
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
