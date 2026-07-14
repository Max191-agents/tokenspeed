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


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def token_cases(world_size: int) -> list[list[int]]:
    cases = [
        [8] * world_size,
        [8 + rank for rank in range(world_size)],
    ]
    if world_size >= 4:
        cases.append([1, 20, 3] + [0] * (world_size - 3))
    else:
        cases.append([3] + [0] * (world_size - 1))
    return cases


def _randn_cpu(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(shape, dtype=torch.float32, generator=generator).clamp(
        -3.0, 3.0
    )
    return values.to(dtype).contiguous()


def _rank_inputs(
    world_size: int,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    seed: int,
) -> list[torch.Tensor]:
    return [
        _randn_cpu(shape, dtype=dtype, seed=seed + rank) for rank in range(world_size)
    ]


def _sum_rank_inputs(rank_inputs: list[torch.Tensor]) -> torch.Tensor:
    acc = torch.zeros_like(rank_inputs[0], dtype=torch.float32)
    for tensor in rank_inputs:
        acc = acc + tensor.float()
    return acc.to(rank_inputs[0].dtype)


def _sum_rank_inputs_for_rmsnorm(rank_inputs: list[torch.Tensor]) -> torch.Tensor:
    acc = torch.zeros_like(rank_inputs[0], dtype=torch.float32)
    for tensor in rank_inputs:
        acc = acc + tensor.float()
    return acc


def _rmsnorm_rows(rows: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = rows.pow(2).mean(dim=-1, keepdim=True)
    return rows * torch.rsqrt(variance + eps) * weight.to(rows.dtype)


def _all_gather_reference(rank_inputs: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(rank_inputs, dim=0)


def _reduce_scatter_reference(
    rank_inputs: list[torch.Tensor],
    tokens_per_rank: list[int],
) -> list[torch.Tensor]:
    reduced = _sum_rank_inputs(rank_inputs)
    outputs: list[torch.Tensor] = []
    offset = 0
    for num_tokens in tokens_per_rank:
        outputs.append(reduced[offset : offset + num_tokens].contiguous())
        offset += num_tokens
    return outputs


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
    rsag, rank: int, world_size: int, tokens: list[int], hidden_size: int, device
) -> None:
    rank_inputs = [
        _randn_cpu(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            seed=1_000 + sum(tokens) + hidden_size + input_rank,
        )
        for input_rank, num_tokens in enumerate(tokens)
    ]
    local = rank_inputs[rank].to(device=device)
    expected = _all_gather_reference(rank_inputs).to(device=device)

    result = all_gather(rsag, local, token_list_in_group=tokens)

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
        rank_inputs = _rank_inputs(
            world_size,
            (numel,),
            dtype=torch.bfloat16,
            seed=3_000 + numel,
        )
        tensor = rank_inputs[rank].to(device=device)
        expected = _sum_rank_inputs(rank_inputs).to(device=device)
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
        rank_inputs = _rank_inputs(
            world_size,
            (tokens, hidden),
            dtype=torch.bfloat16,
            seed=700 + tokens,
        )
        residuals = _rank_inputs(
            world_size,
            (tokens, hidden),
            dtype=torch.bfloat16,
            seed=900 + tokens,
        )
        weight = _randn_cpu((hidden,), dtype=torch.float32, seed=1_100 + tokens)

        reduced = _sum_rank_inputs_for_rmsnorm(rank_inputs)
        ref_residual = reduced + residuals[rank].float()
        ref_norm = _rmsnorm_rows(ref_residual, weight, eps)
        x = rank_inputs[rank].to(device=device)
        residual = residuals[rank].to(device=device)
        weight_device = weight.to(device=device)

        norm_out, residual_out, scale, partial = allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=residual,
            weight=weight_device,
            rank=rank,
            group=dist.group.WORLD,
            eps=eps,
            max_token_num=64,
        )
        assert scale is None
        assert partial is None

        torch.testing.assert_close(
            residual_out.float(),
            ref_residual.to(device=device),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            norm_out.float(),
            ref_norm.to(device=device),
            atol=2e-2,
            rtol=2e-2,
        )


def check_reduce_scatter(
    rsag, rank: int, world_size: int, tokens: list[int], hidden_size: int, device
) -> None:
    total_tokens = sum(tokens)
    rank_inputs = _rank_inputs(
        world_size,
        (total_tokens, hidden_size),
        dtype=torch.bfloat16,
        seed=4_000 + total_tokens + hidden_size,
    )
    full = rank_inputs[rank].to(device=device)
    expected = _reduce_scatter_reference(rank_inputs, tokens)[rank].to(device=device)

    result = reduce_scatter(rsag, full, token_list_in_group=tokens)

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
