# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.prefill_stage2 import (
    _select_stage2_reduce_launch,
    gluon_mxfp4_moe_stage2_reduce_kernel,
)

requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(), reason="AMD GFX950 is required for stage-2 reduction execution"
)


@pytest.mark.parametrize(
    "token_num,n_cols,topk,expected",
    [
        pytest.param(16, 7168, 8, (4, 256, 2), id="kimi-speculative"),
        pytest.param(15, 7168, 8, (32, 256, 1), id="different-token-count"),
        pytest.param(16, 7167, 8, (32, 256, 1), id="different-hidden-size"),
        pytest.param(16, 7168, 4, (32, 256, 1), id="different-topk"),
        pytest.param(4096, 7168, 8, (16, 256, 1), id="large-prefill"),
    ],
)
def test_stage2_reduce_launch_is_scoped_to_kimi_speculative_shape(
    token_num: int,
    n_cols: int,
    topk: int,
    expected: tuple[int, int, int],
) -> None:
    assert _select_stage2_reduce_launch(token_num, n_cols, topk) == expected


def _launch_reduce(
    partials: torch.Tensor,
    out: torch.Tensor,
    config: tuple[int, int, int],
) -> None:
    token_num, topk, n_cols = partials.shape
    block_m, block_n, num_warps = config
    grid = ((token_num + block_m - 1) // block_m * ((n_cols + block_n - 1) // block_n),)
    gluon_mxfp4_moe_stage2_reduce_kernel[grid](
        partials,
        out,
        token_num,
        n_cols,
        topk,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        TOP_K=topk,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )


@requires_gfx950
def test_kimi_speculative_stage2_reduce_is_bitwise_and_deterministic() -> None:
    token_num = 16
    topk = 8
    n_cols = 7168
    generator = torch.Generator(device="cuda").manual_seed(20260724)
    partials = (
        torch.randn(
            (token_num, topk, n_cols),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.125
    ).contiguous()

    expected_fp32 = partials[:, 0].float()
    for slot in range(1, topk):
        expected_fp32.add_(partials[:, slot].float())
    expected = expected_fp32.to(torch.bfloat16)

    reference = torch.full_like(expected, float("nan"))
    actual_first = torch.full_like(expected, float("nan"))
    actual_second = torch.full_like(expected, float("nan"))
    _launch_reduce(partials, reference, (32, 256, 1))
    tuned = _select_stage2_reduce_launch(token_num, n_cols, topk)
    _launch_reduce(partials, actual_first, tuned)
    _launch_reduce(partials, actual_second, tuned)
    torch.cuda.synchronize()

    assert torch.equal(reference, expected)
    assert torch.equal(actual_first, reference)
    assert torch.equal(actual_second, actual_first)


@requires_gfx950
def test_large_prefill_stage2_reduce_layout_matches_fixed_order() -> None:
    token_num = 4096
    topk = 8
    n_cols = 256
    generator = torch.Generator(device="cuda").manual_seed(20260725)
    partials = (
        torch.randn(
            (token_num, topk, n_cols),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.125
    ).contiguous()

    expected_fp32 = partials[:, 0].float()
    for slot in range(1, topk):
        expected_fp32.add_(partials[:, slot].float())
    expected = expected_fp32.to(torch.bfloat16)

    actual = torch.full_like(expected, float("nan"))
    config = _select_stage2_reduce_launch(token_num, n_cols, topk)
    assert config == (16, 256, 1)
    _launch_reduce(partials, actual, config)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
