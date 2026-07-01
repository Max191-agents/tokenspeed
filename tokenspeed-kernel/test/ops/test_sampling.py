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
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_renorm,
    fused_topk_topp_workspace_size,
)
from tokenspeed_kernel.ops.sampling.triton import (
    gather_and_expand_scalars,
    min_p_renorm_prob,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    GatherExpandScalarsInputConfig,
    GatherExpandScalarsInputs,
    MinPRenormInputConfig,
    MinPRenormInputs,
    TopKTopPRenormInputConfig,
    TopKTopPRenormInputs,
    gather_expand_scalars_reference,
    min_p_renorm_reference,
    top_k_top_p_renorm_reference,
)

# Sentinel matching tokenspeed.runtime.sampling.sampling_params._TOP_K_DISABLED.
_TOP_K_DISABLED = 1 << 30

# The fused top-k + top-p kernel ships only as a CUDA build; on ROCm the
# Python entry point resolves to a RuntimeError stub. Gate the tests instead
# of failing loudly on AMD CI.
requires_nvidia = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="fused_topk_topp kernel is NVIDIA-only",
)


@pytest.mark.parametrize("bs", [1, 4, 7])
@pytest.mark.parametrize("n", [1, 4, 8])
def test_gather_full(bs: int, n: int, device: str) -> None:
    pool_rows = 32
    values = GatherExpandScalarsInputs(
        GatherExpandScalarsInputConfig(
            pool_rows=pool_rows,
            batch_size=bs,
            n=n,
            include_min_p=True,
            include_seed=True,
            include_offsets=True,
        )
    ).generate(seed=101, metadata_seed=102 + bs * 10 + n, device=device)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        values.index,
        temperature=values.temperature,
        top_k=values.top_k,
        top_p=values.top_p,
        min_p=values.min_p,
        seed=values.seed,
        offsets=values.offsets,
        n=n,
    )
    refs = gather_expand_scalars_reference(values, n=n)

    for out, ref in zip(
        (temps, top_ks, top_ps, min_ps, seeds, offsets),
        refs,
        strict=True,
    ):
        assert out is not None
        assert ref is not None
        torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("n", [1, 5])
def test_gather_no_min_p_no_seed(n: int, device: str) -> None:
    """Verify path: drop min_p, seed, and offsets."""
    values = GatherExpandScalarsInputs(
        GatherExpandScalarsInputConfig(
            pool_rows=16,
            batch_size=8,
            n=n,
            include_min_p=False,
            include_seed=False,
            include_offsets=False,
        )
    ).generate(seed=103, metadata_seed=104 + n, device=device)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        values.index,
        temperature=values.temperature,
        top_k=values.top_k,
        top_p=values.top_p,
        n=n,
    )
    refs = gather_expand_scalars_reference(values, n=n)

    assert min_ps is None
    assert seeds is None
    assert offsets is None
    torch.testing.assert_close(temps, refs[0])
    torch.testing.assert_close(top_ks, refs[1])
    torch.testing.assert_close(top_ps, refs[2])


def test_gather_sample_basic(device: str) -> None:
    """flashinfer.py sample(): seed + offsets, no min_p, n=1."""
    values = GatherExpandScalarsInputs(
        GatherExpandScalarsInputConfig(
            pool_rows=16,
            batch_size=4,
            n=1,
            include_min_p=False,
            include_seed=True,
            include_offsets=True,
        )
    ).generate(seed=105, metadata_seed=106, device=device)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        values.index,
        temperature=values.temperature,
        top_k=values.top_k,
        top_p=values.top_p,
        seed=values.seed,
        offsets=values.offsets,
        n=1,
    )
    refs = gather_expand_scalars_reference(values, n=1)

    assert min_ps is None
    assert seeds is not None
    assert offsets is not None
    torch.testing.assert_close(temps, refs[0])
    torch.testing.assert_close(top_ks, refs[1])
    torch.testing.assert_close(top_ps, refs[2])
    torch.testing.assert_close(seeds, refs[4])
    torch.testing.assert_close(offsets, refs[5])
    assert offsets.dtype == torch.int64


def test_gather_min_p_only(device: str) -> None:
    """flashinfer_full.py verify(): min_p yes, seed no, offsets no."""
    values = GatherExpandScalarsInputs(
        GatherExpandScalarsInputConfig(
            pool_rows=16,
            batch_size=3,
            n=4,
            include_min_p=True,
            include_seed=False,
            include_offsets=False,
        )
    ).generate(seed=107, metadata_seed=108, device=device)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        values.index,
        temperature=values.temperature,
        top_k=values.top_k,
        top_p=values.top_p,
        min_p=values.min_p,
        n=4,
    )
    refs = gather_expand_scalars_reference(values, n=4)

    assert seeds is None
    assert offsets is None
    assert min_ps is not None
    torch.testing.assert_close(temps, refs[0])
    torch.testing.assert_close(top_ks, refs[1])
    torch.testing.assert_close(top_ps, refs[2])
    torch.testing.assert_close(min_ps, refs[3])


@requires_nvidia
@pytest.mark.parametrize(
    "filter_mode",
    [
        "top_k_only",
        "top_p_only",
        "top_k_top_p",
        "mixed",
    ],
)
def test_fused_topk_topp_matches_pipeline(device: str, filter_mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for fused_topk_topp_renorm test")
    bs, V = 8, 8192
    values = TopKTopPRenormInputs(
        TopKTopPRenormInputConfig(
            num_rows=bs,
            vocab_size=V,
            max_top_k=128,
            filter_mode=filter_mode,  # type: ignore[arg-type]
            disabled_top_k_value=_TOP_K_DISABLED,
        )
    ).generate(seed=109, metadata_seed=110, device=device)

    ref = top_k_top_p_renorm_reference(values.probs, values.top_k, values.top_p)
    ours = fused_topk_topp_renorm(values.probs.clone(), values.top_k, values.top_p)

    # Each kept row should renormalize to 1 within fp32 ulp tolerance.
    torch.testing.assert_close(
        ours.sum(dim=-1), torch.ones(bs, device=device), atol=1e-5, rtol=1e-5
    )
    # The fused kernel must produce the same kept set on every row. Allow
    # a single position of disagreement to absorb ties at the top-p cutoff
    # (cumulative-sum rounding can include/exclude the boundary by 1 entry).
    pos_ref = ref > 0
    pos_ours = ours > 0
    pos_diff = (pos_ref != pos_ours).sum(dim=-1).max().item()
    assert (
        pos_diff <= 1
    ), f"[{filter_mode}] kept-position mismatch up to {pos_diff} per row"
    # Renormalized values: sub-ulp accumulation order differs by row scale,
    # so 1e-5 is the right tolerance (matches the per-row sum bound).
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-4)


@requires_nvidia
def test_fused_topk_topp_workspace_size_grows_with_batch(device: str) -> None:
    """Workspace size must grow monotonically with batch and vocab so callers
    can pre-allocate a buffer sized for ``max_bs × vocab``."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for fused_topk_topp_workspace_size test")
    V = 8192
    small = fused_topk_topp_workspace_size(1, V)
    large = fused_topk_topp_workspace_size(64, V)
    assert small > 0
    assert large > small


@requires_nvidia
def test_fused_topk_topp_external_workspace(device: str) -> None:
    """Pre-allocated workspace path must produce the same result as the
    auto-allocated one, so the runtime can hoist the alloc out of the hot
    path."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for fused_topk_topp_external_workspace test")
    bs, V = 4, 8192
    values = TopKTopPRenormInputs(
        TopKTopPRenormInputConfig(
            num_rows=bs,
            vocab_size=V,
            max_top_k=128,
            include_disabled_top_k=True,
        )
    ).generate(seed=111, metadata_seed=112, device=device)

    auto = fused_topk_topp_renorm(values.probs, values.top_k, values.top_p)
    ws = torch.empty(
        fused_topk_topp_workspace_size(bs, V), dtype=torch.uint8, device=device
    )
    manual = fused_topk_topp_renorm(
        values.probs,
        values.top_k,
        values.top_p,
        workspace=ws,
    )
    torch.testing.assert_close(auto, manual, atol=0.0, rtol=0.0)


def test_gather_empty_batch(device: str) -> None:
    values = GatherExpandScalarsInputs(
        GatherExpandScalarsInputConfig(
            pool_rows=16,
            batch_size=0,
            n=5,
            include_min_p=True,
            include_seed=True,
            include_offsets=True,
        )
    ).generate(seed=113, metadata_seed=114, device=device)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        values.index,
        temperature=values.temperature,
        top_k=values.top_k,
        top_p=values.top_p,
        min_p=values.min_p,
        seed=values.seed,
        offsets=values.offsets,
        n=5,
    )

    assert temps.numel() == 0
    assert top_ks.numel() == 0
    assert top_ps.numel() == 0
    assert min_ps.numel() == 0
    assert seeds.numel() == 0
    assert offsets.numel() == 0


@pytest.mark.parametrize("rows", [1, 3, 5])
@pytest.mark.parametrize("vocab_size", [17, 257, 1025])
def test_min_p_renorm_prob(rows: int, vocab_size: int, device: str) -> None:
    values = MinPRenormInputs(
        MinPRenormInputConfig(
            num_rows=rows,
            vocab_size=vocab_size,
            min_p_dtype=torch.float32,
        )
    ).generate(seed=115, metadata_seed=rows * 1000 + vocab_size, device=device)

    out = min_p_renorm_prob(values.probs, values.min_p)
    ref = min_p_renorm_reference(values.probs, values.min_p)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(rows, device=device))


def test_min_p_renorm_prob_bf16_min_p(device: str) -> None:
    values = MinPRenormInputs(
        MinPRenormInputConfig(
            num_rows=4,
            vocab_size=513,
            min_p_dtype=torch.bfloat16,
        )
    ).generate(seed=116, metadata_seed=117, device=device)

    out = min_p_renorm_prob(values.probs, values.min_p)
    ref = min_p_renorm_reference(values.probs, values.min_p)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_min_p_renorm_prob_empty_batch(device: str) -> None:
    values = MinPRenormInputs(
        MinPRenormInputConfig(
            num_rows=0,
            vocab_size=32,
            min_p_dtype=torch.float32,
        )
    ).generate(seed=118, metadata_seed=119, device=device)

    out = min_p_renorm_prob(values.probs, values.min_p)

    assert out.shape == values.probs.shape
    assert out.dtype == values.probs.dtype
