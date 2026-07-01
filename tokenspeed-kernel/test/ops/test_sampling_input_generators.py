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
from tokenspeed_kernel.ops.sampling import argmax
from tokenspeed_kernel.ops.sampling.cuda import (
    chain_speculative_sampling_target_only,
    fused_topk_topp_renorm,
    verify_chain_greedy,
)
from tokenspeed_kernel.ops.sampling.cute_dsl import argmax_pair
from tokenspeed_kernel.ops.sampling.flashinfer import (
    min_p_sampling_from_probs as flashinfer_min_p_sampling_from_probs,
)
from tokenspeed_kernel.ops.sampling.flashinfer import softmax as flashinfer_softmax
from tokenspeed_kernel.ops.sampling.flashinfer import (
    top_k_renorm_prob as flashinfer_top_k_renorm_prob,
)
from tokenspeed_kernel.ops.sampling.flashinfer import (
    top_p_renorm_prob as flashinfer_top_p_renorm_prob,
)
from tokenspeed_kernel.ops.sampling.flashinfer import (
    top_p_renorm_probs as flashinfer_top_p_renorm_probs,
)
from tokenspeed_kernel.ops.sampling.triton import (
    gather_and_expand_scalars,
    min_p_renorm_prob,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    ArgmaxInputConfig,
    ArgmaxInputs,
    ArgmaxPairInputConfig,
    ArgmaxPairInputs,
    GatherExpandScalarsInputConfig,
    GatherExpandScalarsInputs,
    MinPRenormInputConfig,
    MinPRenormInputs,
    MinPSamplingInputConfig,
    MinPSamplingInputs,
    SoftmaxInputConfig,
    SoftmaxInputs,
    SpeculativeChainSamplingInputConfig,
    SpeculativeChainSamplingInputs,
    SpeculativeGreedyVerifyInputConfig,
    SpeculativeGreedyVerifyInputs,
    TopKTopPRenormInputConfig,
    TopKTopPRenormInputs,
    TopPRenormInputConfig,
    TopPRenormInputs,
    argmax_pair_reference,
    argmax_reference,
    gather_expand_scalars_reference,
    min_p_renorm_reference,
    min_p_sampling_reference,
    softmax_reference,
    speculative_chain_sampling_reference,
    speculative_greedy_verify_reference,
    top_k_top_p_renorm_reference,
    top_p_renorm_reference,
)

requires_nvidia = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="fused top-k/top-p sampling kernel is NVIDIA-only",
)


@pytest.mark.parametrize("solution", ["gluon", "cute_dsl"])
def test_argmax_generator_runs_registered_kernel(
    device: str,
    require,
    solution: str,
) -> None:
    dtype = torch.float32
    require("sampling", "argmax", solution, dtype, "logits")
    values = ArgmaxInputs(
        ArgmaxInputConfig(
            num_rows=8,
            vocab_size=4096,
            dtype=dtype,
            out_dtype=torch.int64,
        )
    ).generate(seed=91, metadata_seed=92, device=device)

    out = argmax(values.logits, out=values.out, solution=solution)
    ref = argmax_reference(values.logits)
    torch.cuda.synchronize()

    assert values.out is not None
    assert out.data_ptr() == values.out.data_ptr()
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_argmax_pair_generator_runs_direct_kernel(device: str) -> None:
    values = ArgmaxPairInputs(
        ArgmaxPairInputConfig(
            num_rows=8,
            vocab_size=4096,
            dtype=torch.float32,
            include_out=True,
        )
    ).generate(seed=92, metadata_seed=93, device=device)

    out = argmax_pair(values.logits, out=values.out)
    ref = argmax_pair_reference(values.logits)
    if values.logits.is_cuda:
        torch.cuda.synchronize()

    assert values.out is not None
    assert out.data_ptr() == values.out.data_ptr()
    torch.testing.assert_close(out[:, 0:1], ref[:, 0:1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[:, 1:2], ref[:, 1:2], atol=0, rtol=0)


def test_gather_expand_scalars_generator_runs_triton_kernel(device: str) -> None:
    config = GatherExpandScalarsInputConfig(
        pool_rows=32,
        batch_size=7,
        n=4,
        include_min_p=True,
        include_seed=True,
        include_offsets=True,
    )
    values = GatherExpandScalarsInputs(config).generate(
        seed=93,
        metadata_seed=94,
        device=device,
    )

    outs = gather_and_expand_scalars(
        values.index,
        temperature=values.temperature,
        top_k=values.top_k,
        top_p=values.top_p,
        min_p=values.min_p,
        seed=values.seed,
        offsets=values.offsets,
        n=config.n,
    )
    refs = gather_expand_scalars_reference(values, n=config.n)
    torch.cuda.synchronize()

    for out, ref in zip(outs, refs, strict=True):
        assert out is not None
        assert ref is not None
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_min_p_renorm_generator_runs_triton_kernel(device: str) -> None:
    values = MinPRenormInputs(
        MinPRenormInputConfig(
            num_rows=5,
            vocab_size=257,
            min_p_dtype=torch.bfloat16,
        )
    ).generate(seed=95, metadata_seed=96, device=device)

    out = min_p_renorm_prob(values.probs, values.min_p)
    ref = min_p_renorm_reference(values.probs, values.min_p)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        out.sum(dim=-1),
        torch.ones(values.probs.shape[0], device=device),
        atol=1e-6,
        rtol=1e-6,
    )


@requires_nvidia
def test_min_p_sampling_generator_runs_flashinfer_kernel(device: str) -> None:
    values = MinPSamplingInputs(
        MinPSamplingInputConfig(
            num_rows=5,
            vocab_size=257,
            min_min_p=0.05,
            max_min_p=0.5,
        )
    ).generate(seed=96, metadata_seed=97, device=device)
    expected = min_p_sampling_reference(values.probs, values.min_p)

    try:
        samples, valid = flashinfer_min_p_sampling_from_probs(
            values.probs,
            values.min_p,
            deterministic=True,
            seed=123,
            offset=0,
            return_valid=True,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer min-p sampling unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(samples, expected.samples, atol=0, rtol=0)
    torch.testing.assert_close(valid, expected.valid, atol=0, rtol=0)


@requires_nvidia
def test_softmax_generator_runs_flashinfer_kernel(device: str) -> None:
    values = SoftmaxInputs(
        SoftmaxInputConfig(
            num_rows=5,
            vocab_size=257,
            dtype=torch.bfloat16,
            temperature_mode="per_row",
        )
    ).generate(seed=97, metadata_seed=98, device=device)

    try:
        out = flashinfer_softmax(
            values.logits,
            temperature=values.temperature,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer softmax unavailable: {exc}")
    torch.cuda.synchronize()

    ref = softmax_reference(values.logits, values.temperature)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@requires_nvidia
def test_top_k_top_p_generator_runs_fused_cuda_kernel(device: str) -> None:
    values = TopKTopPRenormInputs(
        TopKTopPRenormInputConfig(
            num_rows=4,
            vocab_size=8192,
            max_top_k=128,
            include_disabled_top_k=True,
        )
    ).generate(seed=97, metadata_seed=98, device=device)

    out = fused_topk_topp_renorm(values.probs, values.top_k, values.top_p)
    ref = top_k_top_p_renorm_reference(values.probs, values.top_k, values.top_p)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@requires_nvidia
def test_top_k_top_p_generator_runs_flashinfer_renorm_sequence(device: str) -> None:
    values = TopKTopPRenormInputs(
        TopKTopPRenormInputConfig(
            num_rows=4,
            vocab_size=4096,
            max_top_k=128,
            include_disabled_top_k=True,
        )
    ).generate(seed=99, metadata_seed=100, device=device)

    try:
        top_k_out = flashinfer_top_k_renorm_prob(values.probs, values.top_k)
        out = flashinfer_top_p_renorm_prob(
            top_k_out,
            values.top_p,
            is_deterministic=True,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer top-k/top-p renormalization unavailable: {exc}")
    torch.cuda.synchronize()

    ref = top_k_top_p_renorm_reference(values.probs, values.top_k, values.top_p)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@requires_nvidia
def test_top_p_generator_runs_flashinfer_renorm_kernel(device: str) -> None:
    values = TopPRenormInputs(
        TopPRenormInputConfig(
            num_rows=4,
            vocab_size=4096,
            min_top_p=0.5,
            max_top_p=0.95,
        )
    ).generate(seed=101, metadata_seed=102, device=device)

    try:
        out = flashinfer_top_p_renorm_probs(
            values.probs,
            values.top_p,
            is_deterministic=True,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer top-p renormalization unavailable: {exc}")
    torch.cuda.synchronize()

    ref = top_p_renorm_reference(values.probs, values.top_p)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@requires_nvidia
def test_speculative_greedy_verify_generator_runs_cuda_kernel(device: str) -> None:
    config = SpeculativeGreedyVerifyInputConfig(
        batch_size=6,
        num_draft_tokens=4,
        vocab_size=4096,
        min_accepted_tokens=0,
        max_accepted_tokens=3,
    )
    values = SpeculativeGreedyVerifyInputs(config).generate(
        seed=101,
        metadata_seed=102,
        device=device,
    )
    expected = speculative_greedy_verify_reference(values)

    try:
        verify_chain_greedy(
            predicts=values.predicts,
            accept_index=values.accept_index,
            accept_token_num=values.accept_token_num,
            candidates=values.candidates,
            target_predict=values.target_predict,
            batch_size=config.batch_size,
            num_draft_tokens=config.num_draft_tokens,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"CUDA verify_chain_greedy unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(values.predicts, expected.predicts, atol=0, rtol=0)
    torch.testing.assert_close(
        values.accept_index,
        expected.accept_index,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        values.accept_token_num,
        expected.accept_token_num,
        atol=0,
        rtol=0,
    )


@requires_nvidia
def test_speculative_chain_sampling_generator_runs_cuda_kernel(device: str) -> None:
    config = SpeculativeChainSamplingInputConfig(
        batch_size=6,
        num_draft_tokens=4,
        vocab_size=64,
        include_draft_probs=False,
        min_accepted_tokens=0,
        max_accepted_tokens=3,
    )
    values = SpeculativeChainSamplingInputs(config).generate(
        seed=103,
        metadata_seed=104,
        device=device,
    )
    expected = speculative_chain_sampling_reference(
        values,
        threshold_single=config.threshold_single,
        threshold_acc=config.threshold_acc,
    )

    try:
        chain_speculative_sampling_target_only(
            predicts=values.predicts,
            accept_index=values.accept_index,
            accept_token_num=values.accept_token_num,
            candidates=values.candidates,
            uniform_samples=values.uniform_samples,
            uniform_samples_for_final_sampling=(
                values.uniform_samples_for_final_sampling
            ),
            target_probs=values.target_probs,
            draft_probs=values.draft_probs,
            threshold_single=config.threshold_single,
            threshold_acc=config.threshold_acc,
            deterministic=True,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"CUDA chain speculative sampling unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(values.predicts, expected.predicts, atol=0, rtol=0)
    torch.testing.assert_close(
        values.accept_index,
        expected.accept_index,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        values.accept_token_num,
        expected.accept_token_num,
        atol=0,
        rtol=0,
    )
