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
    ArgmaxInputConfig,
    ArgmaxInputs,
    ArgmaxPairInputConfig,
    ArgmaxPairInputs,
    GatherExpandScalarsInputConfig,
    GatherExpandScalarsInputs,
    MinPRenormInputConfig,
    MinPRenormInputs,
    SoftmaxInputConfig,
    SoftmaxInputs,
    SpeculativeChainSamplingInputConfig,
    SpeculativeChainSamplingInputs,
    SpeculativeChainSamplingInputValues,
    SpeculativeGreedyVerifyInputConfig,
    SpeculativeGreedyVerifyInputs,
    SpeculativeGreedyVerifyInputValues,
    TopKTopPRenormInputConfig,
    TopKTopPRenormInputs,
    TopPRenormInputConfig,
    TopPRenormInputs,
    argmax_pair_reference,
    argmax_reference,
    gather_expand_scalars_reference,
    min_p_renorm_reference,
    softmax_reference,
    speculative_chain_sampling_reference,
    speculative_greedy_verify_reference,
    top_k_top_p_renorm_reference,
    top_p_renorm_reference,
)


def test_argmax_inputs_generate_planted_maxima() -> None:
    values = ArgmaxInputs(
        ArgmaxInputConfig(
            num_rows=5,
            vocab_size=17,
            dtype=torch.bfloat16,
            out_dtype=torch.int32,
        )
    ).generate(seed=41, metadata_seed=42, device="cpu")

    assert values.logits.shape == (5, 17)
    assert values.logits.dtype == torch.bfloat16
    assert values.out is not None
    assert values.out.dtype == torch.int32
    torch.testing.assert_close(argmax_reference(values.logits), values.expected_indices)


def test_argmax_metadata_seed_controls_planted_indices() -> None:
    generator = ArgmaxInputs(
        ArgmaxInputConfig(num_rows=5, vocab_size=17, dtype=torch.float32)
    )

    values1 = generator.generate(seed=43, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=44, metadata_seed=99, device="cpu")

    torch.testing.assert_close(values1.expected_indices, values2.expected_indices)
    assert not torch.equal(values1.logits, values2.logits)


def test_argmax_reference_ignores_nans_and_marks_all_nan_rows() -> None:
    logits = torch.tensor(
        [[float("nan"), 1.0, 2.0], [float("nan"), float("nan"), float("nan")]],
        dtype=torch.float32,
    )
    ref = argmax_reference(logits)
    torch.testing.assert_close(ref, torch.tensor([2, -1]))


def test_argmax_pair_inputs_generate_planted_maxima() -> None:
    values = ArgmaxPairInputs(
        ArgmaxPairInputConfig(
            num_rows=5,
            vocab_size=17,
            dtype=torch.float32,
            include_out=True,
        )
    ).generate(seed=44, metadata_seed=45, device="cpu")

    assert values.logits.shape == (5, 17)
    assert values.logits.dtype == torch.float32
    assert values.out is not None
    assert values.out.shape == (5, 2)
    assert values.out.dtype == torch.float32
    torch.testing.assert_close(
        argmax_pair_reference(values.logits),
        values.expected_pair,
    )
    torch.testing.assert_close(
        values.expected_pair[:, 0],
        torch.full((5,), 8.0, dtype=torch.float32),
    )


def test_argmax_pair_reference_returns_first_tied_index() -> None:
    logits = torch.tensor(
        [
            [1.0, 4.0, 4.0, 3.0],
            [-2.0, -1.0, -1.0, -3.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        argmax_pair_reference(logits),
        torch.tensor([[4.0, 1.0], [-1.0, 1.0]], dtype=torch.float32),
    )


def test_softmax_inputs_generate_values_and_reference() -> None:
    values = SoftmaxInputs(
        SoftmaxInputConfig(
            num_rows=4,
            vocab_size=19,
            dtype=torch.bfloat16,
        )
    ).generate(seed=45, device="cpu")

    assert values.logits.shape == (4, 19)
    assert values.logits.dtype == torch.bfloat16
    assert values.temperature is None
    ref = softmax_reference(values.logits, values.temperature)
    assert ref.shape == (4, 19)
    assert ref.dtype == torch.float32
    torch.testing.assert_close(ref.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6)


def test_softmax_inputs_generate_scalar_temperature() -> None:
    values = SoftmaxInputs(
        SoftmaxInputConfig(
            num_rows=3,
            vocab_size=11,
            dtype=torch.float16,
            temperature_mode="scalar",
            min_temperature=0.7,
            max_temperature=0.7,
        )
    ).generate(seed=46, metadata_seed=47, device="cpu")

    assert isinstance(values.temperature, float)
    assert values.temperature == pytest.approx(0.7)
    ref = softmax_reference(values.logits, values.temperature)
    torch.testing.assert_close(ref.sum(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6)


def test_softmax_metadata_seed_controls_temperature_only() -> None:
    generator = SoftmaxInputs(
        SoftmaxInputConfig(
            num_rows=5,
            vocab_size=13,
            dtype=torch.float32,
            temperature_mode="per_row",
        )
    )

    values1 = generator.generate(seed=48, metadata_seed=202, device="cpu")
    values2 = generator.generate(seed=49, metadata_seed=202, device="cpu")

    assert isinstance(values1.temperature, torch.Tensor)
    assert isinstance(values2.temperature, torch.Tensor)
    assert values1.temperature.shape == (5, 1)
    torch.testing.assert_close(values1.temperature, values2.temperature)
    assert not torch.equal(values1.logits, values2.logits)


def test_speculative_greedy_verify_inputs_generate_values_and_reference() -> None:
    config = SpeculativeGreedyVerifyInputConfig(
        batch_size=5,
        num_draft_tokens=4,
        vocab_size=97,
        min_accepted_tokens=0,
        max_accepted_tokens=3,
    )
    values = SpeculativeGreedyVerifyInputs(config).generate(
        seed=141,
        metadata_seed=142,
        device="cpu",
    )
    refs = speculative_greedy_verify_reference(values)

    assert values.predicts.shape == (20,)
    assert values.predicts.dtype == torch.int32
    assert values.accept_index.shape == (5, 4)
    assert values.accept_token_num.shape == (5,)
    assert values.candidates.dtype == torch.int32
    assert values.target_predict.dtype == torch.int64
    assert torch.all(refs.accept_token_num >= 0)
    assert torch.all(refs.accept_token_num <= 3)

    positions = torch.arange(config.num_draft_tokens).unsqueeze(0)
    expected_valid = positions <= refs.accept_token_num.unsqueeze(1)
    assert torch.equal(refs.accept_index >= 0, expected_valid)
    valid_predicts = refs.accept_index[refs.accept_index >= 0].to(torch.int64)
    torch.testing.assert_close(
        refs.predicts[valid_predicts],
        values.target_predict.reshape(-1).to(torch.int32)[valid_predicts],
    )
    invalid_predicts = refs.predicts[refs.predicts < 0]
    torch.testing.assert_close(invalid_predicts, torch.full_like(invalid_predicts, -1))


def test_speculative_greedy_verify_metadata_seed_controls_acceptance() -> None:
    generator = SpeculativeGreedyVerifyInputs(
        SpeculativeGreedyVerifyInputConfig(
            batch_size=7,
            num_draft_tokens=5,
            vocab_size=101,
        )
    )

    values1 = generator.generate(seed=143, metadata_seed=900, device="cpu")
    values2 = generator.generate(seed=144, metadata_seed=900, device="cpu")
    refs1 = speculative_greedy_verify_reference(values1)
    refs2 = speculative_greedy_verify_reference(values2)

    torch.testing.assert_close(refs1.accept_token_num, refs2.accept_token_num)
    assert not torch.equal(values1.candidates, values2.candidates)


def test_speculative_greedy_verify_reference_known_prefixes() -> None:
    values = SpeculativeGreedyVerifyInputValues(
        predicts=torch.full((8,), -1, dtype=torch.int32),
        accept_index=torch.full((2, 4), -1, dtype=torch.int32),
        accept_token_num=torch.empty((2,), dtype=torch.int32),
        candidates=torch.tensor(
            [[10, 20, 30, 40], [11, 12, 13, 14]],
            dtype=torch.int32,
        ),
        target_predict=torch.tensor(
            [[20, 30, 99, 7], [99, 12, 13, 14]],
            dtype=torch.int64,
        ),
    )

    refs = speculative_greedy_verify_reference(values)

    torch.testing.assert_close(
        refs.predicts,
        torch.tensor([20, 30, 99, -1, 99, -1, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        refs.accept_token_num,
        torch.tensor([2, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        refs.accept_index,
        torch.tensor([[0, 1, 2, -1], [4, -1, -1, -1]], dtype=torch.int32),
    )


def test_speculative_chain_sampling_inputs_generate_values_and_reference() -> None:
    config = SpeculativeChainSamplingInputConfig(
        batch_size=5,
        num_draft_tokens=4,
        vocab_size=97,
        include_draft_probs=False,
        min_accepted_tokens=0,
        max_accepted_tokens=3,
    )
    values = SpeculativeChainSamplingInputs(config).generate(
        seed=151,
        metadata_seed=152,
        device="cpu",
    )
    refs = speculative_chain_sampling_reference(
        values,
        threshold_single=config.threshold_single,
        threshold_acc=config.threshold_acc,
    )

    assert values.predicts.shape == (20,)
    assert values.accept_index.shape == (5, 4)
    assert values.candidates.dtype == torch.int32
    assert values.target_probs.shape == (5, 4, 97)
    assert values.draft_probs is None
    assert torch.all(refs.accept_token_num >= 0)
    assert torch.all(refs.accept_token_num <= 3)
    positions = torch.arange(config.num_draft_tokens).unsqueeze(0)
    assert torch.equal(
        refs.accept_index >= 0,
        positions <= refs.accept_token_num.unsqueeze(1),
    )
    valid_indices = refs.accept_index[refs.accept_index >= 0].to(torch.int64)
    assert torch.all(refs.predicts[valid_indices] >= 0)


def test_speculative_chain_sampling_metadata_seed_controls_acceptance() -> None:
    generator = SpeculativeChainSamplingInputs(
        SpeculativeChainSamplingInputConfig(
            batch_size=7,
            num_draft_tokens=5,
            vocab_size=101,
        )
    )

    values1 = generator.generate(seed=153, metadata_seed=901, device="cpu")
    values2 = generator.generate(seed=154, metadata_seed=901, device="cpu")
    refs1 = speculative_chain_sampling_reference(
        values1,
        threshold_single=generator.config.threshold_single,
        threshold_acc=generator.config.threshold_acc,
    )
    refs2 = speculative_chain_sampling_reference(
        values2,
        threshold_single=generator.config.threshold_single,
        threshold_acc=generator.config.threshold_acc,
    )

    torch.testing.assert_close(refs1.accept_token_num, refs2.accept_token_num)
    assert not torch.equal(values1.candidates, values2.candidates)


def test_speculative_chain_sampling_reference_known_rejection() -> None:
    values = SpeculativeChainSamplingInputValues(
        predicts=torch.full((4,), -1, dtype=torch.int32),
        accept_index=torch.full((1, 4), -1, dtype=torch.int32),
        accept_token_num=torch.empty((1,), dtype=torch.int32),
        candidates=torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        uniform_samples=torch.tensor([[0.25, 0.99, 0.25, 0.25]], dtype=torch.float32),
        uniform_samples_for_final_sampling=torch.tensor([0.0], dtype=torch.float32),
        target_probs=torch.zeros((1, 4, 16), dtype=torch.float32),
        draft_probs=torch.zeros((1, 4, 16), dtype=torch.float32),
    )
    values.target_probs[0, 0, 11] = 1.0
    values.target_probs[0, 1, 12] = 0.1
    values.target_probs[0, 1, 5] = 0.9

    refs = speculative_chain_sampling_reference(
        values,
        threshold_single=0.9,
        threshold_acc=1.0,
    )

    torch.testing.assert_close(
        refs.predicts,
        torch.tensor([11, 5, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        refs.accept_token_num, torch.tensor([1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        refs.accept_index,
        torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
    )
    assert refs.draft_probs is not None
    assert refs.draft_probs[0, 1, 12].item() == pytest.approx(0.1)


def test_gather_expand_scalars_inputs_generate_values_and_reference() -> None:
    config = GatherExpandScalarsInputConfig(
        pool_rows=11,
        batch_size=4,
        n=3,
        include_min_p=True,
        include_seed=True,
        include_offsets=True,
    )
    values = GatherExpandScalarsInputs(config).generate(
        seed=45,
        metadata_seed=46,
        device="cpu",
    )

    assert values.index.shape == (4,)
    assert values.temperature.shape == (11,)
    assert values.top_k.dtype == torch.int32
    assert values.min_p is not None
    assert values.seed is not None
    assert values.offsets is not None

    refs = gather_expand_scalars_reference(values, n=config.n)
    assert refs[0].shape == (12,)
    assert refs[3] is not None
    assert refs[4] is not None
    assert refs[5] is not None
    assert refs[5].dtype == torch.int64


def test_gather_expand_scalars_optional_streams() -> None:
    config = GatherExpandScalarsInputConfig(
        pool_rows=8,
        batch_size=3,
        n=2,
        include_min_p=False,
        include_seed=False,
        include_offsets=False,
    )
    values = GatherExpandScalarsInputs(config).generate(seed=47, device="cpu")
    refs = gather_expand_scalars_reference(values, n=config.n)

    assert values.min_p is None
    assert values.seed is None
    assert values.offsets is None
    assert refs[3] is None
    assert refs[4] is None
    assert refs[5] is None


def test_min_p_renorm_inputs_generate_probabilities_and_reference() -> None:
    config = MinPRenormInputConfig(
        num_rows=4,
        vocab_size=23,
        min_p_dtype=torch.bfloat16,
    )
    values = MinPRenormInputs(config).generate(
        seed=48,
        metadata_seed=49,
        device="cpu",
    )

    assert values.probs.shape == (4, 23)
    torch.testing.assert_close(values.probs.sum(dim=-1), torch.ones(4))
    assert values.min_p.dtype == torch.bfloat16
    ref = min_p_renorm_reference(values.probs, values.min_p)
    torch.testing.assert_close(ref.sum(dim=-1), torch.ones(4))
    assert torch.count_nonzero(ref == 0) > 0


def test_top_p_renorm_inputs_generate_values_and_reference() -> None:
    config = TopPRenormInputConfig(
        num_rows=5,
        vocab_size=29,
        dtype=torch.float32,
        min_top_p=0.4,
        max_top_p=0.9,
    )
    values = TopPRenormInputs(config).generate(
        seed=52,
        metadata_seed=53,
        device="cpu",
    )

    assert values.probs.shape == (5, 29)
    assert values.probs.dtype == torch.float32
    torch.testing.assert_close(
        values.probs.float().sum(dim=-1),
        torch.ones(5),
        atol=1e-6,
        rtol=1e-6,
    )
    assert values.top_p.shape == (5,)
    assert values.top_p.min() >= 0.4
    assert values.top_p.max() <= 0.9
    ref = top_p_renorm_reference(values.probs, values.top_p)
    torch.testing.assert_close(
        ref.float().sum(dim=-1),
        torch.ones(5),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.count_nonzero(ref == 0) > 0


def test_top_p_renorm_reference_known_threshold() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    top_p = torch.tensor([0.6], dtype=torch.float32)

    ref = top_p_renorm_reference(probs, top_p)

    torch.testing.assert_close(
        ref,
        torch.tensor([[0.625, 0.375, 0.0]], dtype=torch.float32),
    )


def test_top_k_top_p_renorm_inputs_generate_values_and_reference() -> None:
    config = TopKTopPRenormInputConfig(
        num_rows=6,
        vocab_size=31,
        max_top_k=8,
        include_disabled_top_k=True,
    )
    values = TopKTopPRenormInputs(config).generate(
        seed=50,
        metadata_seed=51,
        device="cpu",
    )

    assert values.probs.shape == (6, 31)
    torch.testing.assert_close(values.probs.sum(dim=-1), torch.ones(6))
    assert values.top_k.shape == (6,)
    assert torch.equal(values.top_k[1::2], torch.full((3,), 31, dtype=torch.int32))
    assert values.top_p.min() >= 0.5
    assert values.top_p.max() <= 1.0
    ref = top_k_top_p_renorm_reference(values.probs, values.top_k, values.top_p)
    torch.testing.assert_close(ref.sum(dim=-1), torch.ones(6), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("bad_dtype", [torch.float8_e4m3fn, torch.int32])
def test_argmax_rejects_invalid_dtype(bad_dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="dtype"):
        ArgmaxInputs(
            ArgmaxInputConfig(
                num_rows=1,
                vocab_size=8,
                dtype=bad_dtype,
            )
        )


def test_gather_expand_rejects_invalid_repeat_count() -> None:
    with pytest.raises(ValueError, match="n"):
        GatherExpandScalarsInputs(
            GatherExpandScalarsInputConfig(
                pool_rows=8,
                batch_size=1,
                n=0,
            )
        )


def test_argmax_pair_reference_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        argmax_pair_reference(torch.randn(4))


def test_softmax_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature_mode"):
        SoftmaxInputs(
            SoftmaxInputConfig(
                num_rows=1,
                vocab_size=8,
                dtype=torch.float32,
                temperature_mode="batch",  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="min_temperature"):
        SoftmaxInputs(
            SoftmaxInputConfig(
                num_rows=1,
                vocab_size=8,
                dtype=torch.float32,
                temperature_mode="scalar",
                min_temperature=0.0,
            )
        )
    with pytest.raises(ValueError, match="temperature tensor"):
        softmax_reference(
            torch.randn(2, 4),
            torch.ones(3, dtype=torch.float32),
        )


def test_top_k_top_p_rejects_too_large_max_top_k() -> None:
    with pytest.raises(ValueError, match="max_top_k"):
        TopKTopPRenormInputs(
            TopKTopPRenormInputConfig(
                num_rows=1,
                vocab_size=8,
                max_top_k=9,
            )
        )


def test_top_p_renorm_rejects_invalid_threshold_bounds() -> None:
    with pytest.raises(ValueError, match="top-p bounds"):
        TopPRenormInputs(
            TopPRenormInputConfig(
                num_rows=1,
                vocab_size=8,
                min_top_p=0.0,
            )
        )
    with pytest.raises(ValueError, match="top_p thresholds"):
        top_p_renorm_reference(
            torch.full((1, 4), 0.25),
            torch.tensor([1.1]),
        )


def test_speculative_greedy_verify_rejects_invalid_accept_range() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        SpeculativeGreedyVerifyInputs(
            SpeculativeGreedyVerifyInputConfig(
                batch_size=1,
                num_draft_tokens=4,
                vocab_size=1,
            )
        )
    with pytest.raises(ValueError, match="max_accepted_tokens"):
        SpeculativeGreedyVerifyInputs(
            SpeculativeGreedyVerifyInputConfig(
                batch_size=1,
                num_draft_tokens=4,
                vocab_size=16,
                max_accepted_tokens=4,
            )
        )
    with pytest.raises(ValueError, match="min_accepted_tokens"):
        SpeculativeGreedyVerifyInputs(
            SpeculativeGreedyVerifyInputConfig(
                batch_size=1,
                num_draft_tokens=4,
                vocab_size=16,
                min_accepted_tokens=3,
                max_accepted_tokens=2,
            )
        )


def test_speculative_chain_sampling_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="threshold_acc"):
        SpeculativeChainSamplingInputs(
            SpeculativeChainSamplingInputConfig(
                batch_size=1,
                num_draft_tokens=4,
                vocab_size=16,
                threshold_acc=0.0,
            )
        )
    with pytest.raises(ValueError, match="threshold_single=0"):
        SpeculativeChainSamplingInputs(
            SpeculativeChainSamplingInputConfig(
                batch_size=1,
                num_draft_tokens=4,
                vocab_size=16,
                threshold_single=0.0,
                min_accepted_tokens=0,
                max_accepted_tokens=2,
            )
        )
    with pytest.raises(ValueError, match="candidate token IDs"):
        speculative_chain_sampling_reference(
            SpeculativeChainSamplingInputValues(
                predicts=torch.full((1,), -1, dtype=torch.int32),
                accept_index=torch.full((1, 1), -1, dtype=torch.int32),
                accept_token_num=torch.empty((1,), dtype=torch.int32),
                candidates=torch.tensor([[99]], dtype=torch.int32),
                uniform_samples=torch.zeros((1, 1), dtype=torch.float32),
                uniform_samples_for_final_sampling=torch.zeros(
                    (1,), dtype=torch.float32
                ),
                target_probs=torch.ones((1, 1, 2), dtype=torch.float32),
                draft_probs=None,
            ),
            threshold_single=0.9,
            threshold_acc=1.0,
        )
