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

from pathlib import Path

import pytest
import torch
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_fp8_mfma_gfx950 import (
    _validate_prefill_query_scratch,
)

_SOURCE = (
    Path(__file__).parents[2] / "python/tokenspeed_kernel_amd/ops/attention/gluon/"
    "dsa_score_fp8_mfma_gfx950.py"
)


def _decompose_range_safe_query(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_fp32 = q.float()
    finite_abs = torch.where(torch.isfinite(q_fp32), q_fp32.abs(), 0.0)
    scale = torch.maximum(
        finite_abs.amax(dim=-1, keepdim=True) / 256.0,
        torch.ones((), dtype=torch.float32),
    )
    inverse_scale = scale.reciprocal()
    normalized = q_fp32 * inverse_scale
    high = normalized.to(torch.float8_e4m3fn).float()
    middle = ((normalized - high) * 32.0).to(torch.float8_e4m3fn).float()
    return high + middle / 32.0, normalized, scale


def test_range_safe_query_decomposition_spans_bf16_dynamic_range() -> None:
    pattern = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32)
    amplitudes = torch.logspace(-3, 30, 32, dtype=torch.float32)
    q = (amplitudes[:, None] * pattern).to(torch.bfloat16)

    approximate, normalized, scale = _decompose_range_safe_query(q)

    assert bool(torch.isfinite(scale).all())
    assert float(scale.min()) >= 1.0
    assert float(normalized.abs().max()) <= 256.0
    torch.testing.assert_close(
        approximate,
        normalized,
        rtol=3.0e-3,
        atol=3.0e-4,
    )


def test_two_stage_kernel_preprocesses_scaled_weights_without_dummy_component() -> None:
    source = _SOURCE.read_text()

    assert "q_scale = gl.maximum(q_amax * (1.0 / 256.0), 1.0)" in source
    assert "scaled_weights = head_weights * head_scale" in source
    assert "_dsa_preprocess_prefill_query_fp8_kernel" in source
    assert "q_tail" not in source
    assert "two_component" not in source


def test_production_short_kernel_uses_cta_uniform_range_normalization() -> None:
    source = _SOURCE.read_text()
    wide_start = source.index("def _dsa_prefill_logits_fp8_tiled_fused_wide_kernel(")
    wide_end = source.index("def _prepare_fused_range_safe_query(")
    wide_source = source[wide_start:wide_end]

    assert "q_mag_bits = q_bf16.to(gl.uint16, bitcast=True) & 0x7FFF" in wide_source
    assert "mfma_wave_mag_bits = q_wave_mag_shared.load" in wide_source
    assert "needs_scaling = gl.max(mfma_wave_mag_bits, axis=0) > 0x4380" in wide_source
    assert wide_source.count("if needs_scaling:") == 2
    assert "q_normalized = q_fp32 / producer_head_scale" in wide_source
    assert "q_mid = (q_scaled_residual * 32.0).to(gl.float8e4nv)" in wide_source
    assert "q_residual = q_fp32 - q_hi.to(gl.float32)" in wide_source
    assert "query_fp8_scratch" not in wide_source
    assert "if seq_len_sum <= 2048:" in source


def test_default_short_dispatch_amortizes_query_work_across_four_waves() -> None:
    source = _SOURCE.read_text()
    wide_start = source.index("def _dsa_prefill_logits_fp8_tiled_fused_wide_kernel(")
    wide_end = source.index("def _prepare_fused_range_safe_query(")
    wide_source = source[wide_start:wide_end]

    assert "_SHORT_BLOCK_N = 128" in source
    assert "_SHORT_NUM_WARPS = 4" in source
    assert "warps_per_cta=[1, NUM_WARPS]" in wide_source
    assert "q_hi_shared.store(q_hi)" in wide_source
    assert "q_mid_shared.store(q_mid)" in wide_source
    assert "gl.barrier()" in wide_source
    assert "q_hi_shared.load(dot_q_layout)" in wide_source
    assert "mfma_k = _load_packed_k_gather_wide(" in wide_source
    assert "gl.convert_layout(mfma_k" not in wide_source
    assert "if seq_len_sum <= 2048:" in source


@pytest.mark.parametrize("slot", [0, 1, 63, 64, 65, 511, 2048, 1_048_575])
def test_unsigned_slot_offsets_match_packed_page_layout(slot: int) -> None:
    page, row = divmod(slot, 64)

    expected_key_byte = page * (64 * 132) + row * 128
    expected_scale_word = page * ((64 * 132) // 4) + (64 * 128) // 4 + row

    assert (slot << 7) + (page << 8) == expected_key_byte
    assert slot + ((page + 1) << 11) == expected_scale_word

    source = _SOURCE.read_text()
    assert "(slot_u << 7) + (pages << 8)" in source
    assert "slot_u + ((pages + 1) << 11)" in source


def test_prefill_query_scratch_contract_accepts_planar_contiguous_buffers() -> None:
    q = torch.empty((3, 32, 128), dtype=torch.bfloat16)
    query_fp8_scratch = torch.empty((3, 2, 32, 128), dtype=torch.float8_e4m3fn)
    scaled_weights_scratch = torch.empty((3, 32), dtype=torch.float32)

    _validate_prefill_query_scratch(q, query_fp8_scratch, scaled_weights_scratch)


def test_prefill_query_scratch_contract_rejects_wrong_planar_shape() -> None:
    q = torch.empty((3, 32, 128), dtype=torch.bfloat16)
    query_fp8_scratch = torch.empty((3, 2, 31, 128), dtype=torch.float8_e4m3fn)
    scaled_weights_scratch = torch.empty((3, 32), dtype=torch.float32)

    with pytest.raises(ValueError, match="query_fp8_scratch must be preallocated"):
        _validate_prefill_query_scratch(q, query_fp8_scratch, scaled_weights_scratch)


def test_prefill_query_scratch_contract_rejects_wrong_weight_dtype() -> None:
    q = torch.empty((3, 32, 128), dtype=torch.bfloat16)
    query_fp8_scratch = torch.empty((3, 2, 32, 128), dtype=torch.float8_e4m3fn)
    scaled_weights_scratch = torch.empty((3, 32), dtype=torch.bfloat16)

    with pytest.raises(
        ValueError, match="scaled_weights_scratch must be preallocated FP32"
    ):
        _validate_prefill_query_scratch(q, query_fp8_scratch, scaled_weights_scratch)


def test_prefill_query_scratch_contract_rejects_noncontiguous_query_storage() -> None:
    q = torch.empty((3, 32, 128), dtype=torch.bfloat16)
    query_fp8_scratch = torch.empty((3, 2, 32, 256), dtype=torch.float8_e4m3fn)[
        ..., ::2
    ]
    scaled_weights_scratch = torch.empty((3, 32), dtype=torch.float32)
    assert query_fp8_scratch.shape == (3, 2, 32, 128)
    assert not query_fp8_scratch.is_contiguous()

    with pytest.raises(ValueError, match="prefill query scratch must be contiguous"):
        _validate_prefill_query_scratch(q, query_fp8_scratch, scaled_weights_scratch)
