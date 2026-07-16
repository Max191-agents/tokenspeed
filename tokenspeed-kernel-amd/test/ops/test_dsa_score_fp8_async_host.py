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

import pytest
import torch
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_fp8_async_gfx950 import (
    launch_dsa_matched_core_fp8_async_gfx950,
)


def _inputs(rows: int, capacity: int) -> tuple[torch.Tensor, ...]:
    q = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((capacity, 128), dtype=torch.float8_e4m3fn)
    scales = torch.empty((capacity,), dtype=torch.float32)
    weights = torch.empty((rows, 32), dtype=torch.float32)
    logits = torch.empty((rows, capacity), dtype=torch.float32)
    return q, k, scales, weights, logits


@pytest.mark.parametrize("rows,capacity", [(0, 7), (3, 0)])
def test_empty_inputs_return_caller_logits_without_launch(rows: int, capacity: int):
    q, k, scales, weights, logits = _inputs(rows, capacity)

    result = launch_dsa_matched_core_fp8_async_gfx950(
        q,
        k,
        scales,
        weights,
        logits,
        seq_len=capacity,
        softmax_scale=1.0,
    )

    assert result is logits


@pytest.mark.parametrize("seq_len", [-1, 9])
def test_sequence_length_must_fit_caller_capacity(seq_len: int):
    q, k, scales, weights, logits = _inputs(rows=1, capacity=8)

    with pytest.raises(ValueError, match=r"seq_len must be in \[0, 8\]"):
        launch_dsa_matched_core_fp8_async_gfx950(
            q,
            k,
            scales,
            weights,
            logits,
            seq_len=seq_len,
            softmax_scale=1.0,
        )


def test_query_must_be_contiguous_e4m3():
    q, k, scales, weights, logits = _inputs(rows=1, capacity=8)
    q = torch.empty((1, 128, 32), dtype=torch.float8_e4m3fn).transpose(1, 2)
    assert q.shape == (1, 32, 128)
    assert not q.is_contiguous()

    with pytest.raises(ValueError, match="matched-core tensors must be contiguous"):
        launch_dsa_matched_core_fp8_async_gfx950(
            q,
            k,
            scales,
            weights,
            logits,
            seq_len=8,
            softmax_scale=1.0,
        )


def test_logits_must_match_preallocated_rectangular_capacity():
    q, k, scales, weights, _ = _inputs(rows=2, capacity=8)
    logits = torch.empty((2, 7), dtype=torch.float32)

    with pytest.raises(ValueError, match="logits must be preallocated FP32"):
        launch_dsa_matched_core_fp8_async_gfx950(
            q,
            k,
            scales,
            weights,
            logits,
            seq_len=7,
            softmax_scale=1.0,
        )
