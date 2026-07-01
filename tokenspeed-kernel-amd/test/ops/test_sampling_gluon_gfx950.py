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

MODEL_VOCABS = {
    "deepseek_v4": 129280,
    "qwen3_5": 151936,
    "minimax_m2": 200064,
}


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "AMD GFX950 is required for Gluon argmax tests", allow_module_level=True
    )


from tokenspeed_kernel_amd.ops.sampling.gluon import (  # noqa: E402
    argmax_gfx950,
)
from tokenspeed_numerics_input_generators import (  # noqa: E402
    ArgmaxInputConfig,
    ArgmaxInputs,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_argmax_matches_torch_for_dtypes(dtype):
    values = ArgmaxInputs(
        ArgmaxInputConfig(num_rows=8, vocab_size=4096, dtype=dtype)
    ).generate(seed=0xA950, metadata_seed=0xA951, device="cuda")
    out = argmax_gfx950.argmax(values.logits)
    torch.testing.assert_close(out, values.expected_indices, atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N",
    [
        (1, MODEL_VOCABS["deepseek_v4"]),
        (2, MODEL_VOCABS["deepseek_v4"]),
        (3, MODEL_VOCABS["deepseek_v4"]),
        (4, MODEL_VOCABS["deepseek_v4"]),
        (16, MODEL_VOCABS["qwen3_5"]),
        (64, MODEL_VOCABS["minimax_m2"]),
        (128, MODEL_VOCABS["qwen3_5"]),
    ],
)
def test_argmax_matches_torch_for_model_shapes(M, N):
    values = ArgmaxInputs(
        ArgmaxInputConfig(num_rows=M, vocab_size=N, dtype=torch.float32)
    ).generate(seed=M ^ N, metadata_seed=(M ^ N ^ 0xA950), device="cuda")
    out = argmax_gfx950.argmax(values.logits)
    torch.testing.assert_close(out, values.expected_indices, atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N,dtype",
    [
        (1, MODEL_VOCABS["deepseek_v4"], torch.float32),
        (4, MODEL_VOCABS["deepseek_v4"], torch.float32),
        (8, MODEL_VOCABS["deepseek_v4"], torch.float16),
        (128, MODEL_VOCABS["deepseek_v4"], torch.bfloat16),
    ],
)
def test_argmax_all_nan_rows_return_sentinel(M, N, dtype):
    values = ArgmaxInputs(
        ArgmaxInputConfig(num_rows=M, vocab_size=N, dtype=dtype, nan_pattern="all")
    ).generate(seed=M ^ N, metadata_seed=(M ^ N ^ 0xA951), device="cuda")
    out = argmax_gfx950.argmax(values.logits)
    torch.testing.assert_close(out, values.expected_indices, atol=0, rtol=0)


@pytest.mark.parametrize("M", [4, 128])
def test_argmax_ignores_nan_but_preserves_valid_negative_infinity(M):
    N = MODEL_VOCABS["deepseek_v4"]
    values = ArgmaxInputs(
        ArgmaxInputConfig(
            num_rows=M,
            vocab_size=N,
            dtype=torch.float32,
            nan_pattern="mixed",
        )
    ).generate(seed=M ^ N, metadata_seed=(M ^ N ^ 0xA952), device="cuda")

    out = argmax_gfx950.argmax(values.logits)
    torch.testing.assert_close(out, values.expected_indices, atol=0, rtol=0)


def test_argmax_returns_first_index_on_ties():
    M, N = 4, 4096
    values = ArgmaxInputs(
        ArgmaxInputConfig(
            num_rows=M,
            vocab_size=N,
            dtype=torch.float32,
            max_pattern="tied",
        )
    ).generate(seed=0xA952, metadata_seed=0xA953, device="cuda")
    torch.testing.assert_close(
        argmax_gfx950.argmax(values.logits),
        values.expected_indices,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_argmax_writes_into_strided_caller_buffer(out_dtype):
    M, N = 8, 4096
    values = ArgmaxInputs(
        ArgmaxInputConfig(num_rows=M, vocab_size=N, dtype=torch.float32)
    ).generate(seed=0xA954, metadata_seed=0xA955, device="cuda")
    storage = torch.empty(M * 2, device="cuda", dtype=out_dtype)
    out = storage[::2]
    returned = argmax_gfx950.argmax(values.logits, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.long(), values.expected_indices, atol=0, rtol=0)


def test_argmax_out_buffer_under_cuda_graph():
    M, N = 16, MODEL_VOCABS["deepseek_v4"]
    values = ArgmaxInputs(
        ArgmaxInputConfig(num_rows=M, vocab_size=N, dtype=torch.float32)
    ).generate(seed=M ^ N ^ 0xC0DE, metadata_seed=M ^ N ^ 0xC0DF, device="cuda")
    x = values.logits
    out = torch.empty(M, dtype=torch.int32, device="cuda")

    argmax_gfx950.argmax(x, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        argmax_gfx950.argmax(x, out=out)

    new_values = ArgmaxInputs(
        ArgmaxInputConfig(num_rows=M, vocab_size=N, dtype=torch.float32)
    ).generate(seed=M ^ N ^ 0xC1DE, metadata_seed=M ^ N ^ 0xC1DF, device="cuda")
    x.copy_(new_values.logits)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.long(), new_values.expected_indices, atol=0, rtol=0)
