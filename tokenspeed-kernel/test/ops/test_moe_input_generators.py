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

import tokenspeed_kernel.numerics.moe  # noqa: F401
import torch
from tokenspeed_kernel.numerics.inputs import get_input_generator
from tokenspeed_kernel.numerics.moe import canonicalize_align_block_size
from tokenspeed_numerics_input_generators import (
    MoeAlignBlockSizeInputValues,
    canonicalize_moe_align_block_size,
    moe_align_block_size_reference,
)


def test_moe_align_block_size_numerics_adapter_uses_generator() -> None:
    generated = get_input_generator(
        "moe",
        "align_block_size",
        dtype=torch.int32,
        traits={},
        device="cpu",
        seed=41,
    ).generate(total_tokens=7, top_k=2, num_experts=5, block_size=4)

    values = MoeAlignBlockSizeInputValues(
        topk_ids=generated["topk_ids"],
        block_size=generated["block_size"],
        num_experts=generated["num_experts"],
    )
    ref = moe_align_block_size_reference(values)

    assert values.topk_ids.shape == (7, 2)
    assert values.topk_ids.dtype == torch.int32
    assert torch.all(values.topk_ids >= 0)
    assert torch.all(values.topk_ids < 5)
    assert ref.sorted_token_ids.numel() % values.block_size == 0
    assert ref.expert_ids.numel() * values.block_size == ref.sorted_token_ids.numel()
    torch.testing.assert_close(
        canonicalize_align_block_size(
            ref.sorted_token_ids,
            ref.expert_ids,
            ref.num_tokens_post_pad,
            values.block_size,
        ),
        canonicalize_moe_align_block_size(ref, block_size=values.block_size),
        atol=0,
        rtol=0,
    )
