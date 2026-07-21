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
from tokenspeed_kernel import hadamard_transform
from tokenspeed_kernel.operation import OperationRegistry

torch.manual_seed(42)


@pytest.mark.parametrize("solution", ["triton", "fast_hadamard_transform"])
def test_hadamard_transform(device: str, solution: str, require) -> None:
    dtype = torch.bfloat16
    require("transform", "hadamard_transform", solution, dtype, "x")

    x = torch.randn((3, 5, 128), device=device, dtype=dtype)
    scale = 128**-0.5

    out = hadamard_transform(x, scale=scale, solution=solution)
    schema = OperationRegistry.get().lookup("transform", "hadamard_transform")
    expected = schema.reference(x, scale=scale)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=2.0e-2,
        rtol=2.0e-2,
    )
