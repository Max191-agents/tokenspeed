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


def _hadamard_reference(x: torch.Tensor, scale: float) -> torch.Tensor:
    transform_dim = x.shape[-1]
    out = x.float().reshape(-1, transform_dim).clone()
    stride = 1
    while stride < transform_dim:
        view = out.reshape(-1, transform_dim // (2 * stride), 2, stride)
        left = view[:, :, 0, :].clone()
        right = view[:, :, 1, :].clone()
        view[:, :, 0, :] = left + right
        view[:, :, 1, :] = left - right
        stride *= 2
    return (out * scale).reshape_as(x).to(x.dtype)


def test_fast_hadamard_transform(device: str) -> None:
    pytest.importorskip("fast_hadamard_transform")
    from tokenspeed_kernel.thirdparty.fast_hadamard_transform import hadamard_transform

    transform_dim = 64
    scale = transform_dim**-0.5
    x = (
        torch.linspace(
            -1,
            1,
            steps=5 * transform_dim,
            device=device,
            dtype=torch.float32,
        )
        .reshape(5, transform_dim)
        .to(torch.bfloat16)
    )
    expected = _hadamard_reference(x, scale)

    try:
        actual = hadamard_transform(x, scale=scale)
    except RuntimeError as exc:
        pytest.skip(f"fast_hadamard_transform unavailable for this device: {exc}")
    if actual.is_cuda:
        torch.cuda.synchronize()

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
