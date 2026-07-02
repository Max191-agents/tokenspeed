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

import math

import pytest
import torch
from tokenspeed_numerics_input_generators import (
    HadamardTransformInputConfig,
    HadamardTransformInputs,
    hadamard_transform_reference,
)


def test_hadamard_inputs_generate_values_and_reference() -> None:
    values = HadamardTransformInputs(
        HadamardTransformInputConfig(
            batch_shape=(3, 2),
            transform_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=11, device="cpu")

    assert values.x.shape == (3, 2, 8)
    assert values.x.dtype == torch.float32
    assert values.scale == 8**-0.5

    ref = hadamard_transform_reference(values.x, scale=values.scale)
    assert ref.shape == values.x.shape
    assert ref.dtype == values.x.dtype
    torch.testing.assert_close(
        ref.norm(dim=-1),
        values.x.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_hadamard_reference_known_unscaled_transform() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    ref = hadamard_transform_reference(x, scale=1.0)
    expected = torch.tensor([[10.0, -2.0, -4.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(ref, expected, atol=0, rtol=0)


def test_hadamard_inputs_allow_empty_batch() -> None:
    values = HadamardTransformInputs(
        HadamardTransformInputConfig(
            batch_shape=(0,),
            transform_dim=16,
            dtype=torch.bfloat16,
        )
    ).generate(seed=12, device="cpu")
    ref = hadamard_transform_reference(values.x, scale=values.scale)

    assert values.x.shape == (0, 16)
    assert values.x.dtype == torch.bfloat16
    assert ref.shape == values.x.shape


def test_hadamard_inputs_seed_controls_values() -> None:
    generator = HadamardTransformInputs(
        HadamardTransformInputConfig(
            batch_shape=(2,),
            transform_dim=8,
            dtype=torch.float32,
        )
    )

    values1 = generator.generate(seed=13, device="cpu")
    values2 = generator.generate(seed=13, device="cpu")
    values3 = generator.generate(seed=14, device="cpu")

    torch.testing.assert_close(values1.x, values2.x)
    assert not torch.equal(values1.x, values3.x)


@pytest.mark.parametrize("bad_dim", [-8, 0, 3, 12])
def test_hadamard_inputs_reject_invalid_transform_dim(bad_dim: int) -> None:
    with pytest.raises(ValueError, match="transform_dim"):
        HadamardTransformInputs(
            HadamardTransformInputConfig(
                batch_shape=(1,),
                transform_dim=bad_dim,
            )
        )


def test_hadamard_inputs_reject_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        HadamardTransformInputs(
            HadamardTransformInputConfig(
                batch_shape=(1,),
                transform_dim=8,
                dtype=torch.int32,
            )
        )


def test_hadamard_inputs_reject_nonfinite_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        HadamardTransformInputs(
            HadamardTransformInputConfig(
                batch_shape=(1,),
                transform_dim=8,
                scale=math.inf,
            )
        )
