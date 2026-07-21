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
from tokenspeed_kernel.contracts.ops.transform.hadamard_transform import (
    HADAMARD_TRANSFORM,
)
from tokenspeed_kernel.operation import OperationRegistry
from tokenspeed_kernel.registry import (
    KernelRegistry,
    load_builtin_kernels,
    register_kernel,
)
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    tensor_format,
)


def _register(
    *,
    signatures,
    traits=None,
    name="test_hadamard",
):
    return register_kernel(
        "transform",
        "hadamard_transform",
        name=name,
        solution="test",
        signatures=signatures,
        traits=traits,
    )


def test_schema_is_published_and_reference_is_not_selectable() -> None:
    assert (
        OperationRegistry.get().lookup("transform", "hadamard_transform")
        is HADAMARD_TRANSFORM
    )
    names = {
        spec.name
        for spec in KernelRegistry.get().get_for_operator(
            "transform", "hadamard_transform"
        )
    }
    assert "hadamard_transform_reference" not in names


def test_reference_matches_butterfly_definition() -> None:
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    actual = HADAMARD_TRANSFORM.reference(x, scale=0.5)

    expected = torch.tensor(
        [[3.0, -1.0, -2.0, 0.0], [11.0, -1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.equal(actual, expected)


def test_reference_zero_pads_non_power_of_two_width() -> None:
    x = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    transform_width = 8
    matrix = torch.tensor(
        [
            [
                (-1.0) ** ((row & column).bit_count())
                for column in range(transform_width)
            ]
            for row in range(transform_width)
        ]
    )
    padded = torch.nn.functional.pad(x, (0, transform_width - x.shape[-1]))

    actual = HADAMARD_TRANSFORM.reference(x)

    expected = (padded @ matrix.T)[..., : x.shape[-1]]
    assert torch.equal(actual, expected)


def test_reference_preserves_float64_precision() -> None:
    x = torch.tensor([[1.0 + 2.0**-40, 1.0]], dtype=torch.float64)

    actual = HADAMARD_TRANSFORM.reference(x)

    expected = torch.stack((x[..., 0] + x[..., 1], x[..., 0] - x[..., 1]), -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("traits", [None, {"last_dim": frozenset({64, 128})}])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_valid_legacy_registration_is_accepted(fresh_registry, traits, dtype) -> None:
    @_register(
        signatures={format_signature(x=dense_tensor_format(dtype))},
        traits=traits,
    )
    def adapter(x, *, scale=1.0):
        return x * scale

    assert KernelRegistry.get().get_impl("test_hadamard") is adapter


@pytest.mark.parametrize(
    "signature, error",
    [
        (
            format_signature(input=dense_tensor_format(torch.float32)),
            "only role 'x'",
        ),
        (
            format_signature(x=tensor_format("mxfp4", torch.uint8)),
            "unscaled dense",
        ),
    ],
)
def test_invalid_signature_claim_is_rejected(
    fresh_registry, signature, error: str
) -> None:
    with pytest.raises(ValueError, match=error):

        @_register(signatures={signature})
        def adapter(x, *, scale=1.0):
            return x * scale


@pytest.mark.parametrize(
    "traits, error",
    [
        ({"width": frozenset({128})}, "unknown.*width"),
        ({"last_dim": {128}}, "non-empty frozenset"),
        ({"last_dim": frozenset({0})}, "positive integers"),
    ],
)
def test_invalid_trait_claim_is_rejected(fresh_registry, traits, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):

        @_register(
            signatures={format_signature(x=dense_tensor_format(torch.float32))},
            traits=traits,
        )
        def adapter(x, *, scale=1.0):
            return x * scale


def test_builtin_registrations_satisfy_schema() -> None:
    load_builtin_kernels()
    specs = KernelRegistry.get().get_for_operator("transform", "hadamard_transform")
    assert specs
    for spec in specs:
        HADAMARD_TRANSFORM.validate_registration(
            spec,
            KernelRegistry.get().get_impl(spec.name),
        )
