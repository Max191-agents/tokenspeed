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
from tokenspeed_kernel.contracts.ops.attention.attn_merge_state import (
    ATTN_MERGE_STATE,
)
from tokenspeed_kernel.operation import OperationRegistry
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    tensor_format,
)


def _signature(
    out_a_dtype: torch.dtype = torch.bfloat16,
    out_b_dtype: torch.dtype = torch.bfloat16,
):
    return format_signature(
        out_a=dense_tensor_format(out_a_dtype),
        out_b=dense_tensor_format(out_b_dtype),
    )


def _adapter(
    *,
    out_a,
    lse_a,
    out_b,
    lse_b,
    lse_scale_log2,
    inplace=False,
    enable_pdl=False,
):
    return ATTN_MERGE_STATE.reference(
        out_a=out_a,
        lse_a=lse_a,
        out_b=out_b,
        lse_b=lse_b,
        lse_scale_log2=lse_scale_log2,
        inplace=inplace,
        enable_pdl=enable_pdl,
    )


def _register(*, signatures, traits=None):
    return register_kernel(
        "attention",
        "attn_merge_state",
        name="test_attn_merge_state",
        solution="test",
        signatures=signatures,
        traits=traits,
    )


def test_schema_is_published_and_reference_is_not_selectable() -> None:
    assert (
        OperationRegistry.get().lookup("attention", "attn_merge_state")
        is ATTN_MERGE_STATE
    )
    names = {
        spec.name
        for spec in KernelRegistry.get().get_for_operator(
            "attention", "attn_merge_state"
        )
    }
    assert "attn_merge_state_reference" not in names


def test_reference_defines_weighted_merge_and_inplace_semantics() -> None:
    out_a = torch.tensor([[[2.0]]])
    out_b = torch.tensor([[[6.0]]])
    lse_a = torch.tensor([[0.0]])
    lse_b = torch.tensor([[0.0]])

    output, lse = ATTN_MERGE_STATE.reference(
        out_a=out_a,
        lse_a=lse_a,
        out_b=out_b,
        lse_b=lse_b,
        inplace=True,
    )

    assert output is out_a
    assert lse is lse_a
    torch.testing.assert_close(output, torch.tensor([[[4.0]]]))
    torch.testing.assert_close(lse, torch.tensor([[math.log(2.0)]]))


@pytest.mark.parametrize("invalid", ["shape", "lse_dtype", "stride"])
def test_reference_rejects_inputs_outside_the_shared_kernel_domain(invalid) -> None:
    out_a = torch.zeros((2, 2, 2))
    out_b = torch.zeros_like(out_a)
    lse_a = torch.zeros((2, 2))
    lse_b = torch.zeros_like(lse_a)
    if invalid == "shape":
        out_b = torch.zeros((1, 2, 2))
    elif invalid == "lse_dtype":
        lse_b = lse_b.to(torch.float16)
    else:
        out_b = out_b.transpose(1, 2)

    with pytest.raises((TypeError, ValueError)):
        ATTN_MERGE_STATE.reference(
            out_a=out_a,
            lse_a=lse_a,
            out_b=out_b,
            lse_b=lse_b,
        )


def test_signature_dtypes_are_kernel_claims_not_schema_policy(fresh_registry) -> None:
    _register(
        signatures={_signature(torch.float32, torch.bfloat16)},
        traits={"head_dim": frozenset({64, 128})},
    )(_adapter)

    assert KernelRegistry.get().get_impl("test_attn_merge_state") is _adapter


@pytest.mark.parametrize(
    "signature, error",
    [
        (format_signature(out_a=dense_tensor_format(torch.float32)), "require roles"),
        (
            format_signature(
                out_a=tensor_format("mxfp8", torch.uint8),
                out_b=tensor_format("mxfp8", torch.uint8),
            ),
            "unscaled dense",
        ),
    ],
)
def test_invalid_signature_claim_is_rejected(
    fresh_registry, signature, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _register(signatures={signature})(_adapter)


@pytest.mark.parametrize(
    "traits, error",
    [
        ({"width": frozenset({128})}, "unknown.*width"),
        ({"head_dim": {128}}, "non-empty frozenset"),
        ({"head_dim": frozenset({0})}, "positive integers"),
    ],
)
def test_invalid_trait_claim_is_rejected(fresh_registry, traits, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _register(signatures={_signature()}, traits=traits)(_adapter)


def test_builtin_registrations_satisfy_schema() -> None:
    specs = KernelRegistry.get().get_for_operator("attention", "attn_merge_state")
    assert specs
    for spec in specs:
        ATTN_MERGE_STATE.validate_registration(
            spec,
            KernelRegistry.get().get_impl(spec.name),
        )
