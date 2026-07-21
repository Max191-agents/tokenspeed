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
from tokenspeed_kernel.contracts.ops.embedding.rope_mla import ROPE_MLA
from tokenspeed_kernel.operation import OperationRegistry
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
)


def _mla_signature(dtype=torch.bfloat16):
    tensor = dense_tensor_format(dtype)
    return format_signature(
        q_rope=tensor,
        k_rope=tensor,
        q_nope=tensor,
        k_nope=tensor,
    )


def _mla_adapter(
    *,
    positions,
    q_rope,
    k_rope,
    q_nope,
    k_nope,
    cos_sin_cache,
    q_rope_out,
    k_rope_out,
    q_nope_out,
    k_nope_out,
    is_neox=True,
    quant_scale_q=1.0,
    quant_scale_kv=1.0,
    enable_pdl=False,
):
    return ROPE_MLA.reference(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        q_rope_out=q_rope_out,
        k_rope_out=k_rope_out,
        q_nope_out=q_nope_out,
        k_nope_out=k_nope_out,
        is_neox=is_neox,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        enable_pdl=enable_pdl,
    )


def test_schema_is_published_and_reference_is_not_selectable() -> None:
    catalog = OperationRegistry.get()
    assert catalog.lookup("embedding", "rope_mla") is ROPE_MLA
    names = {spec.name for spec in KernelRegistry.get().list_kernels()}
    assert "rope_mla_reference" not in names


def test_rope_mla_reference_defines_scale_and_output_cast() -> None:
    positions = torch.tensor([0], dtype=torch.int64)
    cache = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    q_rope = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    k_rope = q_rope.clone()
    q_nope = torch.tensor([[[5.0, 6.0]]])
    k_nope = q_nope.clone()
    outputs = [
        torch.empty(value.shape, dtype=torch.float16)
        for value in (q_rope, k_rope, q_nope, k_nope)
    ]

    ROPE_MLA.reference(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cache,
        q_rope_out=outputs[0],
        k_rope_out=outputs[1],
        q_nope_out=outputs[2],
        k_nope_out=outputs[3],
        quant_scale_q=2.0,
        quant_scale_kv=3.0,
    )

    torch.testing.assert_close(
        outputs[0][0, 0], torch.tensor([-6.0, -8.0, 2.0, 4.0], dtype=torch.float16)
    )
    torch.testing.assert_close(
        outputs[1][0, 0], torch.tensor([-9.0, -12.0, 3.0, 6.0], dtype=torch.float16)
    )
    torch.testing.assert_close(
        outputs[2][0, 0], torch.tensor([10.0, 12.0], dtype=torch.float16)
    )
    torch.testing.assert_close(
        outputs[3][0, 0], torch.tensor([15.0, 18.0], dtype=torch.float16)
    )


def test_rope_mla_registration_keeps_dtype_policy_in_kernel_claims(
    fresh_registry,
) -> None:
    signature = format_signature(
        q_rope=dense_tensor_format(torch.float32),
        k_rope=dense_tensor_format(torch.float16),
        q_nope=dense_tensor_format(torch.bfloat16),
        k_nope=dense_tensor_format(torch.float64),
    )

    register_kernel(
        "embedding",
        "rope_mla",
        name="test_rope_mla",
        solution="test",
        signatures={signature},
        traits={
            "is_neox": frozenset({True}),
            "quantize_dtype": frozenset({torch.float8_e5m2}),
            "has_scale_q_tensor": frozenset({False, True}),
            "has_scale_kv_tensor": frozenset({False, True}),
        },
    )(_mla_adapter)

    assert KernelRegistry.get().get_impl("test_rope_mla") is _mla_adapter


@pytest.mark.parametrize(
    "traits, error",
    [
        (
            {"quantize_dtype": frozenset({torch.float32})},
            "values must be FP8 dtypes",
        ),
        (
            {"quantize_dtype": frozenset({"fp8"})},
            "values must be FP8 dtypes",
        ),
    ],
)
def test_invalid_rope_mla_registration_claim_is_rejected(
    fresh_registry, traits, error
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        register_kernel(
            "embedding",
            "rope_mla",
            name="invalid_rope_mla",
            solution="test",
            signatures={_mla_signature()},
            traits=traits,
        )(_mla_adapter)


def test_builtin_registrations_satisfy_schema() -> None:
    specs = KernelRegistry.get().get_for_operator(*ROPE_MLA.id)
    assert specs
    for spec in specs:
        ROPE_MLA.validate_registration(
            spec,
            KernelRegistry.get().get_impl(spec.name),
        )
