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

import inspect
import math

import pytest
import torch
from tokenspeed_kernel.contracts.attention import MLA_PREFILL
from tokenspeed_kernel.operation import OperationRegistry
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    tensor_format,
)


def _signature(
    q_dtype: torch.dtype = torch.bfloat16,
    k_dtype: torch.dtype = torch.bfloat16,
    v_dtype: torch.dtype = torch.bfloat16,
):
    return format_signature(
        q=dense_tensor_format(q_dtype),
        k=dense_tensor_format(k_dtype),
        v=dense_tensor_format(v_dtype),
    )


def _adapter(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    softmax_scale,
    seq_lens_kv=None,
    is_causal=True,
    logit_cap=0.0,
    return_lse=False,
    out=None,
):
    return MLA_PREFILL.reference(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        softmax_scale=softmax_scale,
        seq_lens_kv=seq_lens_kv,
        is_causal=is_causal,
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
    )


def _register(*, signatures, traits=None):
    return register_kernel(
        "attention",
        "mla_prefill",
        name="test_mla_prefill",
        solution="test",
        signatures=signatures,
        traits=traits,
    )


def test_schema_is_published_with_exact_keyword_abi() -> None:
    assert OperationRegistry.get().lookup("attention", "mla_prefill") is MLA_PREFILL
    parameters = inspect.signature(MLA_PREFILL.reference).parameters
    assert list(parameters) == [
        "q",
        "k",
        "v",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "max_seqlen_q",
        "max_seqlen_kv",
        "softmax_scale",
        "seq_lens_kv",
        "is_causal",
        "logit_cap",
        "return_lse",
        "out",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    names = {
        spec.name
        for spec in KernelRegistry.get().get_for_operator("attention", "mla_prefill")
    }
    assert "mla_prefill_reference" not in names


def test_reference_defines_ragged_gqa_causal_and_output_buffer_semantics() -> None:
    q = torch.zeros((3, 4, 1), dtype=torch.float32)
    k = torch.zeros((5, 2, 1), dtype=torch.float32)
    v = torch.tensor(
        [
            [[1.0], [10.0]],
            [[2.0], [20.0]],
            [[3.0], [30.0]],
            [[4.0], [40.0]],
            [[6.0], [60.0]],
        ]
    )
    out = torch.empty((3, 4, 1), dtype=torch.float64)

    output, lse = MLA_PREFILL.reference(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 5], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_kv=3,
        softmax_scale=1.0,
        seq_lens_kv=torch.tensor([3, 2], dtype=torch.int32),
        out=out,
        return_lse=True,
    )

    assert output is out
    assert output.dtype is torch.float64
    torch.testing.assert_close(
        output[..., 0],
        torch.tensor(
            [
                [2.0, 2.0, 20.0, 20.0],
                [4.0, 4.0, 40.0, 40.0],
                [5.0, 5.0, 50.0, 50.0],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        lse,
        torch.tensor(
            [
                [math.log(3.0)] * 4,
                [0.0] * 4,
                [math.log(2.0)] * 4,
            ]
        ),
    )


def test_reference_defines_positive_logit_cap_and_natural_log_lse() -> None:
    q = torch.tensor([[[2.0]]])
    k = torch.tensor([[[1.0]], [[-1.0]]])
    v = torch.tensor([[[1.0]], [[3.0]]])
    cap = 0.5
    capped_score = cap * math.tanh(2.0)

    output, lse = MLA_PREFILL.reference(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 2], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_kv=2,
        softmax_scale=0.5,
        is_causal=False,
        logit_cap=cap,
        return_lse=True,
    )

    logits = torch.tensor([capped_score, -capped_score])
    expected_output = (torch.softmax(logits, dim=0) * torch.tensor([1.0, 3.0])).sum()
    torch.testing.assert_close(output[0, 0, 0], expected_output)
    torch.testing.assert_close(lse[0, 0], torch.logsumexp(logits, dim=0))


def test_reference_defines_empty_kv_sequence_semantics() -> None:
    output, lse = MLA_PREFILL.reference(
        q=torch.zeros((2, 1, 1)),
        k=torch.zeros((1, 1, 1)),
        v=torch.tensor([[[7.0]]]),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 0, 1], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_kv=1,
        softmax_scale=1.0,
        seq_lens_kv=torch.tensor([0, 1], dtype=torch.int32),
        return_lse=True,
    )

    torch.testing.assert_close(output[:, 0, 0], torch.tensor([0.0, 7.0]))
    assert lse[0, 0] == float("-inf")
    assert lse[1, 0] == 0.0


@pytest.mark.parametrize(
    "override, error",
    [
        ({"seq_lens_kv": torch.tensor([1], dtype=torch.int32)}, "must match"),
        ({"max_seqlen_q": 1}, "cu_seqlens_q.*inconsistent"),
        ({"max_seqlen_kv": 1}, "cu_seqlens_kv.*inconsistent"),
        ({"logit_cap": -1.0}, "non-negative"),
    ],
)
def test_reference_rejects_inconsistent_redundant_metadata(
    override, error: str
) -> None:
    arguments = {
        "q": torch.zeros((2, 1, 1)),
        "k": torch.zeros((2, 1, 1)),
        "v": torch.zeros((2, 1, 1)),
        "cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([0, 2], dtype=torch.int32),
        "max_seqlen_q": 2,
        "max_seqlen_kv": 2,
        "softmax_scale": 1.0,
        "seq_lens_kv": torch.tensor([2], dtype=torch.int32),
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=error):
        MLA_PREFILL.reference(**arguments)


def test_reference_uses_bfloat16_default_output_for_float8_query() -> None:
    fp8 = torch.float8_e4m3fn
    output = MLA_PREFILL.reference(
        q=torch.zeros((1, 1, 1), dtype=fp8),
        k=torch.zeros((1, 1, 1), dtype=fp8),
        v=torch.ones((1, 1, 1), dtype=fp8),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 1], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_kv=1,
        softmax_scale=1.0,
    )

    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output.float(), torch.ones_like(output.float()))


@pytest.mark.parametrize(
    "traits",
    [
        None,
        {
            "qk_head_dim": frozenset({128, 192}),
            "v_head_dim": frozenset({128}),
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    ],
)
def test_valid_string_registration_is_accepted(fresh_registry, traits) -> None:
    _register(signatures={_signature()}, traits=traits)(_adapter)

    assert KernelRegistry.get().get_impl("test_mla_prefill") is _adapter


def test_signature_dtypes_are_kernel_claims_not_schema_policy(fresh_registry) -> None:
    _register(signatures={_signature(torch.float32, torch.bfloat16, torch.float16)})(
        _adapter
    )

    assert KernelRegistry.get().get_impl("test_mla_prefill") is _adapter


@pytest.mark.parametrize(
    "signature, error",
    [
        (format_signature(q=dense_tensor_format(torch.float32)), "require roles"),
        (
            format_signature(
                q=tensor_format("mxfp8", torch.uint8),
                k=tensor_format("mxfp8", torch.uint8),
                v=tensor_format("mxfp8", torch.uint8),
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
        ({"head_dim": frozenset({128})}, "unknown.*head_dim"),
        ({"qk_head_dim": {192}}, "non-empty frozenset"),
        ({"qk_head_dim": frozenset({0})}, "positive integers"),
        ({"v_head_dim": frozenset({True})}, "positive integers"),
        ({"is_causal": frozenset({1})}, "values must be bool"),
    ],
)
def test_invalid_trait_claim_is_rejected(fresh_registry, traits, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _register(signatures={_signature()}, traits=traits)(_adapter)


def test_builtin_registrations_satisfy_schema() -> None:
    specs = KernelRegistry.get().get_for_operator("attention", "mla_prefill")
    assert specs
    for spec in specs:
        MLA_PREFILL.validate_registration(
            spec,
            KernelRegistry.get().get_impl(spec.name),
        )
