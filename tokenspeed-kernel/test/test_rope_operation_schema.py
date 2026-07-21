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
from tokenspeed_kernel.contracts.ops.embedding.rope import ROPE
from tokenspeed_kernel.operation import OperationRegistry
from tokenspeed_kernel.ops.embedding import FusedMLASetKVBufferArg, FusedSetKVBufferArg
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    tensor_format,
)


def _rope_signature(q_dtype=torch.bfloat16, k_dtype=torch.bfloat16):
    return format_signature(
        q=dense_tensor_format(q_dtype),
        k=dense_tensor_format(k_dtype),
    )


def _rope_adapter(
    *,
    positions,
    q,
    k,
    head_size,
    cos_sin_cache,
    is_neox=True,
    fused_set_kv_buffer_arg=None,
    fused_mla_set_kv_buffer_arg=None,
    q_rope_out=None,
    k_rope_out=None,
    enable_pdl=False,
):
    return ROPE.reference(
        positions=positions,
        q=q,
        k=k,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
        fused_mla_set_kv_buffer_arg=fused_mla_set_kv_buffer_arg,
        q_rope_out=q_rope_out,
        k_rope_out=k_rope_out,
        enable_pdl=enable_pdl,
    )


def test_schema_is_published_and_reference_is_not_selectable() -> None:
    catalog = OperationRegistry.get()
    assert catalog.lookup("embedding", "rope") is ROPE
    names = {spec.name for spec in KernelRegistry.get().list_kernels()}
    assert "rope_reference" not in names


@pytest.mark.parametrize(
    "is_neox, expected",
    [
        (True, [-3.0, -4.0, 1.0, 2.0, 9.0]),
        (False, [-2.0, 1.0, -4.0, 3.0, 9.0]),
    ],
)
def test_rope_reference_defines_layout_and_partial_rotation(is_neox, expected) -> None:
    positions = torch.tensor([0], dtype=torch.int64)
    cache = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0, 9.0]])
    k = q.clone()
    q_out = torch.empty_like(q)

    ROPE.reference(
        positions=positions,
        q=q,
        k=k,
        head_size=5,
        cos_sin_cache=cache,
        is_neox=is_neox,
        q_rope_out=q_out,
    )

    torch.testing.assert_close(q, torch.tensor([[1.0, 2.0, 3.0, 4.0, 9.0]]))
    torch.testing.assert_close(q_out, torch.tensor([expected]))
    torch.testing.assert_close(k, torch.tensor([expected]))


def test_rope_reference_defines_standard_fused_cache_write() -> None:
    positions = torch.tensor([0], dtype=torch.int64)
    cache = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    k = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    value = torch.tensor([[[9.0, 10.0, 11.0, 12.0]]])
    cache_loc = torch.tensor([2], dtype=torch.int32)
    k_buffer = torch.zeros((3, 4))
    v_buffer = torch.zeros_like(k_buffer)

    ROPE.reference(
        positions=positions,
        q=q,
        k=k,
        head_size=4,
        cos_sin_cache=cache,
        fused_set_kv_buffer_arg=FusedSetKVBufferArg(
            value=value,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            k_scale=None,
            v_scale=None,
            cache_loc=cache_loc,
        ),
    )

    torch.testing.assert_close(k_buffer[2], k[0])
    torch.testing.assert_close(v_buffer[2], value[0, 0])


def test_rope_reference_defines_mla_fused_cache_write() -> None:
    positions = torch.tensor([0], dtype=torch.int64)
    cache = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    q = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    k = torch.tensor([[[5.0, 6.0, 7.0, 8.0]]])
    k_original = k.clone()
    q_out = torch.empty_like(q)
    kv_buffer = torch.zeros((3, 6))

    ROPE.reference(
        positions=positions,
        q=q,
        k=k,
        head_size=4,
        cos_sin_cache=cache,
        fused_mla_set_kv_buffer_arg=FusedMLASetKVBufferArg(
            k_nope=torch.tensor([[[11.0, 12.0]]]),
            kv_buffer=kv_buffer,
            cache_loc=torch.tensor([1], dtype=torch.int32),
        ),
        q_rope_out=q_out,
    )

    torch.testing.assert_close(q_out[0, 0], torch.tensor([-3.0, -4.0, 1.0, 2.0]))
    torch.testing.assert_close(k, k_original)
    torch.testing.assert_close(
        kv_buffer[1], torch.tensor([11.0, 12.0, -7.0, -8.0, 5.0, 6.0])
    )


def test_rope_registration_keeps_dtype_policy_in_kernel_claims(fresh_registry) -> None:
    register_kernel(
        "embedding",
        "rope",
        name="test_rope",
        solution="test",
        signatures={_rope_signature(torch.float32, torch.float16)},
        traits={
            "head_size": frozenset({64}),
            **{
                name: frozenset({False, True})
                for name in (
                    "partial_rotary",
                    "is_neox",
                    "has_fused_kv",
                    "has_fused_mla_kv",
                    "has_q_out",
                    "has_k_out",
                )
            },
        },
    )(_rope_adapter)

    assert KernelRegistry.get().get_impl("test_rope") is _rope_adapter


@pytest.mark.parametrize(
    "signatures, traits, error",
    [
        (
            {format_signature(q=dense_tensor_format(torch.float32))},
            None,
            "require roles",
        ),
        (
            {
                format_signature(
                    q=tensor_format("mxfp8", torch.uint8),
                    k=tensor_format("mxfp8", torch.uint8),
                )
            },
            None,
            "unscaled dense",
        ),
        (
            {_rope_signature()},
            {"head_dim": frozenset({64})},
            "unknown.*head_dim",
        ),
    ],
)
def test_invalid_rope_registration_claim_is_rejected(
    fresh_registry, signatures, traits, error
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        register_kernel(
            "embedding",
            "rope",
            name="invalid_rope",
            solution="test",
            signatures=signatures,
            traits=traits,
        )(_rope_adapter)


def test_builtin_registrations_satisfy_schema() -> None:
    specs = KernelRegistry.get().get_for_operator(*ROPE.id)
    assert specs
    for spec in specs:
        ROPE.validate_registration(
            spec,
            KernelRegistry.get().get_impl(spec.name),
        )
