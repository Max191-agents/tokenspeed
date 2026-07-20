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
from types import SimpleNamespace

import pytest
import tokenspeed_kernel.contracts.quantization as quantization_contract
import torch
from tokenspeed_kernel.contracts.quantization import FP8
from tokenspeed_kernel.operation import OperationRegistry
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    tensor_format,
)


def _signature(dtype: torch.dtype = torch.bfloat16):
    return format_signature(x=dense_tensor_format(dtype))


def _adapter(
    x: torch.Tensor,
    *,
    scale: float | torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    return FP8.reference(x, scale=scale, enable_pdl=enable_pdl)


@pytest.fixture
def e4m3_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = SimpleNamespace(fp8e4m3fn=SimpleNamespace(dtype=torch.float8_e4m3fn))
    monkeypatch.setattr(quantization_contract, "current_platform", lambda: platform)


def test_schema_is_published_and_reference_is_not_selectable() -> None:
    assert OperationRegistry.get().lookup("quantization", "fp8") is FP8
    parameters = tuple(FP8.signature.parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "x",
        "scale",
        "enable_pdl",
    ]
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:]
    )
    names = {spec.name for spec in KernelRegistry.get().list_kernels()}
    assert "quantize_fp8_reference" not in names


def test_reference_defines_saturating_cast_and_scale(e4m3_platform) -> None:
    x = torch.tensor(
        [[-1000.0, -4.0, 0.0, 4.0, 1000.0]],
        dtype=torch.float64,
    )
    scale = torch.tensor([2.0], dtype=torch.float32)

    actual = FP8.invoke_reference(x, scale=scale, enable_pdl=True)
    finite_limit = torch.finfo(torch.float8_e4m3fn).max
    expected = (
        (x.float() / 2.0).clamp(-finite_limit, finite_limit).to(torch.float8_e4m3fn)
    )

    assert actual.shape == x.shape
    assert actual.dtype is torch.float8_e4m3fn
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_reference_without_scale_is_a_saturating_fp8_cast(e4m3_platform) -> None:
    x = torch.tensor([[-1000.0, -1.0, 1.0, 1000.0]], dtype=torch.float32)

    actual = FP8.invoke_reference(x)

    expected = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("scale", "error"),
    [
        (torch.ones(2), "contain one value"),
        (True, "real number or scalar tensor"),
    ],
)
def test_reference_rejects_non_scalar_or_non_numeric_scale(
    e4m3_platform,
    scale,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        FP8.invoke_reference(torch.ones(2), scale=scale)


def test_out_of_tree_registration_keeps_dtype_policy_in_kernel_claims(
    fresh_registry,
) -> None:
    signature = _signature(torch.float64)
    register_kernel(
        "quantization",
        "fp8",
        name="external_quantize_fp8",
        solution="external",
        signatures={signature},
        traits={"has_scale": frozenset({False, True})},
    )(_adapter)

    spec = KernelRegistry.get().get_by_name("external_quantize_fp8")
    assert spec is not None
    assert spec.format_signatures == frozenset({signature})
    assert KernelRegistry.get().get_impl(spec.name) is _adapter


@pytest.mark.parametrize(
    ("signatures", "traits", "error"),
    [
        (
            {format_signature(y=dense_tensor_format(torch.float32))},
            None,
            "require only role 'x'",
        ),
        (
            {format_signature(x=tensor_format("mxfp8", torch.uint8))},
            None,
            "unscaled dense format",
        ),
        (
            {_signature()},
            {"has_scale_tensor": frozenset({True})},
            "unknown.*has_scale_tensor",
        ),
        (
            {_signature()},
            {"has_scale": frozenset({"yes"})},
            "values must be bool",
        ),
    ],
)
def test_invalid_registration_claim_is_rejected(
    fresh_registry,
    signatures,
    traits,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        register_kernel(
            "quantization",
            "fp8",
            name="invalid_quantize_fp8",
            solution="test",
            signatures=signatures,
            traits=traits,
        )(_adapter)


def test_builtin_registrations_satisfy_schema() -> None:
    specs = KernelRegistry.get().get_for_operator(*FP8.id)
    assert specs
    for spec in specs:
        FP8.validate_registration(
            spec,
            KernelRegistry.get().get_impl(spec.name),
        )

    triton = KernelRegistry.get().get_by_name("triton_quantize_fp8")
    assert triton is not None
    assert triton.solution == "triton"
    assert triton.format_signatures == frozenset(
        {_signature(torch.bfloat16), _signature(torch.float16)}
    )
    assert triton.traits == {"has_scale": frozenset({False, True})}
    assert triton.priority == Priority.PORTABLE
