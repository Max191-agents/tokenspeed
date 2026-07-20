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

from collections.abc import Mapping
from numbers import Real
from typing import Any

import torch
from tokenspeed_kernel.operation import OperationSchema
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.signature import FormatSignature

__all__ = ["FP8", "quantize_fp8_reference"]


def _validate_signatures(signatures: frozenset[FormatSignature]) -> None:
    if not signatures:
        raise ValueError("quantization.fp8 requires a format signature")
    for signature in signatures:
        if not isinstance(signature, FormatSignature):
            raise TypeError("quantization.fp8 signatures must be FormatSignature")
        if tuple(name for name, _ in signature.roles) != ("x",):
            raise ValueError("quantization.fp8 signatures require only role 'x'")
        x_format = signature.format_for("x")
        if x_format is None or x_format.format != "dense" or x_format.scale is not None:
            raise ValueError(
                "quantization.fp8 role 'x' must use an unscaled dense format"
            )


def _validate_traits(traits: Mapping[str, frozenset[Any]]) -> None:
    unknown = set(traits) - {"has_scale"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown quantization.fp8 trait(s): {names}")
    if "has_scale" not in traits:
        return
    values = traits["has_scale"]
    if not isinstance(values, frozenset) or not values:
        raise TypeError(
            "quantization.fp8 trait 'has_scale' must be a non-empty frozenset"
        )
    if any(type(value) is not bool for value in values):
        raise TypeError("quantization.fp8 trait 'has_scale' values must be bool")


def quantize_fp8_reference(
    x: torch.Tensor,
    *,
    scale: float | torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Quantize ``x / scale`` to the platform-native, saturating E4M3 dtype."""
    del enable_pdl
    if not isinstance(x, torch.Tensor):
        raise TypeError("quantization.fp8 input must be a torch.Tensor")
    if x.layout is not torch.strided:
        raise ValueError("quantization.fp8 input must use a dense strided layout")
    if x.is_complex():
        raise TypeError("quantization.fp8 input must contain real values")

    values = x.to(torch.float32)
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("quantization.fp8 scale tensor must contain one value")
        if scale.device != x.device:
            raise ValueError(
                "quantization.fp8 scale tensor must be on the input device"
            )
        if scale.is_complex():
            raise TypeError("quantization.fp8 scale must contain a real value")
        scale_value: float | torch.Tensor = scale.to(torch.float32).reshape(())
    elif scale is None:
        scale_value = 1.0
    elif isinstance(scale, Real) and not isinstance(scale, bool):
        scale_value = float(scale)
    else:
        raise TypeError("quantization.fp8 scale must be a real number or scalar tensor")

    output_dtype = current_platform().fp8e4m3fn.dtype
    finite_limit = torch.finfo(output_dtype).max
    return (values / scale_value).clamp(-finite_limit, finite_limit).to(output_dtype)


FP8 = OperationSchema(
    family="quantization",
    mode="fp8",
    reference=quantize_fp8_reference,
    validate_signatures=_validate_signatures,
    validate_traits=_validate_traits,
).publish()
