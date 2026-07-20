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
from typing import Any

import torch
from tokenspeed_kernel.operation import OperationSchema
from tokenspeed_kernel.signature import FormatSignature

__all__ = ["HADAMARD_TRANSFORM", "hadamard_transform_reference"]


def _validate_signatures(signatures: frozenset[FormatSignature]) -> None:
    if not signatures:
        raise ValueError("transform.hadamard_transform requires a format signature")
    for signature in signatures:
        if not isinstance(signature, FormatSignature):
            raise TypeError("Hadamard format signatures must be FormatSignature values")
        if tuple(name for name, _ in signature.roles) != ("x",):
            raise ValueError("Hadamard format signatures must contain only role 'x'")
        x_format = signature.format_for("x")
        if x_format is None or x_format.format != "dense" or x_format.scale is not None:
            raise ValueError("Hadamard role 'x' must use an unscaled dense format")


def _validate_traits(traits: Mapping[str, frozenset[Any]]) -> None:
    unknown = set(traits) - {"last_dim"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown transform.hadamard_transform trait(s): {names}")
    if "last_dim" not in traits:
        return
    values = traits["last_dim"]
    if not isinstance(values, frozenset) or not values:
        raise TypeError("Hadamard trait 'last_dim' must be a non-empty frozenset")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Hadamard trait 'last_dim' values must be positive integers")


def hadamard_transform_reference(
    x: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a Sylvester Hadamard transform along the last dimension."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Hadamard input must be a torch.Tensor")
    if x.dim() == 0:
        raise ValueError("Hadamard input must have at least one dimension")
    width = x.shape[-1]
    if width <= 0:
        raise ValueError("Hadamard input last dimension must be positive")
    if not x.is_floating_point():
        raise TypeError("Hadamard input must use a floating-point dtype")

    transform_width = 1 << (width - 1).bit_length()
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    result = x.to(compute_dtype).reshape(-1, width)
    if transform_width != width:
        padded = result.new_zeros((result.shape[0], transform_width))
        padded[:, :width] = result
        result = padded
    block = 1
    while block < transform_width:
        rows = result.reshape(
            result.shape[0], transform_width // (2 * block), 2 * block
        )
        left = rows[..., :block]
        right = rows[..., block:]
        result = torch.cat((left + right, left - right), dim=-1).reshape(
            -1, transform_width
        )
        block *= 2
    result = result[:, :width].reshape_as(x)
    return (result * float(scale)).to(x.dtype)


HADAMARD_TRANSFORM = OperationSchema(
    family="transform",
    mode="hadamard_transform",
    reference=hadamard_transform_reference,
    validate_signatures=_validate_signatures,
    validate_traits=_validate_traits,
).publish()
