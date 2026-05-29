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

import importlib
import sys

import pytest
import torch
from tokenspeed_kernel.numerics.inputs import get_input_generator
from tokenspeed_kernel.numerics.reference.layernorm import rmsnorm as torch_rmsnorm
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
from tokenspeed_kernel.signature import format_signatures

pytestmark = pytest.mark.usefixtures("fresh_registry")


def _reload_layernorm_registrations() -> None:
    for name in (
        "tokenspeed_kernel.ops.layernorm",
        "tokenspeed_kernel.ops.layernorm.gluon",
        "tokenspeed_kernel.ops.layernorm.triton",
        "tokenspeed_kernel.numerics.reference.layernorm",
    ):
        sys.modules.pop(name, None)
    importlib.import_module("tokenspeed_kernel.ops.layernorm")


def test_layernorm_side_effect_import_registers_rmsnorm_kernels() -> None:
    _reload_layernorm_registrations()

    specs = KernelRegistry.get().get_for_operator("norm", "rmsnorm")
    names = {spec.name for spec in specs}

    assert {"torch_rmsnorm", "triton_rmsnorm", "gluon_rmsnorm"}.issubset(names)
    gluon_spec = KernelRegistry.get().get_by_name("gluon_rmsnorm")
    assert gluon_spec is not None
    assert gluon_spec.solution == "gluon"
    assert gluon_spec.capability.vendors == frozenset({"amd"})
    assert gluon_spec.format_signatures_for_storage_dtype(torch.bfloat16, "x")


def test_load_builtin_kernels_includes_layernorm() -> None:
    load_builtin_kernels()

    assert KernelRegistry.get().get_by_name("gluon_rmsnorm") is not None
    assert KernelRegistry.get().get_by_name("triton_rmsnorm") is not None
    assert KernelRegistry.get().get_by_name("torch_rmsnorm") is not None


def test_rmsnorm_input_generator_matches_registered_kernel_api() -> None:
    import tokenspeed_kernel.numerics.layernorm  # noqa: F401

    signature = next(iter(format_signatures("x", "dense", {torch.bfloat16})))
    generator = get_input_generator(
        "norm",
        "rmsnorm",
        dtype=torch.bfloat16,
        traits={},
        format_signature=signature,
        device="cpu",
    )

    inputs = generator.generate(num_tokens=3, hidden_size=16)

    assert set(inputs) == {"x", "weight", "eps", "residual", "out"}
    assert inputs["x"].shape == (3, 16)
    assert inputs["x"].dtype == torch.bfloat16
    assert inputs["weight"].shape == (16,)
    assert inputs["weight"].dtype == torch.float32
    assert inputs["residual"] is None
    assert inputs["out"] is None


def test_reference_rmsnorm_matches_formula() -> None:
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.float32)
    eps = 1e-6

    out = torch_rmsnorm(x, weight, eps)

    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    expected = (x_float * torch.rsqrt(variance + eps) * weight).to(x.dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
