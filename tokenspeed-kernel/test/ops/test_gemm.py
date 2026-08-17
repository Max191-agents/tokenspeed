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
from unittest.mock import Mock

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm import (
    bmm_fp8_output,
    linear_attnres_partials,
    linear_attnres_partials_available,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_DENSE_BMM_SIGNATURE = format_signature(
    a=dense_tensor_format(torch.bfloat16),
    b=dense_tensor_format(torch.bfloat16),
)
_BMM_FP8_OUTPUT_SIGNATURE = format_signature(
    a=dense_tensor_format(torch.bfloat16),
    b=dense_tensor_format(torch.bfloat16),
    out=dense_tensor_format(torch.float8_e4m3fn),
)
_BMM_FP8_OUTPUT_TRAITS = {
    "batch": 16,
    "m": 3,
    "n": 512,
    "k": 192,
    "a_inner_stride_one": True,
    "b_n_stride_one": True,
    "out_inner_stride_one": True,
    "a_load_16b_aligned": True,
    "b_load_16b_aligned": True,
}


def test_mm_rejects_bad_out_layout() -> None:
    a = torch.empty((4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)
    out = torch.empty((16, 4), dtype=torch.bfloat16).transpose(0, 1)

    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.mm(a, b, out=out)


def test_mm_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.empty((4, 8), dtype=torch.float32)
    b = torch.empty((16, 8), dtype=torch.float32)
    out = torch.empty((4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_mm out= requires out_dtype"):
        tokenspeed_kernel.mm(a, b, out=out, override="torch_mm")


def test_bmm_rejects_batch_mismatch() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((3, 16, 8), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="batch mismatch"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_rejects_rank2_weights() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((16, 8), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"B with shape \[B, N, K\]"):
        tokenspeed_kernel.bmm(a, b)


def test_bmm_rejects_bad_out_layout() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 16, 8), dtype=torch.bfloat16)
    out = torch.empty((2, 16, 4), dtype=torch.bfloat16).transpose(1, 2)

    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        tokenspeed_kernel.bmm(a, b, out=out)


def test_bmm_reference_rejects_out_dtype_mismatch() -> None:
    a = torch.randn((2, 4, 8), dtype=torch.float32)
    b = torch.randn((2, 16, 8), dtype=torch.float32)
    out = torch.empty((2, 4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="torch_bmm out= requires out_dtype"):
        tokenspeed_kernel.bmm(a, b, out=out, override="torch_bmm")


@pytest.mark.parametrize("operand", ["a", "b"])
def test_bmm_fp8_output_rejects_non_bf16_input(operand: str) -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 16, 8), dtype=torch.bfloat16)
    if operand == "a":
        a = a.float()
    else:
        b = b.float()

    with pytest.raises(TypeError, match="expects BF16 inputs"):
        bmm_fp8_output(a, b)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        pytest.param((0, 4, 8), "positive batch/M/N/K", id="batch"),
        pytest.param((2, 0, 8), "positive batch/M/N/K", id="rows"),
        pytest.param((2, 4, 0), "positive batch/M/N/K", id="reduction"),
    ],
)
def test_bmm_fp8_output_rejects_empty_dimensions(
    shape: tuple[int, int, int], message: str
) -> None:
    batch, _, K = shape
    a = torch.empty(shape, dtype=torch.bfloat16)
    b = torch.empty((batch, 16, K), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=message):
        bmm_fp8_output(a, b)


def test_bmm_fp8_output_rejects_empty_output_dimension() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 0, 8), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="positive batch/M/N/K"):
        bmm_fp8_output(a, b)


def test_bmm_fp8_output_rejects_wrong_output_dtype() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 16, 8), dtype=torch.bfloat16)
    out = torch.empty((2, 4, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="expects dtype"):
        bmm_fp8_output(a, b, out=out)


def test_bmm_fp8_output_rejects_overlapping_output() -> None:
    a = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    b = torch.empty((2, 16, 8), dtype=torch.bfloat16)
    out = torch.empty((4, 16), dtype=torch.float8_e4m3fn).as_strided(
        (2, 4, 16),
        (0, 16, 1),
    )

    with pytest.raises(ValueError, match="must not have internal overlap"):
        bmm_fp8_output(a, b, out=out)


def test_bmm_fp8_output_has_no_nvidia_kernel(h100_platform) -> None:
    registry = KernelRegistry.get()
    registry.clear_cache()
    try:
        with pytest.raises(NoKernelFoundError):
            select_kernel(
                "gemm",
                "bmm_fp8_output",
                _BMM_FP8_OUTPUT_SIGNATURE,
                platform=h100_platform,
                traits=_BMM_FP8_OUTPUT_TRAITS,
            )
    finally:
        registry.clear_cache()


@pytest.mark.parametrize("tokens", range(1, 33))
def test_bmm_fp8_output_selects_specialized_kernel(mi350_platform, tokens: int) -> None:
    registry = KernelRegistry.get()
    traits = {**_BMM_FP8_OUTPUT_TRAITS, "m": tokens}

    registry.clear_cache()
    try:
        selected = select_kernel(
            "gemm",
            "bmm_fp8_output",
            _BMM_FP8_OUTPUT_SIGNATURE,
            platform=mi350_platform,
            traits=traits,
        )
    finally:
        registry.clear_cache()

    if registry.get_by_name("triton_bmm_bf16_fp8_glm52") is not None and tokens in (
        1,
        2,
        4,
        8,
        16,
    ):
        expected = "triton_bmm_bf16_fp8_glm52"
    elif registry.get_by_name("gluon_bmm_a16w16_skinny_gfx950") is not None:
        expected = "gluon_bmm_a16w16_skinny_gfx950"
    else:
        expected = "composite_bmm_fp8_output"
    assert selected.name == expected
    selected_spec = registry.get_by_name(selected.name)
    assert selected_spec is not None
    native_output = spec_matches_traits(
        selected_spec,
        {"native_fp8_output": True},
        require_all_traits=True,
    )
    assert native_output is (expected != "composite_bmm_fp8_output")


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 5, 64, 64), id="minimum-tile"),
        pytest.param((7, 23, 128, 320), id="partial-m-tile"),
        pytest.param((16, 32, 1024, 512), id="implementation-upper-bounds"),
    ],
)
def test_bmm_fp8_output_uses_composite_for_general_shapes(
    mi350_platform,
    shape: tuple[int, int, int, int],
) -> None:
    registry = KernelRegistry.get()
    batch, M, N, K = shape
    traits = {
        **_BMM_FP8_OUTPUT_TRAITS,
        "batch": batch,
        "m": M,
        "n": N,
        "k": K,
    }

    registry.clear_cache()
    try:
        selected = select_kernel(
            "gemm",
            "bmm_fp8_output",
            _BMM_FP8_OUTPUT_SIGNATURE,
            platform=mi350_platform,
            traits=traits,
        )
    finally:
        registry.clear_cache()

    assert selected.name == "composite_bmm_fp8_output"


def test_generic_bmm_keeps_tuned_m1_k128_priority(mi350_platform) -> None:
    registry = KernelRegistry.get()
    traits = {
        "batch": 12,
        "m": 1,
        "n": 512,
        "k": 128,
        "a_inner_stride_one": True,
        "b_n_stride_one": True,
        "out_inner_stride_one": True,
        "out_dtype": torch.bfloat16,
    }

    registry.clear_cache()
    try:
        selected = select_kernel(
            "gemm",
            "bmm",
            _DENSE_BMM_SIGNATURE,
            platform=mi350_platform,
            traits=traits,
        )
    finally:
        registry.clear_cache()

    if registry.get_by_name("gluon_bmm_a16w16_gfx950") is not None:
        expected = "gluon_bmm_a16w16_gfx950"
    else:
        expected = "torch_bmm"
    assert selected.name == expected


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param({"batch": 15}, id="batch"),
        pytest.param({"m": 33}, id="tokens"),
        pytest.param({"n": 1024}, id="output-width"),
        pytest.param({"k": 512}, id="reduction-width"),
        pytest.param({"a_inner_stride_one": False}, id="activation-stride"),
        pytest.param({"b_n_stride_one": False}, id="weight-stride"),
        pytest.param({"out_inner_stride_one": False}, id="output-stride"),
        pytest.param({"a_load_16b_aligned": False}, id="activation-alignment"),
        pytest.param({"b_load_16b_aligned": False}, id="weight-alignment"),
    ],
)
def test_bmm_fp8_output_negative_traits_use_composite(
    mi350_platform,
    updates: dict[str, object],
) -> None:
    registry = KernelRegistry.get()
    traits = {**_BMM_FP8_OUTPUT_TRAITS, **updates}

    registry.clear_cache()
    try:
        selected = select_kernel(
            "gemm",
            "bmm_fp8_output",
            _BMM_FP8_OUTPUT_SIGNATURE,
            platform=mi350_platform,
            traits=traits,
        )
    finally:
        registry.clear_cache()

    assert selected.name == "composite_bmm_fp8_output"


def test_bmm_fp8_output_solution_can_force_composite(mi350_platform) -> None:
    selected = select_kernel(
        "gemm",
        "bmm_fp8_output",
        _BMM_FP8_OUTPUT_SIGNATURE,
        platform=mi350_platform,
        traits={**_BMM_FP8_OUTPUT_TRAITS, "m": 1},
        solution="composite",
    )

    assert selected.name == "composite_bmm_fp8_output"


def test_skinny_bmm_registered_wrapper_delegates_when_native_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = KernelRegistry.get().get_impl("gluon_bmm_a16w16_skinny_gfx950")
    if implementation is None:
        pytest.skip("gfx950 Gluon BMM is not registered")
    gluon = importlib.import_module("tokenspeed_kernel.ops.gemm.gluon")
    triton_module = importlib.import_module("tokenspeed_kernel.ops.gemm.triton")
    native = Mock(return_value=None)
    fallback = Mock()
    monkeypatch.setattr(gluon, "_bmm_a16w16_skinny_impl", native)
    monkeypatch.setattr(triton_module, "composite_bmm_fp8_output", fallback)
    a = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    b = torch.empty((2, 5, 4), dtype=torch.bfloat16)
    out = torch.empty((2, 3, 5), dtype=torch.float8_e4m3fn)
    fallback.return_value = out

    returned = implementation(a, b, out=out, enable_pdl=True)

    assert returned is out
    native.assert_called_once_with(a, b, torch.float8_e4m3fn, out=out)
    fallback.assert_called_once_with(a, b, out=out, enable_pdl=True)


def test_bmm_fp8_output_composite_writes_strided_prefix_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemm_module = importlib.import_module("tokenspeed_kernel.ops.gemm")
    triton_module = importlib.import_module("tokenspeed_kernel.ops.gemm.triton")
    bmm_calls = 0
    quantize_calls = 0

    def fake_bmm(A, B, *, out, out_dtype):
        nonlocal bmm_calls
        bmm_calls += 1
        assert out.is_contiguous()
        assert out_dtype == torch.bfloat16
        return torch.bmm(A, B.transpose(1, 2), out=out)

    def fake_quantize(x, out):
        nonlocal quantize_calls
        quantize_calls += 1
        assert x.dtype == torch.bfloat16
        limit = torch.finfo(out.dtype).max
        out.copy_(x.clamp(min=-limit, max=limit).to(out.dtype))
        return out

    monkeypatch.setattr(gemm_module, "bmm", fake_bmm)
    monkeypatch.setattr(triton_module, "_quantize_bmm_fp8_output", fake_quantize)
    torch.manual_seed(17)
    heads, tokens, k, n = 2, 3, 4, 5
    q_backing = torch.randn(tokens, heads, k + 2, dtype=torch.bfloat16)
    q = q_backing[..., :k].transpose(0, 1)
    weight = torch.randn(heads, k, n, dtype=torch.bfloat16)
    backing = torch.empty(tokens, heads, n + 3, dtype=torch.float8_e4m3fn)
    backing.view(torch.uint8).fill_(0xA5)
    out = backing[..., :n].transpose(0, 1)
    expected = torch.bmm(q, weight).to(torch.float8_e4m3fn)

    returned = bmm_fp8_output(
        q,
        weight.transpose(1, 2),
        out=out,
        enable_pdl=True,
        solution="composite",
    )

    assert returned is out
    assert bmm_calls == 1
    assert quantize_calls == 1
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    assert torch.all(backing[..., n:].view(torch.uint8) == 0xA5)


def test_bmm_fp8_output_auto_dispatch_uses_composite_for_misaligned_input(
    device: str,
    require,
) -> None:
    require("gemm", "bmm_fp8_output", "gluon", torch.bfloat16, "a")
    require("gemm", "bmm_fp8_output", "composite", torch.bfloat16, "a")
    torch.manual_seed(19)
    heads, tokens, k, n = 16, 3, 192, 512
    storage = torch.randn(
        tokens * heads * 256 + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    q = storage[1:].as_strided(
        (heads, tokens, k),
        (256, heads * 256, 1),
    )
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16) * 0.25
    out = torch.empty(heads, tokens, n, device=device, dtype=torch.float8_e4m3fn)

    returned = bmm_fp8_output(
        q,
        weight.transpose(1, 2),
        out=out,
    )

    expected = torch.bmm(q, weight).to(out.dtype)
    assert returned is out
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


def test_bmm_fp8_output_auto_dispatch_uses_gluon_for_production_layout(
    device: str,
    require,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require("gemm", "bmm_fp8_output", "gluon", torch.bfloat16, "a")
    triton_module = importlib.import_module("tokenspeed_kernel.ops.gemm.triton")
    fallback = Mock(side_effect=AssertionError("native Gluon kernel declined"))
    monkeypatch.setattr(triton_module, "composite_bmm_fp8_output", fallback)
    torch.manual_seed(21)
    heads, tokens, k, n = 16, 3, 192, 512
    q_backing = torch.randn(
        tokens,
        heads,
        256,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_backing[..., :k].transpose(0, 1)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    combined = torch.empty(
        tokens,
        heads,
        n + 64,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    combined.view(torch.uint8).fill_(0xA5)
    out = combined[..., :n].transpose(0, 1)

    returned = bmm_fp8_output(
        q,
        weight.transpose(1, 2),
        out=out,
    )

    expected = torch.bmm(q, weight).to(torch.float8_e4m3fn)
    assert returned is out
    fallback.assert_not_called()
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0.5)
    assert torch.all(combined[..., n:].view(torch.uint8) == 0xA5)


def test_bmm_fp8_output_composite_writes_head_major_prefix(
    device: str,
    require,
) -> None:
    require("gemm", "bmm_fp8_output", "composite", torch.bfloat16, "a")
    torch.manual_seed(23)
    heads, tokens, k, n = 3, 2, 8, 16
    q_backing = torch.randn(tokens, heads, k + 4, device=device, dtype=torch.bfloat16)
    q = q_backing[..., :k].transpose(0, 1)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    backing = torch.empty(
        tokens,
        heads,
        n + 7,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    backing.view(torch.uint8).fill_(0xA5)
    out = backing[..., :n].transpose(0, 1)

    returned = bmm_fp8_output(
        q,
        weight.transpose(1, 2),
        out=out,
        solution="composite",
    )

    expected = torch.bmm(q, weight).to(torch.float8_e4m3fn)
    assert returned is out
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    assert torch.all(backing[..., n:].view(torch.uint8) == 0xA5)


def test_bmm_fp8_output_composite_saturates(device: str, require) -> None:
    require("gemm", "bmm_fp8_output", "composite", torch.bfloat16, "a")
    a = torch.full((1, 1, 4), 16.0, device=device, dtype=torch.bfloat16)
    b = torch.full((1, 3, 4), 16.0, device=device, dtype=torch.bfloat16)

    output = bmm_fp8_output(a, b, solution="composite")

    assert output.float().unique().item() == torch.finfo(output.dtype).max


def test_bmm_fp8_output_composite_normalizes_pdl_on_amd(
    device: str,
    require,
) -> None:
    if not current_platform().is_amd:
        pytest.skip("AMD-specific PDL normalization")
    require("gemm", "bmm_fp8_output", "composite", torch.bfloat16, "a")
    torch.manual_seed(29)
    a = torch.randn((2, 3, 8), device=device, dtype=torch.bfloat16)
    b = torch.randn((2, 5, 8), device=device, dtype=torch.bfloat16)

    output = bmm_fp8_output(
        a,
        b,
        solution="composite",
        enable_pdl=True,
    )

    expected = torch.bmm(a, b.transpose(1, 2)).to(torch.float8_e4m3fn)
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


def test_bmm_writes_head_major_strided_output(device: str) -> None:
    heads, tokens, k, n = 3, 1, 8, 16
    a = torch.randn(heads, tokens, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    backing = torch.empty(tokens, heads, n + 4, device=device, dtype=torch.bfloat16)
    out = backing[..., :n].transpose(0, 1)

    returned = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        out=out,
        override="torch_bmm",
    )

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=0, rtol=0)


def test_gluon_bmm_writes_head_major_strided_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    heads, tokens, k, n = 12, 1, 128, 512
    a_backing = torch.randn(tokens, heads, k, device=device, dtype=torch.bfloat16)
    a = a_backing.transpose(0, 1)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    backing = torch.empty(tokens, heads, n + 64, device=device, dtype=torch.bfloat16)
    out = backing[..., :n].transpose(0, 1)

    returned = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        out=out,
        override="gluon_bmm_a16w16_gfx950",
    )

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=0.25, rtol=0.01)


def test_gluon_bmm_allocates_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    a = torch.randn(12, 1, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(12, 128, 512, device=device, dtype=torch.bfloat16)

    output = tokenspeed_kernel.bmm(
        a,
        weight.transpose(1, 2),
        override="gluon_bmm_a16w16_gfx950",
    )

    torch.testing.assert_close(output, torch.bmm(a, weight), atol=0.25, rtol=0.01)


def test_gluon_bmm_falls_back_for_fp32_output(device: str, require) -> None:
    require("gemm", "bmm", "gluon", torch.bfloat16, "a")
    a = torch.randn(12, 1, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(12, 128, 512, device=device, dtype=torch.bfloat16)

    output = tokenspeed_kernel.bmm(a, weight.transpose(1, 2), out_dtype=torch.float32)

    assert output.dtype == torch.float32


@pytest.mark.parametrize("tokens", [1, 16])
def test_bmm_fp8_output_writes_fp8_prefix(device: str, require, tokens: int) -> None:
    require("gemm", "bmm_fp8_output", "triton", torch.bfloat16, "a")
    torch.manual_seed(91)
    heads, k, n = 16, 192, 512
    q_backing = torch.randn(tokens, heads, 256, device=device, dtype=torch.bfloat16)
    q = q_backing[..., :k].transpose(0, 1)
    weight = torch.randn(heads, k, n, device=device, dtype=torch.bfloat16)
    combined = torch.empty(
        tokens, heads, n + 64, device=device, dtype=torch.float8_e4m3fn
    )
    combined.view(torch.uint8).fill_(0xA5)
    out = combined[..., :n].transpose(0, 1)

    actual = bmm_fp8_output(
        q,
        weight.transpose(1, 2),
        out=out,
        override="triton_bmm_bf16_fp8_glm52",
    )

    expected = torch.bmm(q, weight).to(torch.float8_e4m3fn)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.5)
    assert torch.all(combined[..., n:].view(torch.uint8) == 0xA5)


def test_bmm_fp8_output_rounds_through_bf16_before_fp8(device: str, require) -> None:
    require("gemm", "bmm_fp8_output", "triton", torch.bfloat16, "a")
    heads, tokens, k, n = 16, 1, 192, 512
    q_backing = torch.zeros(tokens, heads, 256, device=device, dtype=torch.bfloat16)
    q_backing[..., :2] = 1.0
    q = q_backing[..., :k].transpose(0, 1)
    weight = torch.zeros(heads, k, n, device=device, dtype=torch.bfloat16)
    weight[:, 0, :] = 2.125
    weight[:, 1, :] = 2**-10
    out = torch.empty(heads, tokens, n, device=device, dtype=torch.float8_e4m3fn)

    actual = bmm_fp8_output(
        q,
        weight.transpose(1, 2),
        out=out,
        override="triton_bmm_bf16_fp8_glm52",
    )

    staged = torch.bmm(q, weight).to(torch.float8_e4m3fn)
    direct = torch.tensor(2.125 + 2**-10, device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    assert staged.float().unique().item() == 2.0
    assert direct.float().item() == 2.25
    assert torch.equal(actual.view(torch.uint8), staged.view(torch.uint8))


def test_bmm_fp8_output_saturates_fp8_output(device: str, require) -> None:
    require("gemm", "bmm_fp8_output", "triton", torch.bfloat16, "a")
    heads, tokens, k, n = 16, 1, 192, 512
    q = torch.full((heads, tokens, k), 4.0, device=device, dtype=torch.bfloat16)
    weight = torch.full((heads, k, n), 4.0, device=device, dtype=torch.bfloat16)

    output = bmm_fp8_output(
        q,
        weight.transpose(1, 2),
        override="triton_bmm_bf16_fp8_glm52",
    )

    assert output.float().unique().item() == torch.finfo(output.dtype).max


def test_decode_gemv_writes_preallocated_output() -> None:
    from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

    x = torch.randn(2, 8)
    weight = torch.randn(4, 8)
    out = torch.empty(2, 4)

    returned = decode_gemv(x, weight, out=out)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, x @ weight.t())


def test_linear_attnres_partials_portable_composition() -> None:
    torch.manual_seed(13)
    hidden = torch.randn(2, 6, dtype=torch.bfloat16)
    weight = torch.randn(9, 6, dtype=torch.bfloat16)
    blocks = torch.randn(4, 2, 6, dtype=torch.bfloat16)
    scores = (
        torch.randn(6, dtype=torch.bfloat16),
        torch.randn(6, dtype=torch.bfloat16),
    )
    scratch = tuple(
        (
            torch.empty(2, dtype=torch.float32),
            torch.empty(2, dtype=torch.float32),
            torch.empty(2, 6, dtype=torch.float32),
        )
        for _ in range(2)
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    torch.testing.assert_close(actual, torch.nn.functional.linear(hidden, weight))
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-5)
    for score, outputs in zip(scores, scratch, strict=True):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        maxima = logits.max(dim=0).values
        unnormalized = torch.exp(logits - maxima)
        torch.testing.assert_close(outputs[0], maxima)
        torch.testing.assert_close(outputs[1], unnormalized.sum(dim=0))
        torch.testing.assert_close(
            outputs[2],
            torch.einsum("bt,bth->th", unnormalized, values),
        )


def test_linear_attnres_partials_decode_fallback_uses_gemv(monkeypatch) -> None:
    from tokenspeed_kernel.ops.gemm import triton_gemv

    hidden = torch.randn(1, 6, dtype=torch.bfloat16)
    weight = torch.randn(9, 6, dtype=torch.bfloat16)
    blocks = torch.randn(2, 1, 6, dtype=torch.bfloat16)
    scores = tuple(torch.randn(6, dtype=torch.bfloat16) for _ in range(2))
    scratch = tuple(
        (
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, 6, dtype=torch.float32),
        )
        for _ in range(2)
    )
    expected = torch.empty(1, 9, dtype=torch.bfloat16)
    gemv = Mock(return_value=expected)
    monkeypatch.setattr(triton_gemv, "decode_gemv", gemv)

    assert not linear_attnres_partials_available(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    assert actual is expected
    gemv.assert_called_once_with(hidden, weight)


def test_linear_attnres_partials_cpu_skips_device_kernel_selection(
    monkeypatch,
) -> None:
    module = importlib.import_module(
        "tokenspeed_kernel.ops.gemm.linear_attnres_partials"
    )
    selector = Mock()
    monkeypatch.setattr(module, "select_kernel", selector)
    hidden = torch.randn(1, 6, dtype=torch.bfloat16)
    weight = torch.randn(9, 6, dtype=torch.bfloat16)
    blocks = torch.randn(2, 1, 6, dtype=torch.bfloat16)
    scores = tuple(torch.randn(6, dtype=torch.bfloat16) for _ in range(2))
    scratch = tuple(
        (
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, 6, dtype=torch.float32),
        )
        for _ in range(2)
    )

    assert not linear_attnres_partials_available(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )
    selector.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
    reason="gfx950 is required",
)
def test_linear_attnres_partials_gfx950_matches_composition() -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    hidden = (torch.randn(1, 7168, device="cuda", generator=generator) * 0.1).to(
        torch.bfloat16
    )
    weight = (torch.randn(3648, 7168, device="cuda", generator=generator) * 0.01).to(
        torch.bfloat16
    )
    blocks = (torch.randn(4, 1, 7168, device="cuda", generator=generator) * 0.1).to(
        torch.bfloat16
    )
    scores = tuple(
        (torch.randn(7168, device="cuda", generator=generator) * 0.02).to(
            torch.bfloat16
        )
        for _ in range(2)
    )
    scratch = tuple(
        (
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(1, 7168, device="cuda", dtype=torch.float32),
        )
        for _ in range(2)
    )

    assert linear_attnres_partials_available(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-6,
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-6,
        override="gluon_linear_attnres_partials_gfx950",
    )

    torch.testing.assert_close(
        actual,
        torch.nn.functional.linear(hidden, weight),
        atol=2e-2,
        rtol=2e-2,
    )
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-6)
    for score, outputs in zip(scores, scratch, strict=True):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        maxima = logits.max(dim=0).values
        unnormalized = torch.exp(logits - maxima)
        torch.testing.assert_close(outputs[0], maxima, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(
            outputs[1], unnormalized.sum(dim=0), atol=2e-4, rtol=2e-4
        )
        torch.testing.assert_close(
            outputs[2],
            torch.einsum("bt,bth->th", unnormalized, values),
            atol=2e-4,
            rtol=2e-4,
        )

    with pytest.raises(ValueError, match="output size must be divisible by 16"):
        linear_attnres_partials(
            hidden,
            weight[:17],
            blocks,
            *scores,
            *scratch,
            eps=1e-6,
            override="gluon_linear_attnres_partials_gfx950",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
def test_linear_attnres_partials_cuda_portable_strided_inputs() -> None:
    generator = torch.Generator(device="cuda").manual_seed(37)
    hidden = torch.randn(
        1, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        32, 65, device="cuda", dtype=torch.bfloat16, generator=generator
    )[:, :64]
    blocks = torch.randn(
        3, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )[..., ::2]
    scores = tuple(
        torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=generator)[::2]
        for _ in range(2)
    )
    scratch = tuple(
        (
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(1, device="cuda", dtype=torch.float32),
            torch.empty(1, 64, device="cuda", dtype=torch.float32),
        )
        for _ in range(2)
    )

    actual = linear_attnres_partials(
        hidden,
        weight,
        blocks,
        *scores,
        *scratch,
        eps=1e-5,
    )

    torch.testing.assert_close(actual, torch.nn.functional.linear(hidden, weight))
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-5)
    for score, outputs in zip(scores, scratch, strict=True):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        maxima = logits.max(dim=0).values
        unnormalized = torch.exp(logits - maxima)
        torch.testing.assert_close(outputs[0], maxima)
        torch.testing.assert_close(outputs[1], unnormalized.sum(dim=0))
        torch.testing.assert_close(
            outputs[2],
            torch.einsum("bt,bth->th", unnormalized, values),
        )
