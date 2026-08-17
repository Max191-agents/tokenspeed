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
from unittest.mock import Mock, patch

import pytest
import tokenspeed_kernel.ops.layernorm.triton as triton_layernorm
import torch

from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm


def _make_fused_norm(q_hidden_size: int = 4, kv_hidden_size: int = 4) -> FusedRMSNorm:
    return FusedRMSNorm(
        RMSNorm(q_hidden_size, eps=1e-6),
        RMSNorm(kv_hidden_size, eps=1e-6),
    )


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fused_rmsnorm_forwards_amd_output_to_single_kernel(
    output_dtype: torch.dtype,
) -> None:
    fused_norm = _make_fused_norm(q_hidden_size=4, kv_hidden_size=6)
    q = torch.ones((2, 4), dtype=torch.bfloat16)
    kv = torch.ones((2, 6), dtype=torch.bfloat16)
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv, dtype=output_dtype)
    kernel = Mock()

    with (
        patch("tokenspeed.runtime.layers.layernorm._is_amd", True),
        patch(
            "tokenspeed.runtime.layers.layernorm.triton_rmsnorm_fused_parallel",
            kernel,
        ),
        patch("tokenspeed.runtime.layers.layernorm.pdl_enabled", return_value=True),
    ):
        result = fused_norm(q, kv, output_q_a=q_out, output_kv_a=kv_out)

    kernel.assert_called_once_with(
        input1=q,
        weight1=fused_norm.weight_q_a,
        output1=q_out,
        input2=kv,
        weight2=fused_norm.weight_kv_a,
        output2=kv_out,
        eps=fused_norm.q_a_norm.variance_epsilon,
        enable_pdl=True,
    )
    assert result[0] is q
    assert result[1] is kv


def test_fused_rmsnorm_amd_defaults_to_inplace_outputs() -> None:
    fused_norm = _make_fused_norm(q_hidden_size=4, kv_hidden_size=6)
    q = torch.ones((2, 4), dtype=torch.bfloat16)
    kv = torch.ones((2, 6), dtype=torch.bfloat16)
    kernel = Mock()

    with (
        patch("tokenspeed.runtime.layers.layernorm._is_amd", True),
        patch(
            "tokenspeed.runtime.layers.layernorm.triton_rmsnorm_fused_parallel",
            kernel,
        ),
        patch("tokenspeed.runtime.layers.layernorm.pdl_enabled", return_value=False),
    ):
        result = fused_norm(q, kv)

    kernel.assert_called_once_with(
        input1=q,
        weight1=fused_norm.weight_q_a,
        output1=q,
        input2=kv,
        weight2=fused_norm.weight_kv_a,
        output2=kv,
        eps=fused_norm.q_a_norm.variance_epsilon,
        enable_pdl=False,
    )
    assert result[0] is q
    assert result[1] is kv


def test_fused_rmsnorm_keeps_non_amd_kernel_call_unchanged() -> None:
    fused_norm = _make_fused_norm()
    q = torch.ones((2, 4), dtype=torch.bfloat16)
    kv = torch.ones((2, 4), dtype=torch.bfloat16)
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)
    kernel = Mock()

    with (
        patch("tokenspeed.runtime.layers.layernorm._is_amd", False),
        patch(
            "tokenspeed.runtime.layers.layernorm.rmsnorm_fused_parallel",
            kernel,
            create=True,
        ),
        patch("tokenspeed.runtime.layers.layernorm.pdl_enabled", return_value=False),
    ):
        result = fused_norm(q, kv, output_q_a=q_out, output_kv_a=kv_out)

    kernel.assert_called_once_with(
        input1=q,
        weight1=fused_norm.weight_q_a,
        output1=q_out,
        input2=kv,
        weight2=fused_norm.weight_kv_a,
        output2=kv_out,
        eps=fused_norm.q_a_norm.variance_epsilon,
        enable_pdl=False,
    )
    assert result[0] is q
    assert result[1] is kv


def test_fused_rmsnorm_keeps_original_forward_api() -> None:
    assert tuple(inspect.signature(FusedRMSNorm.forward).parameters) == (
        "self",
        "input_q_a",
        "input_kv_a",
        "output_q_a",
        "output_kv_a",
    )
    assert not hasattr(triton_layernorm, "rmsnorm_fused_parallel_fp8")
