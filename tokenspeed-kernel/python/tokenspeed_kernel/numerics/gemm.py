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
from typing import Any, Literal

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance
from tokenspeed_numerics_input_generators import (
    CustomDType,
    GemmInputConfig,
    GemmInputs,
    InputDType,
    gemm_scale_shape,
)

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

_ATOL = {
    torch.float32: 1e-5,
    # bf16/fp16: error is dominated by the output cast (~1 ulp_rel = 2^-7 ≈ 8e-3
    # for bf16; 2^-10 ≈ 1e-3 for fp16), not by fp32 accumulation, so we set the
    # baseline at the rounding floor and use a K-independent scale.
    torch.float16: 1.5e-2,
    torch.bfloat16: 1.5e-2,
    torch.float8_e4m3fn: 5e-3,
    torch.float8_e4m3fnuz: 5e-3,
    # Packed quantized GEMM formats such as MXFP4/NVFP4 use uint8 storage for
    # the value tensor. The represented values are determined by the paired
    # format-specific scales.
    torch.uint8: 1.0e-1,
}

_QUANTIZED_DTYPES: set[torch.dtype] = {
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.uint8,
}

_BF16_FP16_DTYPES: set[torch.dtype] = {
    torch.float16,
    torch.bfloat16,
}


def tolerance(
    dtype: torch.dtype,
    *,
    K: int | None = None,
    inputs: dict[str, Any] | None = None,
    acc_dtype: torch.dtype = torch.float32,
    **_: Any,
) -> Tolerance:
    """Shape-aware GEMM tolerance.

    - fp32: error grows as sqrt(K) under fp32 accumulation noise.
    - fp16/bf16: K-independent — fp32 accumulation is well below the output
      dtype's rounding floor, so error is dominated by the final cast.
    - fp8: error grows linearly with K for blockwise kernels.
    """
    if dtype not in _ATOL:
        raise KeyError(f"No GEMM tolerance baseline for dtype={dtype}")

    if K is None and inputs is not None and "A" in inputs:
        K = int(inputs["A"].shape[-1])
    if K is None:
        raise ValueError("GEMM tolerance requires K or inputs['A']")

    base = _ATOL[dtype]
    if dtype in _QUANTIZED_DTYPES:
        scale = max(K, 1) / 128.0
    elif dtype in _BF16_FP16_DTYPES:
        scale = 1.0
    else:
        scale = math.sqrt(max(K, 1) / 128.0)
    if acc_dtype != torch.float32:
        scale *= 8.0
    return Tolerance(atol=base * scale, rtol=base * scale)


set_family_tolerance("gemm", tolerance)

# ---------------------------------------------------------------------------
# Input Generator
# ---------------------------------------------------------------------------


class GemmInputGenerator(InputGenerator):
    def _format(self, role: str) -> Any | None:
        if self.format_signature is None:
            return None
        return self.format_signature.format_for(role)

    def _layout(self, role: str) -> str:
        layout = self.traits.get(f"{role}_layout")
        if role == "a" and layout == {"KM"}:
            return "KM"
        if role == "b" and layout == {"KN"}:
            return "KN"
        return "MK" if role == "a" else "NK"

    def _block_size(self, *formats: Any | None) -> list[int] | None:
        for tensor_format in formats:
            scale = tensor_format.scale if tensor_format is not None else None
            if scale is not None and scale.block_shape is not None:
                return list(scale.block_shape)
        return None

    def _input_dtype(self, tensor_format: Any | None) -> InputDType:
        if tensor_format is None:
            return self.dtype
        if tensor_format.format == "mxfp4":
            if tensor_format.storage_dtype != torch.uint8:
                raise ValueError("mxfp4 values must use torch.uint8 storage")
            return CustomDType.MXFP4
        return tensor_format.storage_dtype

    def _scale_dtype(self, tensor_format: Any | None) -> InputDType:
        scale = tensor_format.scale if tensor_format is not None else None
        if scale is None:
            return None
        if tensor_format.format == "mxfp4":
            if scale.storage_dtype != torch.uint8:
                raise ValueError("mxfp4 scales must use torch.uint8 UE8M0 storage")
            return CustomDType.UE8M0
        return scale.storage_dtype

    def _scale_shape(
        self,
        tensor_format: Any | None,
        role: Literal["a", "b"],
        *,
        M: int,
        N: int,
        K: int,
    ) -> tuple[int, ...] | None:
        scale = tensor_format.scale if tensor_format is not None else None
        if scale is None:
            return None
        return gemm_scale_shape(
            scale.granularity,
            role,
            M=M,
            N=N,
            K=K,
            block_shape=scale.block_shape,
        )

    def generate(
        self,
        M: int,
        N: int,
        K: int,
    ) -> dict[str, Any]:
        a_tensor_format = self._format("a")
        b_tensor_format = self._format("b")
        a_dtype = self._input_dtype(a_tensor_format)
        b_dtype = self._input_dtype(b_tensor_format)

        block_size = self._block_size(a_tensor_format, b_tensor_format)
        a_layout = self._layout("a")
        b_layout = self._layout("b")
        out_dtype = torch.bfloat16

        gemm_inputs = GemmInputs(
            GemmInputConfig(
                M=M,
                N=N,
                K=K,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                c_dtype=out_dtype,
                a_layout=a_layout,
                b_layout=b_layout,
                a_scale_dtype=self._scale_dtype(a_tensor_format),
                b_scale_dtype=self._scale_dtype(b_tensor_format),
                a_scale_shape=self._scale_shape(a_tensor_format, "a", M=M, N=N, K=K),
                b_scale_shape=self._scale_shape(b_tensor_format, "b", M=M, N=N, K=K),
            )
        ).generate(seed=self.seed, device=self.device)
        A = gemm_inputs.A
        B = gemm_inputs.B
        A_scales = gemm_inputs.A_scales
        B_scales = gemm_inputs.B_scales
        C = gemm_inputs.C

        alpha = None

        return {
            "A": A,
            "B": B,
            "C": C,
            "A_scales": A_scales,
            "B_scales": B_scales,
            "out_dtype": out_dtype,
            "alpha": alpha,
            "block_size": block_size,
        }


set_input_generator("gemm", "mm", GemmInputGenerator)

# ---------------------------------------------------------------------------
# Shape Presets
# ---------------------------------------------------------------------------


GEMM_MM_STANDARD_SHAPES: list[dict[str, int]] = [
    {"M": 16, "N": 16, "K": 64},
    {"M": 64, "N": 128, "K": 128},
    {"M": 128, "N": 128, "K": 256},
    {"M": 256, "N": 256, "K": 512},
    # DSv3 hot-path shapes — exercise hand-rolled kernels in trtllm dsv3_router /
    # dsv3_fused_a (M ≤ 16, K = 7168, N = num_experts=256 or fused_a=2112) plus
    # off-shape (M = 64) which falls back to cuBLAS inside the same op.
    {"M": 1, "N": 256, "K": 7168},
    {"M": 8, "N": 256, "K": 7168},
    {"M": 16, "N": 256, "K": 7168},
    {"M": 64, "N": 256, "K": 7168},
    {"M": 1, "N": 2112, "K": 7168},
    {"M": 8, "N": 2112, "K": 7168},
    {"M": 16, "N": 2112, "K": 7168},
    {"M": 64, "N": 2112, "K": 7168},
]


GEMM_MM_BENCHMARK_SHAPES: list[dict[str, int]] = [
    {"M": 1, "N": 4096, "K": 4096},
    {"M": 16, "N": 4096, "K": 4096},
    {"M": 128, "N": 4096, "K": 4096},
    {"M": 512, "N": 4096, "K": 4096},
    {"M": 4096, "N": 4096, "K": 4096},
]


set_standard_shapes("gemm", "mm", GEMM_MM_STANDARD_SHAPES)
set_benchmark_shapes("gemm", "mm", GEMM_MM_BENCHMARK_SHAPES)
