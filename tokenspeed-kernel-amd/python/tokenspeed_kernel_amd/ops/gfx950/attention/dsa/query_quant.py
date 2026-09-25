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

"""GFX950 query preparation for the pooled-key scorer.

The scorer consumes one FP8 scale per 128-value query head. This preparation
kernel follows the SGLang/AITER contract: it rounds that scale up to a power of
two, emits E4M3FN queries, and folds both the query scale and model scale into
the FP32 per-head weights. Keeping preparation outside the pool-window loop
lets every scorer window reuse the same converted query.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = ["gluon_dsa_prepare_kpool_query_gfx950"]

_BLOCK_VECTORS = 32
_HEAD_DIM = 128
_NUM_WARPS = 2
_HEAD_SCALE = 32**-0.5
_SOFTMAX_SCALE = _HEAD_DIM**-0.5
_MODEL_SCALE = 1.0 / 64.0


@gluon.constexpr_function
def _query_downcast_layout(
    block_vectors: gl.constexpr,
    head_dim: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Lay out four 32-value hardware scale groups per query vector."""
    return gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


@gluon.jit
def _dsa_prepare_kpool_query_kernel(
    q,
    weights,
    q_out,
    combined_weights_out,
    HEAD_SCALE: gl.constexpr,
    SOFTMAX_SCALE: gl.constexpr,
    BLOCK_VECTORS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
):
    gl.static_assert(BLOCK_VECTORS == 32)
    gl.static_assert(HEAD_DIM == 128)
    gl.static_assert(gl.num_warps() == 2)
    layout: gl.constexpr = _query_downcast_layout(
        BLOCK_VECTORS,
        HEAD_DIM,
        gl.num_warps(),
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    dim_layout: gl.constexpr = gl.SliceLayout(0, layout)

    rows = gl.program_id(0) * BLOCK_VECTORS + gl.arange(
        0,
        BLOCK_VECTORS,
        layout=row_layout,
    )
    dims = gl.arange(0, HEAD_DIM, layout=dim_layout)
    values = gl.load(
        q + rows[:, None].to(gl.int64) * HEAD_DIM + dims[None, :],
    ).to(gl.float32)

    # The public contract has one common scale for all 128 values. The CDNA4
    # conversion instruction accepts 32-value E8M0 groups, so repeat the same
    # exponent across its four physical groups.
    grouped = values.reshape((BLOCK_VECTORS, HEAD_DIM // 32, 32))
    group_amax = gl.max(gl.abs(grouped), axis=2, keep_dims=False)
    vector_amax = gl.max(group_amax, axis=1, keep_dims=False)
    safe_amax = gl.maximum(vector_amax, 1.0e-4)
    scaled_amax = safe_amax / 448.0
    scaled_bits = scaled_amax.to(gl.uint32, bitcast=True)
    biased_exp = (scaled_bits >> 23) & 0xFF
    has_mantissa = (scaled_bits & 0x7FFFFF) != 0
    scale_byte_base = biased_exp + has_mantissa.to(gl.uint32)
    scale_bits = scale_byte_base << 23
    scale = scale_bits.to(gl.float32, bitcast=True)
    scale_byte = (
        scale_byte_base[:, None] + gl.zeros_like(group_amax).to(gl.uint32)
    ).to(gl.uint8)
    quantized = gl.amd.cdna4.scaled_downcast(
        values,
        scale_byte,
        "e4m3",
        axis=1,
    )

    gl.store(
        q_out + rows[:, None].to(gl.int64) * HEAD_DIM + dims[None, :],
        quantized,
    )
    scale_layout: gl.constexpr = scale.type.layout
    weight_rows = gl.program_id(0) * BLOCK_VECTORS + gl.arange(
        0,
        BLOCK_VECTORS,
        layout=scale_layout,
    )
    raw_weights = gl.load(weights + weight_rows).to(gl.float32)
    # Preserve the target association instead of multiplying by an algebraic
    # 1/64 shortcut: SGLang applies head scale, query scale, then dot scale.
    combined_weights = ((raw_weights * HEAD_SCALE) * scale) * SOFTMAX_SCALE
    gl.store(combined_weights_out + weight_rows, combined_weights)


def _validate_output(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} on q.device with shape {shape}; "
            f"got shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, contiguous={value.is_contiguous()}"
        )


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.numel()
        and right.numel()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def gluon_dsa_prepare_kpool_query_gfx950(
    q: torch.Tensor,
    weights: torch.Tensor,
    model_scale: float,
    *,
    q_out: torch.Tensor | None,
    combined_weights_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a complete query chunk once and fold its scorer weights.

    ``q`` must be contiguous BF16 ``[tokens, 32, 128]`` and ``weights`` must
    be contiguous BF16 or FP32 ``[tokens, 32]``. Optional output buffers make
    the operation usable during graph replay without changing their pointers.
    """
    if abs(float(model_scale) - _MODEL_SCALE) > 1.0e-12:
        raise ValueError(
            "GFX950 KPool query preparation requires the GLM-5.3 model scale "
            f"1/64, got {model_scale}"
        )
    if q.dim() != 3 or tuple(q.shape[1:]) != (32, _HEAD_DIM):
        raise ValueError(
            f"KPool query preparation requires q=[tokens, 32, 128], got "
            f"{tuple(q.shape)}"
        )
    if q.dtype != torch.bfloat16:
        raise TypeError(f"KPool query preparation requires BF16 q, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("KPool query preparation requires contiguous q")
    if not q.is_cuda:
        raise RuntimeError("KPool query preparation requires CUDA tensors")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"KPool query preparation requires BF16 or FP32 weights, got "
            f"{weights.dtype}"
        )
    if weights.shape != q.shape[:2] or not weights.is_contiguous():
        raise ValueError("weights must match q and be contiguous")
    if weights.device != q.device:
        raise ValueError("weights must be on the same device as q")

    q_shape = tuple(q.shape)
    weights_shape = tuple(weights.shape)
    if q_out is None:
        q_out = torch.empty(q_shape, dtype=torch.float8_e4m3fn, device=q.device)
    else:
        _validate_output(
            q_out,
            name="q_out",
            shape=q_shape,
            dtype=torch.float8_e4m3fn,
            device=q.device,
        )
    if _shares_storage(q_out, q):
        raise ValueError("q_out must not alias q during graph replay")
    if combined_weights_out is None:
        combined_weights_out = torch.empty(
            weights_shape,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        _validate_output(
            combined_weights_out,
            name="combined_weights_out",
            shape=weights_shape,
            dtype=torch.float32,
            device=q.device,
        )
    if _shares_storage(combined_weights_out, weights):
        raise ValueError(
            "combined_weights_out must not alias weights because graph replay "
            "would apply the folded scale repeatedly"
        )
    if _shares_storage(combined_weights_out, q):
        raise ValueError("combined_weights_out must not alias q during graph replay")

    tokens = int(q.shape[0])
    if tokens == 0:
        return q_out, combined_weights_out
    _dsa_prepare_kpool_query_kernel[(tokens,)](
        q,
        weights,
        q_out,
        combined_weights_out,
        HEAD_SCALE=_HEAD_SCALE,
        SOFTMAX_SCALE=_SOFTMAX_SCALE,
        BLOCK_VECTORS=_BLOCK_VECTORS,
        HEAD_DIM=_HEAD_DIM,
        num_warps=_NUM_WARPS,
        waves_per_eu=4,
    )
    return q_out, combined_weights_out
