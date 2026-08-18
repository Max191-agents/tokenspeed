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

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_FP8_DTYPE = torch.float8_e4m3fn

_MLA_PREPARE_FP8_SIGNATURE = format_signature(
    q_nope=dense_tensor_format(torch.bfloat16),
    projection_weight=dense_tensor_format(torch.bfloat16),
    q_rope=dense_tensor_format(torch.bfloat16),
    k_nope=dense_tensor_format(torch.bfloat16),
    k_rope=dense_tensor_format(torch.bfloat16),
    out=dense_tensor_format(_FP8_DTYPE),
)


def _is_python_one(value: float | torch.Tensor) -> bool:
    return not isinstance(value, torch.Tensor) and float(value) == 1.0


def _producer_traits(
    q_nope: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    cos_sin_cache: torch.Tensor | None,
    absorbed_query: torch.Tensor | None,
    prequantized_key: torch.Tensor | None,
    quant_scale_q: float | torch.Tensor,
) -> dict[str, object]:
    return {
        "num_tokens": q_nope.shape[0],
        "num_heads": q_nope.shape[1],
        "q_nope_dim": q_nope.shape[2],
        "kv_lora_rank": projection_weight.shape[1],
        "has_rope": cos_sin_cache is not None,
        "has_absorbed_query": absorbed_query is not None,
        "has_prequantized_key": prequantized_key is not None,
        "q_scale_is_one": _is_python_one(quant_scale_q),
    }


def _portable_mla_prepare_fp8_query(
    q_nope: torch.Tensor,
    projection_weight: torch.Tensor,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    *,
    absorbed_query: torch.Tensor | None,
    is_neox: bool,
    quant_scale_q: float | torch.Tensor,
    quant_scale_kv: float | torch.Tensor,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.ops.embedding import apply_rope_mla
    from tokenspeed_kernel.ops.gemm import bmm

    tokens, heads, _ = q_nope.shape
    latent_dim = projection_weight.shape[1]
    if absorbed_query is None:
        projected_nope = q_nope.new_empty(tokens, heads, latent_dim)
    else:
        projected_nope = absorbed_query[..., :latent_dim]

    bmm(
        q_nope.transpose(0, 1),
        projection_weight,
        out=projected_nope.transpose(0, 1),
        enable_pdl=enable_pdl,
    )
    return apply_rope_mla(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=projected_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        enable_pdl=enable_pdl,
    )


@register_kernel(
    "attention",
    "mla_prepare_fp8_query",
    name="composite_mla_prepare_fp8_query",
    solution="composite",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=frozenset({_MLA_PREPARE_FP8_SIGNATURE}),
    traits={"has_prequantized_key": frozenset({False})},
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def composite_mla_prepare_fp8_query(
    q_nope: torch.Tensor,
    projection_weight: torch.Tensor,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    *,
    absorbed_query: torch.Tensor | None = None,
    prequantized_key: torch.Tensor | None = None,
    is_neox: bool = True,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a BF16 query BMM with the selected MLA RoPE implementation."""
    if prequantized_key is not None:
        raise ValueError(
            "the composite MLA preparation path requires a raw normalized key"
        )
    return _portable_mla_prepare_fp8_query(
        q_nope,
        projection_weight,
        positions,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        absorbed_query=absorbed_query,
        is_neox=is_neox,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        enable_pdl=enable_pdl,
    )


@register_kernel(
    "attention",
    "mla_prepare_fp8_query",
    name="producer_mla_prepare_fp8_query_amd",
    solution="producer",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=frozenset({_MLA_PREPARE_FP8_SIGNATURE}),
    traits={
        "has_rope": frozenset({True}),
        "has_absorbed_query": frozenset({False}),
        "q_scale_is_one": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
    tags={"latency", "throughput"},
)
def producer_mla_prepare_fp8_query_amd(
    q_nope: torch.Tensor,
    projection_weight: torch.Tensor,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    *,
    absorbed_query: torch.Tensor | None = None,
    prequantized_key: torch.Tensor | None = None,
    is_neox: bool = True,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare AMD MLA inputs with producer-side query quantization."""
    traits = _producer_traits(
        q_nope,
        projection_weight,
        cos_sin_cache=cos_sin_cache,
        absorbed_query=absorbed_query,
        prequantized_key=prequantized_key,
        quant_scale_q=quant_scale_q,
    )
    if any(
        (
            not traits["has_rope"],
            traits["has_absorbed_query"],
            not traits["q_scale_is_one"],
        )
    ):
        if prequantized_key is not None:
            raise ValueError(
                "prequantized_key requires RoPE, no absorbed_query, and unit query scale"
            )
        return _portable_mla_prepare_fp8_query(
            q_nope,
            projection_weight,
            positions,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            absorbed_query=absorbed_query,
            is_neox=is_neox,
            quant_scale_q=quant_scale_q,
            quant_scale_kv=quant_scale_kv,
            enable_pdl=enable_pdl,
        )

    from tokenspeed_kernel.ops.embedding import rope_mla_fp8_producer
    from tokenspeed_kernel.ops.gemm import bmm_fp8_output

    tokens, heads, _ = q_nope.shape
    latent_dim = projection_weight.shape[1]
    query_out = torch.empty(
        (tokens, heads, latent_dim + q_rope.shape[-1]),
        dtype=_FP8_DTYPE,
        device=q_nope.device,
    )
    bmm_fp8_output(
        q_nope.transpose(0, 1),
        projection_weight,
        out=query_out[..., :latent_dim].transpose(0, 1),
        enable_pdl=enable_pdl,
    )
    return rope_mla_fp8_producer(
        positions=positions,
        q_rope=q_rope,
        k_rope=k_rope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        query_out=query_out,
        key_out=prequantized_key,
        enable_pdl=enable_pdl,
    )


def mla_prepare_fp8_query(
    q_nope: torch.Tensor,
    projection_weight: torch.Tensor,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    *,
    absorbed_query: torch.Tensor | None = None,
    prequantized_key: torch.Tensor | None = None,
    is_neox: bool = True,
    quant_scale_q: float | torch.Tensor = 1.0,
    quant_scale_kv: float | torch.Tensor = 1.0,
    enable_pdl: bool = False,
    solution: str | None = None,
    override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project and prepare query/key tensors for FP8 MLA attention.

    The selected strategy delegates query projection to ``bmm_fp8_output``,
    which chooses either a native staged-E4M3 kernel or the ordinary BF16 BMM
    plus conversion. A caller-owned BF16 projection destination instead uses
    the raw BF16 composition so its prefix is populated.

    Args:
        q_nope: Raw query non-RoPE channels shaped ``[tokens, heads, q_dim]``.
        projection_weight: Per-head projection shaped
            ``[heads, latent_dim, q_dim]``.
        positions: Token positions consumed by RoPE.
        q_rope: Raw query RoPE channels shaped ``[tokens, heads, rope_dim]``.
        k_nope: Raw key latent channels shaped
            ``[tokens, kv_heads, latent_dim]``.
        k_rope: Raw key RoPE channels shaped
            ``[tokens, kv_heads, rope_dim]``.
        cos_sin_cache: Packed RoPE cache, or ``None`` to skip rotation.
        absorbed_query: Optional contiguous BF16 destination shaped
            ``[tokens, heads, latent_dim + rope_dim]``. Its latent prefix is
            populated by the query projection and forces the portable path.
        prequantized_key: Optional contiguous E4M3 key tensor whose latent
            prefix is already populated using ``quant_scale_kv``. A
            producer strategy preserves and returns that buffer. Combinations
            that require the raw BF16 composition reject a prequantized key.
        is_neox: Whether to use Neox-style half-split rotation.
        quant_scale_q: Scale applied before query E4M3 conversion.
        quant_scale_kv: Scale applied before key E4M3 conversion.
        enable_pdl: Request Programmatic Dependent Launch where supported.
        solution: Restrict selection to one registered preparation strategy.
        override: Force a specific registered preparation implementation.

    Returns:
        Combined E4M3 query and key tensors, each laid out as
        ``concat(non_rope, rope)`` on the last dimension.
    """
    if q_nope.ndim != 3:
        raise ValueError("q_nope must have shape [tokens, heads, q_dim]")
    tokens, heads, q_dim = q_nope.shape
    if tokens <= 0 or heads <= 0 or q_dim <= 0:
        raise ValueError("q_nope token count, head count, and width must be positive")
    if projection_weight.ndim != 3 or projection_weight.shape[0] != heads:
        raise ValueError("projection_weight must have shape [heads, latent_dim, q_dim]")
    latent_dim = projection_weight.shape[1]
    if latent_dim <= 0 or projection_weight.shape[2] != q_dim:
        raise ValueError("projection_weight must have shape [heads, latent_dim, q_dim]")
    if q_rope.ndim != 3 or q_rope.shape[:2] != (tokens, heads):
        raise ValueError("q_rope must have shape [tokens, heads, rope_dim]")
    rope_dim = q_rope.shape[2]
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("q_rope width must be positive and even")
    if k_nope.ndim != 3 or k_nope.shape[0] != tokens:
        raise ValueError("k_nope must have shape [tokens, kv_heads, latent_dim]")
    kv_heads = k_nope.shape[1]
    if kv_heads <= 0 or k_nope.shape[2] != latent_dim:
        raise ValueError("k_nope must have shape [tokens, kv_heads, latent_dim]")
    if k_rope.shape != (tokens, kv_heads, rope_dim):
        raise ValueError("k_rope must have shape [tokens, kv_heads, rope_dim]")

    channel_inputs = (
        (q_nope, "q_nope"),
        (q_rope, "q_rope"),
        (k_nope, "k_nope"),
        (k_rope, "k_rope"),
    )
    for tensor, name in channel_inputs:
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16, got {tensor.dtype}")
        if tensor.device != q_nope.device:
            raise ValueError(f"{name} must be on {q_nope.device}, got {tensor.device}")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have a contiguous last dimension")
    if projection_weight.dtype != torch.bfloat16:
        raise TypeError(
            "projection_weight must use torch.bfloat16, got "
            f"{projection_weight.dtype}"
        )
    if projection_weight.device != q_nope.device:
        raise ValueError("projection_weight must share the input device")

    if cos_sin_cache is not None:
        if cos_sin_cache.ndim != 2:
            raise ValueError("cos_sin_cache must have shape [positions, rope_dim]")
        if cos_sin_cache.shape[1] != rope_dim:
            raise ValueError("cos_sin_cache width must match rope_dim")
        if cos_sin_cache.device != q_nope.device:
            raise ValueError("cos_sin_cache must share the input device")
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("positions must use torch.int32 or torch.int64")
        if positions.numel() != tokens:
            raise ValueError("positions must contain one entry per token")
        if positions.device != q_nope.device:
            raise ValueError("positions must share the input device")
    for name, scale in (
        ("quant_scale_q", quant_scale_q),
        ("quant_scale_kv", quant_scale_kv),
    ):
        if isinstance(scale, torch.Tensor):
            if scale.numel() != 1:
                raise ValueError(f"{name} must contain one value")
            if scale.device != q_nope.device:
                raise ValueError(f"{name} must share the input device")

    if absorbed_query is not None:
        expected_query_shape = (tokens, heads, latent_dim + rope_dim)
        if absorbed_query.shape != expected_query_shape:
            raise ValueError(
                "absorbed_query must have shape "
                f"{expected_query_shape}, got {tuple(absorbed_query.shape)}"
            )
        if absorbed_query.dtype != torch.bfloat16:
            raise TypeError("absorbed_query must use torch.bfloat16")
        if absorbed_query.device != q_nope.device:
            raise ValueError("absorbed_query must share the input device")
        if not absorbed_query.is_contiguous():
            raise ValueError("absorbed_query must be contiguous")

    if prequantized_key is not None:
        expected_key_shape = (tokens, kv_heads, latent_dim + rope_dim)
        if prequantized_key.shape != expected_key_shape:
            raise ValueError(
                "prequantized_key must have shape "
                f"{expected_key_shape}, got {tuple(prequantized_key.shape)}"
            )
        if prequantized_key.dtype != _FP8_DTYPE:
            raise TypeError(f"prequantized_key must use {_FP8_DTYPE}")
        if prequantized_key.device != q_nope.device:
            raise ValueError("prequantized_key must share the input device")
        if not prequantized_key.is_contiguous():
            raise ValueError("prequantized_key must be contiguous")
        if (
            cos_sin_cache is None
            or absorbed_query is not None
            or not _is_python_one(quant_scale_q)
        ):
            raise ValueError(
                "prequantized_key requires RoPE, no absorbed_query, and unit query scale"
            )

    traits = _producer_traits(
        q_nope,
        projection_weight,
        cos_sin_cache=cos_sin_cache,
        absorbed_query=absorbed_query,
        prequantized_key=prequantized_key,
        quant_scale_q=quant_scale_q,
    )
    kernel = select_kernel(
        "attention",
        "mla_prepare_fp8_query",
        _MLA_PREPARE_FP8_SIGNATURE,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "num_tokens": tokens,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "q_nope_dim": q_dim,
        "kv_lora_rank": latent_dim,
        "rope_dim": rope_dim,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_prepare_fp8_query",
        kernel.name,
        q_nope.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "mla_prepare_fp8_query",
        q_nope.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q_nope,
            projection_weight,
            positions,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            absorbed_query=absorbed_query,
            prequantized_key=prequantized_key,
            is_neox=is_neox,
            quant_scale_q=quant_scale_q,
            quant_scale_kv=quant_scale_kv,
            enable_pdl=enable_pdl,
        )


__all__ = ["mla_prepare_fp8_query"]
