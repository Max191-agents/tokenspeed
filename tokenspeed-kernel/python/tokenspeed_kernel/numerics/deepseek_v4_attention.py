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

"""TokenSpeed DeepSeek V4 helper-kernel reference adapters.

The input generator package exposes operation-level CSA values. This module
keeps TokenSpeed helper-kernel tests in terms of the concrete DeepSeek V4
helper operation names and performs any packed FP8/MXFP4 adaptation outside the
generator package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from tokenspeed_numerics_input_generators.attention._core import (
    DeviceLike,
    PageTableInput,
    PageTableInputConfig,
    TensorInput,
    _CSA_FP8_MAX,
    _CSA_FP8_QUANT_BLOCK,
    _CSA_HEAD_DIM,
    _CSA_INDEXER_DIM,
    _CSA_INDEXER_MXFP4_BLOCK_SIZE,
    _CSA_INDEXER_MXFP4_HALF_BLOCK,
    _CSA_INDEXER_MXFP4_SCALE_BYTES,
    _CSA_INDEXER_MXFP4_VALUE_BYTES,
    _CSA_NOPE_DIM,
    _CSA_ROPE_DIM,
    _CSA_SWA_SCALE_DIM,
    _CSA_SWA_TOKEN_STRIDE,
    _check_float_dtype,
    _check_nonnegative,
    _check_positive,
    _child_seed,
    _compression_boundary_positions,
    _compression_non_boundary_positions,
    _generate_compression_positions,
    _generate_random_positions,
    _generate_random_uint8_tensor,
    _generate_slot_mapping_tensor,
    _generate_token_to_request_indices,
    _generate_valid_mask,
    _require_tensor,
    _resolve_attention_seeds,
    _resolve_device,
    _rng_for_device,
    build_rope_cos_sin_cache,
)
from tokenspeed_numerics_input_generators.attention.csa import (
    CSACompressorStateValues,
    CSAIndexerQueryValues,
    CSAInputValues,
    CSAPagedIndexValues,
    CSASparsePrefillIndexValues,
    csa_build_dense_prefill_local_compressed_indices_reference,
    csa_combine_dense_swa_indices_reference,
    csa_combine_topk_swa_indices_reference,
    csa_compressed_slot_mapping_reference,
    csa_compute_global_topk_indices_and_lens_reference,
    csa_decode_swa_indices_and_lens_reference,
    csa_indexer_decode_metadata_reference,
    csa_save_compressor_state_reference,
)

DeepSeekV4CompressorStateInputValues = CSACompressorStateValues
DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues = CSAIndexerQueryValues
DeepSeekV4PagedIndexValues = CSAPagedIndexValues
DeepSeekV4SparsePrefillIndexValues = CSASparsePrefillIndexValues

deepseek_v4_build_dense_prefill_local_compressed_indices_reference = (
    csa_build_dense_prefill_local_compressed_indices_reference
)
deepseek_v4_combine_dense_swa_indices_reference = (
    csa_combine_dense_swa_indices_reference
)
deepseek_v4_combine_topk_swa_indices_reference = csa_combine_topk_swa_indices_reference
deepseek_v4_compressed_slot_mapping_reference = csa_compressed_slot_mapping_reference
deepseek_v4_compute_global_topk_indices_and_lens_reference = (
    csa_compute_global_topk_indices_and_lens_reference
)
deepseek_v4_decode_swa_indices_and_lens_reference = (
    csa_decode_swa_indices_and_lens_reference
)
deepseek_v4_indexer_decode_metadata_reference = csa_indexer_decode_metadata_reference
deepseek_v4_save_compressor_state_reference = csa_save_compressor_state_reference

def _deepseek_v4_apply_indexer_q_rope(
    values: CSAIndexerQueryValues,
) -> torch.Tensor:
    q = values.index_q.float()
    positions = values.positions.to(torch.int64)
    positions = positions[:, None].expand(q.shape[0], q.shape[1]).reshape(-1)
    rotated = _deepseek_v4_apply_indexer_rope_rows(
        q.reshape(-1, _CSA_INDEXER_DIM),
        positions,
        values.cos_sin_cache,
    )
    return rotated.reshape_as(q)


def _deepseek_v4_apply_indexer_rope_rows(
    rows: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    rows = rows.float()
    nope_dim = _CSA_INDEXER_DIM - _CSA_ROPE_DIM
    half_rope = _CSA_ROPE_DIM // 2
    rope = rows[..., nope_dim:]
    rope_even = rope[..., 0::2]
    rope_odd = rope[..., 1::2]
    cos_sin = cos_sin_cache[positions.to(torch.int64)]
    cos_v = cos_sin[..., :half_rope].float()
    sin_v = cos_sin[..., half_rope:].float()
    rotated_rope = torch.empty_like(rope)
    rotated_rope[..., 0::2] = rope_even * cos_v - rope_odd * sin_v
    rotated_rope[..., 1::2] = rope_odd * cos_v + rope_even * sin_v
    rotated = torch.cat((rows[..., :nope_dim], rotated_rope), dim=-1)
    return rotated.to(torch.bfloat16).to(torch.float32)


def _deepseek_v4_indexer_hadamard_signs(device: torch.device) -> torch.Tensor:
    indices = torch.arange(
        _CSA_INDEXER_DIM,
        dtype=torch.int32,
        device=device,
    )
    bits = indices[:, None] & indices[None, :]
    parity = bits ^ (bits >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    parity = parity & 1
    return torch.where(
        parity == 0,
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(-1.0, dtype=torch.float32, device=device),
    )


def _deepseek_v4_indexer_q_hadamard(
    rotated: torch.Tensor,
) -> torch.Tensor:
    signs = _deepseek_v4_indexer_hadamard_signs(rotated.device)
    flat = rotated.reshape(-1, _CSA_INDEXER_DIM)
    projected = flat @ signs
    projected = projected * (_CSA_INDEXER_DIM**-0.5)
    projected = projected.reshape_as(rotated)
    return projected.to(torch.bfloat16).to(torch.float32)


def _validate_deepseek_v4_indexer_q_rope_hadamard_mxfp4_values(
    values: CSAIndexerQueryValues,
) -> None:
    if values.index_q.ndim != 3:
        raise ValueError(f"index_q must be rank-3, got {values.index_q.ndim}")
    if values.index_q.shape[-1] != _CSA_INDEXER_DIM:
        raise ValueError(
            f"index_q width must be {_CSA_INDEXER_DIM}, "
            f"got {values.index_q.shape[-1]}"
        )
    if not values.index_q.is_floating_point():
        raise TypeError(f"index_q must be floating point, got {values.index_q.dtype}")
    if values.positions.ndim != 1:
        raise ValueError("positions must be rank-1")
    if values.positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be integer, got {values.positions.dtype}")
    if values.positions.numel() != values.index_q.shape[0]:
        raise ValueError("positions length must match index_q token dimension")
    if values.cos_sin_cache.ndim != 2:
        raise ValueError("cos_sin_cache must be rank-2")
    if values.cos_sin_cache.shape[1] != _CSA_ROPE_DIM:
        raise ValueError(
            f"cos_sin_cache width must be {_CSA_ROPE_DIM}, "
            f"got {values.cos_sin_cache.shape[1]}"
        )
    if values.weights.shape != values.index_q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(values.index_q.shape[:2])}, "
            f"got {tuple(values.weights.shape)}"
        )
    if not values.weights.is_floating_point():
        raise TypeError(f"weights must be floating point, got {values.weights.dtype}")
    if (
        values.positions.device != values.index_q.device
        or values.cos_sin_cache.device != values.index_q.device
        or values.weights.device != values.index_q.device
    ):
        raise ValueError(
            "index_q, positions, cos_sin_cache, and weights must share device"
        )
    if values.positions.numel():
        min_pos = int(values.positions.min().item())
        max_pos = int(values.positions.max().item())
        if min_pos < 0 or max_pos >= values.cos_sin_cache.shape[0]:
            raise ValueError(
                "positions must be within cos_sin_cache rows, got range "
                f"[{min_pos}, {max_pos}] for cache length {values.cos_sin_cache.shape[0]}"
            )
    if not math.isfinite(values.softmax_scale):
        raise ValueError(f"softmax_scale must be finite, got {values.softmax_scale}")
    if not math.isfinite(values.head_scale):
        raise ValueError(f"head_scale must be finite, got {values.head_scale}")


def deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference(
    values: CSAIndexerQueryValues,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return packed MXFP4 indexer Q and scaled weights for CSA."""

    _validate_deepseek_v4_indexer_q_rope_hadamard_mxfp4_values(values)
    num_tokens, num_heads, _ = values.index_q.shape
    q_packed = torch.empty(
        (
            num_tokens,
            num_heads,
            _CSA_INDEXER_MXFP4_VALUE_BYTES,
        ),
        dtype=torch.uint8,
        device=values.index_q.device,
    )
    q_scale_bytes = torch.empty(
        (
            num_tokens,
            num_heads,
            _CSA_INDEXER_MXFP4_SCALE_BYTES,
        ),
        dtype=torch.uint8,
        device=values.index_q.device,
    )
    weights_out = (
        values.weights.float() * values.softmax_scale * values.head_scale
    ).contiguous()
    if num_tokens == 0:
        return (
            q_packed,
            q_scale_bytes.view(torch.int32).squeeze(-1).contiguous(),
        ), weights_out

    rotated = _deepseek_v4_apply_indexer_q_rope(values)
    hadamard = _deepseek_v4_indexer_q_hadamard(rotated)
    packed_rows = q_packed.reshape(-1, _CSA_INDEXER_MXFP4_VALUE_BYTES)
    scale_rows = q_scale_bytes.reshape(-1, _CSA_INDEXER_MXFP4_SCALE_BYTES)
    for row_idx, row in enumerate(hadamard.reshape(-1, _CSA_INDEXER_DIM)):
        packed, scales = _deepseek_v4_indexer_mxfp4_row_reference(row)
        packed_rows[row_idx] = packed
        scale_rows[row_idx] = scales
    return (
        q_packed.contiguous(),
        q_scale_bytes.view(torch.int32).squeeze(-1).contiguous(),
    ), weights_out


@dataclass
class DeepSeekV4InvRoPEFP8QuantInputValues:
    """Generated values for CSA inverse-RoPE FP8 quantization."""

    o: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    n_groups: int
    heads_per_group: int
    nope_dim: int
    rope_dim: int
    quant_group_size: int
    tma_aligned_scales: bool


@dataclass
class _DeepSeekV4InvRoPEFP8QuantConfig:
    """Initialization parameters for inverse-RoPE FP8 output quantization.

    The represented operation starts from attention output rows shaped as
    ``[num_tokens, num_heads, head_dim]``. It applies the inverse RoPE transform
    to the final rotary channels, groups heads into ``n_groups`` output groups,
    and block-quantizes each grouped row to FP8 E4M3 with power-of-two scales.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of attention-output token rows.
    num_tokens: int

    # Required: number of output-projection groups.
    n_groups: int

    # Required: number of heads packed into each output-projection group.
    heads_per_group: int

    # Required: generated dtype for attention output rows before FP8 quant.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional shape and quantization configuration.
    # ------------------------------------------------------------------

    # Optional: attention output width per head. CSA uses 512.
    head_dim: int = _CSA_HEAD_DIM

    # Optional: non-rotary prefix width per head. CSA uses 448.
    nope_dim: int = _CSA_NOPE_DIM

    # Optional: rotary suffix width per head. CSA uses 64.
    rope_dim: int = _CSA_ROPE_DIM

    # Optional: number of contiguous values sharing one FP8 scale.
    quant_group_size: int = _CSA_FP8_QUANT_BLOCK * 2

    # Optional: when true, pack each head's four UE8M0 scale bytes into int32
    # values laid out for the DeepGEMM TMA-aligned path.
    tma_aligned_scales: bool = True

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: number of RoPE cache rows. Generated positions are in
    # [0, max_position), so this controls the generated absolute-position range.
    max_position: int = 1024

    # Optional: RoPE frequency base used to build the generated cos/sin cache.
    rope_base: float = 10000.0

    # Optional: dtype for generated position metadata.
    position_dtype: torch.dtype = torch.int64

    # Optional: scale applied to generated output values before quantization.
    value_scale: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DeepSeekV4InvRoPEFP8QuantBuilder:
    """Generator for CSA inverse-RoPE FP8 output quantization inputs."""

    config: _DeepSeekV4InvRoPEFP8QuantConfig
    o_input: TensorInput | None

    def __init__(self, config: _DeepSeekV4InvRoPEFP8QuantConfig) -> None:
        self.config = config
        self.o_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.o_input = self.o_input or TensorInput(
            self._o_shape(),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4InvRoPEFP8QuantInputValues:
        self.__post_init__()
        if self.o_input is None:
            raise ValueError("o_input must be initialized")
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        self.o_input.shape = self._o_shape()
        self.o_input.dtype = self.config.dtype
        o = _require_tensor(
            self.o_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "o",
        )
        values = DeepSeekV4InvRoPEFP8QuantInputValues(
            o=(o.float() * self.config.value_scale).to(o.dtype).contiguous(),
            positions=self._generate_positions(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            cos_sin_cache=build_rope_cos_sin_cache(
                rotary_dim=self.config.rope_dim,
                max_position=self.config.max_position,
                base=self.config.rope_base,
                device=target_device,
            ),
            n_groups=self.config.n_groups,
            heads_per_group=self.config.heads_per_group,
            nope_dim=self.config.nope_dim,
            rope_dim=self.config.rope_dim,
            quant_group_size=self.config.quant_group_size,
            tma_aligned_scales=self.config.tma_aligned_scales,
        )
        _validate_deepseek_v4_inv_rope_fp8_quant_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens",
            self.config.num_tokens,
        )
        self.config.n_groups = _check_positive("n_groups", self.config.n_groups)
        self.config.heads_per_group = _check_positive(
            "heads_per_group",
            self.config.heads_per_group,
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.nope_dim = _check_nonnegative("nope_dim", self.config.nope_dim)
        self.config.rope_dim = _check_positive("rope_dim", self.config.rope_dim)
        if self.config.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {self.config.rope_dim}")
        if self.config.nope_dim + self.config.rope_dim != self.config.head_dim:
            raise ValueError(
                "nope_dim + rope_dim must equal head_dim for inverse-RoPE "
                f"quantization, got {self.config.nope_dim} + "
                f"{self.config.rope_dim} != {self.config.head_dim}"
            )
        self.config.quant_group_size = _check_positive(
            "quant_group_size",
            self.config.quant_group_size,
        )
        if self.config.head_dim % self.config.quant_group_size != 0:
            raise ValueError(
                "head_dim must be divisible by quant_group_size, got "
                f"head_dim={self.config.head_dim}, "
                f"quant_group_size={self.config.quant_group_size}"
            )
        if self.config.rope_dim > self.config.quant_group_size:
            raise ValueError(
                "rope_dim must fit in the final quantization group, got "
                f"rope_dim={self.config.rope_dim}, "
                f"quant_group_size={self.config.quant_group_size}"
            )
        chunks_per_head = self.config.head_dim // self.config.quant_group_size
        if self.config.tma_aligned_scales and chunks_per_head != 4:
            raise ValueError(
                "tma_aligned_scales requires exactly four quantization groups "
                f"per head, got {chunks_per_head}"
            )
        self.config.max_position = _check_positive(
            "max_position",
            self.config.max_position,
        )
        self.config.rope_base = float(self.config.rope_base)
        if self.config.rope_base <= 0.0 or not math.isfinite(self.config.rope_base):
            raise ValueError(f"rope_base must be positive, got {self.config.rope_base}")
        if self.config.position_dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "position_dtype must be torch.int32 or torch.int64, got "
                f"{self.config.position_dtype}"
            )
        self.config.value_scale = float(self.config.value_scale)
        if self.config.value_scale < 0.0 or not math.isfinite(self.config.value_scale):
            raise ValueError(
                f"value_scale must be finite and non-negative, got {self.config.value_scale}"
            )

    def _num_heads(self) -> int:
        return self.config.n_groups * self.config.heads_per_group

    def _o_shape(self) -> tuple[int, int, int]:
        return (self.config.num_tokens, self._num_heads(), self.config.head_dim)

    def _generate_positions(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_random_positions(
            num_tokens=self.config.num_tokens,
            max_position=self.config.max_position,
            dtype=self.config.position_dtype,
            seed=seed,
            device=device,
        )


def _validate_deepseek_v4_inv_rope_fp8_quant_values(
    values: DeepSeekV4InvRoPEFP8QuantInputValues,
) -> None:
    if values.o.ndim != 3:
        raise ValueError(f"o must be rank-3, got {values.o.ndim}")
    n_groups = _check_positive("n_groups", values.n_groups)
    heads_per_group = _check_positive("heads_per_group", values.heads_per_group)
    expected_heads = n_groups * heads_per_group
    if values.o.shape[1] != expected_heads:
        raise ValueError(
            f"o head dimension must be n_groups * heads_per_group={expected_heads}, "
            f"got {values.o.shape[1]}"
        )
    if not values.o.is_floating_point():
        raise TypeError(f"o must be floating point, got {values.o.dtype}")
    if values.positions.ndim != 1:
        raise ValueError("positions must be rank-1")
    if values.positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be integer, got {values.positions.dtype}")
    if values.positions.numel() != values.o.shape[0]:
        raise ValueError("positions length must match o token dimension")
    if values.cos_sin_cache.ndim != 2:
        raise ValueError("cos_sin_cache must be rank-2")
    rope_dim = _check_positive("rope_dim", values.rope_dim)
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    if values.cos_sin_cache.shape[1] != rope_dim:
        raise ValueError(
            f"cos_sin_cache width must be rope_dim={rope_dim}, "
            f"got {values.cos_sin_cache.shape[1]}"
        )
    nope_dim = _check_nonnegative("nope_dim", values.nope_dim)
    head_dim = values.o.shape[-1]
    if nope_dim + rope_dim != head_dim:
        raise ValueError(
            f"nope_dim + rope_dim must match o head_dim={head_dim}, "
            f"got {nope_dim} + {rope_dim}"
        )
    quant_group_size = _check_positive("quant_group_size", values.quant_group_size)
    if head_dim % quant_group_size != 0:
        raise ValueError("o head_dim must be divisible by quant_group_size")
    if rope_dim > quant_group_size:
        raise ValueError("rope_dim must fit in the final quantization group")
    chunks_per_head = head_dim // quant_group_size
    if values.tma_aligned_scales and chunks_per_head != 4:
        raise ValueError(
            "tma_aligned_scales requires exactly four quantization groups per head"
        )
    if (
        values.positions.device != values.o.device
        or values.cos_sin_cache.device != values.o.device
    ):
        raise ValueError("o, positions, and cos_sin_cache must share device")
    if values.positions.numel():
        min_pos = int(values.positions.min().item())
        max_pos = int(values.positions.max().item())
        if min_pos < 0 or max_pos >= values.cos_sin_cache.shape[0]:
            raise ValueError(
                "positions must be within cos_sin_cache rows, got range "
                f"[{min_pos}, {max_pos}] for cache length {values.cos_sin_cache.shape[0]}"
            )


def _deepseek_v4_apply_inverse_rope(
    values: DeepSeekV4InvRoPEFP8QuantInputValues,
) -> torch.Tensor:
    out = values.o.float().clone()
    half_rope = values.rope_dim // 2
    rope = out[..., values.nope_dim : values.nope_dim + values.rope_dim]
    even = rope[..., 0::2].clone()
    odd = rope[..., 1::2].clone()
    cos_sin = values.cos_sin_cache[values.positions.to(torch.int64)]
    cos = cos_sin[:, None, :half_rope].float()
    sin = cos_sin[:, None, half_rope:].float()
    rope[..., 0::2] = even * cos + odd * sin
    rope[..., 1::2] = odd * cos - even * sin
    return out


def _deepseek_v4_grouped_fp8_quant(
    x: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    quant_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _, head_dim = x.shape
    grouped = x.reshape(num_tokens, n_groups, heads_per_group * head_dim)
    num_scale_blocks = grouped.shape[-1] // quant_group_size
    block_view = grouped.reshape(
        num_tokens,
        n_groups,
        num_scale_blocks,
        quant_group_size,
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    max_abs = block_view.abs().amax(dim=-1).clamp(min=1.0e-10)
    scales = torch.exp2(torch.ceil(torch.log2(max_abs / fp8_max))).to(torch.float32)
    scaled = (block_view / scales.unsqueeze(-1)).clamp(
        min=-fp8_max,
        max=fp8_max,
    )
    quantized = scaled.to(torch.float8_e4m3fn).reshape_as(grouped)
    return quantized.contiguous(), scales.contiguous()


def _deepseek_v4_pack_tma_aligned_scale_int32(
    scales: torch.Tensor,
    *,
    heads_per_group: int,
    chunks_per_head: int,
) -> torch.Tensor:
    scale_exp = torch.round(torch.log2(scales)).to(torch.int32)
    scale_bytes = (scale_exp + 127).clamp(min=0, max=255).to(torch.int32)
    scale_bytes = scale_bytes.reshape(
        scales.shape[0],
        scales.shape[1],
        heads_per_group,
        chunks_per_head,
    )
    shifts = torch.arange(chunks_per_head, dtype=torch.int32, device=scales.device) * 8
    return torch.sum(scale_bytes << shifts, dim=-1).to(torch.int32).contiguous()


def deepseek_v4_inv_rope_fp8_quant_reference(
    values: DeepSeekV4InvRoPEFP8QuantInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return grouped FP8 output rows and scales after inverse RoPE."""

    _validate_deepseek_v4_inv_rope_fp8_quant_values(values)
    inv_rope = _deepseek_v4_apply_inverse_rope(values)
    fp8, scales = _deepseek_v4_grouped_fp8_quant(
        inv_rope,
        n_groups=values.n_groups,
        heads_per_group=values.heads_per_group,
        quant_group_size=values.quant_group_size,
    )
    if not values.tma_aligned_scales:
        return fp8, scales
    return fp8, _deepseek_v4_pack_tma_aligned_scale_int32(
        scales,
        heads_per_group=values.heads_per_group,
        chunks_per_head=values.o.shape[-1] // values.quant_group_size,
    )


@dataclass
class DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues:
    """Generated values for CSA indexer MXFP4 cache inserts."""

    state_cache: torch.Tensor
    token_to_req_indices: torch.Tensor
    positions: torch.Tensor
    compressor_slot_mapping: torch.Tensor
    block_table: torch.Tensor
    compressor_block_size: int
    rms_norm_weight: torch.Tensor
    rms_norm_eps: float
    cos_sin_cache: torch.Tensor
    kv_cache_2d: torch.Tensor
    kv_slot_mapping: torch.Tensor
    kv_cache_block_size: int
    compress_ratio: int
    block_table_base_offsets: torch.Tensor | None


@dataclass
class _DeepSeekV4CSAIndexerMXFP4CacheInsertConfig:
    """Initialization parameters for CSA indexer cache inserts.

    The represented operation compresses an overlapping CSA state window,
    normalizes it, applies indexer RoPE/Hadamard, quantizes to MXFP4, and writes
    the result into a paged indexer cache.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of candidate token rows.
    num_tokens: int

    # Required: number of request rows in the generated block table.
    batch_size: int

    # Required: maximum generated sequence position plus one.
    max_seq_len: int

    # Required: number of physical pages in the compressor state cache.
    num_state_cache_blocks: int

    # Required: number of token rows in each compressor state cache page.
    compressor_block_size: int

    # Required: number of physical pages in the output MXFP4 indexer cache.
    num_kv_cache_blocks: int

    # Required: number of indexer rows in each output MXFP4 cache page.
    kv_cache_block_size: int

    # Required: generated dtype for compressor state and RMSNorm weight values.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: generated rows whose position is not a compression boundary.
    non_boundary_token_count: int = 0

    # Optional: rows with compressor_slot_mapping == -1. These rows are skipped.
    negative_compressor_slot_count: int = 0

    # Optional: rows with kv_slot_mapping == -1. These rows are skipped.
    negative_kv_slot_count: int = 0

    # Optional: include zero block-table base offsets in generated values.
    include_block_table_base_offsets: bool = False

    # Optional: CSA indexer compression ratio. CSA indexer uses 4.
    compress_ratio: int = 4

    # Optional: RMSNorm epsilon.
    rms_norm_eps: float = 1.0e-5

    # Optional: RoPE frequency base used to build the generated cos/sin cache.
    rope_base: float = 10000.0

    # Optional: scale applied to generated state-cache values.
    value_scale: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DeepSeekV4CSAIndexerMXFP4CacheInsertBuilder:
    """Generator for CSA indexer MXFP4 cache-insert inputs."""

    config: _DeepSeekV4CSAIndexerMXFP4CacheInsertConfig
    state_cache_input: TensorInput | None
    rms_norm_weight_input: TensorInput | None

    def __init__(
        self,
        config: _DeepSeekV4CSAIndexerMXFP4CacheInsertConfig,
    ) -> None:
        self.config = config
        self.state_cache_input = None
        self.rms_norm_weight_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.state_cache_input = self.state_cache_input or TensorInput(
            self._state_cache_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.rms_norm_weight_input = self.rms_norm_weight_input or TensorInput(
            (_CSA_INDEXER_DIM,),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues:
        self.__post_init__()
        if self.state_cache_input is None or self.rms_norm_weight_input is None:
            raise ValueError("CSA indexer child generators must be initialized")
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        self.state_cache_input.shape = self._state_cache_shape()
        self.state_cache_input.dtype = self.config.dtype
        self.rms_norm_weight_input.shape = (_CSA_INDEXER_DIM,)
        self.rms_norm_weight_input.dtype = self.config.dtype
        state_cache = _require_tensor(
            self.state_cache_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "state_cache",
        )
        rms_norm_weight = _require_tensor(
            self.rms_norm_weight_input.generate(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ).values,
            "rms_norm_weight",
        )
        values = DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues(
            state_cache=(state_cache.float() * self.config.value_scale)
            .to(state_cache.dtype)
            .contiguous(),
            token_to_req_indices=self._generate_token_to_req_indices(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            positions=self._generate_positions(
                seed=_child_seed(metadata_seed, 2),
                device=target_device,
            ),
            compressor_slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 3),
                total_slots=self._total_state_slots(),
                negative_count=self.config.negative_compressor_slot_count,
                device=target_device,
            ),
            block_table=self._generate_block_table(
                seed=_child_seed(metadata_seed, 4),
                device=target_device,
            ),
            compressor_block_size=self.config.compressor_block_size,
            rms_norm_weight=rms_norm_weight.contiguous(),
            rms_norm_eps=self.config.rms_norm_eps,
            cos_sin_cache=build_rope_cos_sin_cache(
                rotary_dim=_CSA_ROPE_DIM,
                max_position=self.config.max_seq_len,
                base=self.config.rope_base,
                device=target_device,
            ),
            kv_cache_2d=self._generate_kv_cache(
                seed=_child_seed(value_seed, 3),
                device=target_device,
            ),
            kv_slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 5),
                total_slots=self._total_kv_slots(),
                negative_count=self.config.negative_kv_slot_count,
                device=target_device,
            ),
            kv_cache_block_size=self.config.kv_cache_block_size,
            compress_ratio=self.config.compress_ratio,
            block_table_base_offsets=(
                torch.zeros(
                    (self.config.batch_size,),
                    dtype=torch.int32,
                    device=target_device,
                )
                if self.config.include_block_table_base_offsets
                else None
            ),
        )
        _validate_deepseek_v4_csa_indexer_mxfp4_cache_insert_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens",
            self.config.num_tokens,
        )
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.max_seq_len = _check_positive(
            "max_seq_len",
            self.config.max_seq_len,
        )
        self.config.num_state_cache_blocks = _check_positive(
            "num_state_cache_blocks",
            self.config.num_state_cache_blocks,
        )
        self.config.compressor_block_size = _check_positive(
            "compressor_block_size",
            self.config.compressor_block_size,
        )
        self.config.num_kv_cache_blocks = _check_positive(
            "num_kv_cache_blocks",
            self.config.num_kv_cache_blocks,
        )
        self.config.kv_cache_block_size = _check_positive(
            "kv_cache_block_size",
            self.config.kv_cache_block_size,
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.compress_ratio = _check_positive(
            "compress_ratio",
            self.config.compress_ratio,
        )
        if self.config.compress_ratio != 4:
            raise ValueError(
                f"CSA indexer uses compress_ratio=4, got {self.config.compress_ratio}"
            )
        self.config.non_boundary_token_count = _check_nonnegative(
            "non_boundary_token_count",
            self.config.non_boundary_token_count,
        )
        self.config.negative_compressor_slot_count = _check_nonnegative(
            "negative_compressor_slot_count",
            self.config.negative_compressor_slot_count,
        )
        self.config.negative_kv_slot_count = _check_nonnegative(
            "negative_kv_slot_count",
            self.config.negative_kv_slot_count,
        )
        for name, value in (
            ("non_boundary_token_count", self.config.non_boundary_token_count),
            (
                "negative_compressor_slot_count",
                self.config.negative_compressor_slot_count,
            ),
            ("negative_kv_slot_count", self.config.negative_kv_slot_count),
        ):
            if value > self.config.num_tokens:
                raise ValueError(f"{name} must be <= num_tokens")
        if self.config.num_tokens - self.config.negative_compressor_slot_count > (
            self._total_state_slots()
        ):
            raise ValueError("generated compressor slots must fit in state_cache")
        if self.config.num_tokens - self.config.negative_kv_slot_count > (
            self._total_kv_slots()
        ):
            raise ValueError("generated kv slots must fit in kv_cache_2d")
        if not self._boundary_positions().numel() and self.config.num_tokens:
            raise ValueError(
                "max_seq_len must contain at least one compression boundary"
            )
        if (
            self.config.non_boundary_token_count
            and not self._non_boundary_positions().numel()
        ):
            raise ValueError(
                "max_seq_len must contain non-boundary positions when "
                "non_boundary_token_count is non-zero"
            )
        self.config.rms_norm_eps = float(self.config.rms_norm_eps)
        if self.config.rms_norm_eps <= 0.0 or not math.isfinite(
            self.config.rms_norm_eps
        ):
            raise ValueError(
                f"rms_norm_eps must be finite and positive, got {self.config.rms_norm_eps}"
            )
        self.config.rope_base = float(self.config.rope_base)
        if self.config.rope_base <= 0.0 or not math.isfinite(self.config.rope_base):
            raise ValueError(f"rope_base must be positive, got {self.config.rope_base}")
        self.config.value_scale = float(self.config.value_scale)
        if self.config.value_scale < 0.0 or not math.isfinite(self.config.value_scale):
            raise ValueError(
                f"value_scale must be finite and non-negative, got {self.config.value_scale}"
            )

    def _state_width(self) -> int:
        return _CSA_INDEXER_DIM * 2

    def _state_cache_shape(self) -> tuple[int, int, int]:
        return (
            self.config.num_state_cache_blocks,
            self.config.compressor_block_size,
            self._state_width() * 2,
        )

    def _total_state_slots(self) -> int:
        return self.config.num_state_cache_blocks * self.config.compressor_block_size

    def _total_kv_slots(self) -> int:
        return self.config.num_kv_cache_blocks * self.config.kv_cache_block_size

    def _block_table_width(self) -> int:
        return math.ceil(self.config.max_seq_len / self.config.compressor_block_size)

    def _boundary_positions(self) -> torch.Tensor:
        return _compression_boundary_positions(
            max_seq_len=self.config.max_seq_len,
            compress_ratio=self.config.compress_ratio,
        )

    def _non_boundary_positions(self) -> torch.Tensor:
        return _compression_non_boundary_positions(
            max_seq_len=self.config.max_seq_len,
            compress_ratio=self.config.compress_ratio,
        )

    def _generate_token_to_req_indices(
        self, *, seed: int, device: torch.device
    ) -> torch.Tensor:
        return _generate_token_to_request_indices(
            num_tokens=self.config.num_tokens,
            batch_size=self.config.batch_size,
            seed=seed,
            device=device,
        )

    def _generate_positions(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_compression_positions(
            num_tokens=self.config.num_tokens,
            max_seq_len=self.config.max_seq_len,
            compress_ratio=self.config.compress_ratio,
            non_boundary_token_count=self.config.non_boundary_token_count,
            seed=seed,
            device=device,
        )

    def _generate_slot_mapping(
        self,
        *,
        seed: int,
        total_slots: int,
        negative_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        return _generate_slot_mapping_tensor(
            num_rows=self.config.num_tokens,
            total_slots=total_slots,
            negative_count=negative_count,
            seed=seed,
            device=device,
        )

    def _generate_block_table(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        table = torch.randint(
            0,
            self.config.num_state_cache_blocks,
            (self.config.batch_size, self._block_table_width()),
            dtype=torch.int64,
            generator=rng,
        )
        return table.to(device)

    def _generate_kv_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_random_uint8_tensor(
            shape=(
                self.config.num_kv_cache_blocks,
                self.config.kv_cache_block_size
                * (
                    _CSA_INDEXER_MXFP4_VALUE_BYTES
                    + _CSA_INDEXER_MXFP4_SCALE_BYTES
                ),
            ),
            seed=seed,
            device=device,
        )


def _validate_deepseek_v4_csa_indexer_mxfp4_cache_insert_values(
    values: DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues,
) -> None:
    if values.state_cache.ndim != 3:
        raise ValueError("state_cache must be rank-3")
    if not values.state_cache.is_floating_point():
        raise TypeError(
            f"state_cache must be floating point, got {values.state_cache.dtype}"
        )
    state_width = values.state_cache.shape[-1] // 2
    if values.state_cache.shape[-1] != _CSA_INDEXER_DIM * 4:
        raise ValueError(
            "CSA indexer state_cache last dimension must be "
            f"{_CSA_INDEXER_DIM * 4}, got {values.state_cache.shape[-1]}"
        )
    if state_width != _CSA_INDEXER_DIM * 2:
        raise ValueError("state_cache state width must be two indexer rows")
    compressor_block_size = _check_positive(
        "compressor_block_size",
        values.compressor_block_size,
    )
    if values.state_cache.shape[1] != compressor_block_size:
        raise ValueError("compressor_block_size must match state_cache.shape[1]")
    if values.compress_ratio != 4:
        raise ValueError(f"compress_ratio must be 4, got {values.compress_ratio}")
    if values.kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {values.kv_cache_2d.dtype}")
    if values.kv_cache_2d.ndim != 2:
        raise ValueError("kv_cache_2d must be rank-2")
    kv_cache_block_size = _check_positive(
        "kv_cache_block_size",
        values.kv_cache_block_size,
    )
    min_row_bytes = kv_cache_block_size * (
        _CSA_INDEXER_MXFP4_VALUE_BYTES + _CSA_INDEXER_MXFP4_SCALE_BYTES
    )
    if values.kv_cache_2d.shape[1] < min_row_bytes:
        raise ValueError(
            f"kv_cache_2d row width must be at least {min_row_bytes}, "
            f"got {values.kv_cache_2d.shape[1]}"
        )
    for name, tensor in (
        ("token_to_req_indices", values.token_to_req_indices),
        ("positions", values.positions),
        ("compressor_slot_mapping", values.compressor_slot_mapping),
        ("kv_slot_mapping", values.kv_slot_mapping),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
        if tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be integer, got {tensor.dtype}")
        if tensor.device != values.state_cache.device:
            raise ValueError(f"{name} must share the state_cache device")
    num_actual = min(
        values.compressor_slot_mapping.numel(),
        values.positions.numel(),
        values.kv_slot_mapping.numel(),
    )
    if values.token_to_req_indices.numel() < num_actual:
        raise ValueError("token_to_req_indices length must cover generated rows")
    if values.block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    if values.block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"block_table must be integer, got {values.block_table.dtype}")
    if values.block_table.device != values.state_cache.device:
        raise ValueError("block_table must share the state_cache device")
    if values.block_table.numel():
        if int(values.block_table.min().item()) < 0:
            raise ValueError("block_table entries must be non-negative")
        if int(values.block_table.max().item()) >= values.state_cache.shape[0]:
            raise ValueError("block_table entries must be valid state cache pages")
    if values.block_table_base_offsets is not None:
        if values.block_table_base_offsets.ndim != 1:
            raise ValueError("block_table_base_offsets must be rank-1")
        if values.block_table_base_offsets.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_table_base_offsets must be integer")
        if values.block_table_base_offsets.numel() != values.block_table.shape[0]:
            raise ValueError("block_table_base_offsets length must match batch size")
        if values.block_table_base_offsets.device != values.state_cache.device:
            raise ValueError(
                "block_table_base_offsets must share the state_cache device"
            )
    if values.rms_norm_weight.shape != (_CSA_INDEXER_DIM,):
        raise ValueError(
            f"rms_norm_weight must have shape {(_CSA_INDEXER_DIM,)}, "
            f"got {tuple(values.rms_norm_weight.shape)}"
        )
    if not values.rms_norm_weight.is_floating_point():
        raise TypeError("rms_norm_weight must be floating point")
    if values.rms_norm_weight.device != values.state_cache.device:
        raise ValueError("rms_norm_weight must share the state_cache device")
    if values.rms_norm_eps <= 0.0 or not math.isfinite(values.rms_norm_eps):
        raise ValueError("rms_norm_eps must be finite and positive")
    if values.cos_sin_cache.shape[-1] != _CSA_ROPE_DIM:
        raise ValueError(
            f"cos_sin_cache width must be {_CSA_ROPE_DIM}, "
            f"got {values.cos_sin_cache.shape[-1]}"
        )
    if values.cos_sin_cache.device != values.state_cache.device:
        raise ValueError("cos_sin_cache must share the state_cache device")
    if num_actual:
        reqs = values.token_to_req_indices[:num_actual].to(torch.int64)
        if (
            int(reqs.min().item()) < 0
            or int(reqs.max().item()) >= values.block_table.shape[0]
        ):
            raise ValueError("token_to_req_indices entries must be valid request rows")
        positions = values.positions[:num_actual].to(torch.int64)
        if int(positions.min().item()) < 0:
            raise ValueError("positions must be non-negative")
        if int(positions.max().item()) >= values.cos_sin_cache.shape[0]:
            raise ValueError("positions must fit in cos_sin_cache")
        table_idx = positions // compressor_block_size
        if int(table_idx.max().item()) >= values.block_table.shape[1]:
            raise ValueError("block_table must cover generated positions")
        state_slots = values.compressor_slot_mapping[:num_actual].to(torch.int64)
        valid_state = state_slots >= 0
        if bool(valid_state.any().item()):
            if int(state_slots[valid_state].max().item()) >= (
                values.state_cache.shape[0] * compressor_block_size
            ):
                raise ValueError("compressor_slot_mapping entries exceed state cache")
        kv_slots = values.kv_slot_mapping[:num_actual].to(torch.int64)
        writable = (
            valid_state
            & (kv_slots >= 0)
            & (torch.remainder(positions + 1, values.compress_ratio) == 0)
        )
        if bool(writable.any().item()):
            write_slots = kv_slots[writable]
            if int(write_slots.max().item()) >= (
                values.kv_cache_2d.shape[0] * kv_cache_block_size
            ):
                raise ValueError("kv_slot_mapping entries exceed kv cache")
            if int(torch.unique(write_slots).numel()) != int(write_slots.numel()):
                raise ValueError("writable kv_slot_mapping entries must be unique")


def _deepseek_v4_csa_indexer_compress_rows(
    values: DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_actual = min(
        values.compressor_slot_mapping.numel(),
        values.positions.numel(),
        values.kv_slot_mapping.numel(),
    )
    if num_actual == 0:
        return (
            torch.empty(
                (0, _CSA_INDEXER_DIM),
                dtype=torch.float32,
                device=values.state_cache.device,
            ),
            torch.empty((0,), dtype=torch.bool, device=values.state_cache.device),
        )

    positions = values.positions[:num_actual].to(torch.int64)
    state_slots = values.compressor_slot_mapping[:num_actual].to(torch.int64)
    kv_slots = values.kv_slot_mapping[:num_actual].to(torch.int64)
    valid_token = (
        (state_slots >= 0)
        & (kv_slots >= 0)
        & (torch.remainder(positions + 1, values.compress_ratio) == 0)
    )
    window = values.compress_ratio * 2
    offsets = torch.arange(window, dtype=torch.int64, device=values.state_cache.device)
    window_positions = positions[:, None] - window + 1 + offsets[None, :]
    table_idx_raw = torch.div(
        window_positions,
        values.compressor_block_size,
        rounding_mode="floor",
    )
    reqs = values.token_to_req_indices[:num_actual].to(torch.int64)
    if values.block_table_base_offsets is not None:
        table_idx_raw = (
            table_idx_raw
            - values.block_table_base_offsets.to(torch.int64)[reqs][:, None]
        )
    valid_window = (
        (window_positions >= 0)
        & (table_idx_raw >= 0)
        & (table_idx_raw < values.block_table.shape[1])
    )
    table_idx = table_idx_raw.clamp(0, max(values.block_table.shape[1] - 1, 0))
    block_numbers = values.block_table[reqs[:, None], table_idx].to(torch.int64)
    valid_window = valid_window & (block_numbers >= 0)
    safe_block = block_numbers.clamp_min(0)
    pos_in_block = torch.remainder(
        window_positions.clamp_min(0),
        values.compressor_block_size,
    )
    rows = values.state_cache[safe_block, pos_in_block]
    state_width = values.state_cache.shape[-1] // 2
    head_offsets = torch.where(
        offsets >= values.compress_ratio,
        torch.full_like(offsets, _CSA_INDEXER_DIM),
        torch.zeros_like(offsets),
    )
    dim_indices = (
        head_offsets[:, None]
        + torch.arange(
            _CSA_INDEXER_DIM,
            dtype=torch.int64,
            device=values.state_cache.device,
        )[None, :]
    )
    dim_indices = dim_indices[None, :, :].expand(num_actual, -1, -1)
    kv_rows = torch.gather(rows[..., :state_width], -1, dim_indices).float()
    score_rows = torch.gather(rows[..., state_width:], -1, dim_indices).float()
    valid_window_f = valid_window.unsqueeze(-1)
    score_rows = torch.where(
        valid_window_f,
        score_rows,
        score_rows.new_full((), -1.0e30),
    )
    weights = torch.softmax(score_rows, dim=1)
    kv_rows = torch.where(valid_window_f, kv_rows, torch.zeros_like(kv_rows))
    compressed = torch.sum(kv_rows * weights, dim=1)
    variance = compressed.square().sum(dim=-1, keepdim=True) / float(
        _CSA_INDEXER_DIM
    )
    normed = compressed * torch.rsqrt(variance + values.rms_norm_eps)
    return normed * values.rms_norm_weight.float(), valid_token


def deepseek_v4_csa_indexer_mxfp4_cache_insert_reference(
    values: DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues,
) -> torch.Tensor:
    """Return cache bytes after applying CSA indexer MXFP4 inserts."""

    _validate_deepseek_v4_csa_indexer_mxfp4_cache_insert_values(values)
    out = values.kv_cache_2d.clone()
    normed, valid = _deepseek_v4_csa_indexer_compress_rows(values)
    if normed.numel() == 0:
        return out.contiguous()

    compressed_positions = (
        torch.div(
            values.positions[: normed.shape[0]].to(torch.int64),
            values.compress_ratio,
            rounding_mode="floor",
        )
        * values.compress_ratio
    )
    rotated = _deepseek_v4_apply_indexer_rope_rows(
        normed,
        compressed_positions,
        values.cos_sin_cache,
    )
    hadamard = _deepseek_v4_indexer_q_hadamard(rotated)
    flat = out.reshape(-1)
    slots = values.kv_slot_mapping[: normed.shape[0]].to(torch.int64)
    for row_idx, row in enumerate(hadamard):
        if not bool(valid[row_idx].item()):
            continue
        slot = int(slots[row_idx].item())
        page = slot // values.kv_cache_block_size
        pos = slot % values.kv_cache_block_size
        page_base = page * out.stride(0)
        value_base = page_base + pos * _CSA_INDEXER_MXFP4_VALUE_BYTES
        scale_base = (
            page_base
            + values.kv_cache_block_size * _CSA_INDEXER_MXFP4_VALUE_BYTES
            + pos * _CSA_INDEXER_MXFP4_SCALE_BYTES
        )
        packed, scales = _deepseek_v4_indexer_mxfp4_row_reference(row)
        flat[value_base : value_base + _CSA_INDEXER_MXFP4_VALUE_BYTES] = packed
        flat[scale_base : scale_base + _CSA_INDEXER_MXFP4_SCALE_BYTES] = scales
    return out.contiguous()


@dataclass
class DeepSeekV4SparseCompressCacheInsertInputValues:
    """Generated values for CSA sparse-compress K-cache inserts."""

    state_cache: torch.Tensor
    token_to_req_indices: torch.Tensor
    positions: torch.Tensor
    compressor_slot_mapping: torch.Tensor
    block_table: torch.Tensor
    compressor_block_size: int
    rms_norm_weight: torch.Tensor
    rms_norm_eps: float
    cos_sin_cache: torch.Tensor
    kv_cache_2d: torch.Tensor
    kv_slot_mapping: torch.Tensor
    kv_cache_block_size: int
    compress_ratio: int
    overlap: bool
    block_table_base_offsets: torch.Tensor | None


@dataclass
class _DeepSeekV4SparseCompressCacheInsertConfig:
    """Initialization parameters for CSA sparse-compress cache inserts.

    The represented operation compresses state-cache windows into one 512-wide
    K row, normalizes the row, stores the NoPE prefix as block-scaled FP8, and
    stores the RoPE suffix as BF16 in a paged sparse-window attention cache.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of candidate token rows.
    num_tokens: int

    # Required: number of request rows in the generated block table.
    batch_size: int

    # Required: maximum generated sequence position plus one.
    max_seq_len: int

    # Required: number of physical pages in the compressor state cache.
    num_state_cache_blocks: int

    # Required: number of token rows in each compressor state cache page.
    compressor_block_size: int

    # Required: number of physical pages in the output K cache.
    num_kv_cache_blocks: int

    # Required: number of K rows in each output cache page.
    kv_cache_block_size: int

    # Required: compression interval. Runtime HCA uses 128; CSA uses 4.
    compress_ratio: int

    # Required: whether the state window uses the overlapping CSA layout.
    overlap: bool

    # Required: generated dtype for compressor state and RMSNorm weight values.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: generated rows whose position is not a compression boundary.
    non_boundary_token_count: int = 0

    # Optional: rows with compressor_slot_mapping == -1. These rows are skipped.
    negative_compressor_slot_count: int = 0

    # Optional: rows with kv_slot_mapping == -1. These rows are skipped.
    negative_kv_slot_count: int = 0

    # Optional: include zero block-table base offsets in generated values.
    include_block_table_base_offsets: bool = False

    # Optional: RMSNorm epsilon.
    rms_norm_eps: float = 1.0e-5

    # Optional: RoPE frequency base used to build the generated cos/sin cache.
    rope_base: float = 10000.0

    # Optional: scale applied to generated state-cache values.
    value_scale: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DeepSeekV4SparseCompressCacheInsertBuilder:
    """Generator for CSA sparse-compress K-cache insert inputs."""

    config: _DeepSeekV4SparseCompressCacheInsertConfig
    state_cache_input: TensorInput | None
    rms_norm_weight_input: TensorInput | None

    def __init__(
        self,
        config: _DeepSeekV4SparseCompressCacheInsertConfig,
    ) -> None:
        self.config = config
        self.state_cache_input = None
        self.rms_norm_weight_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.state_cache_input = self.state_cache_input or TensorInput(
            self._state_cache_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.rms_norm_weight_input = self.rms_norm_weight_input or TensorInput(
            (_CSA_HEAD_DIM,),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4SparseCompressCacheInsertInputValues:
        self.__post_init__()
        if self.state_cache_input is None or self.rms_norm_weight_input is None:
            raise ValueError("sparse-compress child generators must be initialized")
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        self.state_cache_input.shape = self._state_cache_shape()
        self.state_cache_input.dtype = self.config.dtype
        self.rms_norm_weight_input.shape = (_CSA_HEAD_DIM,)
        self.rms_norm_weight_input.dtype = self.config.dtype
        state_cache = _require_tensor(
            self.state_cache_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "state_cache",
        )
        rms_norm_weight = _require_tensor(
            self.rms_norm_weight_input.generate(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ).values,
            "rms_norm_weight",
        )
        values = DeepSeekV4SparseCompressCacheInsertInputValues(
            state_cache=(state_cache.float() * self.config.value_scale)
            .to(state_cache.dtype)
            .contiguous(),
            token_to_req_indices=self._generate_token_to_req_indices(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            positions=self._generate_positions(
                seed=_child_seed(metadata_seed, 2),
                device=target_device,
            ),
            compressor_slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 3),
                total_slots=self._total_state_slots(),
                negative_count=self.config.negative_compressor_slot_count,
                device=target_device,
            ),
            block_table=self._generate_block_table(
                seed=_child_seed(metadata_seed, 4),
                device=target_device,
            ),
            compressor_block_size=self.config.compressor_block_size,
            rms_norm_weight=rms_norm_weight.contiguous(),
            rms_norm_eps=self.config.rms_norm_eps,
            cos_sin_cache=build_rope_cos_sin_cache(
                rotary_dim=_CSA_ROPE_DIM,
                max_position=self.config.max_seq_len,
                base=self.config.rope_base,
                device=target_device,
            ),
            kv_cache_2d=self._generate_kv_cache(
                seed=_child_seed(value_seed, 3),
                device=target_device,
            ),
            kv_slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 5),
                total_slots=self._total_kv_slots(),
                negative_count=self.config.negative_kv_slot_count,
                device=target_device,
            ),
            kv_cache_block_size=self.config.kv_cache_block_size,
            compress_ratio=self.config.compress_ratio,
            overlap=self.config.overlap,
            block_table_base_offsets=(
                torch.zeros(
                    (self.config.batch_size,),
                    dtype=torch.int32,
                    device=target_device,
                )
                if self.config.include_block_table_base_offsets
                else None
            ),
        )
        _validate_deepseek_v4_sparse_compress_cache_insert_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens",
            self.config.num_tokens,
        )
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.max_seq_len = _check_positive(
            "max_seq_len",
            self.config.max_seq_len,
        )
        self.config.num_state_cache_blocks = _check_positive(
            "num_state_cache_blocks",
            self.config.num_state_cache_blocks,
        )
        self.config.compressor_block_size = _check_positive(
            "compressor_block_size",
            self.config.compressor_block_size,
        )
        self.config.num_kv_cache_blocks = _check_positive(
            "num_kv_cache_blocks",
            self.config.num_kv_cache_blocks,
        )
        self.config.kv_cache_block_size = _check_positive(
            "kv_cache_block_size",
            self.config.kv_cache_block_size,
        )
        self.config.compress_ratio = _check_positive(
            "compress_ratio",
            self.config.compress_ratio,
        )
        if not isinstance(self.config.overlap, bool):
            raise TypeError("overlap must be a bool")
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.non_boundary_token_count = _check_nonnegative(
            "non_boundary_token_count",
            self.config.non_boundary_token_count,
        )
        self.config.negative_compressor_slot_count = _check_nonnegative(
            "negative_compressor_slot_count",
            self.config.negative_compressor_slot_count,
        )
        self.config.negative_kv_slot_count = _check_nonnegative(
            "negative_kv_slot_count",
            self.config.negative_kv_slot_count,
        )
        for name, value in (
            ("non_boundary_token_count", self.config.non_boundary_token_count),
            (
                "negative_compressor_slot_count",
                self.config.negative_compressor_slot_count,
            ),
            ("negative_kv_slot_count", self.config.negative_kv_slot_count),
        ):
            if value > self.config.num_tokens:
                raise ValueError(f"{name} must be <= num_tokens")
        if self.config.num_tokens - self.config.negative_compressor_slot_count > (
            self._total_state_slots()
        ):
            raise ValueError("generated compressor slots must fit in state_cache")
        if self.config.num_tokens - self.config.negative_kv_slot_count > (
            self._total_kv_slots()
        ):
            raise ValueError("generated kv slots must fit in kv_cache_2d")
        if not self._boundary_positions().numel() and self.config.num_tokens:
            raise ValueError(
                "max_seq_len must contain at least one compression boundary"
            )
        if (
            self.config.non_boundary_token_count
            and not self._non_boundary_positions().numel()
        ):
            raise ValueError(
                "max_seq_len must contain non-boundary positions when "
                "non_boundary_token_count is non-zero"
            )
        self.config.rms_norm_eps = float(self.config.rms_norm_eps)
        if self.config.rms_norm_eps <= 0.0 or not math.isfinite(
            self.config.rms_norm_eps
        ):
            raise ValueError(
                f"rms_norm_eps must be finite and positive, got {self.config.rms_norm_eps}"
            )
        self.config.rope_base = float(self.config.rope_base)
        if self.config.rope_base <= 0.0 or not math.isfinite(self.config.rope_base):
            raise ValueError(f"rope_base must be positive, got {self.config.rope_base}")
        self.config.value_scale = float(self.config.value_scale)
        if self.config.value_scale < 0.0 or not math.isfinite(self.config.value_scale):
            raise ValueError(
                f"value_scale must be finite and non-negative, got {self.config.value_scale}"
            )

    def _state_width(self) -> int:
        return _CSA_HEAD_DIM * (2 if self.config.overlap else 1)

    def _state_cache_shape(self) -> tuple[int, int, int]:
        return (
            self.config.num_state_cache_blocks,
            self.config.compressor_block_size,
            self._state_width() * 2,
        )

    def _total_state_slots(self) -> int:
        return self.config.num_state_cache_blocks * self.config.compressor_block_size

    def _total_kv_slots(self) -> int:
        return self.config.num_kv_cache_blocks * self.config.kv_cache_block_size

    def _block_table_width(self) -> int:
        return math.ceil(self.config.max_seq_len / self.config.compressor_block_size)

    def _boundary_positions(self) -> torch.Tensor:
        return _compression_boundary_positions(
            max_seq_len=self.config.max_seq_len,
            compress_ratio=self.config.compress_ratio,
        )

    def _non_boundary_positions(self) -> torch.Tensor:
        return _compression_non_boundary_positions(
            max_seq_len=self.config.max_seq_len,
            compress_ratio=self.config.compress_ratio,
        )

    def _generate_token_to_req_indices(
        self, *, seed: int, device: torch.device
    ) -> torch.Tensor:
        return _generate_token_to_request_indices(
            num_tokens=self.config.num_tokens,
            batch_size=self.config.batch_size,
            seed=seed,
            device=device,
        )

    def _generate_positions(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_compression_positions(
            num_tokens=self.config.num_tokens,
            max_seq_len=self.config.max_seq_len,
            compress_ratio=self.config.compress_ratio,
            non_boundary_token_count=self.config.non_boundary_token_count,
            seed=seed,
            device=device,
        )

    def _generate_slot_mapping(
        self,
        *,
        seed: int,
        total_slots: int,
        negative_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        return _generate_slot_mapping_tensor(
            num_rows=self.config.num_tokens,
            total_slots=total_slots,
            negative_count=negative_count,
            seed=seed,
            device=device,
        )

    def _generate_block_table(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        table = torch.randint(
            0,
            self.config.num_state_cache_blocks,
            (self.config.batch_size, self._block_table_width()),
            dtype=torch.int64,
            generator=rng,
        )
        return table.to(device)

    def _generate_kv_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_random_uint8_tensor(
            shape=(
                self.config.num_kv_cache_blocks,
                self.config.kv_cache_block_size
                * (_CSA_SWA_TOKEN_STRIDE + _CSA_SWA_SCALE_DIM),
            ),
            seed=seed,
            device=device,
        )


def _validate_deepseek_v4_sparse_compress_cache_insert_values(
    values: DeepSeekV4SparseCompressCacheInsertInputValues,
) -> None:
    if values.state_cache.ndim != 3:
        raise ValueError("state_cache must be rank-3")
    if not values.state_cache.is_floating_point():
        raise TypeError(
            f"state_cache must be floating point, got {values.state_cache.dtype}"
        )
    expected_state_width = _CSA_HEAD_DIM * (2 if values.overlap else 1)
    if values.state_cache.shape[-1] != expected_state_width * 2:
        raise ValueError(
            "state_cache last dimension must be "
            f"{expected_state_width * 2}, got {values.state_cache.shape[-1]}"
        )
    compressor_block_size = _check_positive(
        "compressor_block_size",
        values.compressor_block_size,
    )
    if values.state_cache.shape[1] != compressor_block_size:
        raise ValueError("compressor_block_size must match state_cache.shape[1]")
    compress_ratio = _check_positive("compress_ratio", values.compress_ratio)
    if not isinstance(values.overlap, bool):
        raise TypeError("overlap must be a bool")
    if values.kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {values.kv_cache_2d.dtype}")
    if values.kv_cache_2d.ndim != 2:
        raise ValueError("kv_cache_2d must be rank-2")
    kv_cache_block_size = _check_positive(
        "kv_cache_block_size",
        values.kv_cache_block_size,
    )
    min_row_bytes = kv_cache_block_size * (
        _CSA_SWA_TOKEN_STRIDE + _CSA_SWA_SCALE_DIM
    )
    if values.kv_cache_2d.shape[1] < min_row_bytes:
        raise ValueError(
            f"kv_cache_2d row width must be at least {min_row_bytes}, "
            f"got {values.kv_cache_2d.shape[1]}"
        )
    for name, tensor in (
        ("token_to_req_indices", values.token_to_req_indices),
        ("positions", values.positions),
        ("compressor_slot_mapping", values.compressor_slot_mapping),
        ("kv_slot_mapping", values.kv_slot_mapping),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
        if tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be integer, got {tensor.dtype}")
        if tensor.device != values.state_cache.device:
            raise ValueError(f"{name} must share the state_cache device")
    num_actual = min(
        values.compressor_slot_mapping.numel(),
        values.positions.numel(),
        values.kv_slot_mapping.numel(),
    )
    if values.token_to_req_indices.numel() < num_actual:
        raise ValueError("token_to_req_indices length must cover generated rows")
    if values.block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    if values.block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"block_table must be integer, got {values.block_table.dtype}")
    if values.block_table.device != values.state_cache.device:
        raise ValueError("block_table must share the state_cache device")
    if values.block_table.numel():
        if int(values.block_table.min().item()) < 0:
            raise ValueError("block_table entries must be non-negative")
        if int(values.block_table.max().item()) >= values.state_cache.shape[0]:
            raise ValueError("block_table entries must be valid state cache pages")
    if values.block_table_base_offsets is not None:
        if values.block_table_base_offsets.ndim != 1:
            raise ValueError("block_table_base_offsets must be rank-1")
        if values.block_table_base_offsets.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_table_base_offsets must be integer")
        if values.block_table_base_offsets.numel() != values.block_table.shape[0]:
            raise ValueError("block_table_base_offsets length must match batch size")
        if values.block_table_base_offsets.device != values.state_cache.device:
            raise ValueError(
                "block_table_base_offsets must share the state_cache device"
            )
    if values.rms_norm_weight.shape != (_CSA_HEAD_DIM,):
        raise ValueError(
            f"rms_norm_weight must have shape {(_CSA_HEAD_DIM,)}, "
            f"got {tuple(values.rms_norm_weight.shape)}"
        )
    if not values.rms_norm_weight.is_floating_point():
        raise TypeError("rms_norm_weight must be floating point")
    if values.rms_norm_weight.device != values.state_cache.device:
        raise ValueError("rms_norm_weight must share the state_cache device")
    if values.rms_norm_eps <= 0.0 or not math.isfinite(values.rms_norm_eps):
        raise ValueError("rms_norm_eps must be finite and positive")
    if values.cos_sin_cache.shape[-1] != _CSA_ROPE_DIM:
        raise ValueError(
            f"cos_sin_cache width must be {_CSA_ROPE_DIM}, "
            f"got {values.cos_sin_cache.shape[-1]}"
        )
    if values.cos_sin_cache.device != values.state_cache.device:
        raise ValueError("cos_sin_cache must share the state_cache device")
    if num_actual:
        reqs = values.token_to_req_indices[:num_actual].to(torch.int64)
        if (
            int(reqs.min().item()) < 0
            or int(reqs.max().item()) >= values.block_table.shape[0]
        ):
            raise ValueError("token_to_req_indices entries must be valid request rows")
        positions = values.positions[:num_actual].to(torch.int64)
        if int(positions.min().item()) < 0:
            raise ValueError("positions must be non-negative")
        if int(positions.max().item()) >= values.cos_sin_cache.shape[0]:
            raise ValueError("positions must fit in cos_sin_cache")
        table_idx = positions // compressor_block_size
        if int(table_idx.max().item()) >= values.block_table.shape[1]:
            raise ValueError("block_table must cover generated positions")
        state_slots = values.compressor_slot_mapping[:num_actual].to(torch.int64)
        valid_state = state_slots >= 0
        if bool(valid_state.any().item()):
            if int(state_slots[valid_state].max().item()) >= (
                values.state_cache.shape[0] * compressor_block_size
            ):
                raise ValueError("compressor_slot_mapping entries exceed state cache")
        kv_slots = values.kv_slot_mapping[:num_actual].to(torch.int64)
        writable = (
            valid_state
            & (kv_slots >= 0)
            & (torch.remainder(positions + 1, compress_ratio) == 0)
        )
        if bool(writable.any().item()):
            write_slots = kv_slots[writable]
            if int(write_slots.max().item()) >= (
                values.kv_cache_2d.shape[0] * kv_cache_block_size
            ):
                raise ValueError("kv_slot_mapping entries exceed kv cache")
            if int(torch.unique(write_slots).numel()) != int(write_slots.numel()):
                raise ValueError("writable kv_slot_mapping entries must be unique")


def _deepseek_v4_sparse_compress_rows(
    values: DeepSeekV4SparseCompressCacheInsertInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_actual = min(
        values.compressor_slot_mapping.numel(),
        values.positions.numel(),
        values.kv_slot_mapping.numel(),
    )
    if num_actual == 0:
        return (
            torch.empty(
                (0, _CSA_HEAD_DIM),
                dtype=torch.float32,
                device=values.state_cache.device,
            ),
            torch.empty((0,), dtype=torch.bool, device=values.state_cache.device),
        )

    positions = values.positions[:num_actual].to(torch.int64)
    state_slots = values.compressor_slot_mapping[:num_actual].to(torch.int64)
    kv_slots = values.kv_slot_mapping[:num_actual].to(torch.int64)
    valid_token = (
        (state_slots >= 0)
        & (kv_slots >= 0)
        & (torch.remainder(positions + 1, values.compress_ratio) == 0)
    )
    window = values.compress_ratio * (2 if values.overlap else 1)
    offsets = torch.arange(window, dtype=torch.int64, device=values.state_cache.device)
    window_positions = positions[:, None] - window + 1 + offsets[None, :]
    table_idx_raw = torch.div(
        window_positions,
        values.compressor_block_size,
        rounding_mode="floor",
    )
    reqs = values.token_to_req_indices[:num_actual].to(torch.int64)
    if values.block_table_base_offsets is not None:
        table_idx_raw = (
            table_idx_raw
            - values.block_table_base_offsets.to(torch.int64)[reqs][:, None]
        )
    valid_window = (
        (window_positions >= 0)
        & (table_idx_raw >= 0)
        & (table_idx_raw < values.block_table.shape[1])
    )
    table_idx = table_idx_raw.clamp(0, max(values.block_table.shape[1] - 1, 0))
    block_numbers = values.block_table[reqs[:, None], table_idx].to(torch.int64)
    valid_window = valid_window & (block_numbers >= 0)
    safe_block = block_numbers.clamp_min(0)
    pos_in_block = torch.remainder(
        window_positions.clamp_min(0),
        values.compressor_block_size,
    )
    rows = values.state_cache[safe_block, pos_in_block]
    state_width = values.state_cache.shape[-1] // 2
    head_offsets = (
        torch.where(
            offsets >= values.compress_ratio,
            torch.full_like(offsets, _CSA_HEAD_DIM),
            torch.zeros_like(offsets),
        )
        if values.overlap
        else torch.zeros_like(offsets)
    )
    dim_indices = (
        head_offsets[:, None]
        + torch.arange(
            _CSA_HEAD_DIM,
            dtype=torch.int64,
            device=values.state_cache.device,
        )[None, :]
    )
    dim_indices = dim_indices[None, :, :].expand(num_actual, -1, -1)
    kv_rows = torch.gather(rows[..., :state_width], -1, dim_indices).float()
    score_rows = torch.gather(rows[..., state_width:], -1, dim_indices).float()
    valid_window_f = valid_window.unsqueeze(-1)
    score_rows = torch.where(
        valid_window_f,
        score_rows,
        score_rows.new_full((), -1.0e30),
    )
    weights = torch.softmax(score_rows, dim=1)
    kv_rows = torch.where(valid_window_f, kv_rows, torch.zeros_like(kv_rows))
    compressed = torch.sum(kv_rows * weights, dim=1)
    variance = compressed.square().sum(dim=-1, keepdim=True) / float(
        _CSA_HEAD_DIM
    )
    normed = compressed * torch.rsqrt(variance + values.rms_norm_eps)
    return normed * values.rms_norm_weight.float(), valid_token


def _deepseek_v4_apply_k_rope_rows(
    rows: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    rows = rows.float().clone()
    rope = rows[..., _CSA_NOPE_DIM:]
    even = rope[..., 0::2].clone()
    odd = rope[..., 1::2].clone()
    half_rope = _CSA_ROPE_DIM // 2
    cos_sin = cos_sin_cache[positions.to(torch.int64)]
    cos = cos_sin[..., :half_rope].float()
    sin = cos_sin[..., half_rope:].float()
    rope[..., 0::2] = even * cos - odd * sin
    rope[..., 1::2] = even * sin + odd * cos
    return rows


def _deepseek_v4_sparse_cache_row_bytes(
    normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quant_input = normed.to(torch.bfloat16).to(torch.float32)
    quant_blocks = quant_input.reshape(
        quant_input.shape[0],
        _CSA_HEAD_DIM // _CSA_FP8_QUANT_BLOCK,
        _CSA_FP8_QUANT_BLOCK,
    )
    absmax = quant_blocks.abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / _CSA_FP8_MAX))
    scaled = torch.clamp(
        quant_blocks * torch.exp2(-exponent).unsqueeze(-1),
        -_CSA_FP8_MAX,
        _CSA_FP8_MAX,
    )
    value_bytes = (
        scaled.to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(
            quant_input.shape[0],
            _CSA_HEAD_DIM,
        )
    )
    scale_bytes = torch.clamp(exponent + 127.0, 0.0, 255.0).to(torch.uint8)
    scale_bytes[..., -1] = 0
    rotated = _deepseek_v4_apply_k_rope_rows(normed, positions, cos_sin_cache).to(
        torch.bfloat16
    )
    rope_bytes = rotated[..., _CSA_NOPE_DIM:].contiguous().view(torch.uint8)
    rope_bytes = rope_bytes.reshape(normed.shape[0], _CSA_ROPE_DIM * 2)
    return value_bytes[..., :_CSA_NOPE_DIM], scale_bytes, rope_bytes


def deepseek_v4_sparse_compress_cache_insert_reference(
    values: DeepSeekV4SparseCompressCacheInsertInputValues,
) -> torch.Tensor:
    """Return cache bytes after applying CSA sparse-compress inserts."""

    _validate_deepseek_v4_sparse_compress_cache_insert_values(values)
    out = values.kv_cache_2d.clone()
    normed, valid = _deepseek_v4_sparse_compress_rows(values)
    if normed.numel() == 0:
        return out.contiguous()
    compressed_positions = (
        torch.div(
            values.positions[: normed.shape[0]].to(torch.int64),
            values.compress_ratio,
            rounding_mode="floor",
        )
        * values.compress_ratio
    )
    value_bytes, scale_bytes, rope_bytes = _deepseek_v4_sparse_cache_row_bytes(
        normed,
        compressed_positions,
        values.cos_sin_cache,
    )
    flat = out.reshape(-1)
    slots = values.kv_slot_mapping[: normed.shape[0]].to(torch.int64)
    for row_idx in range(normed.shape[0]):
        if not bool(valid[row_idx].item()):
            continue
        slot = int(slots[row_idx].item())
        page = slot // values.kv_cache_block_size
        pos = slot % values.kv_cache_block_size
        page_base = page * out.stride(0)
        token_base = page_base + pos * _CSA_SWA_TOKEN_STRIDE
        scale_base = (
            page_base
            + values.kv_cache_block_size * _CSA_SWA_TOKEN_STRIDE
            + pos * _CSA_SWA_SCALE_DIM
        )
        flat[token_base : token_base + _CSA_NOPE_DIM] = value_bytes[row_idx]
        rope_base = token_base + _CSA_NOPE_DIM
        flat[rope_base : rope_base + _CSA_ROPE_DIM * 2] = rope_bytes[row_idx]
        flat[scale_base : scale_base + _CSA_SWA_SCALE_DIM] = scale_bytes[
            row_idx
        ]
    return out.contiguous()


@dataclass
class DeepSeekV4IndexerMXFP4CacheWriteInputValues:
    """Generated values for writing CSA indexer K rows to MXFP4 cache."""

    index_k: torch.Tensor
    cache_2d: torch.Tensor
    slot_mapping: torch.Tensor
    valid: torch.Tensor
    block_size: int


@dataclass
class _DeepSeekV4IndexerMXFP4CacheWriteConfig:
    """Initialization parameters for CSA indexer MXFP4 cache writes.

    The represented operation quantizes 128-channel indexer K rows into MXFP4
    storage and writes them into a paged byte cache. Cache rows are selected by
    ``slot_mapping``. A row is written only when ``valid`` is true and its slot
    is non-negative.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of generated indexer K rows.
    num_rows: int

    # Required: number of physical MXFP4 cache pages.
    num_cache_blocks: int

    # Required: number of indexer rows in each physical cache page.
    block_size: int

    # Required: generated dtype for indexer K rows.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: rows with slot_mapping == -1. These rows are skipped.
    negative_slot_count: int = 0

    # Optional: rows with valid == False. These rows are skipped even when they
    # have a non-negative slot.
    masked_row_count: int = 0

    # Optional: factor applied to generated indexer K rows to keep MXFP4 scales
    # in a representative range.
    value_scale: float = 1.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DeepSeekV4IndexerMXFP4CacheWriteBuilder:
    """Generator for CSA indexer MXFP4 cache-write inputs."""

    config: _DeepSeekV4IndexerMXFP4CacheWriteConfig
    index_k_input: TensorInput | None

    def __init__(self, config: _DeepSeekV4IndexerMXFP4CacheWriteConfig) -> None:
        self.config = config
        self.index_k_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.index_k_input = self.index_k_input or TensorInput(
            self._index_k_shape(),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4IndexerMXFP4CacheWriteInputValues:
        self.__post_init__()
        if self.index_k_input is None:
            raise ValueError("index_k_input must be initialized")
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        self.index_k_input.shape = self._index_k_shape()
        self.index_k_input.dtype = self.config.dtype
        index_k = _require_tensor(
            self.index_k_input.generate(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ).values,
            "index_k",
        )
        values = DeepSeekV4IndexerMXFP4CacheWriteInputValues(
            index_k=(index_k.float() * self.config.value_scale)
            .to(index_k.dtype)
            .contiguous(),
            cache_2d=self._generate_cache(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ),
            slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            valid=self._generate_valid(
                seed=_child_seed(metadata_seed, 2),
                device=target_device,
            ),
            block_size=self.config.block_size,
        )
        _validate_deepseek_v4_indexer_mxfp4_cache_write_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.num_cache_blocks = _check_positive(
            "num_cache_blocks",
            self.config.num_cache_blocks,
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.negative_slot_count = _check_nonnegative(
            "negative_slot_count",
            self.config.negative_slot_count,
        )
        self.config.masked_row_count = _check_nonnegative(
            "masked_row_count",
            self.config.masked_row_count,
        )
        if self.config.negative_slot_count > self.config.num_rows:
            raise ValueError("negative_slot_count must be <= num_rows")
        if self.config.masked_row_count > self.config.num_rows:
            raise ValueError("masked_row_count must be <= num_rows")
        non_negative_count = self.config.num_rows - self.config.negative_slot_count
        total_slots = self.config.num_cache_blocks * self.config.block_size
        if non_negative_count > total_slots:
            raise ValueError(
                "generated non-negative slots must fit in the MXFP4 cache, got "
                f"non_negative_count={non_negative_count}, total_slots={total_slots}"
            )
        self.config.value_scale = float(self.config.value_scale)
        if self.config.value_scale < 0.0:
            raise ValueError(
                f"value_scale must be non-negative, got {self.config.value_scale}"
            )

    def _index_k_shape(self) -> tuple[int, int]:
        return (self.config.num_rows, _CSA_INDEXER_DIM)

    def _cache_row_bytes(self) -> int:
        return self.config.block_size * (
            _CSA_INDEXER_MXFP4_VALUE_BYTES
            + _CSA_INDEXER_MXFP4_SCALE_BYTES
        )

    def _generate_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_random_uint8_tensor(
            shape=(self.config.num_cache_blocks, self._cache_row_bytes()),
            seed=seed,
            device=device,
        )

    def _generate_slot_mapping(
        self, *, seed: int, device: torch.device
    ) -> torch.Tensor:
        return _generate_slot_mapping_tensor(
            num_rows=self.config.num_rows,
            total_slots=self.config.num_cache_blocks * self.config.block_size,
            negative_count=self.config.negative_slot_count,
            seed=seed,
            device=device,
        )

    def _generate_valid(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_valid_mask(
            num_rows=self.config.num_rows,
            false_count=self.config.masked_row_count,
            seed=seed,
            device=device,
        )


def _deepseek_v4_mxfp4_nibble_reference(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.minimum(x.abs(), torch.tensor(6.0, device=x.device))
    code = torch.where(
        abs_x <= 0.25,
        torch.zeros_like(abs_x, dtype=torch.uint8),
        torch.where(
            abs_x <= 0.75,
            torch.ones_like(abs_x, dtype=torch.uint8),
            torch.where(
                abs_x <= 1.25,
                torch.full_like(abs_x, 2, dtype=torch.uint8),
                torch.where(
                    abs_x <= 1.75,
                    torch.full_like(abs_x, 3, dtype=torch.uint8),
                    torch.where(
                        abs_x <= 2.5,
                        torch.full_like(abs_x, 4, dtype=torch.uint8),
                        torch.where(
                            abs_x <= 3.5,
                            torch.full_like(abs_x, 5, dtype=torch.uint8),
                            torch.where(
                                abs_x <= 5.0,
                                torch.full_like(abs_x, 6, dtype=torch.uint8),
                                torch.full_like(abs_x, 7, dtype=torch.uint8),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    sign = ((x < 0) & (code != 0)).to(torch.uint8)
    return code | (sign << 3)


def _deepseek_v4_indexer_mxfp4_row_reference(
    row: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    row = row.float()
    packed = torch.empty(
        (_CSA_INDEXER_MXFP4_VALUE_BYTES,),
        dtype=torch.uint8,
        device=row.device,
    )
    scales = torch.empty(
        (_CSA_INDEXER_MXFP4_SCALE_BYTES,),
        dtype=torch.uint8,
        device=row.device,
    )
    for block_idx in range(_CSA_INDEXER_MXFP4_SCALE_BYTES):
        block_base = block_idx * _CSA_INDEXER_MXFP4_BLOCK_SIZE
        block = row[block_base : block_base + _CSA_INDEXER_MXFP4_BLOCK_SIZE]
        lo = block[0::2]
        hi = block[1::2]
        amax = torch.maximum(
            torch.maximum(lo.abs().max(), hi.abs().max()),
            torch.tensor(1.0e-4, dtype=torch.float32, device=row.device),
        )
        exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
        inv_scale = torch.exp2(-exponent)
        lo_nibbles = _deepseek_v4_mxfp4_nibble_reference(lo * inv_scale)
        hi_nibbles = _deepseek_v4_mxfp4_nibble_reference(hi * inv_scale)
        start = block_idx * _CSA_INDEXER_MXFP4_HALF_BLOCK
        packed[start : start + _CSA_INDEXER_MXFP4_HALF_BLOCK] = lo_nibbles | (
            hi_nibbles << 4
        )
        scales[block_idx] = int(exponent.item()) + 127
    return packed, scales


def _validate_deepseek_v4_indexer_mxfp4_cache_write_values(
    values: DeepSeekV4IndexerMXFP4CacheWriteInputValues,
) -> None:
    if values.index_k.ndim != 2:
        raise ValueError(f"index_k must be rank-2, got {values.index_k.ndim}")
    if values.index_k.shape[1] != _CSA_INDEXER_DIM:
        raise ValueError(
            f"index_k width must be {_CSA_INDEXER_DIM}, "
            f"got {values.index_k.shape[1]}"
        )
    if values.cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {values.cache_2d.dtype}")
    if values.cache_2d.ndim != 2:
        raise ValueError(f"cache_2d must be rank-2, got {values.cache_2d.ndim}")
    block_size = _check_positive("block_size", values.block_size)
    min_row_bytes = block_size * (
        _CSA_INDEXER_MXFP4_VALUE_BYTES + _CSA_INDEXER_MXFP4_SCALE_BYTES
    )
    if values.cache_2d.shape[1] < min_row_bytes:
        raise ValueError(
            f"cache_2d row width must be at least {min_row_bytes}, "
            f"got {values.cache_2d.shape[1]}"
        )
    if values.slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be rank-1")
    if values.valid.ndim != 1:
        raise ValueError("valid must be rank-1")
    if values.slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"slot_mapping must be integer, got {values.slot_mapping.dtype}"
        )
    if values.valid.dtype != torch.bool:
        raise TypeError(f"valid must be bool, got {values.valid.dtype}")
    num_rows = min(
        values.index_k.shape[0],
        values.slot_mapping.numel(),
        values.valid.numel(),
    )
    if values.slot_mapping.numel() != values.index_k.shape[0]:
        raise ValueError("slot_mapping length must match index_k rows")
    if values.valid.numel() != values.index_k.shape[0]:
        raise ValueError("valid length must match index_k rows")
    if not values.index_k.is_floating_point():
        raise TypeError(f"index_k must be floating point, got {values.index_k.dtype}")
    if (
        values.cache_2d.device != values.index_k.device
        or values.slot_mapping.device != values.index_k.device
        or values.valid.device != values.index_k.device
    ):
        raise ValueError("index_k, cache_2d, slot_mapping, and valid must share device")
    slots = values.slot_mapping[:num_rows].to(torch.int64)
    writable = values.valid[:num_rows] & (slots >= 0)
    if not bool(writable.any().item()):
        return
    total_slots = values.cache_2d.shape[0] * block_size
    write_slots = slots[writable]
    if int(write_slots.max().item()) >= total_slots:
        raise ValueError(
            f"writable slot_mapping entries must be < {total_slots}, "
            f"got {int(write_slots.max().item())}"
        )
    unique_count = int(torch.unique(write_slots).numel())
    if unique_count != int(write_slots.numel()):
        raise ValueError("writable slot_mapping entries must be unique")


def deepseek_v4_indexer_mxfp4_cache_write_reference(
    values: DeepSeekV4IndexerMXFP4CacheWriteInputValues,
) -> torch.Tensor:
    """Return cache bytes after applying CSA indexer MXFP4 writes."""

    _validate_deepseek_v4_indexer_mxfp4_cache_write_values(values)
    out = values.cache_2d.clone()
    num_rows = min(
        values.index_k.shape[0],
        values.slot_mapping.numel(),
        values.valid.numel(),
    )
    slots = values.slot_mapping[:num_rows].to(torch.int64)
    for row_idx in range(num_rows):
        if not bool(values.valid[row_idx].item()):
            continue
        slot = int(slots[row_idx].item())
        if slot < 0:
            continue
        page = slot // values.block_size
        pos = slot % values.block_size
        page_base = page * out.stride(0)
        value_base = page_base + pos * _CSA_INDEXER_MXFP4_VALUE_BYTES
        scale_base = (
            page_base
            + values.block_size * _CSA_INDEXER_MXFP4_VALUE_BYTES
            + pos * _CSA_INDEXER_MXFP4_SCALE_BYTES
        )
        packed, scales = _deepseek_v4_indexer_mxfp4_row_reference(
            values.index_k[row_idx]
        )
        flat = out.reshape(-1)
        flat[value_base : value_base + _CSA_INDEXER_MXFP4_VALUE_BYTES] = packed
        flat[scale_base : scale_base + _CSA_INDEXER_MXFP4_SCALE_BYTES] = scales
    return out.contiguous()


@dataclass
class DeepSeekV4IndexerMXFP4CacheGatherInputValues:
    """Generated values for gathering CSA indexer MXFP4 cache rows."""

    cache_2d: torch.Tensor
    slot_mapping: torch.Tensor
    values_out: torch.Tensor
    scales_out: torch.Tensor
    block_size: int


@dataclass
class _DeepSeekV4IndexerMXFP4CacheGatherConfig:
    """Initialization parameters for CSA indexer MXFP4 cache gathers.

    The represented operation reads packed MXFP4 value bytes and UE8M0 scale
    bytes from a paged cache into dense workspaces. Negative slot ids produce
    zero-filled output rows.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of rows to gather.
    num_rows: int

    # Required: number of physical MXFP4 cache pages.
    num_cache_blocks: int

    # Required: number of indexer rows in each physical cache page.
    block_size: int

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: rows with slot_mapping == -1. These rows gather zeros.
    negative_slot_count: int = 0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DeepSeekV4IndexerMXFP4CacheGatherBuilder:
    """Generator for CSA indexer MXFP4 cache-gather inputs."""

    config: _DeepSeekV4IndexerMXFP4CacheGatherConfig

    def __init__(self, config: _DeepSeekV4IndexerMXFP4CacheGatherConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4IndexerMXFP4CacheGatherInputValues:
        self.__post_init__()
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        values = DeepSeekV4IndexerMXFP4CacheGatherInputValues(
            cache_2d=self._generate_cache(
                seed=_child_seed(value_seed, 1),
                device=target_device,
            ),
            slot_mapping=self._generate_slot_mapping(
                seed=_child_seed(metadata_seed, 1),
                device=target_device,
            ),
            values_out=self._generate_output(
                seed=_child_seed(value_seed, 2),
                shape=(self.config.num_rows, _CSA_INDEXER_MXFP4_VALUE_BYTES),
                device=target_device,
            ),
            scales_out=self._generate_output(
                seed=_child_seed(value_seed, 3),
                shape=(self.config.num_rows, _CSA_INDEXER_MXFP4_SCALE_BYTES),
                device=target_device,
            ),
            block_size=self.config.block_size,
        )
        _validate_deepseek_v4_indexer_mxfp4_cache_gather_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.num_cache_blocks = _check_positive(
            "num_cache_blocks",
            self.config.num_cache_blocks,
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.negative_slot_count = _check_nonnegative(
            "negative_slot_count",
            self.config.negative_slot_count,
        )
        if self.config.negative_slot_count > self.config.num_rows:
            raise ValueError("negative_slot_count must be <= num_rows")
        total_slots = self.config.num_cache_blocks * self.config.block_size
        if self.config.num_rows > 0 and total_slots <= 0:
            raise ValueError("gather cache must contain at least one slot")

    def _cache_row_bytes(self) -> int:
        return self.config.block_size * (
            _CSA_INDEXER_MXFP4_VALUE_BYTES
            + _CSA_INDEXER_MXFP4_SCALE_BYTES
        )

    def _generate_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        return _generate_random_uint8_tensor(
            shape=(self.config.num_cache_blocks, self._cache_row_bytes()),
            seed=seed,
            device=device,
        )

    def _generate_output(
        self,
        *,
        seed: int,
        shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        return _generate_random_uint8_tensor(
            shape=shape,
            seed=seed,
            device=device,
        )

    def _generate_slot_mapping(
        self, *, seed: int, device: torch.device
    ) -> torch.Tensor:
        return _generate_slot_mapping_tensor(
            num_rows=self.config.num_rows,
            total_slots=self.config.num_cache_blocks * self.config.block_size,
            negative_count=self.config.negative_slot_count,
            unique=False,
            seed=seed,
            device=device,
        )


def _validate_deepseek_v4_indexer_mxfp4_cache_gather_values(
    values: DeepSeekV4IndexerMXFP4CacheGatherInputValues,
) -> None:
    if values.cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {values.cache_2d.dtype}")
    if values.values_out.dtype != torch.uint8:
        raise TypeError(f"values_out must be uint8, got {values.values_out.dtype}")
    if values.scales_out.dtype != torch.uint8:
        raise TypeError(f"scales_out must be uint8, got {values.scales_out.dtype}")
    if values.cache_2d.ndim != 2:
        raise ValueError(f"cache_2d must be rank-2, got {values.cache_2d.ndim}")
    if values.slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be rank-1")
    if values.values_out.ndim != 2 or values.scales_out.ndim != 2:
        raise ValueError("values_out and scales_out must be rank-2")
    block_size = _check_positive("block_size", values.block_size)
    min_row_bytes = block_size * (
        _CSA_INDEXER_MXFP4_VALUE_BYTES + _CSA_INDEXER_MXFP4_SCALE_BYTES
    )
    if values.cache_2d.shape[1] < min_row_bytes:
        raise ValueError(
            f"cache_2d row width must be at least {min_row_bytes}, "
            f"got {values.cache_2d.shape[1]}"
        )
    if values.cache_2d.stride(1) != 1:
        raise ValueError("cache_2d must be contiguous in the byte dimension")
    rows = values.slot_mapping.numel()
    if values.values_out.shape[0] < rows or values.scales_out.shape[0] < rows:
        raise ValueError(
            "gather output workspaces must have at least slot_mapping rows"
        )
    if values.values_out.shape[1] < _CSA_INDEXER_MXFP4_VALUE_BYTES:
        raise ValueError("values_out has insufficient value bytes")
    if values.scales_out.shape[1] < _CSA_INDEXER_MXFP4_SCALE_BYTES:
        raise ValueError("scales_out has insufficient scale bytes")
    if values.values_out.stride(1) != 1 or values.scales_out.stride(1) != 1:
        raise ValueError(
            "gather output workspaces must be contiguous in the byte dimension"
        )
    if values.slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"slot_mapping must be integer, got {values.slot_mapping.dtype}"
        )
    if (
        values.slot_mapping.device != values.cache_2d.device
        or values.values_out.device != values.cache_2d.device
        or values.scales_out.device != values.cache_2d.device
    ):
        raise ValueError(
            "cache_2d, slot_mapping, values_out, and scales_out must share device"
        )
    slots = values.slot_mapping.to(torch.int64)
    valid = slots >= 0
    if not bool(valid.any().item()):
        return
    total_slots = values.cache_2d.shape[0] * block_size
    valid_slots = slots[valid]
    if int(valid_slots.max().item()) >= total_slots:
        raise ValueError(
            f"slot_mapping entries must be < {total_slots}, "
            f"got {int(valid_slots.max().item())}"
        )


def deepseek_v4_indexer_mxfp4_cache_gather_reference(
    values: DeepSeekV4IndexerMXFP4CacheGatherInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense value and scale byte workspaces gathered from MXFP4 cache."""

    _validate_deepseek_v4_indexer_mxfp4_cache_gather_values(values)
    values_out = values.values_out.clone()
    scales_out = values.scales_out.clone()
    flat_cache = values.cache_2d.reshape(-1)
    slots = values.slot_mapping.to(torch.int64)
    for row_idx, slot_value in enumerate(slots.tolist()):
        if slot_value < 0:
            values_out[row_idx, :_CSA_INDEXER_MXFP4_VALUE_BYTES].zero_()
            scales_out[row_idx, :_CSA_INDEXER_MXFP4_SCALE_BYTES].zero_()
            continue
        page = slot_value // values.block_size
        pos = slot_value % values.block_size
        page_base = page * values.cache_2d.stride(0)
        value_base = page_base + pos * _CSA_INDEXER_MXFP4_VALUE_BYTES
        scale_base = (
            page_base
            + values.block_size * _CSA_INDEXER_MXFP4_VALUE_BYTES
            + pos * _CSA_INDEXER_MXFP4_SCALE_BYTES
        )
        values_out[
            row_idx,
            :_CSA_INDEXER_MXFP4_VALUE_BYTES,
        ] = flat_cache[value_base : value_base + _CSA_INDEXER_MXFP4_VALUE_BYTES]
        scales_out[
            row_idx,
            :_CSA_INDEXER_MXFP4_SCALE_BYTES,
        ] = flat_cache[scale_base : scale_base + _CSA_INDEXER_MXFP4_SCALE_BYTES]
    return values_out.contiguous(), scales_out.contiguous()


@dataclass
class DeepSeekV4KCacheGatherInputValues:
    """Generated values for CSA sparse K-cache gather/dequantization."""

    out: torch.Tensor
    cache_2d: torch.Tensor
    seq_lens: torch.Tensor
    gather_lens: torch.Tensor | None
    block_table: torch.Tensor
    block_size: int
    offset: int
    block_table_base_offsets: torch.Tensor | None = None


@dataclass
class _DeepSeekV4KCacheGatherConfig:
    """Initialization parameters for CSA K-cache gather/dequantization.

    The represented operation reads paged sparse-window attention K-cache rows.
    Each cache page stores per-token FP8 NoPE bytes, BF16 RoPE bytes, and UE8M0
    NoPE scale bytes. The generated metadata selects the suffix of each
    request to gather into a BF16 output workspace.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of independent request rows.
    batch_size: int

    # Required: maximum visible K-token sequence length per request.
    max_seq_len: int

    # Required: number of tokens in each physical cache page.
    block_size: int

    # ------------------------------------------------------------------
    # Optional metadata/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: maximum suffix length gathered for any request. Defaults to
    # max_seq_len. Actual per-request gather lengths are generated <= seq_len.
    max_gather_len: int | None = None

    # Optional: output-token offset where gathered rows are written.
    offset: int = 0

    # Optional: physical cache page count. Defaults to one unique page-table
    # entry for every generated request/page coordinate.
    num_cache_blocks: int | None = None

    # Optional: generate explicit gather_lens. When false, the operation
    # gathers the full generated seq_lens suffix for each request.
    include_gather_lens: bool = True

    # Optional: include block-table base offsets. Generated offsets are zero so
    # the metadata remains a valid unsliced page table while exercising the API.
    include_block_table_base_offsets: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DeepSeekV4KCacheGatherBuilder:
    """Generator for CSA paged K-cache gather/dequantization inputs."""

    config: _DeepSeekV4KCacheGatherConfig

    def __init__(self, config: _DeepSeekV4KCacheGatherConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DeepSeekV4KCacheGatherInputValues:
        self.__post_init__()
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        metadata_generator = _rng_for_device(
            target_device, _child_seed(metadata_seed, 1)
        )
        seq_lens = self._generate_seq_lens(
            generator=metadata_generator,
            device=target_device,
        )
        gather_lens = self._generate_gather_lens(
            seq_lens=seq_lens,
            generator=metadata_generator,
        )
        block_table = self._generate_block_table(
            seed=_child_seed(metadata_seed, 2),
            device=target_device,
        )
        base_offsets = None
        if self.config.include_block_table_base_offsets:
            base_offsets = torch.zeros(
                (self.config.batch_size,),
                dtype=torch.int32,
                device=target_device,
            )
        out_rows = self.config.offset + self._out_gather_width()
        values = DeepSeekV4KCacheGatherInputValues(
            out=self._generate_out(
                seed=_child_seed(value_seed, 1),
                out_rows=out_rows,
                device=target_device,
            ),
            cache_2d=self._generate_cache(
                seed=_child_seed(value_seed, 2),
                device=target_device,
            ),
            seq_lens=seq_lens.contiguous(),
            gather_lens=None if gather_lens is None else gather_lens.contiguous(),
            block_table=block_table.contiguous(),
            block_size=self.config.block_size,
            offset=self.config.offset,
            block_table_base_offsets=(
                None if base_offsets is None else base_offsets.contiguous()
            ),
        )
        _validate_deepseek_v4_k_cache_gather_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_nonnegative(
            "batch_size", self.config.batch_size
        )
        self.config.max_seq_len = _check_positive(
            "max_seq_len", self.config.max_seq_len
        )
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        if self.config.max_gather_len is None:
            self.config.max_gather_len = self.config.max_seq_len
        self.config.max_gather_len = _check_positive(
            "max_gather_len", self.config.max_gather_len
        )
        if self.config.max_gather_len > self.config.max_seq_len:
            raise ValueError(
                "max_gather_len must be <= max_seq_len; got "
                f"{self.config.max_gather_len} > {self.config.max_seq_len}"
            )
        self.config.offset = _check_nonnegative("offset", self.config.offset)
        required_pages = self._required_page_table_entries()
        if self.config.num_cache_blocks is None:
            self.config.num_cache_blocks = max(1, required_pages)
        self.config.num_cache_blocks = _check_positive(
            "num_cache_blocks", self.config.num_cache_blocks
        )
        if self.config.num_cache_blocks < required_pages:
            raise ValueError(
                "num_cache_blocks must cover generated page-table entries; got "
                f"{self.config.num_cache_blocks} < {required_pages}"
            )

    def _max_blocks_per_seq(self) -> int:
        return math.ceil(self.config.max_seq_len / self.config.block_size)

    def _required_page_table_entries(self) -> int:
        return self.config.batch_size * self._max_blocks_per_seq()

    def _cache_row_bytes(self) -> int:
        return self.config.block_size * (
            _CSA_SWA_TOKEN_STRIDE + _CSA_SWA_SCALE_DIM
        )

    def _out_gather_width(self) -> int:
        if self.config.include_gather_lens:
            return int(self.config.max_gather_len)
        return int(self.config.max_seq_len)

    def _generate_seq_lens(
        self,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        if self.config.batch_size == 0:
            return torch.empty((0,), dtype=torch.int32, device=device)
        return torch.randint(
            1,
            self.config.max_seq_len + 1,
            (self.config.batch_size,),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )

    def _generate_gather_lens(
        self,
        *,
        seq_lens: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        if not self.config.include_gather_lens:
            return None
        gather_lens = torch.empty_like(seq_lens)
        for row, seq_len in enumerate(seq_lens.tolist()):
            upper = min(int(seq_len), int(self.config.max_gather_len))
            if upper <= 1:
                gather_lens[row] = upper
            else:
                gather_lens[row] = torch.randint(
                    1,
                    upper + 1,
                    (),
                    dtype=torch.int32,
                    device=seq_lens.device,
                    generator=generator,
                )
        return gather_lens

    def _generate_block_table(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        width = self._max_blocks_per_seq()
        if self.config.batch_size == 0:
            return torch.empty((0, width), dtype=torch.int32, device=device)
        assert self.config.num_cache_blocks is not None
        return PageTableInput(
            PageTableInputConfig(
                batch_size=self.config.batch_size,
                max_pages_per_request=width,
                indexing="random",
                num_physical_pages=self.config.num_cache_blocks,
            )
        ).generate(seed=seed, device=device).page_table

    def _generate_out(
        self,
        *,
        seed: int,
        out_rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        generator = _rng_for_device(device, seed)
        out = torch.randn(
            (self.config.batch_size, out_rows, _CSA_HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        return out.to(torch.bfloat16).contiguous()

    def _generate_cache(self, *, seed: int, device: torch.device) -> torch.Tensor:
        assert self.config.num_cache_blocks is not None
        generator = _rng_for_device(device, seed)
        cache = torch.empty(
            (self.config.num_cache_blocks, self._cache_row_bytes()),
            dtype=torch.uint8,
            device=device,
        )
        token_area = cache[:, : self.config.block_size * _CSA_SWA_TOKEN_STRIDE]
        token_area = token_area.reshape(
            self.config.num_cache_blocks,
            self.config.block_size,
            _CSA_SWA_TOKEN_STRIDE,
        )
        scale_area = cache[:, self.config.block_size * _CSA_SWA_TOKEN_STRIDE :]
        scale_area = scale_area.reshape(
            self.config.num_cache_blocks,
            self.config.block_size,
            _CSA_SWA_SCALE_DIM,
        )
        nope_fp8 = (
            torch.randn(
                (
                    self.config.num_cache_blocks,
                    self.config.block_size,
                    _CSA_NOPE_DIM,
                ),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            .clamp(-4.0, 4.0)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        rope_bf16 = (
            torch.randn(
                (
                    self.config.num_cache_blocks,
                    self.config.block_size,
                    _CSA_ROPE_DIM,
                ),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            .to(torch.bfloat16)
            .view(torch.uint8)
        )
        scale_bytes = torch.randint(
            123,
            131,
            (
                self.config.num_cache_blocks,
                self.config.block_size,
                _CSA_SWA_SCALE_DIM,
            ),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        scale_bytes[..., -1] = 127
        token_area[..., :_CSA_NOPE_DIM] = nope_fp8
        token_area[
            ...,
            _CSA_NOPE_DIM:_CSA_SWA_TOKEN_STRIDE,
        ] = rope_bf16
        scale_area.copy_(scale_bytes)
        return cache.contiguous()


def _validate_deepseek_v4_k_cache_gather_values(
    values: DeepSeekV4KCacheGatherInputValues,
) -> None:
    if values.out.dtype != torch.bfloat16:
        raise TypeError(f"out must be bfloat16, got {values.out.dtype}")
    if values.cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {values.cache_2d.dtype}")
    if values.out.ndim != 3:
        raise ValueError(f"out must be rank-3, got {values.out.ndim}")
    if values.out.shape[-1] != _CSA_HEAD_DIM:
        raise ValueError(
            f"out hidden width must be {_CSA_HEAD_DIM}, "
            f"got {values.out.shape[-1]}"
        )
    if values.out.stride(-1) != 1:
        raise ValueError("out must be contiguous in the hidden dimension")
    if values.cache_2d.ndim != 2:
        raise ValueError(f"cache_2d must be rank-2, got {values.cache_2d.ndim}")
    block_size = _check_positive("block_size", values.block_size)
    row_bytes = block_size * (
        _CSA_SWA_TOKEN_STRIDE + _CSA_SWA_SCALE_DIM
    )
    if values.cache_2d.shape[1] < row_bytes:
        raise ValueError(
            f"cache_2d row width must be at least {row_bytes}, "
            f"got {values.cache_2d.shape[1]}"
        )
    if values.cache_2d.stride(1) != 1:
        raise ValueError("cache_2d must be contiguous in the byte dimension")
    if values.seq_lens.ndim != 1:
        raise ValueError("seq_lens must be rank-1")
    if values.seq_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"seq_lens must be integer, got {values.seq_lens.dtype}")
    if values.gather_lens is not None:
        if values.gather_lens.ndim != 1:
            raise ValueError("gather_lens must be rank-1")
        if values.gather_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"gather_lens must be integer, got {values.gather_lens.dtype}"
            )
        if values.gather_lens.shape != values.seq_lens.shape:
            raise ValueError("gather_lens shape must match seq_lens")
    if values.block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    batch_size = values.seq_lens.numel()
    if values.out.shape[0] != batch_size or values.block_table.shape[0] != batch_size:
        raise ValueError("out, block_table, and seq_lens batch dimensions must match")
    if values.block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"block_table must be integer, got {values.block_table.dtype}")
    if values.block_table_base_offsets is not None:
        if values.block_table_base_offsets.shape != values.seq_lens.shape:
            raise ValueError("block_table_base_offsets shape must match seq_lens")
        if values.block_table_base_offsets.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "block_table_base_offsets must be integer, got "
                f"{values.block_table_base_offsets.dtype}"
            )
    if (
        values.out.device != values.cache_2d.device
        or values.seq_lens.device != values.cache_2d.device
        or values.block_table.device != values.cache_2d.device
        or (
            values.gather_lens is not None
            and values.gather_lens.device != values.cache_2d.device
        )
        or (
            values.block_table_base_offsets is not None
            and values.block_table_base_offsets.device != values.cache_2d.device
        )
    ):
        raise ValueError("K-cache gather values must share a device")
    offset = _check_nonnegative("offset", values.offset)
    seq_lens = values.seq_lens.to(torch.int64)
    gather_lens = (
        seq_lens if values.gather_lens is None else values.gather_lens.to(torch.int64)
    )
    if bool((seq_lens < 0).any().item()):
        raise ValueError("seq_lens entries must be non-negative")
    if bool((gather_lens < 0).any().item()):
        raise ValueError("gather_lens entries must be non-negative")
    if bool((gather_lens > seq_lens).any().item()):
        raise ValueError("gather_lens entries must be <= seq_lens")
    if (
        gather_lens.numel()
        and offset + int(gather_lens.max().item()) > values.out.shape[1]
    ):
        raise ValueError("out token dimension is too small for offset + gather_lens")
    if values.block_table.numel():
        if bool((values.block_table < 0).any().item()):
            raise ValueError("block_table entries must be non-negative")
        if int(values.block_table.max().item()) >= values.cache_2d.shape[0]:
            raise ValueError("block_table entries must index cache_2d rows")
    base_offsets = (
        torch.zeros_like(seq_lens)
        if values.block_table_base_offsets is None
        else values.block_table_base_offsets.to(torch.int64)
    )
    if bool((base_offsets < 0).any().item()):
        raise ValueError("block_table_base_offsets entries must be non-negative")
    max_blocks = values.block_table.shape[1]
    for row in range(batch_size):
        seq_len = int(seq_lens[row].item())
        gather_len = int(gather_lens[row].item())
        start_pos = seq_len - gather_len
        base = int(base_offsets[row].item())
        for pos in range(start_pos, seq_len):
            table_idx = pos // block_size - base
            if table_idx < 0 or table_idx >= max_blocks:
                raise ValueError("block_table is too narrow for generated positions")


def deepseek_v4_dequantize_and_gather_k_cache_reference(
    values: DeepSeekV4KCacheGatherInputValues,
) -> torch.Tensor:
    """Return BF16 K rows gathered/dequantized from CSA paged cache."""

    _validate_deepseek_v4_k_cache_gather_values(values)
    out = values.out.clone()
    block_size = values.block_size
    seq_lens = values.seq_lens.to(torch.int64)
    gather_lens = (
        seq_lens if values.gather_lens is None else values.gather_lens.to(torch.int64)
    )
    base_offsets = (
        torch.zeros_like(seq_lens)
        if values.block_table_base_offsets is None
        else values.block_table_base_offsets.to(torch.int64)
    )
    n_quant_blocks = _CSA_NOPE_DIM // _CSA_FP8_QUANT_BLOCK
    for batch_idx in range(seq_lens.numel()):
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = int(gather_lens[batch_idx].item())
        start_pos = seq_len - gather_len
        base = int(base_offsets[batch_idx].item())
        for gather_idx in range(gather_len):
            pos = start_pos + gather_idx
            table_idx = pos // block_size - base
            out_row = out[batch_idx, values.offset + gather_idx]
            physical = int(values.block_table[batch_idx, table_idx].item())
            pos_in_block = pos % block_size
            cache_row = values.cache_2d[physical]
            token_base = pos_in_block * _CSA_SWA_TOKEN_STRIDE
            scale_base = (
                block_size * _CSA_SWA_TOKEN_STRIDE
                + pos_in_block * _CSA_SWA_SCALE_DIM
            )
            for qblock in range(n_quant_blocks):
                qstart = token_base + qblock * _CSA_FP8_QUANT_BLOCK
                qend = qstart + _CSA_FP8_QUANT_BLOCK
                x_fp8 = cache_row[qstart:qend].view(torch.float8_e4m3fn).float()
                scale_exp = int(cache_row[scale_base + qblock].item()) - 127
                scale = math.pow(2.0, scale_exp)
                out_row[
                    qblock
                    * _CSA_FP8_QUANT_BLOCK : (qblock + 1)
                    * _CSA_FP8_QUANT_BLOCK
                ] = (x_fp8 * scale).to(torch.bfloat16)
            rope_start = token_base + _CSA_NOPE_DIM
            rope_end = rope_start + _CSA_ROPE_DIM * 2
            out_row[_CSA_NOPE_DIM:] = cache_row[rope_start:rope_end].view(
                torch.bfloat16
            )
    return out.contiguous()




def _deepseek_v4_csa_shapes(values: CSAInputValues) -> tuple[int, int, int, int, torch.dtype]:
    compressed = values.compressed
    if compressed is None:
        raise ValueError("DeepSeek V4 CSA helper values require compressed CSA inputs")
    batch_size = int(values.sliding_window.seq_lens.numel())
    num_tokens = int(values.sliding_window.q.shape[0])
    max_seq_len = max(1, int(values.sliding_window.seq_lens.max().item()))
    block_size = int(values.sliding_window.block_size)
    return batch_size, num_tokens, max_seq_len, block_size, values.sliding_window.q.dtype


def deepseek_v4_inv_rope_fp8_quant_values_from_csa(
    values: CSAInputValues,
    *,
    n_groups: int,
    heads_per_group: int,
) -> DeepSeekV4InvRoPEFP8QuantInputValues:
    if values.indexer is None:
        raise ValueError("inverse-RoPE FP8 helper values require CSA indexer inputs")
    return DeepSeekV4InvRoPEFP8QuantInputValues(
        o=values.sliding_window.q,
        positions=values.sliding_window.positions.to(torch.int64),
        cos_sin_cache=values.indexer.query.cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=_CSA_NOPE_DIM,
        rope_dim=_CSA_ROPE_DIM,
        quant_group_size=_CSA_FP8_QUANT_BLOCK * 2,
        tma_aligned_scales=True,
    )


def deepseek_v4_csa_indexer_mxfp4_cache_insert_values_from_csa(
    values: CSAInputValues,
    *,
    seed: int | None = None,
    metadata_seed: int | None = None,
    value_seed: int | None = None,
    device: DeviceLike = None,
) -> DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues:
    if values.indexer is None or values.compressed is None:
        raise ValueError("indexer cache-insert helper values require CSA indexer inputs")
    batch_size, num_tokens, max_seq_len, block_size, dtype = _deepseek_v4_csa_shapes(values)
    state = values.compressed.compressor_state
    return _DeepSeekV4CSAIndexerMXFP4CacheInsertBuilder(
        _DeepSeekV4CSAIndexerMXFP4CacheInsertConfig(
            num_tokens=num_tokens,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_state_cache_blocks=int(state.state_cache.shape[0]),
            compressor_block_size=int(state.block_size),
            num_kv_cache_blocks=max(1, batch_size),
            kv_cache_block_size=block_size,
            dtype=dtype,
            compress_ratio=int(state.compress_ratio),
        )
    ).generate(seed=seed, metadata_seed=metadata_seed, value_seed=value_seed, device=device)


def deepseek_v4_sparse_compress_cache_insert_values_from_csa(
    values: CSAInputValues,
    *,
    seed: int | None = None,
    metadata_seed: int | None = None,
    value_seed: int | None = None,
    device: DeviceLike = None,
) -> DeepSeekV4SparseCompressCacheInsertInputValues:
    if values.compressed is None:
        raise ValueError("sparse-compress helper values require compressed CSA inputs")
    batch_size, num_tokens, max_seq_len, block_size, dtype = _deepseek_v4_csa_shapes(values)
    state = values.compressed.compressor_state
    return _DeepSeekV4SparseCompressCacheInsertBuilder(
        _DeepSeekV4SparseCompressCacheInsertConfig(
            num_tokens=num_tokens,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_state_cache_blocks=int(state.state_cache.shape[0]),
            compressor_block_size=int(state.block_size),
            num_kv_cache_blocks=max(1, batch_size),
            kv_cache_block_size=block_size,
            compress_ratio=int(state.compress_ratio),
            overlap=int(state.compress_ratio) == 4,
            dtype=dtype,
        )
    ).generate(seed=seed, metadata_seed=metadata_seed, value_seed=value_seed, device=device)


def deepseek_v4_indexer_mxfp4_cache_write_values_from_csa(
    values: CSAInputValues,
    *,
    seed: int | None = None,
    metadata_seed: int | None = None,
    value_seed: int | None = None,
    device: DeviceLike = None,
) -> DeepSeekV4IndexerMXFP4CacheWriteInputValues:
    _, num_tokens, _, block_size, dtype = _deepseek_v4_csa_shapes(values)
    return _DeepSeekV4IndexerMXFP4CacheWriteBuilder(
        _DeepSeekV4IndexerMXFP4CacheWriteConfig(
            num_rows=num_tokens,
            num_cache_blocks=max(1, math.ceil(max(1, num_tokens) / block_size)),
            block_size=block_size,
            dtype=dtype,
        )
    ).generate(seed=seed, metadata_seed=metadata_seed, value_seed=value_seed, device=device)


def deepseek_v4_indexer_mxfp4_cache_gather_values_from_csa(
    values: CSAInputValues,
    *,
    seed: int | None = None,
    metadata_seed: int | None = None,
    value_seed: int | None = None,
    device: DeviceLike = None,
) -> DeepSeekV4IndexerMXFP4CacheGatherInputValues:
    _, num_tokens, _, block_size, _ = _deepseek_v4_csa_shapes(values)
    return _DeepSeekV4IndexerMXFP4CacheGatherBuilder(
        _DeepSeekV4IndexerMXFP4CacheGatherConfig(
            num_rows=num_tokens,
            num_cache_blocks=max(1, math.ceil(max(1, num_tokens) / block_size)),
            block_size=block_size,
        )
    ).generate(seed=seed, metadata_seed=metadata_seed, value_seed=value_seed, device=device)


def deepseek_v4_k_cache_gather_values_from_csa(
    values: CSAInputValues,
    *,
    seed: int | None = None,
    metadata_seed: int | None = None,
    value_seed: int | None = None,
    device: DeviceLike = None,
) -> DeepSeekV4KCacheGatherInputValues:
    batch_size, _, max_seq_len, block_size, _ = _deepseek_v4_csa_shapes(values)
    return _DeepSeekV4KCacheGatherBuilder(
        _DeepSeekV4KCacheGatherConfig(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            block_size=block_size,
            max_gather_len=min(max_seq_len, int(values.sliding_window.window_size)),
            offset=0,
            num_cache_blocks=max(1, batch_size * math.ceil(max_seq_len / block_size)),
            include_gather_lens=True,
            include_block_table_base_offsets=(
                values.compressed is not None
                and values.compressed.paged_index.block_table_base_offsets is not None
            ),
        )
    ).generate(seed=seed, metadata_seed=metadata_seed, value_seed=value_seed, device=device)


__all__ = [
    "DeepSeekV4CompressorStateInputValues",
    "DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues",
    "DeepSeekV4IndexerMXFP4CacheGatherInputValues",
    "DeepSeekV4IndexerMXFP4CacheWriteInputValues",
    "DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues",
    "DeepSeekV4InvRoPEFP8QuantInputValues",
    "DeepSeekV4KCacheGatherInputValues",
    "DeepSeekV4PagedIndexValues",
    "DeepSeekV4SparseCompressCacheInsertInputValues",
    "DeepSeekV4SparsePrefillIndexValues",
    "deepseek_v4_build_dense_prefill_local_compressed_indices_reference",
    "deepseek_v4_combine_dense_swa_indices_reference",
    "deepseek_v4_combine_topk_swa_indices_reference",
    "deepseek_v4_compressed_slot_mapping_reference",
    "deepseek_v4_compute_global_topk_indices_and_lens_reference",
    "deepseek_v4_csa_indexer_mxfp4_cache_insert_reference",
    "deepseek_v4_csa_indexer_mxfp4_cache_insert_values_from_csa",
    "deepseek_v4_decode_swa_indices_and_lens_reference",
    "deepseek_v4_dequantize_and_gather_k_cache_reference",
    "deepseek_v4_indexer_decode_metadata_reference",
    "deepseek_v4_indexer_mxfp4_cache_gather_reference",
    "deepseek_v4_indexer_mxfp4_cache_gather_values_from_csa",
    "deepseek_v4_indexer_mxfp4_cache_write_reference",
    "deepseek_v4_indexer_mxfp4_cache_write_values_from_csa",
    "deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference",
    "deepseek_v4_inv_rope_fp8_quant_reference",
    "deepseek_v4_inv_rope_fp8_quant_values_from_csa",
    "deepseek_v4_k_cache_gather_values_from_csa",
    "deepseek_v4_save_compressor_state_reference",
    "deepseek_v4_sparse_compress_cache_insert_reference",
    "deepseek_v4_sparse_compress_cache_insert_values_from_csa",
]
