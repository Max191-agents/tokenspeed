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

"""Attention-family input generators for numerical correctness tests."""

from __future__ import annotations

from ._core import (
    DeviceLike,
    NumericsInputGenerator,
    PageTableIndexing,
    PageTableInput,
    PageTableInputConfig,
    PageTableValues,
    TensorInput,
    _DSA_SPARSE_DECODE_BF16_BYTES,
    _DSA_SPARSE_DECODE_FP8_E4M3_MAX,
    _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK,
    _DSA_SPARSE_DECODE_FP8_SCALE_BYTES,
    _check_float_dtype,
    _check_matches,
    _check_nonnegative,
    _check_page_table_indexing,
    _check_positive,
    _check_power_of_2,
    _child_seed,
    _require_tensor,
    _resolve_device,
    _rng_for_device,
    dataclass,
    torch,
)


def dsa_sparse_decode_row_bytes(nope_dim: int, rope_dim: int) -> int:
    """Return bytes per packed sparse-decode KV row."""

    nope_dim = int(nope_dim)
    rope_dim = int(rope_dim)
    if nope_dim % _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK != 0:
        raise ValueError(
            "dynamic sparse attention decode NoPE dim must be divisible by "
            f"{_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK}, got {nope_dim}"
        )
    return (
        nope_dim
        + nope_dim
        // _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK
        * _DSA_SPARSE_DECODE_FP8_SCALE_BYTES
        + rope_dim * _DSA_SPARSE_DECODE_BF16_BYTES
    )


@dataclass
class DSASparseDecodeKVPackInputValues:
    """Generated values for dynamic sparse attention decode KV row packing.

    The represented output row layout is:

    ```text
    [NoPE FP8 E4M3 bytes][FP32 scale bytes per 128 NoPE channels][RoPE BF16 bytes]
    ```
    """

    out: torch.Tensor
    loc: torch.Tensor
    cache_k_nope: torch.Tensor
    cache_k_rope: torch.Tensor


@dataclass
class DSASparseDecodeKVPackInputConfig:
    """Initialization parameters for dynamic sparse attention KV row packing.

    The operation packs per-token sparse-decode K-cache rows into physical
    cache slots. Non-RoPE key channels are dynamically scaled per 128-channel
    block and stored as FP8 E4M3 bytes. RoPE key channels are copied as raw
    BF16 bytes after the FP32 scale sidecar.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows to pack.
    num_tokens: int

    # Required: number of physical output rows available for sparse decode.
    num_slots: int

    # Required: non-RoPE key width. Must be a power of two and divisible by 128.
    nope_dim: int

    # Required: RoPE key width. Must be a power of two.
    rope_dim: int

    # ------------------------------------------------------------------
    # Optional generation configuration.
    # ------------------------------------------------------------------

    # Optional: include the singleton KV-head axis accepted by some callers,
    # generating [tokens, 1, dim] source tensors instead of [tokens, dim].
    include_head_axis: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DSASparseDecodeKVPackBuilder:
    """Generator for dynamic sparse attention decode KV row packing."""

    config: DSASparseDecodeKVPackInputConfig
    cache_k_nope_input: TensorInput | None
    cache_k_rope_input: TensorInput | None

    def __init__(self, config: DSASparseDecodeKVPackInputConfig) -> None:
        self.config = config
        self.cache_k_nope_input = None
        self.cache_k_rope_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_slots = _check_positive("num_slots", self.config.num_slots)
        if self.config.num_tokens > self.config.num_slots:
            raise ValueError(
                "num_tokens must be <= num_slots so generated slot locations "
                f"are unique, got num_tokens={self.config.num_tokens}, "
                f"num_slots={self.config.num_slots}"
            )
        self.config.nope_dim = _check_power_of_2("nope_dim", self.config.nope_dim)
        if self.config.nope_dim % _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK != 0:
            raise ValueError(
                "nope_dim must be divisible by "
                f"{_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK}, got {self.config.nope_dim}"
            )
        self.config.rope_dim = _check_power_of_2("rope_dim", self.config.rope_dim)
        self.config.include_head_axis = bool(self.config.include_head_axis)
        self.cache_k_nope_input = self.cache_k_nope_input or TensorInput(
            self._source_shape(self.config.nope_dim),
            torch.bfloat16,
            device=self.config.device,
        )
        self.cache_k_rope_input = self.cache_k_rope_input or TensorInput(
            self._source_shape(self.config.rope_dim),
            torch.bfloat16,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DSASparseDecodeKVPackInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        if self.cache_k_nope_input is None or self.cache_k_rope_input is None:
            raise ValueError(
                "_DSASparseDecodeKVPackBuilder child generators must be initialized"
            )

        self.cache_k_nope_input.shape = self._source_shape(self.config.nope_dim)
        self.cache_k_nope_input.dtype = torch.bfloat16
        self.cache_k_nope_input.device = self.config.device
        self.cache_k_rope_input.shape = self._source_shape(self.config.rope_dim)
        self.cache_k_rope_input.dtype = torch.bfloat16
        self.cache_k_rope_input.device = self.config.device

        cache_k_nope = self.cache_k_nope_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        ).values
        cache_k_rope = self.cache_k_rope_input.generate(
            seed=_child_seed(seed, 2),
            device=target_device,
        ).values
        if cache_k_nope is None or cache_k_rope is None:
            raise ValueError(
                "dynamic sparse attention K source tensors must be generated"
            )

        out = self._generate_out(seed=_child_seed(seed, 3), device=target_device)
        loc_seed = seed if metadata_seed is None else metadata_seed
        loc = self._generate_loc(seed=_child_seed(loc_seed, 4), device=target_device)
        values = DSASparseDecodeKVPackInputValues(
            out=out,
            loc=loc,
            cache_k_nope=cache_k_nope,
            cache_k_rope=cache_k_rope,
        )
        _validate_dsa_sparse_decode_kv_pack_values(values)
        return values

    def _source_shape(self, dim: int) -> tuple[int, ...]:
        if self.config.include_head_axis:
            return (self.config.num_tokens, 1, int(dim))
        return (self.config.num_tokens, int(dim))

    def _generate_out(self, *, seed: int, device: torch.device) -> torch.Tensor:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        row_bytes = dsa_sparse_decode_row_bytes(
            self.config.nope_dim,
            self.config.rope_dim,
        )
        return torch.randint(
            0,
            256,
            (self.config.num_slots, row_bytes),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def _generate_loc(self, *, seed: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randperm(
            self.config.num_slots,
            dtype=torch.int64,
            generator=generator,
        )[: self.config.num_tokens].to(device)


def _dsa_sparse_decode_source_2d(
    name: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[1] != 1:
            raise ValueError(f"{name} rank-3 form must have one KV head")
        tensor = tensor.squeeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2 or rank-3, got {tensor.ndim}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16, got {tensor.dtype}")
    return tensor.contiguous()


def _validate_dsa_sparse_decode_kv_pack_values(
    values: DSASparseDecodeKVPackInputValues,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if values.out.dtype != torch.uint8:
        raise TypeError(f"out must be uint8, got {values.out.dtype}")
    if values.out.ndim != 2:
        raise ValueError(f"out must be rank-2, got {values.out.ndim}")
    if values.loc.ndim != 1:
        raise ValueError(f"loc must be rank-1, got {values.loc.ndim}")
    if values.loc.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"loc must use an integer dtype, got {values.loc.dtype}")
    if values.loc.device != values.out.device:
        raise ValueError("loc and out must be on the same device")

    cache_k_nope = _dsa_sparse_decode_source_2d(
        "cache_k_nope",
        values.cache_k_nope,
    )
    cache_k_rope = _dsa_sparse_decode_source_2d(
        "cache_k_rope",
        values.cache_k_rope,
    )
    if (
        cache_k_nope.device != values.out.device
        or cache_k_rope.device != values.out.device
    ):
        raise ValueError("cache_k_nope, cache_k_rope, loc, and out must share device")
    if cache_k_nope.shape[0] != values.loc.numel():
        raise ValueError("cache_k_nope token dimension must match loc length")
    if cache_k_rope.shape[0] != values.loc.numel():
        raise ValueError("cache_k_rope token dimension must match loc length")

    nope_dim = _check_power_of_2("nope_dim", int(cache_k_nope.shape[1]))
    if nope_dim % _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK != 0:
        raise ValueError(
            "nope_dim must be divisible by "
            f"{_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK}, got {nope_dim}"
        )
    rope_dim = _check_power_of_2("rope_dim", int(cache_k_rope.shape[1]))
    expected_row_bytes = dsa_sparse_decode_row_bytes(nope_dim, rope_dim)
    if values.out.shape[1] != expected_row_bytes:
        raise ValueError(
            f"out row width must be {expected_row_bytes}, got {values.out.shape[1]}"
        )

    if values.loc.numel() > 0:
        loc = values.loc.to(torch.int64)
        if int(loc.min().item()) < 0 or int(loc.max().item()) >= values.out.shape[0]:
            raise ValueError("loc entries must be valid output row indices")
        unique_count = int(torch.unique(loc).numel())
        if unique_count != values.loc.numel():
            raise ValueError("loc entries must be unique to avoid output row races")

    return cache_k_nope, cache_k_rope, nope_dim, rope_dim


def dsa_sparse_decode_kv_pack_reference(
    values: DSASparseDecodeKVPackInputValues,
) -> torch.Tensor:
    """Return sparse-decode KV rows packed into the generated output buffer."""

    cache_k_nope, cache_k_rope, nope_dim, rope_dim = (
        _validate_dsa_sparse_decode_kv_pack_values(values)
    )
    out = values.out.clone()
    if values.loc.numel() == 0:
        return out

    num_nope_blocks = nope_dim // _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK
    nope_blocks = cache_k_nope.float().reshape(
        values.loc.numel(),
        num_nope_blocks,
        _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK,
    )
    scales = nope_blocks.abs().amax(dim=-1) / _DSA_SPARSE_DECODE_FP8_E4M3_MAX
    scales = torch.clamp(scales, min=1.0e-26)
    fp8_nope = torch.clamp(
        nope_blocks / scales.unsqueeze(-1),
        -_DSA_SPARSE_DECODE_FP8_E4M3_MAX,
        _DSA_SPARSE_DECODE_FP8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    fp8_nope_bytes = (
        fp8_nope.contiguous()
        .view(torch.uint8)
        .reshape(
            values.loc.numel(),
            nope_dim,
        )
    )
    scale_bytes = (
        scales.float()
        .contiguous()
        .view(torch.uint8)
        .reshape(
            values.loc.numel(),
            num_nope_blocks * _DSA_SPARSE_DECODE_FP8_SCALE_BYTES,
        )
    )
    rope_bytes = (
        cache_k_rope.contiguous()
        .view(torch.uint8)
        .reshape(
            values.loc.numel(),
            rope_dim * _DSA_SPARSE_DECODE_BF16_BYTES,
        )
    )

    loc = values.loc.to(torch.int64)
    scale_offset = nope_dim
    rope_offset = scale_offset + scale_bytes.shape[1]
    out[loc, :nope_dim] = fp8_nope_bytes
    out[loc, scale_offset:rope_offset] = scale_bytes
    out[loc, rope_offset:] = rope_bytes
    return out


@dataclass
class DSADecodeTopKInputValues:
    """Generated values for deterministic dynamic sparse attention top-k."""

    logits: torch.Tensor
    out: torch.Tensor
    valid_lens: torch.Tensor
    topk: int


@dataclass
class DSADecodeTopKInputConfig:
    """Initialization parameters for deterministic dynamic sparse attention top-k.

    The represented operation selects the top-k local context offsets from each
    row of pre-masked dynamic sparse attention decode indexer logits. Logits
    beyond each row's valid context length are set to ``-inf``. Ties are
    resolved by choosing the smaller local offset first.
    """

    # Required: number of independent decode rows.
    num_rows: int

    # Required: maximum number of local context offsets in each row.
    vocab_size: int

    # Required: number of selected offsets per row.
    topk: int

    # Required: generated logit dtype.
    dtype: torch.dtype

    # Optional: minimum generated valid context length. Defaults to topk so
    # every row has enough finite logits to produce top-k valid offsets.
    min_valid_len: int | None = None

    # Optional: maximum generated valid context length. Defaults to vocab_size.
    max_valid_len: int | None = None

    # Optional: plant a deterministic equal-logit boundary case when the row
    # has at least one spare valid offset beyond topk.
    include_boundary_tie: bool = True

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DSADecodeTopKBuilder:
    """Generator for deterministic dynamic sparse attention top-k buffers."""

    config: DSADecodeTopKInputConfig

    def __init__(self, config: DSADecodeTopKInputConfig) -> None:
        self.config = config
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.num_rows = _check_nonnegative("num_rows", self.config.num_rows)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.topk = _check_positive("topk", self.config.topk)
        if self.config.topk > self.config.vocab_size:
            raise ValueError("topk must be <= vocab_size")
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        min_valid_len = (
            self.config.topk
            if self.config.min_valid_len is None
            else _check_positive("min_valid_len", self.config.min_valid_len)
        )
        max_valid_len = (
            self.config.vocab_size
            if self.config.max_valid_len is None
            else _check_positive("max_valid_len", self.config.max_valid_len)
        )
        if min_valid_len < self.config.topk:
            raise ValueError("min_valid_len must be >= topk")
        if max_valid_len > self.config.vocab_size:
            raise ValueError("max_valid_len must be <= vocab_size")
        if min_valid_len > max_valid_len:
            raise ValueError("min_valid_len must be <= max_valid_len")
        self.config.min_valid_len = min_valid_len
        self.config.max_valid_len = max_valid_len

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> DSADecodeTopKInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        value_generator = _rng_for_device(target_device, _child_seed(seed, 1))
        logits = -1.0 - 3.0 * torch.rand(
            (self.config.num_rows, self.config.vocab_size),
            dtype=torch.float32,
            device=target_device,
            generator=value_generator,
        )
        valid_lens = self._generate_valid_lens(seed=_child_seed(seed, 2)).to(
            target_device
        )
        for row_idx, valid_len_tensor in enumerate(valid_lens.cpu()):
            valid_len = int(valid_len_tensor.item())
            if valid_len < self.config.vocab_size:
                logits[row_idx, valid_len:] = -float("inf")
            self._plant_topk_row(logits[row_idx], valid_len=valid_len)
        logits = logits.to(self.config.dtype)
        out = torch.full(
            (self.config.num_rows, self.config.topk),
            -1,
            dtype=torch.int32,
            device=target_device,
        )
        values = DSADecodeTopKInputValues(
            logits=logits,
            out=out,
            valid_lens=valid_lens,
            topk=self.config.topk,
        )
        _validate_dsa_decode_topk_values(values)
        return values

    def _generate_valid_lens(self, *, seed: int) -> torch.Tensor:
        if self.config.num_rows == 0:
            return torch.empty((0,), dtype=torch.int32)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(
            int(self.config.min_valid_len),
            int(self.config.max_valid_len) + 1,
            (self.config.num_rows,),
            dtype=torch.int32,
            generator=rng,
        )

    def _plant_topk_row(self, row: torch.Tensor, *, valid_len: int) -> None:
        if valid_len <= 0:
            return
        if self.config.include_boundary_tie and valid_len > self.config.topk:
            for rank in range(self.config.topk - 1):
                row[rank] = 32.0 - float(rank)
            row[valid_len - 2] = 1.0
            row[valid_len - 1] = 1.0
            return
        for rank in range(self.config.topk):
            row[rank] = 32.0 - float(rank)


def _validate_dsa_decode_topk_values(values: DSADecodeTopKInputValues) -> None:
    if values.logits.ndim != 2:
        raise ValueError("logits must be rank-2")
    if values.out.ndim != 2:
        raise ValueError("out must be rank-2")
    if values.valid_lens.ndim != 1:
        raise ValueError("valid_lens must be rank-1")
    if not torch.is_floating_point(values.logits):
        raise TypeError("logits must use a floating dtype")
    if values.out.dtype != torch.int32:
        raise TypeError(f"out must be int32, got {values.out.dtype}")
    if values.valid_lens.dtype != torch.int32:
        raise TypeError(f"valid_lens must be int32, got {values.valid_lens.dtype}")
    if values.topk <= 0:
        raise ValueError("topk must be positive")
    num_rows, vocab_size = values.logits.shape
    if values.out.shape != (num_rows, values.topk):
        raise ValueError("out must have shape [num_rows, topk]")
    if values.valid_lens.shape != (num_rows,):
        raise ValueError("valid_lens must have one entry per logit row")
    if values.topk > vocab_size:
        raise ValueError("topk must be <= logits width")
    if (
        values.logits.device != values.out.device
        or values.logits.device != values.valid_lens.device
    ):
        raise ValueError("logits, out, and valid_lens must share device")
    valid_lens = values.valid_lens.cpu().tolist()
    logits_cpu = values.logits.detach().cpu()
    for row_idx, valid_len in enumerate(valid_lens):
        valid_len = int(valid_len)
        if valid_len < values.topk or valid_len > vocab_size:
            raise ValueError("valid_lens entries must be in [topk, logits width]")
        valid_logits = logits_cpu[row_idx, :valid_len]
        if not torch.isfinite(valid_logits).all():
            raise ValueError("valid logits must be finite")
        if (
            valid_len < vocab_size
            and not torch.isneginf(logits_cpu[row_idx, valid_len:]).all()
        ):
            raise ValueError("logits after valid_lens must be masked with -inf")


def dsa_decode_topk_reference(values: DSADecodeTopKInputValues) -> torch.Tensor:
    """Return stable top-k offsets for dynamic sparse attention decode logits."""

    _validate_dsa_decode_topk_values(values)
    logits_cpu = values.logits.detach().cpu().float()
    result = torch.empty(
        values.out.shape,
        dtype=torch.int32,
        device=values.logits.device,
    )
    for row_idx in range(logits_cpu.shape[0]):
        ordered = sorted(
            range(logits_cpu.shape[1]),
            key=lambda index: (-float(logits_cpu[row_idx, index]), index),
        )
        result[row_idx] = torch.tensor(
            ordered[: values.topk],
            dtype=torch.int32,
            device=values.logits.device,
        )
    return result


@dataclass
class DSATopKSlotInputValues:
    """Generated values for dynamic sparse attention top-k slot conversion."""

    local_topk_offsets: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    block_table_cpu: list[list[int]]
    block_table_values: PageTableValues
    block_size: int
    topk: int


@dataclass
class DSATopKSlotInputConfig:
    """Initialization parameters for dynamic sparse attention slot conversion.

    The represented metadata operation maps per-token local offsets through a
    token-row page table. Local offsets address logical positions in the
    per-token context; global slots address physical cache rows.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of token rows.
    num_tokens: int

    # Required: number of top-k offsets per token row.
    topk: int

    # Required: physical page size used for local-offset to page/offset mapping.
    block_size: int

    # Required: page-table width per token row.
    max_pages_per_token: int

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: upper bound for generated per-token sequence lengths. Defaults
    # to max_pages_per_token * block_size.
    max_seq_len: int | None = None

    # Optional: physical-page assignment policy for the token-row block table.
    indexing: PageTableIndexing = "random"

    # Optional: nested page-table config. If supplied, its batch size must match
    # num_tokens; its width is raised as needed during generation.
    page_table_input: PageTableInputConfig | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass
class DSAInputValues:
    """Generated core values for dynamic sparse attention.

    The values describe the mathematical inputs shared by sparse decode KV
    packing, decode top-k selection, and local-offset to cache-slot mapping.
    Helper kernels can adapt these fields into their narrower argument lists.
    """

    cache_k_nope: torch.Tensor
    cache_k_rope: torch.Tensor
    packed_kv_out: torch.Tensor
    slot_mapping: torch.Tensor
    logits: torch.Tensor
    topk_out: torch.Tensor
    valid_lens: torch.Tensor
    local_topk_offsets: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    block_table_cpu: list[list[int]]
    block_table_values: PageTableValues
    block_size: int
    topk: int


@dataclass
class DSAInputConfig:
    """Initialization parameters for ``DSAInputs``.

    This config describes one dynamic sparse attention decode surface. It
    generates source K rows, physical write slots, masked indexer logits,
    selected local offsets, and page-table metadata. Helper kernels that only
    need one subset should adapt from the generated values.
    """

    # Required: number of token rows represented by the generated values.
    num_tokens: int

    # Required: number of physical output rows available for sparse KV packing.
    num_slots: int

    # Required: non-RoPE key width. Must be a power of two and divisible by 128.
    nope_dim: int

    # Required: RoPE key width. Must be a power of two.
    rope_dim: int

    # Required: maximum number of local context offsets in each logit row.
    vocab_size: int

    # Required: number of selected offsets per token row.
    topk: int

    # Required: physical page size for local-offset to page/offset mapping.
    block_size: int

    # Required: page-table width per token row.
    max_pages_per_token: int

    # Required: generated logit dtype.
    dtype: torch.dtype

    # Optional: include singleton KV-head axis for generated source K tensors.
    include_head_axis: bool = False

    # Optional: minimum generated valid context length. Defaults to topk.
    min_valid_len: int | None = None

    # Optional: maximum generated valid context length. Defaults to vocab_size.
    max_valid_len: int | None = None

    # Optional: plant a deterministic equal-logit boundary case.
    include_boundary_tie: bool = True

    # Optional: upper bound for generated per-token sequence lengths. Defaults
    # to max_pages_per_token * block_size.
    max_seq_len: int | None = None

    # Optional: physical-page assignment policy for the token-row page table.
    indexing: PageTableIndexing = "random"

    # Optional: nested page-table config override.
    page_table_input: PageTableInputConfig | None = None

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _DSATopKSlotBuilder:
    """Generator for dynamic sparse attention top-k slot metadata."""

    config: DSATopKSlotInputConfig
    page_table_input: PageTableInput | None

    def __init__(self, config: DSATopKSlotInputConfig) -> None:
        self.config = config
        self.page_table_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.page_table_input = self.page_table_input or PageTableInput(
            self.config.page_table_input or self._make_page_table_config()
        )
        self._verify_page_table_config_matches_parent()
        self.config.page_table_input = self.page_table_input.config

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> DSATopKSlotInputValues:
        self.__post_init__()
        if self.page_table_input is None:
            raise ValueError("page_table_input must be initialized")
        target_device = _resolve_device(self.config.device, device)
        self.page_table_input.config.batch_size = max(1, self.config.num_tokens)
        self.page_table_input.config.max_pages_per_request = max(
            self.page_table_input.config.max_pages_per_request,
            self.config.max_pages_per_token,
        )
        self.page_table_input.config.indexing = self.config.indexing
        page_table_values = self.page_table_input.generate(
            seed=_child_seed(seed, 1),
            device=target_device,
        )
        max_context_len = page_table_values.page_table.shape[1] * self.config.block_size
        max_seq_len = (
            max_context_len
            if self.config.max_seq_len is None
            else min(self.config.max_seq_len, max_context_len)
        )
        seq_lens_cpu = self._generate_seq_lens(
            seed=_child_seed(seed, 2),
            max_seq_len=max_seq_len,
        )
        local_topk_cpu = self._generate_local_topk_offsets(
            seed=_child_seed(seed, 3),
            seq_lens=seq_lens_cpu,
        )

        return DSATopKSlotInputValues(
            local_topk_offsets=local_topk_cpu.to(target_device),
            seq_lens=torch.tensor(
                seq_lens_cpu, dtype=torch.int32, device=target_device
            ),
            block_table=page_table_values.page_table,
            block_table_cpu=page_table_values.page_table_cpu,
            block_table_values=page_table_values,
            block_size=self.config.block_size,
            topk=self.config.topk,
        )

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.topk = _check_positive("topk", self.config.topk)
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.max_pages_per_token = _check_positive(
            "max_pages_per_token", self.config.max_pages_per_token
        )
        if self.config.max_seq_len is not None:
            self.config.max_seq_len = _check_positive(
                "max_seq_len", self.config.max_seq_len
            )
        self.config.indexing = _check_page_table_indexing(self.config.indexing)

    def _make_page_table_config(self) -> PageTableInputConfig:
        return PageTableInputConfig(
            batch_size=max(1, self.config.num_tokens),
            max_pages_per_request=self.config.max_pages_per_token,
            indexing=self.config.indexing,
            device=self.config.device,
        )

    def _verify_page_table_config_matches_parent(self) -> None:
        if self.page_table_input is None:
            raise ValueError("page_table_input must be initialized")
        _check_matches(
            parent_name="DSATopKSlotInputConfig.num_tokens",
            child_name="page_table_input.batch_size",
            parent_value=max(1, self.config.num_tokens),
            child_value=self.page_table_input.config.batch_size,
        )

    def _generate_seq_lens(self, *, seed: int, max_seq_len: int) -> list[int]:
        if self.config.num_tokens == 0:
            return []
        rng = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(
            1,
            max_seq_len + 1,
            (self.config.num_tokens,),
            dtype=torch.int32,
            generator=rng,
        ).tolist()

    def _generate_local_topk_offsets(
        self,
        *,
        seed: int,
        seq_lens: list[int],
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        local_topk = torch.full(
            (self.config.num_tokens, self.config.topk),
            -1,
            dtype=torch.int32,
        )
        for token_idx, seq_len in enumerate(seq_lens):
            valid_count = int(
                torch.randint(
                    1,
                    min(int(seq_len), self.config.topk) + 1,
                    (1,),
                    generator=rng,
                ).item()
            )
            selected = torch.randperm(int(seq_len), generator=rng)[:valid_count].to(
                torch.int32
            )
            local_topk[token_idx, :valid_count] = selected
        return local_topk


def _validate_dsa_topk_slot_values(values: DSATopKSlotInputValues) -> None:
    if values.local_topk_offsets.dtype != torch.int32:
        raise TypeError(
            f"local_topk_offsets must be int32, got {values.local_topk_offsets.dtype}"
        )
    if values.seq_lens.dtype != torch.int32:
        raise TypeError(f"seq_lens must be int32, got {values.seq_lens.dtype}")
    if values.block_table.dtype != torch.int32:
        raise TypeError(f"block_table must be int32, got {values.block_table.dtype}")
    if values.local_topk_offsets.ndim != 2:
        raise ValueError("local_topk_offsets must be rank-2")
    num_tokens, topk = values.local_topk_offsets.shape
    if values.topk != topk:
        raise ValueError("topk must match local_topk_offsets width")
    if values.seq_lens.shape != (num_tokens,):
        raise ValueError("seq_lens must have shape [num_tokens]")
    if values.block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    if values.block_table.shape[0] < num_tokens:
        raise ValueError("block_table must have at least one row per token")
    if values.block_table.shape[1] <= 0:
        raise ValueError("block_table must have at least one page column")
    if values.block_size <= 0:
        raise ValueError("block_size must be positive")
    if values.topk <= 0:
        raise ValueError("topk must be positive")
    if (
        values.local_topk_offsets.device != values.seq_lens.device
        or values.local_topk_offsets.device != values.block_table.device
    ):
        raise ValueError(
            "local_topk_offsets, seq_lens, and block_table must share device"
        )


def dsa_local_topk_to_global_slots_reference(
    values: DSATopKSlotInputValues,
    *,
    use_seq_lens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global cache slots for dynamic sparse attention local offsets."""

    _validate_dsa_topk_slot_values(values)
    local = values.local_topk_offsets
    device = local.device
    num_tokens, topk = local.shape
    global_slots = torch.full_like(local, -1)
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    block_table_cpu = values.block_table.cpu()
    seq_lens_cpu = values.seq_lens.cpu().tolist()
    max_context_len = values.block_table.shape[1] * values.block_size
    for token_idx in range(num_tokens):
        seq_len = int(seq_lens_cpu[token_idx]) if use_seq_lens else max_context_len
        count = 0
        for slot in range(topk):
            local_idx = int(local[token_idx, slot].item())
            if local_idx < 0 or local_idx >= seq_len:
                continue
            block_idx = local_idx // values.block_size
            if block_idx < 0 or block_idx >= values.block_table.shape[1]:
                continue
            block_offset = local_idx % values.block_size
            page = int(block_table_cpu[token_idx, block_idx].item())
            global_slots[token_idx, slot] = page * values.block_size + block_offset
            count += 1
        lens[token_idx] = count
    return global_slots, lens


def dsa_full_context_topk_to_global_slots_reference(
    values: DSATopKSlotInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global slots for the first top-k positions of each token context."""

    _validate_dsa_topk_slot_values(values)
    device = values.seq_lens.device
    num_tokens = int(values.seq_lens.numel())
    topk = int(values.topk)
    global_slots = torch.full(
        (num_tokens, topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    block_table_cpu = values.block_table.cpu()
    seq_lens_cpu = values.seq_lens.cpu().tolist()
    max_context_len = values.block_table.shape[1] * values.block_size
    for token_idx, seq_len in enumerate(seq_lens_cpu):
        capped_seq_len = min(int(seq_len), max_context_len)
        lens[token_idx] = min(capped_seq_len, topk)
        for offset in range(topk):
            block_idx = offset // values.block_size
            if offset >= int(seq_len) or block_idx >= values.block_table.shape[1]:
                continue
            block_offset = offset % values.block_size
            page = int(block_table_cpu[token_idx, block_idx].item())
            global_slots[token_idx, offset] = page * values.block_size + block_offset
    return global_slots, lens


@dataclass

class DSAInputs(NumericsInputGenerator):
    """Family generator for dynamic sparse attention inputs."""

    config: DSAInputConfig
    cache_k_nope_input: TensorInput | None
    cache_k_rope_input: TensorInput | None
    page_table_input: PageTableInput | None

    def __init__(self, config: DSAInputConfig) -> None:
        self.config = config
        self.cache_k_nope_input = None
        self.cache_k_rope_input = None
        self.page_table_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.cache_k_nope_input = self.cache_k_nope_input or TensorInput(
            self._source_shape(self.config.nope_dim),
            torch.bfloat16,
            device=self.config.device,
        )
        self.cache_k_rope_input = self.cache_k_rope_input or TensorInput(
            self._source_shape(self.config.rope_dim),
            torch.bfloat16,
            device=self.config.device,
        )
        self.page_table_input = self.page_table_input or PageTableInput(
            self.config.page_table_input or self._make_page_table_config()
        )
        self._verify_page_table_config_matches_parent()
        self.config.page_table_input = self.page_table_input.config

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> DSAInputValues:
        self.__post_init__()
        if (
            self.cache_k_nope_input is None
            or self.cache_k_rope_input is None
            or self.page_table_input is None
        ):
            raise ValueError("DSAInputs child generators must be initialized")
        target_device = _resolve_device(self.config.device, device)
        metadata_seed = seed if metadata_seed is None else metadata_seed

        self.cache_k_nope_input.shape = self._source_shape(self.config.nope_dim)
        self.cache_k_nope_input.dtype = torch.bfloat16
        self.cache_k_nope_input.device = self.config.device
        self.cache_k_rope_input.shape = self._source_shape(self.config.rope_dim)
        self.cache_k_rope_input.dtype = torch.bfloat16
        self.cache_k_rope_input.device = self.config.device

        cache_k_nope = _require_tensor(
            self.cache_k_nope_input.generate(
                seed=_child_seed(seed, 1),
                device=target_device,
            ).values,
            "cache_k_nope",
        )
        cache_k_rope = _require_tensor(
            self.cache_k_rope_input.generate(
                seed=_child_seed(seed, 2),
                device=target_device,
            ).values,
            "cache_k_rope",
        )
        packed_kv_out = self._generate_packed_kv_out(
            seed=_child_seed(seed, 3),
            device=target_device,
        )
        slot_mapping = self._generate_slot_mapping(
            seed=_child_seed(metadata_seed, 4),
            device=target_device,
        )
        logits, valid_lens = self._generate_logits_and_lens(
            seed=_child_seed(seed, 5),
            metadata_seed=_child_seed(metadata_seed, 5),
            device=target_device,
        )
        topk_out = torch.full(
            (self.config.num_tokens, self.config.topk),
            -1,
            dtype=torch.int32,
            device=target_device,
        )
        page_table_values = self._generate_page_table(
            seed=_child_seed(metadata_seed, 6),
            device=target_device,
        )
        seq_lens_cpu = self._generate_seq_lens(
            seed=_child_seed(metadata_seed, 7),
            max_seq_len=self._max_generated_seq_len(page_table_values),
        )
        local_topk_offsets = self._generate_local_topk_offsets(
            seed=_child_seed(metadata_seed, 8),
            seq_lens=seq_lens_cpu,
        ).to(target_device)

        values = DSAInputValues(
            cache_k_nope=cache_k_nope,
            cache_k_rope=cache_k_rope,
            packed_kv_out=packed_kv_out,
            slot_mapping=slot_mapping,
            logits=logits,
            topk_out=topk_out,
            valid_lens=valid_lens,
            local_topk_offsets=local_topk_offsets,
            seq_lens=torch.tensor(
                seq_lens_cpu,
                dtype=torch.int32,
                device=target_device,
            ),
            block_table=page_table_values.page_table,
            block_table_cpu=page_table_values.page_table_cpu,
            block_table_values=page_table_values,
            block_size=self.config.block_size,
            topk=self.config.topk,
        )
        self._validate_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens",
            self.config.num_tokens,
        )
        self.config.num_slots = _check_positive("num_slots", self.config.num_slots)
        if self.config.num_tokens > self.config.num_slots:
            raise ValueError(
                "num_tokens must be <= num_slots so generated slot locations "
                f"are unique, got num_tokens={self.config.num_tokens}, "
                f"num_slots={self.config.num_slots}"
            )
        self.config.nope_dim = _check_power_of_2("nope_dim", self.config.nope_dim)
        if self.config.nope_dim % _DSA_SPARSE_DECODE_FP8_QUANT_BLOCK != 0:
            raise ValueError(
                "nope_dim must be divisible by "
                f"{_DSA_SPARSE_DECODE_FP8_QUANT_BLOCK}, got {self.config.nope_dim}"
            )
        self.config.rope_dim = _check_power_of_2("rope_dim", self.config.rope_dim)
        self.config.vocab_size = _check_positive("vocab_size", self.config.vocab_size)
        self.config.topk = _check_positive("topk", self.config.topk)
        if self.config.topk > self.config.vocab_size:
            raise ValueError("topk must be <= vocab_size")
        self.config.block_size = _check_positive("block_size", self.config.block_size)
        self.config.max_pages_per_token = _check_positive(
            "max_pages_per_token",
            self.config.max_pages_per_token,
        )
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        min_valid_len = (
            self.config.topk
            if self.config.min_valid_len is None
            else _check_positive("min_valid_len", self.config.min_valid_len)
        )
        max_valid_len = (
            self.config.vocab_size
            if self.config.max_valid_len is None
            else _check_positive("max_valid_len", self.config.max_valid_len)
        )
        if min_valid_len < self.config.topk:
            raise ValueError("min_valid_len must be >= topk")
        if max_valid_len > self.config.vocab_size:
            raise ValueError("max_valid_len must be <= vocab_size")
        if min_valid_len > max_valid_len:
            raise ValueError("min_valid_len must be <= max_valid_len")
        self.config.min_valid_len = min_valid_len
        self.config.max_valid_len = max_valid_len
        if self.config.max_seq_len is not None:
            self.config.max_seq_len = _check_positive(
                "max_seq_len",
                self.config.max_seq_len,
            )
        self.config.indexing = _check_page_table_indexing(self.config.indexing)

    def _source_shape(self, dim: int) -> tuple[int, ...]:
        if self.config.include_head_axis:
            return (self.config.num_tokens, 1, int(dim))
        return (self.config.num_tokens, int(dim))

    def _make_page_table_config(self) -> PageTableInputConfig:
        return PageTableInputConfig(
            batch_size=max(1, self.config.num_tokens),
            max_pages_per_request=self.config.max_pages_per_token,
            indexing=self.config.indexing,
            device=self.config.device,
        )

    def _verify_page_table_config_matches_parent(self) -> None:
        if self.page_table_input is None:
            raise ValueError("page_table_input must be initialized")
        _check_matches(
            parent_name="DSAInputConfig.num_tokens",
            child_name="page_table_input.batch_size",
            parent_value=max(1, self.config.num_tokens),
            child_value=self.page_table_input.config.batch_size,
        )

    def _generate_packed_kv_out(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        generator = _rng_for_device(device, seed)
        row_bytes = dsa_sparse_decode_row_bytes(
            self.config.nope_dim,
            self.config.rope_dim,
        )
        return torch.randint(
            0,
            256,
            (self.config.num_slots, row_bytes),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    def _generate_slot_mapping(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randperm(
            self.config.num_slots,
            dtype=torch.int64,
            generator=generator,
        )[: self.config.num_tokens].to(device)

    def _generate_logits_and_lens(
        self,
        *,
        seed: int,
        metadata_seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value_generator = _rng_for_device(device, seed)
        logits = -1.0 - 3.0 * torch.rand(
            (self.config.num_tokens, self.config.vocab_size),
            dtype=torch.float32,
            device=device,
            generator=value_generator,
        )
        valid_lens = self._generate_valid_lens(seed=metadata_seed).to(device)
        for row_idx, valid_len_tensor in enumerate(valid_lens.cpu()):
            valid_len = int(valid_len_tensor.item())
            if valid_len < self.config.vocab_size:
                logits[row_idx, valid_len:] = -float("inf")
            self._plant_topk_row(logits[row_idx], valid_len=valid_len)
        return logits.to(self.config.dtype), valid_lens

    def _generate_valid_lens(self, *, seed: int) -> torch.Tensor:
        if self.config.num_tokens == 0:
            return torch.empty((0,), dtype=torch.int32)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(
            int(self.config.min_valid_len),
            int(self.config.max_valid_len) + 1,
            (self.config.num_tokens,),
            dtype=torch.int32,
            generator=rng,
        )

    def _plant_topk_row(self, row: torch.Tensor, *, valid_len: int) -> None:
        if valid_len <= 0:
            return
        if self.config.include_boundary_tie and valid_len > self.config.topk:
            for rank in range(self.config.topk - 1):
                row[rank] = 32.0 - float(rank)
            row[valid_len - 2] = 1.0
            row[valid_len - 1] = 1.0
            return
        for rank in range(self.config.topk):
            row[rank] = 32.0 - float(rank)

    def _generate_page_table(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> PageTableValues:
        if self.page_table_input is None:
            raise ValueError("page_table_input must be initialized")
        self.page_table_input.config.batch_size = max(1, self.config.num_tokens)
        self.page_table_input.config.max_pages_per_request = max(
            self.page_table_input.config.max_pages_per_request,
            self.config.max_pages_per_token,
        )
        self.page_table_input.config.indexing = self.config.indexing
        return self.page_table_input.generate(seed=seed, device=device)

    def _max_generated_seq_len(self, page_table_values: PageTableValues) -> int:
        max_context_len = page_table_values.page_table.shape[1] * self.config.block_size
        if self.config.max_seq_len is None:
            return max_context_len
        return min(self.config.max_seq_len, max_context_len)

    def _generate_seq_lens(self, *, seed: int, max_seq_len: int) -> list[int]:
        if self.config.num_tokens == 0:
            return []
        rng = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(
            1,
            max_seq_len + 1,
            (self.config.num_tokens,),
            dtype=torch.int32,
            generator=rng,
        ).tolist()

    def _generate_local_topk_offsets(
        self,
        *,
        seed: int,
        seq_lens: list[int],
    ) -> torch.Tensor:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        local_topk = torch.full(
            (self.config.num_tokens, self.config.topk),
            -1,
            dtype=torch.int32,
        )
        for token_idx, seq_len in enumerate(seq_lens):
            valid_count = int(
                torch.randint(
                    1,
                    min(int(seq_len), self.config.topk) + 1,
                    (1,),
                    generator=rng,
                ).item()
            )
            selected = torch.randperm(int(seq_len), generator=rng)[:valid_count].to(
                torch.int32
            )
            local_topk[token_idx, :valid_count] = selected
        return local_topk

    def _validate_values(self, values: DSAInputValues) -> None:
        _validate_dsa_sparse_decode_kv_pack_values(
            DSASparseDecodeKVPackInputValues(
                out=values.packed_kv_out,
                loc=values.slot_mapping,
                cache_k_nope=values.cache_k_nope,
                cache_k_rope=values.cache_k_rope,
            )
        )
        _validate_dsa_decode_topk_values(
            DSADecodeTopKInputValues(
                logits=values.logits,
                out=values.topk_out,
                valid_lens=values.valid_lens,
                topk=values.topk,
            )
        )
        _validate_dsa_topk_slot_values(
            DSATopKSlotInputValues(
                local_topk_offsets=values.local_topk_offsets,
                seq_lens=values.seq_lens,
                block_table=values.block_table,
                block_table_cpu=values.block_table_cpu,
                block_table_values=values.block_table_values,
                block_size=values.block_size,
                topk=values.topk,
            )
        )
