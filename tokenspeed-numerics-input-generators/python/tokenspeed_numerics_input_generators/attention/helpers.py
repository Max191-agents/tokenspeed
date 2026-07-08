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
    TensorInput,
    _check_float_dtype,
    _check_nonnegative,
    _check_positive,
    _child_seed,
    _generate_packed_rows,
    _require_tensor,
    _resolve_device,
    dataclass,
    math,
    torch,
)


@dataclass
class AttentionMergeStateInputValues:
    """Generated values for merging two partial attention states."""

    out_a: torch.Tensor
    lse_a: torch.Tensor
    out_b: torch.Tensor
    lse_b: torch.Tensor
    lse_scale_log2: float


@dataclass
class AttentionMergeStateInputConfig:
    """Initialization parameters for attention merge-state inputs.

    The represented operation merges two partial attention outputs and their
    log-sum-exp values into one equivalent state. ``lse_scale_log2`` converts
    the LSE inputs into log2 space before the numerically stable merge.
    """

    # Required: total query rows being merged.
    total_q: int

    # Required: number of attention heads.
    num_heads: int

    # Required: per-head output dimension.
    head_dim: int

    # Optional: generated dtype for partial output tensors.
    dtype: torch.dtype = torch.bfloat16

    # Optional: scalar that maps input LSE values to log2 space. The default
    # corresponds to natural-log LSE inputs.
    lse_scale_log2: float = math.log2(math.e)

    # Optional: generated LSE values are sampled from [-lse_bound, lse_bound].
    lse_bound: float = 6.0

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _AttentionMergeStateBuilder:
    """Generator for attention merge-state inputs."""

    config: AttentionMergeStateInputConfig
    out_a_input: TensorInput | None
    out_b_input: TensorInput | None

    def __init__(self, config: AttentionMergeStateInputConfig) -> None:
        self.config = config
        self.out_a_input = None
        self.out_b_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.config.total_q = _check_nonnegative("total_q", self.config.total_q)
        self.config.num_heads = _check_positive("num_heads", self.config.num_heads)
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        if self.config.lse_scale_log2 <= 0.0:
            raise ValueError(
                f"lse_scale_log2 must be positive, got {self.config.lse_scale_log2}"
            )
        if self.config.lse_bound <= 0.0:
            raise ValueError(f"lse_bound must be positive, got {self.config.lse_bound}")
        output_shape = (
            self.config.total_q,
            self.config.num_heads,
            self.config.head_dim,
        )
        self.out_a_input = self.out_a_input or TensorInput(
            output_shape,
            self.config.dtype,
            device=self.config.device,
        )
        self.out_b_input = self.out_b_input or TensorInput(
            output_shape,
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        metadata_seed: int | None = None,
        device: DeviceLike = None,
    ) -> AttentionMergeStateInputValues:
        del metadata_seed
        self.__post_init__()
        if self.out_a_input is None or self.out_b_input is None:
            raise ValueError(
                "_AttentionMergeStateBuilder child generators must be initialized"
            )
        target_device = _resolve_device(self.config.device, device)
        lse_generator = torch.Generator(
            device="cuda" if target_device.type == "cuda" else "cpu"
        ).manual_seed(_child_seed(seed, 3))
        lse_shape = (self.config.total_q, self.config.num_heads)
        lse_a = (
            torch.rand(
                lse_shape,
                dtype=torch.float32,
                device=target_device,
                generator=lse_generator,
            )
            * (2.0 * self.config.lse_bound)
            - self.config.lse_bound
        )
        lse_b = (
            torch.rand(
                lse_shape,
                dtype=torch.float32,
                device=target_device,
                generator=lse_generator,
            )
            * (2.0 * self.config.lse_bound)
            - self.config.lse_bound
        )
        return AttentionMergeStateInputValues(
            out_a=_require_tensor(
                self.out_a_input.generate(
                    seed=_child_seed(seed, 1), device=device
                ).values,
                "out_a",
            ).contiguous(),
            lse_a=lse_a.contiguous(),
            out_b=_require_tensor(
                self.out_b_input.generate(
                    seed=_child_seed(seed, 2), device=device
                ).values,
                "out_b",
            ).contiguous(),
            lse_b=lse_b.contiguous(),
            lse_scale_log2=float(self.config.lse_scale_log2),
        )


def attention_merge_state_reference(
    values: AttentionMergeStateInputValues,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the merged partial attention output and LSE state."""

    if values.out_a.shape != values.out_b.shape:
        raise ValueError(
            f"out_a/out_b shape mismatch: {tuple(values.out_a.shape)} vs {tuple(values.out_b.shape)}"
        )
    if values.out_a.ndim != 3:
        raise ValueError(f"out tensors must be rank 3, got {values.out_a.ndim}")
    expected_lse_shape = values.out_a.shape[:2]
    if (
        values.lse_a.shape != expected_lse_shape
        or values.lse_b.shape != expected_lse_shape
    ):
        raise ValueError(
            "lse tensors must match out tensor leading dimensions; "
            f"expected {tuple(expected_lse_shape)}, "
            f"got {tuple(values.lse_a.shape)} and {tuple(values.lse_b.shape)}"
        )
    if values.lse_scale_log2 <= 0.0:
        raise ValueError(
            f"lse_scale_log2 must be positive, got {values.lse_scale_log2}"
        )
    lse_a_log2 = values.lse_a.to(torch.float32) * values.lse_scale_log2
    lse_b_log2 = values.lse_b.to(torch.float32) * values.lse_scale_log2
    lse_max_log2 = torch.maximum(lse_a_log2, lse_b_log2)
    weight_a = torch.exp2(lse_a_log2 - lse_max_log2)
    weight_b = torch.exp2(lse_b_log2 - lse_max_log2)
    denom = weight_a + weight_b
    out = (
        values.out_a.to(torch.float32) * weight_a[..., None]
        + values.out_b.to(torch.float32) * weight_b[..., None]
    ) / denom[..., None]
    lse = (lse_max_log2 + torch.log2(denom)) / values.lse_scale_log2
    return out.to(values.out_a.dtype), lse


@dataclass
class GDNQKVSplitInputValues:
    """Generated values for packed GDN QKV splitting."""

    mixed_qkv: torch.Tensor
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    head_q: int
    head_k: int
    head_v: int
    fuse_l2norm: bool
    l2norm_eps: float


@dataclass
class GDNQKVSplitInputConfig:
    """Initialization parameters for packed GDN QKV splitting.

    The represented operation splits a packed post-projection QKV row into
    contiguous query, key, and value tensors. When ``fuse_l2norm`` is true, Q
    and K are independently normalized over each head dimension before they are
    returned; V is always copied as-is.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of packed token rows.
    num_tokens: int

    # Required: number of Q heads in the packed Q segment.
    num_q_heads: int

    # Required: number of K heads in the packed K segment.
    num_k_heads: int

    # Required: number of V heads in the packed V segment.
    num_v_heads: int

    # Required: per-head Q dimension.
    head_q: int

    # Required: per-head K dimension.
    head_k: int

    # Required: per-head V dimension.
    head_v: int

    # Required: dtype for generated packed QKV values.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: normalize Q and K per head after splitting. This models the
    # fused fast path used by some GDN prefill implementations.
    fuse_l2norm: bool = False

    # Optional: epsilon used by the per-head L2 normalization.
    l2norm_eps: float = 1.0e-6

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _GDNQKVSplitBuilder:
    """Generator for packed GDN QKV split inputs."""

    config: GDNQKVSplitInputConfig
    mixed_qkv_input: TensorInput | None

    def __init__(self, config: GDNQKVSplitInputConfig) -> None:
        self.config = config
        self.mixed_qkv_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.mixed_qkv_input = self.mixed_qkv_input or TensorInput(
            (self.config.num_tokens, self._qkv_dim()),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> GDNQKVSplitInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        mixed_qkv = _generate_packed_rows(
            tensor_input=self.mixed_qkv_input,
            shape=(self.config.num_tokens, self._qkv_dim()),
            dtype=self.config.dtype,
            seed=seed,
            device=target_device,
            name="mixed_qkv",
        )
        return GDNQKVSplitInputValues(
            mixed_qkv=mixed_qkv,
            num_q_heads=self.config.num_q_heads,
            num_k_heads=self.config.num_k_heads,
            num_v_heads=self.config.num_v_heads,
            head_q=self.config.head_q,
            head_k=self.config.head_k,
            head_v=self.config.head_v,
            fuse_l2norm=bool(self.config.fuse_l2norm),
            l2norm_eps=float(self.config.l2norm_eps),
        )

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_q_heads = _check_positive(
            "num_q_heads", self.config.num_q_heads
        )
        self.config.num_k_heads = _check_positive(
            "num_k_heads", self.config.num_k_heads
        )
        self.config.num_v_heads = _check_positive(
            "num_v_heads", self.config.num_v_heads
        )
        self.config.head_q = _check_positive("head_q", self.config.head_q)
        self.config.head_k = _check_positive("head_k", self.config.head_k)
        self.config.head_v = _check_positive("head_v", self.config.head_v)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.l2norm_eps = float(self.config.l2norm_eps)
        if self.config.l2norm_eps <= 0.0:
            raise ValueError(
                f"l2norm_eps must be positive, got {self.config.l2norm_eps}"
            )

    def _qkv_dim(self) -> int:
        return (
            self.config.num_q_heads * self.config.head_q
            + self.config.num_k_heads * self.config.head_k
            + self.config.num_v_heads * self.config.head_v
        )


def gdn_qkv_split_reference(
    values: GDNQKVSplitInputValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return split Q/K/V tensors, optionally with per-head Q/K L2 norm."""

    if values.mixed_qkv.ndim != 2:
        raise ValueError(f"mixed_qkv must be rank-2, got {values.mixed_qkv.ndim}")
    for name, value in (
        ("num_q_heads", values.num_q_heads),
        ("num_k_heads", values.num_k_heads),
        ("num_v_heads", values.num_v_heads),
        ("head_q", values.head_q),
        ("head_k", values.head_k),
        ("head_v", values.head_v),
    ):
        _check_positive(name, value)
    if values.l2norm_eps <= 0.0:
        raise ValueError(f"l2norm_eps must be positive, got {values.l2norm_eps}")

    q_dim = values.num_q_heads * values.head_q
    k_dim = values.num_k_heads * values.head_k
    v_dim = values.num_v_heads * values.head_v
    qkv_dim = q_dim + k_dim + v_dim
    if values.mixed_qkv.shape[1] != qkv_dim:
        raise ValueError(
            f"mixed_qkv last dimension must be {qkv_dim}, "
            f"got {values.mixed_qkv.shape[1]}"
        )

    tokens = values.mixed_qkv.shape[0]
    q = values.mixed_qkv[:, :q_dim].reshape(tokens, values.num_q_heads, values.head_q)
    k = values.mixed_qkv[:, q_dim : q_dim + k_dim].reshape(
        tokens,
        values.num_k_heads,
        values.head_k,
    )
    v = values.mixed_qkv[:, q_dim + k_dim :].reshape(
        tokens,
        values.num_v_heads,
        values.head_v,
    )
    if values.fuse_l2norm:
        q_norm = torch.sqrt(
            (q.float() * q.float()).sum(dim=-1, keepdim=True) + values.l2norm_eps
        )
        k_norm = torch.sqrt(
            (k.float() * k.float()).sum(dim=-1, keepdim=True) + values.l2norm_eps
        )
        q = (q.float() / q_norm).to(values.mixed_qkv.dtype)
        k = (k.float() / k_norm).to(values.mixed_qkv.dtype)
    return (
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
    )


@dataclass

class PackedQKVComplexRotaryInputValues:
    """Generated values for packed QKV complex rotary splitting."""

    qkv: torch.Tensor
    freqs_cis: torch.Tensor
    num_heads: int
    head_dim: int
    copy_v: bool


@dataclass
class PackedQKVComplexRotaryInputConfig:
    """Initialization parameters for packed QKV complex RoPE.

    The represented operation splits a packed QKV tensor into equal-width Q, K,
    and V segments. Q and K are interpreted as pairs of real channels and are
    multiplied by unit complex rotary frequencies. V is not rotated.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of packed token rows.
    num_tokens: int

    # Required: number of Q/K/V heads. This utility expects all three packed
    # segments to use the same head count.
    num_heads: int

    # Required: per-head Q/K/V dimension. Must be even for complex pairs.
    head_dim: int

    # Required: dtype for generated packed QKV values.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional configuration fields.
    # ------------------------------------------------------------------

    # Optional: materialize V as an output tensor instead of returning a view of
    # the packed V segment. The mathematical values are the same either way.
    copy_v: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


@dataclass(init=False)
class _PackedQKVComplexRotaryBuilder:
    """Generator for packed QKV complex rotary inputs."""

    config: PackedQKVComplexRotaryInputConfig
    qkv_input: TensorInput | None

    def __init__(self, config: PackedQKVComplexRotaryInputConfig) -> None:
        self.config = config
        self.qkv_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.qkv_input = self.qkv_input or TensorInput(
            (self.config.num_tokens, self._packed_dim()),
            self.config.dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int,
        device: DeviceLike = None,
    ) -> PackedQKVComplexRotaryInputValues:
        self.__post_init__()
        target_device = _resolve_device(self.config.device, device)
        qkv = _generate_packed_rows(
            tensor_input=self.qkv_input,
            shape=(self.config.num_tokens, self._packed_dim()),
            dtype=self.config.dtype,
            seed=_child_seed(seed, 1),
            device=target_device,
            name="qkv",
        )
        freqs_cis = self._generate_freqs_cis(
            seed=_child_seed(seed, 2),
            device=target_device,
        )
        return PackedQKVComplexRotaryInputValues(
            qkv=qkv,
            freqs_cis=freqs_cis.contiguous(),
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            copy_v=bool(self.config.copy_v),
        )

    def _normalize_config(self) -> None:
        self.config.num_tokens = _check_nonnegative(
            "num_tokens", self.config.num_tokens
        )
        self.config.num_heads = _check_positive("num_heads", self.config.num_heads)
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        if self.config.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {self.config.head_dim}")
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)

    def _packed_dim(self) -> int:
        return 3 * self.config.num_heads * self.config.head_dim

    def _generate_freqs_cis(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        angles = (
            torch.rand(
                (self.config.num_tokens, self.config.head_dim // 2),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * (2.0 * math.pi)
            - math.pi
        )
        return torch.complex(torch.cos(angles), torch.sin(angles))


def _apply_complex_rotary(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    real = freqs_cis.real[:, None, :].float()
    imag = freqs_cis.imag[:, None, :].float()
    out = torch.empty_like(x.float())
    out[..., 0::2] = x_even * real - x_odd * imag
    out[..., 1::2] = x_odd * real + x_even * imag
    return out.to(x.dtype)


def packed_qkv_complex_rotary_reference(
    values: PackedQKVComplexRotaryInputValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q/K after complex RoPE and unrotated V."""

    if values.qkv.ndim != 2:
        raise ValueError(f"qkv must be rank-2, got {values.qkv.ndim}")
    _check_positive("num_heads", values.num_heads)
    _check_positive("head_dim", values.head_dim)
    if values.head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {values.head_dim}")
    if not values.freqs_cis.is_complex():
        raise TypeError("freqs_cis must be a complex tensor")
    if values.qkv.device != values.freqs_cis.device:
        raise ValueError("qkv and freqs_cis must be on the same device")
    total_tokens = values.qkv.shape[0]
    q_size = values.num_heads * values.head_dim
    kv_size = q_size
    packed_dim = q_size + 2 * kv_size
    if values.qkv.shape[1] != packed_dim:
        raise ValueError(
            f"qkv last dimension must be {packed_dim}, got {values.qkv.shape[1]}"
        )
    expected_freqs_shape = (total_tokens, values.head_dim // 2)
    if values.freqs_cis.shape != expected_freqs_shape:
        raise ValueError(
            f"freqs_cis must have shape {expected_freqs_shape}, "
            f"got {tuple(values.freqs_cis.shape)}"
        )

    q = values.qkv[:, :q_size].reshape(total_tokens, values.num_heads, values.head_dim)
    k = values.qkv[:, q_size : q_size + kv_size].reshape(
        total_tokens,
        values.num_heads,
        values.head_dim,
    )
    v = values.qkv[:, q_size + kv_size :].reshape(
        total_tokens,
        values.num_heads,
        values.head_dim,
    )
    q_out = _apply_complex_rotary(q, values.freqs_cis).contiguous()
    k_out = _apply_complex_rotary(k, values.freqs_cis).contiguous()
    v_out = v.clone().contiguous() if values.copy_v else v
    return q_out, k_out, v_out
