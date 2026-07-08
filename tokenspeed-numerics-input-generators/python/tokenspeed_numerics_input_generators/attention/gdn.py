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
    LengthMode,
    MHARequestMetadataInput,
    MHARequestMetadataInputConfig,
    MHARequestMetadataValues,
    NumericsInputGenerator,
    TensorInput,
    _GDN_CHUNK_SIZE,
    _check_float_dtype,
    _check_positive,
    _child_seed,
    _require_tensor,
    _resolve_attention_seeds,
    _resolve_device,
    dataclass,
    torch,
)


@dataclass
class GDNChunkPrefillInputValues:
    """Generated values for gated-delta-rule chunked prefill.

    The operation consumes a flattened prompt stream partitioned by
    ``cu_seqlens``. Q and K are already L2-normalized, ``g`` is the log-space
    state decay, and ``beta`` gates the delta update written into the recurrent
    state.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens_cpu: list[int]
    scale: float | None
    output_h: bool


@dataclass
class GDNChunkPrefillReferenceValues:
    """Reference outputs for gated-delta-rule chunked prefill."""

    out: torch.Tensor
    final_state: torch.Tensor
    state_checkpoints: torch.Tensor | None = None
    checkpoint_cu_starts: torch.Tensor | None = None


@dataclass
class GDNChunkPrefillInputConfig:
    """Initialization parameters for GDN chunked prefill.

    This generator represents the prompt-side recurrent scan for Gated
    DeltaNet-style linear attention. It generates normalized Q/K rows, V rows,
    log-space decay gates, beta update gates, initial recurrent state, and
    cumulative sequence metadata for a flattened variable-length prompt stream.
    """

    # ------------------------------------------------------------------
    # Required configuration fields.
    # ------------------------------------------------------------------

    # Required: number of independent prompt sequences in cu_seqlens.
    batch_size: int

    # Required: total flattened prefill tokens across all sequences.
    total_tokens: int

    # Required: number of query/key heads. K shares the Q head count.
    num_q_heads: int

    # Required: number of value/state heads. Must be >= num_q_heads and an
    # integer multiple of num_q_heads for the generated GVA mapping.
    num_v_heads: int

    # Required: per-head Q/K/V dimension.
    head_dim: int

    # Required: generated dtype for Q, K, and V.
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Optional request-layout configuration.
    # ------------------------------------------------------------------

    # Optional: split policy for total_tokens across batch_size sequences.
    length_mode: LengthMode = "ragged"

    # Optional: upper bound for generated tokens in one sequence.
    max_tokens_per_sequence: int | None = None

    # Optional: include a leading singleton batch axis accepted by TokenSpeed's
    # wrapper. The operation semantics remain a flattened cu_seqlens stream.
    include_batch_dim: bool = False

    # ------------------------------------------------------------------
    # Optional recurrence/value generation configuration.
    # ------------------------------------------------------------------

    # Optional: explicit output scale. Defaults to 1 / sqrt(head_dim).
    scale: float | None = None

    # Optional: dtype for the generated initial recurrent state.
    initial_state_dtype: torch.dtype = torch.float32

    # Optional: factor applied to the generated initial state to keep the
    # recurrence in a numerically useful range.
    initial_state_scale: float = 0.05

    # Optional: lower bound for exp(g), where g is generated in log space.
    min_decay: float = 0.75

    # Optional: upper bound for exp(g), where g is generated in log space.
    max_decay: float = 0.99

    # Optional: epsilon used when normalizing generated Q and K rows.
    l2norm_eps: float = 1.0e-6

    # Optional: request intermediate recurrent-state checkpoints after each
    # full 64-token chunk in every sequence.
    output_h: bool = False

    # Optional: generated tensor device override.
    device: DeviceLike = None


GDNInputValues = GDNChunkPrefillInputValues
GDNInputConfig = GDNChunkPrefillInputConfig


@dataclass(init=False)
class GDNInputs(NumericsInputGenerator):
    """Generator for core Gated DeltaNet recurrent attention inputs."""

    config: GDNInputConfig
    q_input: TensorInput | None
    k_input: TensorInput | None
    v_input: TensorInput | None
    initial_state_input: TensorInput | None

    def __init__(self, config: GDNInputConfig) -> None:
        self.config = config
        self.q_input = None
        self.k_input = None
        self.v_input = None
        self.initial_state_input = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self._normalize_config()
        self.q_input = self.q_input or TensorInput(
            self._qk_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.k_input = self.k_input or TensorInput(
            self._qk_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.v_input = self.v_input or TensorInput(
            self._v_shape(),
            self.config.dtype,
            device=self.config.device,
        )
        self.initial_state_input = self.initial_state_input or TensorInput(
            self._state_shape(),
            self.config.initial_state_dtype,
            device=self.config.device,
        )

    def generate(
        self,
        *,
        seed: int | None = None,
        metadata_seed: int | None = None,
        value_seed: int | None = None,
        device: DeviceLike = None,
    ) -> GDNChunkPrefillInputValues:
        self.__post_init__()
        if (
            self.q_input is None
            or self.k_input is None
            or self.v_input is None
            or self.initial_state_input is None
        ):
            raise ValueError("GDNInputs child generators must be initialized")
        metadata_seed, value_seed = _resolve_attention_seeds(
            seed=seed,
            metadata_seed=metadata_seed,
            value_seed=value_seed,
        )
        target_device = _resolve_device(self.config.device, device)
        metadata = self._generate_metadata(
            seed=metadata_seed,
            device=target_device,
        )

        self.q_input.shape = self._qk_shape()
        self.q_input.dtype = self.config.dtype
        self.k_input.shape = self._qk_shape()
        self.k_input.dtype = self.config.dtype
        self.v_input.shape = self._v_shape()
        self.v_input.dtype = self.config.dtype
        self.initial_state_input.shape = self._state_shape()
        self.initial_state_input.dtype = self.config.initial_state_dtype

        q = self._normalize_qk(
            _require_tensor(
                self.q_input.generate(
                    seed=_child_seed(value_seed, 1),
                    device=target_device,
                ).values,
                "q",
            )
        )
        k = self._normalize_qk(
            _require_tensor(
                self.k_input.generate(
                    seed=_child_seed(value_seed, 2),
                    device=target_device,
                ).values,
                "k",
            )
        )
        v = _require_tensor(
            self.v_input.generate(
                seed=_child_seed(value_seed, 3),
                device=target_device,
            ).values,
            "v",
        ).contiguous()
        initial_state = _require_tensor(
            self.initial_state_input.generate(
                seed=_child_seed(value_seed, 4),
                device=target_device,
            ).values,
            "initial_state",
        )
        initial_state = (
            initial_state.float() * float(self.config.initial_state_scale)
        ).to(self.config.initial_state_dtype)
        g, beta = self._generate_gates(
            seed=_child_seed(value_seed, 5),
            device=target_device,
        )

        if self.config.include_batch_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

        values = GDNChunkPrefillInputValues(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            initial_state=initial_state.contiguous(),
            cu_seqlens=metadata.cu_seqlens_q.contiguous(),
            seq_lens_cpu=list(metadata.new_q_lens_cpu),
            scale=self.config.scale,
            output_h=bool(self.config.output_h),
        )
        _validate_gdn_chunk_prefill_values(values)
        return values

    def _normalize_config(self) -> None:
        self.config.batch_size = _check_positive("batch_size", self.config.batch_size)
        self.config.total_tokens = _check_positive(
            "total_tokens",
            self.config.total_tokens,
        )
        if self.config.total_tokens < self.config.batch_size:
            raise ValueError(
                "total_tokens must be >= batch_size so every sequence has at "
                f"least one token, got total_tokens={self.config.total_tokens}, "
                f"batch_size={self.config.batch_size}"
            )
        self.config.num_q_heads = _check_positive(
            "num_q_heads",
            self.config.num_q_heads,
        )
        self.config.num_v_heads = _check_positive(
            "num_v_heads",
            self.config.num_v_heads,
        )
        if self.config.num_v_heads < self.config.num_q_heads:
            raise ValueError(
                "num_v_heads must be >= num_q_heads for the supported GVA/equal-head "
                f"mapping, got num_v_heads={self.config.num_v_heads}, "
                f"num_q_heads={self.config.num_q_heads}"
            )
        if self.config.num_v_heads % self.config.num_q_heads != 0:
            raise ValueError(
                "num_v_heads must be an integer multiple of num_q_heads, got "
                f"num_v_heads={self.config.num_v_heads}, "
                f"num_q_heads={self.config.num_q_heads}"
            )
        self.config.head_dim = _check_positive("head_dim", self.config.head_dim)
        self.config.dtype = _check_float_dtype("dtype", self.config.dtype)
        self.config.initial_state_dtype = _check_float_dtype(
            "initial_state_dtype",
            self.config.initial_state_dtype,
        )
        if self.config.length_mode not in ("ragged", "regular", "fixed_per_request"):
            raise ValueError(
                "length_mode must be 'ragged', 'regular', or 'fixed_per_request', "
                f"got {self.config.length_mode!r}"
            )
        if self.config.max_tokens_per_sequence is not None:
            self.config.max_tokens_per_sequence = _check_positive(
                "max_tokens_per_sequence",
                self.config.max_tokens_per_sequence,
            )
        if self.config.scale is not None:
            self.config.scale = float(self.config.scale)
            if self.config.scale <= 0.0:
                raise ValueError(f"scale must be positive, got {self.config.scale}")
        self.config.initial_state_scale = float(self.config.initial_state_scale)
        if self.config.initial_state_scale < 0.0:
            raise ValueError(
                "initial_state_scale must be non-negative, got "
                f"{self.config.initial_state_scale}"
            )
        self.config.min_decay = float(self.config.min_decay)
        self.config.max_decay = float(self.config.max_decay)
        if not (0.0 < self.config.min_decay <= self.config.max_decay <= 1.0):
            raise ValueError(
                "decay bounds must satisfy 0 < min_decay <= max_decay <= 1, got "
                f"min_decay={self.config.min_decay}, "
                f"max_decay={self.config.max_decay}"
            )
        self.config.l2norm_eps = float(self.config.l2norm_eps)
        if self.config.l2norm_eps <= 0.0:
            raise ValueError(
                f"l2norm_eps must be positive, got {self.config.l2norm_eps}"
            )
        self.config.include_batch_dim = bool(self.config.include_batch_dim)
        self.config.output_h = bool(self.config.output_h)

    def _qk_shape(self) -> tuple[int, int, int]:
        return (
            self.config.total_tokens,
            self.config.num_q_heads,
            self.config.head_dim,
        )

    def _v_shape(self) -> tuple[int, int, int]:
        return (
            self.config.total_tokens,
            self.config.num_v_heads,
            self.config.head_dim,
        )

    def _state_shape(self) -> tuple[int, int, int, int]:
        return (
            self.config.batch_size,
            self.config.num_v_heads,
            self.config.head_dim,
            self.config.head_dim,
        )

    def _generate_metadata(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> MHARequestMetadataValues:
        metadata_input = MHARequestMetadataInput(
            MHARequestMetadataInputConfig(
                batch_size=self.config.batch_size,
                total_cached_tokens=0,
                total_new_q_tokens=self.config.total_tokens,
                new_q_length_mode=self.config.length_mode,
                max_new_q_tokens_per_request=self.config.max_tokens_per_sequence,
                cache_layout="none",
                device=self.config.device,
            )
        )
        return metadata_input.generate(seed=seed, device=device)

    def _normalize_qk(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_f32 = tensor.float()
        norm = torch.sqrt(
            (tensor_f32 * tensor_f32).sum(dim=-1, keepdim=True) + self.config.l2norm_eps
        )
        return (tensor_f32 / norm).to(tensor.dtype).contiguous()

    def _generate_gates(
        self,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        gate_shape = (
            self.config.total_tokens,
            self.config.num_v_heads,
        )
        alpha = torch.rand(
            gate_shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        alpha = alpha * (self.config.max_decay - self.config.min_decay)
        alpha = alpha + self.config.min_decay
        g = torch.log(alpha)
        beta = torch.rand(
            gate_shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        beta = beta * 0.9 + 0.05
        return g.contiguous(), beta.contiguous()


def _gdn_as_3d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} rank-4 form must have leading batch size 1")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be rank-3 or rank-4, got {tensor.ndim}")
    return tensor.contiguous()


def _gdn_gate_as_2d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} rank-3 form must have leading batch size 1")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2 or rank-3, got {tensor.ndim}")
    return tensor.contiguous()


def _validate_gdn_chunk_prefill_values(
    values: GDNChunkPrefillInputValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _gdn_as_3d("q", values.q)
    k = _gdn_as_3d("k", values.k)
    v = _gdn_as_3d("v", values.v)
    g = _gdn_gate_as_2d("g", values.g)
    beta = _gdn_gate_as_2d("beta", values.beta)

    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have matching shapes, got {q.shape} and {k.shape}"
        )
    if q.shape[0] != v.shape[0]:
        raise ValueError("q/k and v must have matching token dimensions")
    if q.shape[-1] != v.shape[-1]:
        raise ValueError("q/k and v must have matching head_dim")
    num_q_heads = q.shape[1]
    num_v_heads = v.shape[1]
    if num_v_heads < num_q_heads:
        raise ValueError("num_v_heads must be >= num_q_heads")
    if num_v_heads % num_q_heads != 0:
        raise ValueError("num_v_heads must be an integer multiple of num_q_heads")
    if g.shape != (v.shape[0], num_v_heads):
        raise ValueError(
            f"g must have shape {(v.shape[0], num_v_heads)}, got {tuple(g.shape)}"
        )
    if beta.shape != g.shape:
        raise ValueError(
            f"beta must have shape {tuple(g.shape)}, got {tuple(beta.shape)}"
        )
    if values.initial_state.shape != (
        values.cu_seqlens.numel() - 1,
        num_v_heads,
        q.shape[-1],
        v.shape[-1],
    ):
        raise ValueError(
            "initial_state must have shape "
            f"{(values.cu_seqlens.numel() - 1, num_v_heads, q.shape[-1], v.shape[-1])}, "
            f"got {tuple(values.initial_state.shape)}"
        )
    if values.cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be rank-1, got {values.cu_seqlens.ndim}")
    if values.cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"cu_seqlens must use an integer dtype, got {values.cu_seqlens.dtype}"
        )
    cu = values.cu_seqlens.detach().cpu().to(torch.int64)
    if cu.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if int(cu[0].item()) != 0:
        raise ValueError("cu_seqlens must start at zero")
    if int(cu[-1].item()) != q.shape[0]:
        raise ValueError(
            f"cu_seqlens must end at total token count {q.shape[0]}, "
            f"got {int(cu[-1].item())}"
        )
    if bool((cu[1:] < cu[:-1]).any().item()):
        raise ValueError("cu_seqlens must be monotonically nondecreasing")
    seq_lens = (cu[1:] - cu[:-1]).tolist()
    if values.seq_lens_cpu != [int(length) for length in seq_lens]:
        raise ValueError("seq_lens_cpu must match cu_seqlens differences")
    if any(length <= 0 for length in values.seq_lens_cpu):
        raise ValueError(
            "all generated GDN prefill sequences must have at least one token"
        )
    if values.scale is not None and values.scale <= 0.0:
        raise ValueError(f"scale must be positive, got {values.scale}")
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", values.initial_state),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if values.initial_state.device != q.device:
        raise ValueError("initial_state must be on the same device as q")
    if (
        k.device != q.device
        or v.device != q.device
        or g.device != q.device
        or beta.device != q.device
    ):
        raise ValueError("q, k, v, g, and beta must share a device")
    if values.cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be on the same device as q")
    return q, k, v, g, beta


def gdn_chunk_prefill_reference(
    values: GDNChunkPrefillInputValues,
) -> GDNChunkPrefillReferenceValues:
    """Return a sequential reference for GDN chunked prefill.

    The reference follows the recurrent delta-rule form for each sequence and
    value head:

    ```text
    state = exp(g_t) * state
    delta = beta_t * (v_t - k_t @ state)
    state = state + outer(k_t, delta)
    out_t = scale * (q_t @ state)
    ```
    """

    q, k, v, g, beta = _validate_gdn_chunk_prefill_values(values)
    scale = values.scale if values.scale is not None else q.shape[-1] ** -0.5
    state = values.initial_state.float().clone()
    out = torch.empty_like(v)
    heads_per_q = v.shape[1] // q.shape[1]
    cu = values.cu_seqlens.detach().cpu().to(torch.int64).tolist()

    checkpoint_counts: list[int] = []
    checkpoints: list[torch.Tensor] = []
    for seq_idx, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=True)):
        start = int(start)
        end = int(end)
        checkpoint_count = 0
        for local_idx, token_idx in enumerate(range(start, end), start=1):
            for v_head in range(v.shape[1]):
                q_head = v_head // heads_per_q
                state_view = state[seq_idx, v_head]
                state_view.mul_(torch.exp(g[token_idx, v_head].float()))
                k_vec = k[token_idx, q_head].float()
                v_vec = v[token_idx, v_head].float()
                read = torch.matmul(k_vec, state_view)
                delta = (v_vec - read) * beta[token_idx, v_head].float()
                state_view.add_(torch.outer(k_vec, delta))
                out[token_idx, v_head] = (
                    torch.matmul(q[token_idx, q_head].float(), state_view) * scale
                ).to(out.dtype)
            if values.output_h and local_idx % _GDN_CHUNK_SIZE == 0:
                checkpoints.append(state[seq_idx].clone())
                checkpoint_count += 1
        checkpoint_counts.append(checkpoint_count)

    if values.q.ndim == 4:
        out = out.unsqueeze(0)

    if not values.output_h:
        return GDNChunkPrefillReferenceValues(
            out=out.contiguous(),
            final_state=state.contiguous(),
        )

    if checkpoints:
        state_checkpoints = torch.stack(checkpoints, dim=0).contiguous()
    else:
        state_checkpoints = torch.empty(
            (0, v.shape[1], q.shape[-1], v.shape[-1]),
            dtype=torch.float32,
            device=q.device,
        )
    checkpoint_cu = [0]
    for count in checkpoint_counts:
        checkpoint_cu.append(checkpoint_cu[-1] + count)
    checkpoint_cu_starts = torch.tensor(
        checkpoint_cu,
        dtype=torch.int64,
        device=q.device,
    )
    return GDNChunkPrefillReferenceValues(
        out=out.contiguous(),
        final_state=state.contiguous(),
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
    )
