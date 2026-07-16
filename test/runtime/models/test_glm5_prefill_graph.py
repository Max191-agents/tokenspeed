from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.glm5 import (
    GlmDsaIndexerOutput,
    GlmMoeDsaAttention,
    _glm_dsa_real_token_count,
)


def _context(extend_lens, *, bs, num_extends, spec_width=1):
    backend = SimpleNamespace(
        chunked_prefill_metadata=(
            SimpleNamespace(extend_seq_lens_cpu=extend_lens)
            if extend_lens is not None
            else None
        ),
        forward_decode_metadata=None,
        spec_num_tokens=spec_width,
    )
    active_lens = (
        extend_lens.tolist()
        if isinstance(extend_lens, torch.Tensor)
        else (extend_lens or [])
    )
    input_num_tokens = sum(active_lens[:num_extends]) + max(bs - num_extends, 0) * (
        spec_width or 1
    )
    return ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(),
        bs=bs,
        num_extends=num_extends,
        input_num_tokens=input_num_tokens,
        forward_mode=ForwardMode.from_num_extends(num_extends, bs),
    )


@pytest.mark.parametrize(
    ("extend_lens", "bs", "num_extends", "spec_width", "expected"),
    [
        ([1], 1, 1, 1, 1),
        ([16], 1, 1, 1, 16),
        ([2, 3], 2, 2, 1, 5),
        ([2, 3], 4, 2, 1, 7),
        (torch.tensor([2, 3], dtype=torch.int32), 4, 2, 6, 17),
    ],
)
def test_glm_dsa_real_token_count_for_prefill_graph(
    extend_lens, bs, num_extends, spec_width, expected
):
    ctx = _context(
        extend_lens,
        bs=bs,
        num_extends=num_extends,
        spec_width=spec_width,
    )

    assert _glm_dsa_real_token_count(ctx) is None
    with active_forward(ctx):
        assert _glm_dsa_real_token_count(ctx) == expected


def test_glm_dsa_real_token_count_requires_prefill_metadata():
    ctx = _context(None, bs=1, num_extends=1)

    with active_forward(ctx), pytest.raises(
        RuntimeError, match="requires CPU extend-length metadata"
    ):
        _glm_dsa_real_token_count(ctx)


@pytest.mark.parametrize(
    ("extend_lens", "num_extends", "message"),
    [
        ([1], 2, "incomplete extend-length metadata"),
        ([-1], 1, "requires nonnegative extend lengths"),
    ],
)
def test_glm_dsa_real_token_count_rejects_invalid_lengths(
    extend_lens, num_extends, message
):
    ctx = _context(extend_lens, bs=num_extends, num_extends=num_extends)

    with active_forward(ctx), pytest.raises(RuntimeError, match=message):
        _glm_dsa_real_token_count(ctx)


class _CommRecorder:
    def __init__(self):
        self.rows = []

    def pre_attn_comm(self, tensor, ctx):
        self.rows.append(tensor.shape[0])
        return tensor


class _TokenPool:
    def __init__(self):
        self.cache_write_rows = None

    def set_index_k_buffer(self, layer_id, out_cache_loc, key):
        self.cache_write_rows = (out_cache_loc.shape[0], key.shape[0])


class _FakeAttention:
    q_lora_rank = 2
    kv_lora_rank = 2
    qk_rope_head_dim = 1
    num_local_heads = 1
    v_head_dim = 1
    skip_indexer_topk = False
    is_nextn = False
    attn_mqa = SimpleNamespace(layer_id=0)

    def __init__(self):
        self.indexer_rows = None
        self.prefill_rows = None

    def fused_qkv_a_proj_with_mqa(self, hidden_states, *args):
        return torch.zeros((hidden_states.shape[0], 5), dtype=hidden_states.dtype)

    def fused_qk_layernorm(self, *, input_q_a, input_kv_a, output_q_a):
        output_q_a.copy_(input_q_a)

    def indexer(self, hidden_states, q_norm, positions):
        self.indexer_rows = (
            hidden_states.shape[0],
            q_norm.shape[0],
            positions.shape[0],
        )
        rows = positions.shape[0]
        return GlmDsaIndexerOutput(
            query=torch.zeros((rows, 1, 1)),
            key=torch.zeros((rows, 1)),
            weights=torch.zeros((rows, 1)),
        )

    def _compute_prefill_topk_indices(self, indexer_output, ctx, num_tokens):
        assert indexer_output.query.shape[0] == num_tokens
        return object()

    def _compute_decode_topk_indices(self, indexer_output, ctx):
        raise AssertionError("pure prefill must not compute decode top-k")

    def q_b_proj(self, q_norm):
        return torch.zeros((q_norm.shape[0], 1)), None

    def forward_dsa_sparse_prefill(
        self,
        positions,
        q,
        latent_cache,
        ctx,
        out_cache_loc,
        output,
        *,
        prefill_topk,
    ):
        self.prefill_rows = tuple(
            tensor.shape[0]
            for tensor in (positions, q, latent_cache, out_cache_loc, output)
        )
        output.fill_(1)

    def o_proj(self, output):
        return output, None

    _resolve_decode_window = staticmethod(GlmMoeDsaAttention._resolve_decode_window)


@pytest.mark.parametrize("real_rows", [1, 16])
def test_glm_dsa_prefill_graph_slices_after_both_collectives(real_rows):
    ctx = _context([real_rows], bs=1, num_extends=1)
    token_pool = _TokenPool()
    ctx.token_to_kv_pool = token_pool
    comm = _CommRecorder()
    attention = _FakeAttention()
    padded_rows = 16

    with active_forward(ctx):
        result = GlmMoeDsaAttention.forward.__wrapped__(
            attention,
            torch.arange(padded_rows),
            torch.zeros((padded_rows, 7)),
            ctx,
            torch.arange(padded_rows),
            comm,
        )

    assert comm.rows == [padded_rows, padded_rows]
    assert attention.indexer_rows == (real_rows, real_rows, real_rows)
    assert token_pool.cache_write_rows == (real_rows, real_rows)
    assert attention.prefill_rows == (real_rows,) * 5
    assert result.shape == (real_rows, 1)


def test_glm_dsa_prefill_token_mismatch_stays_strict():
    ctx = _context([1], bs=1, num_extends=1)
    ctx.attn_backend.chunked_prefill_metadata.extend_prefix_lens = torch.tensor([0])
    ctx.attn_backend.chunked_prefill_metadata.extend_seq_lens = torch.tensor([1])

    with pytest.raises(RuntimeError, match="metadata=1, tokens=2"):
        GlmMoeDsaAttention._compute_prefill_topk_indices(
            SimpleNamespace(), SimpleNamespace(), ctx, 2
        )
