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

import pytest
import torch
from tokenspeed_numerics_input_generators import (
    AttentionMergeStateInputValues,
    CSAHistoryConfig,
    CSAIndexerConfig,
    CSAInputConfig,
    CSAInputs,
    DSADecodeTopKInputValues,
    DSAInputConfig,
    DSAInputValues,
    DSAInputs,
    DSASparseDecodeKVPackInputValues,
    DSATopKSlotInputValues,
    GDNInputConfig,
    GDNInputValues,
    GDNInputs,
    GDNQKVSplitInputValues,
    KVCacheInput,
    KVCacheInputConfig,
    MHAInputConfig,
    MHAInputs,
    MHAInputValues,
    MHARequestMetadataInputConfig,
    MLAInputConfig,
    MLAInputs,
    MLAKVCacheInput,
    MLAKVCacheInputConfig,
    PackedQKVComplexRotaryInputValues,
    PageTableInput,
    PageTableInputConfig,
    SlotMappingInput,
    SlotMappingInputConfig,
    attention_merge_state_reference,
    deepseek_v4_combine_dense_swa_indices_reference,
    deepseek_v4_compute_global_topk_indices_and_lens_reference,
    deepseek_v4_csa_indexer_mxfp4_cache_insert_reference,
    deepseek_v4_dequantize_and_gather_k_cache_reference,
    deepseek_v4_indexer_mxfp4_cache_gather_reference,
    deepseek_v4_indexer_mxfp4_cache_write_reference,
    deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference,
    deepseek_v4_save_compressor_state_reference,
    deepseek_v4_sparse_compress_cache_insert_reference,
    dsa_decode_topk_reference,
    dsa_full_context_topk_to_global_slots_reference,
    dsa_local_topk_to_global_slots_reference,
    dsa_sparse_decode_kv_pack_reference,
    dsa_sparse_decode_row_bytes,
    gdn_chunk_prefill_reference,
    gdn_qkv_split_reference,
    mha_reference,
    mla_reference,
    packed_qkv_complex_rotary_reference,
)

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
)


def _page_ids(inputs: MHAInputValues) -> list[int]:
    assert inputs.cache is not None
    assert inputs.cache.page_table_cpu is not None
    return [page_id for row in inputs.cache.page_table_cpu for page_id in row]


def _mha_config(
    *,
    batch_size: int,
    total_cached_tokens: int,
    total_new_q_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    cache_layout: str = "none",
    page_size: int | None = None,
    indexing: str | None = None,
    include_sinks: bool = False,
    metadata_kwargs: dict[str, object] | None = None,
) -> MHAInputConfig:
    return MHAInputConfig(
        batch_size=batch_size,
        total_cached_tokens=total_cached_tokens,
        total_new_q_tokens=total_new_q_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        cache_layout=cache_layout,  # type: ignore[arg-type]
        page_size=page_size,
        indexing=indexing,  # type: ignore[arg-type]
        include_sinks=include_sinks,
        metadata_input=MHARequestMetadataInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            cache_layout=cache_layout,  # type: ignore[arg-type]
            **(metadata_kwargs or {}),
        ),
    )


def _mla_config(
    *,
    batch_size: int,
    total_cached_tokens: int,
    total_new_q_tokens: int,
    num_q_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    q_dtype: torch.dtype,
    cache_layout: str = "none",
    page_size: int | None = None,
    indexing: str | None = None,
    num_kv_heads: int | None = None,
    metadata_kwargs: dict[str, object] | None = None,
) -> MLAInputConfig:
    metadata_fields = {
        "allow_untied_non_cached_kv": True,
        "new_q_length_mode": (
            "fixed_per_request" if cache_layout != "none" else "ragged"
        ),
        **(metadata_kwargs or {}),
    }
    return MLAInputConfig(
        batch_size=batch_size,
        total_cached_tokens=total_cached_tokens,
        total_new_q_tokens=total_new_q_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        q_dtype=q_dtype,
        cache_layout=cache_layout,  # type: ignore[arg-type]
        page_size=page_size,
        indexing=indexing,  # type: ignore[arg-type]
        metadata_input=MHARequestMetadataInputConfig(
            batch_size=batch_size,
            total_cached_tokens=total_cached_tokens,
            total_new_q_tokens=total_new_q_tokens,
            cache_layout=cache_layout,  # type: ignore[arg-type]
            **metadata_fields,
        ),
    )


def _attention_merge_state_values(
    *,
    total_q: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
    lse_scale_log2: float = 1.4426950408889634,
    lse_bound: float = 6.0,
) -> AttentionMergeStateInputValues:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out_shape = (total_q, num_heads, head_dim)
    lse_shape = (total_q, num_heads)
    return AttentionMergeStateInputValues(
        out_a=torch.randn(out_shape, dtype=dtype, generator=generator).contiguous(),
        lse_a=(
            torch.rand(lse_shape, dtype=torch.float32, generator=generator)
            * (2.0 * lse_bound)
            - lse_bound
        ).contiguous(),
        out_b=torch.randn(out_shape, dtype=dtype, generator=generator).contiguous(),
        lse_b=(
            torch.rand(lse_shape, dtype=torch.float32, generator=generator)
            * (2.0 * lse_bound)
            - lse_bound
        ).contiguous(),
        lse_scale_log2=lse_scale_log2,
    )


def _packed_qkv_complex_rotary_values(
    *,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    copy_v: bool,
    seed: int,
) -> PackedQKVComplexRotaryInputValues:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    qkv = torch.randn(
        (num_tokens, 3 * num_heads * head_dim),
        dtype=dtype,
        generator=generator,
    ).contiguous()
    angles = (
        torch.rand(
            (num_tokens, head_dim // 2),
            dtype=torch.float32,
            generator=generator,
        )
        * (2.0 * torch.pi)
        - torch.pi
    )
    return PackedQKVComplexRotaryInputValues(
        qkv=qkv,
        freqs_cis=torch.complex(torch.cos(angles), torch.sin(angles)).contiguous(),
        num_heads=num_heads,
        head_dim=head_dim,
        copy_v=copy_v,
    )


def _gdn_qkv_split_values(values: GDNInputValues) -> GDNQKVSplitInputValues:
    q = values.q.squeeze(0) if values.q.ndim == 4 else values.q
    k = values.k.squeeze(0) if values.k.ndim == 4 else values.k
    v = values.v.squeeze(0) if values.v.ndim == 4 else values.v
    return GDNQKVSplitInputValues(
        mixed_qkv=torch.cat(
            [q.reshape(q.shape[0], -1), k.reshape(k.shape[0], -1), v.reshape(v.shape[0], -1)],
            dim=-1,
        ).contiguous(),
        num_q_heads=q.shape[1],
        num_k_heads=k.shape[1],
        num_v_heads=v.shape[1],
        head_q=q.shape[2],
        head_k=k.shape[2],
        head_v=v.shape[2],
        fuse_l2norm=False,
        l2norm_eps=1.0e-6,
    )


def _dsa_pack_values(values: DSAInputValues) -> DSASparseDecodeKVPackInputValues:
    return DSASparseDecodeKVPackInputValues(
        out=values.packed_kv_out,
        loc=values.slot_mapping,
        cache_k_nope=values.cache_k_nope,
        cache_k_rope=values.cache_k_rope,
    )


def _dsa_topk_values(values: DSAInputValues) -> DSADecodeTopKInputValues:
    return DSADecodeTopKInputValues(
        logits=values.logits,
        out=values.topk_out,
        valid_lens=values.valid_lens,
        topk=values.topk,
    )


def _dsa_slot_values(values: DSAInputValues) -> DSATopKSlotInputValues:
    return DSATopKSlotInputValues(
        local_topk_offsets=values.local_topk_offsets,
        seq_lens=values.seq_lens,
        block_table=values.block_table,
        block_table_cpu=values.block_table_cpu,
        block_table_values=values.block_table_values,
        block_size=values.block_size,
        topk=values.topk,
    )


def test_attention_merge_state_inputs_generate_values_and_reference() -> None:
    values = _attention_merge_state_values(
        total_q=7,
        num_heads=3,
        head_dim=16,
        dtype=torch.float32,
        lse_bound=4.0,
        seed=1,
    )

    assert values.out_a.shape == (7, 3, 16)
    assert values.out_b.shape == (7, 3, 16)
    assert values.lse_a.shape == (7, 3)
    assert values.lse_b.shape == (7, 3)
    assert values.lse_a.dtype == torch.float32
    assert values.lse_b.dtype == torch.float32
    assert torch.all(values.lse_a >= -4.0)
    assert torch.all(values.lse_a <= 4.0)

    out, lse = attention_merge_state_reference(values)
    lse_ref = torch.maximum(values.lse_a, values.lse_b)
    weight_a = torch.exp(values.lse_a - lse_ref)
    weight_b = torch.exp(values.lse_b - lse_ref)
    denom = weight_a + weight_b
    out_ref = (
        values.out_a.float() * weight_a[..., None]
        + values.out_b.float() * weight_b[..., None]
    ) / denom[..., None]
    lse_ref = lse_ref + torch.log(denom)

    torch.testing.assert_close(out.float(), out_ref)
    torch.testing.assert_close(lse, lse_ref)


def test_attention_merge_state_inputs_support_log2_lse_scale() -> None:
    values = _attention_merge_state_values(
        total_q=5,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        lse_scale_log2=1.0,
        seed=2,
    )

    out, lse = attention_merge_state_reference(values)
    lse_ref = torch.maximum(values.lse_a, values.lse_b)
    weight_a = torch.exp2(values.lse_a - lse_ref)
    weight_b = torch.exp2(values.lse_b - lse_ref)
    denom = weight_a + weight_b
    out_ref = (
        values.out_a.float() * weight_a[..., None]
        + values.out_b.float() * weight_b[..., None]
    ) / denom[..., None]
    lse_ref = lse_ref + torch.log2(denom)

    torch.testing.assert_close(out.float(), out_ref)
    torch.testing.assert_close(lse, lse_ref)


def test_attention_merge_state_rejects_invalid_config() -> None:
    values = _attention_merge_state_values(
        total_q=7,
        num_heads=3,
        head_dim=16,
        dtype=torch.float32,
        lse_scale_log2=0.0,
        seed=3,
    )
    with pytest.raises(ValueError, match="lse_scale_log2"):
        attention_merge_state_reference(values)


def test_gdn_qkv_split_inputs_generate_plain_split_reference() -> None:
    gdn_values = GDNInputs(
        GDNInputConfig(
            batch_size=1,
            total_tokens=5,
            num_q_heads=2,
            num_v_heads=4,
            head_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=101, device="cpu")
    values = _gdn_qkv_split_values(gdn_values)

    q, k, v = gdn_qkv_split_reference(values)

    assert values.mixed_qkv.shape == (5, 64)
    assert q.shape == (1, 5, 2, 8)
    assert k.shape == (1, 5, 2, 8)
    assert v.shape == (1, 5, 4, 8)
    torch.testing.assert_close(q.reshape(5, -1), values.mixed_qkv[:, :16])
    torch.testing.assert_close(k.reshape(5, -1), values.mixed_qkv[:, 16:32])
    torch.testing.assert_close(v.reshape(5, -1), values.mixed_qkv[:, 32:])


def test_gdn_qkv_split_inputs_reject_invalid_configs_and_values() -> None:
    gdn_values = GDNInputs(
        GDNInputConfig(
            batch_size=1,
            total_tokens=5,
            num_q_heads=2,
            num_v_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=103, device="cpu")
    values = _gdn_qkv_split_values(gdn_values)
    values.mixed_qkv = values.mixed_qkv[:, :-1]
    with pytest.raises(ValueError, match="last dimension"):
        gdn_qkv_split_reference(values)


def test_gdn_chunk_prefill_inputs_generate_values_and_reference() -> None:
    values = GDNInputs(
        GDNInputConfig(
            batch_size=3,
            total_tokens=18,
            num_q_heads=2,
            num_v_heads=4,
            head_dim=8,
            dtype=torch.float32,
            max_tokens_per_sequence=8,
        )
    ).generate(metadata_seed=201, value_seed=301, device="cpu")

    assert values.q.shape == (18, 2, 8)
    assert values.k.shape == (18, 2, 8)
    assert values.v.shape == (18, 4, 8)
    assert values.g.shape == (18, 4)
    assert values.beta.shape == (18, 4)
    assert values.initial_state.shape == (3, 4, 8, 8)
    assert values.cu_seqlens.tolist()[-1] == 18
    assert sum(values.seq_lens_cpu) == 18
    assert max(values.seq_lens_cpu) <= 8
    torch.testing.assert_close(
        torch.linalg.vector_norm(values.q.float(), dim=-1),
        torch.ones((18, 2)),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(values.k.float(), dim=-1),
        torch.ones((18, 2)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.all(torch.exp(values.g) >= 0.75)
    assert torch.all(torch.exp(values.g) <= 0.99)
    assert torch.all(values.beta >= 0.05)
    assert torch.all(values.beta <= 0.95)

    reference = gdn_chunk_prefill_reference(values)

    assert reference.out.shape == (18, 4, 8)
    assert reference.out.dtype == torch.float32
    assert reference.final_state.shape == (3, 4, 8, 8)
    assert reference.final_state.dtype == torch.float32
    assert reference.state_checkpoints is None
    assert reference.checkpoint_cu_starts is None
    assert torch.isfinite(reference.out).all()
    assert torch.isfinite(reference.final_state).all()


def test_gdn_chunk_prefill_inputs_support_batch_axis_and_checkpoints() -> None:
    values = GDNInputs(
        GDNInputConfig(
            batch_size=2,
            total_tokens=128,
            num_q_heads=2,
            num_v_heads=4,
            head_dim=8,
            dtype=torch.float32,
            length_mode="fixed_per_request",
            include_batch_dim=True,
            output_h=True,
        )
    ).generate(seed=401, device="cpu")

    reference = gdn_chunk_prefill_reference(values)

    assert values.q.shape == (1, 128, 2, 8)
    assert values.g.shape == (1, 128, 4)
    assert values.seq_lens_cpu == [64, 64]
    assert reference.out.shape == (1, 128, 4, 8)
    assert reference.state_checkpoints is not None
    assert reference.state_checkpoints.shape == (2, 4, 8, 8)
    assert reference.checkpoint_cu_starts is not None
    assert reference.checkpoint_cu_starts.tolist() == [0, 1, 2]


def test_gdn_chunk_prefill_inputs_keep_metadata_seed_independent() -> None:
    generator = GDNInputs(
        GDNInputConfig(
            batch_size=4,
            total_tokens=23,
            num_q_heads=1,
            num_v_heads=2,
            head_dim=8,
            dtype=torch.float32,
            max_tokens_per_sequence=9,
        )
    )

    first = generator.generate(metadata_seed=501, value_seed=601, device="cpu")
    same_metadata = generator.generate(
        metadata_seed=501,
        value_seed=602,
        device="cpu",
    )
    same_values = generator.generate(
        metadata_seed=502,
        value_seed=601,
        device="cpu",
    )

    torch.testing.assert_close(first.cu_seqlens, same_metadata.cu_seqlens)
    assert not torch.equal(first.q, same_metadata.q)
    torch.testing.assert_close(first.q, same_values.q)


def test_gdn_chunk_prefill_inputs_reject_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="num_v_heads must be >= num_q_heads"):
        GDNInputs(
            GDNInputConfig(
                batch_size=2,
                total_tokens=8,
                num_q_heads=4,
                num_v_heads=2,
                head_dim=8,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="integer multiple"):
        GDNInputs(
            GDNInputConfig(
                batch_size=2,
                total_tokens=8,
                num_q_heads=2,
                num_v_heads=3,
                head_dim=8,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="total_tokens must be >= batch_size"):
        GDNInputs(
            GDNInputConfig(
                batch_size=4,
                total_tokens=3,
                num_q_heads=2,
                num_v_heads=2,
                head_dim=8,
                dtype=torch.float32,
            )
        )

    values = GDNInputs(
        GDNInputConfig(
            batch_size=2,
            total_tokens=8,
            num_q_heads=2,
            num_v_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=701, device="cpu")
    values.beta = values.beta[:, :1]
    with pytest.raises(ValueError, match="beta must have shape"):
        gdn_chunk_prefill_reference(values)


def test_packed_qkv_complex_rotary_inputs_generate_values_and_reference() -> None:
    values = _packed_qkv_complex_rotary_values(
        num_tokens=5,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        copy_v=False,
        seed=111,
    )

    q, k, v = packed_qkv_complex_rotary_reference(values)

    assert values.qkv.shape == (5, 48)
    assert values.freqs_cis.shape == (5, 4)
    assert values.freqs_cis.dtype == torch.complex64
    torch.testing.assert_close(
        torch.abs(values.freqs_cis),
        torch.ones_like(values.freqs_cis.real),
    )
    assert q.shape == (5, 2, 8)
    assert k.shape == (5, 2, 8)
    assert v.shape == (5, 2, 8)

    q_raw = values.qkv[:, :16].reshape(5, 2, 8)
    freqs = values.freqs_cis
    q_even = q_raw[..., 0::2]
    q_odd = q_raw[..., 1::2]
    q_ref = torch.empty_like(q_raw)
    q_ref[..., 0::2] = q_even * freqs.real[:, None, :] - q_odd * freqs.imag[:, None, :]
    q_ref[..., 1::2] = q_odd * freqs.real[:, None, :] + q_even * freqs.imag[:, None, :]

    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(v.reshape(5, -1), values.qkv[:, 32:])


def test_packed_qkv_complex_rotary_inputs_support_copy_v() -> None:
    values = _packed_qkv_complex_rotary_values(
        num_tokens=3,
        num_heads=2,
        head_dim=6,
        dtype=torch.float32,
        copy_v=True,
        seed=112,
    )

    _, _, v = packed_qkv_complex_rotary_reference(values)

    assert v.is_contiguous()
    torch.testing.assert_close(v.reshape(3, -1), values.qkv[:, 24:])


def test_packed_qkv_complex_rotary_inputs_reject_invalid_configs_and_values() -> None:
    values = _packed_qkv_complex_rotary_values(
        num_tokens=5,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        copy_v=False,
        seed=113,
    )
    values.head_dim = 7
    with pytest.raises(ValueError, match="head_dim"):
        packed_qkv_complex_rotary_reference(values)

    values = _packed_qkv_complex_rotary_values(
        num_tokens=5,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        copy_v=False,
        seed=114,
    )
    values.freqs_cis = values.freqs_cis[:-1]
    with pytest.raises(ValueError, match="freqs_cis"):
        packed_qkv_complex_rotary_reference(values)

    values = _packed_qkv_complex_rotary_values(
        num_tokens=5,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        copy_v=False,
        seed=115,
    )
    values.freqs_cis = values.freqs_cis.real
    with pytest.raises(TypeError, match="complex"):
        packed_qkv_complex_rotary_reference(values)


def test_dsa_sparse_decode_kv_pack_inputs_generate_values_and_reference() -> None:
    values = DSAInputs(
        DSAInputConfig(
            num_tokens=5,
            num_slots=9,
            nope_dim=128,
            rope_dim=64,
            vocab_size=12,
            topk=3,
            block_size=8,
            max_pages_per_token=4,
            dtype=torch.float32,
        )
    ).generate(seed=118, metadata_seed=119, device="cpu")
    pack_values = _dsa_pack_values(values)

    expected_row_bytes = dsa_sparse_decode_row_bytes(128, 64)
    packed = dsa_sparse_decode_kv_pack_reference(pack_values)
    loc = values.slot_mapping.to(torch.int64)
    scale_offset = 128
    rope_offset = scale_offset + 4

    assert values.packed_kv_out.shape == (9, expected_row_bytes)
    assert values.slot_mapping.shape == (5,)
    assert torch.unique(values.slot_mapping).numel() == values.slot_mapping.numel()
    assert values.cache_k_nope.shape == (5, 128)
    assert values.cache_k_rope.shape == (5, 64)
    assert packed.shape == values.packed_kv_out.shape
    assert packed.dtype == torch.uint8

    written_mask = torch.zeros(values.packed_kv_out.shape[0], dtype=torch.bool)
    written_mask[loc.cpu()] = True
    assert torch.equal(packed[~written_mask], values.packed_kv_out[~written_mask])

    scales = packed[loc, scale_offset:rope_offset].contiguous().view(torch.float32)
    assert scales.shape == (5, 1)
    assert torch.all(scales > 0)
    expected_rope_bytes = values.cache_k_rope.contiguous().view(torch.uint8)
    assert torch.equal(packed[loc, rope_offset:], expected_rope_bytes)


def test_dsa_sparse_decode_kv_pack_supports_head_axis_and_zero_tokens() -> None:
    values = DSAInputs(
        DSAInputConfig(
            num_tokens=0,
            num_slots=3,
            nope_dim=128,
            rope_dim=32,
            vocab_size=8,
            topk=2,
            block_size=4,
            max_pages_per_token=3,
            dtype=torch.float32,
            include_head_axis=True,
        )
    ).generate(seed=120, device="cpu")
    pack_values = _dsa_pack_values(values)

    packed = dsa_sparse_decode_kv_pack_reference(pack_values)

    assert values.slot_mapping.shape == (0,)
    assert values.cache_k_nope.shape == (0, 1, 128)
    assert values.cache_k_rope.shape == (0, 1, 32)
    assert torch.equal(packed, values.packed_kv_out)


def test_dsa_sparse_decode_kv_pack_rejects_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="num_tokens"):
        DSAInputs(
            DSAInputConfig(
                num_tokens=4,
                num_slots=3,
                nope_dim=128,
                rope_dim=64,
                vocab_size=8,
                topk=2,
                block_size=4,
                max_pages_per_token=3,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="divisible"):
        DSAInputs(
            DSAInputConfig(
                num_tokens=2,
                num_slots=3,
                nope_dim=64,
                rope_dim=64,
                vocab_size=8,
                topk=2,
                block_size=4,
                max_pages_per_token=3,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="power of two"):
        DSAInputs(
            DSAInputConfig(
                num_tokens=2,
                num_slots=3,
                nope_dim=128,
                rope_dim=48,
                vocab_size=8,
                topk=2,
                block_size=4,
                max_pages_per_token=3,
                dtype=torch.float32,
            )
        )

    values = DSAInputs(
        DSAInputConfig(
            num_tokens=3,
            num_slots=5,
            nope_dim=128,
            rope_dim=64,
            vocab_size=8,
            topk=2,
            block_size=4,
            max_pages_per_token=3,
            dtype=torch.float32,
        )
    ).generate(seed=121, device="cpu")
    pack_values = _dsa_pack_values(values)
    pack_values.loc = torch.tensor([0, 0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="unique"):
        dsa_sparse_decode_kv_pack_reference(pack_values)


def test_dsa_decode_topk_inputs_generate_stable_tie_values() -> None:
    values = DSAInputs(
        DSAInputConfig(
            num_tokens=4,
            num_slots=6,
            nope_dim=128,
            rope_dim=64,
            vocab_size=12,
            topk=3,
            block_size=8,
            max_pages_per_token=4,
            dtype=torch.float32,
            min_valid_len=6,
            max_valid_len=12,
        )
    ).generate(seed=188, device="cpu")
    topk_values = _dsa_topk_values(values)

    expected = dsa_decode_topk_reference(topk_values)

    assert values.logits.shape == (4, 12)
    assert values.topk_out.shape == (4, 3)
    assert values.topk_out.dtype == torch.int32
    assert values.valid_lens.shape == (4,)
    assert values.valid_lens.dtype == torch.int32
    assert expected.shape == (4, 3)
    assert expected.dtype == torch.int32
    for row, valid_len_tensor in enumerate(values.valid_lens):
        valid_len = int(valid_len_tensor.item())
        assert torch.isneginf(values.logits[row, valid_len:]).all()
        if valid_len > values.topk:
            assert expected[row, -1].item() == valid_len - 2


def test_dsa_decode_topk_reference_matches_manual_ordering() -> None:
    values = DSAInputs(
        DSAInputConfig(
            num_tokens=1,
            num_slots=2,
            nope_dim=128,
            rope_dim=64,
            vocab_size=5,
            topk=3,
            block_size=4,
            max_pages_per_token=2,
            dtype=torch.float32,
            min_valid_len=5,
            max_valid_len=5,
        )
    ).generate(seed=189, device="cpu")
    topk_values = _dsa_topk_values(values)
    topk_values.logits[0] = torch.tensor([1.0, 3.0, 3.0, 2.0, -float("inf")])
    topk_values.valid_lens[0] = 4

    expected = dsa_decode_topk_reference(topk_values)

    torch.testing.assert_close(
        expected,
        torch.tensor([[1, 2, 3]], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_dsa_decode_topk_inputs_reject_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="topk must be <= vocab_size"):
        DSAInputs(
            DSAInputConfig(
                num_tokens=1,
                num_slots=2,
                nope_dim=128,
                rope_dim=64,
                vocab_size=4,
                topk=5,
                block_size=4,
                max_pages_per_token=2,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="min_valid_len must be >= topk"):
        DSAInputs(
            DSAInputConfig(
                num_tokens=1,
                num_slots=2,
                nope_dim=128,
                rope_dim=64,
                vocab_size=8,
                topk=4,
                block_size=4,
                max_pages_per_token=2,
                dtype=torch.float32,
                min_valid_len=3,
            )
        )

    values = DSAInputs(
        DSAInputConfig(
            num_tokens=1,
            num_slots=2,
            nope_dim=128,
            rope_dim=64,
            vocab_size=8,
            topk=4,
            block_size=4,
            max_pages_per_token=2,
            dtype=torch.float32,
            min_valid_len=6,
            max_valid_len=6,
        )
    ).generate(seed=190, device="cpu")
    topk_values = _dsa_topk_values(values)
    topk_values.logits[0, int(topk_values.valid_lens[0].item()) :] = 0.0

    with pytest.raises(ValueError, match="masked with -inf"):
        dsa_decode_topk_reference(topk_values)


def test_dsa_topk_slot_inputs_generate_values_and_references() -> None:
    values = DSAInputs(
        DSAInputConfig(
            num_tokens=5,
            num_slots=7,
            nope_dim=128,
            rope_dim=64,
            vocab_size=16,
            topk=4,
            block_size=8,
            max_pages_per_token=4,
            max_seq_len=24,
            dtype=torch.float32,
            indexing="identity",
        )
    ).generate(seed=121, device="cpu")
    slot_values = _dsa_slot_values(values)

    local_slots, local_lens = dsa_local_topk_to_global_slots_reference(slot_values)
    full_slots, full_lens = dsa_full_context_topk_to_global_slots_reference(slot_values)

    assert values.local_topk_offsets.shape == (5, 4)
    assert values.seq_lens.shape == (5,)
    assert values.block_table.shape == (5, 4)
    assert local_slots.shape == (5, 4)
    assert full_slots.shape == (5, 4)
    assert local_lens.shape == (5,)
    assert full_lens.shape == (5,)
    assert torch.all(local_lens > 0)
    assert torch.all(local_lens <= 4)
    assert torch.all(full_lens <= 4)

    for token_idx in range(5):
        for slot_idx in range(4):
            local_idx = int(values.local_topk_offsets[token_idx, slot_idx].item())
            if local_idx < 0:
                assert int(local_slots[token_idx, slot_idx].item()) == -1
                continue
            block = local_idx // values.block_size
            offset = local_idx % values.block_size
            expected = int(values.block_table[token_idx, block].item()) * 8 + offset
            assert int(local_slots[token_idx, slot_idx].item()) == expected


def test_dsa_topk_slot_reference_supports_no_seq_lens_local_mode() -> None:
    values = DSAInputs(
        DSAInputConfig(
            num_tokens=3,
            num_slots=5,
            nope_dim=128,
            rope_dim=64,
            vocab_size=12,
            topk=5,
            block_size=4,
            max_pages_per_token=3,
            max_seq_len=12,
            dtype=torch.float32,
            indexing="identity",
        )
    ).generate(seed=122, device="cpu")
    slot_values = _dsa_slot_values(values)

    slots, lens = dsa_local_topk_to_global_slots_reference(
        slot_values,
        use_seq_lens=False,
    )

    assert slots.shape == (3, 5)
    assert lens.shape == (3,)
    assert torch.all(lens > 0)


def test_dsa_topk_slot_inputs_support_zero_tokens() -> None:
    values = DSAInputs(
        DSAInputConfig(
            num_tokens=0,
            num_slots=3,
            nope_dim=128,
            rope_dim=64,
            vocab_size=8,
            topk=5,
            block_size=4,
            max_pages_per_token=3,
            dtype=torch.float32,
        )
    ).generate(seed=124, device="cpu")
    slot_values = _dsa_slot_values(values)

    slots, lens = dsa_full_context_topk_to_global_slots_reference(slot_values)

    assert values.local_topk_offsets.shape == (0, 5)
    assert values.seq_lens.shape == (0,)
    assert values.block_table.shape == (1, 3)
    assert slots.shape == (0, 5)
    assert lens.shape == (0,)


def test_dsa_topk_slot_inputs_reject_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="topk"):
        DSAInputs(
            DSAInputConfig(
                num_tokens=3,
                num_slots=5,
                nope_dim=128,
                rope_dim=64,
                vocab_size=8,
                topk=0,
                block_size=8,
                max_pages_per_token=2,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="page_table_input.batch_size"):
        DSAInputs(
            DSAInputConfig(
                num_tokens=3,
                num_slots=5,
                nope_dim=128,
                rope_dim=64,
                vocab_size=8,
                topk=4,
                block_size=8,
                max_pages_per_token=2,
                dtype=torch.float32,
                page_table_input=PageTableInputConfig(
                    batch_size=2,
                    max_pages_per_request=2,
                ),
            )
        )

    values = DSAInputs(
        DSAInputConfig(
            num_tokens=3,
            num_slots=5,
            nope_dim=128,
            rope_dim=64,
            vocab_size=8,
            topk=4,
            block_size=8,
            max_pages_per_token=2,
            dtype=torch.float32,
        )
    ).generate(seed=123, device="cpu")
    slot_values = _dsa_slot_values(values)
    slot_values.seq_lens = slot_values.seq_lens[:-1]
    with pytest.raises(ValueError, match="seq_lens"):
        dsa_local_topk_to_global_slots_reference(slot_values)


def test_compressed_sequence_attention_generates_swa_only_values() -> None:
    values = CSAInputs(
        CSAInputConfig(
            batch_size=2,
            total_cached_tokens=8,
            total_new_q_tokens=4,
            dtype=torch.bfloat16,
            num_q_heads=3,
            page_size=4,
            window_size=5,
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=2,
                total_cached_tokens=8,
                total_new_q_tokens=4,
                cache_layout="paged",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=911, device="cpu")

    assert values.kind == "swa"
    assert values.compressed is None
    assert values.indexer is None
    assert values.sliding_window.q.shape == (4, 3, 512)
    assert values.sliding_window.positions.shape == (4,)
    assert values.sliding_window.token_to_req_indices.tolist() == [0, 0, 1, 1]
    assert values.sliding_window.cache_2d.dtype == torch.uint8
    assert values.sliding_window.page_table.shape[0] == 2


def test_compressed_sequence_attention_generates_hca_values() -> None:
    values = CSAInputs(
        CSAInputConfig(
            batch_size=2,
            total_cached_tokens=16,
            total_new_q_tokens=4,
            dtype=torch.float32,
            num_q_heads=2,
            page_size=4,
            window_size=5,
            compressed=CSAHistoryConfig(
                compress_ratio=128,
                topk=3,
                num_state_cache_blocks=2,
                compressor_block_size=8,
                num_kv_cache_blocks=2,
                kv_cache_block_size=4,
            ),
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=2,
                total_cached_tokens=16,
                total_new_q_tokens=4,
                cache_layout="paged",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=912, device="cpu")

    assert values.kind == "hca"
    assert values.indexer is None
    assert values.compressed is not None
    assert values.compressed.cache_insert.compress_ratio == 128
    assert values.compressed.cache_insert.overlap is False
    assert values.compressed.compressor_state.compress_ratio == 128
    assert values.compressed.paged_index.compress_ratio == 128
    assert (
        values.compressed.paged_index.metadata.new_q_lens_cpu
        == values.sliding_window.metadata.new_q_lens_cpu
    )
    assert (
        values.compressed.sparse_prefill_index.metadata.visible_kv_lens_cpu
        == values.sliding_window.metadata.visible_kv_lens_cpu
    )
    assert torch.equal(values.compressed.paged_index.block_table, values.sliding_window.page_table)

    deepseek_v4_sparse_compress_cache_insert_reference(values.compressed.cache_insert)
    deepseek_v4_save_compressor_state_reference(values.compressed.compressor_state)
    deepseek_v4_compute_global_topk_indices_and_lens_reference(
        values.compressed.paged_index
    )
    deepseek_v4_combine_dense_swa_indices_reference(
        values.compressed.sparse_prefill_index
    )
    deepseek_v4_dequantize_and_gather_k_cache_reference(values.compressed.k_cache_gather)


def test_compressed_sequence_attention_generates_csa_indexer_values() -> None:
    values = CSAInputs(
        CSAInputConfig(
            batch_size=2,
            total_cached_tokens=16,
            total_new_q_tokens=4,
            dtype=torch.float32,
            num_q_heads=2,
            page_size=4,
            window_size=5,
            compressed=CSAHistoryConfig(
                compress_ratio=4,
                topk=3,
                num_state_cache_blocks=2,
                compressor_block_size=4,
                num_kv_cache_blocks=2,
                kv_cache_block_size=4,
            ),
            indexer=CSAIndexerConfig(
                num_heads=2,
                num_state_cache_blocks=2,
                compressor_block_size=4,
                num_kv_cache_blocks=2,
                kv_cache_block_size=4,
                num_cache_blocks=2,
                block_size=4,
            ),
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=2,
                total_cached_tokens=16,
                total_new_q_tokens=4,
                cache_layout="paged",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(metadata_seed=913, value_seed=914, device="cpu")

    assert values.kind == "csa"
    assert values.compressed is not None
    assert values.indexer is not None
    assert values.compressed.cache_insert.compress_ratio == 4
    assert values.compressed.cache_insert.overlap is True
    assert values.indexer.cache_insert.compress_ratio == 4
    assert values.indexer.q_rope_hadamard_mxfp4.index_q.shape == (4, 2, 128)
    assert (
        values.compressed.paged_index.metadata.new_q_lens_cpu
        == values.sliding_window.metadata.new_q_lens_cpu
    )
    assert (
        values.compressed.sparse_prefill_index.token_to_req_indices.tolist()
        == values.sliding_window.token_to_req_indices.tolist()
    )

    deepseek_v4_sparse_compress_cache_insert_reference(values.compressed.cache_insert)
    deepseek_v4_csa_indexer_mxfp4_cache_insert_reference(values.indexer.cache_insert)
    deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference(
        values.indexer.q_rope_hadamard_mxfp4
    )
    deepseek_v4_indexer_mxfp4_cache_write_reference(values.indexer.cache_write)
    deepseek_v4_indexer_mxfp4_cache_gather_reference(values.indexer.cache_gather)


def test_compressed_sequence_attention_rejects_invalid_component_mix() -> None:
    with pytest.raises(ValueError, match="indexer requires compressed"):
        CSAInputs(
            CSAInputConfig(
                batch_size=1,
                total_cached_tokens=1,
                total_new_q_tokens=1,
                dtype=torch.float32,
                num_q_heads=1,
                page_size=1,
                window_size=1,
                indexer=CSAIndexerConfig(),
            )
        )

    with pytest.raises(ValueError, match="only valid for CSA"):
        CSAInputs(
            CSAInputConfig(
                batch_size=1,
                total_cached_tokens=1,
                total_new_q_tokens=1,
                dtype=torch.float32,
                num_q_heads=1,
                page_size=1,
                window_size=1,
                compressed=CSAHistoryConfig(
                    compress_ratio=128,
                ),
                indexer=CSAIndexerConfig(),
            )
        )


def test_page_table_input_generates_identity_and_random_indexing() -> None:
    identity = PageTableInput(
        PageTableInputConfig(
            batch_size=3,
            max_pages_per_request=5,
            indexing="identity",
        )
    ).generate(seed=1, device="cpu")
    random = PageTableInput(
        PageTableInputConfig(
            batch_size=3,
            max_pages_per_request=5,
        )
    ).generate(seed=1, device="cpu")

    expected_pages = list(range(15))

    assert identity.page_table.shape == (3, 5)
    assert identity.page_ids() == expected_pages
    assert sorted(random.page_ids()) == expected_pages
    assert random.page_ids() != expected_pages


def test_page_table_input_can_index_larger_physical_page_pool() -> None:
    values = PageTableInput(
        PageTableInputConfig(
            batch_size=2,
            max_pages_per_request=3,
            indexing="random",
            num_physical_pages=11,
        )
    ).generate(seed=4, device="cpu")

    assert values.page_table.shape == (2, 3)
    assert values.num_pages == 11
    assert len(set(values.page_ids())) == 6
    assert min(values.page_ids()) >= 0
    assert max(values.page_ids()) < 11


def test_slot_mapping_input_generates_unique_nullable_slots() -> None:
    values = SlotMappingInput(
        SlotMappingInputConfig(
            num_rows=8,
            total_slots=16,
            negative_count=3,
        )
    ).generate(seed=5, device="cpu")

    slots = values.slot_mapping.cpu()
    mapped = slots[slots >= 0]
    assert values.slot_mapping.dtype == torch.int64
    assert int((slots < 0).sum().item()) == 3
    assert mapped.numel() == 5
    assert torch.unique(mapped).numel() == mapped.numel()
    assert int(mapped.max().item()) < 16


def test_slot_mapping_input_can_allow_duplicate_gather_slots() -> None:
    values = SlotMappingInput(
        SlotMappingInputConfig(
            num_rows=6,
            total_slots=2,
            unique=False,
            dtype=torch.int32,
        )
    ).generate(seed=6, device="cpu")

    assert values.slot_mapping.dtype == torch.int32
    assert values.slot_mapping.shape == (6,)
    assert int(values.slot_mapping.min().item()) >= 0
    assert int(values.slot_mapping.max().item()) < 2


def test_kv_cache_input_generates_dense_cache() -> None:
    cache = KVCacheInput(
        KVCacheInputConfig(
            cache_layout="dense",
            batch_size=3,
            max_seqlen_k=17,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=1, device="cpu")

    assert cache.k_cache is not None
    assert cache.v_cache is not None
    assert cache.page_table is None
    assert cache.page_table_cpu is None
    assert cache.page_table_values is None
    assert cache.k_cache.shape == (3, 17, 2, 8)
    assert cache.v_cache.shape == (3, 17, 2, 8)


def test_kv_cache_input_requires_real_cache_layout() -> None:
    with pytest.raises(ValueError, match="cache_layout must be 'dense' or 'paged'"):
        KVCacheInput(
            KVCacheInputConfig(
                cache_layout="none",
                batch_size=3,
                max_seqlen_k=17,
                num_kv_heads=2,
                head_dim=8,
                dtype=torch.float32,
            )
        )


def test_kv_cache_input_generates_paged_cache_with_nested_page_table() -> None:
    cache_input = KVCacheInput(
        KVCacheInputConfig(
            cache_layout="paged",
            batch_size=3,
            max_seqlen_k=17,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            page_size=4,
            page_table_input=PageTableInputConfig(
                batch_size=3,
                max_pages_per_request=1,
                indexing="identity",
            ),
        )
    )
    assert cache_input.page_table_input is not None
    cache_input.page_table_input.config.indexing = "random"

    cache = cache_input.generate(seed=1, device="cpu")

    assert cache.k_cache is not None
    assert cache.v_cache is not None
    assert cache.page_table is not None
    assert cache.page_table.shape == (3, 5)
    assert cache.k_cache.shape == (15, 4, 2, 8)
    assert cache.v_cache.shape == (15, 4, 2, 8)
    assert cache.page_table_values is not None
    assert sorted(cache.page_table_values.page_ids()) == list(range(15))
    assert cache.page_table_values.page_ids() != list(range(15))


def test_kv_cache_input_uses_nested_page_table_config() -> None:
    page_table_config = PageTableInputConfig(
        batch_size=1,
        max_pages_per_request=1,
        indexing="identity",
    )
    cache_input = KVCacheInput(
        KVCacheInputConfig(
            cache_layout="paged",
            batch_size=3,
            max_seqlen_k=17,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            page_size=4,
            page_table_input=page_table_config,
        )
    )

    assert cache_input.page_table_input is not None
    assert cache_input.page_table_input.config is page_table_config

    cache = cache_input.generate(seed=1, device="cpu")

    assert cache.page_table_values is not None
    assert cache.page_table_values.page_ids() == list(range(15))


def test_mla_kv_cache_input_generates_paged_compressed_cache() -> None:
    cache = MLAKVCacheInput(
        MLAKVCacheInputConfig(
            cache_layout="paged",
            batch_size=2,
            max_seqlen_k=9,
            kv_lora_rank=16,
            qk_rope_head_dim=4,
            dtype=torch.float32,
            page_size=4,
            page_table_input=PageTableInputConfig(
                batch_size=2,
                max_pages_per_request=1,
                indexing="identity",
            ),
        )
    ).generate(seed=1, device="cpu")

    assert cache.page_table is not None
    assert cache.page_table.shape == (2, 3)
    assert cache.kv_cache.shape == (6, 4, 1, 20)
    assert cache.page_table_values is not None
    assert cache.page_table_values.page_ids() == list(range(6))


def test_mha_inputs_without_cache_leaves_cache_input_none() -> None:
    generator = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=0,
            total_new_q_tokens=12,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            q_dtype=torch.float32,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "fixed_per_request"},
        )
    )
    assert generator.cache_input is None

    inputs = generator.generate(metadata_seed=31, value_seed=41, device="cpu")

    assert generator.cache_input is None
    assert inputs.cache is None


def test_mha_inputs_keep_metadata_seed_independent_from_values() -> None:
    first = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=31,
            total_new_q_tokens=9,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=8,
        )
    ).generate(metadata_seed=11, value_seed=21, device="cpu")
    second = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=31,
            total_new_q_tokens=9,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=8,
        )
    ).generate(metadata_seed=11, value_seed=22, device="cpu")

    assert first.cache is not None
    assert second.cache is not None
    assert first.metadata.cached_lens_cpu == second.metadata.cached_lens_cpu
    assert first.metadata.new_q_lens_cpu == second.metadata.new_q_lens_cpu
    assert first.metadata.visible_kv_lens_cpu == second.metadata.visible_kv_lens_cpu
    assert torch.equal(first.cache.page_table, second.cache.page_table)
    assert torch.equal(first.metadata.cu_seqlens_q, second.metadata.cu_seqlens_q)
    assert first.q is not None
    assert second.q is not None
    assert not torch.equal(first.q, second.q)


def test_mha_inputs_generate_dense_qkv_view() -> None:
    inputs = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=0,
            total_new_q_tokens=12,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            q_dtype=torch.float32,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "fixed_per_request"},
        )
    ).generate(metadata_seed=31, value_seed=41, device="cpu")

    q, k, v = inputs.dense_qkv()

    assert q.shape == (3, 4, 4, 8)
    assert k.shape == (3, 4, 2, 8)
    assert v.shape == (3, 4, 2, 8)
    assert inputs.metadata.cu_seqlens_q_cpu == [0, 4, 8, 12]


def test_mha_reference_matches_manual_causal_prefill() -> None:
    inputs = MHAInputs(
        _mha_config(
            batch_size=2,
            total_cached_tokens=0,
            total_new_q_tokens=6,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=4,
            q_dtype=torch.float32,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "fixed_per_request"},
        )
    ).generate(metadata_seed=31, value_seed=41, device="cpu")
    assert inputs.q is not None and inputs.k is not None and inputs.v is not None

    ref = mha_reference(inputs)

    manual_outputs = []
    manual_lses = []
    scale = inputs.q.shape[-1] ** -0.5
    for request_idx in range(2):
        start = request_idx * 3
        end = start + 3
        q = inputs.q[start:end].float()
        k = inputs.k[start:end].float().expand(-1, 2, -1)
        v = inputs.v[start:end].float().expand(-1, 2, -1)
        scores = torch.einsum("qhd,khd->qkh", q, k) * scale
        causal_mask = torch.triu(
            torch.ones((3, 3), dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(-1), float("-inf"))
        probs = torch.softmax(scores, dim=1)
        manual_outputs.append(torch.einsum("qkh,khd->qhd", probs, v))
        manual_lses.append(torch.logsumexp(scores, dim=1))

    torch.testing.assert_close(ref.out, torch.cat(manual_outputs, dim=0))
    torch.testing.assert_close(ref.lse, torch.cat(manual_lses, dim=0))


def test_mha_reference_uses_paged_cache_values() -> None:
    inputs = MHAInputs(
        _mha_config(
            batch_size=2,
            total_cached_tokens=6,
            total_new_q_tokens=2,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=4,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=2,
            indexing="identity",
            metadata_kwargs={
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=7, value_seed=9, device="cpu")
    assert inputs.q is not None
    assert inputs.cache is not None
    assert inputs.cache.k_cache is not None
    assert inputs.cache.v_cache is not None
    assert inputs.cache.page_table_cpu is not None

    ref = mha_reference(inputs, is_causal=True)

    manual_outputs = []
    manual_lses = []
    scale = inputs.q.shape[-1] ** -0.5
    page_size = inputs.cache.k_cache.shape[1]
    for batch_idx in range(2):
        q = inputs.q[batch_idx : batch_idx + 1].float()
        rows_k = []
        rows_v = []
        for pos in range(int(inputs.metadata.cache_seqlens[batch_idx].item())):
            page_col = pos // page_size
            page_offset = pos % page_size
            physical_page = inputs.cache.page_table_cpu[batch_idx][page_col]
            rows_k.append(inputs.cache.k_cache[physical_page, page_offset])
            rows_v.append(inputs.cache.v_cache[physical_page, page_offset])
        k = torch.stack(rows_k, dim=0).float().expand(-1, 2, -1)
        v = torch.stack(rows_v, dim=0).float().expand(-1, 2, -1)
        scores = torch.einsum("qhd,khd->qkh", q, k) * scale
        probs = torch.softmax(scores, dim=1)
        manual_outputs.append(torch.einsum("qkh,khd->qhd", probs, v))
        manual_lses.append(torch.logsumexp(scores, dim=1))

    torch.testing.assert_close(ref.out, torch.cat(manual_outputs, dim=0))
    torch.testing.assert_close(ref.lse, torch.cat(manual_lses, dim=0))


def test_mha_inputs_generate_bounded_ragged_request_lengths() -> None:
    inputs = MHAInputs(
        _mha_config(
            batch_size=5,
            total_cached_tokens=27,
            total_new_q_tokens=23,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=8,
            metadata_kwargs={
                "max_cached_tokens_per_request": 9,
                "max_new_q_tokens_per_request": 8,
            },
        )
    ).generate(metadata_seed=1, value_seed=99, device="cpu")

    assert sum(inputs.metadata.cached_lens_cpu) == 27
    assert sum(inputs.metadata.new_q_lens_cpu) == 23
    assert max(inputs.metadata.cached_lens_cpu) <= 9
    assert max(inputs.metadata.new_q_lens_cpu) <= 8
    assert len(set(inputs.metadata.cached_lens_cpu)) > 1
    assert len(set(inputs.metadata.new_q_lens_cpu)) > 1


def test_mha_inputs_generate_identity_page_table_indexing() -> None:
    inputs = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=36,
            total_new_q_tokens=12,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=4,
            indexing="identity",
            metadata_kwargs={
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)

    assert page_ids == list(range(len(page_ids)))


def test_mha_inputs_generate_random_page_table_indexing_by_default() -> None:
    inputs = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=36,
            total_new_q_tokens=12,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=4,
            metadata_kwargs={
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)

    assert sorted(page_ids) == list(range(len(page_ids)))
    assert page_ids != list(range(len(page_ids)))


def test_mha_inputs_allows_nested_page_table_generator_configuration() -> None:
    generator = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=36,
            total_new_q_tokens=12,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=4,
            metadata_kwargs={
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    )
    assert generator.cache_input is not None
    assert generator.cache_input.page_table_input is not None
    generator.cache_input.page_table_input.config.indexing = "identity"

    inputs = generator.generate(metadata_seed=1, value_seed=99, device="cpu")
    page_ids = _page_ids(inputs)

    assert page_ids == list(range(len(page_ids)))


def test_mha_inputs_accepts_nested_config_objects() -> None:
    config = MHAInputConfig(
        batch_size=3,
        total_cached_tokens=36,
        total_new_q_tokens=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        page_size=4,
        indexing="identity",
        metadata_input=MHARequestMetadataInputConfig(
            batch_size=3,
            total_cached_tokens=36,
            total_new_q_tokens=12,
            cached_length_mode="regular",
            new_q_length_mode="fixed_per_request",
            cache_layout="paged",
        ),
        cache_input=KVCacheInputConfig(
            cache_layout="paged",
            batch_size=3,
            max_seqlen_k=1,
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.float32,
            page_size=4,
            page_table_input=PageTableInputConfig(
                batch_size=3,
                max_pages_per_request=1,
                indexing="identity",
            ),
        ),
    )

    generator = MHAInputs(config)
    assert generator.q_input is not None
    generator.q_input.dtype = torch.float64

    assert generator.config is config
    assert generator.metadata_input is not None
    assert generator.cache_input is not None
    assert generator.cache_input.page_table_input is not None
    assert generator.metadata_input.config is config.metadata_input
    assert generator.cache_input.config is config.cache_input
    assert (
        generator.cache_input.page_table_input.config
        is config.cache_input.page_table_input
    )

    inputs = generator.generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)
    assert page_ids == list(range(len(page_ids)))
    assert inputs.q is not None
    assert inputs.q.dtype == torch.float64


def test_mha_input_config_builds_children_from_common_parent_fields() -> None:
    config = MHAInputConfig(
        batch_size=3,
        total_cached_tokens=36,
        total_new_q_tokens=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        page_size=4,
        indexing="identity",
    )

    generator = MHAInputs(config)

    assert generator.metadata_input is not None
    assert generator.cache_input is not None
    assert generator.cache_input.page_table_input is not None
    assert generator.metadata_input.config.batch_size == config.batch_size
    assert (
        generator.metadata_input.config.total_cached_tokens
        == config.total_cached_tokens
    )
    assert (
        generator.metadata_input.config.total_new_q_tokens == config.total_new_q_tokens
    )
    assert generator.metadata_input.config.cache_layout == config.cache_layout
    assert generator.cache_input.config.batch_size == config.batch_size
    assert generator.cache_input.config.cache_layout == config.cache_layout
    assert generator.cache_input.config.page_size == config.page_size
    assert generator.cache_input.page_table_input.config.indexing == config.indexing

    inputs = generator.generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)
    assert page_ids == list(range(len(page_ids)))


def test_mha_input_config_rejects_mismatched_metadata_override() -> None:
    with pytest.raises(
        ValueError,
        match="metadata_input.batch_size=.*MHAInputConfig.batch_size",
    ):
        MHAInputs(
            MHAInputConfig(
                batch_size=3,
                total_cached_tokens=36,
                total_new_q_tokens=12,
                num_q_heads=4,
                num_kv_heads=2,
                head_dim=16,
                q_dtype=torch.float32,
                cache_layout="paged",
                page_size=4,
                metadata_input=MHARequestMetadataInputConfig(
                    batch_size=2,
                    total_cached_tokens=36,
                    total_new_q_tokens=12,
                    cache_layout="paged",
                ),
            )
        )


def test_mha_input_config_rejects_mismatched_cache_override() -> None:
    with pytest.raises(
        ValueError,
        match="cache_input.page_size=.*MHAInputConfig.page_size",
    ):
        MHAInputs(
            MHAInputConfig(
                batch_size=3,
                total_cached_tokens=36,
                total_new_q_tokens=12,
                num_q_heads=4,
                num_kv_heads=2,
                head_dim=16,
                q_dtype=torch.float32,
                cache_layout="paged",
                page_size=8,
                cache_input=KVCacheInputConfig(
                    cache_layout="paged",
                    batch_size=3,
                    max_seqlen_k=1,
                    num_kv_heads=2,
                    head_dim=16,
                    dtype=torch.float32,
                    page_size=4,
                ),
            )
        )


def test_mha_input_config_rejects_mismatched_page_table_override() -> None:
    with pytest.raises(
        ValueError,
        match="cache_input.page_table_input.indexing=.*MHAInputConfig.indexing",
    ):
        MHAInputs(
            MHAInputConfig(
                batch_size=3,
                total_cached_tokens=36,
                total_new_q_tokens=12,
                num_q_heads=4,
                num_kv_heads=2,
                head_dim=16,
                q_dtype=torch.float32,
                cache_layout="paged",
                page_size=4,
                indexing="identity",
                cache_input=KVCacheInputConfig(
                    cache_layout="paged",
                    batch_size=3,
                    max_seqlen_k=1,
                    num_kv_heads=2,
                    head_dim=16,
                    dtype=torch.float32,
                    page_size=4,
                    page_table_input=PageTableInputConfig(
                        batch_size=3,
                        max_pages_per_request=1,
                        indexing="random",
                    ),
                ),
            )
        )


def test_mha_inputs_uses_nested_cache_config() -> None:
    cache_config = KVCacheInputConfig(
        cache_layout="paged",
        batch_size=3,
        max_seqlen_k=1,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        page_size=4,
        page_table_input=PageTableInputConfig(
            batch_size=3,
            max_pages_per_request=1,
            indexing="identity",
        ),
    )

    generator = MHAInputs(
        MHAInputConfig(
            batch_size=3,
            total_cached_tokens=36,
            total_new_q_tokens=12,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_dtype=torch.float32,
            cache_layout="paged",
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=3,
                total_cached_tokens=36,
                total_new_q_tokens=12,
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
                cache_layout="paged",
            ),
            cache_input=cache_config,
        )
    )

    assert generator.cache_input is not None
    assert generator.cache_input.config is cache_config

    inputs = generator.generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)
    assert page_ids == list(range(len(page_ids)))


def test_mla_prefill_generator_allows_independent_q_and_kv_lengths() -> None:
    inputs = MLAInputs(
        _mla_config(
            batch_size=3,
            total_cached_tokens=0,
            total_new_q_tokens=12,
            num_q_heads=4,
            num_kv_heads=2,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            kv_lora_rank=16,
            v_head_dim=6,
            q_dtype=torch.float32,
            cache_layout="none",
            metadata_kwargs={
                "total_new_kv_tokens": 15,
                "tie_new_kv_to_query": False,
                "new_q_length_mode": "regular",
                "new_kv_length_mode": "regular",
            },
        )
    ).generate(metadata_seed=1, value_seed=2, device="cpu")

    assert inputs.cache is None
    assert inputs.q is not None
    assert inputs.k is not None
    assert inputs.v is not None
    assert inputs.q.shape == (12, 4, 12)
    assert inputs.k.shape == (15, 2, 12)
    assert inputs.v.shape == (15, 2, 6)
    assert inputs.metadata.cu_seqlens_q_cpu[-1] == 12
    assert inputs.metadata.cu_seqlens_kv_cpu[-1] == 15


def test_mla_reference_prefill_supports_grouped_kv_heads() -> None:
    values = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=0,
            total_new_q_tokens=6,
            num_q_heads=4,
            num_kv_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            kv_lora_rank=8,
            v_head_dim=5,
            q_dtype=torch.float32,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "fixed_per_request"},
        )
    ).generate(metadata_seed=11, value_seed=12, device="cpu")

    ref = mla_reference(values, is_causal=False)

    assert values.q is not None
    assert values.v is not None
    assert ref.out.shape == (6, 4, 5)
    assert ref.out.dtype == torch.float32
    assert ref.lse.shape == (6, 4)
    assert torch.isfinite(ref.out).all()
    assert torch.isfinite(ref.lse).all()

    req = 0
    q_start = values.metadata.cu_seqlens_q_cpu[req]
    q_end = values.metadata.cu_seqlens_q_cpu[req + 1]
    kv_start = values.metadata.cu_seqlens_kv_cpu[req]
    kv_end = values.metadata.cu_seqlens_kv_cpu[req + 1]
    assert values.k is not None
    k = values.k[kv_start:kv_end].repeat_interleave(2, dim=1)
    v = values.v[kv_start:kv_end].repeat_interleave(2, dim=1)
    scores = (
        torch.einsum("qhd,khd->qkh", values.q[q_start:q_end], k) * values.softmax_scale
    )
    probs = torch.softmax(scores, dim=1)
    expected = torch.einsum("qkh,khd->qhd", probs, v)
    torch.testing.assert_close(ref.out[q_start:q_end], expected)


@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_mla_prefill_generator_supports_fp8_reference(dtype: torch.dtype) -> None:
    values = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=0,
            total_new_q_tokens=6,
            num_q_heads=4,
            num_kv_heads=2,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            kv_lora_rank=12,
            v_head_dim=6,
            q_dtype=dtype,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "fixed_per_request"},
        )
    ).generate(metadata_seed=17, value_seed=18, device="cpu")

    ref = mla_reference(values, is_causal=True)

    assert values.q is not None
    assert values.k is not None
    assert values.v is not None
    assert values.q.dtype == dtype
    assert values.k.dtype == dtype
    assert values.v.dtype == dtype
    assert ref.out.shape == (6, 4, 6)
    assert ref.out.dtype == torch.bfloat16
    assert ref.lse.shape == (6, 4)
    assert torch.isfinite(ref.out).all()
    assert torch.isfinite(ref.lse).all()


def test_mla_paged_decode_generator_shapes() -> None:
    inputs = MLAInputs(
        _mla_config(
            batch_size=3,
            total_cached_tokens=21,
            total_new_q_tokens=3,
            num_q_heads=4,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            kv_lora_rank=16,
            v_head_dim=6,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=4,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": 9,
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=1, value_seed=2, device="cpu")

    assert inputs.q is not None
    assert inputs.k is None
    assert inputs.v is None
    assert inputs.cache is not None
    assert inputs.q.shape == (3, 1, 4, 20)
    assert inputs.cache.kv_cache.shape == (9, 4, 1, 20)
    assert inputs.cache.page_table is not None
    assert inputs.cache.page_table.shape == (3, 3)
    assert inputs.metadata.cache_seqlens.tolist() == [8, 8, 8]


def test_mla_reference_paged_decode_shapes() -> None:
    values = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=10,
            total_new_q_tokens=2,
            num_q_heads=4,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            kv_lora_rank=12,
            v_head_dim=6,
            q_dtype=torch.float32,
            cache_layout="paged",
            page_size=4,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": 6,
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=13, value_seed=14, device="cpu")

    ref = mla_reference(values)

    assert values.q is not None
    assert ref.out.shape == (2, 1, 4, 12)
    assert ref.out.dtype == torch.float32
    assert ref.lse.shape == (2, 1, 4)
    assert torch.isfinite(ref.out).all()
    assert torch.isfinite(ref.lse).all()


@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_mla_paged_decode_generator_supports_fp8_reference(
    dtype: torch.dtype,
) -> None:
    values = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=10,
            total_new_q_tokens=2,
            num_q_heads=4,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            kv_lora_rank=12,
            v_head_dim=6,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=4,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": 6,
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=19, value_seed=20, device="cpu")

    ref = mla_reference(values)

    assert values.q is not None
    assert values.cache is not None
    assert values.q.dtype == dtype
    assert values.cache.kv_cache.dtype == dtype
    assert ref.out.shape == (2, 1, 4, 12)
    assert ref.out.dtype == torch.bfloat16
    assert ref.lse.shape == (2, 1, 4)
    assert torch.isfinite(ref.out).all()
    assert torch.isfinite(ref.lse).all()


def test_mla_reference_rejects_incompatible_prefill_heads() -> None:
    values = MLAInputs(
        _mla_config(
            batch_size=1,
            total_cached_tokens=0,
            total_new_q_tokens=4,
            num_q_heads=4,
            num_kv_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            kv_lora_rank=8,
            v_head_dim=5,
            q_dtype=torch.float32,
            cache_layout="none",
        )
    ).generate(metadata_seed=15, value_seed=16, device="cpu")
    assert values.k is not None
    assert values.v is not None
    values.k = torch.cat((values.k, values.k[:, :1, :]), dim=1)
    values.v = torch.cat((values.v, values.v[:, :1, :]), dim=1)

    with pytest.raises(ValueError, match="divisible"):
        mla_reference(values)
