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
    AttentionMergeStateInputConfig,
    AttentionMergeStateInputs,
    CompressedSequenceAttentionHistoryConfig,
    CompressedSequenceAttentionIndexerConfig,
    CompressedSequenceAttentionInputConfig,
    CompressedSequenceAttentionInputs,
    DeepSeekV4CompressorStateInputConfig,
    DeepSeekV4CompressorStateInputs,
    DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig,
    DeepSeekV4CSAIndexerMXFP4CacheInsertInputs,
    DeepSeekV4IndexerMXFP4CacheGatherInputConfig,
    DeepSeekV4IndexerMXFP4CacheGatherInputs,
    DeepSeekV4IndexerMXFP4CacheWriteInputConfig,
    DeepSeekV4IndexerMXFP4CacheWriteInputs,
    DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig,
    DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs,
    DeepSeekV4InvRoPEFP8QuantInputConfig,
    DeepSeekV4InvRoPEFP8QuantInputs,
    DeepSeekV4KCacheGatherInputConfig,
    DeepSeekV4KCacheGatherInputs,
    DeepSeekV4PagedIndexInputConfig,
    DeepSeekV4PagedIndexInputs,
    DeepSeekV4SparseCompressCacheInsertInputConfig,
    DeepSeekV4SparseCompressCacheInsertInputs,
    DeepSeekV4SparsePrefillIndexInputConfig,
    DeepSeekV4SparsePrefillIndexInputs,
    DSADecodeTopKInputConfig,
    DSADecodeTopKInputs,
    DSASparseDecodeKVPackInputConfig,
    DSASparseDecodeKVPackInputs,
    DSATopKSlotInputConfig,
    DSATopKSlotInputs,
    GDNChunkPrefillInputConfig,
    GDNChunkPrefillInputs,
    GDNQKVSplitInputConfig,
    GDNQKVSplitInputs,
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
    MLAPrefillFP8InputConfig,
    MLAPrefillFP8Inputs,
    PackedQKVComplexRotaryInputConfig,
    PackedQKVComplexRotaryInputs,
    PageTableInput,
    PageTableInputConfig,
    SlotMappingInput,
    SlotMappingInputConfig,
    attention_merge_state_reference,
    deepseek_v4_build_dense_prefill_local_compressed_indices_reference,
    deepseek_v4_combine_dense_swa_indices_reference,
    deepseek_v4_combine_topk_swa_indices_reference,
    deepseek_v4_compressed_slot_mapping_reference,
    deepseek_v4_compute_global_topk_indices_and_lens_reference,
    deepseek_v4_csa_indexer_mxfp4_cache_insert_reference,
    deepseek_v4_decode_swa_indices_and_lens_reference,
    deepseek_v4_dequantize_and_gather_k_cache_reference,
    deepseek_v4_indexer_decode_metadata_reference,
    deepseek_v4_indexer_mxfp4_cache_gather_reference,
    deepseek_v4_indexer_mxfp4_cache_write_reference,
    deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference,
    deepseek_v4_inv_rope_fp8_quant_reference,
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
    mla_prefill_fp8_reference,
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


def test_attention_merge_state_inputs_generate_values_and_reference() -> None:
    values = AttentionMergeStateInputs(
        AttentionMergeStateInputConfig(
            total_q=7,
            num_heads=3,
            head_dim=16,
            dtype=torch.float32,
            lse_bound=4.0,
        )
    ).generate(seed=1, device="cpu")

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
    values = AttentionMergeStateInputs(
        AttentionMergeStateInputConfig(
            total_q=5,
            num_heads=2,
            head_dim=8,
            dtype=torch.float32,
            lse_scale_log2=1.0,
        )
    ).generate(seed=2, device="cpu")

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
    with pytest.raises(ValueError, match="lse_scale_log2"):
        AttentionMergeStateInputs(
            AttentionMergeStateInputConfig(
                total_q=7,
                num_heads=3,
                head_dim=16,
                lse_scale_log2=0.0,
            )
        )


def test_mla_prefill_fp8_inputs_generate_values_and_reference() -> None:
    values = MLAPrefillFP8Inputs(
        MLAPrefillFP8InputConfig(
            batch_size=3,
            total_tokens=12,
            num_heads=4,
            qk_head_dim=8,
            v_head_dim=6,
            source_dtype=torch.bfloat16,
            length_mode="regular",
            fp8_dtype=torch.float8_e4m3fn,
        )
    ).generate(seed=21, metadata_seed=22, device="cpu")

    assert values.query.shape == (12, 4, 8)
    assert values.key.shape == (12, 4, 8)
    assert values.value.shape == (12, 4, 6)
    assert values.query.dtype == torch.float8_e4m3fn
    assert values.key.dtype == torch.float8_e4m3fn
    assert values.value.dtype == torch.float8_e4m3fn
    assert values.metadata.cu_seqlens_q_cpu == values.metadata.cu_seqlens_kv_cpu
    assert values.metadata.max_seqlen_q == 4
    assert values.metadata.resolved_max_seqlen_k == 4

    refs = mla_prefill_fp8_reference(values, is_causal=True)
    assert refs.out.shape == (12, 4, 6)
    assert refs.out.dtype == torch.bfloat16
    assert refs.lse.shape == (12, 4)
    assert refs.lse.dtype == torch.float32
    assert torch.isfinite(refs.out.float()).all()
    assert torch.isfinite(refs.lse).all()


def test_mla_prefill_fp8_metadata_seed_controls_sequence_layout() -> None:
    generator = MLAPrefillFP8Inputs(
        MLAPrefillFP8InputConfig(
            batch_size=4,
            total_tokens=19,
            num_heads=2,
            qk_head_dim=6,
            v_head_dim=5,
            source_dtype=torch.float16,
            length_mode="ragged",
            max_tokens_per_request=8,
        )
    )

    values1 = generator.generate(seed=31, metadata_seed=99, device="cpu")
    values2 = generator.generate(seed=32, metadata_seed=99, device="cpu")

    assert values1.metadata.cu_seqlens_q_cpu == values2.metadata.cu_seqlens_q_cpu
    assert values1.metadata.cu_seqlens_kv_cpu == values2.metadata.cu_seqlens_kv_cpu
    assert not torch.equal(
        values1.query.view(torch.uint8),
        values2.query.view(torch.uint8),
    )


def test_mla_prefill_fp8_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="source_dtype"):
        MLAPrefillFP8Inputs(
            MLAPrefillFP8InputConfig(
                batch_size=1,
                total_tokens=4,
                num_heads=2,
                qk_head_dim=8,
                v_head_dim=6,
                source_dtype=torch.int32,
            )
        )

    with pytest.raises(ValueError, match="fp8_dtype"):
        MLAPrefillFP8Inputs(
            MLAPrefillFP8InputConfig(
                batch_size=1,
                total_tokens=4,
                num_heads=2,
                qk_head_dim=8,
                v_head_dim=6,
                source_dtype=torch.float32,
                fp8_dtype=torch.bfloat16,
            )
        )


def test_mla_prefill_fp8_reference_rejects_bad_shapes() -> None:
    values = MLAPrefillFP8Inputs(
        MLAPrefillFP8InputConfig(
            batch_size=2,
            total_tokens=6,
            num_heads=2,
            qk_head_dim=8,
            v_head_dim=6,
            source_dtype=torch.bfloat16,
            length_mode="fixed_per_request",
        )
    ).generate(seed=41, device="cpu")
    values.key = values.key[:, :1, :]

    with pytest.raises(ValueError, match="matching head counts"):
        mla_prefill_fp8_reference(values)


def test_gdn_qkv_split_inputs_generate_plain_split_reference() -> None:
    values = GDNQKVSplitInputs(
        GDNQKVSplitInputConfig(
            num_tokens=5,
            num_q_heads=4,
            num_k_heads=2,
            num_v_heads=2,
            head_q=8,
            head_k=8,
            head_v=6,
            dtype=torch.float32,
        )
    ).generate(seed=101, device="cpu")

    q, k, v = gdn_qkv_split_reference(values)

    assert values.mixed_qkv.shape == (5, 60)
    assert q.shape == (1, 5, 4, 8)
    assert k.shape == (1, 5, 2, 8)
    assert v.shape == (1, 5, 2, 6)
    torch.testing.assert_close(q.reshape(5, -1), values.mixed_qkv[:, :32])
    torch.testing.assert_close(k.reshape(5, -1), values.mixed_qkv[:, 32:48])
    torch.testing.assert_close(v.reshape(5, -1), values.mixed_qkv[:, 48:])


def test_gdn_qkv_split_inputs_generate_l2norm_reference() -> None:
    values = GDNQKVSplitInputs(
        GDNQKVSplitInputConfig(
            num_tokens=7,
            num_q_heads=3,
            num_k_heads=2,
            num_v_heads=2,
            head_q=8,
            head_k=8,
            head_v=8,
            dtype=torch.float32,
            fuse_l2norm=True,
        )
    ).generate(seed=102, device="cpu")

    q, k, v = gdn_qkv_split_reference(values)

    torch.testing.assert_close(
        torch.linalg.vector_norm(q, dim=-1),
        torch.ones((1, 7, 3)),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(k, dim=-1),
        torch.ones((1, 7, 2)),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(v.reshape(7, -1), values.mixed_qkv[:, -16:])


def test_gdn_qkv_split_inputs_reject_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="head_q"):
        GDNQKVSplitInputs(
            GDNQKVSplitInputConfig(
                num_tokens=5,
                num_q_heads=4,
                num_k_heads=2,
                num_v_heads=2,
                head_q=0,
                head_k=8,
                head_v=8,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="l2norm_eps"):
        GDNQKVSplitInputs(
            GDNQKVSplitInputConfig(
                num_tokens=5,
                num_q_heads=4,
                num_k_heads=2,
                num_v_heads=2,
                head_q=8,
                head_k=8,
                head_v=8,
                dtype=torch.float32,
                l2norm_eps=0.0,
            )
        )

    values = GDNQKVSplitInputs(
        GDNQKVSplitInputConfig(
            num_tokens=5,
            num_q_heads=4,
            num_k_heads=2,
            num_v_heads=2,
            head_q=8,
            head_k=8,
            head_v=8,
            dtype=torch.float32,
        )
    ).generate(seed=103, device="cpu")
    values.mixed_qkv = values.mixed_qkv[:, :-1]
    with pytest.raises(ValueError, match="last dimension"):
        gdn_qkv_split_reference(values)


def test_gdn_chunk_prefill_inputs_generate_values_and_reference() -> None:
    values = GDNChunkPrefillInputs(
        GDNChunkPrefillInputConfig(
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
    values = GDNChunkPrefillInputs(
        GDNChunkPrefillInputConfig(
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
    generator = GDNChunkPrefillInputs(
        GDNChunkPrefillInputConfig(
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
        GDNChunkPrefillInputs(
            GDNChunkPrefillInputConfig(
                batch_size=2,
                total_tokens=8,
                num_q_heads=4,
                num_v_heads=2,
                head_dim=8,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="integer multiple"):
        GDNChunkPrefillInputs(
            GDNChunkPrefillInputConfig(
                batch_size=2,
                total_tokens=8,
                num_q_heads=2,
                num_v_heads=3,
                head_dim=8,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="total_tokens must be >= batch_size"):
        GDNChunkPrefillInputs(
            GDNChunkPrefillInputConfig(
                batch_size=4,
                total_tokens=3,
                num_q_heads=2,
                num_v_heads=2,
                head_dim=8,
                dtype=torch.float32,
            )
        )

    values = GDNChunkPrefillInputs(
        GDNChunkPrefillInputConfig(
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
    values = PackedQKVComplexRotaryInputs(
        PackedQKVComplexRotaryInputConfig(
            num_tokens=5,
            num_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=111, device="cpu")

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
    values = PackedQKVComplexRotaryInputs(
        PackedQKVComplexRotaryInputConfig(
            num_tokens=3,
            num_heads=2,
            head_dim=6,
            dtype=torch.float32,
            copy_v=True,
        )
    ).generate(seed=112, device="cpu")

    _, _, v = packed_qkv_complex_rotary_reference(values)

    assert v.is_contiguous()
    torch.testing.assert_close(v.reshape(3, -1), values.qkv[:, 24:])


def test_packed_qkv_complex_rotary_inputs_reject_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        PackedQKVComplexRotaryInputs(
            PackedQKVComplexRotaryInputConfig(
                num_tokens=5,
                num_heads=2,
                head_dim=7,
                dtype=torch.float32,
            )
        )

    values = PackedQKVComplexRotaryInputs(
        PackedQKVComplexRotaryInputConfig(
            num_tokens=5,
            num_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=113, device="cpu")
    values.freqs_cis = values.freqs_cis[:-1]
    with pytest.raises(ValueError, match="freqs_cis"):
        packed_qkv_complex_rotary_reference(values)

    values = PackedQKVComplexRotaryInputs(
        PackedQKVComplexRotaryInputConfig(
            num_tokens=5,
            num_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )
    ).generate(seed=114, device="cpu")
    values.freqs_cis = values.freqs_cis.real
    with pytest.raises(TypeError, match="complex"):
        packed_qkv_complex_rotary_reference(values)


def test_dsa_sparse_decode_kv_pack_inputs_generate_values_and_reference() -> None:
    values = DSASparseDecodeKVPackInputs(
        DSASparseDecodeKVPackInputConfig(
            num_tokens=5,
            num_slots=9,
            nope_dim=128,
            rope_dim=64,
        )
    ).generate(seed=118, metadata_seed=119, device="cpu")

    expected_row_bytes = dsa_sparse_decode_row_bytes(128, 64)
    packed = dsa_sparse_decode_kv_pack_reference(values)
    loc = values.loc.to(torch.int64)
    scale_offset = 128
    rope_offset = scale_offset + 4

    assert values.out.shape == (9, expected_row_bytes)
    assert values.loc.shape == (5,)
    assert torch.unique(values.loc).numel() == values.loc.numel()
    assert values.cache_k_nope.shape == (5, 128)
    assert values.cache_k_rope.shape == (5, 64)
    assert packed.shape == values.out.shape
    assert packed.dtype == torch.uint8

    written_mask = torch.zeros(values.out.shape[0], dtype=torch.bool)
    written_mask[loc.cpu()] = True
    assert torch.equal(packed[~written_mask], values.out[~written_mask])

    scales = packed[loc, scale_offset:rope_offset].contiguous().view(torch.float32)
    assert scales.shape == (5, 1)
    assert torch.all(scales > 0)
    expected_rope_bytes = values.cache_k_rope.contiguous().view(torch.uint8)
    assert torch.equal(packed[loc, rope_offset:], expected_rope_bytes)


def test_dsa_sparse_decode_kv_pack_supports_head_axis_and_zero_tokens() -> None:
    values = DSASparseDecodeKVPackInputs(
        DSASparseDecodeKVPackInputConfig(
            num_tokens=0,
            num_slots=3,
            nope_dim=128,
            rope_dim=32,
            include_head_axis=True,
        )
    ).generate(seed=120, device="cpu")

    packed = dsa_sparse_decode_kv_pack_reference(values)

    assert values.loc.shape == (0,)
    assert values.cache_k_nope.shape == (0, 1, 128)
    assert values.cache_k_rope.shape == (0, 1, 32)
    assert torch.equal(packed, values.out)


def test_dsa_sparse_decode_kv_pack_rejects_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="num_tokens"):
        DSASparseDecodeKVPackInputs(
            DSASparseDecodeKVPackInputConfig(
                num_tokens=4,
                num_slots=3,
                nope_dim=128,
                rope_dim=64,
            )
        )

    with pytest.raises(ValueError, match="divisible"):
        DSASparseDecodeKVPackInputs(
            DSASparseDecodeKVPackInputConfig(
                num_tokens=2,
                num_slots=3,
                nope_dim=64,
                rope_dim=64,
            )
        )

    with pytest.raises(ValueError, match="power of two"):
        DSASparseDecodeKVPackInputs(
            DSASparseDecodeKVPackInputConfig(
                num_tokens=2,
                num_slots=3,
                nope_dim=128,
                rope_dim=48,
            )
        )

    values = DSASparseDecodeKVPackInputs(
        DSASparseDecodeKVPackInputConfig(
            num_tokens=3,
            num_slots=5,
            nope_dim=128,
            rope_dim=64,
        )
    ).generate(seed=121, device="cpu")
    values.loc = torch.tensor([0, 0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="unique"):
        dsa_sparse_decode_kv_pack_reference(values)


def test_dsa_decode_topk_inputs_generate_stable_tie_values() -> None:
    values = DSADecodeTopKInputs(
        DSADecodeTopKInputConfig(
            num_rows=4,
            vocab_size=12,
            topk=3,
            dtype=torch.float32,
            min_valid_len=6,
            max_valid_len=12,
        )
    ).generate(seed=188, device="cpu")

    expected = dsa_decode_topk_reference(values)

    assert values.logits.shape == (4, 12)
    assert values.out.shape == (4, 3)
    assert values.out.dtype == torch.int32
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
    values = DSADecodeTopKInputs(
        DSADecodeTopKInputConfig(
            num_rows=1,
            vocab_size=5,
            topk=3,
            dtype=torch.float32,
            min_valid_len=5,
            max_valid_len=5,
        )
    ).generate(seed=189, device="cpu")
    values.logits[0] = torch.tensor([1.0, 3.0, 3.0, 2.0, -float("inf")])
    values.valid_lens[0] = 4

    expected = dsa_decode_topk_reference(values)

    torch.testing.assert_close(
        expected,
        torch.tensor([[1, 2, 3]], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_dsa_decode_topk_inputs_reject_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="topk must be <= vocab_size"):
        DSADecodeTopKInputs(
            DSADecodeTopKInputConfig(
                num_rows=1,
                vocab_size=4,
                topk=5,
                dtype=torch.float32,
            )
        )

    with pytest.raises(ValueError, match="min_valid_len must be >= topk"):
        DSADecodeTopKInputs(
            DSADecodeTopKInputConfig(
                num_rows=1,
                vocab_size=8,
                topk=4,
                dtype=torch.float32,
                min_valid_len=3,
            )
        )

    values = DSADecodeTopKInputs(
        DSADecodeTopKInputConfig(
            num_rows=1,
            vocab_size=8,
            topk=4,
            dtype=torch.float32,
            min_valid_len=6,
            max_valid_len=6,
        )
    ).generate(seed=190, device="cpu")
    values.logits[0, int(values.valid_lens[0].item()) :] = 0.0

    with pytest.raises(ValueError, match="masked with -inf"):
        dsa_decode_topk_reference(values)


def test_dsa_topk_slot_inputs_generate_values_and_references() -> None:
    values = DSATopKSlotInputs(
        DSATopKSlotInputConfig(
            num_tokens=5,
            topk=4,
            block_size=8,
            max_pages_per_token=4,
            max_seq_len=24,
            indexing="identity",
        )
    ).generate(seed=121, device="cpu")

    local_slots, local_lens = dsa_local_topk_to_global_slots_reference(values)
    full_slots, full_lens = dsa_full_context_topk_to_global_slots_reference(values)

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
    values = DSATopKSlotInputs(
        DSATopKSlotInputConfig(
            num_tokens=3,
            topk=5,
            block_size=4,
            max_pages_per_token=3,
            max_seq_len=12,
            indexing="identity",
        )
    ).generate(seed=122, device="cpu")

    slots, lens = dsa_local_topk_to_global_slots_reference(
        values,
        use_seq_lens=False,
    )

    assert slots.shape == (3, 5)
    assert lens.shape == (3,)
    assert torch.all(lens > 0)


def test_dsa_topk_slot_inputs_support_zero_tokens() -> None:
    values = DSATopKSlotInputs(
        DSATopKSlotInputConfig(
            num_tokens=0,
            topk=5,
            block_size=4,
            max_pages_per_token=3,
        )
    ).generate(seed=124, device="cpu")

    slots, lens = dsa_full_context_topk_to_global_slots_reference(values)

    assert values.local_topk_offsets.shape == (0, 5)
    assert values.seq_lens.shape == (0,)
    assert values.block_table.shape == (1, 3)
    assert slots.shape == (0, 5)
    assert lens.shape == (0,)


def test_dsa_topk_slot_inputs_reject_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="topk"):
        DSATopKSlotInputs(
            DSATopKSlotInputConfig(
                num_tokens=3,
                topk=0,
                block_size=8,
                max_pages_per_token=2,
            )
        )

    with pytest.raises(ValueError, match="page_table_input.batch_size"):
        DSATopKSlotInputs(
            DSATopKSlotInputConfig(
                num_tokens=3,
                topk=4,
                block_size=8,
                max_pages_per_token=2,
                page_table_input=PageTableInputConfig(
                    batch_size=2,
                    max_pages_per_request=2,
                ),
            )
        )

    values = DSATopKSlotInputs(
        DSATopKSlotInputConfig(
            num_tokens=3,
            topk=4,
            block_size=8,
            max_pages_per_token=2,
        )
    ).generate(seed=123, device="cpu")
    values.seq_lens = values.seq_lens[:-1]
    with pytest.raises(ValueError, match="seq_lens"):
        dsa_local_topk_to_global_slots_reference(values)


def test_compressed_sequence_attention_generates_swa_only_values() -> None:
    values = CompressedSequenceAttentionInputs(
        CompressedSequenceAttentionInputConfig(
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
    values = CompressedSequenceAttentionInputs(
        CompressedSequenceAttentionInputConfig(
            batch_size=2,
            total_cached_tokens=16,
            total_new_q_tokens=4,
            dtype=torch.float32,
            num_q_heads=2,
            page_size=4,
            window_size=5,
            compressed=CompressedSequenceAttentionHistoryConfig(
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
    values = CompressedSequenceAttentionInputs(
        CompressedSequenceAttentionInputConfig(
            batch_size=2,
            total_cached_tokens=16,
            total_new_q_tokens=4,
            dtype=torch.float32,
            num_q_heads=2,
            page_size=4,
            window_size=5,
            compressed=CompressedSequenceAttentionHistoryConfig(
                compress_ratio=4,
                topk=3,
                num_state_cache_blocks=2,
                compressor_block_size=4,
                num_kv_cache_blocks=2,
                kv_cache_block_size=4,
            ),
            indexer=CompressedSequenceAttentionIndexerConfig(
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
        CompressedSequenceAttentionInputs(
            CompressedSequenceAttentionInputConfig(
                batch_size=1,
                total_cached_tokens=1,
                total_new_q_tokens=1,
                dtype=torch.float32,
                num_q_heads=1,
                page_size=1,
                window_size=1,
                indexer=CompressedSequenceAttentionIndexerConfig(),
            )
        )

    with pytest.raises(ValueError, match="only valid for CSA"):
        CompressedSequenceAttentionInputs(
            CompressedSequenceAttentionInputConfig(
                batch_size=1,
                total_cached_tokens=1,
                total_new_q_tokens=1,
                dtype=torch.float32,
                num_q_heads=1,
                page_size=1,
                window_size=1,
                compressed=CompressedSequenceAttentionHistoryConfig(
                    compress_ratio=128,
                ),
                indexer=CompressedSequenceAttentionIndexerConfig(),
            )
        )


def test_deepseek_v4_compressor_state_inputs_generate_values_and_reference() -> None:
    values = DeepSeekV4CompressorStateInputs(
        DeepSeekV4CompressorStateInputConfig(
            num_tokens=6,
            state_width=16,
            num_cache_blocks=2,
            block_size=4,
            compress_ratio=8,
            dtype=torch.bfloat16,
            invalid_token_count=2,
        )
    ).generate(metadata_seed=23, value_seed=37, device="cpu")

    expected = deepseek_v4_save_compressor_state_reference(values)

    assert values.kv.shape == (6, 16)
    assert values.score.shape == (6, 16)
    assert values.ape.shape == (8, 16)
    assert values.state_cache.shape == (2, 4, 32)
    assert values.positions.tolist() == list(range(6))
    assert int((values.slot_mapping < 0).sum().item()) == 2
    assert expected.shape == values.state_cache.shape

    slots = values.slot_mapping.to(torch.int64)
    for token_idx, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block_idx = slot // values.block_size
        pos_in_block = slot % values.block_size
        ape_row = int(values.positions[token_idx].item()) % values.compress_ratio
        torch.testing.assert_close(
            expected[block_idx, pos_in_block, :16],
            values.kv[token_idx].float(),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            expected[block_idx, pos_in_block, 16:],
            values.score[token_idx].float() + values.ape[ape_row],
            rtol=0.0,
            atol=0.0,
        )

    written_slots = {int(slot) for slot in slots.tolist() if slot >= 0}
    for slot in range(values.state_cache.shape[0] * values.block_size):
        if slot in written_slots:
            continue
        block_idx = slot // values.block_size
        pos_in_block = slot % values.block_size
        torch.testing.assert_close(
            expected[block_idx, pos_in_block],
            values.state_cache[block_idx, pos_in_block],
            rtol=0.0,
            atol=0.0,
        )


def test_deepseek_v4_compressor_state_reference_matches_c4_overlap_ape_layout() -> None:
    values = DeepSeekV4CompressorStateInputs(
        DeepSeekV4CompressorStateInputConfig(
            num_tokens=1,
            state_width=8,
            num_cache_blocks=1,
            block_size=1,
            compress_ratio=4,
            dtype=torch.float32,
        )
    ).generate(seed=41, device="cpu")
    values.kv = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    values.score = torch.zeros((1, 8), dtype=torch.float32)
    values.ape = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    values.state_cache = torch.zeros((1, 1, 16), dtype=torch.float32)
    values.slot_mapping = torch.tensor([0], dtype=torch.int64)
    values.positions = torch.tensor([1], dtype=torch.int64)

    expected = deepseek_v4_save_compressor_state_reference(values)

    torch.testing.assert_close(expected[0, 0, :8], values.kv[0])
    torch.testing.assert_close(
        expected[0, 0, 8:],
        torch.tensor([4, 5, 6, 7, 20, 21, 22, 23], dtype=torch.float32),
    )


def test_deepseek_v4_compressor_state_inputs_keep_metadata_seed_independent() -> None:
    generator = DeepSeekV4CompressorStateInputs(
        DeepSeekV4CompressorStateInputConfig(
            num_tokens=5,
            state_width=8,
            num_cache_blocks=2,
            block_size=3,
            compress_ratio=4,
            dtype=torch.float32,
            invalid_token_count=1,
        )
    )

    first = generator.generate(metadata_seed=51, value_seed=61, device="cpu")
    same_metadata = generator.generate(metadata_seed=51, value_seed=62, device="cpu")
    same_values = generator.generate(metadata_seed=52, value_seed=61, device="cpu")

    torch.testing.assert_close(first.slot_mapping, same_metadata.slot_mapping)
    torch.testing.assert_close(first.positions, same_metadata.positions)
    assert not torch.equal(first.kv, same_metadata.kv)
    torch.testing.assert_close(first.kv, same_values.kv)
    assert not torch.equal(first.slot_mapping, same_values.slot_mapping)


def test_deepseek_v4_compressor_state_rejects_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="valid generated tokens"):
        DeepSeekV4CompressorStateInputs(
            DeepSeekV4CompressorStateInputConfig(
                num_tokens=5,
                state_width=8,
                num_cache_blocks=1,
                block_size=4,
                compress_ratio=4,
                dtype=torch.float32,
            )
        )

    values = DeepSeekV4CompressorStateInputs(
        DeepSeekV4CompressorStateInputConfig(
            num_tokens=2,
            state_width=8,
            num_cache_blocks=1,
            block_size=4,
            compress_ratio=4,
            dtype=torch.float32,
        )
    ).generate(seed=71, device="cpu")
    values.slot_mapping = torch.tensor([0, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="unique"):
        deepseek_v4_save_compressor_state_reference(values)


def test_deepseek_v4_indexer_q_rope_hadamard_mxfp4_generate_reference() -> None:
    values = DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(
        DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig(
            num_tokens=4,
            num_heads=3,
            dtype=torch.bfloat16,
            max_position=32,
            softmax_scale=0.25,
            head_scale=2.0,
        )
    ).generate(metadata_seed=51, value_seed=61, device="cpu")

    (q_packed, q_scale), weights = deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference(
        values
    )

    assert values.index_q.shape == (4, 3, 128)
    assert values.positions.shape == (4,)
    assert values.cos_sin_cache.shape == (32, 64)
    assert values.weights.shape == (4, 3)
    assert q_packed.shape == (4, 3, 64)
    assert q_scale.shape == (4, 3)
    assert q_packed.dtype == torch.uint8
    assert q_scale.dtype == torch.int32
    assert weights.shape == values.weights.shape
    assert torch.isfinite(weights).all()
    assert int(values.positions.min().item()) >= 0
    assert int(values.positions.max().item()) < values.cos_sin_cache.shape[0]


def test_deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference_zero_row() -> None:
    values = DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(
        DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig(
            num_tokens=1,
            num_heads=1,
            dtype=torch.float32,
            max_position=4,
        )
    ).generate(seed=71, device="cpu")
    values.index_q = torch.zeros_like(values.index_q)
    values.weights = torch.ones_like(values.weights)

    (q_packed, q_scale), weights = deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference(
        values
    )

    torch.testing.assert_close(q_packed, torch.zeros_like(q_packed), rtol=0, atol=0)
    expected_scale = torch.full((1, 1, 4), 112, dtype=torch.uint8).view(torch.int32)
    torch.testing.assert_close(q_scale, expected_scale.squeeze(-1), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_deepseek_v4_indexer_q_rope_hadamard_mxfp4_keeps_metadata_seed_independent() -> (
    None
):
    generator = DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(
        DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig(
            num_tokens=4,
            num_heads=2,
            dtype=torch.float32,
            max_position=128,
        )
    )

    first = generator.generate(metadata_seed=81, value_seed=91, device="cpu")
    same_metadata = generator.generate(metadata_seed=81, value_seed=92, device="cpu")
    same_values = generator.generate(metadata_seed=82, value_seed=91, device="cpu")

    torch.testing.assert_close(first.positions, same_metadata.positions)
    torch.testing.assert_close(first.cos_sin_cache, same_metadata.cos_sin_cache)
    assert not torch.equal(first.index_q, same_metadata.index_q)
    assert not torch.equal(first.weights, same_metadata.weights)
    torch.testing.assert_close(first.index_q, same_values.index_q)
    torch.testing.assert_close(first.weights, same_values.weights)
    assert not torch.equal(first.positions, same_values.positions)


def test_deepseek_v4_indexer_q_rope_hadamard_mxfp4_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_position"):
        DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(
            DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig(
                num_tokens=1,
                num_heads=1,
                dtype=torch.float32,
                max_position=0,
            )
        )
    with pytest.raises(TypeError, match="position_dtype"):
        DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(
            DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig(
                num_tokens=1,
                num_heads=1,
                dtype=torch.float32,
                position_dtype=torch.float32,
            )
        )

    values = DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(
        DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig(
            num_tokens=1,
            num_heads=1,
            dtype=torch.float32,
            max_position=2,
        )
    ).generate(seed=101, device="cpu")
    values.positions = torch.tensor([2], dtype=torch.int64)
    with pytest.raises(ValueError, match="positions"):
        deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference(values)


def test_deepseek_v4_inv_rope_fp8_quant_inputs_generate_tma_reference() -> None:
    values = DeepSeekV4InvRoPEFP8QuantInputs(
        DeepSeekV4InvRoPEFP8QuantInputConfig(
            num_tokens=5,
            n_groups=2,
            heads_per_group=2,
            dtype=torch.bfloat16,
            max_position=32,
            value_scale=0.25,
        )
    ).generate(metadata_seed=111, value_seed=121, device="cpu")

    fp8, scales = deepseek_v4_inv_rope_fp8_quant_reference(values)

    assert values.o.shape == (5, 4, 512)
    assert values.positions.shape == (5,)
    assert values.cos_sin_cache.shape == (32, 64)
    assert fp8.shape == (5, 2, 1024)
    assert fp8.dtype == torch.float8_e4m3fn
    assert scales.shape == (5, 2, 2)
    assert scales.dtype == torch.int32
    assert int(values.positions.min().item()) >= 0
    assert int(values.positions.max().item()) < values.cos_sin_cache.shape[0]


def test_deepseek_v4_inv_rope_fp8_quant_inputs_generate_float_scale_reference() -> None:
    values = DeepSeekV4InvRoPEFP8QuantInputs(
        DeepSeekV4InvRoPEFP8QuantInputConfig(
            num_tokens=3,
            n_groups=1,
            heads_per_group=2,
            dtype=torch.float32,
            tma_aligned_scales=False,
            value_scale=0.25,
        )
    ).generate(seed=131, device="cpu")

    fp8, scales = deepseek_v4_inv_rope_fp8_quant_reference(values)

    assert fp8.shape == (3, 1, 1024)
    assert scales.shape == (3, 1, 8)
    assert scales.dtype == torch.float32
    assert torch.all(scales > 0)


def test_deepseek_v4_inv_rope_fp8_quant_keeps_metadata_seed_independent() -> None:
    generator = DeepSeekV4InvRoPEFP8QuantInputs(
        DeepSeekV4InvRoPEFP8QuantInputConfig(
            num_tokens=4,
            n_groups=2,
            heads_per_group=1,
            dtype=torch.float32,
            max_position=128,
        )
    )

    first = generator.generate(metadata_seed=141, value_seed=151, device="cpu")
    same_metadata = generator.generate(metadata_seed=141, value_seed=152, device="cpu")
    same_values = generator.generate(metadata_seed=142, value_seed=151, device="cpu")

    torch.testing.assert_close(first.positions, same_metadata.positions)
    torch.testing.assert_close(first.cos_sin_cache, same_metadata.cos_sin_cache)
    assert not torch.equal(first.o, same_metadata.o)
    torch.testing.assert_close(first.o, same_values.o)
    assert not torch.equal(first.positions, same_values.positions)


def test_deepseek_v4_inv_rope_fp8_quant_rejects_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="nope_dim \\+ rope_dim"):
        DeepSeekV4InvRoPEFP8QuantInputs(
            DeepSeekV4InvRoPEFP8QuantInputConfig(
                num_tokens=1,
                n_groups=1,
                heads_per_group=1,
                dtype=torch.float32,
                head_dim=512,
                nope_dim=384,
                rope_dim=64,
            )
        )
    with pytest.raises(ValueError, match="tma_aligned_scales"):
        DeepSeekV4InvRoPEFP8QuantInputs(
            DeepSeekV4InvRoPEFP8QuantInputConfig(
                num_tokens=1,
                n_groups=1,
                heads_per_group=1,
                dtype=torch.float32,
                head_dim=256,
                nope_dim=192,
                rope_dim=64,
                tma_aligned_scales=True,
            )
        )

    values = DeepSeekV4InvRoPEFP8QuantInputs(
        DeepSeekV4InvRoPEFP8QuantInputConfig(
            num_tokens=1,
            n_groups=1,
            heads_per_group=1,
            dtype=torch.float32,
            max_position=2,
        )
    ).generate(seed=161, device="cpu")
    values.positions = torch.tensor([2], dtype=torch.int64)
    with pytest.raises(ValueError, match="positions"):
        deepseek_v4_inv_rope_fp8_quant_reference(values)


def test_deepseek_v4_csa_indexer_mxfp4_cache_insert_inputs_generate_reference() -> None:
    values = DeepSeekV4CSAIndexerMXFP4CacheInsertInputs(
        DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig(
            num_tokens=6,
            batch_size=2,
            max_seq_len=12,
            num_state_cache_blocks=4,
            compressor_block_size=4,
            num_kv_cache_blocks=2,
            kv_cache_block_size=4,
            dtype=torch.bfloat16,
            non_boundary_token_count=1,
            include_block_table_base_offsets=True,
            value_scale=0.25,
        )
    ).generate(metadata_seed=171, value_seed=181, device="cpu")

    expected = deepseek_v4_csa_indexer_mxfp4_cache_insert_reference(values)

    assert values.state_cache.shape == (4, 4, 512)
    assert values.block_table.shape == (2, 3)
    assert values.kv_cache_2d.shape == (2, 4 * 68)
    assert values.rms_norm_weight.shape == (128,)
    assert values.cos_sin_cache.shape == (12, 64)
    assert values.block_table_base_offsets is not None
    assert expected.shape == values.kv_cache_2d.shape

    writable_slots = []
    for row_idx in range(values.positions.numel()):
        slot = int(values.kv_slot_mapping[row_idx].item())
        writable = (
            int(values.compressor_slot_mapping[row_idx].item()) >= 0
            and slot >= 0
            and (int(values.positions[row_idx].item()) + 1) % values.compress_ratio == 0
        )
        if not writable:
            continue
        writable_slots.append(slot)
        page = slot // values.kv_cache_block_size
        pos = slot % values.kv_cache_block_size
        value_base = pos * 64
        scale_base = values.kv_cache_block_size * 64 + pos * 4
        assert not torch.equal(
            expected[page, value_base : value_base + 64],
            values.kv_cache_2d[page, value_base : value_base + 64],
        )
        assert not torch.equal(
            expected[page, scale_base : scale_base + 4],
            values.kv_cache_2d[page, scale_base : scale_base + 4],
        )
    assert writable_slots


def test_deepseek_v4_csa_indexer_mxfp4_cache_insert_skips_invalid_rows() -> None:
    values = DeepSeekV4CSAIndexerMXFP4CacheInsertInputs(
        DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig(
            num_tokens=3,
            batch_size=1,
            max_seq_len=4,
            num_state_cache_blocks=1,
            compressor_block_size=4,
            num_kv_cache_blocks=1,
            kv_cache_block_size=4,
            dtype=torch.float32,
            non_boundary_token_count=1,
            negative_compressor_slot_count=1,
            negative_kv_slot_count=1,
        )
    ).generate(seed=191, device="cpu")
    values.positions = torch.tensor([3, 2, 3], dtype=torch.int64)
    values.compressor_slot_mapping = torch.tensor([0, 1, -1], dtype=torch.int64)
    values.kv_slot_mapping = torch.tensor([0, -1, 2], dtype=torch.int64)

    expected = deepseek_v4_csa_indexer_mxfp4_cache_insert_reference(values)

    assert not torch.equal(expected[0, :64], values.kv_cache_2d[0, :64])
    torch.testing.assert_close(
        expected[0, 64:128],
        values.kv_cache_2d[0, 64:128],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        expected[0, 128:192],
        values.kv_cache_2d[0, 128:192],
        rtol=0,
        atol=0,
    )


def test_deepseek_v4_csa_indexer_mxfp4_cache_insert_keeps_metadata_seed_independent() -> (
    None
):
    generator = DeepSeekV4CSAIndexerMXFP4CacheInsertInputs(
        DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig(
            num_tokens=5,
            batch_size=2,
            max_seq_len=8,
            num_state_cache_blocks=3,
            compressor_block_size=4,
            num_kv_cache_blocks=2,
            kv_cache_block_size=4,
            dtype=torch.float32,
            non_boundary_token_count=1,
        )
    )

    first = generator.generate(metadata_seed=201, value_seed=211, device="cpu")
    same_metadata = generator.generate(metadata_seed=201, value_seed=212, device="cpu")
    same_values = generator.generate(metadata_seed=202, value_seed=211, device="cpu")

    torch.testing.assert_close(first.positions, same_metadata.positions)
    torch.testing.assert_close(
        first.compressor_slot_mapping,
        same_metadata.compressor_slot_mapping,
    )
    torch.testing.assert_close(first.kv_slot_mapping, same_metadata.kv_slot_mapping)
    torch.testing.assert_close(first.block_table, same_metadata.block_table)
    assert not torch.equal(first.state_cache, same_metadata.state_cache)
    torch.testing.assert_close(first.state_cache, same_values.state_cache)
    assert not torch.equal(first.positions, same_values.positions)


def test_deepseek_v4_csa_indexer_mxfp4_cache_insert_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="compressor slots"):
        DeepSeekV4CSAIndexerMXFP4CacheInsertInputs(
            DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig(
                num_tokens=5,
                batch_size=1,
                max_seq_len=4,
                num_state_cache_blocks=1,
                compressor_block_size=4,
                num_kv_cache_blocks=2,
                kv_cache_block_size=4,
                dtype=torch.float32,
            )
        )
    with pytest.raises(ValueError, match="compress_ratio=4"):
        DeepSeekV4CSAIndexerMXFP4CacheInsertInputs(
            DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig(
                num_tokens=1,
                batch_size=1,
                max_seq_len=4,
                num_state_cache_blocks=1,
                compressor_block_size=4,
                num_kv_cache_blocks=1,
                kv_cache_block_size=4,
                dtype=torch.float32,
                compress_ratio=8,
            )
        )

    values = DeepSeekV4CSAIndexerMXFP4CacheInsertInputs(
        DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig(
            num_tokens=2,
            batch_size=1,
            max_seq_len=4,
            num_state_cache_blocks=1,
            compressor_block_size=4,
            num_kv_cache_blocks=1,
            kv_cache_block_size=4,
            dtype=torch.float32,
        )
    ).generate(seed=221, device="cpu")
    values.kv_slot_mapping = torch.tensor([0, 0], dtype=torch.int64)
    values.positions = torch.tensor([3, 3], dtype=torch.int64)
    with pytest.raises(ValueError, match="unique"):
        deepseek_v4_csa_indexer_mxfp4_cache_insert_reference(values)


@pytest.mark.parametrize(
    ("overlap", "expected_state_width"),
    [
        (False, 1024),
        (True, 2048),
    ],
)
def test_deepseek_v4_sparse_compress_cache_insert_inputs_generate_reference(
    overlap: bool,
    expected_state_width: int,
) -> None:
    values = DeepSeekV4SparseCompressCacheInsertInputs(
        DeepSeekV4SparseCompressCacheInsertInputConfig(
            num_tokens=6,
            batch_size=2,
            max_seq_len=8,
            num_state_cache_blocks=4,
            compressor_block_size=4,
            num_kv_cache_blocks=2,
            kv_cache_block_size=4,
            compress_ratio=4,
            overlap=overlap,
            dtype=torch.bfloat16,
            non_boundary_token_count=1,
            include_block_table_base_offsets=True,
            value_scale=0.25,
        )
    ).generate(metadata_seed=231, value_seed=241, device="cpu")

    expected = deepseek_v4_sparse_compress_cache_insert_reference(values)

    assert values.state_cache.shape == (4, 4, expected_state_width)
    assert values.block_table.shape == (2, 2)
    assert values.kv_cache_2d.shape == (2, 4 * (576 + 8))
    assert values.rms_norm_weight.shape == (512,)
    assert values.cos_sin_cache.shape == (8, 64)
    assert values.block_table_base_offsets is not None
    assert expected.shape == values.kv_cache_2d.shape

    writable_slots = []
    for row_idx in range(values.positions.numel()):
        slot = int(values.kv_slot_mapping[row_idx].item())
        writable = (
            int(values.compressor_slot_mapping[row_idx].item()) >= 0
            and slot >= 0
            and (int(values.positions[row_idx].item()) + 1) % values.compress_ratio == 0
        )
        if not writable:
            continue
        writable_slots.append(slot)
        page = slot // values.kv_cache_block_size
        pos = slot % values.kv_cache_block_size
        token_base = pos * 576
        scale_base = values.kv_cache_block_size * 576 + pos * 8
        assert not torch.equal(
            expected[page, token_base : token_base + 576],
            values.kv_cache_2d[page, token_base : token_base + 576],
        )
        assert not torch.equal(
            expected[page, scale_base : scale_base + 8],
            values.kv_cache_2d[page, scale_base : scale_base + 8],
        )
    assert writable_slots


def test_deepseek_v4_sparse_compress_cache_insert_skips_invalid_rows() -> None:
    values = DeepSeekV4SparseCompressCacheInsertInputs(
        DeepSeekV4SparseCompressCacheInsertInputConfig(
            num_tokens=3,
            batch_size=1,
            max_seq_len=4,
            num_state_cache_blocks=1,
            compressor_block_size=4,
            num_kv_cache_blocks=1,
            kv_cache_block_size=4,
            compress_ratio=4,
            overlap=True,
            dtype=torch.float32,
            non_boundary_token_count=1,
            negative_compressor_slot_count=1,
        )
    ).generate(seed=251, device="cpu")
    values.token_to_req_indices.zero_()
    values.positions = torch.tensor([3, 2, 3], dtype=torch.int64)
    values.compressor_slot_mapping = torch.tensor([0, 1, -1], dtype=torch.int64)
    values.kv_slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int64)
    values.block_table.zero_()
    values.kv_cache_2d.zero_()
    values.state_cache.zero_()
    values.state_cache[:, :, 512:1024] = 1.0
    values.rms_norm_weight.fill_(1.0)

    expected = deepseek_v4_sparse_compress_cache_insert_reference(values)

    assert not torch.equal(expected[0, :576], values.kv_cache_2d[0, :576])
    torch.testing.assert_close(
        expected[0, 576:1152],
        values.kv_cache_2d[0, 576:1152],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        expected[0, 1152:1728],
        values.kv_cache_2d[0, 1152:1728],
        rtol=0,
        atol=0,
    )


def test_deepseek_v4_sparse_compress_cache_insert_keeps_metadata_seed_independent() -> (
    None
):
    generator = DeepSeekV4SparseCompressCacheInsertInputs(
        DeepSeekV4SparseCompressCacheInsertInputConfig(
            num_tokens=5,
            batch_size=2,
            max_seq_len=8,
            num_state_cache_blocks=3,
            compressor_block_size=4,
            num_kv_cache_blocks=2,
            kv_cache_block_size=4,
            compress_ratio=4,
            overlap=False,
            dtype=torch.float32,
            non_boundary_token_count=1,
        )
    )

    first = generator.generate(metadata_seed=261, value_seed=271, device="cpu")
    same_metadata = generator.generate(metadata_seed=261, value_seed=272, device="cpu")
    same_values = generator.generate(metadata_seed=262, value_seed=271, device="cpu")

    torch.testing.assert_close(first.positions, same_metadata.positions)
    torch.testing.assert_close(
        first.compressor_slot_mapping,
        same_metadata.compressor_slot_mapping,
    )
    torch.testing.assert_close(first.kv_slot_mapping, same_metadata.kv_slot_mapping)
    torch.testing.assert_close(first.block_table, same_metadata.block_table)
    assert not torch.equal(first.state_cache, same_metadata.state_cache)
    torch.testing.assert_close(first.state_cache, same_values.state_cache)
    assert not torch.equal(first.positions, same_values.positions)


def test_deepseek_v4_sparse_compress_cache_insert_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="compressor slots"):
        DeepSeekV4SparseCompressCacheInsertInputs(
            DeepSeekV4SparseCompressCacheInsertInputConfig(
                num_tokens=5,
                batch_size=1,
                max_seq_len=4,
                num_state_cache_blocks=1,
                compressor_block_size=4,
                num_kv_cache_blocks=2,
                kv_cache_block_size=4,
                compress_ratio=4,
                overlap=False,
                dtype=torch.float32,
            )
        )
    with pytest.raises(ValueError, match="compression boundary"):
        DeepSeekV4SparseCompressCacheInsertInputs(
            DeepSeekV4SparseCompressCacheInsertInputConfig(
                num_tokens=1,
                batch_size=1,
                max_seq_len=3,
                num_state_cache_blocks=1,
                compressor_block_size=4,
                num_kv_cache_blocks=1,
                kv_cache_block_size=4,
                compress_ratio=4,
                overlap=True,
                dtype=torch.float32,
            )
        )

    values = DeepSeekV4SparseCompressCacheInsertInputs(
        DeepSeekV4SparseCompressCacheInsertInputConfig(
            num_tokens=2,
            batch_size=1,
            max_seq_len=4,
            num_state_cache_blocks=1,
            compressor_block_size=4,
            num_kv_cache_blocks=1,
            kv_cache_block_size=4,
            compress_ratio=4,
            overlap=True,
            dtype=torch.float32,
        )
    ).generate(seed=281, device="cpu")
    values.kv_slot_mapping = torch.tensor([0, 0], dtype=torch.int64)
    values.positions = torch.tensor([3, 3], dtype=torch.int64)
    with pytest.raises(ValueError, match="unique"):
        deepseek_v4_sparse_compress_cache_insert_reference(values)


def test_deepseek_v4_indexer_mxfp4_cache_write_inputs_generate_reference() -> None:
    values = DeepSeekV4IndexerMXFP4CacheWriteInputs(
        DeepSeekV4IndexerMXFP4CacheWriteInputConfig(
            num_rows=5,
            num_cache_blocks=2,
            block_size=4,
            dtype=torch.bfloat16,
            negative_slot_count=1,
            masked_row_count=1,
        )
    ).generate(metadata_seed=81, value_seed=91, device="cpu")

    expected = deepseek_v4_indexer_mxfp4_cache_write_reference(values)

    assert values.index_k.shape == (5, 128)
    assert values.cache_2d.shape == (2, 4 * 68)
    assert values.slot_mapping.shape == (5,)
    assert values.valid.shape == (5,)
    assert int((values.slot_mapping < 0).sum().item()) == 1
    assert int((~values.valid).sum().item()) == 1
    assert expected.shape == values.cache_2d.shape

    slots = values.slot_mapping.to(torch.int64)
    writable_slots = []
    for row_idx, slot in enumerate(slots.tolist()):
        if slot < 0 or not bool(values.valid[row_idx].item()):
            continue
        writable_slots.append(slot)
        page = slot // values.block_size
        pos = slot % values.block_size
        value_base = pos * 64
        scale_base = values.block_size * 64 + pos * 4
        assert not torch.equal(
            expected[page, value_base : value_base + 64],
            values.cache_2d[page, value_base : value_base + 64],
        )
        assert not torch.equal(
            expected[page, scale_base : scale_base + 4],
            values.cache_2d[page, scale_base : scale_base + 4],
        )

    assert writable_slots
    writable_slot_set = set(writable_slots)
    for slot in range(values.cache_2d.shape[0] * values.block_size):
        if slot in writable_slot_set:
            continue
        page = slot // values.block_size
        pos = slot % values.block_size
        value_base = pos * 64
        scale_base = values.block_size * 64 + pos * 4
        torch.testing.assert_close(
            expected[page, value_base : value_base + 64],
            values.cache_2d[page, value_base : value_base + 64],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            expected[page, scale_base : scale_base + 4],
            values.cache_2d[page, scale_base : scale_base + 4],
            rtol=0,
            atol=0,
        )


def test_deepseek_v4_indexer_mxfp4_cache_write_reference_known_zero_row() -> None:
    values = DeepSeekV4IndexerMXFP4CacheWriteInputs(
        DeepSeekV4IndexerMXFP4CacheWriteInputConfig(
            num_rows=1,
            num_cache_blocks=1,
            block_size=1,
            dtype=torch.float32,
        )
    ).generate(seed=101, device="cpu")
    values.index_k = torch.zeros((1, 128), dtype=torch.float32)
    values.cache_2d = torch.full((1, 68), 255, dtype=torch.uint8)
    values.slot_mapping = torch.tensor([0], dtype=torch.int64)
    values.valid = torch.tensor([True], dtype=torch.bool)

    expected = deepseek_v4_indexer_mxfp4_cache_write_reference(values)

    torch.testing.assert_close(expected[0, :64], torch.zeros(64, dtype=torch.uint8))
    torch.testing.assert_close(
        expected[0, 64:68],
        torch.full((4,), 112, dtype=torch.uint8),
    )


def test_deepseek_v4_indexer_mxfp4_cache_write_keeps_metadata_seed_independent() -> (
    None
):
    generator = DeepSeekV4IndexerMXFP4CacheWriteInputs(
        DeepSeekV4IndexerMXFP4CacheWriteInputConfig(
            num_rows=4,
            num_cache_blocks=1,
            block_size=4,
            dtype=torch.float32,
            negative_slot_count=1,
            masked_row_count=1,
        )
    )

    first = generator.generate(metadata_seed=111, value_seed=121, device="cpu")
    same_metadata = generator.generate(metadata_seed=111, value_seed=122, device="cpu")
    same_values = generator.generate(metadata_seed=112, value_seed=121, device="cpu")

    torch.testing.assert_close(first.slot_mapping, same_metadata.slot_mapping)
    torch.testing.assert_close(first.valid, same_metadata.valid)
    assert not torch.equal(first.index_k, same_metadata.index_k)
    torch.testing.assert_close(first.index_k, same_values.index_k)
    assert not torch.equal(first.slot_mapping, same_values.slot_mapping)


def test_deepseek_v4_indexer_mxfp4_cache_write_rejects_invalid_configs_and_values() -> (
    None
):
    with pytest.raises(ValueError, match="non-negative slots"):
        DeepSeekV4IndexerMXFP4CacheWriteInputs(
            DeepSeekV4IndexerMXFP4CacheWriteInputConfig(
                num_rows=5,
                num_cache_blocks=1,
                block_size=4,
                dtype=torch.float32,
            )
        )

    values = DeepSeekV4IndexerMXFP4CacheWriteInputs(
        DeepSeekV4IndexerMXFP4CacheWriteInputConfig(
            num_rows=2,
            num_cache_blocks=1,
            block_size=4,
            dtype=torch.float32,
        )
    ).generate(seed=131, device="cpu")
    values.slot_mapping = torch.tensor([0, 0], dtype=torch.int64)
    values.valid = torch.tensor([True, True], dtype=torch.bool)
    with pytest.raises(ValueError, match="unique"):
        deepseek_v4_indexer_mxfp4_cache_write_reference(values)


def test_deepseek_v4_indexer_mxfp4_cache_gather_inputs_generate_reference() -> None:
    values = DeepSeekV4IndexerMXFP4CacheGatherInputs(
        DeepSeekV4IndexerMXFP4CacheGatherInputConfig(
            num_rows=6,
            num_cache_blocks=2,
            block_size=4,
            negative_slot_count=2,
        )
    ).generate(metadata_seed=141, value_seed=151, device="cpu")

    expected_values, expected_scales = deepseek_v4_indexer_mxfp4_cache_gather_reference(
        values
    )

    assert values.cache_2d.shape == (2, 4 * 68)
    assert values.slot_mapping.shape == (6,)
    assert values.values_out.shape == (6, 64)
    assert values.scales_out.shape == (6, 4)
    assert int((values.slot_mapping < 0).sum().item()) == 2
    assert expected_values.shape == values.values_out.shape
    assert expected_scales.shape == values.scales_out.shape

    for row_idx, slot in enumerate(values.slot_mapping.to(torch.int64).tolist()):
        if slot < 0:
            torch.testing.assert_close(
                expected_values[row_idx, :64],
                torch.zeros((64,), dtype=torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                expected_scales[row_idx, :4],
                torch.zeros((4,), dtype=torch.uint8),
                rtol=0,
                atol=0,
            )
            continue

        page = slot // values.block_size
        pos = slot % values.block_size
        value_base = pos * 64
        scale_base = values.block_size * 64 + pos * 4
        torch.testing.assert_close(
            expected_values[row_idx, :64],
            values.cache_2d[page, value_base : value_base + 64],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            expected_scales[row_idx, :4],
            values.cache_2d[page, scale_base : scale_base + 4],
            rtol=0,
            atol=0,
        )


def test_deepseek_v4_indexer_mxfp4_cache_gather_keeps_metadata_seed_independent() -> (
    None
):
    generator = DeepSeekV4IndexerMXFP4CacheGatherInputs(
        DeepSeekV4IndexerMXFP4CacheGatherInputConfig(
            num_rows=4,
            num_cache_blocks=2,
            block_size=2,
            negative_slot_count=1,
        )
    )

    first = generator.generate(metadata_seed=161, value_seed=171, device="cpu")
    same_metadata = generator.generate(metadata_seed=161, value_seed=172, device="cpu")
    same_values = generator.generate(metadata_seed=162, value_seed=171, device="cpu")

    torch.testing.assert_close(first.slot_mapping, same_metadata.slot_mapping)
    assert not torch.equal(first.cache_2d, same_metadata.cache_2d)
    assert not torch.equal(first.values_out, same_metadata.values_out)
    torch.testing.assert_close(first.cache_2d, same_values.cache_2d)
    torch.testing.assert_close(first.values_out, same_values.values_out)
    torch.testing.assert_close(first.scales_out, same_values.scales_out)


def test_deepseek_v4_indexer_mxfp4_cache_gather_rejects_invalid_configs_and_values() -> (
    None
):
    with pytest.raises(ValueError, match="negative_slot_count"):
        DeepSeekV4IndexerMXFP4CacheGatherInputs(
            DeepSeekV4IndexerMXFP4CacheGatherInputConfig(
                num_rows=1,
                num_cache_blocks=1,
                block_size=2,
                negative_slot_count=2,
            )
        )

    values = DeepSeekV4IndexerMXFP4CacheGatherInputs(
        DeepSeekV4IndexerMXFP4CacheGatherInputConfig(
            num_rows=1,
            num_cache_blocks=1,
            block_size=2,
        )
    ).generate(seed=181, device="cpu")
    values.slot_mapping = torch.tensor([2], dtype=torch.int64)
    with pytest.raises(ValueError, match="slot_mapping entries"):
        deepseek_v4_indexer_mxfp4_cache_gather_reference(values)


def test_deepseek_v4_k_cache_gather_inputs_generate_reference() -> None:
    values = DeepSeekV4KCacheGatherInputs(
        DeepSeekV4KCacheGatherInputConfig(
            batch_size=3,
            max_seq_len=9,
            block_size=4,
            max_gather_len=5,
            offset=2,
            include_gather_lens=True,
            include_block_table_base_offsets=True,
        )
    ).generate(metadata_seed=191, value_seed=201, device="cpu")

    expected = deepseek_v4_dequantize_and_gather_k_cache_reference(values)

    assert values.out.shape == (3, 7, 512)
    assert values.cache_2d.shape == (9, 4 * (576 + 8))
    assert values.seq_lens.shape == (3,)
    assert values.gather_lens is not None
    assert values.gather_lens.shape == (3,)
    assert values.block_table.shape == (3, 3)
    assert values.block_table_base_offsets is not None
    torch.testing.assert_close(expected[:, :2], values.out[:, :2], rtol=0, atol=0)

    for batch_idx in range(values.seq_lens.numel()):
        seq_len = int(values.seq_lens[batch_idx].item())
        gather_len = int(values.gather_lens[batch_idx].item())
        start_pos = seq_len - gather_len
        base_offset = int(values.block_table_base_offsets[batch_idx].item())
        for gather_idx in range(gather_len):
            pos = start_pos + gather_idx
            table_idx = pos // values.block_size - base_offset
            physical = int(values.block_table[batch_idx, table_idx].item())
            pos_in_block = pos % values.block_size
            rope_start = pos_in_block * 576 + 448
            rope_end = rope_start + 128
            torch.testing.assert_close(
                expected[batch_idx, values.offset + gather_idx, 448:],
                values.cache_2d[physical, rope_start:rope_end].view(torch.bfloat16),
                rtol=0,
                atol=0,
            )


def test_deepseek_v4_k_cache_gather_keeps_metadata_seed_independent() -> None:
    generator = DeepSeekV4KCacheGatherInputs(
        DeepSeekV4KCacheGatherInputConfig(
            batch_size=2,
            max_seq_len=8,
            block_size=4,
            max_gather_len=4,
        )
    )

    first = generator.generate(metadata_seed=211, value_seed=221, device="cpu")
    same_metadata = generator.generate(metadata_seed=211, value_seed=222, device="cpu")
    same_values = generator.generate(metadata_seed=212, value_seed=221, device="cpu")

    torch.testing.assert_close(first.seq_lens, same_metadata.seq_lens)
    assert first.gather_lens is not None and same_metadata.gather_lens is not None
    torch.testing.assert_close(first.gather_lens, same_metadata.gather_lens)
    torch.testing.assert_close(first.block_table, same_metadata.block_table)
    assert not torch.equal(first.cache_2d, same_metadata.cache_2d)
    torch.testing.assert_close(first.cache_2d, same_values.cache_2d)
    torch.testing.assert_close(first.out, same_values.out)


def test_deepseek_v4_k_cache_gather_supports_full_sequence_gather() -> None:
    values = DeepSeekV4KCacheGatherInputs(
        DeepSeekV4KCacheGatherInputConfig(
            batch_size=2,
            max_seq_len=6,
            block_size=3,
            include_gather_lens=False,
        )
    ).generate(seed=231, device="cpu")

    expected = deepseek_v4_dequantize_and_gather_k_cache_reference(values)

    assert values.gather_lens is None
    assert expected.shape == values.out.shape
    assert torch.isfinite(expected.float()).all()


def test_deepseek_v4_k_cache_gather_rejects_invalid_configs_and_values() -> None:
    with pytest.raises(ValueError, match="max_gather_len"):
        DeepSeekV4KCacheGatherInputs(
            DeepSeekV4KCacheGatherInputConfig(
                batch_size=1,
                max_seq_len=4,
                block_size=2,
                max_gather_len=5,
            )
        )
    with pytest.raises(ValueError, match="num_cache_blocks"):
        DeepSeekV4KCacheGatherInputs(
            DeepSeekV4KCacheGatherInputConfig(
                batch_size=2,
                max_seq_len=4,
                block_size=2,
                num_cache_blocks=3,
            )
        )

    values = DeepSeekV4KCacheGatherInputs(
        DeepSeekV4KCacheGatherInputConfig(
            batch_size=1,
            max_seq_len=4,
            block_size=2,
            max_gather_len=2,
        )
    ).generate(seed=241, device="cpu")
    assert values.gather_lens is not None
    values.gather_lens = values.seq_lens + 1
    with pytest.raises(ValueError, match="gather_lens entries"):
        deepseek_v4_dequantize_and_gather_k_cache_reference(values)


def test_deepseek_v4_paged_index_inputs_generate_values_and_refs() -> None:
    values = DeepSeekV4PagedIndexInputs(
        DeepSeekV4PagedIndexInputConfig(
            batch_size=3,
            total_cached_tokens=18,
            total_new_q_tokens=9,
            block_size=4,
            compress_ratio=3,
            window_size=5,
            topk=4,
            indexing="identity",
            include_valid_token_mask=True,
            invalid_token_probability=0.85,
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=3,
                total_cached_tokens=18,
                total_new_q_tokens=9,
                cache_layout="paged",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=11, device="cpu")

    assert values.positions.shape == (9,)
    assert values.token_to_req_indices.shape == (9,)
    assert values.local_topk_indices.shape == (9, 4)
    assert values.seq_lens.shape == (3,)
    assert values.block_table.shape[0] == 3
    assert values.block_table.shape[1] >= 3
    assert values.is_valid_token is not None
    assert values.is_valid_token.shape == (9,)

    global_topk, global_lens = (
        deepseek_v4_compute_global_topk_indices_and_lens_reference(values)
    )
    swa_indices, swa_lens = deepseek_v4_decode_swa_indices_and_lens_reference(values)
    slot_mapping = deepseek_v4_compressed_slot_mapping_reference(values)
    context_lens, out_block_tables = deepseek_v4_indexer_decode_metadata_reference(
        values
    )

    assert global_topk.shape == values.local_topk_indices.shape
    assert global_lens.shape == (9,)
    assert swa_indices.shape == (9, values.window_size)
    assert swa_lens.shape == (9,)
    assert slot_mapping.shape == (9,)
    assert context_lens.shape == (9,)
    assert out_block_tables.shape == (9, values.max_blocks)
    assert torch.all(global_lens <= values.topk)
    assert torch.all(swa_lens <= values.window_size)
    invalid = ~values.is_valid_token
    assert torch.all(global_lens[invalid] == 0)
    assert torch.all(swa_lens[invalid] == 0)

    valid_compressed = [
        idx
        for idx, pos in enumerate(values.positions.tolist())
        if (pos + 1) % values.compress_ratio == 0
    ]
    assert valid_compressed
    assert torch.all(slot_mapping[valid_compressed] >= 0)
    assert torch.all(context_lens >= 0)


def test_deepseek_v4_paged_index_inputs_support_base_offsets() -> None:
    values = DeepSeekV4PagedIndexInputs(
        DeepSeekV4PagedIndexInputConfig(
            batch_size=2,
            total_cached_tokens=16,
            total_new_q_tokens=6,
            block_size=4,
            compress_ratio=2,
            window_size=4,
            topk=3,
            include_block_table_base_offsets=True,
            indexing="identity",
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=2,
                total_cached_tokens=16,
                total_new_q_tokens=6,
                cache_layout="paged",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=12, device="cpu")

    assert values.block_table_base_offsets is not None
    assert values.block_table_base_offsets.shape == (2,)

    swa_indices, swa_lens = deepseek_v4_decode_swa_indices_and_lens_reference(values)
    context_lens, out_block_tables = deepseek_v4_indexer_decode_metadata_reference(
        values
    )

    assert swa_indices.shape == (6, 4)
    assert torch.all(swa_lens <= 4)
    assert context_lens.shape == (6,)
    assert out_block_tables.shape == (6, values.max_blocks)


def test_deepseek_v4_paged_index_inputs_keep_metadata_seeded() -> None:
    config = DeepSeekV4PagedIndexInputConfig(
        batch_size=2,
        total_cached_tokens=13,
        total_new_q_tokens=7,
        block_size=4,
        compress_ratio=2,
        window_size=4,
        topk=3,
        indexing="random",
    )

    first = DeepSeekV4PagedIndexInputs(config).generate(seed=13, device="cpu")
    second = DeepSeekV4PagedIndexInputs(config).generate(seed=13, device="cpu")
    third = DeepSeekV4PagedIndexInputs(config).generate(seed=14, device="cpu")

    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.token_to_req_indices, second.token_to_req_indices)
    assert torch.equal(first.block_table, second.block_table)
    assert torch.equal(first.local_topk_indices, second.local_topk_indices)
    assert not torch.equal(first.block_table, third.block_table)


def test_deepseek_v4_paged_index_inputs_reject_invalid_configs() -> None:
    with pytest.raises(ValueError, match="compress_ratio"):
        DeepSeekV4PagedIndexInputs(
            DeepSeekV4PagedIndexInputConfig(
                batch_size=2,
                total_cached_tokens=8,
                total_new_q_tokens=4,
                block_size=4,
                compress_ratio=1,
                window_size=4,
                topk=2,
            )
        )

    with pytest.raises(ValueError, match="metadata_input.total_new_kv_tokens"):
        DeepSeekV4PagedIndexInputs(
            DeepSeekV4PagedIndexInputConfig(
                batch_size=2,
                total_cached_tokens=8,
                total_new_q_tokens=4,
                block_size=4,
                compress_ratio=2,
                window_size=4,
                topk=2,
                metadata_input=MHARequestMetadataInputConfig(
                    batch_size=2,
                    total_cached_tokens=8,
                    total_new_q_tokens=4,
                    total_new_kv_tokens=3,
                    cache_layout="paged",
                    tie_new_kv_to_query=False,
                ),
            )
        )

    with pytest.raises(ValueError, match="page_table_input.batch_size"):
        DeepSeekV4PagedIndexInputs(
            DeepSeekV4PagedIndexInputConfig(
                batch_size=2,
                total_cached_tokens=8,
                total_new_q_tokens=4,
                block_size=4,
                compress_ratio=2,
                window_size=4,
                topk=2,
                page_table_input=PageTableInputConfig(
                    batch_size=1,
                    max_pages_per_request=4,
                ),
            )
        )


def test_deepseek_v4_sparse_prefill_indices_generate_values_and_refs() -> None:
    values = DeepSeekV4SparsePrefillIndexInputs(
        DeepSeekV4SparsePrefillIndexInputConfig(
            batch_size=3,
            total_cached_tokens=17,
            total_new_q_tokens=9,
            topk=4,
            window_size=5,
            compress_ratio=3,
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=3,
                total_cached_tokens=17,
                total_new_q_tokens=9,
                cache_layout="dense",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=17, device="cpu")

    assert values.topk_indices.shape == (9, 4)
    assert values.positions.shape == (9,)
    assert values.token_to_req_indices.shape == (9,)
    assert values.seq_lens.shape == (3,)
    assert values.gather_lens.shape == (3,)
    assert values.compressed_lens.shape == (3,)
    assert values.topk_indices.dtype == torch.int32
    assert values.positions.dtype == torch.int32
    assert values.workspace_width >= (
        values.compressed_base + int(values.gather_lens.max().item())
    )

    local = deepseek_v4_build_dense_prefill_local_compressed_indices_reference(values)
    topk_indices, topk_lens = deepseek_v4_combine_topk_swa_indices_reference(values)
    dense_indices, dense_lens = deepseek_v4_combine_dense_swa_indices_reference(values)

    assert local.shape == (9, values.compressed_base)
    assert topk_indices.shape == (9, 128)
    assert dense_indices.shape == (9, 128)
    assert topk_lens.shape == (9,)
    assert dense_lens.shape == (9,)
    assert torch.all(topk_lens > 0)
    assert torch.all(dense_lens > 0)
    for token_idx, pos in enumerate(values.positions.tolist()):
        req = int(values.token_to_req_indices[token_idx].item())
        expected_topk_len = min((pos + 1) // values.compress_ratio, values.topk)
        expected_swa_len = min(pos + 1, values.window_size)
        expected_dense_len = min(
            (pos + 1) // values.compress_ratio,
            int(values.compressed_lens[req].item()),
        )
        assert int(topk_lens[token_idx].item()) == (
            expected_topk_len + expected_swa_len
        )
        assert int(dense_lens[token_idx].item()) == (
            expected_dense_len + expected_swa_len
        )
        topk_len = int(topk_lens[token_idx].item())
        dense_len = int(dense_lens[token_idx].item())
        assert torch.all(topk_indices[token_idx, topk_len:] == -1)
        assert torch.all(dense_indices[token_idx, dense_len:] == -1)


def test_deepseek_v4_sparse_prefill_index_inputs_keep_seeded_metadata_stable() -> None:
    config = DeepSeekV4SparsePrefillIndexInputConfig(
        batch_size=2,
        total_cached_tokens=20,
        total_new_q_tokens=8,
        topk=3,
        window_size=4,
        compress_ratio=2,
        topk_indexing="random",
        metadata_input=MHARequestMetadataInputConfig(
            batch_size=2,
            total_cached_tokens=20,
            total_new_q_tokens=8,
            cache_layout="dense",
            cached_length_mode="regular",
            new_q_length_mode="fixed_per_request",
        ),
    )

    first = DeepSeekV4SparsePrefillIndexInputs(config).generate(seed=23, device="cpu")
    second = DeepSeekV4SparsePrefillIndexInputs(config).generate(seed=23, device="cpu")
    third = DeepSeekV4SparsePrefillIndexInputs(config).generate(seed=24, device="cpu")

    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.token_to_req_indices, second.token_to_req_indices)
    assert torch.equal(first.topk_indices, second.topk_indices)
    assert not torch.equal(first.topk_indices, third.topk_indices)


def test_deepseek_v4_sparse_prefill_index_inputs_support_identity_topk() -> None:
    values = DeepSeekV4SparsePrefillIndexInputs(
        DeepSeekV4SparsePrefillIndexInputConfig(
            batch_size=1,
            total_cached_tokens=12,
            total_new_q_tokens=4,
            topk=4,
            window_size=3,
            compress_ratio=2,
            topk_indexing="identity",
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=1,
                total_cached_tokens=12,
                total_new_q_tokens=4,
                cache_layout="dense",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=31, device="cpu")

    for token_idx, pos in enumerate(values.positions.tolist()):
        valid = min((pos + 1) // values.compress_ratio, values.topk)
        assert torch.equal(
            values.topk_indices[token_idx, :valid],
            torch.arange(valid, dtype=torch.int32),
        )


def test_deepseek_v4_sparse_prefill_index_inputs_reject_invalid_configs() -> None:
    with pytest.raises(ValueError, match="compress_ratio"):
        DeepSeekV4SparsePrefillIndexInputs(
            DeepSeekV4SparsePrefillIndexInputConfig(
                batch_size=2,
                total_cached_tokens=0,
                total_new_q_tokens=4,
                topk=2,
                window_size=4,
                compress_ratio=1,
            )
        )

    with pytest.raises(ValueError, match="total_new_kv_tokens"):
        DeepSeekV4SparsePrefillIndexInputs(
            DeepSeekV4SparsePrefillIndexInputConfig(
                batch_size=2,
                total_cached_tokens=0,
                total_new_q_tokens=4,
                topk=2,
                window_size=4,
                compress_ratio=2,
                metadata_input=MHARequestMetadataInputConfig(
                    batch_size=2,
                    total_cached_tokens=0,
                    total_new_q_tokens=4,
                    total_new_kv_tokens=3,
                    tie_new_kv_to_query=False,
                    allow_untied_non_cached_kv=True,
                ),
            )
        )

    with pytest.raises(ValueError, match="workspace_width"):
        DeepSeekV4SparsePrefillIndexInputs(
            DeepSeekV4SparsePrefillIndexInputConfig(
                batch_size=1,
                total_cached_tokens=8,
                total_new_q_tokens=4,
                topk=2,
                window_size=4,
                compress_ratio=2,
                compressed_base=6,
                workspace_width=7,
            )
        ).generate(seed=1, device="cpu")


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
