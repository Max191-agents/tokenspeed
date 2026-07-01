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
from tokenspeed_kernel import (
    attn_merge_state,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
    mla_decode_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.ops.attention.tokenspeed_mla import mla_kv_pack_quantize_fp8
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)
from tokenspeed_kernel.ops.attention.triton.qkv_rotary import packed_qkv_complex_rotary
from tokenspeed_kernel.ops.attention.triton.dsa_sparse_layout import (
    full_context_topk_to_global_slots,
    local_topk_to_global_slots,
    pack_sparse_decode_kv,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_build_dense_prefill_local_compressed_indices,
    deepseek_v4_compressed_slot_mapping,
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_indexer_decode_metadata_compute,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.numerics.attention_kernel_kwargs import (
    mla_decode_with_kvcache_kwargs,
    mla_prefill_kwargs,
    mha_decode_with_kvcache_kwargs,
    mha_extend_with_kvcache_kwargs,
    mha_prefill_kwargs,
)
from tokenspeed_numerics_input_generators import (
    AttentionMergeStateInputConfig,
    AttentionMergeStateInputs,
    DSASparseDecodeKVPackInputConfig,
    DSASparseDecodeKVPackInputs,
    DSATopKSlotInputConfig,
    DSATopKSlotInputs,
    GDNQKVSplitInputConfig,
    GDNQKVSplitInputs,
    DeepSeekV4PagedIndexInputConfig,
    DeepSeekV4PagedIndexInputs,
    DeepSeekV4SparsePrefillIndexInputConfig,
    DeepSeekV4SparsePrefillIndexInputs,
    MHAInputConfig,
    MHAInputs,
    MLAInputConfig,
    MLAInputs,
    MLAKVPackQuantizeFP8InputConfig,
    MLAKVPackQuantizeFP8Inputs,
    MHARequestMetadataInputConfig,
    PackedQKVComplexRotaryInputConfig,
    PackedQKVComplexRotaryInputs,
    attention_merge_state_reference,
    dsa_sparse_decode_kv_pack_reference,
    dsa_full_context_topk_to_global_slots_reference,
    dsa_local_topk_to_global_slots_reference,
    deepseek_v4_compressed_slot_mapping_reference,
    deepseek_v4_compute_global_topk_indices_and_lens_reference,
    deepseek_v4_decode_swa_indices_and_lens_reference,
    deepseek_v4_build_dense_prefill_local_compressed_indices_reference,
    deepseek_v4_combine_dense_swa_indices_reference,
    deepseek_v4_combine_topk_swa_indices_reference,
    deepseek_v4_indexer_decode_metadata_reference,
    gdn_qkv_split_reference,
    mla_kv_pack_quantize_fp8_reference,
    packed_qkv_complex_rotary_reference,
)


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


def test_attention_merge_state_generator_runs_triton_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    solution = "triton"
    require("attention", "attn_merge_state", solution, dtype, "out_a")

    values = AttentionMergeStateInputs(
        AttentionMergeStateInputConfig(
            total_q=31,
            num_heads=8,
            head_dim=64,
            dtype=dtype,
        )
    ).generate(seed=301, device=device)
    expected_out, expected_lse = attention_merge_state_reference(values)

    out, lse = attn_merge_state(
        values.out_a,
        values.lse_a,
        values.out_b,
        values.lse_b,
        lse_scale_log2=values.lse_scale_log2,
        solution=solution,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), expected_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_mha_prefill_generator_runs_attention_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("attention", "mha_prefill", solution, dtype, "q")

    inputs = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=0,
            total_new_q_tokens=33,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            q_dtype=dtype,
            cache_layout="none",
            include_sinks=True,
            metadata_kwargs={"new_q_length_mode": "ragged"},
        )
    ).generate(metadata_seed=101, value_seed=201, device=device)

    out = mha_prefill(**mha_prefill_kwargs(inputs), solution=solution)

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()


def test_mha_paged_extend_generator_runs_triton_attention_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    solution = "triton"
    require("attention", "mha_extend_with_kvcache", solution, dtype, "q")

    inputs = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=47,
            total_new_q_tokens=11,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=64,
            include_sinks=True,
            metadata_kwargs={
                "cached_length_mode": "ragged",
                "new_q_length_mode": "ragged",
            },
        )
    ).generate(metadata_seed=102, value_seed=202, device=device)

    out = mha_extend_with_kvcache(
        **mha_extend_with_kvcache_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()


def test_mla_prefill_generator_runs_attention_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    solution = "triton"
    require("attention", "mla_prefill", solution, dtype, "q")

    inputs = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=0,
            total_new_q_tokens=33,
            num_q_heads=8,
            num_kv_heads=8,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=128,
            v_head_dim=128,
            q_dtype=dtype,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "ragged"},
        )
    ).generate(metadata_seed=104, value_seed=204, device=device)

    out = mla_prefill(
        **mla_prefill_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert inputs.v is not None
    assert out.shape == (inputs.q.shape[0], inputs.q.shape[1], inputs.v.shape[-1])
    assert not torch.isnan(out.float()).any()


def test_mla_paged_decode_generator_runs_attention_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    solution = "triton"
    require("attention", "mla_decode_with_kvcache", solution, dtype, "q")

    inputs = MLAInputs(
        _mla_config(
            batch_size=2,
            total_cached_tokens=12,
            total_new_q_tokens=2,
            num_q_heads=8,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=128,
            v_head_dim=128,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=4,
            indexing="identity",
            metadata_kwargs={
                "max_seqlen_k": 7,
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=105, value_seed=205, device=device)

    out = mla_decode_with_kvcache(
        **mla_decode_with_kvcache_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == (inputs.q.shape[0], inputs.q.shape[1], inputs.q.shape[2], 128)
    assert not torch.isnan(out.float()).any()


def test_mla_kv_pack_quantize_fp8_generator_runs_tokenspeed_mla_kernel(
    device: str,
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("tokenspeed_mla K/V pack+quantize is NVIDIA-only")

    values = MLAKVPackQuantizeFP8Inputs(
        MLAKVPackQuantizeFP8InputConfig(
            num_tokens=32,
            num_kv_heads=4,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=24,
            input_dtype=torch.bfloat16,
            k_scale_inv=0.5,
            v_scale_inv=1.7,
            fp8_dtype=torch.float8_e4m3fn,
        )
    ).generate(seed=206, device=device)
    expected_k, expected_v = mla_kv_pack_quantize_fp8_reference(values)

    actual_k, actual_v = mla_kv_pack_quantize_fp8(
        values.k_nope,
        values.k_pe,
        values.v,
        k_scale_inv=values.k_scale_inv,
        v_scale_inv=values.v_scale_inv,
        fp8_dtype=values.fp8_dtype,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(actual_v.view(torch.uint8), expected_v.view(torch.uint8))


@pytest.mark.parametrize("fuse_l2norm", [False, True], ids=["split", "split-l2"])
def test_gdn_qkv_split_generator_runs_tokenspeed_triton(
    device: str,
    fuse_l2norm: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for GDN QKV split Triton test")

    values = GDNQKVSplitInputs(
        GDNQKVSplitInputConfig(
            num_tokens=13,
            num_q_heads=4,
            num_k_heads=2,
            num_v_heads=2,
            head_q=16,
            head_k=16,
            head_v=16,
            dtype=torch.bfloat16,
            fuse_l2norm=fuse_l2norm,
        )
    ).generate(seed=208, device=device)
    expected_q, expected_k, expected_v = gdn_qkv_split_reference(values)

    actual_q, actual_k, actual_v = fused_qkv_split_gdn_prefill(
        values.mixed_qkv,
        values.num_q_heads,
        values.num_k_heads,
        values.num_v_heads,
        values.head_q,
        values.head_k,
        values.head_v,
        fuse_l2norm=values.fuse_l2norm,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_q.float(), expected_q.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        actual_k.float(), expected_k.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(actual_v.float(), expected_v.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("copy_v", [False, True], ids=["view-v", "copy-v"])
def test_packed_qkv_complex_rotary_generator_runs_tokenspeed_triton(
    device: str,
    copy_v: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for packed QKV rotary Triton test")

    values = PackedQKVComplexRotaryInputs(
        PackedQKVComplexRotaryInputConfig(
            num_tokens=17,
            num_heads=4,
            head_dim=16,
            dtype=torch.bfloat16,
            copy_v=copy_v,
        )
    ).generate(seed=211, device=device)
    expected_q, expected_k, expected_v = packed_qkv_complex_rotary_reference(values)
    q_size = values.num_heads * values.head_dim
    kv_size = q_size

    actual_q, actual_k, actual_v = packed_qkv_complex_rotary(
        values.qkv,
        q_size,
        kv_size,
        values.num_heads,
        values.head_dim,
        values.freqs_cis,
        copy_v=values.copy_v,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_q.float(), expected_q.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        actual_k.float(), expected_k.float(), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(actual_v.float(), expected_v.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("include_head_axis", [False, True], ids=["rank2", "rank3"])
def test_dsa_sparse_decode_kv_pack_generator_runs_tokenspeed_triton(
    device: str,
    include_head_axis: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for DSA sparse decode pack Triton test")

    values = DSASparseDecodeKVPackInputs(
        DSASparseDecodeKVPackInputConfig(
            num_tokens=7,
            num_slots=11,
            nope_dim=128,
            rope_dim=64,
            include_head_axis=include_head_axis,
        )
    ).generate(seed=213, metadata_seed=214, device=device)
    expected = dsa_sparse_decode_kv_pack_reference(values)
    actual = values.out.clone()

    pack_sparse_decode_kv(
        out=actual,
        loc=values.loc,
        cache_k_nope=values.cache_k_nope,
        cache_k_rope=values.cache_k_rope,
    )
    torch.cuda.synchronize()

    loc = values.loc.to(torch.int64)
    nope_dim = values.cache_k_nope.shape[-1]
    num_nope_blocks = nope_dim // 128
    scale_offset = nope_dim
    rope_offset = scale_offset + num_nope_blocks * 4

    written_mask = torch.zeros(actual.shape[0], dtype=torch.bool, device=actual.device)
    written_mask[loc] = True
    assert torch.equal(actual[~written_mask], expected[~written_mask])
    assert torch.equal(actual[loc, :scale_offset], expected[loc, :scale_offset])
    assert torch.equal(actual[loc, rope_offset:], expected[loc, rope_offset:])

    actual_scales = (
        actual[loc, scale_offset:rope_offset].contiguous().view(torch.float32)
    )
    expected_scales = (
        expected[loc, scale_offset:rope_offset].contiguous().view(torch.float32)
    )
    torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-6, atol=1e-9)


def test_dsa_topk_slot_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for DSA top-k slot Triton test")

    values = DSATopKSlotInputs(
        DSATopKSlotInputConfig(
            num_tokens=7,
            topk=6,
            block_size=8,
            max_pages_per_token=4,
            max_seq_len=24,
        )
    ).generate(seed=212, device=device)

    expected_local_slots, expected_local_lens = (
        dsa_local_topk_to_global_slots_reference(values)
    )
    actual_local_slots, actual_local_lens = local_topk_to_global_slots(
        local_topk_offsets=values.local_topk_offsets,
        block_table=values.block_table,
        block_size=values.block_size,
        seq_lens=values.seq_lens,
    )

    expected_full_slots, expected_full_lens = (
        dsa_full_context_topk_to_global_slots_reference(values)
    )
    actual_full_slots, actual_full_lens = full_context_topk_to_global_slots(
        seq_lens=values.seq_lens,
        block_table=values.block_table,
        block_size=values.block_size,
        topk=values.topk,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_local_slots, expected_local_slots)
    assert torch.equal(actual_local_lens, expected_local_lens)
    assert torch.equal(actual_full_slots, expected_full_slots)
    assert torch.equal(actual_full_lens, expected_full_lens)


def test_deepseek_v4_global_topk_generator_runs_tokenspeed_cpu() -> None:
    values = DeepSeekV4PagedIndexInputs(
        DeepSeekV4PagedIndexInputConfig(
            batch_size=3,
            total_cached_tokens=18,
            total_new_q_tokens=9,
            block_size=4,
            compress_ratio=3,
            window_size=8,
            topk=4,
            include_valid_token_mask=True,
            invalid_token_probability=0.5,
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=3,
                total_cached_tokens=18,
                total_new_q_tokens=9,
                cache_layout="paged",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=209, device="cpu")
    expected_indices, expected_lens = (
        deepseek_v4_compute_global_topk_indices_and_lens_reference(values)
    )

    actual_indices, actual_lens = deepseek_v4_compute_global_topk_indices_and_lens(
        topk_indices=values.local_topk_indices,
        token_to_req_indices=values.token_to_req_indices,
        block_table=values.block_table,
        block_size=values.block_size,
        is_valid_token=values.is_valid_token,
    )

    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_lens, expected_lens)


def test_deepseek_v4_paged_index_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton paged-index tests")

    values = DeepSeekV4PagedIndexInputs(
        DeepSeekV4PagedIndexInputConfig(
            batch_size=3,
            total_cached_tokens=18,
            total_new_q_tokens=9,
            block_size=4,
            compress_ratio=3,
            window_size=8,
            topk=4,
            indexing="identity",
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=3,
                total_cached_tokens=18,
                total_new_q_tokens=9,
                cache_layout="paged",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=210, device=device)

    expected_topk_indices, expected_topk_lens = (
        deepseek_v4_compute_global_topk_indices_and_lens_reference(values)
    )
    actual_topk_indices, actual_topk_lens = (
        deepseek_v4_compute_global_topk_indices_and_lens(
            topk_indices=values.local_topk_indices,
            token_to_req_indices=values.token_to_req_indices,
            block_table=values.block_table,
            block_size=values.block_size,
            is_valid_token=values.is_valid_token,
        )
    )

    expected_swa_indices, expected_swa_lens = (
        deepseek_v4_decode_swa_indices_and_lens_reference(values)
    )
    actual_swa_indices, actual_swa_lens = deepseek_v4_decode_swa_indices_and_lens(
        query_start_loc=values.metadata.cu_seqlens_q,
        seq_lens=values.seq_lens,
        token_to_req_indices=values.token_to_req_indices,
        block_table=values.block_table,
        window_size=values.window_size,
        block_size=values.block_size,
        block_table_base_offsets=values.block_table_base_offsets,
        is_valid_token=values.is_valid_token,
    )

    expected_slot_mapping = deepseek_v4_compressed_slot_mapping_reference(values)
    actual_slot_mapping = deepseek_v4_compressed_slot_mapping(
        num_tokens=values.positions.numel(),
        query_start_loc=values.metadata.cu_seqlens_q,
        seq_lens=values.seq_lens,
        block_table=values.block_table,
        block_size=values.block_size,
        compress_ratio=values.compress_ratio,
    )

    expected_context_lens, expected_block_tables = (
        deepseek_v4_indexer_decode_metadata_reference(values)
    )
    actual_context_lens = torch.empty(
        values.positions.numel(),
        dtype=torch.int32,
        device=device,
    )
    actual_block_tables = torch.empty(
        values.positions.numel(),
        values.max_blocks,
        dtype=torch.int32,
        device=device,
    )
    deepseek_v4_indexer_decode_metadata_compute(
        positions=values.positions,
        token_to_req_indices=values.token_to_req_indices,
        block_table=values.block_table,
        cache_block_size=values.block_size,
        compress_ratio=values.compress_ratio,
        max_blocks=values.max_blocks,
        out_context_lens=actual_context_lens,
        out_block_tables=actual_block_tables,
        block_table_base_offsets=values.block_table_base_offsets,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_topk_indices, expected_topk_indices)
    assert torch.equal(actual_topk_lens, expected_topk_lens)
    assert torch.equal(actual_swa_indices, expected_swa_indices)
    assert torch.equal(actual_swa_lens, expected_swa_lens)
    assert torch.equal(actual_slot_mapping, expected_slot_mapping)
    assert torch.equal(actual_context_lens, expected_context_lens)
    assert torch.equal(actual_block_tables, expected_block_tables)


def test_deepseek_v4_local_compressed_generator_runs_tokenspeed_cpu() -> None:
    values = DeepSeekV4SparsePrefillIndexInputs(
        DeepSeekV4SparsePrefillIndexInputConfig(
            batch_size=3,
            total_cached_tokens=18,
            total_new_q_tokens=9,
            topk=4,
            window_size=8,
            compress_ratio=3,
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=3,
                total_cached_tokens=18,
                total_new_q_tokens=9,
                cache_layout="dense",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=207, device="cpu")
    out = torch.empty(
        values.positions.numel(),
        values.compressed_base,
        dtype=torch.int32,
        device="cpu",
    )

    actual = deepseek_v4_build_dense_prefill_local_compressed_indices(
        positions=values.positions,
        compress_ratio=values.compress_ratio,
        width=values.compressed_base,
        out=out,
    )
    expected = deepseek_v4_build_dense_prefill_local_compressed_indices_reference(
        values
    )

    assert torch.equal(actual, expected)


def test_deepseek_v4_sparse_prefill_combine_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton sparse-prefill index tests")

    values = DeepSeekV4SparsePrefillIndexInputs(
        DeepSeekV4SparsePrefillIndexInputConfig(
            batch_size=3,
            total_cached_tokens=18,
            total_new_q_tokens=9,
            topk=4,
            window_size=8,
            compress_ratio=3,
            metadata_input=MHARequestMetadataInputConfig(
                batch_size=3,
                total_cached_tokens=18,
                total_new_q_tokens=9,
                cache_layout="dense",
                cached_length_mode="regular",
                new_q_length_mode="fixed_per_request",
            ),
        )
    ).generate(seed=208, device=device)

    expected_topk_indices, expected_topk_lens = (
        deepseek_v4_combine_topk_swa_indices_reference(values)
    )
    actual_topk_indices, actual_topk_lens = deepseek_v4_combine_topk_swa_indices(
        topk_indices=values.topk_indices,
        query_start_loc=values.metadata.cu_seqlens_q,
        seq_lens=values.seq_lens,
        gather_lens=values.gather_lens,
        window_size=values.window_size,
        compress_ratio=values.compress_ratio,
        topk=values.topk,
        workspace_width=values.workspace_width,
        compressed_base=values.compressed_base,
    )
    expected_dense_indices, expected_dense_lens = (
        deepseek_v4_combine_dense_swa_indices_reference(values)
    )
    actual_dense_indices, actual_dense_lens = deepseek_v4_combine_dense_swa_indices(
        positions=values.positions,
        token_to_req_indices=values.token_to_req_indices,
        seq_lens=values.seq_lens,
        compressed_lens=values.compressed_lens,
        gather_lens=values.gather_lens,
        window_size=values.window_size,
        compress_ratio=values.compress_ratio,
        workspace_width=values.workspace_width,
        compressed_base=values.compressed_base,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_topk_indices, expected_topk_indices)
    assert torch.equal(actual_topk_lens, expected_topk_lens)
    assert torch.equal(actual_dense_indices, expected_dense_indices)
    assert torch.equal(actual_dense_lens, expected_dense_lens)


@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_mha_paged_decode_generator_runs_attention_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("attention", "mha_decode_with_kvcache", solution, dtype, "q")

    inputs = MHAInputs(
        _mha_config(
            batch_size=4,
            total_cached_tokens=51,
            total_new_q_tokens=4,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=64,
            include_sinks=True,
            metadata_kwargs={
                "cached_length_mode": "ragged",
                "new_q_length_mode": "fixed_per_request",
            },
        )
    ).generate(metadata_seed=103, value_seed=203, device=device)

    out = mha_decode_with_kvcache(
        **mha_decode_with_kvcache_kwargs(inputs),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()
