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

import math

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
from tokenspeed_kernel.numerics.attention_kernel_kwargs import (
    mha_decode_with_kvcache_kwargs,
    mha_extend_with_kvcache_kwargs,
    mha_prefill_kwargs,
    mla_decode_with_kvcache_kwargs,
    mla_prefill_kwargs,
)
from tokenspeed_kernel.ops.attention.flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
)
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.ops.attention.flashinfer import (
    gated_delta_rule as flashinfer_gdn,
)
from tokenspeed_kernel.ops.attention.flashinfer.dsa_topk import (
    deterministic_decode_topk,
    has_deterministic_decode_topk,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_build_dense_prefill_local_compressed_indices,
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compressed_slot_mapping,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_dequantize_and_gather_k_cache,
    deepseek_v4_fused_csa_indexer_mxfp4_cache_insert,
    deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4,
    deepseek_v4_fused_inv_rope_fp8_quant,
    deepseek_v4_fused_sparse_compress_cache_insert,
    deepseek_v4_gather_indexer_mxfp4_cache,
    deepseek_v4_indexer_decode_metadata_compute,
    deepseek_v4_save_compressor_state,
    write_deepseek_v4_indexer_mxfp4_cache_cuda,
)
from tokenspeed_kernel.ops.attention.triton.dsa_sparse_layout import (
    full_context_topk_to_global_slots,
    local_topk_to_global_slots,
    pack_sparse_decode_kv,
)
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)
from tokenspeed_kernel.ops.attention.triton.qkv_rotary import packed_qkv_complex_rotary
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    AttentionMergeStateInputConfig,
    AttentionMergeStateInputs,
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
    MHAInputConfig,
    MHAInputs,
    MHARequestMetadataInputConfig,
    MLAInputConfig,
    MLAInputs,
    PackedQKVComplexRotaryInputConfig,
    PackedQKVComplexRotaryInputs,
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
    gdn_chunk_prefill_reference,
    gdn_qkv_split_reference,
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


def _skip_if_attention_extension_unavailable(exc: BaseException) -> None:
    message = str(exc)
    skip_fragments = (
        "Kernel implementation not found",
        "No module named",
        "requires Hopper",
        "requires SM",
    )
    if any(fragment in message for fragment in skip_fragments):
        pytest.skip(message)
    raise exc


def _attention_output(result: object) -> torch.Tensor:
    if isinstance(result, tuple):
        return result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError(
            f"attention result must be a tensor or tuple, got {type(result)}"
        )
    return result


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


def test_mha_generator_runs_direct_flash_attn_dense_kernel(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("FlashAttention dense smoke test requires NVIDIA Hopper+")

    dtype = torch.bfloat16
    head_dim = 64
    inputs = MHAInputs(
        _mha_config(
            batch_size=2,
            total_cached_tokens=0,
            total_new_q_tokens=16,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=head_dim,
            q_dtype=dtype,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "fixed_per_request"},
        )
    ).generate(metadata_seed=111, value_seed=211, device=device)
    q, k, v = inputs.dense_qkv()

    try:
        result = flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_attention_extension_unavailable(exc)
    out = _attention_output(result)
    if out.is_cuda:
        torch.cuda.synchronize()

    assert out.shape == q.shape
    assert not torch.isnan(out.float()).any()


def test_mha_generator_runs_direct_flash_attn_varlen_kernel(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("FlashAttention varlen smoke test requires NVIDIA Hopper+")

    dtype = torch.bfloat16
    head_dim = 64
    inputs = MHAInputs(
        _mha_config(
            batch_size=3,
            total_cached_tokens=0,
            total_new_q_tokens=29,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=head_dim,
            q_dtype=dtype,
            cache_layout="none",
            metadata_kwargs={"new_q_length_mode": "ragged"},
        )
    ).generate(metadata_seed=112, value_seed=212, device=device)
    assert inputs.q is not None
    assert inputs.k is not None
    assert inputs.v is not None

    try:
        result = flash_attn_varlen_func(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            cu_seqlens_q=inputs.metadata.cu_seqlens_q,
            cu_seqlens_k=inputs.metadata.cu_seqlens_q,
            max_seqlen_q=inputs.metadata.max_seqlen_q,
            max_seqlen_k=inputs.metadata.max_seqlen_q,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_attention_extension_unavailable(exc)
    out = _attention_output(result)
    if out.is_cuda:
        torch.cuda.synchronize()

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


def test_mla_generator_runs_direct_flash_mla_paged_decode_kernel(device: str) -> None:
    platform = current_platform()
    if not platform.is_nvidia or not platform.is_hopper_plus:
        pytest.skip("FlashMLA paged decode smoke test requires NVIDIA Hopper+")

    dtype = torch.bfloat16
    head_dim_v = 512
    qk_rope_head_dim = 64
    kv_cache_dim = head_dim_v + qk_rope_head_dim
    inputs = MLAInputs(
        _mla_config(
            batch_size=4,
            total_cached_tokens=128,
            total_new_q_tokens=4,
            num_q_heads=16,
            qk_nope_head_dim=head_dim_v,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=head_dim_v,
            v_head_dim=head_dim_v,
            q_dtype=dtype,
            cache_layout="paged",
            page_size=64,
            indexing="identity",
            metadata_kwargs={
                "cached_length_mode": "regular",
                "new_q_length_mode": "fixed_per_request",
                "max_seqlen_k": 64,
            },
        )
    ).generate(metadata_seed=113, value_seed=213, device=device)
    assert inputs.q is not None
    assert inputs.cache is not None
    assert inputs.cache.page_table is not None

    try:
        tile_scheduler_metadata, _ = get_mla_metadata()
        out, lse = flash_mla_with_kvcache(
            q=inputs.q,
            k_cache=inputs.cache.kv_cache,
            block_table=inputs.cache.page_table,
            cache_seqlens=inputs.metadata.cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            softmax_scale=1.0 / math.sqrt(kv_cache_dim),
            causal=True,
        )
    except (RuntimeError, ModuleNotFoundError) as exc:
        _skip_if_attention_extension_unavailable(exc)
    if out.is_cuda:
        torch.cuda.synchronize()

    assert out.shape == (4, 1, 16, head_dim_v)
    assert lse.shape == (4, 16, 1)
    assert not torch.isnan(out.float()).any()
    assert not torch.isnan(lse.float()).any()


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


def test_gdn_chunk_prefill_generator_runs_tokenspeed_flashinfer(device: str) -> None:
    if not flashinfer_gdn.is_available():
        pytest.skip("FlashInfer GDN chunk-prefill fast path is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for FlashInfer GDN chunk-prefill test")

    values = GDNChunkPrefillInputs(
        GDNChunkPrefillInputConfig(
            batch_size=2,
            total_tokens=32,
            num_q_heads=2,
            num_v_heads=4,
            head_dim=flashinfer_gdn.SUPPORTED_HEAD_DIM,
            dtype=torch.bfloat16,
            length_mode="regular",
            include_batch_dim=True,
            initial_state_dtype=torch.float32,
        )
    ).generate(seed=209, device=device)
    expected = gdn_chunk_prefill_reference(values)

    actual_out, actual_state = flashinfer_gdn.gdn_chunk_prefill(
        values.q,
        values.k,
        values.v,
        values.g,
        values.beta,
        scale=values.scale,
        initial_state=values.initial_state,
        cu_seqlens=values.cu_seqlens,
    )
    torch.cuda.synchronize()

    assert actual_out.shape == expected.out.shape
    assert actual_state.shape == expected.final_state.shape
    assert actual_out.dtype == values.q.dtype
    torch.testing.assert_close(
        actual_out.float(),
        expected.out.float(),
        rtol=1e-2,
        atol=1e-1,
    )
    assert (actual_state.float() - expected.final_state).abs().mean() < 1e-2


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


def test_dsa_decode_topk_generator_runs_flashinfer_kernel(device: str) -> None:
    if not torch.cuda.is_available() or not has_deterministic_decode_topk():
        pytest.skip("FlashInfer deterministic DSA top-k requires CUDA and flashinfer")

    values = DSADecodeTopKInputs(
        DSADecodeTopKInputConfig(
            num_rows=5,
            vocab_size=32,
            topk=6,
            dtype=torch.float32,
            min_valid_len=8,
            max_valid_len=32,
        )
    ).generate(seed=213, device=device)
    expected = dsa_decode_topk_reference(values)

    deterministic_decode_topk(values.logits, values.out, values.topk)
    torch.cuda.synchronize()

    assert torch.equal(values.out, expected)


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


def test_deepseek_v4_compressor_state_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton compressor-state test")

    values = DeepSeekV4CompressorStateInputs(
        DeepSeekV4CompressorStateInputConfig(
            num_tokens=6,
            state_width=16,
            num_cache_blocks=2,
            block_size=4,
            compress_ratio=4,
            dtype=torch.bfloat16,
            invalid_token_count=1,
        )
    ).generate(seed=2091, device=device)
    expected = deepseek_v4_save_compressor_state_reference(values)
    actual = values.state_cache.clone()

    deepseek_v4_save_compressor_state(
        kv=values.kv,
        score=values.score,
        ape=values.ape,
        state_cache=actual,
        slot_mapping=values.slot_mapping,
        positions=values.positions,
        block_size=values.block_size,
        compress_ratio=values.compress_ratio,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_deepseek_v4_indexer_q_rope_hadamard_mxfp4_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton indexer-Q MXFP4 test")

    values = DeepSeekV4IndexerQRoPEHadamardMXFP4Inputs(
        DeepSeekV4IndexerQRoPEHadamardMXFP4InputConfig(
            num_tokens=4,
            num_heads=3,
            dtype=torch.bfloat16,
            max_position=32,
            softmax_scale=0.25,
            head_scale=2.0,
        )
    ).generate(seed=2092, device=device)
    values.index_q.zero_()
    values.index_q[..., 0] = 4.0
    (expected_q_packed, expected_q_scale), expected_weights = (
        deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference(values)
    )

    (actual_q_packed, actual_q_scale), actual_weights = (
        deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4(
            index_q=values.index_q,
            positions=values.positions,
            cos_sin_cache=values.cos_sin_cache,
            weights=values.weights,
            softmax_scale=values.softmax_scale,
            head_scale=values.head_scale,
        )
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q_packed, expected_q_packed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_q_scale, expected_q_scale, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-6)


def test_deepseek_v4_inv_rope_fp8_quant_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton inverse-RoPE FP8 test")

    values = DeepSeekV4InvRoPEFP8QuantInputs(
        DeepSeekV4InvRoPEFP8QuantInputConfig(
            num_tokens=5,
            n_groups=2,
            heads_per_group=2,
            dtype=torch.bfloat16,
            max_position=32,
            value_scale=0.25,
        )
    ).generate(seed=2093, device=device)
    expected_fp8, expected_scale = deepseek_v4_inv_rope_fp8_quant_reference(values)

    actual_fp8, actual_scale = deepseek_v4_fused_inv_rope_fp8_quant(
        values.o,
        values.positions,
        values.cos_sin_cache,
        n_groups=values.n_groups,
        heads_per_group=values.heads_per_group,
        nope_dim=values.nope_dim,
        rope_dim=values.rope_dim,
        quant_group_size=values.quant_group_size,
        tma_aligned_scales=values.tma_aligned_scales,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_fp8.float(), expected_fp8.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


def test_deepseek_v4_csa_indexer_mxfp4_cache_insert_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton CSA indexer cache test")

    values = DeepSeekV4CSAIndexerMXFP4CacheInsertInputs(
        DeepSeekV4CSAIndexerMXFP4CacheInsertInputConfig(
            num_tokens=4,
            batch_size=1,
            max_seq_len=4,
            num_state_cache_blocks=1,
            compressor_block_size=4,
            num_kv_cache_blocks=1,
            kv_cache_block_size=4,
            dtype=torch.bfloat16,
            value_scale=0.0,
        )
    ).generate(seed=2094, device=device)
    values.token_to_req_indices.zero_()
    values.positions.fill_(3)
    values.compressor_slot_mapping = torch.arange(4, dtype=torch.int64, device=device)
    values.kv_slot_mapping = torch.arange(4, dtype=torch.int64, device=device)
    values.block_table.zero_()
    values.state_cache.zero_()
    values.state_cache[:, :, 128:256] = 1.0
    values.rms_norm_weight.fill_(1.0)
    expected = deepseek_v4_csa_indexer_mxfp4_cache_insert_reference(values)
    actual = values.kv_cache_2d.clone()

    deepseek_v4_fused_csa_indexer_mxfp4_cache_insert(
        state_cache=values.state_cache,
        token_to_req_indices=values.token_to_req_indices,
        positions=values.positions,
        compressor_slot_mapping=values.compressor_slot_mapping,
        block_table=values.block_table,
        compressor_block_size=values.compressor_block_size,
        rms_norm_weight=values.rms_norm_weight,
        rms_norm_eps=values.rms_norm_eps,
        cos_sin_cache=values.cos_sin_cache,
        kv_cache_2d=actual,
        kv_slot_mapping=values.kv_slot_mapping,
        kv_cache_block_size=values.kv_cache_block_size,
        compress_ratio=values.compress_ratio,
        block_table_base_offsets=values.block_table_base_offsets,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_deepseek_v4_sparse_compress_cache_insert_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton sparse-compress test")

    values = DeepSeekV4SparseCompressCacheInsertInputs(
        DeepSeekV4SparseCompressCacheInsertInputConfig(
            num_tokens=4,
            batch_size=1,
            max_seq_len=4,
            num_state_cache_blocks=1,
            compressor_block_size=4,
            num_kv_cache_blocks=1,
            kv_cache_block_size=4,
            compress_ratio=4,
            overlap=True,
            dtype=torch.bfloat16,
            value_scale=0.0,
        )
    ).generate(seed=2095, device=device)
    values.token_to_req_indices.zero_()
    values.positions.fill_(3)
    values.compressor_slot_mapping = torch.arange(4, dtype=torch.int64, device=device)
    values.kv_slot_mapping = torch.arange(4, dtype=torch.int64, device=device)
    values.block_table.zero_()
    values.state_cache.zero_()
    values.state_cache[:, :, 512:1024] = 1.0
    values.rms_norm_weight.fill_(1.0)
    expected = deepseek_v4_sparse_compress_cache_insert_reference(values)
    actual = values.kv_cache_2d.clone()

    deepseek_v4_fused_sparse_compress_cache_insert(
        state_cache=values.state_cache,
        token_to_req_indices=values.token_to_req_indices,
        positions=values.positions,
        compressor_slot_mapping=values.compressor_slot_mapping,
        block_table=values.block_table,
        compressor_block_size=values.compressor_block_size,
        rms_norm_weight=values.rms_norm_weight,
        rms_norm_eps=values.rms_norm_eps,
        cos_sin_cache=values.cos_sin_cache,
        kv_cache_2d=actual,
        kv_slot_mapping=values.kv_slot_mapping,
        kv_cache_block_size=values.kv_cache_block_size,
        compress_ratio=values.compress_ratio,
        overlap=values.overlap,
        block_table_base_offsets=values.block_table_base_offsets,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_deepseek_v4_indexer_mxfp4_cache_write_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton MXFP4 cache-write test")

    values = DeepSeekV4IndexerMXFP4CacheWriteInputs(
        DeepSeekV4IndexerMXFP4CacheWriteInputConfig(
            num_rows=5,
            num_cache_blocks=2,
            block_size=4,
            dtype=torch.bfloat16,
            negative_slot_count=1,
            masked_row_count=1,
        )
    ).generate(seed=2092, device=device)
    expected = deepseek_v4_indexer_mxfp4_cache_write_reference(values)
    actual = values.cache_2d.clone()

    write_deepseek_v4_indexer_mxfp4_cache_cuda(
        values.index_k,
        actual,
        values.slot_mapping,
        values.valid,
        values.block_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_deepseek_v4_indexer_mxfp4_cache_gather_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton MXFP4 cache-gather test")

    values = DeepSeekV4IndexerMXFP4CacheGatherInputs(
        DeepSeekV4IndexerMXFP4CacheGatherInputConfig(
            num_rows=6,
            num_cache_blocks=2,
            block_size=4,
            negative_slot_count=2,
        )
    ).generate(seed=2093, device=device)
    expected_values, expected_scales = deepseek_v4_indexer_mxfp4_cache_gather_reference(
        values
    )
    actual_values = values.values_out.clone()
    actual_scales = values.scales_out.clone()

    deepseek_v4_gather_indexer_mxfp4_cache(
        cache_2d=values.cache_2d,
        slot_mapping=values.slot_mapping,
        values_out=actual_values,
        scales_out=actual_scales,
        block_size=values.block_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_values, expected_values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0.0, atol=0.0)


def test_deepseek_v4_k_cache_gather_generator_runs_tokenspeed_triton(
    device: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU is required for Triton K-cache gather test")

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
    ).generate(seed=2094, device=device)
    expected = deepseek_v4_dequantize_and_gather_k_cache_reference(values)
    actual = values.out.clone()

    deepseek_v4_dequantize_and_gather_k_cache(
        out=actual,
        cache_2d=values.cache_2d,
        seq_lens=values.seq_lens,
        gather_lens=values.gather_lens,
        block_table=values.block_table,
        block_size=values.block_size,
        offset=values.offset,
        block_table_base_offsets=values.block_table_base_offsets,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


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
