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
    KVCacheInput,
    KVCacheInputConfig,
    MHAInputConfig,
    MHAInputValues,
    MHAInputs,
    MLAInputConfig,
    MLAInputs,
    MLAKVCacheInput,
    MLAKVCacheInputConfig,
    MHARequestMetadataInputConfig,
    PageTableInput,
    PageTableInputConfig,
    attention_merge_state_reference,
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
