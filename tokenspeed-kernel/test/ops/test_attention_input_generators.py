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
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.numerics.input_generators import (
    KVCacheInput,
    KVCacheInputConfig,
    MHAInputConfig,
    MHAInputValues,
    MHAInputs,
    MHARequestMetadataInputConfig,
    PageTableInput,
    PageTableInputConfig,
    TensorInputConfig,
)


def _page_ids(inputs: MHAInputValues) -> list[int]:
    assert inputs.cache is not None
    assert inputs.cache.page_table_cpu is not None
    return [page_id for row in inputs.cache.page_table_cpu for page_id in row]


def test_page_table_input_generates_identity_and_random_indexing() -> None:
    identity = PageTableInput(
        batch_size=3,
        max_pages_per_request=5,
        indexing="identity",
    ).generate(seed=1, device="cpu")
    random = PageTableInput(
        batch_size=3,
        max_pages_per_request=5,
    ).generate(seed=1, device="cpu")

    expected_pages = list(range(15))

    assert identity.page_table.shape == (3, 5)
    assert identity.page_ids() == expected_pages
    assert sorted(random.page_ids()) == expected_pages
    assert random.page_ids() != expected_pages


def test_kv_cache_input_generates_dense_cache() -> None:
    cache = KVCacheInput(
        cache_layout="dense",
        batch_size=3,
        max_seqlen_k=17,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
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
            cache_layout="none",
            batch_size=3,
            max_seqlen_k=17,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
        )


def test_kv_cache_input_generates_paged_cache_with_nested_page_table() -> None:
    cache_input = KVCacheInput(
        cache_layout="paged",
        batch_size=3,
        max_seqlen_k=17,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        page_size=4,
        page_table_indexing="identity",
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


def test_kv_cache_input_preserves_nested_page_table_generator_object() -> None:
    page_table_input = PageTableInput(
        batch_size=1,
        max_pages_per_request=1,
        indexing="identity",
    )
    cache_input = KVCacheInput(
        cache_layout="paged",
        batch_size=3,
        max_seqlen_k=17,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        page_size=4,
        page_table_input=page_table_input,
    )

    assert cache_input.page_table_input is page_table_input

    cache = cache_input.generate(seed=1, device="cpu")

    assert cache.page_table_values is not None
    assert cache.page_table_values.page_ids() == list(range(15))


def test_mha_inputs_without_cache_leaves_cache_input_none() -> None:
    generator = MHAInputs(
        batch_size=3,
        total_cached_tokens=0,
        total_new_q_tokens=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=8,
        q_dtype=torch.float32,
        cache_layout="none",
        new_q_length_mode="fixed_per_request",
    )
    assert generator.cache_input is None

    inputs = generator.generate(metadata_seed=31, value_seed=41, device="cpu")

    assert generator.cache_input is None
    assert inputs.cache is None


def test_mha_inputs_keep_metadata_seed_independent_from_values() -> None:
    first = MHAInputs(
        batch_size=4,
        total_cached_tokens=31,
        total_new_q_tokens=9,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        page_size=8,
    ).generate(metadata_seed=11, value_seed=21, device="cpu")
    second = MHAInputs(
        batch_size=4,
        total_cached_tokens=31,
        total_new_q_tokens=9,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        page_size=8,
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
        batch_size=3,
        total_cached_tokens=0,
        total_new_q_tokens=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=8,
        q_dtype=torch.float32,
        cache_layout="none",
        new_q_length_mode="fixed_per_request",
    ).generate(metadata_seed=31, value_seed=41, device="cpu")

    q, k, v = inputs.dense_qkv()

    assert q.shape == (3, 4, 4, 8)
    assert k.shape == (3, 4, 2, 8)
    assert v.shape == (3, 4, 2, 8)
    assert inputs.metadata.cu_seqlens_q_cpu == [0, 4, 8, 12]


def test_mha_inputs_generate_bounded_ragged_request_lengths() -> None:
    inputs = MHAInputs(
        batch_size=5,
        total_cached_tokens=27,
        total_new_q_tokens=23,
        max_cached_tokens_per_request=9,
        max_new_q_tokens_per_request=8,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        page_size=8,
    ).generate(metadata_seed=1, value_seed=99, device="cpu")

    assert sum(inputs.metadata.cached_lens_cpu) == 27
    assert sum(inputs.metadata.new_q_lens_cpu) == 23
    assert max(inputs.metadata.cached_lens_cpu) <= 9
    assert max(inputs.metadata.new_q_lens_cpu) <= 8
    assert len(set(inputs.metadata.cached_lens_cpu)) > 1
    assert len(set(inputs.metadata.new_q_lens_cpu)) > 1


def test_mha_inputs_generate_identity_page_table_indexing() -> None:
    inputs = MHAInputs(
        batch_size=3,
        total_cached_tokens=36,
        total_new_q_tokens=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        cached_length_mode="regular",
        new_q_length_mode="fixed_per_request",
        page_size=4,
        page_table_indexing="identity",
    ).generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)

    assert page_ids == list(range(len(page_ids)))


def test_mha_inputs_generate_random_page_table_indexing_by_default() -> None:
    inputs = MHAInputs(
        batch_size=3,
        total_cached_tokens=36,
        total_new_q_tokens=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        cached_length_mode="regular",
        new_q_length_mode="fixed_per_request",
        page_size=4,
    ).generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)

    assert sorted(page_ids) == list(range(len(page_ids)))
    assert page_ids != list(range(len(page_ids)))


def test_mha_inputs_allows_nested_page_table_generator_configuration() -> None:
    generator = MHAInputs(
        batch_size=3,
        total_cached_tokens=36,
        total_new_q_tokens=12,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        cache_layout="paged",
        cached_length_mode="regular",
        new_q_length_mode="fixed_per_request",
        page_size=4,
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
        q_input=TensorInputConfig((0, 4, 16), torch.float64),
    )

    generator = MHAInputs(config)

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
    assert generator.q_input is not None
    assert generator.q_input.config is config.q_input

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


def test_mha_inputs_preserves_nested_cache_generator_object() -> None:
    cache_input = KVCacheInput(
        cache_layout="paged",
        batch_size=3,
        max_seqlen_k=1,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        page_size=4,
    )
    assert cache_input.page_table_input is not None
    cache_input.page_table_input.config.indexing = "identity"

    generator = MHAInputs(
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.float32,
        metadata_input=MHARequestMetadataInputConfig(
            batch_size=3,
            total_cached_tokens=36,
            total_new_q_tokens=12,
            cached_length_mode="regular",
            new_q_length_mode="fixed_per_request",
            cache_layout="paged",
        ),
        cache_input=cache_input,
    )

    assert generator.cache_input is cache_input

    inputs = generator.generate(metadata_seed=1, value_seed=99, device="cpu")

    page_ids = _page_ids(inputs)
    assert page_ids == list(range(len(page_ids)))


@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_mha_prefill_generator_runs_attention_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("attention", "mha_prefill", solution, dtype, "q")

    inputs = MHAInputs(
        batch_size=3,
        total_cached_tokens=0,
        total_new_q_tokens=33,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        q_dtype=dtype,
        cache_layout="none",
        new_q_length_mode="ragged",
        include_sinks=True,
    ).generate(metadata_seed=101, value_seed=201, device=device)

    out = mha_prefill(**inputs.as_mha_prefill_kwargs(), solution=solution)

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
        batch_size=4,
        total_cached_tokens=47,
        total_new_q_tokens=11,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        q_dtype=dtype,
        cache_layout="paged",
        cached_length_mode="ragged",
        new_q_length_mode="ragged",
        page_size=64,
        include_sinks=True,
    ).generate(metadata_seed=102, value_seed=202, device=device)

    out = mha_extend_with_kvcache(
        **inputs.as_mha_extend_with_kvcache_kwargs(),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()


@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_mha_paged_decode_generator_runs_attention_kernel(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("attention", "mha_decode_with_kvcache", solution, dtype, "q")

    inputs = MHAInputs(
        batch_size=4,
        total_cached_tokens=51,
        total_new_q_tokens=4,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        q_dtype=dtype,
        cache_layout="paged",
        cached_length_mode="ragged",
        new_q_length_mode="fixed_per_request",
        page_size=64,
        include_sinks=True,
    ).generate(metadata_seed=103, value_seed=203, device=device)

    out = mha_decode_with_kvcache(
        **inputs.as_mha_decode_with_kvcache_kwargs(),
        solution=solution,
    )

    assert inputs.q is not None
    assert out.shape == inputs.q.shape
    assert not torch.isnan(out.float()).any()
