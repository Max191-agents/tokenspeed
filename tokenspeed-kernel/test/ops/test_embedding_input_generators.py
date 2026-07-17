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
from tokenspeed_kernel.numerics.reference.embedding import (
    mla_rope_quantize_fp8_reference,
    rope_reference,
)
from tokenspeed_kernel.ops.embedding import FusedSetKVBufferArg, apply_rope
from tokenspeed_kernel.ops.embedding.flashinfer import mla_rope_quantize_fp8
from tokenspeed_kernel.platform import current_platform
from tokenspeed_numerics_input_generators import (
    MLARopeInputConfig,
    MLARopeInputs,
    RopeInputConfig,
    RopeInputs,
)

platform = current_platform()


def test_rope_reference_preserves_partial_rotary_tail() -> None:
    config = RopeInputConfig(
        num_tokens=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=16,
        rotary_dim=8,
        dtype=torch.float32,
        is_neox=True,
    )
    values = RopeInputs(config).generate(seed=23, metadata_seed=24, device="cpu")

    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )

    torch.testing.assert_close(
        q_ref.view(3, 2, 16)[..., 8:],
        values.query.view(3, 2, 16)[..., 8:],
    )
    torch.testing.assert_close(
        k_ref.view(3, 1, 16)[..., 8:],
        values.key.view(3, 1, 16)[..., 8:],
    )


@pytest.mark.parametrize(
    ("k_rank", "num_kv_heads", "expected_key_shape"),
    [(2, 1, (4, 13)), (3, 2, (4, 2, 13))],
)
def test_mla_rope_quantize_fp8_reference(
    k_rank: int,
    num_kv_heads: int,
    expected_key_shape: tuple[int, ...],
) -> None:
    quant_scale_q = 0.75
    quant_scale_kv = 1.25
    fp8_dtype = torch.float8_e4m3fn
    config = MLARopeInputConfig(
        num_tokens=4,
        num_q_heads=3,
        qk_nope_head_dim=5,
        qk_rope_head_dim=8,
        dtype=torch.bfloat16,
        k_rank=k_rank,
        num_kv_heads=num_kv_heads,
        max_position=32,
    )
    values = MLARopeInputs(config).generate(
        seed=31,
        metadata_seed=32,
        device="cpu",
    )

    ref = mla_rope_quantize_fp8_reference(
        values,
        fp8_dtype=fp8_dtype,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        is_neox=config.is_neox,
    )

    assert ref.query.shape == (4, 3, 13)
    assert ref.key.shape == expected_key_shape
    assert ref.query.dtype == fp8_dtype
    assert ref.key.dtype == fp8_dtype
    torch.testing.assert_close(
        ref.q_nope.view(torch.uint8),
        (values.q_nope.float() * quant_scale_q).to(fp8_dtype).view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("is_neox", [True, False])
def test_rope_generator_runs_triton_kernel_with_output_buffers(
    device: str,
    require,
    is_neox: bool,
) -> None:
    dtype = torch.bfloat16
    require("embedding", "rope", "triton", dtype, "query")
    config = RopeInputConfig(
        num_tokens=9,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=64,
        rotary_dim=32,
        dtype=dtype,
        is_neox=is_neox,
        with_q_output=True,
        with_k_output=True,
    )
    values = RopeInputs(config).generate(seed=71, metadata_seed=72, device=device)
    query_orig = values.query.clone()
    key_orig = values.key.clone()
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )

    q_out, k_out = apply_rope(
        positions=values.positions,
        query=values.query,
        key=values.key,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
        output_q_rope=values.output_q_rope,
        output_k_rope=values.output_k_rope,
        solution="triton",
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(values.query, query_orig, atol=0, rtol=0)
    torch.testing.assert_close(values.key, key_orig, atol=0, rtol=0)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


def test_rope_generator_runs_triton_fused_kv_kernel(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("embedding", "rope", "triton", dtype, "query")
    config = RopeInputConfig(
        num_tokens=7,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=64,
        dtype=dtype,
        with_fused_kv=True,
        cache_size=16,
        with_q_output=True,
    )
    values = RopeInputs(config).generate(seed=73, metadata_seed=74, device=device)
    assert values.fused_kv is not None
    assert values.output_q_rope is not None
    query_orig = values.query.clone()
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )

    q_out, k_out = apply_rope(
        positions=values.positions,
        query=values.query,
        key=values.key,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
        fused_set_kv_buffer_arg=FusedSetKVBufferArg(
            value=values.fused_kv.value,
            k_buffer=values.fused_kv.k_buffer,
            v_buffer=values.fused_kv.v_buffer,
            k_scale=values.fused_kv.k_scale,
            v_scale=values.fused_kv.v_scale,
            cache_loc=values.fused_kv.cache_loc,
        ),
        output_q_rope=values.output_q_rope,
        solution="triton",
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(values.query, query_orig, atol=0, rtol=0)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        values.fused_kv.k_buffer.index_select(0, values.fused_kv.cache_loc),
        k_ref,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        values.fused_kv.v_buffer.index_select(0, values.fused_kv.cache_loc),
        values.fused_kv.value.reshape(config.num_tokens, -1),
        atol=0,
        rtol=0,
    )


def test_rope_generator_runs_cuda_kernel_with_output_buffers(
    device: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("embedding", "rope", "cuda", dtype, "query")
    config = RopeInputConfig(
        num_tokens=8,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=64,
        rotary_dim=64,
        dtype=dtype,
        is_neox=True,
        with_q_output=True,
        with_k_output=True,
    )
    values = RopeInputs(config).generate(seed=75, metadata_seed=76, device=device)
    query_orig = values.query.clone()
    key_orig = values.key.clone()
    q_ref, k_ref = rope_reference(
        values.query,
        values.key,
        values.positions,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
    )

    q_out, k_out = apply_rope(
        positions=values.positions,
        query=values.query,
        key=values.key,
        head_size=config.head_size,
        cos_sin_cache=values.cos_sin_cache,
        is_neox=config.is_neox,
        rotary_dim=config.rotary_dim,
        output_q_rope=values.output_q_rope,
        output_k_rope=values.output_k_rope,
        solution="cuda",
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(values.query, query_orig, atol=0, rtol=0)
    torch.testing.assert_close(values.key, key_orig, atol=0, rtol=0)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not platform.is_nvidia,
    reason="FlashInfer MLA RoPE FP8 quantization requires NVIDIA CUDA.",
)
def test_mla_rope_generator_adapts_to_flashinfer_fp8_kernel(device: str) -> None:
    fp8_dtype = torch.float8_e4m3fn
    quant_scale_q = 1.0
    quant_scale_kv = 0.75
    config = MLARopeInputConfig(
        num_tokens=5,
        num_q_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=16,
        dtype=torch.bfloat16,
        max_position=64,
    )
    values = MLARopeInputs(config).generate(
        seed=77,
        metadata_seed=78,
        device=device,
    )
    q_rope_out = torch.empty_like(values.q_rope, dtype=fp8_dtype)
    k_rope_out = torch.empty_like(values.k_rope, dtype=fp8_dtype)
    q_nope_out = torch.empty_like(values.q_nope, dtype=fp8_dtype)
    k_nope_out = torch.empty_like(values.k_nope, dtype=fp8_dtype)
    expected = mla_rope_quantize_fp8_reference(
        values,
        fp8_dtype=fp8_dtype,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        is_neox=config.is_neox,
    )

    try:
        mla_rope_quantize_fp8(
            q_rope=values.q_rope,
            k_rope=values.k_rope,
            q_nope=values.q_nope,
            k_nope=values.k_nope,
            cos_sin_cache=values.cos_sin_cache,
            pos_ids=values.positions,
            is_neox=config.is_neox,
            quantize_dtype=fp8_dtype,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
            q_nope_out=q_nope_out,
            k_nope_out=k_nope_out,
            quant_scale_q=quant_scale_q,
            quant_scale_kv=quant_scale_kv,
            enable_pdl=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer MLA RoPE FP8 quantization unavailable: {exc}")
    torch.cuda.synchronize()

    torch.testing.assert_close(
        q_nope_out.view(torch.uint8),
        expected.q_nope.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        q_rope_out.view(torch.uint8),
        expected.q_rope.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        k_nope_out.view(torch.uint8),
        expected.k_nope.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        k_rope_out.view(torch.uint8),
        expected.k_rope.view(torch.uint8),
        atol=0,
        rtol=0,
    )
