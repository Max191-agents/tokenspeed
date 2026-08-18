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
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for dense16 Gluon GEMM tests",
        allow_module_level=True,
    )


from tokenspeed_kernel.ops.gemm import bmm_fp8_output  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (  # noqa: E402
    _supports_largem_shape,
    gluon_mm_a16w16_largem_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (  # noqa: E402
    BUFFER_OFFSET_I32_MAX,
    _choose_mfma_lds_mediumm_config,
    _choose_skinny_bmm_mfma_config,
    _fits_i32_buffer_offsets,
    _fits_i32_element_offsets,
    _get_partial_scratch,
    _has_skinny_bmm_load_alignment,
    _supports_mfma_lds_smallm,
    _use_mfma_lds_largem,
    _use_mfma_lds_mediumm,
    _use_mfma_lds_smallm,
    _use_warp_reduce_smallm,
    gluon_bmm_a16w16_gfx950,
    gluon_bmm_a16w16_skinny_gfx950,
    gluon_mm_a16w16_gfx950,
    gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
    gluon_mm_a16w16_mfma_lds_smallm_gfx950,
    gluon_mm_a16w16_warp_reduce_smallm_gfx950,
)

_CORRECTNESS_CASES = [
    pytest.param(
        gluon_mm_a16w16_warp_reduce_smallm_gfx950,
        (2, 128, 1024),
        id="warp-reduce",
    ),
    pytest.param(
        gluon_mm_a16w16_mfma_lds_smallm_gfx950,
        (4, 256, 2048),
        id="splitk-smallm",
    ),
    pytest.param(
        gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
        (8, 128, 64),
        id="mediumm-one-buffer",
    ),
    pytest.param(
        gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
        (8, 1280, 512),
        id="mediumm-two-buffer",
    ),
    pytest.param(
        gluon_mm_a16w16_mfma_lds_mediumm_gfx950,
        (16, 1280, 768),
        id="mediumm-three-buffer",
    ),
    pytest.param(
        gluon_mm_a16w16_largem_gfx950,
        (256, 256, 256),
        id="largem",
    ),
]


@pytest.mark.parametrize("kernel,shape", _CORRECTNESS_CASES)
def test_dense16_kernel_variant_correctness(
    kernel, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25

    out = kernel(a, b, dtype)
    assert out is not None

    torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("kernel,shape", _CORRECTNESS_CASES)
def test_dense16_kernel_variant_writes_strided_out(
    kernel, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25
    backing = torch.empty((m, n + 17), device="cuda", dtype=dtype)
    out = backing[:, :n]

    actual = kernel(a, b, dtype, out=out)

    assert actual is out
    torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch", [12, 16])
def test_dense16_bmm_writes_strided_out(batch: int) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = 1, 512, 128
    a_backing = torch.randn((m, batch, k), device="cuda", dtype=dtype) * 0.25
    a = a_backing.transpose(0, 1)
    weight = torch.randn((batch, k, n), device="cuda", dtype=dtype) * 0.25
    b = weight.transpose(1, 2)
    backing = torch.empty((m, batch, n + 17), device="cuda", dtype=dtype)
    out = backing[..., :n].transpose(0, 1)

    actual = gluon_bmm_a16w16_gfx950(a, b, dtype, out=out)

    assert actual is out
    torch.testing.assert_close(out, torch.bmm(a, weight), atol=1e-2, rtol=1e-2)


def test_dense16_bmm_rejects_unsupported_shape() -> None:
    a = torch.empty((12, 2, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((12, 512, 128), device="cuda", dtype=torch.bfloat16)

    assert gluon_bmm_a16w16_gfx950(a, b, torch.bfloat16) is None


@pytest.mark.parametrize("tokens", [1, 4, 5, 16, 17, 32])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_skinny_bmm_writes_strided_output(tokens: int, out_dtype: torch.dtype) -> None:
    torch.manual_seed(71)
    heads, k, n = 16, 192, 512
    q_backing = (
        torch.randn((tokens, heads, 256), device="cuda", dtype=torch.bfloat16) * 0.25
    )
    q = q_backing[..., :k].transpose(0, 1)
    weight = torch.randn((heads, k, n), device="cuda", dtype=torch.bfloat16) * 0.25
    b = weight.transpose(1, 2)
    combined = torch.full(
        (tokens, heads, n + 64),
        1.0,
        device="cuda",
        dtype=out_dtype,
    )
    out = combined[..., :n].transpose(0, 1)
    suffix_bytes = combined[..., n:].view(torch.uint8).clone()

    assert q.stride() == (256, 4096, 1)
    assert b.stride() == (98304, 1, 512)
    assert out.stride() == (576, 9216, 1)

    actual = gluon_bmm_a16w16_skinny_gfx950(q, b, out_dtype, out=out)

    expected = torch.bmm(q, weight).to(out_dtype)
    assert actual is out
    if out_dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.5)
    else:
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.25)
    assert torch.equal(combined[..., n:].view(torch.uint8), suffix_bytes)


@pytest.mark.parametrize("tokens", [1, 16])
def test_skinny_fp8_bmm_rounds_through_bf16(tokens: int) -> None:
    heads, k, n = 16, 192, 512
    q_backing = torch.zeros((tokens, heads, 256), device="cuda", dtype=torch.bfloat16)
    q_backing[..., :2] = 1.0
    q = q_backing[..., :k].transpose(0, 1)
    weight = torch.zeros((heads, k, n), device="cuda", dtype=torch.bfloat16)
    weight[:, 0, :] = 2.125
    weight[:, 1, :] = 2**-10

    actual = gluon_bmm_a16w16_skinny_gfx950(
        q, weight.transpose(1, 2), torch.float8_e4m3fn
    )

    assert actual is not None
    staged = torch.bmm(q, weight).to(torch.float8_e4m3fn)
    direct = torch.tensor(2.125 + 2**-10, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    assert staged.float().unique().item() == 2.0
    assert direct.float().item() == 2.25
    assert torch.equal(actual.view(torch.uint8), staged.view(torch.uint8))


@pytest.mark.parametrize("tokens", [3, 16])
def test_skinny_fp8_bmm_saturates_output(tokens: int) -> None:
    heads, k, n = 16, 192, 512
    q = torch.full((heads, tokens, k), 4.0, device="cuda", dtype=torch.bfloat16)
    weight = torch.full((heads, k, n), 4.0, device="cuda", dtype=torch.bfloat16)

    output = gluon_bmm_a16w16_skinny_gfx950(
        q,
        weight.transpose(1, 2),
        torch.float8_e4m3fn,
    )

    assert output is not None
    assert output.float().unique().item() == torch.finfo(output.dtype).max


@pytest.mark.parametrize(
    ("shape", "mutate"),
    [
        pytest.param((65, 1, 64, 64), None, id="batch-too-large"),
        pytest.param((1, 33, 64, 64), None, id="m-too-large"),
        pytest.param((1, 1, 48, 64), None, id="n-unaligned"),
        pytest.param((1, 1, 64, 96), None, id="k-unaligned"),
        pytest.param((1, 1, 64, 64), "a-stride", id="activation-stride"),
        pytest.param((1, 1, 64, 64), "b-stride", id="weight-stride"),
    ],
)
def test_skinny_bmm_falls_back_for_each_unsupported_override_shape(
    shape: tuple[int, int, int, int], mutate: str | None
) -> None:
    batch, M, N, K = shape
    a = torch.zeros((batch, M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.zeros((batch, K, N), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    if mutate == "a-stride":
        a = torch.zeros((batch, M, K * 2), device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif mutate == "b-stride":
        b = torch.zeros((batch, N, K), device="cuda", dtype=torch.bfloat16)

    actual = bmm_fp8_output(
        a,
        b,
        override="gluon_bmm_a16w16_skinny_gfx950",
    )

    expected = torch.bmm(a, b.transpose(1, 2)).to(torch.float8_e4m3fn)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_skinny_bmm_forced_override_accepts_n_aligned_to_32(device: str) -> None:
    batch, M, N, K = 3, 5, 96, 192
    a = torch.randn((batch, M, K), device=device, dtype=torch.bfloat16)
    weight = torch.randn((batch, K, N), device=device, dtype=torch.bfloat16)

    actual = bmm_fp8_output(
        a,
        weight.transpose(1, 2),
        override="gluon_bmm_a16w16_skinny_gfx950",
    )

    expected = torch.bmm(a, weight).to(torch.float8_e4m3fn)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.5)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_skinny_bmm_executes_advertised_batch_and_depth_boundary(
    out_dtype: torch.dtype,
) -> None:
    torch.manual_seed(73)
    batch, M, N, K = 64, 32, 64, 1024
    a = torch.randn((batch, M, K), device="cuda", dtype=torch.bfloat16) * 0.0625
    weight = torch.randn((batch, K, N), device="cuda", dtype=torch.bfloat16) * 0.0625

    actual = gluon_bmm_a16w16_skinny_gfx950(a, weight.transpose(1, 2), out_dtype)

    assert actual is not None
    expected = torch.bmm(a, weight).to(out_dtype)
    if out_dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.5)
    else:
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.25)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param((5, 64, 64), (16, 32, 64, 2, 2, 1), id="one-buffer"),
        pytest.param((16, 512, 128), (16, 32, 64, 2, 2, 2), id="two-buffer"),
        pytest.param((16, 512, 192), (16, 32, 64, 2, 2, 3), id="glm"),
        pytest.param((16, 512, 512), (16, 32, 256, 2, 2, 2), id="bk256"),
        pytest.param((17, 512, 320), (32, 32, 64, 2, 2, 3), id="bm32"),
        pytest.param((32, 512, 1024), (16, 32, 512, 2, 2, 2), id="k1024"),
    ],
)
def test_choose_skinny_bmm_mfma_config(
    shape: tuple[int, int, int], expected: tuple[int, int, int, int, int, int]
) -> None:
    assert _choose_skinny_bmm_mfma_config(*shape) == expected


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4, 512, 192), id="row-policy"),
        pytest.param((33, 512, 192), id="m-too-large"),
        pytest.param((16, 48, 192), id="n-unaligned"),
        pytest.param((16, 512, 32), id="k-too-small"),
        pytest.param((16, 512, 1088), id="k-too-large"),
    ],
)
def test_choose_skinny_bmm_mfma_config_rejects_unsupported_shapes(
    shape: tuple[int, int, int],
) -> None:
    assert _choose_skinny_bmm_mfma_config(*shape) is None


def test_choose_skinny_bmm_mfma_config_covers_advertised_domain() -> None:
    for M in range(5, 33):
        for K in range(64, 1025, 64):
            config = _choose_skinny_bmm_mfma_config(M, 64, K)
            assert config is not None
            block_m, block_n, block_k, warps_m, warps_n, num_buffers = config
            assert block_m in (16, 32)
            assert block_n == 32
            assert K % block_k == 0
            assert warps_m * warps_n == 4
            k_tiles = K // block_k
            if num_buffers == 1:
                assert k_tiles == 1
            else:
                assert k_tiles >= num_buffers


def test_skinny_bmm_buffer_offsets_are_bounded_for_async_copy() -> None:
    assert _fits_i32_buffer_offsets((64, 32, 1024), (32768, 1024, 1), 2)
    assert _fits_i32_buffer_offsets((16, 512, 192), (98304, 1, 512), 2)
    assert _fits_i32_buffer_offsets((2,), (BUFFER_OFFSET_I32_MAX // 2 - 1,), 2)
    assert not _fits_i32_buffer_offsets((2,), (BUFFER_OFFSET_I32_MAX // 2,), 2)
    assert not _fits_i32_buffer_offsets((2, 1), (2**31, 1), 1)
    assert not _fits_i32_buffer_offsets((2, 1), (-1, 1), 1)
    assert not _fits_i32_buffer_offsets((2, 1), (1, 1), 0)
    assert not _fits_i32_buffer_offsets((0, 1), (1, 1), 1)


def test_skinny_bmm_output_offsets_use_element_indices() -> None:
    assert _fits_i32_element_offsets((2,), (BUFFER_OFFSET_I32_MAX,))
    assert not _fits_i32_element_offsets((2,), (BUFFER_OFFSET_I32_MAX + 1,))


@pytest.mark.parametrize("operand", ["activation", "weight"])
def test_skinny_bmm_rejects_misaligned_load_base_on_cpu(operand: str) -> None:
    batch, M, N, K = 2, 5, 64, 64
    a = torch.empty((batch, M, K), dtype=torch.bfloat16)
    weight = torch.empty((batch, K, N), dtype=torch.bfloat16)
    b = weight.transpose(1, 2)
    assert _has_skinny_bmm_load_alignment(a, b)

    if operand == "activation":
        storage = torch.empty(batch * M * K + 1, dtype=torch.bfloat16)
        a = storage[1:].view(batch, M, K)
        assert a.data_ptr() % 16 == torch.bfloat16.itemsize
    else:
        storage = torch.empty(batch * K * N + 1, dtype=torch.bfloat16)
        b = storage[1:].view(batch, K, N).transpose(1, 2)
        assert b.data_ptr() % 16 == torch.bfloat16.itemsize

    assert not _has_skinny_bmm_load_alignment(a, b)


class _MetadataOnlyCudaTensor:
    dtype = torch.bfloat16
    ndim = 3
    is_cuda = True
    device = torch.device("cuda")

    def __init__(
        self,
        shape: tuple[int, int, int],
        strides: tuple[int, int, int],
        data_ptr: int = 0x1000,
    ) -> None:
        self.shape = shape
        self._strides = strides
        self._data_ptr = data_ptr

    def stride(self, dim: int) -> int:
        return self._strides[dim]

    def data_ptr(self) -> int:
        return self._data_ptr

    def element_size(self) -> int:
        return torch.bfloat16.itemsize


@pytest.mark.parametrize(
    ("a_strides", "b_strides", "a_ptr", "b_ptr"),
    [
        pytest.param((320, 64, 1), (4096, 1, 64), 0x1002, 0x2000, id="a-base"),
        pytest.param((320, 64, 1), (4096, 1, 64), 0x1000, 0x2002, id="b-base"),
        pytest.param((321, 64, 1), (4096, 1, 64), 0x1000, 0x2000, id="a-batch"),
        pytest.param((328, 65, 1), (4096, 1, 64), 0x1000, 0x2000, id="a-row"),
        pytest.param((320, 64, 1), (4097, 1, 64), 0x1000, 0x2000, id="b-batch"),
        pytest.param((320, 64, 1), (4160, 1, 65), 0x1000, 0x2000, id="b-k"),
    ],
)
def test_skinny_bmm_wrapper_rejects_unsafe_load_alignment_without_gpu(
    a_strides: tuple[int, int, int],
    b_strides: tuple[int, int, int],
    a_ptr: int,
    b_ptr: int,
) -> None:
    a = _MetadataOnlyCudaTensor((2, 5, 64), a_strides, a_ptr)
    b = _MetadataOnlyCudaTensor((2, 64, 64), b_strides, b_ptr)

    assert gluon_bmm_a16w16_skinny_gfx950(a, b, torch.bfloat16) is None


def test_splitk_smallm_out_handles_padded_reducer_rows() -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    m, n, k = 2, 256, 2048
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((n, k), device="cuda", dtype=dtype) * 0.25
    backing = torch.empty((m, n + 17), device="cuda", dtype=dtype)
    out = backing[:, :n]

    actual = gluon_mm_a16w16_mfma_lds_smallm_gfx950(a, b, dtype, out=out)

    assert actual is out
    torch.testing.assert_close(out, torch.mm(a, b.T), atol=1e-2, rtol=1e-2)


def test_use_warp_reduce_covers_small_k_decode_shapes() -> None:
    assert _use_warp_reduce_smallm(1, 1280, 1024)
    assert _use_warp_reduce_smallm(2, 2560, 2048)
    assert _use_warp_reduce_smallm(4, 1280, 512)
    assert _use_warp_reduce_smallm(4, 1280, 1024)


def test_use_warp_reduce_rejects_splitk_or_medium_shapes() -> None:
    assert not _use_warp_reduce_smallm(1, 1280, 2880)
    assert not _use_warp_reduce_smallm(4, 2560, 2048)
    assert not _use_warp_reduce_smallm(8, 1280, 512)


def test_supports_splitk_covers_smallm_high_k_shapes() -> None:
    assert _supports_mfma_lds_smallm(1, 4096, 4096)
    assert _supports_mfma_lds_smallm(2, 4096, 4096)
    assert _supports_mfma_lds_smallm(1, 1280, 2880)
    assert _supports_mfma_lds_smallm(4, 1280, 1024)
    assert _supports_mfma_lds_smallm(4, 2560, 2048)
    assert _supports_mfma_lds_smallm(4, 8192, 8192)


def test_supports_splitk_rejects_non_target_shapes() -> None:
    assert not _supports_mfma_lds_smallm(4, 3968, 4096)
    assert not _supports_mfma_lds_smallm(4, 1280, 960)
    assert not _supports_mfma_lds_smallm(4, 1280, 1216)
    assert not _supports_mfma_lds_smallm(4, 4224, 4096)
    assert not _supports_mfma_lds_smallm(3, 4096, 4096)
    assert not _supports_mfma_lds_smallm(8, 8192, 4096)


def test_use_splitk_is_disabled_for_default_routing() -> None:
    assert not _use_mfma_lds_smallm(1, 4096, 4096)
    assert not _use_mfma_lds_smallm(4, 2560, 2048)
    assert not _use_mfma_lds_smallm(1, 2560, 2048)
    assert not _use_mfma_lds_smallm(2, 2560, 2048)
    assert not _use_mfma_lds_smallm(4, 1280, 1024)


def test_dispatcher_falls_back_for_splitk_shapes() -> None:
    dtype = torch.bfloat16
    a = torch.empty((1, 4096), device="cuda", dtype=dtype)
    b = torch.empty((4096, 4096), device="cuda", dtype=dtype)

    assert _supports_mfma_lds_smallm(1, 4096, 4096)
    assert gluon_mm_a16w16_gfx950(a, b, dtype) is None


def test_splitk_partial_scratch_is_stream_local() -> None:
    device = torch.device("cuda")
    first = _get_partial_scratch(device, 2, 8, 256, 4)
    second = _get_partial_scratch(device, 2, 8, 256, 4)

    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        other_first = _get_partial_scratch(device, 2, 8, 256, 4)
        other_second = _get_partial_scratch(device, 2, 8, 256, 4)
    other_stream.synchronize()

    assert first.shape == second.shape == other_first.shape == other_second.shape
    assert first.data_ptr() == second.data_ptr()
    assert other_first.data_ptr() == other_second.data_ptr()
    assert first.data_ptr() != other_first.data_ptr()


def test_choose_mfma_lds_mediumm_config_uses_tuned_medium_m_tiles() -> None:
    assert _choose_mfma_lds_mediumm_config(8, 1280, 64) == (16, 32, 64, 2, 2, 1)
    assert _choose_mfma_lds_mediumm_config(8, 1280, 512) == (16, 32, 256, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(16, 1280, 768) == (16, 32, 256, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(8, 1280, 1024) == (16, 32, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(32, 2560, 2048) == (16, 16, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(64, 1280, 1024) == (32, 32, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(64, 1280, 2048) == (32, 32, 512, 2, 2, 2)
    assert _choose_mfma_lds_mediumm_config(64, 2560, 2048) == (32, 32, 128, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(128, 2560, 2048) == (32, 32, 64, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(128, 1280, 2880) == (32, 32, 64, 2, 2, 3)
    assert _choose_mfma_lds_mediumm_config(128, 4096, 4096) == (16, 128, 64, 1, 4, 3)
    assert _choose_mfma_lds_mediumm_config(768, 3584, 7168) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )
    assert _choose_mfma_lds_mediumm_config(1024, 3584, 7168) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )
    assert _choose_mfma_lds_mediumm_config(384, 7168, 3584) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )
    assert _choose_mfma_lds_mediumm_config(512, 7168, 3584) == (
        128,
        128,
        64,
        2,
        4,
        3,
    )


def test_choose_mfma_lds_mediumm_config_falls_back_for_slow_shapes() -> None:
    assert _choose_mfma_lds_mediumm_config(16, 1280, 2880) is None
    assert _choose_mfma_lds_mediumm_config(16, 2560, 2048) is None
    assert _choose_mfma_lds_mediumm_config(32, 1280, 1024) is None
    assert _choose_mfma_lds_mediumm_config(16, 1280, 8192) is None
    assert _choose_mfma_lds_mediumm_config(32, 1280, 4096) is None
    assert _choose_mfma_lds_mediumm_config(128, 4096, 64) is None
    assert _choose_mfma_lds_mediumm_config(128, 4096, 512) is None
    assert _choose_mfma_lds_mediumm_config(64, 4096, 2048) is None
    assert _choose_mfma_lds_mediumm_config(256, 1280, 1024) is None
    assert _choose_mfma_lds_mediumm_config(512, 4096, 4096) is None
    assert _choose_mfma_lds_mediumm_config(1024, 8192, 8192) is None
    assert _choose_mfma_lds_mediumm_config(640, 3584, 7168) is None
    assert _choose_mfma_lds_mediumm_config(1152, 3584, 7168) is None
    assert _choose_mfma_lds_mediumm_config(320, 7168, 3584) is None
    assert _choose_mfma_lds_mediumm_config(576, 7168, 3584) is None


def test_use_mediumm_routes_configured_shapes() -> None:
    assert _use_mfma_lds_mediumm(8, 1280, 1024)
    assert _use_mfma_lds_mediumm(64, 1280, 2880)
    assert _use_mfma_lds_mediumm(128, 4096, 4096)
    assert _use_mfma_lds_mediumm(768, 3584, 7168)
    assert _use_mfma_lds_mediumm(512, 7168, 3584)
    assert not _use_mfma_lds_mediumm(4, 1280, 1024)
    assert not _use_mfma_lds_mediumm(256, 1280, 1024)
    assert not _use_mfma_lds_mediumm(640, 3584, 7168)


def test_supports_largem_shape_covers_aligned_prefill_tiles() -> None:
    assert _supports_largem_shape(256, 256, 256)
    assert _supports_largem_shape(2048, 8192, 8192)


def test_supports_largem_shape_rejects_unaligned_or_medium_shapes() -> None:
    assert not _supports_largem_shape(128, 4096, 4096)
    assert not _supports_largem_shape(256, 128, 256)
    assert not _supports_largem_shape(256, 256, 128)
    assert not _supports_largem_shape(256, 1280, 2880)
    assert not _supports_largem_shape(384, 4096, 4096)
    assert not _supports_largem_shape(512, 3968, 4096)


def test_use_largem_routes_only_dispatch_target_shapes() -> None:
    assert _use_mfma_lds_largem(2048, 4096, 4096)
    assert not _use_mfma_lds_largem(1024, 8192, 8192)
    assert not _use_mfma_lds_largem(2048, 1280, 2880)
