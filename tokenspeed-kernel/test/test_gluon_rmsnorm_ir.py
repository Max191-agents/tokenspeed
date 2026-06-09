from __future__ import annotations

import pytest

from tokenspeed_kernel._triton import gl
from tokenspeed_kernel._triton import _import_triton_module
from tokenspeed_kernel.ops.layernorm.gluon import (
    _rmsnorm_block_full_aiter_aligned_kernel,
    _rmsnorm_block_full_aiter_kernel,
    _rmsnorm_block_full_aiter_pipelined_aligned_kernel,
)


GPUTarget = _import_triton_module("backends.compiler").GPUTarget
MockTensor = _import_triton_module("runtime.jit").MockTensor
run_parser = _import_triton_module("_filecheck").run_parser


def _parse_aiter_kernel(kernel):
    if not hasattr(gl.amd.cdna4, "make_buffer_descriptor"):
        pytest.skip("custom tokenspeed_triton descriptor build is not installed")

    layout = gl.BlockedLayout([8], [64], [4], [0])
    dtype = gl.bfloat16
    return run_parser(
        kernel,
        (
            MockTensor(dtype),
            MockTensor(dtype),
            MockTensor(dtype),
            MockTensor(dtype),
            MockTensor(dtype),
            2880,
            1e-6,
            4096,
            True,
            8,
            8,
            8,
            layout,
        ),
        {},
        target=GPUTarget("hip", "gfx950", 64),
    )


def _parse_pipelined_aiter_kernel(
    rows_per_cta: int = 4,
    num_buffers: int = 2,
    *,
    interleaved_rows: bool = False,
):
    if not hasattr(gl.amd.cdna4, "make_buffer_descriptor"):
        pytest.skip("custom tokenspeed_triton descriptor build is not installed")

    layout = gl.BlockedLayout([8], [64], [4], [0])
    dtype = gl.bfloat16
    return run_parser(
        _rmsnorm_block_full_aiter_pipelined_aligned_kernel,
        (
            MockTensor(dtype),
            MockTensor(dtype),
            MockTensor(dtype),
            MockTensor(dtype),
            MockTensor(dtype),
            2880,
            1e-6,
            4096,
            512,
            rows_per_cta,
            num_buffers,
            min(rows_per_cta, num_buffers),
            interleaved_rows,
            8,
            8,
            8,
            layout,
        ),
        {},
        target=GPUTarget("hip", "gfx950", 64),
    )


def _assert_weight_load_after_rstd(ir: str) -> None:
    lines = ir.splitlines()
    load_indices = [i for i, line in enumerate(lines) if "amdg.buffer_load" in line]
    rstd_index = next(i for i, line in enumerate(lines) if "math.rsqrt" in line)

    assert len(load_indices) == 3
    assert load_indices[0] < rstd_index
    assert load_indices[1] < rstd_index
    assert load_indices[2] > rstd_index


def test_block_full_aiter_general_uses_descriptor_bounds_with_load_masks() -> None:
    mod = _parse_aiter_kernel(_rmsnorm_block_full_aiter_kernel)
    ir = mod.str_nodebug()

    assert ir.count("validBytes =") == 3
    assert "sizePerThread = [8]" in ir
    assert "amdg.buffer_load" in ir
    assert "tt.addptr" in ir
    load_lines = [line for line in ir.splitlines() if "amdg.buffer_load" in line]
    assert len(load_lines) == 3
    for line in load_lines:
        before_valid_bytes = line.split(" validBytes", maxsplit=1)[0]
        assert ", %" in before_valid_bytes
    _assert_weight_load_after_rstd(ir)


def test_block_full_aiter_aligned_uses_descriptor_bounds_without_load_masks() -> None:
    mod = _parse_aiter_kernel(_rmsnorm_block_full_aiter_aligned_kernel)
    ir = mod.str_nodebug()

    assert ir.count("validBytes =") == 3
    assert "sizePerThread = [8]" in ir
    assert "amdg.buffer_load" in ir
    assert "tt.addptr" in ir
    load_lines = [line for line in ir.splitlines() if "amdg.buffer_load" in line]
    assert len(load_lines) == 3
    for line in load_lines:
        before_valid_bytes = line.split(" validBytes", maxsplit=1)[0]
        assert ", %" not in before_valid_bytes
    _assert_weight_load_after_rstd(ir)


@pytest.mark.parametrize("num_buffers", [1, 2, 3, 4])
def test_block_full_aiter_pipelined_uses_async_lds_pipeline(
    num_buffers: int,
) -> None:
    rows_per_cta = 4
    initial_groups = min(rows_per_cta, num_buffers)
    mod = _parse_pipelined_aiter_kernel(
        rows_per_cta=rows_per_cta,
        num_buffers=num_buffers,
    )
    ir = mod.str_nodebug()

    assert "sizePerThread = [8]" in ir
    assert "ttg.local_alloc" in ir
    assert "amdg.buffer_load_to_local" in ir
    assert "ttg.async_commit_group" in ir
    assert "ttg.async_wait" in ir
    assert "ttg.local_load" in ir
    assert "ttg.amdg.syncedViaAsyncWait = true" in ir
    assert f"!ttg.memdesc<{num_buffers}x4096xbf16" in ir
    assert ir.count("amdg.buffer_load_to_local") == rows_per_cta
    assert ir.count("amdg.buffer_store") == rows_per_cta
    assert ir.count("math.rsqrt") == rows_per_cta

    lines = ir.splitlines()
    first_wait = next(i for i, line in enumerate(lines) if "ttg.async_wait" in line)
    first_weight_load = next(
        i
        for i, line in enumerate(lines[:first_wait])
        if "amdg.buffer_load " in line and "buffer_load_to_local" not in line
    )
    prologue_commits = [
        i
        for i, line in enumerate(lines[:first_wait])
        if "ttg.async_commit_group" in line
    ]
    assert len(prologue_commits) == initial_groups
    assert prologue_commits[-1] < first_weight_load < first_wait


def test_block_full_aiter_pipelined_interleaved_strides_rows_by_grid() -> None:
    contiguous_ir = _parse_pipelined_aiter_kernel(
        rows_per_cta=4,
        num_buffers=2,
        interleaved_rows=False,
    ).str_nodebug()
    interleaved_ir = _parse_pipelined_aiter_kernel(
        rows_per_cta=4,
        num_buffers=2,
        interleaved_rows=True,
    ).str_nodebug()

    assert "c512_i32" not in contiguous_ir
    assert "c512_i32" in interleaved_ir
    assert "c1024_i32" in interleaved_ir
    assert "c1536_i32" in interleaved_ir
