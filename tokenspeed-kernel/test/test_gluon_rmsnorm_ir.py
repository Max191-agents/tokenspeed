from __future__ import annotations

import pytest

from tokenspeed_kernel._triton import gl
from tokenspeed_kernel._triton import _import_triton_module
from tokenspeed_kernel.ops.layernorm.gluon import (
    _rmsnorm_block_full_aiter_aligned_kernel,
    _rmsnorm_block_full_aiter_kernel,
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
