from __future__ import annotations

import pytest

from tokenspeed_kernel._triton import gl
from tokenspeed_kernel._triton import _import_triton_module
from tokenspeed_kernel.ops.layernorm.gluon import _rmsnorm_block_full_aiter_kernel


GPUTarget = _import_triton_module("backends.compiler").GPUTarget
MockTensor = _import_triton_module("runtime.jit").MockTensor
run_parser = _import_triton_module("_filecheck").run_parser


def test_block_full_aiter_uses_descriptor_bounds() -> None:
    if not hasattr(gl.amd.cdna4, "make_buffer_descriptor"):
        pytest.skip("custom tokenspeed_triton descriptor build is not installed")

    layout = gl.BlockedLayout([16], [64], [4], [0])
    dtype = gl.bfloat16
    mod = run_parser(
        _rmsnorm_block_full_aiter_kernel,
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
            16,
            16,
            16,
            layout,
        ),
        {},
        target=GPUTarget("hip", "gfx950", 64),
    )
    ir = mod.str_nodebug()

    assert ir.count("validBytes =") == 3
    assert "amdg.buffer_load" in ir
    assert "tt.addptr" in ir
