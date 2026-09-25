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

"""Import-free checks for the graph-safe gfx950 decode scorer route."""

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[5]
AMD_DSA = (
    ROOT / "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsa"
)
KPOOL = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/kpool"


def _function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_graph_length_bound_stays_runtime_data() -> None:
    kernel = _function(
        AMD_DSA / "decode_kpool_folded.py",
        "_dsa_kpool_decode_logits_fp8_folded_kernel",
    )
    decorator = kernel.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    options = {key.arg: ast.literal_eval(key.value) for key in decorator.keywords}
    assert "max_candidates" in options["do_not_specialize"]
    assert "max_candidates" in options["do_not_specialize_on_alignment"]

    public = _function(KPOOL / "__init__.py", "kpool_decode_topk")
    traits = next(
        node.value
        for node in public.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "traits"
            for target in node.targets
        )
    )
    assert isinstance(traits, ast.Dict)
    trait_values = {
        ast.literal_eval(key): ast.unparse(value)
        for key, value in zip(traits.keys, traits.values, strict=True)
    }
    assert trait_values["q_len"] == "q_len_per_req"
    assert trait_values["batch_size"] == "int(tokens // q_len_per_req)"

    candidate = _function(
        KPOOL / "gluon.py", "gluon_kpool_decode_topk_fp8_query_gfx950"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "seq_lens"
        and node.func.attr in {"item", "tolist", "cpu"}
        for node in ast.walk(candidate)
    )


def test_public_decode_route_covers_only_measured_mtp_shapes() -> None:
    module = ast.parse((KPOOL / "gluon.py").read_text())
    registration = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_kernel"
        and len(node.args) >= 2
        and ast.literal_eval(node.args[1]) == "kpool_decode_topk"
    )
    args = {key.arg: key.value for key in registration.keywords}
    assert ast.literal_eval(args["name"]) == "gluon_kpool_decode_topk_fp8_query_gfx950"
    assert "features" in args and "tags" not in args
    traits = args["traits"]
    assert isinstance(traits, ast.Dict)
    options = {
        ast.literal_eval(key): frozenset(ast.literal_eval(value.args[0]))
        for key, value in zip(traits.keys, traits.values, strict=True)
    }
    assert options["batch_size"] == {2, 4, 8, 16}
    assert options["q_len"] == {4}
    assert options["index_heads"] == {32}
    assert options["index_k_format"] == {"fp8_scaled"}


def test_fused_selection_rejects_incomplete_score_rows() -> None:
    epilogue = _function(AMD_DSA / "sparse_mla.py", "_kpool_fused_slot_epilogue")
    source = ast.unparse(epilogue)
    assert "score_end == num_pools" in source
    assert "num_pools <= logits_cols" in source
    assert "valid &= complete_scores" in source


def test_non_glm_scale_falls_back_before_query_conversion(monkeypatch) -> None:
    if torch.version.hip is None:
        pytest.skip("gfx950 selector requires a ROCm build")

    from tokenspeed_kernel.ops.attention import kpool as public_kpool
    from tokenspeed_kernel.ops.attention.kpool import gluon as kpool_gluon
    from tokenspeed_kernel.platform import ArchVersion, current_platform

    if current_platform().arch_version != ArchVersion(9, 5):
        pytest.skip("gfx950 selector requires gfx950")

    q = torch.empty((8, 32, 128), dtype=torch.bfloat16)
    weights = torch.empty((8, 32), dtype=torch.float32)
    lengths = torch.empty((2,), dtype=torch.int32)
    table = torch.empty((2, 64), dtype=torch.int32)
    cache = torch.empty((64, 16 * (128 + 4)), dtype=torch.uint8)
    sentinel = (object(), object())
    calls = []
    selected_names = []

    def portable(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    real_select = public_kpool.select_kernel

    def observe_selection(*args, **kwargs):
        selected = real_select(*args, **kwargs)
        selected_names.append(selected.name)
        return selected

    monkeypatch.setattr(kpool_gluon, "triton_dense_kpool_decode_topk", portable)
    monkeypatch.setattr(public_kpool, "select_kernel", observe_selection)
    monkeypatch.delenv(
        "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_KPOOL_DECODE_TOPK", raising=False
    )
    result = public_kpool.kpool_decode_topk(
        q,
        cache,
        weights,
        lengths,
        table,
        table,
        pool_size=4,
        page_size=16,
        kv_page_size=64,
        topk_pools=512,
        softmax_scale=128**-0.5,
        q_len_per_req=4,
        max_seq_len=4096,
    )

    assert result is sentinel
    assert selected_names == ["gluon_kpool_decode_topk_fp8_query_gfx950"]
    assert len(calls) == 1
    assert calls[0][1]["softmax_scale"] == 128**-0.5
    assert kpool_gluon._uses_native_fp8_decode_scorer(
        num_tokens=8,
        q_len_per_req=4,
        max_seq_len=4096,
        max_num_pools=1024,
        topk_pools=512,
        softmax_scale=1.0 / 64.0,
    )
