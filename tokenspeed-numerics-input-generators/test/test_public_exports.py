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

import importlib

import pytest
import tokenspeed_numerics_input_generators as public

_PUBLIC_MODULES = (
    "attention",
    "attention.cache",
    "attention.metadata",
    "core",
    "embedding",
    "gemm",
    "layernorm",
    "moe",
    "quantization",
    "rotary",
    "sampling",
)

_REMOVED_MODULES = (
    "activation",
    "attention_cache",
    "attention_metadata",
    "communication",
    "kvcache",
    "transforms",
)


def test_package_root_reexports_public_family_names() -> None:
    root_exports = set(public.__all__)
    missing: dict[str, list[str]] = {}

    for module_name in _PUBLIC_MODULES:
        module = importlib.import_module(
            f"tokenspeed_numerics_input_generators.{module_name}"
        )
        module_exports = set(getattr(module, "__all__", ()))
        module_missing = sorted(module_exports - root_exports)
        if module_missing:
            missing[module_name] = module_missing

    assert missing == {}


def test_attention_package_reexports_support_module_names() -> None:
    attention = importlib.import_module(
        "tokenspeed_numerics_input_generators.attention"
    )
    attention_exports = set(getattr(attention, "__all__", ()))
    missing: dict[str, list[str]] = {}

    for module_name in ("cache", "metadata"):
        module = importlib.import_module(
            f"tokenspeed_numerics_input_generators.attention.{module_name}"
        )
        module_exports = set(getattr(module, "__all__", ()))
        module_missing = sorted(module_exports - attention_exports)
        if module_missing:
            missing[module_name] = module_missing

    assert missing == {}


def test_package_all_names_are_bound() -> None:
    missing = sorted(name for name in public.__all__ if not hasattr(public, name))

    assert missing == []


def test_embedding_family_does_not_export_removed_specializations() -> None:
    embedding = importlib.import_module(
        "tokenspeed_numerics_input_generators.embedding"
    )
    removed_names = {
        "MLARopeQuantizeFP8InputConfig",
        "MLARopeQuantizeFP8Inputs",
        "MLARopeQuantizeFP8InputValues",
        "MLARopeQuantizeFP8ReferenceValues",
        "mla_rope_quantize_fp8_reference",
        "rope_reference",
    }

    assert removed_names.isdisjoint(embedding.__all__)
    assert removed_names.isdisjoint(public.__all__)
    for name in removed_names:
        assert not hasattr(embedding, name)
        assert not hasattr(public, name)


def test_removed_family_modules_are_not_importable() -> None:
    for module_name in _REMOVED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(
                f"tokenspeed_numerics_input_generators.{module_name}"
            )
