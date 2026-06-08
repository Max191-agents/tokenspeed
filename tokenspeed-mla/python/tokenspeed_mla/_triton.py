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

"""Triton vendor-package indirection used by the standalone MLA package."""

import importlib
import os

_TRITON_DEFAULT_PACKAGE = "tokenspeed_triton"
_TRITON_PACKAGE_ENV = "TOKENSPEED_TRITON_PACKAGE"
_TRITON_DST = os.environ.get(_TRITON_PACKAGE_ENV, _TRITON_DEFAULT_PACKAGE).strip()
if not _TRITON_DST:
    _TRITON_DST = _TRITON_DEFAULT_PACKAGE


def _import_triton_module(suffix: str = ""):
    module_name = _TRITON_DST if not suffix else f"{_TRITON_DST}.{suffix}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Unable to import Triton package {module_name!r}. "
            f"Set {_TRITON_PACKAGE_ENV} to the import package that provides the "
            "Triton API, or install the default tokenspeed_triton package."
        ) from exc


triton = _import_triton_module()
tl = _import_triton_module("language")

try:
    TensorDescriptor = _import_triton_module("tools.tensor_descriptor").TensorDescriptor
except ImportError:
    TensorDescriptor = None

__all__ = ["TensorDescriptor", "tl", "triton"]
