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

from typing import Any

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance
from tokenspeed_numerics_input_generators import (
    ArgmaxInputConfig,
    ArgmaxInputs,
)


def tolerance(
    dtype: torch.dtype,
    **_: Any,
) -> Tolerance:
    """Argmax returns discrete indices, so any mismatch is correctness failure."""

    return Tolerance(atol=0.0, rtol=0.0)


set_family_tolerance("sampling", tolerance)


class SamplingArgmaxInputGenerator(InputGenerator):
    """Generates logits for row-wise argmax sampling kernels."""

    def generate(
        self,
        M: int,
        N: int,
        *,
        max_pattern: str = "unique",
        nan_pattern: str = "none",
    ) -> dict[str, Any]:
        values = ArgmaxInputs(
            ArgmaxInputConfig(
                num_rows=M,
                vocab_size=N,
                dtype=self.dtype,
                max_pattern=max_pattern,
                nan_pattern=nan_pattern,
                device=self.device,
            )
        ).generate(
            seed=self.seed,
            metadata_seed=self.seed + 1,
            device=self.device,
        )
        return {"logits": values.logits}


set_input_generator("sampling", "argmax", SamplingArgmaxInputGenerator)


_ARGMAX_STANDARD_SHAPES: list[dict[str, Any]] = [
    {"M": 1, "N": 4096},
    {"M": 8, "N": 4096},
    {"M": 8, "N": 4096, "max_pattern": "tied"},
    {"M": 32, "N": 8192},
]

set_standard_shapes("sampling", "argmax", _ARGMAX_STANDARD_SHAPES)
set_benchmark_shapes("sampling", "argmax", _ARGMAX_STANDARD_SHAPES)
