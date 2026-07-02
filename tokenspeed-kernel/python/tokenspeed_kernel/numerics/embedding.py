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
from tokenspeed_kernel.numerics.outputs import set_output_extractor
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance
from tokenspeed_numerics_input_generators import RopeInputConfig, RopeInputs


def tolerance(
    dtype: torch.dtype,
    **_: Any,
) -> Tolerance:
    """RoPE kernels round rotated fp32 intermediates back to fp16/bf16 outputs."""

    return Tolerance(atol=2e-2, rtol=2e-2)


set_family_tolerance("embedding", tolerance)


class EmbeddingRopeInputGenerator(InputGenerator):
    """Generates operation-level inputs for rotary positional embedding."""

    def generate(
        self,
        num_tokens: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        *,
        rotary_dim: int | None = None,
        max_position: int = 1024,
        rope_base: float = 10000.0,
        is_neox: bool = True,
        with_fused_kv: bool = False,
        cache_size: int | None = None,
        with_q_output: bool = False,
        with_k_output: bool = False,
    ) -> dict[str, Any]:
        values = RopeInputs(
            RopeInputConfig(
                num_tokens=num_tokens,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position=max_position,
                rope_base=rope_base,
                dtype=self.dtype,
                is_neox=is_neox,
                with_fused_kv=with_fused_kv,
                cache_size=cache_size,
                with_q_output=with_q_output,
                with_k_output=with_k_output,
                device=self.device,
            )
        ).generate(
            seed=self.seed,
            metadata_seed=self.seed + 1,
            device=self.device,
        )
        return {
            "positions": values.positions,
            "query": values.query,
            "key": values.key,
            "head_size": head_size,
            "cos_sin_cache": values.cos_sin_cache,
            "is_neox": is_neox,
            "rotary_dim": rotary_dim,
            "fused_set_kv_buffer_arg": values.fused_kv,
            "output_q_rope": values.output_q_rope,
            "output_k_rope": values.output_k_rope,
            "enable_pdl": False,
        }


def _rope_outputs(
    inputs: dict[str, Any],
    result: Any,
) -> tuple[torch.Tensor, ...]:
    del result
    query_output = inputs["output_q_rope"]
    key_output = inputs["output_k_rope"]
    outputs: list[torch.Tensor] = [
        query_output if query_output is not None else inputs["query"],
        key_output if key_output is not None else inputs["key"],
    ]
    fused_kv = inputs["fused_set_kv_buffer_arg"]
    if fused_kv is not None:
        cache_loc = fused_kv.cache_loc.to(torch.int64)
        outputs.append(fused_kv.k_buffer.index_select(0, cache_loc))
        outputs.append(fused_kv.v_buffer.index_select(0, cache_loc))
    return tuple(outputs)


set_input_generator("embedding", "rope", EmbeddingRopeInputGenerator)
set_output_extractor("embedding", "rope", _rope_outputs)


_ROPE_STANDARD_SHAPES: list[dict[str, Any]] = [
    {
        "num_tokens": 1,
        "num_q_heads": 8,
        "num_kv_heads": 1,
        "head_size": 64,
        "rotary_dim": 64,
        "is_neox": True,
    },
    {
        "num_tokens": 8,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_size": 64,
        "rotary_dim": 64,
        "is_neox": False,
    },
    {
        "num_tokens": 8,
        "num_q_heads": 4,
        "num_kv_heads": 1,
        "head_size": 128,
        "rotary_dim": 64,
        "is_neox": True,
    },
    {
        "num_tokens": 7,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_size": 64,
        "rotary_dim": 64,
        "is_neox": True,
        "with_fused_kv": True,
        "cache_size": 16,
        "with_q_output": True,
    },
    {
        "num_tokens": 9,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_size": 64,
        "rotary_dim": 32,
        "is_neox": False,
        "with_q_output": True,
        "with_k_output": True,
    },
]

set_standard_shapes("embedding", "rope", _ROPE_STANDARD_SHAPES)
set_benchmark_shapes("embedding", "rope", _ROPE_STANDARD_SHAPES)
