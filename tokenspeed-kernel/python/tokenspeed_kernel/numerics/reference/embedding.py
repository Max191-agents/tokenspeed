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

"""Reference embedding kernels."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_numerics_input_generators import rope_reference


@register_kernel(
    "embedding",
    "rope",
    name="torch_embedding_rope",
    solution="reference",
    signatures=format_signatures(
        ("query", "key"), "dense", {torch.float16, torch.bfloat16}
    ),
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
)
def torch_embedding_rope(
    *,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    rotary_dim: int | None = None,
    fused_set_kv_buffer_arg: Any = None,
    output_q_rope: torch.Tensor | None = None,
    output_k_rope: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Reference rotary embedding with the same mutating contract as kernels."""

    del enable_pdl
    q_ref, k_ref = rope_reference(
        query,
        key,
        positions,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        rotary_dim=rotary_dim,
    )
    q_target = output_q_rope if output_q_rope is not None else query
    k_target = output_k_rope if output_k_rope is not None else key
    q_target.copy_(q_ref)
    k_target.copy_(k_ref)

    if fused_set_kv_buffer_arg is not None:
        cache_loc = fused_set_kv_buffer_arg.cache_loc.to(torch.int64)
        fused_set_kv_buffer_arg.k_buffer.index_copy_(0, cache_loc, k_target)
        fused_set_kv_buffer_arg.v_buffer.index_copy_(
            0,
            cache_loc,
            fused_set_kv_buffer_arg.value.reshape_as(k_target),
        )
