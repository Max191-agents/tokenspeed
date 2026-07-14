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

"""Attention-family input generators for numerical correctness tests."""

from __future__ import annotations

from ._core import AttentionGeneratedValues, attention_generate
from .helpers import (
    AttentionMergeStateInputValues,
    GDNQKVSplitInputValues,
    PackedQKVComplexRotaryInputValues,
    attention_merge_state_reference,
    gdn_qkv_split_reference,
    packed_qkv_complex_rotary_reference,
)
from .gdn import (
    GDNChunkPrefillInputValues,
    GDNChunkPrefillReferenceValues,
    GDNInputConfig,
    GDNInputValues,
    GDNInputs,
    gdn_chunk_prefill_reference,
)
from .dsa import (
    DSADecodeTopKInputValues,
    DSAInputConfig,
    DSAInputValues,
    DSAInputs,
    DSASparseDecodeKVPackInputValues,
    DSATopKSlotInputValues,
    dsa_decode_topk_reference,
    dsa_full_context_topk_to_global_slots_reference,
    dsa_local_topk_to_global_slots_reference,
    dsa_sparse_decode_kv_pack_reference,
    dsa_sparse_decode_row_bytes,
)
from .csa import (
    CSAHistoryConfig,
    CSAHistoryValues,
    CSAIndexerConfig,
    CSAIndexerQueryValues,
    CSAIndexerValues,
    CSAInputConfig,
    CSAInputValues,
    CSAInputs,
    CSASlidingWindowValues,
)
from .mha import MHAInputConfig, MHAInputValues, MHAInputs, MHAReferenceValues, mha_reference
from .mla import MLAInputConfig, MLAInputValues, MLAInputs, MLAReferenceValues, mla_reference

__all__ = [
    'AttentionGeneratedValues',
    'AttentionMergeStateInputValues',
    'attention_generate',
    'attention_merge_state_reference',
    'CSAInputConfig',
    'CSAInputs',
    'CSAInputValues',
    'CSAHistoryConfig',
    'CSAHistoryValues',
    'CSAIndexerConfig',
    'CSAIndexerQueryValues',
    'CSAIndexerValues',
    'CSASlidingWindowValues',
    'DSAInputConfig',
    'DSAInputValues',
    'DSAInputs',
    'DSADecodeTopKInputValues',
    'dsa_decode_topk_reference',
    'DSASparseDecodeKVPackInputValues',
    'DSATopKSlotInputValues',
    'dsa_sparse_decode_kv_pack_reference',
    'dsa_sparse_decode_row_bytes',
    'dsa_full_context_topk_to_global_slots_reference',
    'dsa_local_topk_to_global_slots_reference',
    'GDNInputs',
    'GDNInputConfig',
    'GDNInputValues',
    'GDNQKVSplitInputValues',
    'GDNChunkPrefillInputValues',
    'GDNChunkPrefillReferenceValues',
    'gdn_chunk_prefill_reference',
    'gdn_qkv_split_reference',
    'MHAInputConfig',
    'MHAInputValues',
    'MHAInputs',
    'MHAReferenceValues',
    'mha_reference',
    'MLAInputConfig',
    'MLAInputValues',
    'MLAInputs',
    'MLAReferenceValues',
    'mla_reference',
    'PackedQKVComplexRotaryInputValues',
    'packed_qkv_complex_rotary_reference',
]
