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

"""TokenSpeed DeepSeek V4 helper-kernel reference adapters.

The input generator package names these values by their CSA role. This module
keeps TokenSpeed helper-kernel tests and call sites in terms of the concrete
DeepSeek V4 helper operation names.
"""

from __future__ import annotations

from tokenspeed_numerics_input_generators.attention.csa import (
    CSACompressorStateValues,
    CSAIndexerMXFP4CacheGatherValues,
    CSAIndexerMXFP4CacheInsertValues,
    CSAIndexerMXFP4CacheWriteValues,
    CSAIndexerQRoPEHadamardMXFP4Values,
    CSAInvRoPEFP8QuantValues,
    CSAKCacheGatherValues,
    CSAPagedIndexValues,
    CSASparseCompressCacheInsertValues,
    CSASparsePrefillIndexValues,
    csa_build_dense_prefill_local_compressed_indices_reference,
    csa_combine_dense_swa_indices_reference,
    csa_combine_topk_swa_indices_reference,
    csa_compressed_slot_mapping_reference,
    csa_compute_global_topk_indices_and_lens_reference,
    csa_decode_swa_indices_and_lens_reference,
    csa_dequantize_and_gather_k_cache_reference,
    csa_indexer_decode_metadata_reference,
    csa_indexer_mxfp4_cache_gather_reference,
    csa_indexer_mxfp4_cache_insert_reference,
    csa_indexer_mxfp4_cache_write_reference,
    csa_indexer_q_rope_hadamard_mxfp4_reference,
    csa_inv_rope_fp8_quant_reference,
    csa_save_compressor_state_reference,
    csa_sparse_compress_cache_insert_reference,
)

DeepSeekV4CompressorStateInputValues = CSACompressorStateValues
DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues = CSAIndexerMXFP4CacheInsertValues
DeepSeekV4IndexerMXFP4CacheGatherInputValues = CSAIndexerMXFP4CacheGatherValues
DeepSeekV4IndexerMXFP4CacheWriteInputValues = CSAIndexerMXFP4CacheWriteValues
DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues = CSAIndexerQRoPEHadamardMXFP4Values
DeepSeekV4InvRoPEFP8QuantInputValues = CSAInvRoPEFP8QuantValues
DeepSeekV4KCacheGatherInputValues = CSAKCacheGatherValues
DeepSeekV4PagedIndexValues = CSAPagedIndexValues
DeepSeekV4SparseCompressCacheInsertInputValues = CSASparseCompressCacheInsertValues
DeepSeekV4SparsePrefillIndexValues = CSASparsePrefillIndexValues

deepseek_v4_build_dense_prefill_local_compressed_indices_reference = (
    csa_build_dense_prefill_local_compressed_indices_reference
)
deepseek_v4_combine_dense_swa_indices_reference = (
    csa_combine_dense_swa_indices_reference
)
deepseek_v4_combine_topk_swa_indices_reference = csa_combine_topk_swa_indices_reference
deepseek_v4_compressed_slot_mapping_reference = csa_compressed_slot_mapping_reference
deepseek_v4_compute_global_topk_indices_and_lens_reference = (
    csa_compute_global_topk_indices_and_lens_reference
)
deepseek_v4_csa_indexer_mxfp4_cache_insert_reference = (
    csa_indexer_mxfp4_cache_insert_reference
)
deepseek_v4_decode_swa_indices_and_lens_reference = (
    csa_decode_swa_indices_and_lens_reference
)
deepseek_v4_dequantize_and_gather_k_cache_reference = (
    csa_dequantize_and_gather_k_cache_reference
)
deepseek_v4_indexer_decode_metadata_reference = csa_indexer_decode_metadata_reference
deepseek_v4_indexer_mxfp4_cache_gather_reference = (
    csa_indexer_mxfp4_cache_gather_reference
)
deepseek_v4_indexer_mxfp4_cache_write_reference = (
    csa_indexer_mxfp4_cache_write_reference
)
deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference = (
    csa_indexer_q_rope_hadamard_mxfp4_reference
)
deepseek_v4_inv_rope_fp8_quant_reference = csa_inv_rope_fp8_quant_reference
deepseek_v4_save_compressor_state_reference = csa_save_compressor_state_reference
deepseek_v4_sparse_compress_cache_insert_reference = (
    csa_sparse_compress_cache_insert_reference
)

__all__ = [
    "DeepSeekV4CompressorStateInputValues",
    "DeepSeekV4CSAIndexerMXFP4CacheInsertInputValues",
    "DeepSeekV4IndexerMXFP4CacheGatherInputValues",
    "DeepSeekV4IndexerMXFP4CacheWriteInputValues",
    "DeepSeekV4IndexerQRoPEHadamardMXFP4InputValues",
    "DeepSeekV4InvRoPEFP8QuantInputValues",
    "DeepSeekV4KCacheGatherInputValues",
    "DeepSeekV4PagedIndexValues",
    "DeepSeekV4SparseCompressCacheInsertInputValues",
    "DeepSeekV4SparsePrefillIndexValues",
    "deepseek_v4_build_dense_prefill_local_compressed_indices_reference",
    "deepseek_v4_combine_dense_swa_indices_reference",
    "deepseek_v4_combine_topk_swa_indices_reference",
    "deepseek_v4_compressed_slot_mapping_reference",
    "deepseek_v4_compute_global_topk_indices_and_lens_reference",
    "deepseek_v4_csa_indexer_mxfp4_cache_insert_reference",
    "deepseek_v4_decode_swa_indices_and_lens_reference",
    "deepseek_v4_dequantize_and_gather_k_cache_reference",
    "deepseek_v4_indexer_decode_metadata_reference",
    "deepseek_v4_indexer_mxfp4_cache_gather_reference",
    "deepseek_v4_indexer_mxfp4_cache_write_reference",
    "deepseek_v4_indexer_q_rope_hadamard_mxfp4_reference",
    "deepseek_v4_inv_rope_fp8_quant_reference",
    "deepseek_v4_save_compressor_state_reference",
    "deepseek_v4_sparse_compress_cache_insert_reference",
]
