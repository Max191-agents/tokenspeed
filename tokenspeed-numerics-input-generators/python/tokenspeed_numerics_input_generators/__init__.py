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

"""Composable input generators for numerical correctness tests.

The package exports the stable public input-generator surface while keeping
core tensor primitives separate from op/layer-family generators.
"""

from __future__ import annotations

from tokenspeed_numerics_input_generators.activation import (
    FusedGateSigmoidMulAddInputConfig,
    FusedGateSigmoidMulAddInputs,
    FusedGateSigmoidMulAddInputValues,
    FusedSwiGLUFP8UE8M0InputConfig,
    FusedSwiGLUFP8UE8M0Inputs,
    FusedSwiGLUFP8UE8M0InputValues,
    GatedActivationInputConfig,
    GatedActivationInputs,
    GatedActivationInputValues,
    SigmoidMulInputConfig,
    SigmoidMulInputs,
    SigmoidMulInputValues,
)
from tokenspeed_numerics_input_generators.attention import (
    MHAInputConfig,
    MHAInputs,
    MHAInputValues,
    MLAInputConfig,
    MLAInputs,
    MLAInputValues,
)
from tokenspeed_numerics_input_generators.attention_cache import (
    AttentionCacheInput,
    KVCacheInput,
    KVCacheInputConfig,
    KVCacheLayout,
    KVCacheValues,
    MLAKVCacheInput,
    MLAKVCacheInputConfig,
    MLAKVCacheValues,
    PageTableInput,
    PageTableInputConfig,
    PageTableValues,
)
from tokenspeed_numerics_input_generators.attention_metadata import (
    MHARequestMetadataInputConfig,
    MHARequestMetadataInput,
    MHARequestMetadataValues,
)
from tokenspeed_numerics_input_generators.core import (
    CustomDType,
    DeviceLike,
    InputDType,
    NumericsInputGenerator,
    TensorInput,
    TensorValues,
)
from tokenspeed_numerics_input_generators.gemm import (
    GemmInputConfig,
    GemmInputValues,
    GemmInputs,
    gemm_scale_shape,
    mxfp4_gemm_input_config,
)
from tokenspeed_numerics_input_generators.moe import (
    MoeInputConfig,
    MoeInputs,
    MoeInputValues,
)
from tokenspeed_numerics_input_generators.quantization import (
    FP8QuantizationInputConfig,
    FP8QuantizationInputs,
    FP8QuantizationInputValues,
    fp8_quantization_reference,
    fp8_scale_shape,
)

__all__ = [
    "AttentionCacheInput",
    "CustomDType",
    "DeviceLike",
    "FusedGateSigmoidMulAddInputConfig",
    "FusedGateSigmoidMulAddInputValues",
    "FusedGateSigmoidMulAddInputs",
    "FusedSwiGLUFP8UE8M0InputConfig",
    "FusedSwiGLUFP8UE8M0InputValues",
    "FusedSwiGLUFP8UE8M0Inputs",
    "FP8QuantizationInputConfig",
    "FP8QuantizationInputValues",
    "FP8QuantizationInputs",
    "GemmInputConfig",
    "GemmInputValues",
    "GemmInputs",
    "GatedActivationInputConfig",
    "GatedActivationInputValues",
    "GatedActivationInputs",
    "InputDType",
    "KVCacheInput",
    "KVCacheInputConfig",
    "KVCacheLayout",
    "KVCacheValues",
    "MHAInputConfig",
    "MHAInputValues",
    "MHAInputs",
    "MLAInputConfig",
    "MLAInputValues",
    "MLAInputs",
    "MLAKVCacheInput",
    "MLAKVCacheInputConfig",
    "MLAKVCacheValues",
    "MHARequestMetadataInputConfig",
    "MHARequestMetadataInput",
    "MHARequestMetadataValues",
    "MoeInputConfig",
    "MoeInputValues",
    "MoeInputs",
    "NumericsInputGenerator",
    "PageTableInput",
    "PageTableInputConfig",
    "PageTableValues",
    "SigmoidMulInputConfig",
    "SigmoidMulInputValues",
    "SigmoidMulInputs",
    "TensorInput",
    "TensorValues",
    "fp8_quantization_reference",
    "fp8_scale_shape",
    "gemm_scale_shape",
    "mxfp4_gemm_input_config",
]
