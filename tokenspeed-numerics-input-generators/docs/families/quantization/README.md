# Quantization Generators

Quantization generators produce inputs for storage formats where value ranges,
scale shapes, and packed layouts are part of the mathematical representation.
The family covers FP8 casts, dynamic FP8 scaling, MXFP8, MXFP4, NVFP4, and
packed GPTQ-to-Marlin layout repacking.

## Generators

- `FP8QuantizationInputs`: generates floating input tensors and optional tensor,
  token, or token-group scales for FP8 quantization. Tensor-wide scales may be
  fixed by configuration for static scaled casts, or generated dynamically from
  the input values.
- `MXFP8QuantizationInputs`: generates bf16/fp16 values that are valid for
  MXFP8 E4M3 storage with UE8M0 group scales.
- `MXFP4QuantizationInputs`: generates bf16/fp16 values that quantize cleanly to
  packed E2M1 nibbles with UE8M0 group scales.
- `NVFP4QuantizationInputs`: generates bf16/fp16 values, global scale, and
  local FP8 scale relationships for packed NVFP4 storage.
- `GPTQMarlinRepackInputs`: generates int32 packed GPTQ weight words and
  optional act-order permutation metadata for exact Marlin tiled-layout
  repacking.

## Generated Values

Generated low-bit values are often chosen from exactly representable values
times power-of-two scales. That keeps correctness tests focused on kernel
behavior rather than accidental ambiguity from arbitrary real-number
quantization.

References return operation-level quantized storage and scale tensors. Backend
scale swizzles or packed ABI details should be adapter responsibilities unless a
specific generator explicitly models that representation. Marlin repacking is
one such explicit layout generator because the layout transform is the
operation under test.
