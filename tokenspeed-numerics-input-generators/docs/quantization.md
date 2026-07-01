# Quantization Input Generators

Quantization generators cover operations that map floating point tensors into
lower-precision storage formats. The input generator owns the source tensor and
the semantic scale relationships needed to define the quantized values.

## Operation Semantics

### FP8 Quantization

`FP8QuantizationInputs` represents FP8 casting with optional dynamic scales:

```text
scale = max(abs(x_group)) / fp8_max
q = clamp(x / scale, fp8_min, fp8_max).to(fp8_dtype)
```

The scale group depends on `granularity`:

- `none`: no scale, pure FP8 cast.
- `tensor`: one scale for the entire tensor.
- `token`: one scale per row/token, along the last dimension.
- `token_group`: one scale per row/token and contiguous group along the last
  dimension.

The generator returns the source tensor and semantic float32 scales. Encoded
scale layouts, such as UE8M0 or backend-specific packed layouts, are adapter
concerns unless a future generator explicitly models that encoded format.

### MXFP4 Quantization

`MXFP4QuantizationInputs` represents packed MXFP4 quantization:

```text
scale[group] = 2 ** floor(log2(max(abs(x_group))) - 2)
q_group = nearest_e2m1(x_group / scale[group])
packed = pack_two_e2m1_values_per_byte(q_group)
encoded_scale = ue8m0(scale[group])
```

The generator currently models the linear scale layout with `scale_size=32`,
which is the layout used by the TokenSpeed Triton MXFP4 quantization path. To
make correctness checks stable, generated values are sampled from exactly
representable E2M1 values multiplied by power-of-two group scales. This avoids
tests being dominated by incidental rounding tie behavior while still
exercising packed values, signs, zeros, and per-group scale metadata.

### MXFP8 Quantization

`MXFP8QuantizationInputs` represents microscaled FP8 quantization:

```text
scale[group] = 2 ** ceil(log2(max(abs(x_group)) / fp8_max))
q_group = clamp(x_group / scale[group], fp8_min, fp8_max).to(fp8_e4m3)
encoded_scale = ue8m0(scale[group])
```

The semantic scale group contains 32 contiguous values along the last
dimension. Generated values are sampled from FP8-representable values multiplied
by power-of-two group scales so the reference can recover the exact generated
tensor after dequantization.

### NVFP4 Quantization

`NVFP4QuantizationInputs` represents NVFP4 quantization with two-level scaling:

```text
local_scale[group] = (max(abs(x_group)) / (global_scale * 6)).to(fp8_e4m3)
q_group = nearest_e2m1(x_group / (global_scale * local_scale[group]))
packed = pack_two_e2m1_values_per_byte(q_group)
```

Each semantic scale group contains 16 contiguous values along the last
dimension. The generator returns the source tensor, a scalar FP32 global scale,
and linear scale-layout metadata. Backend-specific swizzled FP4 scale layouts
remain adapter concerns.

## Validation Contract

The generator rejects invalid quantization inputs before values are returned:

- shapes must have at least one positive dimension
- input dtype must be a floating torch dtype, but not already FP8
- output dtype must be a supported FP8 dtype
- `token_group` requires a positive `group_size`
- the last dimension must be divisible by `group_size` for token-group scales
- non-float scale encodings are only accepted for token-group semantics
- MXFP4 input dtype must be bf16 or fp16
- MXFP4 currently requires `scale_size=32` and `scale_layout="linear"`
- MXFP4 last dimension must be divisible by the scale size
- MXFP8 input dtype must be bf16 or fp16
- MXFP8 currently requires `scale_size=32` and `scale_layout="linear"`
- MXFP8 last dimension must be divisible by the scale size
- NVFP4 input dtype must be bf16 or fp16
- NVFP4 currently requires `scale_size=16`, `scale_layout="linear"`, and a
  positive scalar global scale
- NVFP4 last dimension must be divisible by the scale size

These checks are operation-level constraints. Kernel-specific layout constraints
belong in adapters or tests.

## TokenSpeed API Mapping

TokenSpeed has these relevant quantization surfaces today:

- legacy numerics modes `fp8_tensor`, `fp8_token`, and `fp8_token_group_128`
  compare the quantized FP8 values cast back to float32
- runtime APIs such as `quantize_fp8` and `quantize_fp8_with_scale` return FP8
  tensors and, for dynamic modes, backend scale tensors
- runtime APIs such as `quantize_mxfp4` return packed uint8 MXFP4 values and
  uint8 UE8M0 scale bytes
- runtime APIs such as `quantize_mxfp8` return FP8 MXFP8 values plus encoded
  scale tensors whose physical scale layout may be backend-specific
- runtime APIs such as `quantize_nvfp4` return packed E2M1x2 values plus FP8
  scale tensors, with optional backend-specific scale swizzling

`FP8QuantizationInputs` and `MXFP4QuantizationInputs` can feed those APIs. For
the legacy FP8 numerics modes, adapters only need to pass `values.x` and choose
the generator granularity that matches the mode. For runtime API tests,
`values.scale` can be used as the semantic FP8 reference scale while
backend-specific scale tensors are checked separately for shape, dtype, or
encoded-layout expectations. MXFP4 tests can compare both packed values and
encoded scales directly for the linear layout. MXFP8 and NVFP4 tests can compare
quantized payload values directly for exactly representable generated inputs and
can treat backend-specific scale layouts as shape/layout adapter checks.
