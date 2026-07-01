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

`FP8QuantizationInputs` and `MXFP4QuantizationInputs` can feed those APIs. For
the legacy FP8 numerics modes, adapters only need to pass `values.x` and choose
the generator granularity that matches the mode. For runtime API tests,
`values.scale` can be used as the semantic FP8 reference scale while
backend-specific scale tensors are checked separately for shape, dtype, or
encoded-layout expectations. MXFP4 tests can compare both packed values and
encoded scales directly for the linear layout.
