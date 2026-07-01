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

## Validation Contract

The generator rejects invalid quantization inputs before values are returned:

- shapes must have at least one positive dimension
- input dtype must be a floating torch dtype, but not already FP8
- output dtype must be a supported FP8 dtype
- `token_group` requires a positive `group_size`
- the last dimension must be divisible by `group_size` for token-group scales
- non-float scale encodings are only accepted for token-group semantics

These checks are operation-level constraints. Kernel-specific layout constraints
belong in adapters or tests.

## TokenSpeed API Mapping

TokenSpeed has two relevant FP8 quantization surfaces today:

- legacy numerics modes `fp8_tensor`, `fp8_token`, and `fp8_token_group_128`
  compare the quantized FP8 values cast back to float32
- runtime APIs such as `quantize_fp8` and `quantize_fp8_with_scale` return FP8
  tensors and, for dynamic modes, backend scale tensors

`FP8QuantizationInputs` can feed both. For the legacy numerics modes, adapters
only need to pass `values.x` and choose the generator granularity that matches
the mode. For runtime API tests, `values.scale` can be used as the semantic
reference scale while backend-specific scale tensors are checked separately for
shape, dtype, or encoded-layout expectations.
