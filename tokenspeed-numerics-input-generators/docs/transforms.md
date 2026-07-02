# Transform Input Generators

Transform generators model deterministic tensor transforms used as reusable
building blocks in TokenSpeed kernels and runtime paths.

## Operation Semantics

### Walsh-Hadamard Transform

`HadamardTransformInputs` represents a Walsh-Hadamard transform over the last
dimension of `x`. All leading dimensions are batch dimensions. The transform
dimension must be a positive power of two because the transform is defined
recursively by splitting each row into equal halves:

```text
H_1(x) = x
H_2n([a, b]) = [H_n(a) + H_n(b), H_n(a) - H_n(b)]
out = scale * H_n(x)
```

The default `scale` is `transform_dim ** -0.5`, which makes the transform
orthonormal. That is the common TokenSpeed usage for attention/indexer
rotations because it preserves the row norm and keeps generated values in a
similar range before and after the transform.

## Validation Contract

The generator rejects invalid transform inputs before values are returned:

- batch dimensions must be non-negative
- `transform_dim` must be a positive power of two
- dtype must be fp16, bf16, fp32, or fp64
- scale must be finite after defaults are resolved

Tensor values are generated from `seed`. There is no separate metadata stream
for this generator because the operation has no structured index metadata.

## TokenSpeed API Mapping

TokenSpeed imports `hadamard_transform` from
`tokenspeed_kernel.thirdparty.fast_hadamard_transform`. Tests or adapters can
pass `HadamardTransformInputValues.x` and `.scale` directly to that wrapper:

```python
values = HadamardTransformInputs(config).generate(seed=1, device="cuda")
out = hadamard_transform(values.x, scale=values.scale)
```

The generator is independent of that extension. If the extension is not
available, package tests can still validate operation semantics with
`hadamard_transform_reference`, while TokenSpeed compatibility tests can skip
only the backend-specific execution.
