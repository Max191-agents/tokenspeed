# Transform Generators

Transform generators cover standalone deterministic tensor transforms. These
operations are not tied to a model layer by themselves, but they often appear
inside attention/indexer code as reusable mathematical building blocks.

## Generators

- `HadamardTransformInputs`: generates a tensor whose last dimension is a
  power-of-two Walsh-Hadamard transform axis, plus the concrete output scale.

## Generated Values

Generated values include the input tensor `x` and the `scale` multiplier passed
to the transform. The generator defaults to orthonormal scaling,
`transform_dim ** -0.5`, so random inputs keep a similar value range before and
after transformation.

The reference applies the recursive Walsh-Hadamard transform directly over the
last dimension and returns the same shape and dtype as the input.

## Verification

Verification enforces that the transformed dimension is a positive power of
two, batch dimensions are non-negative, dtype is a regular floating torch dtype,
and scale is finite. These are operation-level constraints: backend-specific
limits such as CUDA availability, supported dimensions, or extension import
availability belong in adapters or tests.
