# Core Generator Components

Core components define the shared API and dtype semantics used by operation
families. They are not an operation family themselves, but they hold the common
building blocks that prevent every family from reimplementing tensor generation
and custom dtype handling.

## Components

- `NumericsInputGenerator`: base protocol for objects with a `generate` method.
- `TensorInput`: generates ordinary floating tensors and optional scale
  sidecars.
- `TensorValues`: returns generated tensor storage and optional scales.
- `CustomDType`: names non-Torch semantic storage formats such as MXFP4,
  MXINT4, and UE8M0 scale bytes.

## Semantics

`TensorInput` owns generic tensor validity rules: floating torch dtypes use
normal distributions, scaled tensors require compatible scale shape and dtype
pairs, and custom quantized dtypes enforce their required sidecars. Operation
families compose these primitives and add the operation-specific shape and
metadata relationships.

MXFP4 values use packed E2M1 bytes plus UE8M0 scale bytes. MXINT4 values use
packed signed INT4 bytes plus BF16 group scales. Both custom dtypes require an
explicit scale shape so callers cannot generate semantically incomplete
quantized storage.

This split keeps datatype-specific verification in one place while preserving
operation-level meaning in family generators.
