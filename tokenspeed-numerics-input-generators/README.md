# Numerics Input Generator Design Philosophy

Numerics input generators are a standalone library for producing meaningful
test inputs for mathematically defined operations. They exist because numerical
correctness is not only a question of running an implementation and checking a
tolerance. Complex datatypes, quantization schemes, reductions, caches, routing,
and ragged metadata all impose constraints on value ranges, distributions, and
shape relationships. Those constraints belong to the operation and datatype
semantics, not to any one GPU kernel or framework integration.

TokenSpeed kernel tests guide which generators are added first, but the
generators themselves should not be defined in terms of TokenSpeed kernels. A
kernel, registry adapter, benchmark, or reference runner is a consumer of the
generated values. The generator should describe the operation-level inputs in a
way that remains useful even if the implementation being tested changes.

## Operation-First Semantics

Every generator family should start from a documented operation definition. The
definition does not need to be a formal proof, but it should be precise enough
that input generation has a clear meaning:

- What computation or layer family is represented?
- What tensors and metadata are inputs to that computation?
- What does each input mean mathematically?
- Which shape relationships must hold?
- Which datatypes or custom formats are supported?
- Which generated values are intentionally random, and which are structured
  metadata?

For example, a GEMM generator is meaningful because the operation is clearly
defined by operands such as `A`, `B`, and `C` and shape parameters such as
`M`, `N`, and `K`. The same GEMM generator can optionally attach scale tensors
to operands when values and scales together define the represented numerical
values. Attention generators are meaningful only if they define the request
metadata, query tokens, optional new key/value tokens, and optional cache
contents. MoE generators should describe hidden states, routing metadata,
expert weights, and intermediate activation relationships rather than only
matching a particular fused kernel signature.

If the operation cannot be described independently of an implementation API, the
generator is probably too implementation-specific and should live in an adapter
or test helper instead.

## Library Contract

Input generators should have a consistent, explicit API:

```python
config = FamilyInputConfig(
    required_shape_or_semantic_parameters=...,
    optional_generation_controls=...,
)

generator = FamilyInputs(config)
values = generator.generate(seed=metadata_or_value_seed, device="cuda")
```

The exact class names may vary by family, but the concepts should remain stable:

- A config object describes what should be generated.
- A generator object owns reusable child generators and family-specific logic.
- `generate(...)` returns a values object containing generated tensors and
  metadata. Generation should not rely on callers reading mutated generator
  fields to get output values.
- Seeds are explicit. Structured metadata and numerical values should be able to
  use distinct seeds so tests can compare different value draws with identical
  request layouts or page mappings.
- Devices are caller-controlled. A generator may support direct generation on a
  target device, and callers may also choose to generate on host and move values
  later when a test needs that control.

The values object should be as simple as possible. Plain tensors should be
stored directly. Composite generated values, such as scaled tensors with values
and scales or attention caches with page tables, should use small typed value
objects.

## Verification

Input generators should include the minimum verification needed to make invalid
operation inputs unrepresentable through normal generator use. A generator call
should either fail before returning or return values that satisfy the documented
operation-level constraints for that family.

This is a misuse-resistance contract. Once a caller chooses the appropriate
operation-family generator and provides a valid configuration, they should not
need private knowledge of hidden shape, dtype, metadata, cache, or quantization
rules to avoid malformed inputs. The point of using the generator library is to
have confidence that input generation was done correctly because the relevant
operation and datatype constraints were checked in one well-defined place.

Verification in this layer is about input validity, not numerical agreement. It
should reject invalid configuration choices, inconsistent child generator
overrides, impossible metadata layouts, unsupported dtype/scale combinations,
out-of-range indices, cache/page-table mismatches, and optional input
combinations that do not define a valid computation. Numerical validators and
kernel tests can then focus on comparing reference and test outputs instead of
defending against broken generated inputs.

The scope should stay operation-level. Input generators should enforce
invariants that belong to the operation or datatype, but they should not become
complete validators for every backend that consumes the values. Backend-specific
preconditions, launch constraints, ABI details, preferred tile sizes, and
performance-oriented layout choices belong in adapters or tests when they are
tied to one implementation rather than to the operation itself.

Checks should be placed where the invariant is owned:

- Config objects validate static operation constraints, such as required
  dimensions, supported dtype combinations, cache layout choices, and quantized
  scale requirements.
- Generators validate constraints that are only known after defaults, random
  metadata, nested generators, or device selection have been resolved.
- Values objects describe already-valid generated inputs; they should not
  become second configuration objects that consumers must repair or normalize.

The amount of verification should be enough to prevent misuse, but not so broad
that the generator becomes a backend conformance suite. If a public generator
call can produce broken operation inputs for the operation it claims to model,
that is a bug in the generator. Generator tests should cover that contract
directly with both invalid configurations that must be rejected and valid
generated values that satisfy the documented invariants.

## Configuration Ownership

Configuration fields should have one clear owner. Parent generators may expose a
small set of common convenience fields, but they should not duplicate detailed
child configuration state unless there is a strong reason. If a parent needs a
child setting, it should read the child config directly or through a small
accessor.

Duplicated config state creates synchronization problems. Consistency checks are
useful when a parent exposes common overrides and also accepts child config
objects, but copied state should not become the primary way for the parent to
learn what the child generator resolved.

## Input Generation Is Not Kernel Adaptation

Generators should return operation-level inputs. They should not try to match
every detail of a kernel registry entry point.

Adapters can translate generated values into implementation-specific call
signatures:

```python
values = MHAInputs(config).generate(metadata_seed=1, value_seed=2)
kwargs = mha_kernel_kwargs(values)
out = implementation(**kwargs)
```

The generator should know what an attention cache is. It should not need to know
that one backend names an argument `page_table` while another names it
`block_tables`, or that one registry mode expects a flattened tensor while a
different wrapper expects a batch-major view. Those are adapter concerns.

## Value Ranges And Datatypes

Random input values must be chosen with datatype and operation semantics in
mind. A normal distribution centered around zero is a useful default for
ordinary floating point tensors, but some datatypes need custom handling:

- Custom quantized storage may need paired scale tensors.
- Packed formats may have storage shapes that differ from logical shapes.
- Scale formats may require bounded exponent choices.
- Reductions may need ranges that avoid meaningless overflow or underflow.

The goal is not to perfectly quantize arbitrary real-valued tensors for every
format. The goal is to generate valid, useful numerical inputs whose represented
values exercise the operation without accidentally making the comparison
dominated by avoidable range pathologies.

Core dtype and tensor utilities should live in shared components so families do
not reimplement the same quantized storage rules. Family generators should
compose those core generators and add only the operation-specific shape and
semantic relationships.

## Sharing Implementation

Shared implementation is valuable when it makes the code easier to understand
and harder to misuse. It is not valuable when it only hides a function call or
forces unrelated families through an awkward abstraction.

Prefer sharing code that represents a real common operation:

- generating an ordinary or scaled tensor from a dtype-aware tensor generator
- splitting request lengths into consistent ragged metadata
- generating page tables and cache storage
- applying common attention tensor/cache generation once shapes have been chosen

Avoid sharing code that makes each family harder to read. Family generators
should still visibly define the operation-specific choices: which tensors exist,
what their shapes mean, which caches are valid, and how child generators are
composed.

As an example, MHA and MLA attention can share metadata and cache-generation
plumbing, but they should still explicitly choose different tensor shapes:

```python
if cache_layout == "none":
    q_shape = prefill_q_shape(...)
    k_shape = prefill_k_shape(...)
    v_shape = prefill_v_shape(...)
    cache_config = None
else:
    q_shape = cached_q_shape(...)
    k_shape = None
    v_shape = None
    cache_config = compressed_or_regular_cache(...)

generated = attention_generate(
    metadata=metadata,
    q_shape=q_shape,
    k_shape=k_shape,
    v_shape=v_shape,
    cache_config=cache_config,
)
```

The shared helper should perform the common generation work. The family still
owns the meaning of the shapes and cache choice.

## Documentation Expectations

Each high-level operation family should have a README that acts as a navigation
point. It should describe:

- The operation family and its mathematical or compute semantics.
- The generators defined for that family.
- The major configuration concepts and common scenarios.
- The generated values and how consumers are expected to use them.
- Any important datatype, range, layout, or metadata constraints.

Family READMEs should not list every field exhaustively. Detailed field
documentation belongs in the config and values classes. The README should give a
reader enough context to choose the right generator and know where to look next.

The library should be organized around those family boundaries, with shared core
components and utilities factored out when they are genuinely reusable. The
current package keeps stable package-root exports for public names and a
separate family-doc navigation tree:

```text
tokenspeed_numerics_input_generators/
  core.py                   # dtype, tensor, base interfaces
  gemm.py                   # GEMM-family generators
  attention/                # MHA/MLA/CSA/DSA/GDN generators
    cache.py                # reusable cache/page-table generators
    metadata.py             # request metadata and slot mappings
  moe.py                    # MoE-family generators
  transforms.py             # standalone deterministic tensor transforms

docs/families/
  README.md                 # operation-family navigation
  core/README.md
  gemm/README.md
  attention/README.md
  moe/README.md
  transforms/README.md
```

The exact names and module boundaries can evolve, but the conceptual split
should remain: core primitives, optional shared utilities, and documented
operation-family generators.

The implementation modules keep shared attention support under the attention
package. Public names remain exported from the package root, and attention
support modules are imported as
`tokenspeed_numerics_input_generators.attention.cache` and
`tokenspeed_numerics_input_generators.attention.metadata`.

## Growth Model

The library will grow alongside numerical testing needs in TokenSpeed. That is a
practical prioritization mechanism, not a semantic dependency. When a TokenSpeed
kernel needs a new generator, the work should still define the operation in
implementation-independent terms first, then add any TokenSpeed-specific call
adaptation outside the generator.

This keeps the generator library useful for:

- direct unit tests for generated input invariants
- kernel correctness tests against references
- comparison across multiple implementations of the same operation
- future model-layer tests that need realistic operation inputs

The standard for adding a generator is therefore:

1. Define the operation and inputs clearly.
2. Document the config and generated values.
3. Compose existing core/family generators when that simplifies the code.
4. Add verification for the operation and datatype invariants that prevent
   malformed generated inputs.
5. Keep implementation-specific kwargs and registry details in adapters.
6. Add tests for both generated invariants and at least one realistic consumer.
