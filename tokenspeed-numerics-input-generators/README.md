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

Input generators should perform just enough verification that the library cannot
be misused to silently produce invalid operation inputs. If a generator accepts
a config and returns values, those values should satisfy the operation-level
constraints documented for that generator. Correctness tests should get
well-formed inputs by construction as long as they stay inside the public
generator API.

In other words, the generator should be the trust boundary for input validity.
Consumers should be able to use generated values with confidence that the
operation's generic invariants have already been checked, including the
relationships between nested generators, generated metadata, tensor dtypes,
scale tensors, and cache/index structures.

The verification standard is misuse resistance, not exhaustive defensive
programming. The public generator path should make invalid or broken operation
inputs unrepresentable, or reject them before generation completes. Consumers
may still check kernel-specific ABI details, but they should not need to
rediscover whether an operation-family config is coherent, whether metadata can
describe a valid computation, or whether generated tensors satisfy the generic
shape and dtype relationships for that operation.

This confidence is part of the library contract. It should be difficult to call
a generator in a way that produces broken tensors, inconsistent metadata,
unsupported dtype combinations, or operation inputs with no valid mathematical
interpretation. Tests that intentionally need invalid inputs should construct
those cases outside the normal generator path so the generator contract stays
clear.

Concretely, each generator should guarantee that:

- Required operation-defining fields are present before generation starts.
- Shape relationships are internally consistent, including relationships
  between parent configs and nested child generators.
- Dtype and custom-format requirements are enforced where the corresponding
  tensor is generated.
- Randomly generated metadata cannot describe impossible layouts, out-of-range
  indices, invalid request lengths, or cache/page state that does not match the
  generated tensors.
- Returned values are ready for reference implementations or kernel adapters
  without requiring each consumer to repeat generic validity checks.

Verification belongs at the same semantic level as generation. It should check
shape relationships, datatype compatibility, required metadata, cache/page
constraints, scale requirements, and other invariants that define whether the
generated inputs are meaningful. Examples include rejecting an MXFP4 tensor
without scales, rejecting incompatible scale shapes, ensuring page-table
metadata is consistent with cache layout, and requiring attention head counts or
request-length metadata to satisfy the operation contract.

The practical rule is that misuse should fail inside the generator library
rather than later inside a reference implementation, kernel adapter, or backend
kernel. If a public config can produce inputs that do not satisfy the documented
operation semantics, the generator contract is incomplete and should be fixed at
the source.

Verification is part of the public API for each generator, not an optional
debug aid. A generator should reject unsupported or contradictory configurations
instead of guessing what the caller intended. It should also validate generated
relationships that depend on inferred defaults, child generators, random
metadata, or target-device choices. This makes the generator the trusted place
where operation-specific constraints are encoded.

The useful verification boundary is the point where misuse would create broken
or meaningless inputs:

- Config objects should reject impossible or contradictory operation
  descriptions.
- `generate(...)` should verify relationships that are only known after child
  configs, inferred defaults, devices, dtypes, or generated metadata are
  resolved.
- Values returned by a generator should not need additional generic validity
  checks before being passed to a reference implementation or kernel adapter.

The goal is not to duplicate every assertion a kernel might make about its
private ABI. Kernel-specific requirements still belong in adapters or tests.
The generator should instead prevent misuse of the operation-family API itself:
bad configs should fail early, and generated values should not require every
consumer to rediscover the same validity checks.

This keeps numerical tests focused on implementation correctness. If a test uses
the generator library, failures should not be caused by malformed inputs unless
the test deliberately mutates the generated values outside the generator
contract.

Tests for generators should cover that contract directly. Each family should
include focused tests for invalid configurations that must be rejected and for
valid generated values satisfying the documented invariants. Consumer tests can
then rely on the library to provide well-formed inputs and spend their checks on
reference agreement or implementation-specific adaptation.

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
components and utilities factored out when they are genuinely reusable:

```text
input_generators/
  README.md                 # overall philosophy and navigation
  core/                     # dtype, tensor, scaled tensor, base interfaces
  utils/                    # shared helpers used by multiple families
  gemm/                     # GEMM-family generators and README
  attention/                # MHA/MLA/cache/metadata generators and README
  moe/                      # MoE-family generators and README
```

The exact names and module boundaries can evolve, but the conceptual split
should remain: core primitives, optional shared utilities, and documented
operation-family generators.

The current family documentation lives under
[`docs/families`](docs/families/README.md). The implementation modules still use
a flat Python import surface for compatibility, while the docs provide the
family navigation structure expected by this design.

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
4. Keep implementation-specific kwargs and registry details in adapters.
5. Add tests for both generated invariants and at least one realistic consumer.
