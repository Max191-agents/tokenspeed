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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterable

import torch
from tokenspeed_kernel.numerics.comparison import (
    ComparisonResult,
    compare_outputs,
    format_comparison,
)
from tokenspeed_kernel.numerics.inputs import (
    get_input_generator,
    get_standard_shapes,
)
from tokenspeed_kernel.numerics.outputs import get_output_extractor
from tokenspeed_kernel.numerics.tolerance import (
    Tolerance,
    ToleranceFn,
    ToleranceOverride,
    get_family_tolerance,
)
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.selection import (
    ref_compatible_with_spec,
    spec_matches_shape_traits,
)
from tokenspeed_kernel.signature import FormatSignature

# isort: split
import tokenspeed_kernel.numerics.embedding  # noqa: F401
import tokenspeed_kernel.numerics.gemm  # noqa: F401
import tokenspeed_kernel.numerics.moe  # noqa: F401
import tokenspeed_kernel.numerics.quantize  # noqa: F401
import tokenspeed_kernel.numerics.sampling  # noqa: F401

__all__ = ["verify_kernel"]


def _as_tolerance_fn(override: ToleranceOverride | None) -> ToleranceFn | None:
    if override is None:
        return None
    if isinstance(override, Tolerance):
        return lambda _dtype, **_kwargs: override
    if callable(override):
        return override
    raise TypeError(
        "tolerance override must be Tolerance, callable, or None; "
        f"got {type(override)!r}"
    )


def _compatible_reference_for_signature(
    registry: KernelRegistry,
    spec: KernelSpec,
    signature: FormatSignature,
) -> KernelSpec | None:
    ref_specs = registry.get_for_operator(
        spec.family,
        spec.mode,
        format_signature=signature,
        solution="reference",
    )
    for ref in ref_specs:
        if ref.name == spec.name:
            continue
        if ref_compatible_with_spec(ref, spec):
            return ref
    return None


def _verification_signature_and_reference(
    registry: KernelRegistry,
    spec: KernelSpec,
    dtype: torch.dtype,
    dtype_role: str | Iterable[str],
) -> tuple[FormatSignature | None, KernelSpec | None]:
    signatures = spec.format_signatures_for_storage_dtype(dtype, dtype_role)
    for signature in signatures:
        ref_spec = _compatible_reference_for_signature(registry, spec, signature)
        if ref_spec is not None:
            return signature, ref_spec
    return (signatures[0], None) if signatures else (None, None)


def _as_output_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Sequence) and all(
        isinstance(item, torch.Tensor) for item in value
    ):
        return tuple(value)
    raise TypeError(
        "compare_outputs currently expects tensor outputs or a sequence of "
        f"tensor outputs; got {type(value)!r}"
    )


def verify_kernel(
    kernel_name: str,
    *,
    shapes: list[dict[str, Any]] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    dtype_role: str | Iterable[str],
    tolerance: ToleranceOverride | None = None,
    verbose: bool = False,
    device: str | None = None,
    seed: int = 42,
) -> list[ComparisonResult]:
    """Verify one registered kernel against a reference kernel."""
    load_builtin_kernels()
    registry = KernelRegistry.get()
    spec = registry.get_by_name(kernel_name)
    if spec is None:
        raise ValueError(f"Kernel {kernel_name!r} is not registered")

    kernel = registry.get_impl(kernel_name)
    if kernel is None:
        raise ValueError(f"Kernel implementation for {kernel_name!r} is missing")

    signature, ref_spec = _verification_signature_and_reference(
        registry, spec, dtype, dtype_role
    )
    if signature is None:
        raise ValueError(
            f"Kernel {kernel_name!r} does not support storage dtype={dtype} "
            f"on dtype filter role(s) {dtype_role}"
        )
    if ref_spec is None:
        raise ValueError(
            "No compatible reference kernel found for "
            f"{spec.family}.{spec.mode} and dtype={dtype}; "
            f"kernel={spec.name} traits={spec.traits}"
        )
    ref_kernel = registry.get_impl(ref_spec.name)
    if ref_kernel is None:
        raise ValueError(f"Reference implementation {ref_spec.name!r} is missing")

    generator = get_input_generator(
        spec.family,
        spec.mode,
        dtype=dtype,
        traits=spec.traits,
        format_signature=signature,
        device=device,
        seed=seed,
    )
    test_shapes = shapes or get_standard_shapes(spec.family, spec.mode)
    tol_fn = _as_tolerance_fn(tolerance) or get_family_tolerance(spec.family)
    output_extractor = get_output_extractor(spec.family, spec.mode)

    results: list[ComparisonResult] = []
    for shape in test_shapes:
        if not spec_matches_shape_traits(spec, shape):
            if verbose:
                print(f"[SKIP] {kernel_name} shape={shape} incompatible with traits")
            continue

        if output_extractor is None:
            inputs = generator.generate(**shape)
            expected = ref_kernel(**inputs)
            actual = kernel(**inputs)
        else:
            reference_inputs = generator.generate(**shape)
            inputs = generator.generate(**shape)
            expected_result = ref_kernel(**reference_inputs)
            actual_result = kernel(**inputs)
            expected = output_extractor(reference_inputs, expected_result)
            actual = output_extractor(inputs, actual_result)

        actual_outputs = _as_output_tuple(actual)
        expected_outputs = _as_output_tuple(expected)
        if len(actual_outputs) != len(expected_outputs):
            raise ValueError(
                "Output count mismatch: "
                f"actual={len(actual_outputs)} expected={len(expected_outputs)}"
            )

        tol = tol_fn(dtype, family=spec.family, mode=spec.mode, inputs=inputs, **shape)
        for output_index, (actual_tensor, expected_tensor) in enumerate(
            zip(actual_outputs, expected_outputs, strict=True)
        ):
            result = compare_outputs(actual_tensor, expected_tensor, tolerance=tol)
            if verbose:
                print(
                    format_comparison(
                        result,
                        f"{kernel_name} shape={shape} output={output_index}",
                    )
                )
            results.append(result)

    return results
