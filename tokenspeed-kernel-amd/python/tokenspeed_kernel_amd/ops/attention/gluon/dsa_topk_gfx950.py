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

"""DSA top-k Gluon kernels for AMD GFX950."""

from __future__ import annotations

import importlib.metadata
import os
from collections import OrderedDict
from functools import cache
from threading import Lock
from weakref import ref

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_gfx950 import (
    _check_packed_fp8_inputs,
    _dsa_decode_logits_fp8_kernel,
    _dsa_prefill_logits_fp8_kernel,
)

try:
    _COMPILED_RUNNER_ABI_SUPPORTED = (
        importlib.metadata.version("tokenspeed-triton") == "3.8.10.post20260709"
    )
except importlib.metadata.PackageNotFoundError:
    _COMPILED_RUNNER_ABI_SUPPORTED = False

_RADIX_TOPK_MIN_COLS = 65536
_RADIX_TOPK_BLOCK_N = 4096
_RADIX_TOPK_TARGET_GROUPS_PER_CU = 4
_PREFILL_RADIX_BITS = 11
_PREFILL_RADIX_BUCKETS = 1 << _PREFILL_RADIX_BITS
_PREFILL_RADIX_SCHEDULE = ((21, 11), (10, 11), (0, 10))
_PREFILL_RADIX_BLOCK_N = 4096
_PREFILL_RADIX_HIST_TARGET_GROUPS_PER_CU = 2
_PREFILL_RADIX_SCATTER_TARGET_GROUPS_PER_CU = 2
_PREFILL_LOCAL_GROUP_PREFIX_MIN_COLS = 262144
_ONEBLOCK_RADIX_SCHEDULE = (12, 12, 8)
_ONEBLOCK_RADIX_BUCKETS = 1 << max(_ONEBLOCK_RADIX_SCHEDULE)
_ONEBLOCK_DECODE_RADIX_BLOCK_N = 8192
_ONEBLOCK_DECODE_SHORT_LOAD_ELEMS = 4
_ONEBLOCK_DECODE_LONG_LOAD_ELEMS = 8
_ONEBLOCK_PREFILL_RADIX_BLOCK_N = 4096
_ONEBLOCK_COMPACT_FINAL_BLOCK_N = 4096
_ONEBLOCK_COMPACT_FINAL_MIN_COLS = 65536
_ONEBLOCK_DECODE_EARLY_STOP_MIN_COLS = 65536
_ONEBLOCK_DECODE_RUNTIME_MAX_COLS = 256 * 1024
_ONEBLOCK_RADIX_MAX_COLS = 90000
_PREFILL_RUNTIME_RADIX_MIN_COLS = 98304
_PREFILL_RUNTIME_RADIX_MAX_COLS = 196608
_PREFILL_HIST_DERIVED_MIN_COLS = 524288
_PERSISTENT_PREFILL_MIN_COLS = 128 * 1024
_PERSISTENT_PREFILL_FOUR_GROUP_MIN_COLS = 256 * 1024
_PERSISTENT_PREFILL_MAX_COLS = 1024 * 1024
_PERSISTENT_PREFILL_MIN_ROWS = 32
_PERSISTENT_DECODE_MIN_COLS = 90000
_PERSISTENT_DECODE_MAX_COLS = 256 * 1024
_PERSISTENT_DECODE_MAX_GROUPS = 8
_PERSISTENT_PREFILL_TOPK = 2048
_PERSISTENT_PREFILL_BLOCK_N = 16384
_PERSISTENT_PREFILL_NUM_WARPS = 16
_PERSISTENT_PREFILL_NUM_BUCKETS = gl.constexpr(1 << 11)
_PERSISTENT_PREFILL_NUM_PASSES = gl.constexpr(3)
_PERSISTENT_PREFILL_COUNTER_STRIDE = gl.constexpr(32)
_PERSISTENT_PREFILL_WORKSPACE_CACHE_MAXSIZE = 16

_persistent_topk_workspace_cache: OrderedDict[
    tuple[int, int, int], tuple[torch.Tensor, ...]
] = OrderedDict()
_persistent_topk_graph_workspace_keys: set[tuple[int, int, int]] = set()
_persistent_topk_workspace_lock = Lock()
_COMPILED_RUNNER_CACHE_SIZE = 64
_COMPILED_RUNNER_CANONICAL_KNOBS = (None, None, True, True, True, False)
_COMPILED_RUNNER_KNOB_ENV_VARS = (
    "TRITON_OVERRIDE_ARCH",
    "TRITON_F32_DEFAULT",
    "TRITON_DEFAULT_FP_FUSION",
    "AMDGCN_USE_BUFFER_OPS",
    "AMDGCN_USE_BUFFER_ATOMICS",
    "AMDGCN_ANALYZE_SMALL_TENSOR_RANGE",
)
_COMPILED_RUNNER_KNOB_ENV_BYTE_KEYS = tuple(
    os.fsencode(name) for name in _COMPILED_RUNNER_KNOB_ENV_VARS
)
_COMPILED_RUNNER_CANONICAL_ENVIRONMENT = (None,) * len(_COMPILED_RUNNER_KNOB_ENV_VARS)
_TRIVIAL_DECODE_POINTER_DTYPES = (torch.int32,) * 4
_RUNTIME_RADIX_POINTER_DTYPES = (torch.float32,) + (torch.int32,) * 5
_MANUAL_RADIX_POINTER_DTYPES = (torch.float32,) + (torch.int32,) * 6
_PERSISTENT_RADIX_POINTER_DTYPES = (torch.float32,) + (torch.int32,) * 10


class _CompiledRunnerPlan:
    __slots__ = (
        "runner",
        "ordinary_runner",
        "driver",
        "device",
        "mutable_state",
        "target_getter",
        "target_key",
        "knob_key",
        "environment_key",
        "pointer_refs",
        "compiled",
        "raw_launch",
        "launch_cooperative_grid",
        "grid_x",
        "grid_y",
        "grid_z",
        "function",
        "packed_metadata",
        "warp_size",
        "arg_annotations",
        "kernel_signature",
        "enter_hook_chain",
        "exit_hook_chain",
    )

    def __init__(
        self,
        runner: object,
        driver: object,
        device: int,
        mutable_state: tuple[object, ...],
        target_getter: object,
        target_key: tuple[object, ...],
        knob_key: tuple[object, ...],
        pointer_args: tuple[object, ...],
        *,
        compiled: object | None = None,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        self.runner = runner
        self.ordinary_runner = runner
        self.driver = driver
        self.device = device
        self.mutable_state = mutable_state
        self.target_getter = target_getter
        self.target_key = target_key
        self.knob_key = knob_key
        self.environment_key = _compiled_runner_environment_key()
        self.pointer_refs = tuple(ref(pointer) for pointer in pointer_args)
        self.compiled = None
        self.raw_launch = None
        self.launch_cooperative_grid = False
        self.grid_x = 0
        self.grid_y = 0
        self.grid_z = 0
        self.function = None
        self.packed_metadata = None
        self.warp_size = 0
        self.arg_annotations = None
        self.kernel_signature = None
        self.enter_hook_chain = None
        self.exit_hook_chain = None

        if compiled is None or grid is None or not _COMPILED_RUNNER_ABI_SUPPORTED:
            return
        launcher = compiled.run
        # The raw launcher bypasses CompiledKernel's lazy launch metadata callback.
        source = getattr(compiled, "src", None)
        jit_kernel = getattr(source, "fn", None)
        runtime = triton.knobs.runtime
        enter_hook_chain = runtime.launch_enter_hook
        exit_hook_chain = runtime.launch_exit_hook
        if (
            jit_kernel is None
            or not hasattr(jit_kernel, "launch_metadata")
            or jit_kernel.launch_metadata is not None
            or launcher.global_scratch_size != 0
            or launcher.profile_scratch_size != 0
            or launcher.launch is not driver.utils.launch
            or not isinstance(getattr(enter_hook_chain, "calls", None), list)
            or not isinstance(getattr(exit_hook_chain, "calls", None), list)
        ):
            return

        self.compiled = compiled
        self.raw_launch = launcher.launch
        self.launch_cooperative_grid = launcher.launch_cooperative_grid
        self.grid_x, self.grid_y, self.grid_z = grid
        self.function = compiled.function
        self.packed_metadata = compiled.packed_metadata
        self.warp_size = launcher.warp_size
        self.arg_annotations = launcher.arg_annotations
        self.kernel_signature = launcher.kernel_signature
        self.enter_hook_chain = enter_hook_chain
        self.exit_hook_chain = exit_hook_chain
        self.runner = self._launch_raw

    def _launch_raw(self, *args: object) -> None:
        runtime = triton.knobs.runtime
        enter_hook_chain = self.enter_hook_chain
        exit_hook_chain = self.exit_hook_chain
        # HookChain mutation is unsynchronized. Sequential add/remove falls back;
        # concurrent mutation is not an atomic launch protocol supported by Triton.
        if (
            runtime.launch_enter_hook is not enter_hook_chain
            or runtime.launch_exit_hook is not exit_hook_chain
            or enter_hook_chain.calls
            or exit_hook_chain.calls
        ):
            self.ordinary_runner(*args)
            return

        stream = self.driver.get_current_stream(self.device)
        self.raw_launch(
            self.launch_cooperative_grid,
            self.grid_x,
            self.grid_y,
            self.grid_z,
            stream,
            self.function,
            None,
            None,
            self.packed_metadata,
            None,
            None,
            None,
            self.warp_size,
            self.arg_annotations,
            self.kernel_signature,
            args,
        )

    def pointers_match(self, args: tuple[object, ...]) -> bool:
        count = len(self.pointer_refs)
        if count == 4:
            return (
                self.pointer_refs[0]() is args[0]
                and self.pointer_refs[1]() is args[1]
                and self.pointer_refs[2]() is args[2]
                and self.pointer_refs[3]() is args[3]
            )
        if count == 6:
            return (
                self.pointer_refs[0]() is args[0]
                and self.pointer_refs[1]() is args[1]
                and self.pointer_refs[2]() is args[2]
                and self.pointer_refs[3]() is args[3]
                and self.pointer_refs[4]() is args[4]
                and self.pointer_refs[5]() is args[5]
            )
        if count == 7:
            return (
                self.pointer_refs[0]() is args[0]
                and self.pointer_refs[1]() is args[1]
                and self.pointer_refs[2]() is args[2]
                and self.pointer_refs[3]() is args[3]
                and self.pointer_refs[4]() is args[4]
                and self.pointer_refs[5]() is args[5]
                and self.pointer_refs[6]() is args[6]
            )
        if count == 11:
            return (
                self.pointer_refs[0]() is args[0]
                and self.pointer_refs[1]() is args[1]
                and self.pointer_refs[2]() is args[2]
                and self.pointer_refs[3]() is args[3]
                and self.pointer_refs[4]() is args[4]
                and self.pointer_refs[5]() is args[5]
                and self.pointer_refs[6]() is args[6]
                and self.pointer_refs[7]() is args[7]
                and self.pointer_refs[8]() is args[8]
                and self.pointer_refs[9]() is args[9]
                and self.pointer_refs[10]() is args[10]
            )
        return all(
            pointer_ref() is pointer
            for pointer_ref, pointer in zip(self.pointer_refs, args, strict=True)
        )

    def update_pointers(self, pointer_args: tuple[object, ...]) -> None:
        self.pointer_refs = tuple(ref(pointer) for pointer in pointer_args)


_compiled_runner_cache: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)
_compiled_runner_cache_lock = Lock()
_compiled_runner_environments: dict[
    tuple[object, int],
    tuple[tuple[object, ...], tuple[object, ...], object],
] = {}
_trivial_decode_runner_plans: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)
_runtime_decode_runner_plans: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)
_manual_decode_runner_plans: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)
_trivial_prefill_runner_plans: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)
_runtime_prefill_runner_plans: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)
_manual_prefill_runner_plans: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)
_persistent_runner_plans: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] = (
    OrderedDict()
)

__all__ = [
    "gluon_dsa_decode_topk_fp8_gfx950",
    "gluon_dsa_prefill_topk_fp8_gfx950",
]


@cache
def _compiled_runner_signature_supported(
    kernel: object,
    pointer_count: int,
    native_scalar_count: int,
) -> bool:
    runtime_arg_count = pointer_count + native_scalar_count
    return len(kernel.params) == runtime_arg_count + sum(
        parameter.is_constexpr for parameter in kernel.params
    ) and all(
        (
            not parameter.is_constexpr
            if index < runtime_arg_count
            else parameter.is_constexpr
        )
        for index, parameter in enumerate(kernel.params)
    )


def _compiled_runner_target_key(driver: object) -> tuple[object, ...]:
    target = driver.get_current_target()
    return (target.backend, target.arch, target.warp_size)


def _compiled_runner_knob_key() -> tuple[object, ...]:
    knobs = triton.knobs
    return (
        knobs.runtime.override_arch,
        knobs.language.fp32_default,
        knobs.language.default_fp_fusion,
        knobs.amd.use_buffer_ops,
        knobs.amd.use_buffer_atomics,
        knobs.amd.buffer_ops_analyze_small_tensor_range,
    )


def _compiled_runner_environment_key() -> tuple[object, ...]:
    data = getattr(os.environ, "_data", None)
    if os.name == "posix" and isinstance(data, dict):
        # POSIX _Environ stores exact byte values; fixed lookups avoid six
        # descriptor getenv calls. Other implementations use the public API.
        keys = _COMPILED_RUNNER_KNOB_ENV_BYTE_KEYS
        return (
            data.get(keys[0]),
            data.get(keys[1]),
            data.get(keys[2]),
            data.get(keys[3]),
            data.get(keys[4]),
            data.get(keys[5]),
        )
    return tuple(
        None if (value := os.environ.get(name)) is None else os.fsencode(value)
        for name in _COMPILED_RUNNER_KNOB_ENV_VARS
    )


def _compiled_runner_knob_mutation_guard() -> tuple[object, ...]:
    knobs = triton.knobs
    return (
        tuple(knobs.runtime.__dict__.items()),
        tuple(knobs.language.__dict__.items()),
        tuple(knobs.amd.__dict__.items()),
        tuple(knobs.compilation.__dict__.items()),
        len(os.environ),
    )


def _compiled_runner_mutable_state(kernel: object) -> tuple[object, ...]:
    knobs = triton.knobs
    return (
        kernel.debug,
        bool(kernel.pre_run_hooks),
        bool(kernel.used_global_vals),
        knobs.runtime.debug,
        knobs.runtime.add_stages_inspection_hook,
        knobs.compilation.instrumentation_mode,
        _compiled_runner_knob_mutation_guard(),
    )


def _compiled_runner_state_supported(state: tuple[object, ...]) -> bool:
    return (
        not state[0]
        and not state[1]
        and not state[2]
        and not state[3]
        and state[4] is None
        and not state[5]
    )


def _compiled_runner_plan_state_matches(
    plan: _CompiledRunnerPlan,
    kernel: object,
    driver: object,
) -> bool:
    knobs = triton.knobs
    if (
        driver is not plan.driver
        or kernel.debug
        or kernel.pre_run_hooks
        or kernel.used_global_vals
        or knobs.runtime.debug
        or knobs.runtime.add_stages_inspection_hook is not None
        or knobs.compilation.instrumentation_mode
        or knobs.runtime.__dict__
        or knobs.language.__dict__
        or knobs.amd.__dict__
        or knobs.compilation.__dict__
        or _compiled_runner_environment_key() != plan.environment_key
    ):
        return False
    # Empty knob dictionaries plus the exact six-variable environment key imply
    # the descriptor values under which supported plans are compiled.
    return (
        driver.get_current_device() == plan.device
        and _compiled_runner_target_getter(driver) is plan.target_getter
        # For the gated HIP ABI, the target can otherwise change only with the
        # device or override_arch, both checked above. Misses revalidate it fully.
        and plan.target_key == ("hip", "gfx950", 64)
        and plan.knob_key == _COMPILED_RUNNER_CANONICAL_KNOBS
        and plan.environment_key == _COMPILED_RUNNER_CANONICAL_ENVIRONMENT
    )


def _compiled_runner_target_getter(driver: object) -> object:
    getter = driver.get_current_target
    return getattr(getter, "__func__", getter)


def _compiled_pointer_args_supported(
    args: tuple[object, ...],
    pointer_dtypes: tuple[torch.dtype, ...],
    device: int,
) -> bool:
    pointers = args[: len(pointer_dtypes)]
    if not all(isinstance(pointer, torch.Tensor) for pointer in pointers):
        return False
    if any(
        pointer.dtype != dtype or pointer.get_device() != device
        for pointer, dtype in zip(pointers, pointer_dtypes, strict=True)
    ):
        return False
    alignment_bits = 0
    for pointer in pointers:
        alignment_bits |= pointer.data_ptr()
    return alignment_bits % 16 == 0


def _compiled_runner_route_key(
    kernel: object,
    grid: tuple[int, int, int],
    pointer_dtypes: tuple[torch.dtype, ...],
    specialization_key: tuple[object, ...],
    driver: object,
    device: int,
    native_scalar_count: int,
    num_warps: int,
) -> tuple[object, ...]:
    return (
        kernel,
        driver,
        device,
        pointer_dtypes,
        16,
        specialization_key,
        grid,
        native_scalar_count,
        ("num_warps", num_warps),
    )


def _get_cached_compiled_runner_plan(
    cache: OrderedDict[tuple[object, ...], _CompiledRunnerPlan],
    key: tuple[object, ...],
) -> _CompiledRunnerPlan | None:
    try:
        plan = cache[key]
        cache.move_to_end(key)
    except KeyError:
        return None
    return plan


def _cache_compiled_runner_plan(
    cache: OrderedDict[tuple[object, ...], _CompiledRunnerPlan],
    key: tuple[object, ...],
    plan: _CompiledRunnerPlan,
) -> None:
    cache[key] = plan
    try:
        cache.move_to_end(key)
    except KeyError:
        return
    while len(cache) > _COMPILED_RUNNER_CACHE_SIZE:
        try:
            cache.popitem(last=False)
        except KeyError:
            return


def _launch_warmed_compiled_kernel(
    kernel: object,
    grid: tuple[int, int, int],
    args: tuple[object, ...],
    pointer_dtypes: tuple[torch.dtype, ...],
    specialization_key: tuple[object, ...],
    *,
    dispatch_cache: OrderedDict[tuple[object, ...], _CompiledRunnerPlan] | None = None,
    dispatch_key: tuple[object, ...] | None = None,
    native_scalar_count: int = 0,
    num_warps: int,
) -> None:
    if not _COMPILED_RUNNER_ABI_SUPPORTED:
        kernel[grid](*args, num_warps=num_warps)
        return

    if dispatch_cache is not None and dispatch_key is not None:
        plan = _get_cached_compiled_runner_plan(dispatch_cache, dispatch_key)
        if plan is not None:
            driver = triton.runtime.driver.active
            pointer_args = args[: len(pointer_dtypes)]
            same_pointers = plan.pointers_match(pointer_args)
            pointers_supported = same_pointers
            same_driver = driver is plan.driver
            if same_driver and not pointers_supported:
                pointers_supported = _compiled_pointer_args_supported(
                    args,
                    pointer_dtypes,
                    plan.device,
                )
            if (
                same_driver
                and _compiled_runner_plan_state_matches(plan, kernel, driver)
                and pointers_supported
            ):
                if not same_pointers:
                    plan.update_pointers(pointer_args)
                plan.runner(*args)
                return

            kernel[grid](*args, num_warps=num_warps)
            return

    driver = triton.runtime.driver.active
    device = driver.get_current_device()
    key = _compiled_runner_route_key(
        kernel,
        grid,
        pointer_dtypes,
        specialization_key,
        driver,
        device,
        native_scalar_count,
        num_warps,
    )
    plan = _get_cached_compiled_runner_plan(_compiled_runner_cache, key)
    mutable_state = _compiled_runner_mutable_state(kernel)
    if plan is not None:
        pointer_args = args[: len(pointer_dtypes)]
        same_pointers = plan.pointers_match(pointer_args)
        pointers_supported = same_pointers
        if not pointers_supported:
            pointers_supported = _compiled_pointer_args_supported(
                args,
                pointer_dtypes,
                device,
            )
        if (
            _compiled_runner_plan_state_matches(plan, kernel, driver)
            and pointers_supported
        ):
            if not same_pointers:
                plan.update_pointers(pointer_args)
            if dispatch_cache is not None and dispatch_key is not None:
                _cache_compiled_runner_plan(dispatch_cache, dispatch_key, plan)
            plan.runner(*args)
            return

        kernel[grid](*args, num_warps=num_warps)
        return

    pointers_supported = _compiled_pointer_args_supported(
        args,
        pointer_dtypes,
        device,
    )
    if (
        not _compiled_runner_state_supported(mutable_state)
        or not pointers_supported
        or torch.cuda.is_current_stream_capturing()
    ):
        kernel[grid](*args, num_warps=num_warps)
        return

    with _compiled_runner_cache_lock:
        plan = _get_cached_compiled_runner_plan(_compiled_runner_cache, key)
        if plan is None:
            target_key = _compiled_runner_target_key(driver)
            target_getter = _compiled_runner_target_getter(driver)
            knob_key = _compiled_runner_knob_key()
            environment_key = (driver, device)
            environment = (target_key, knob_key, target_getter)
            cached_environment = _compiled_runner_environments.get(environment_key)
            knobs = triton.knobs
            can_compile = not (
                target_key != ("hip", "gfx950", 64)
                or knob_key != _COMPILED_RUNNER_CANONICAL_KNOBS
                or knobs.runtime.__dict__
                or knobs.language.__dict__
                or knobs.amd.__dict__
                or knobs.compilation.__dict__
                or any(name in os.environ for name in _COMPILED_RUNNER_KNOB_ENV_VARS)
                or (
                    cached_environment is not None and cached_environment != environment
                )
                or not _compiled_runner_signature_supported(
                    kernel,
                    len(pointer_dtypes),
                    native_scalar_count,
                )
            )
            compiled = (
                kernel.warmup(
                    *args,
                    grid=grid,
                    num_warps=num_warps,
                    enable_fp_fusion=knob_key[2],
                )
                if can_compile
                else None
            )
            environment_unchanged = (
                triton.runtime.driver.active is driver
                and driver.get_current_device() == device
                and _compiled_runner_target_key(driver) == target_key
                and _compiled_runner_target_getter(driver) is target_getter
                and _compiled_runner_knob_key() == knob_key
                and _compiled_runner_mutable_state(kernel) == mutable_state
                and _compiled_runner_environment_key()
                == _COMPILED_RUNNER_CANONICAL_ENVIRONMENT
            )
            if compiled is not None and environment_unchanged:
                plan = _CompiledRunnerPlan(
                    compiled[grid],
                    driver,
                    device,
                    mutable_state,
                    target_getter,
                    target_key,
                    knob_key,
                    args[: len(pointer_dtypes)],
                    compiled=compiled,
                    grid=grid,
                )
                _cache_compiled_runner_plan(_compiled_runner_cache, key, plan)
                _compiled_runner_environments[environment_key] = environment
    if plan is not None:
        pointer_args = args[: len(pointer_dtypes)]
        same_pointers = plan.pointers_match(pointer_args)
        if _compiled_runner_plan_state_matches(plan, kernel, driver) and (
            same_pointers or pointers_supported
        ):
            if not same_pointers:
                plan.update_pointers(pointer_args)
            if dispatch_cache is not None and dispatch_key is not None:
                _cache_compiled_runner_plan(dispatch_cache, dispatch_key, plan)
            plan.runner(*args)
            return

    kernel[grid](*args, num_warps=num_warps)


@gluon.constexpr_function
def _vector_layout(
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    return gl.BlockedLayout([LOAD_ELEMS], [64], [NUM_WARPS], [0])


@gluon.jit
def _fp32_to_ordered_key(x):
    bits = x.to(gl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ gl.where(sign != 0, 0xFFFFFFFF, 0x80000000)


@gluon.jit
def _fp32_to_topk_key(x):
    bits = x.to(gl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ gl.where(sign != 0, 0, 0x7FFFFFFF)


@gluon.jit
def _topk_add(a, b):
    return a + b


@gluon.jit
def _rank_four_items_per_thread(
    values,
    start,
    thread_layout: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    values_2x2 = values.reshape([BLOCK_N // 4, 2, 2])
    even, odd = gl.split(values_2x2)
    value0, value2 = gl.split(even)
    value1, value3 = gl.split(odd)
    value0 = gl.convert_layout(value0, thread_layout)
    value1 = gl.convert_layout(value1, thread_layout)
    value2 = gl.convert_layout(value2, thread_layout)
    value3 = gl.convert_layout(value3, thread_layout)

    thread_count = value0 + value1 + value2 + value3
    thread_inclusive = gl.associative_scan(thread_count, 0, _topk_add)
    position0 = start + thread_inclusive - thread_count
    position1 = position0 + value0
    position2 = position1 + value1
    position3 = position2 + value2

    even_positions = gl.join(position0, position2)
    odd_positions = gl.join(position1, position3)
    positions = gl.join(even_positions, odd_positions).reshape([BLOCK_N])
    return gl.convert_layout(positions, value_layout), gl.sum(thread_count, axis=0).to(
        gl.int32
    )


@gluon.jit
def _persistent_wait_until_at_least(address, target, thread_offset):
    return gl.inline_asm_elementwise(
        asm="""
            v_cmp_eq_u32_e32 vcc, 0, $4
            s_and_saveexec_b64 $0, vcc
            s_cbranch_execz 2f
        1:
            flat_load_dword $1, $2 sc0 sc1
            s_waitcnt vmcnt(0) lgkmcnt(0)
            buffer_inv sc0 sc1
            v_cmp_ge_u32_e32 vcc, $1, $3
            s_cbranch_vccnz 2f
            s_sleep 1
            s_branch 1b
        2:
            s_or_b64 exec, exec, $0
        """,
        constraints="=&s,=&v,v,v,v",
        args=[address, target, thread_offset],
        dtype=(gl.int64, gl.int32),
        is_pure=False,
        pack=1,
    )


@gluon.jit(noinline=True)
def _persistent_histogram_tail(
    row_logits,
    shared_histogram,
    row_start,
    row_end,
    n_cols,
    full_tiles,
    pass_index,
    threshold_shift,
    threshold,
    shift,
    bucket_mask,
    BLOCK_N: gl.constexpr,
    value_layout: gl.constexpr,
):
    offsets = full_tiles * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=value_layout,
    )
    offsets = gl.max_contiguous(
        gl.multiple_of(offsets.to(gl.int32), 4),
        4,
    )
    values = gl.amd.cdna4.buffer_load(
        ptr=row_logits,
        offsets=offsets,
        mask=offsets < n_cols,
        other=-float("inf"),
    )
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < n_cols)
    keys = _fp32_to_topk_key(values)
    if pass_index == 0:
        prefix_match = valid
    else:
        prefix_match = valid & (
            ((keys >> threshold_shift) << threshold_shift) == threshold
        )
    buckets = (keys >> shift) & bucket_mask
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )


@gluon.jit(noinline=True)
def _persistent_compact_histogram_tail(
    row_logits,
    shared_histogram,
    shared_output_counters,
    shared_compact_keys,
    shared_compact_offsets,
    output_counters,
    out,
    row,
    row_start,
    row_end,
    n_cols,
    full_tiles,
    threshold_shift,
    threshold,
    shift,
    bucket_mask,
    out_stride,
    TOPK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    COUNTER_STRIDE: gl.constexpr,
    value_layout: gl.constexpr,
):
    offsets = full_tiles * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=value_layout,
    )
    offsets = gl.max_contiguous(
        gl.multiple_of(offsets.to(gl.int32), 4),
        4,
    )
    values = gl.amd.cdna4.buffer_load(
        ptr=row_logits,
        offsets=offsets,
        mask=offsets < n_cols,
        other=-float("inf"),
    )
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < n_cols)
    keys = _fp32_to_topk_key(values)
    truncated_keys = (keys >> threshold_shift) << threshold_shift
    prefix_match = valid & (truncated_keys == threshold)
    buckets = (keys >> shift) & bucket_mask
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )

    compact_position = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        gl.zeros([BLOCK_N], gl.int32, layout=value_layout),
        axis=0,
        mask=prefix_match,
    )
    shared_compact_keys.atomic_scatter_xchg(
        keys.to(gl.int32, bitcast=True),
        compact_position,
        axis=0,
        mask=prefix_match & (compact_position < TOPK),
    )
    shared_compact_offsets.atomic_scatter_xchg(
        offsets.to(gl.int32),
        compact_position,
        axis=0,
        mask=prefix_match & (compact_position < TOPK),
    )

    definite_winner = valid & (truncated_keys < threshold)
    direct_counter_offsets = gl.zeros(
        [BLOCK_N],
        gl.int32,
        layout=value_layout,
    )
    direct_position = gl.atomic_add(
        output_counters + (row * 2) * COUNTER_STRIDE + direct_counter_offsets,
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        mask=definite_winner,
        sem="acq_rel",
        scope="gpu",
    )
    gl.store(
        out + row * out_stride + direct_position,
        offsets.to(gl.int32),
        mask=definite_winner & (direct_position < TOPK),
    )


@gluon.jit(noinline=True)
def _persistent_emit_tail(
    row_logits,
    shared_output_counters,
    shared_greater_offsets,
    shared_equal_offsets,
    row_start,
    row_end,
    n_cols,
    full_tiles,
    threshold_shift,
    threshold,
    TOPK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    value_layout: gl.constexpr,
):
    offsets = full_tiles * BLOCK_N + gl.arange(
        0,
        BLOCK_N,
        layout=value_layout,
    )
    offsets = gl.max_contiguous(
        gl.multiple_of(offsets.to(gl.int32), 4),
        4,
    )
    values = gl.amd.cdna4.buffer_load(
        ptr=row_logits,
        offsets=offsets,
        mask=offsets < n_cols,
        other=-float("inf"),
    )
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < n_cols)
    keys = _fp32_to_topk_key(values)
    truncated_keys = (keys >> threshold_shift) << threshold_shift
    greater = valid & (truncated_keys < threshold)
    equal = valid & (truncated_keys == threshold)
    reservation_mask = greater | equal
    reservation_counter = gl.where(greater, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    shared_greater_offsets.atomic_scatter_xchg(
        offsets.to(gl.int32),
        reservation,
        axis=0,
        mask=greater & (reservation < TOPK),
    )
    shared_equal_offsets.atomic_scatter_xchg(
        offsets.to(gl.int32),
        reservation,
        axis=0,
        mask=equal & (reservation < TOPK),
    )


@gluon.jit(
    do_not_specialize=(
        "logits_stride",
        "block_table_stride",
        "out_stride",
        "block_table_cols",
        "n_cols",
    ),
)
def _dsa_persistent_radix_topk_kernel(
    logits,
    histograms,
    arrivals,
    pass_done,
    reset_arrivals,
    output_counters,
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride,
    block_table_stride,
    out_stride,
    block_table_cols,
    n_cols,
    page_size: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    GROUPS_PER_ROW: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
    NUM_PASSES: gl.constexpr,
    COUNTER_STRIDE: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    value_layout: gl.constexpr = _vector_layout(
        BLOCK_N,
        gl.num_warps(),
        BLOCK_N // (64 * gl.num_warps()),
    )
    output_layout: gl.constexpr = _vector_layout(
        TOPK,
        gl.num_warps(),
        TOPK // (64 * gl.num_warps()),
    )
    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        row_start = gl.full([], 0, gl.int32)
        row_end = gl.load(row_ends + req).to(gl.int32) - (q_len_per_req - 1) + q_offset
    else:
        req = gl.full([], 0, gl.int32)
        row_start = gl.load(row_starts + row).to(gl.int32)
        row_end = gl.load(row_ends + row).to(gl.int32)
    row_end = gl.maximum(row_end, row_start)
    row_len = gl.maximum(row_end - row_start, 0)
    selected_count = gl.minimum(row_len, TOPK)
    if group == 0:
        gl.store(lens_out + row, selected_count)

    if row_len <= TOPK:
        if group == 0:
            output_offsets = gl.arange(0, TOPK, layout=output_layout)
            valid = output_offsets < row_len
            output_values = row_start + output_offsets
            if IS_DECODE:
                block_idx = output_values // page_size
                physical_page = gl.load(
                    block_table + req * block_table_stride + block_idx,
                    mask=valid & (block_idx < block_table_cols),
                    other=0,
                )
                output_values = physical_page * page_size + output_values % page_size
            gl.store(
                out + row * out_stride + output_offsets,
                gl.where(valid, output_values, -1),
            )
        return

    hist_layout: gl.constexpr = _vector_layout(
        NUM_BUCKETS,
        gl.num_warps(),
        NUM_BUCKETS // (64 * gl.num_warps()),
    )
    group_layout: gl.constexpr = _vector_layout(
        NUM_BUCKETS // 2,
        gl.num_warps(),
        1,
    )
    wait_threads: gl.constexpr = 64 * gl.num_warps()
    wait_layout: gl.constexpr = _vector_layout(
        wait_threads,
        gl.num_warps(),
        1,
    )
    wait_offsets = gl.arange(0, wait_threads, layout=wait_layout)
    hist_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[NUM_BUCKETS, 1]],
        [NUM_BUCKETS],
        [0],
    )
    histogram_zeros = gl.zeros(
        [NUM_BUCKETS],
        gl.int32,
        layout=hist_layout,
    )
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [NUM_BUCKETS],
        hist_shared_layout,
        value=histogram_zeros,
    )
    compact_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[TOPK, 1]],
        [TOPK],
        [0],
    )
    shared_greater_offsets = gl.allocate_shared_memory(
        gl.int32,
        [TOPK],
        compact_shared_layout,
    )
    shared_equal_offsets = gl.allocate_shared_memory(
        gl.int32,
        [TOPK],
        compact_shared_layout,
    )
    output_counter_layout: gl.constexpr = _vector_layout(
        2,
        gl.num_warps(),
        1,
    )
    output_counter_shared_layout: gl.constexpr = (
        gl.PaddedSharedLayout.with_identity_for(
            [[2, 1]],
            [2],
            [0],
        )
    )
    output_counter_zeros = gl.zeros(
        [2],
        gl.int32,
        layout=output_counter_layout,
    )
    shared_output_counters = gl.allocate_shared_memory(
        gl.int32,
        [2],
        output_counter_shared_layout,
        value=output_counter_zeros,
    )
    gl.barrier()

    bucket_offsets = gl.arange(
        0,
        NUM_BUCKETS,
        layout=hist_layout,
    )
    row_logits = logits + row * logits_stride
    full_tiles = n_cols // BLOCK_N
    tail_size = n_cols - full_tiles * BLOCK_N
    tail_owner = full_tiles % GROUPS_PER_ROW
    threshold = gl.full([], 0, gl.uint32)
    threshold_shift = gl.full([], 32, gl.int32)
    remaining = gl.full([], TOPK, gl.int32)
    done = gl.full([], False, gl.int1)
    pass_index = gl.full([], 0, gl.int32)
    if not IS_DECODE:
        candidate_count = row_len
        compact_count = gl.full([], 0, gl.int32)
        compact_ready = gl.full([], False, gl.int1)

    while (pass_index < NUM_PASSES) & ~done:
        if pass_index != 0:
            gl.barrier()
            shared_histogram.store(histogram_zeros)
            gl.barrier()

        shift = gl.maximum(21 - pass_index * 11, 0)
        bucket_mask = gl.where(pass_index == 2, 0x3FF, 0x7FF)
        if IS_DECODE:
            for tile in range(group, full_tiles, GROUPS_PER_ROW):
                offsets = tile * BLOCK_N + gl.arange(
                    0,
                    BLOCK_N,
                    layout=value_layout,
                )
                offsets = gl.max_contiguous(
                    gl.multiple_of(offsets.to(gl.int32), 4),
                    4,
                )
                values = gl.amd.cdna4.buffer_load(
                    ptr=row_logits,
                    offsets=offsets,
                )
                valid = (offsets >= row_start) & (offsets < row_end)
                keys = _fp32_to_topk_key(values)
                if pass_index == 0:
                    prefix_match = valid
                else:
                    prefix_match = valid & (
                        ((keys >> threshold_shift) << threshold_shift) == threshold
                    )
                buckets = (keys >> shift) & bucket_mask
                shared_histogram.atomic_scatter_add(
                    gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
                    buckets.to(gl.int32),
                    axis=0,
                    mask=prefix_match,
                )

            if (tail_size != 0) & (group == tail_owner):
                _persistent_histogram_tail(
                    row_logits,
                    shared_histogram,
                    row_start,
                    row_end,
                    n_cols,
                    full_tiles,
                    pass_index,
                    threshold_shift,
                    threshold,
                    shift,
                    bucket_mask,
                    BLOCK_N,
                    value_layout,
                )
        else:
            # The prior global histogram gives an exact population bound, so no
            # group can overflow once the selected prefix fits in this buffer.
            compact_this_pass = (
                ~compact_ready & (pass_index != 0) & (candidate_count <= TOPK)
            )
            if compact_ready:
                compact_positions = gl.arange(0, TOPK, layout=output_layout)
                compact_keys = shared_greater_offsets.load(output_layout).to(
                    gl.uint32,
                    bitcast=True,
                )
                compact_valid = compact_positions < compact_count
                compact_prefix_match = compact_valid & (
                    ((compact_keys >> threshold_shift) << threshold_shift) == threshold
                )
                compact_buckets = (compact_keys >> shift) & bucket_mask
                shared_histogram.atomic_scatter_add(
                    gl.full([TOPK], 1, gl.int32, layout=output_layout),
                    compact_buckets.to(gl.int32),
                    axis=0,
                    mask=compact_prefix_match,
                )
            else:
                for tile in range(group, full_tiles, GROUPS_PER_ROW):
                    offsets = tile * BLOCK_N + gl.arange(
                        0,
                        BLOCK_N,
                        layout=value_layout,
                    )
                    offsets = gl.max_contiguous(
                        gl.multiple_of(offsets.to(gl.int32), 4),
                        4,
                    )
                    values = gl.amd.cdna4.buffer_load(
                        ptr=row_logits,
                        offsets=offsets,
                    )
                    valid = (offsets >= row_start) & (offsets < row_end)
                    keys = _fp32_to_topk_key(values)
                    if pass_index == 0:
                        prefix_match = valid
                    else:
                        prefix_match = valid & (
                            ((keys >> threshold_shift) << threshold_shift) == threshold
                        )
                    buckets = (keys >> shift) & bucket_mask
                    shared_histogram.atomic_scatter_add(
                        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
                        buckets.to(gl.int32),
                        axis=0,
                        mask=prefix_match,
                    )

                    if compact_this_pass:
                        # Match AITER's three-way pass: direct-write candidates
                        # already known to win and retain only the boundary set.
                        compact_position = shared_output_counters.atomic_scatter_add(
                            gl.full(
                                [BLOCK_N],
                                1,
                                gl.int32,
                                layout=value_layout,
                            ),
                            gl.zeros(
                                [BLOCK_N],
                                gl.int32,
                                layout=value_layout,
                            ),
                            axis=0,
                            mask=prefix_match,
                        )
                        shared_greater_offsets.atomic_scatter_xchg(
                            keys.to(gl.int32, bitcast=True),
                            compact_position,
                            axis=0,
                            mask=prefix_match & (compact_position < TOPK),
                        )
                        shared_equal_offsets.atomic_scatter_xchg(
                            offsets.to(gl.int32),
                            compact_position,
                            axis=0,
                            mask=prefix_match & (compact_position < TOPK),
                        )

                        truncated_keys = (keys >> threshold_shift) << threshold_shift
                        definite_winner = valid & (truncated_keys < threshold)
                        direct_counter_offsets = gl.zeros(
                            [BLOCK_N],
                            gl.int32,
                            layout=value_layout,
                        )
                        direct_position = gl.atomic_add(
                            output_counters
                            + (row * 2) * COUNTER_STRIDE
                            + direct_counter_offsets,
                            gl.full(
                                [BLOCK_N],
                                1,
                                gl.int32,
                                layout=value_layout,
                            ),
                            mask=definite_winner,
                            sem="acq_rel",
                            scope="gpu",
                        )
                        gl.store(
                            out + row * out_stride + direct_position,
                            offsets.to(gl.int32),
                            mask=definite_winner & (direct_position < TOPK),
                        )

                if (tail_size != 0) & (group == tail_owner):
                    if compact_this_pass:
                        _persistent_compact_histogram_tail(
                            row_logits,
                            shared_histogram,
                            shared_output_counters,
                            shared_greater_offsets,
                            shared_equal_offsets,
                            output_counters,
                            out,
                            row,
                            row_start,
                            row_end,
                            n_cols,
                            full_tiles,
                            threshold_shift,
                            threshold,
                            shift,
                            bucket_mask,
                            out_stride,
                            TOPK,
                            BLOCK_N,
                            COUNTER_STRIDE,
                            value_layout,
                        )
                    else:
                        _persistent_histogram_tail(
                            row_logits,
                            shared_histogram,
                            row_start,
                            row_end,
                            n_cols,
                            full_tiles,
                            pass_index,
                            threshold_shift,
                            threshold,
                            shift,
                            bucket_mask,
                            BLOCK_N,
                            value_layout,
                        )

        gl.barrier()
        if not IS_DECODE:
            if compact_this_pass:
                compact_counter_offsets = gl.arange(
                    0,
                    2,
                    layout=output_counter_layout,
                )
                compact_counters = shared_output_counters.load(output_counter_layout)
                compact_count = gl.sum(
                    gl.where(
                        compact_counter_offsets == 0,
                        compact_counters,
                        0,
                    ),
                    axis=0,
                ).to(gl.int32)
                compact_ready = gl.full([], True, gl.int1)
        local_counts = shared_histogram.load(hist_layout)
        row_histogram = histograms + (row * NUM_PASSES + pass_index) * NUM_BUCKETS
        gl.atomic_add(
            row_histogram + bucket_offsets,
            local_counts,
            mask=local_counts != 0,
            sem="relaxed",
            scope="gpu",
        )
        gl.barrier()

        old = gl.atomic_add(
            arrivals + (row * NUM_PASSES + pass_index) * COUNTER_STRIDE,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        if old == GROUPS_PER_ROW - 1:
            gl.atomic_add(
                pass_done + row * COUNTER_STRIDE,
                1,
                sem="release",
                scope="gpu",
            )
        else:
            _persistent_wait_until_at_least(
                pass_done + row * COUNTER_STRIDE,
                pass_index + 1,
                wait_offsets,
            )
        gl.barrier()

        total_counts = gl.load(
            row_histogram + bucket_offsets,
            volatile=True,
        )
        count_pairs = total_counts.reshape([NUM_BUCKETS // 2, 2])
        count_low, count_high = gl.split(count_pairs)
        count_low = gl.convert_layout(count_low, group_layout)
        count_high = gl.convert_layout(count_high, group_layout)
        group_counts = count_low + count_high
        cumulative = gl.associative_scan(group_counts, 0, _topk_add)
        before_group = cumulative - group_counts
        selected_group = (before_group < remaining) & (cumulative >= remaining)
        bucket_pairs = bucket_offsets.reshape([NUM_BUCKETS // 2, 2])
        bucket_low, bucket_high = gl.split(bucket_pairs)
        bucket_low = gl.convert_layout(bucket_low, group_layout)
        bucket_high = gl.convert_layout(bucket_high, group_layout)
        select_low = before_group + count_low >= remaining
        group_bucket = gl.where(select_low, bucket_low, bucket_high)
        group_selected_before = before_group + gl.where(select_low, 0, count_low)
        group_selected_count = gl.where(select_low, count_low, count_high)
        selected_bucket = gl.sum(
            gl.where(selected_group, group_bucket, 0),
            axis=0,
        ).to(gl.int32)
        selected_greater = gl.sum(
            gl.where(selected_group, group_selected_before, 0),
            axis=0,
        ).to(gl.int32)
        selected_bucket_count = gl.sum(
            gl.where(selected_group, group_selected_count, 0),
            axis=0,
        ).to(gl.int32)
        threshold |= selected_bucket.to(gl.uint32) << shift
        threshold_shift = shift
        remaining -= selected_greater
        done = selected_bucket_count == remaining
        if not IS_DECODE:
            candidate_count = selected_bucket_count
        pass_index += 1

    full_emit = True
    if not IS_DECODE:
        full_emit = ~compact_ready
    if full_emit:
        for tile in range(group, full_tiles, GROUPS_PER_ROW):
            offsets = tile * BLOCK_N + gl.arange(
                0,
                BLOCK_N,
                layout=value_layout,
            )
            offsets = gl.max_contiguous(
                gl.multiple_of(offsets.to(gl.int32), 4),
                4,
            )
            values = gl.amd.cdna4.buffer_load(
                ptr=row_logits,
                offsets=offsets,
            )
            valid = (offsets >= row_start) & (offsets < row_end)
            keys = _fp32_to_topk_key(values)
            truncated_keys = (keys >> threshold_shift) << threshold_shift
            greater = valid & (truncated_keys < threshold)
            equal = valid & (truncated_keys == threshold)
            reservation_mask = greater | equal
            reservation_counter = gl.where(greater, 0, 1).to(gl.int32)
            reservation = shared_output_counters.atomic_scatter_add(
                gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
                reservation_counter,
                axis=0,
                mask=reservation_mask,
            )
            shared_greater_offsets.atomic_scatter_xchg(
                offsets.to(gl.int32),
                reservation,
                axis=0,
                mask=greater & (reservation < TOPK),
            )
            shared_equal_offsets.atomic_scatter_xchg(
                offsets.to(gl.int32),
                reservation,
                axis=0,
                mask=equal & (reservation < TOPK),
            )

        if (tail_size != 0) & (group == tail_owner):
            _persistent_emit_tail(
                row_logits,
                shared_output_counters,
                shared_greater_offsets,
                shared_equal_offsets,
                row_start,
                row_end,
                n_cols,
                full_tiles,
                threshold_shift,
                threshold,
                TOPK,
                BLOCK_N,
                value_layout,
            )
    if not IS_DECODE:
        if compact_ready:
            compact_positions = gl.arange(0, TOPK, layout=output_layout)
            compact_keys = shared_greater_offsets.load(output_layout).to(
                gl.uint32,
                bitcast=True,
            )
            compact_offsets = shared_equal_offsets.load(output_layout)
            compact_valid = compact_positions < compact_count
            gl.barrier()
            shared_output_counters.store(output_counter_zeros)
            gl.barrier()

            compact_truncated_keys = (
                compact_keys >> threshold_shift
            ) << threshold_shift
            greater = compact_valid & (compact_truncated_keys < threshold)
            equal = compact_valid & (compact_truncated_keys == threshold)
            reservation_mask = greater | equal
            reservation_counter = gl.where(greater, 0, 1).to(gl.int32)
            reservation = shared_output_counters.atomic_scatter_add(
                gl.full([TOPK], 1, gl.int32, layout=output_layout),
                reservation_counter,
                axis=0,
                mask=reservation_mask,
            )
            shared_greater_offsets.atomic_scatter_xchg(
                compact_offsets,
                reservation,
                axis=0,
                mask=greater & (reservation < TOPK),
            )
            shared_equal_offsets.atomic_scatter_xchg(
                compact_offsets,
                reservation,
                axis=0,
                mask=equal & (reservation < TOPK),
            )

    gl.barrier()
    output_counter_offsets = gl.arange(0, 2, layout=output_counter_layout)
    local_output_counts = shared_output_counters.load(output_counter_layout)
    local_greater = gl.sum(
        gl.where(output_counter_offsets == 0, local_output_counts, 0),
        axis=0,
    ).to(gl.int32)
    local_equal = gl.sum(
        gl.where(output_counter_offsets == 1, local_output_counts, 0),
        axis=0,
    ).to(gl.int32)
    greater_start = gl.atomic_add(
        output_counters + (row * 2) * COUNTER_STRIDE,
        local_greater,
        sem="acq_rel",
        scope="gpu",
    )
    equal_start = gl.atomic_add(
        output_counters + (row * 2 + 1) * COUNTER_STRIDE,
        local_equal,
        sem="acq_rel",
        scope="gpu",
    )
    copy_offsets = gl.arange(0, TOPK, layout=output_layout)
    greater_values = shared_greater_offsets.load(output_layout)
    equal_values = shared_equal_offsets.load(output_layout)
    greater_positions = greater_start + copy_offsets
    equal_positions = equal_start + copy_offsets
    greater_write = (copy_offsets < local_greater) & (greater_positions < TOPK)
    equal_write = (copy_offsets < local_equal) & (equal_positions < remaining)
    if IS_DECODE:
        greater_block_idx = greater_values // page_size
        greater_page = gl.load(
            block_table + req * block_table_stride + greater_block_idx,
            mask=(greater_block_idx >= 0)
            & (greater_block_idx < block_table_cols)
            & greater_write,
            other=0,
        )
        greater_values = greater_page * page_size + greater_values % page_size
        equal_block_idx = equal_values // page_size
        equal_page = gl.load(
            block_table + req * block_table_stride + equal_block_idx,
            mask=(equal_block_idx >= 0)
            & (equal_block_idx < block_table_cols)
            & equal_write,
            other=0,
        )
        equal_values = equal_page * page_size + equal_values % page_size
    gl.store(
        out + row * out_stride + greater_positions,
        greater_values,
        mask=greater_write,
    )
    gl.store(
        out + row * out_stride + (TOPK - 1 - equal_positions),
        equal_values,
        mask=equal_write,
    )

    gl.barrier()
    reset_old = gl.atomic_add(
        reset_arrivals + row * COUNTER_STRIDE,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if reset_old == GROUPS_PER_ROW - 1:
        for reset_pass in gl.static_range(NUM_PASSES):
            gl.store(
                histograms
                + (row * NUM_PASSES + reset_pass) * NUM_BUCKETS
                + bucket_offsets,
                histogram_zeros,
            )
            gl.store(
                arrivals + (row * NUM_PASSES + reset_pass) * COUNTER_STRIDE,
                0,
            )
        gl.store(
            pass_done + row * COUNTER_STRIDE,
            0,
        )
        gl.store(
            output_counters + (row * 2) * COUNTER_STRIDE,
            0,
        )
        gl.store(
            output_counters + (row * 2 + 1) * COUNTER_STRIDE,
            0,
        )
        gl.store(
            reset_arrivals + row * COUNTER_STRIDE,
            0,
        )


@gluon.jit
def _find_topk_threshold_key(
    values,
    valid,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    layout: gl.constexpr,
):
    keys = _fp32_to_ordered_key(values)
    prefix = 0
    remaining = topk

    # Ordered FP32 keys are searched from the most-significant 4-bit nibble down.
    for shift in gl.static_range(28, -1, -4):
        if shift == 28:
            prefix_match = valid
        else:
            prefix_match = valid & ((keys >> (shift + 4)) == prefix)
        bucket = (keys >> shift) & 0xF
        cumulative = 0
        selected = 0
        selected_remaining = remaining
        found = 0

        for bucket_id in gl.static_range(15, -1, -1):
            in_bucket = prefix_match & (bucket == bucket_id)
            count = gl.sum(
                gl.where(
                    in_bucket,
                    gl.full([BLOCK_N], 1, gl.int32, layout=layout),
                    gl.full([BLOCK_N], 0, gl.int32, layout=layout),
                ),
                axis=0,
            ).to(gl.int32)
            take = (found == 0) & (remaining <= cumulative + count)
            selected = gl.where(take, bucket_id, selected)
            selected_remaining = gl.where(
                take, remaining - cumulative, selected_remaining
            )
            cumulative += gl.where(found == 0, count, 0)
            found = gl.where(take, 1, found)

        prefix = (prefix << 4) | selected
        remaining = selected_remaining

    return prefix


@gluon.jit
def _dsa_decode_select_topk_kernel(
    logits,
    block_table,
    seq_lens,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    topk_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    top_offsets = gl.arange(0, topk, layout=topk_layout)
    req = row // q_len_per_req
    q_offset = row - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    lens = gl.minimum(seq_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(out + row * out_stride + top_offsets, -1)

    if seq_len <= topk:
        valid_top = top_offsets < seq_len
        local = top_offsets.to(gl.int32)
        block_idx = local // page_size
        block_offset = local - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=valid_top & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        slots = page * page_size + block_offset
        gl.store(
            out + row * out_stride + top_offsets,
            gl.where(valid_top, slots, -1),
            mask=top_offsets < topk,
        )
        return

    valid = offsets < seq_len
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    threshold = _find_topk_threshold_key(values, valid, topk, BLOCK_N, layout)
    keys = _fp32_to_ordered_key(values)
    greater = valid & (keys > threshold)
    equal = valid & (keys == threshold)
    greater_i32 = greater.to(gl.int32)
    equal_i32 = equal.to(gl.int32)
    count_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
    greater_pos = gl.associative_scan(greater_i32, 0, _topk_add) - 1
    equal_pos = count_greater + gl.associative_scan(equal_i32, 0, _topk_add) - 1
    greater_write = greater & (greater_pos < topk)
    equal_write = equal & (equal_pos < topk)
    local = offsets.to(gl.int32)
    block_idx = local // page_size
    block_offset = local - block_idx * page_size
    page = gl.load(
        block_table + req * block_table_stride + block_idx,
        mask=(greater_write | equal_write) & (block_idx < block_table_cols),
        other=0,
    ).to(gl.int32)
    slots = page * page_size + block_offset
    gl.store(out + row * out_stride + greater_pos, slots, mask=greater_write)
    gl.store(out + row * out_stride + equal_pos, slots, mask=equal_write)


@gluon.jit
def _dsa_prefill_select_topk_kernel(
    logits,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    topk_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    top_offsets = gl.arange(0, topk, layout=topk_layout)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(row_end - row_start, 0)
    lens = gl.minimum(candidate_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(out + row * out_stride + top_offsets, -1)

    if candidate_len <= topk:
        local = row_start + top_offsets.to(gl.int32)
        valid_top = top_offsets < candidate_len
        gl.store(
            out + row * out_stride + top_offsets,
            gl.where(valid_top, local, -1),
            mask=top_offsets < topk,
        )
        return

    valid = (offsets >= row_start) & (offsets < row_end)
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    threshold = _find_topk_threshold_key(values, valid, topk, BLOCK_N, layout)
    keys = _fp32_to_ordered_key(values)
    greater = valid & (keys > threshold)
    equal = valid & (keys == threshold)
    greater_i32 = greater.to(gl.int32)
    equal_i32 = equal.to(gl.int32)
    count_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
    greater_pos = gl.associative_scan(greater_i32, 0, _topk_add) - 1
    equal_pos = count_greater + gl.associative_scan(equal_i32, 0, _topk_add) - 1
    greater_write = greater & (greater_pos < topk)
    equal_write = equal & (equal_pos < topk)
    local = offsets.to(gl.int32)
    gl.store(out + row * out_stride + greater_pos, local, mask=greater_write)
    gl.store(out + row * out_stride + equal_pos, local, mask=equal_write)


@gluon.jit
def _dsa_trivial_topk_kernel(
    block_table,
    seq_lens,
    row_starts,
    row_ends,
    out,
    lens_out,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, topk, layout=layout)

    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_len = gl.load(seq_lens + req).to(gl.int32)
        if q_len_per_req != 1:
            candidate_len = candidate_len - (q_len_per_req - 1) + q_offset
        valid = offsets < candidate_len
        block_idx = offsets.to(gl.int32) // page_size
        block_offset = offsets.to(gl.int32) - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=valid & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        indices = page * page_size + block_offset
    else:
        row_start = gl.load(row_starts + row).to(gl.int32)
        row_end = gl.load(row_ends + row).to(gl.int32)
        candidate_len = gl.maximum(row_end - row_start, 0)
        valid = offsets < candidate_len
        indices = row_start + offsets.to(gl.int32)

    gl.store(out + row * out_stride + offsets, gl.where(valid, indices, -1))
    gl.store(lens_out + row, gl.minimum(candidate_len, topk).to(gl.int32))


@gluon.jit(
    do_not_specialize=[
        "block_table_stride",
        "out_stride",
        "block_table_cols",
    ]
)
def _dsa_trivial_decode_topk2048_kernel(
    block_table,
    seq_lens,
    out,
    lens_out,
    block_table_stride,
    out_stride,
    block_table_cols,
    page_size: gl.constexpr,
    q_len_per_req: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [32 // gl.num_warps(), 1],
        [1, 64],
        [gl.num_warps(), 1],
        [1, 0],
    )
    page_indices = gl.expand_dims(
        gl.arange(0, 32, layout=gl.SliceLayout(1, layout)),
        1,
    )
    page_offsets = gl.expand_dims(
        gl.arange(0, 64, layout=gl.SliceLayout(0, layout)),
        0,
    )
    offsets = page_indices * 64 + page_offsets

    if q_len_per_req == 1:
        req = row
        candidate_len = gl.load(seq_lens + row).to(gl.int32)
    else:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_len = gl.load(seq_lens + req).to(gl.int32)
        candidate_len = candidate_len - (q_len_per_req - 1) + q_offset

    valid = offsets < candidate_len
    page = gl.load(
        block_table + req * block_table_stride + page_indices,
        mask=(page_indices * 64 < candidate_len) & (page_indices < block_table_cols),
        other=0,
    ).to(gl.int32)
    indices = page * 64 + page_offsets
    gl.store(
        out + row * out_stride + offsets,
        gl.where(valid, indices, -1),
    )
    gl.store(lens_out + row, gl.minimum(candidate_len, 2048).to(gl.int32))


@gluon.jit(do_not_specialize=("out_stride",))
def _dsa_trivial_prefill_topk2048_kernel(
    row_starts,
    row_ends,
    out,
    lens_out,
    out_stride,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(2048, gl.num_warps(), 4)
    offsets = gl.arange(0, 2048, layout=layout)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(row_end - row_start, 0)
    valid = offsets < candidate_len
    indices = row_start + offsets.to(gl.int32)
    gl.store(
        out + row * out_stride + offsets,
        gl.where(valid, indices, -1),
    )
    gl.store(lens_out + row, gl.minimum(candidate_len, 2048).to(gl.int32))


@gluon.jit
def _load_oneblock_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets = tile_start + gl.arange(0, BLOCK_N, layout=value_layout)
    offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
    if IS_TAIL:
        valid = offsets < candidate_len
        vector_mask = gl.max_constancy(offsets < vector_end, 4)
        vector_values = gl.amd.cdna4.buffer_load(
            ptr=candidate_logits,
            offsets=offsets,
            mask=vector_mask,
            other=-float("inf"),
        )
        tail_mask = (offsets >= vector_end) & valid
        tail_values = gl.load(
            candidate_logits + offsets,
            mask=tail_mask,
            other=-float("inf"),
        )
        values = gl.where(tail_mask, tail_values, vector_values)
    else:
        valid = gl.full([BLOCK_N], True, gl.int1, layout=value_layout)
        values = gl.amd.cdna4.buffer_load(
            ptr=candidate_logits,
            offsets=offsets,
        )
    return offsets, values, valid


@gluon.jit
def _accumulate_oneblock_histogram_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    prefix,
    shared_histogram,
    shift: gl.constexpr,
    radix_bits: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    FIRST_PASS: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    _, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    if FIRST_PASS:
        prefix_match = valid
    else:
        prefix_match = valid & ((keys >> (shift + radix_bits)) == prefix)
    buckets = (keys >> shift) & ((1 << radix_bits) - 1)
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )


@gluon.jit
def _emit_oneblock_topk_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    candidate_start,
    prefix,
    count_greater,
    remaining,
    selected_count,
    shared_output_counters,
    out,
    row,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PREFIX_SHIFT: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    compared_keys = keys if PREFIX_SHIFT == 0 else keys >> PREFIX_SHIFT
    greater_mask = valid & (compared_keys < prefix)
    equal_mask = valid & (compared_keys == prefix)
    reservation_mask = greater_mask | equal_mask
    reservation_counter = gl.where(greater_mask, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    greater_position = reservation
    equal_rank = reservation
    equal_position = count_greater + equal_rank
    greater_write = greater_mask & (greater_position < selected_count)
    equal_write = equal_mask & (equal_rank < remaining)
    logical_offsets = candidate_start + offsets.to(gl.int32)

    gl.store(
        out + row * out_stride + greater_position,
        logical_offsets,
        mask=greater_write,
    )
    gl.store(
        out + row * out_stride + equal_position,
        logical_offsets,
        mask=equal_write,
    )


@gluon.jit
def _accumulate_runtime_radix_histogram_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    prefix,
    prefix_shift,
    shift,
    pass_index,
    shared_histogram,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    _, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    if pass_index == 0:
        prefix_match = valid
    else:
        positioned_keys = (keys >> prefix_shift) << prefix_shift
        prefix_match = valid & (positioned_keys == prefix)
    buckets = (keys >> shift) & 0xFFF
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )


@gluon.jit
def _emit_runtime_radix_topk_tile(
    candidate_logits,
    candidate_start,
    tile_start,
    candidate_len,
    vector_end,
    prefix,
    prefix_shift,
    count_greater,
    remaining,
    selected_count,
    shared_output_counters,
    out,
    row,
    out_stride: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    positioned_keys = (keys >> prefix_shift) << prefix_shift
    greater_mask = valid & (positioned_keys < prefix)
    equal_mask = valid & (positioned_keys == prefix)
    reservation_mask = greater_mask | equal_mask
    reservation_counter = gl.where(greater_mask, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    greater_position = reservation
    equal_rank = reservation
    equal_position = count_greater + equal_rank
    logical_offsets = candidate_start + offsets.to(gl.int32)
    gl.store(
        out + row * out_stride + greater_position,
        logical_offsets,
        mask=greater_mask & (greater_position < selected_count),
    )
    gl.store(
        out + row * out_stride + equal_position,
        logical_offsets,
        mask=equal_mask & (equal_rank < remaining),
    )


@gluon.jit
def _emit_runtime_radix_topk_tile_deterministic(
    candidate_logits,
    candidate_start,
    tile_start,
    candidate_len,
    vector_end,
    prefix,
    prefix_shift,
    count_greater,
    remaining,
    selected_count,
    greater_cursor,
    equal_cursor,
    out,
    row,
    out_stride: gl.constexpr,
    thread_layout: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    positioned_keys = (keys >> prefix_shift) << prefix_shift
    greater_mask = valid & (positioned_keys < prefix)
    equal_mask = valid & (positioned_keys == prefix)
    greater_position, tile_greater = _rank_four_items_per_thread(
        greater_mask.to(gl.int32),
        greater_cursor,
        thread_layout,
        value_layout,
        BLOCK_N,
    )
    equal_rank, tile_equal = _rank_four_items_per_thread(
        equal_mask.to(gl.int32),
        equal_cursor,
        thread_layout,
        value_layout,
        BLOCK_N,
    )
    equal_position = count_greater + equal_rank
    logical_offsets = candidate_start + offsets.to(gl.int32)
    gl.store(
        out + row * out_stride + greater_position,
        logical_offsets,
        mask=greater_mask & (greater_position < selected_count),
    )
    gl.store(
        out + row * out_stride + equal_position,
        logical_offsets,
        mask=equal_mask & (equal_rank < remaining),
    )
    return tile_greater, tile_equal


@gluon.jit
def _accumulate_compact_final_histogram_tile(
    candidate_logits,
    tile_start,
    candidate_len,
    vector_end,
    candidate_start,
    prefix,
    shared_histogram,
    shared_output_counters,
    shared_compact_keys,
    shared_compact_offsets,
    block_table,
    out,
    row,
    req,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    IS_DECODE: gl.constexpr,
    value_layout: gl.constexpr,
    BLOCK_N: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    IS_TAIL: gl.constexpr,
):
    offsets, values, valid = _load_oneblock_tile(
        candidate_logits,
        tile_start,
        candidate_len,
        vector_end,
        value_layout,
        BLOCK_N,
        IS_TAIL,
    )
    keys = _fp32_to_topk_key(values)
    high_prefix = keys >> RADIX_BITS
    prefix_match = valid & (high_prefix == prefix)
    buckets = keys & ((1 << RADIX_BITS) - 1)
    shared_histogram.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        buckets.to(gl.int32),
        axis=0,
        mask=prefix_match,
    )

    definite_winner = valid & (high_prefix < prefix)
    reservation_mask = definite_winner | prefix_match
    reservation_counter = gl.where(definite_winner, 0, 2).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([BLOCK_N], 1, gl.int32, layout=value_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    logical_offsets = candidate_start + offsets.to(gl.int32)
    shared_compact_keys.atomic_scatter_xchg(
        keys,
        reservation,
        axis=0,
        mask=prefix_match,
    )
    shared_compact_offsets.atomic_scatter_xchg(
        logical_offsets,
        reservation,
        axis=0,
        mask=prefix_match,
    )

    if IS_DECODE:
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=definite_winner & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        indices = page * page_size + block_offset
    else:
        indices = logical_offsets
    gl.store(
        out + row * out_stride + reservation,
        indices,
        mask=definite_winner,
    )


@gluon.jit
def _emit_compact_final_topk(
    shared_compact_keys,
    shared_compact_offsets,
    compact_count,
    prefix,
    count_greater,
    remaining,
    selected_count,
    shared_output_counters,
    block_table,
    out,
    row,
    req,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    IS_DECODE: gl.constexpr,
    output_layout: gl.constexpr,
):
    compact_positions = gl.arange(0, topk, layout=output_layout)
    valid = compact_positions < compact_count
    keys = shared_compact_keys.load(output_layout)
    logical_offsets = shared_compact_offsets.load(output_layout)
    greater_mask = valid & (keys < prefix)
    equal_mask = valid & (keys == prefix)
    reservation_mask = greater_mask | equal_mask
    reservation_counter = gl.where(greater_mask, 0, 1).to(gl.int32)
    reservation = shared_output_counters.atomic_scatter_add(
        gl.full([topk], 1, gl.int32, layout=output_layout),
        reservation_counter,
        axis=0,
        mask=reservation_mask,
    )
    greater_position = reservation
    equal_rank = reservation
    equal_position = count_greater + equal_rank
    greater_write = greater_mask & (greater_position < selected_count)
    equal_write = equal_mask & (equal_rank < remaining)
    write_mask = greater_write | equal_write

    if IS_DECODE:
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=write_mask & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        indices = page * page_size + block_offset
    else:
        indices = logical_offsets
    gl.store(
        out + row * out_stride + greater_position,
        indices,
        mask=greater_write,
    )
    gl.store(
        out + row * out_stride + equal_position,
        indices,
        mask=equal_write,
    )


@gluon.jit(
    do_not_specialize=[
        "logits_stride",
        "block_table_stride",
        "out_stride",
        "block_table_cols",
    ]
)
def _dsa_oneblock_manual_radix_topk_kernel(
    logits,
    block_table,
    seq_lens,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride,
    block_table_stride,
    out_stride,
    block_table_cols,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    RADIX0_BITS: gl.constexpr,
    RADIX1_BITS: gl.constexpr,
    RADIX2_BITS: gl.constexpr,
    MAX_BUCKETS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    COMPACT_FINAL_BLOCK_N: gl.constexpr,
    USE_COMPACT_FINAL: gl.constexpr,
    USE_RADIX_EARLY_STOP: gl.constexpr,
):
    row = gl.program_id(0)
    value_layout: gl.constexpr = _vector_layout(
        BLOCK_N,
        gl.num_warps(),
        triton.cdiv(BLOCK_N, 64 * gl.num_warps()),
    )
    histogram_layout: gl.constexpr = _vector_layout(
        MAX_BUCKETS,
        gl.num_warps(),
        triton.cdiv(MAX_BUCKETS, 64 * gl.num_warps()),
    )
    group_layout: gl.constexpr = _vector_layout(
        MAX_BUCKETS // 2,
        gl.num_warps(),
        triton.cdiv(MAX_BUCKETS // 2, 64 * gl.num_warps()),
    )
    output_layout: gl.constexpr = _vector_layout(
        topk,
        gl.num_warps(),
        triton.cdiv(topk, 64 * gl.num_warps()),
    )
    if USE_COMPACT_FINAL:
        compact_final_layout: gl.constexpr = _vector_layout(
            COMPACT_FINAL_BLOCK_N,
            gl.num_warps(),
            triton.cdiv(COMPACT_FINAL_BLOCK_N, 64 * gl.num_warps()),
        )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[MAX_BUCKETS, 1]],
        [MAX_BUCKETS],
        [0],
    )
    histogram_zeros = gl.zeros([MAX_BUCKETS], gl.int32, layout=histogram_layout)
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [MAX_BUCKETS],
        shared_layout,
        value=histogram_zeros,
    )
    output_counter_count: gl.constexpr = 4 if USE_COMPACT_FINAL else 2
    output_counter_layout: gl.constexpr = _vector_layout(
        output_counter_count, gl.num_warps(), 1
    )
    output_counter_shared_layout: gl.constexpr = (
        gl.PaddedSharedLayout.with_identity_for(
            [[output_counter_count, 1]], [output_counter_count], [0]
        )
    )
    output_counter_zeros = gl.zeros(
        [output_counter_count], gl.int32, layout=output_counter_layout
    )
    shared_output_counters = gl.allocate_shared_memory(
        gl.int32,
        [output_counter_count],
        output_counter_shared_layout,
        value=output_counter_zeros,
    )
    if USE_COMPACT_FINAL:
        compact_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[topk, 1]],
            [topk],
            [0],
        )
        shared_compact_keys = gl.allocate_shared_memory(
            gl.uint32,
            [topk],
            compact_shared_layout,
        )
        shared_compact_offsets = gl.allocate_shared_memory(
            gl.int32,
            [topk],
            compact_shared_layout,
        )
    gl.barrier()

    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_start = gl.full([], 0, gl.int32)
        candidate_end = gl.load(seq_lens + req).to(gl.int32)
        if q_len_per_req != 1:
            candidate_end = candidate_end - (q_len_per_req - 1) + q_offset
    else:
        req = row
        candidate_start = gl.load(row_starts + row).to(gl.int32)
        candidate_end = gl.load(row_ends + row).to(gl.int32)

    candidate_len = gl.maximum(candidate_end - candidate_start, 0)
    selected_count = gl.minimum(candidate_len, topk).to(gl.int32)
    output_offsets = gl.arange(0, topk, layout=output_layout)
    gl.store(lens_out + row, selected_count)
    gl.store(out + row * out_stride + output_offsets, -1)

    if candidate_len <= topk:
        valid = output_offsets < candidate_len
        logical_offsets = candidate_start + output_offsets.to(gl.int32)
        if IS_DECODE:
            block_idx = logical_offsets // page_size
            block_offset = logical_offsets - block_idx * page_size
            page = gl.load(
                block_table + req * block_table_stride + block_idx,
                mask=valid & (block_idx < block_table_cols),
                other=0,
            ).to(gl.int32)
            indices = page * page_size + block_offset
        else:
            indices = logical_offsets
        gl.store(
            out + row * out_stride + output_offsets,
            gl.where(valid, indices, -1),
        )
        return

    candidate_logits = logits + row * logits_stride + candidate_start
    vector_end = candidate_len & -4
    prefix = gl.full([], 0, gl.uint32)
    remaining = selected_count
    if USE_COMPACT_FINAL:
        compact_count = candidate_len
    bucket_offsets = gl.arange(0, MAX_BUCKETS, layout=histogram_layout)

    for pass_index in gl.static_range(3):
        radix_bits = RADIX0_BITS
        shift = 32 - RADIX0_BITS
        if pass_index == 1:
            radix_bits = RADIX1_BITS
            shift = 32 - RADIX0_BITS - RADIX1_BITS
        elif pass_index == 2:
            radix_bits = RADIX2_BITS
            shift = 0

        if pass_index != 0:
            gl.barrier()
            shared_histogram.store(histogram_zeros)
            gl.barrier()

        full_end = candidate_len & -BLOCK_N
        if USE_COMPACT_FINAL and pass_index == 2:
            if compact_count <= topk:
                compact_full_end = candidate_len & -COMPACT_FINAL_BLOCK_N
                for tile_start in range(0, compact_full_end, COMPACT_FINAL_BLOCK_N):
                    _accumulate_compact_final_histogram_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        shared_histogram,
                        shared_output_counters,
                        shared_compact_keys,
                        shared_compact_offsets,
                        block_table,
                        out,
                        row,
                        req,
                        block_table_stride,
                        out_stride,
                        block_table_cols,
                        page_size,
                        IS_DECODE,
                        compact_final_layout,
                        COMPACT_FINAL_BLOCK_N,
                        radix_bits,
                        False,
                    )
                if compact_full_end < candidate_len:
                    _accumulate_compact_final_histogram_tile(
                        candidate_logits,
                        compact_full_end,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        shared_histogram,
                        shared_output_counters,
                        shared_compact_keys,
                        shared_compact_offsets,
                        block_table,
                        out,
                        row,
                        req,
                        block_table_stride,
                        out_stride,
                        block_table_cols,
                        page_size,
                        IS_DECODE,
                        compact_final_layout,
                        COMPACT_FINAL_BLOCK_N,
                        radix_bits,
                        True,
                    )
            else:
                for tile_start in range(0, full_end, BLOCK_N):
                    _accumulate_oneblock_histogram_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        prefix,
                        shared_histogram,
                        shift,
                        radix_bits,
                        value_layout,
                        BLOCK_N,
                        False,
                        False,
                    )
                if full_end < candidate_len:
                    _accumulate_oneblock_histogram_tile(
                        candidate_logits,
                        full_end,
                        candidate_len,
                        vector_end,
                        prefix,
                        shared_histogram,
                        shift,
                        radix_bits,
                        value_layout,
                        BLOCK_N,
                        False,
                        True,
                    )
        else:
            for tile_start in range(0, full_end, BLOCK_N):
                _accumulate_oneblock_histogram_tile(
                    candidate_logits,
                    tile_start,
                    candidate_len,
                    vector_end,
                    prefix,
                    shared_histogram,
                    shift,
                    radix_bits,
                    value_layout,
                    BLOCK_N,
                    pass_index == 0,
                    False,
                )
            if full_end < candidate_len:
                _accumulate_oneblock_histogram_tile(
                    candidate_logits,
                    full_end,
                    candidate_len,
                    vector_end,
                    prefix,
                    shared_histogram,
                    shift,
                    radix_bits,
                    value_layout,
                    BLOCK_N,
                    pass_index == 0,
                    True,
                )

        gl.barrier()
        counts = shared_histogram.load(histogram_layout)
        count_pairs = counts.reshape([MAX_BUCKETS // 2, 2])
        count_low, count_high = gl.split(count_pairs)
        count_low = gl.convert_layout(count_low, group_layout)
        count_high = gl.convert_layout(count_high, group_layout)
        group_counts = count_low + count_high
        group_cumulative = gl.associative_scan(group_counts, 0, _topk_add)
        group_greater = group_cumulative - group_counts
        selected_group = (group_greater < remaining) & (group_cumulative >= remaining)
        bucket_pairs = bucket_offsets.reshape([MAX_BUCKETS // 2, 2])
        bucket_low, bucket_high = gl.split(bucket_pairs)
        bucket_low = gl.convert_layout(bucket_low, group_layout)
        bucket_high = gl.convert_layout(bucket_high, group_layout)
        select_low = group_greater + count_low >= remaining
        group_bucket = gl.where(select_low, bucket_low, bucket_high)
        group_selected_greater = group_greater + gl.where(select_low, 0, count_low)
        if pass_index == 1 and (USE_COMPACT_FINAL or USE_RADIX_EARLY_STOP):
            group_selected_count = gl.where(select_low, count_low, count_high)
        selected_bucket = gl.sum(gl.where(selected_group, group_bucket, 0), axis=0).to(
            gl.int32
        )
        selected_greater = gl.sum(
            gl.where(selected_group, group_selected_greater, 0), axis=0
        ).to(gl.int32)
        if pass_index == 1 and (USE_COMPACT_FINAL or USE_RADIX_EARLY_STOP):
            selected_bucket_count = gl.sum(
                gl.where(selected_group, group_selected_count, 0), axis=0
            ).to(gl.int32)
        prefix = (prefix << radix_bits) | selected_bucket.to(gl.uint32)
        remaining -= selected_greater
        if USE_RADIX_EARLY_STOP and pass_index == 1:
            if selected_bucket_count == remaining:
                count_greater = selected_count - remaining
                emit_full_end = candidate_len & -BLOCK_N
                for tile_start in range(0, emit_full_end, BLOCK_N):
                    _emit_oneblock_topk_tile(
                        candidate_logits,
                        tile_start,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        count_greater,
                        remaining,
                        selected_count,
                        shared_output_counters,
                        out,
                        row,
                        out_stride,
                        value_layout,
                        BLOCK_N,
                        shift,
                        False,
                    )
                if emit_full_end < candidate_len:
                    _emit_oneblock_topk_tile(
                        candidate_logits,
                        emit_full_end,
                        candidate_len,
                        vector_end,
                        candidate_start,
                        prefix,
                        count_greater,
                        remaining,
                        selected_count,
                        shared_output_counters,
                        out,
                        row,
                        out_stride,
                        value_layout,
                        BLOCK_N,
                        shift,
                        True,
                    )

                if IS_DECODE:
                    gl.barrier()
                    valid_output = output_offsets < selected_count
                    logical_offsets = gl.load(
                        out + row * out_stride + output_offsets,
                        mask=valid_output,
                        other=0,
                    ).to(gl.int32)
                    block_idx = logical_offsets // page_size
                    block_offset = logical_offsets - block_idx * page_size
                    page = gl.load(
                        block_table + req * block_table_stride + block_idx,
                        mask=valid_output & (block_idx < block_table_cols),
                        other=0,
                    ).to(gl.int32)
                    gl.store(
                        out + row * out_stride + output_offsets,
                        page * page_size + block_offset,
                        mask=valid_output,
                    )
                return
        if USE_COMPACT_FINAL and pass_index == 1:
            compact_count = selected_bucket_count

    count_greater = selected_count - remaining
    if USE_COMPACT_FINAL:
        if compact_count <= topk:
            _emit_compact_final_topk(
                shared_compact_keys,
                shared_compact_offsets,
                compact_count,
                prefix,
                count_greater,
                remaining,
                selected_count,
                shared_output_counters,
                block_table,
                out,
                row,
                req,
                block_table_stride,
                out_stride,
                block_table_cols,
                page_size,
                topk,
                IS_DECODE,
                output_layout,
            )
            return

    full_end = candidate_len & -BLOCK_N
    for tile_start in range(0, full_end, BLOCK_N):
        _emit_oneblock_topk_tile(
            candidate_logits,
            tile_start,
            candidate_len,
            vector_end,
            candidate_start,
            prefix,
            count_greater,
            remaining,
            selected_count,
            shared_output_counters,
            out,
            row,
            out_stride,
            value_layout,
            BLOCK_N,
            0,
            False,
        )
    if full_end < candidate_len:
        _emit_oneblock_topk_tile(
            candidate_logits,
            full_end,
            candidate_len,
            vector_end,
            candidate_start,
            prefix,
            count_greater,
            remaining,
            selected_count,
            shared_output_counters,
            out,
            row,
            out_stride,
            value_layout,
            BLOCK_N,
            0,
            True,
        )

    if IS_DECODE:
        gl.barrier()
        valid_output = output_offsets < selected_count
        logical_offsets = gl.load(
            out + row * out_stride + output_offsets,
            mask=valid_output,
            other=0,
        ).to(gl.int32)
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=valid_output & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        gl.store(
            out + row * out_stride + output_offsets,
            page * page_size + block_offset,
            mask=valid_output,
        )


@gluon.jit(
    do_not_specialize=[
        "logits_stride",
        "block_table_stride",
        "out_stride",
        "block_table_cols",
    ]
)
def _dsa_runtime_radix_topk_kernel(
    logits,
    block_table,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride,
    block_table_stride,
    out_stride,
    block_table_cols,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    DETERMINISTIC_EMIT: gl.constexpr,
    MAX_BUCKETS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    value_layout: gl.constexpr = _vector_layout(
        BLOCK_N,
        gl.num_warps(),
        LOAD_ELEMS,
    )
    thread_layout: gl.constexpr = _vector_layout(
        BLOCK_N // 4,
        gl.num_warps(),
        1,
    )
    histogram_layout: gl.constexpr = _vector_layout(
        MAX_BUCKETS,
        gl.num_warps(),
        triton.cdiv(MAX_BUCKETS, 64 * gl.num_warps()),
    )
    group_layout: gl.constexpr = _vector_layout(
        MAX_BUCKETS // 2,
        gl.num_warps(),
        triton.cdiv(MAX_BUCKETS // 2, 64 * gl.num_warps()),
    )
    output_layout: gl.constexpr = _vector_layout(
        topk,
        gl.num_warps(),
        triton.cdiv(topk, 64 * gl.num_warps()),
    )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[MAX_BUCKETS, 1]],
        [MAX_BUCKETS],
        [0],
    )
    histogram_zeros = gl.zeros([MAX_BUCKETS], gl.int32, layout=histogram_layout)
    shared_histogram = gl.allocate_shared_memory(
        gl.int32,
        [MAX_BUCKETS],
        shared_layout,
    )
    output_counter_layout: gl.constexpr = _vector_layout(2, gl.num_warps(), 1)
    output_counter_shared_layout: gl.constexpr = (
        gl.PaddedSharedLayout.with_identity_for([[2, 1]], [2], [0])
    )
    shared_output_counters = gl.allocate_shared_memory(
        gl.int32,
        [2],
        output_counter_shared_layout,
        value=gl.zeros([2], gl.int32, layout=output_counter_layout),
    )
    gl.barrier()

    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_start = gl.full([], 0, gl.int32)
        candidate_end = gl.load(row_ends + req).to(gl.int32)
        if q_len_per_req != 1:
            candidate_end = candidate_end - (q_len_per_req - 1) + q_offset
    else:
        req = row
        candidate_start = gl.load(row_starts + row).to(gl.int32)
        candidate_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(candidate_end - candidate_start, 0)
    selected_count = gl.minimum(candidate_len, topk).to(gl.int32)
    output_offsets = gl.arange(0, topk, layout=output_layout)
    gl.store(lens_out + row, selected_count)
    gl.store(out + row * out_stride + output_offsets, -1)

    if candidate_len <= topk:
        valid = output_offsets < candidate_len
        logical_offsets = candidate_start + output_offsets.to(gl.int32)
        if IS_DECODE:
            block_idx = logical_offsets // page_size
            block_offset = logical_offsets - block_idx * page_size
            page = gl.load(
                block_table + req * block_table_stride + block_idx,
                mask=valid & (block_idx < block_table_cols),
                other=0,
            ).to(gl.int32)
            indices = page * page_size + block_offset
        else:
            indices = logical_offsets
        gl.store(
            out + row * out_stride + output_offsets,
            gl.where(valid, indices, -1),
        )
        return

    candidate_logits = logits + row * logits_stride + candidate_start
    vector_end = candidate_len & -4
    prefix = gl.full([], 0, gl.uint32)
    prefix_shift = gl.full([], 0, gl.int32)
    remaining = selected_count
    pass_index = gl.full([], 0, gl.int32)
    done = gl.full([], 0, gl.int1)
    bucket_offsets = gl.arange(0, MAX_BUCKETS, layout=histogram_layout)

    while (pass_index < 3) & ~done:
        gl.barrier()
        shared_histogram.store(histogram_zeros)
        gl.barrier()

        shift = gl.maximum(20 - pass_index * 12, 0)
        full_end = candidate_len & -BLOCK_N
        for tile_start in range(0, full_end, BLOCK_N):
            _accumulate_runtime_radix_histogram_tile(
                candidate_logits,
                tile_start,
                candidate_len,
                vector_end,
                prefix,
                prefix_shift,
                shift,
                pass_index,
                shared_histogram,
                value_layout,
                BLOCK_N,
                False,
            )
        if full_end < candidate_len:
            _accumulate_runtime_radix_histogram_tile(
                candidate_logits,
                full_end,
                candidate_len,
                vector_end,
                prefix,
                prefix_shift,
                shift,
                pass_index,
                shared_histogram,
                value_layout,
                BLOCK_N,
                True,
            )

        gl.barrier()
        counts = shared_histogram.load(histogram_layout)
        count_pairs = counts.reshape([MAX_BUCKETS // 2, 2])
        count_low, count_high = gl.split(count_pairs)
        count_low = gl.convert_layout(count_low, group_layout)
        count_high = gl.convert_layout(count_high, group_layout)
        group_counts = count_low + count_high
        group_cumulative = gl.associative_scan(group_counts, 0, _topk_add)
        group_greater = group_cumulative - group_counts
        selected_group = (group_greater < remaining) & (group_cumulative >= remaining)
        bucket_pairs = bucket_offsets.reshape([MAX_BUCKETS // 2, 2])
        bucket_low, bucket_high = gl.split(bucket_pairs)
        bucket_low = gl.convert_layout(bucket_low, group_layout)
        bucket_high = gl.convert_layout(bucket_high, group_layout)
        select_low = group_greater + count_low >= remaining
        group_bucket = gl.where(select_low, bucket_low, bucket_high)
        group_selected_greater = group_greater + gl.where(select_low, 0, count_low)
        group_selected_count = gl.where(select_low, count_low, count_high)
        selected_bucket = gl.sum(gl.where(selected_group, group_bucket, 0), axis=0).to(
            gl.int32
        )
        selected_greater = gl.sum(
            gl.where(selected_group, group_selected_greater, 0), axis=0
        ).to(gl.int32)
        selected_bucket_count = gl.sum(
            gl.where(selected_group, group_selected_count, 0), axis=0
        ).to(gl.int32)
        prefix |= selected_bucket.to(gl.uint32) << shift
        remaining -= selected_greater
        prefix_shift = shift
        done = selected_bucket_count == remaining
        pass_index += 1

    count_greater = selected_count - remaining
    emit_full_end = candidate_len & -BLOCK_N
    if IS_DECODE or not DETERMINISTIC_EMIT:
        for tile_start in range(0, emit_full_end, BLOCK_N):
            _emit_runtime_radix_topk_tile(
                candidate_logits,
                candidate_start,
                tile_start,
                candidate_len,
                vector_end,
                prefix,
                prefix_shift,
                count_greater,
                remaining,
                selected_count,
                shared_output_counters,
                out,
                row,
                out_stride,
                value_layout,
                BLOCK_N,
                False,
            )
        if emit_full_end < candidate_len:
            _emit_runtime_radix_topk_tile(
                candidate_logits,
                candidate_start,
                emit_full_end,
                candidate_len,
                vector_end,
                prefix,
                prefix_shift,
                count_greater,
                remaining,
                selected_count,
                shared_output_counters,
                out,
                row,
                out_stride,
                value_layout,
                BLOCK_N,
                True,
            )
    else:
        greater_cursor = 0
        equal_cursor = 0
        for tile_start in range(0, emit_full_end, BLOCK_N):
            tile_greater, tile_equal = _emit_runtime_radix_topk_tile_deterministic(
                candidate_logits,
                candidate_start,
                tile_start,
                candidate_len,
                vector_end,
                prefix,
                prefix_shift,
                count_greater,
                remaining,
                selected_count,
                greater_cursor,
                equal_cursor,
                out,
                row,
                out_stride,
                thread_layout,
                value_layout,
                BLOCK_N,
                False,
            )
            greater_cursor += tile_greater
            equal_cursor += tile_equal
        if emit_full_end < candidate_len:
            _emit_runtime_radix_topk_tile_deterministic(
                candidate_logits,
                candidate_start,
                emit_full_end,
                candidate_len,
                vector_end,
                prefix,
                prefix_shift,
                count_greater,
                remaining,
                selected_count,
                greater_cursor,
                equal_cursor,
                out,
                row,
                out_stride,
                thread_layout,
                value_layout,
                BLOCK_N,
                True,
            )

    if IS_DECODE:
        gl.barrier()
        valid_output = output_offsets < selected_count
        logical_offsets = gl.load(
            out + row * out_stride + output_offsets,
            mask=valid_output,
            other=0,
        ).to(gl.int32)
        block_idx = logical_offsets // page_size
        block_offset = logical_offsets - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=valid_output & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        gl.store(
            out + row * out_stride + output_offsets,
            page * page_size + block_offset,
            mask=valid_output,
        )


@gluon.jit
def _dsa_decode_radix_init_kernel(
    seq_lens,
    out,
    lens_out,
    prefixes,
    remaining,
    counters,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    top_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    top_offsets = gl.arange(0, topk, layout=top_layout)
    req = row // q_len_per_req
    q_offset = row - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    lens = gl.minimum(seq_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(prefixes + row, 0)
    gl.store(remaining + row, lens)
    gl.store(counters + row * 2, 0)
    gl.store(counters + row * 2 + 1, 0)
    gl.store(out + row * out_stride + top_offsets, -1, mask=top_offsets < topk)


@gluon.jit
def _dsa_prefill_radix_init_kernel(
    row_starts,
    row_ends,
    out,
    lens_out,
    prefixes,
    remaining,
    counters,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    top_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    top_offsets = gl.arange(0, topk, layout=top_layout)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(row_end - row_start, 0)
    lens = gl.minimum(candidate_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(prefixes + row, 0)
    gl.store(remaining + row, lens)
    gl.store(counters + row * 2, 0)
    gl.store(counters + row * 2 + 1, 0)
    gl.store(out + row * out_stride + top_offsets, -1, mask=top_offsets < topk)
    if candidate_len <= topk:
        valid = top_offsets < candidate_len
        gl.store(
            out + row * out_stride + top_offsets,
            gl.where(valid, row_start + top_offsets.to(gl.int32), -1),
            mask=top_offsets < topk,
        )


@gluon.jit
def _dsa_radix_hist_kernel(
    logits,
    prefixes,
    hist,
    logits_stride: gl.constexpr,
    hist_tiles: gl.constexpr,
    n_cols: gl.constexpr,
    shift: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    tile = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    mask = offsets < n_cols
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    keys = _fp32_to_ordered_key(values)
    prefix = gl.load(prefixes + row).to(gl.uint32)
    if shift == 28:
        prefix_match = mask
    else:
        prefix_match = (keys >> (shift + 4)) == prefix
    bucket = (keys >> shift) & 0xF
    base = (row * hist_tiles + tile) * 16
    for bucket_id in gl.static_range(0, 16):
        count = gl.sum(
            gl.where(
                mask & prefix_match & (bucket == bucket_id),
                gl.full([BLOCK_N], 1, gl.int32, layout=layout),
                gl.full([BLOCK_N], 0, gl.int32, layout=layout),
            ),
            axis=0,
        ).to(gl.int32)
        gl.store(hist + base + bucket_id, count)


@gluon.jit
def _dsa_radix_grouped_hist_kernel(
    logits,
    prefixes,
    hist,
    logits_stride: gl.constexpr,
    tiles_per_row: gl.constexpr,
    groups_per_row: gl.constexpr,
    n_cols: gl.constexpr,
    shift: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    hist_layout: gl.constexpr = _vector_layout(16, gl.num_warps(), 1)
    prefix = gl.load(prefixes + row).to(gl.uint32)
    counts = gl.zeros([16], gl.int32, layout=hist_layout)

    for tile in range(group, tiles_per_row, groups_per_row):
        offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
        mask = offsets < n_cols
        values = gl.load(
            logits + row * logits_stride + offsets,
            mask=mask,
            other=-float("inf"),
        )
        keys = _fp32_to_ordered_key(values)
        if shift == 28:
            prefix_match = mask
        else:
            prefix_match = mask & ((keys >> (shift + 4)) == prefix)
        bucket = (keys >> shift) & 0xF
        counts += gl.histogram(
            bucket.to(gl.int32),
            16,
            mask=prefix_match,
            layout=hist_layout,
        )

    base = (row * groups_per_row + group) * 16
    bucket_offsets = gl.arange(0, 16, layout=hist_layout)
    gl.store(hist + base + bucket_offsets, counts)


@gluon.jit
def _dsa_prefill_wide_radix_grouped_hist_kernel(
    logits,
    prefixes,
    row_starts,
    row_ends,
    hist,
    logits_stride: gl.constexpr,
    tiles_per_row: gl.constexpr,
    groups_per_row: gl.constexpr,
    row_groups_stride: gl.constexpr,
    hist_group_stride: gl.constexpr,
    n_cols: gl.constexpr,
    shift: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    IS_DECODE: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    hist_layout: gl.constexpr = _vector_layout(
        NUM_BUCKETS,
        gl.num_warps(),
        triton.cdiv(NUM_BUCKETS, 64 * gl.num_warps()),
    )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[NUM_BUCKETS, 1]],
        [NUM_BUCKETS],
        [0],
    )
    shared_hist = gl.allocate_shared_memory(
        gl.int32,
        [NUM_BUCKETS],
        shared_layout,
        value=gl.zeros(
            [NUM_BUCKETS],
            gl.int32,
            layout=hist_layout,
        ),
    )
    gl.barrier()
    prefix = gl.load(prefixes + row).to(gl.uint32)
    if IS_DECODE:
        req = row // q_len_per_req
        q_offset = row - req * q_len_per_req
        candidate_start = gl.full([], 0, gl.int32)
        candidate_end = gl.load(row_ends + req).to(gl.int32)
        if q_len_per_req != 1:
            candidate_end = candidate_end - (q_len_per_req - 1) + q_offset
    else:
        candidate_start = gl.load(row_starts + row).to(gl.int32)
        candidate_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(candidate_end - candidate_start, 0)
    if not IS_DECODE:
        if candidate_len <= topk:
            bucket_offsets = gl.arange(0, NUM_BUCKETS, layout=hist_layout)
            base = (row * row_groups_stride + group) * hist_group_stride
            gl.store(hist + base + bucket_offsets, 0)
            return

    for tile in range(group, tiles_per_row, groups_per_row):
        if IS_DECODE:
            candidate_logits = logits + row * logits_stride + candidate_start
            vector_end = candidate_len & -4
            offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
            offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
            if tile * BLOCK_N + BLOCK_N <= candidate_len:
                values = gl.amd.cdna4.buffer_load(
                    ptr=candidate_logits,
                    offsets=offsets,
                )
                valid = gl.full([BLOCK_N], True, gl.int1, layout=layout)
            else:
                valid = offsets < candidate_len
                vector_mask = gl.max_constancy(offsets < vector_end, 4)
                vector_values = gl.amd.cdna4.buffer_load(
                    ptr=candidate_logits,
                    offsets=offsets,
                    mask=vector_mask,
                    other=-float("inf"),
                )
                tail_mask = (offsets >= vector_end) & valid
                tail_values = gl.load(
                    candidate_logits + offsets,
                    mask=tail_mask,
                    other=-float("inf"),
                )
                values = gl.where(tail_mask, tail_values, vector_values)
        else:
            offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
            valid = offsets < n_cols
            values = gl.load(
                logits + row * logits_stride + offsets,
                mask=valid,
                other=-float("inf"),
            )
        keys = _fp32_to_ordered_key(values)
        if shift + RADIX_BITS == 32:
            prefix_match = valid
        else:
            prefix_match = valid & ((keys >> (shift + RADIX_BITS)) == prefix)
        bucket = (keys >> shift) & (NUM_BUCKETS - 1)
        shared_hist.atomic_scatter_add(
            gl.full([BLOCK_N], 1, gl.int32, layout=layout),
            bucket.to(gl.int32),
            axis=0,
            mask=prefix_match,
        )

    gl.barrier()
    counts = shared_hist.load(hist_layout)
    bucket_offsets = gl.arange(
        0,
        NUM_BUCKETS,
        layout=hist_layout,
    )
    base = (row * row_groups_stride + group) * hist_group_stride
    gl.store(hist + base + bucket_offsets, counts)


@gluon.jit
def _dsa_prefill_wide_radix_update_kernel(
    prefixes,
    remaining,
    hist,
    groups_per_row: gl.constexpr,
    row_groups_stride: gl.constexpr,
    hist_group_stride: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(
        NUM_BUCKETS,
        gl.num_warps(),
        LOAD_ELEMS,
    )
    buckets = gl.arange(0, NUM_BUCKETS, layout=layout)
    row_hist = hist + row * row_groups_stride * hist_group_stride
    counts = gl.zeros(
        [NUM_BUCKETS],
        gl.int32,
        layout=layout,
    )
    for group in range(0, groups_per_row):
        counts += gl.load(row_hist + group * hist_group_stride + buckets)

    descending = gl.associative_scan(counts, 0, _topk_add, reverse=True)
    greater = descending - counts
    kth = gl.load(remaining + row).to(gl.int32)
    selected_mask = (greater < kth) & (descending >= kth)
    selected = gl.sum(gl.where(selected_mask, buckets, 0), axis=0).to(gl.int32)
    selected_greater = gl.sum(gl.where(selected_mask, greater, 0), axis=0).to(gl.int32)
    prefix = gl.load(prefixes + row).to(gl.uint32)
    gl.store(
        prefixes + row,
        ((prefix << RADIX_BITS) | selected.to(gl.uint32)).to(gl.int32),
    )
    gl.store(remaining + row, kth - selected_greater)


@gluon.jit
def _dsa_prefill_derive_group_counts_from_hist_kernel(
    prefixes,
    hist,
    group_offsets,
    groups_per_row: gl.constexpr,
    hist_group_stride: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    NUM_BUCKETS: gl.constexpr,
    PASS_INDEX: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(
        NUM_BUCKETS,
        gl.num_warps(),
        LOAD_ELEMS,
    )
    buckets = gl.arange(0, NUM_BUCKETS, layout=layout)
    counts = gl.load(
        hist + (row * groups_per_row + group) * hist_group_stride + buckets
    )
    selected = gl.load(prefixes + row).to(gl.uint32) & (NUM_BUCKETS - 1)
    greater = gl.sum(gl.where(buckets > selected, counts, 0), axis=0).to(gl.int32)
    offset_base = (row * groups_per_row + group) * 2
    if PASS_INDEX == 0:
        gl.store(group_offsets + offset_base, greater)
    else:
        accumulated = gl.load(group_offsets + offset_base).to(gl.int32)
        gl.store(group_offsets + offset_base, accumulated + greater)
    if PASS_INDEX == 2:
        equal = gl.sum(gl.where(buckets == selected, counts, 0), axis=0).to(gl.int32)
        gl.store(group_offsets + offset_base + 1, equal)


@gluon.jit
def _dsa_radix_update_kernel(
    prefixes,
    remaining,
    hist,
    hist_groups: gl.constexpr,
    BLOCK_GROUPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_GROUPS, gl.num_warps(), LOAD_ELEMS)
    group_offsets = gl.arange(0, BLOCK_GROUPS, layout=layout)
    group_mask = group_offsets < hist_groups
    row_hist = hist + row * hist_groups * 16
    kth = gl.load(remaining + row).to(gl.int32)
    cumulative = 0
    selected = 0
    selected_remaining = kth
    found = 0

    for bucket_desc in gl.static_range(0, 16):
        bucket_id = 15 - bucket_desc
        counts = gl.load(
            row_hist + group_offsets * 16 + bucket_id,
            mask=group_mask,
            other=0,
        )
        count = gl.sum(counts, axis=0).to(gl.int32)
        take = (found == 0) & (kth <= cumulative + count)
        selected = gl.where(take, bucket_id, selected)
        selected_remaining = gl.where(take, kth - cumulative, selected_remaining)
        cumulative += gl.where(found == 0, count, 0)
        found = gl.where(take, 1, found)

    prefix = gl.load(prefixes + row).to(gl.uint32)
    gl.store(prefixes + row, ((prefix << 4) | selected).to(gl.int32))
    gl.store(remaining + row, selected_remaining)


@gluon.jit
def _dsa_decode_radix_scatter_slots_kernel(
    logits,
    prefixes,
    remaining,
    counters,
    block_table,
    seq_lens,
    out,
    logits_stride: gl.constexpr,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    n_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    tiles_per_row: gl.constexpr,
    groups_per_row: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    req = row // q_len_per_req
    q_offset = row - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    threshold = gl.load(prefixes + row).to(gl.uint32)
    keep_equal = gl.load(remaining + row).to(gl.int32)
    count_greater = topk - keep_equal

    for tile in range(group, tiles_per_row, groups_per_row):
        offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
        mask = (offsets < n_cols) & (offsets < seq_len)
        values = gl.load(
            logits + row * logits_stride + offsets,
            mask=mask,
            other=-float("inf"),
        )
        keys = _fp32_to_ordered_key(values)
        greater = mask & (keys > threshold)
        equal = mask & (keys == threshold)

        greater_i32 = greater.to(gl.int32)
        equal_i32 = equal.to(gl.int32)
        tile_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
        tile_equal = gl.sum(equal_i32, axis=0).to(gl.int32)
        greater_start = gl.atomic_add(
            counters + row * 2, tile_greater, sem="acq_rel", scope="gpu"
        )
        equal_start = gl.atomic_add(
            counters + row * 2 + 1, tile_equal, sem="acq_rel", scope="gpu"
        )

        greater_pos = greater_start + gl.associative_scan(greater_i32, 0, _topk_add) - 1
        equal_pos = (
            count_greater
            + equal_start
            + gl.associative_scan(equal_i32, 0, _topk_add)
            - 1
        )
        greater_write = greater & (greater_pos < topk)
        equal_write = (
            equal & (equal_pos < topk) & (equal_pos < count_greater + keep_equal)
        )

        block_idx = offsets.to(gl.int32) // page_size
        block_offset = offsets.to(gl.int32) - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=(greater_write | equal_write) & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        slots = page * page_size + block_offset
        gl.store(out + row * out_stride + greater_pos, slots, mask=greater_write)
        gl.store(out + row * out_stride + equal_pos, slots, mask=equal_write)


@gluon.jit
def _dsa_prefill_radix_scatter_kernel(
    logits,
    prefixes,
    remaining,
    counters,
    row_starts,
    row_ends,
    out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    n_cols: gl.constexpr,
    topk: gl.constexpr,
    tiles_per_row: gl.constexpr,
    groups_per_row: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    if row_end - row_start <= topk:
        return
    threshold = gl.load(prefixes + row).to(gl.uint32)
    keep_equal = gl.load(remaining + row).to(gl.int32)
    count_greater = topk - keep_equal

    for tile in range(group, tiles_per_row, groups_per_row):
        offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
        mask = (offsets >= row_start) & (offsets < row_end) & (offsets < n_cols)
        values = gl.load(
            logits + row * logits_stride + offsets,
            mask=mask,
            other=-float("inf"),
        )
        keys = _fp32_to_ordered_key(values)
        greater = mask & (keys > threshold)
        equal = mask & (keys == threshold)

        greater_i32 = greater.to(gl.int32)
        equal_i32 = equal.to(gl.int32)
        tile_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
        tile_equal = gl.sum(equal_i32, axis=0).to(gl.int32)
        greater_start = gl.atomic_add(
            counters + row * 2, tile_greater, sem="acq_rel", scope="gpu"
        )
        equal_start = gl.atomic_add(
            counters + row * 2 + 1, tile_equal, sem="acq_rel", scope="gpu"
        )

        greater_pos = greater_start + gl.associative_scan(greater_i32, 0, _topk_add) - 1
        equal_pos = (
            count_greater
            + equal_start
            + gl.associative_scan(equal_i32, 0, _topk_add)
            - 1
        )
        greater_write = greater & (greater_pos < topk)
        equal_write = (
            equal & (equal_pos < topk) & (equal_pos < count_greater + keep_equal)
        )
        local = offsets.to(gl.int32)
        gl.store(out + row * out_stride + greater_pos, local, mask=greater_write)
        gl.store(out + row * out_stride + equal_pos, local, mask=equal_write)


@gluon.jit
def _dsa_prefill_radix_group_count_kernel(
    logits,
    prefixes,
    row_starts,
    row_ends,
    group_offsets,
    logits_stride: gl.constexpr,
    n_cols: gl.constexpr,
    tiles_per_row: gl.constexpr,
    groups_per_row: gl.constexpr,
    row_groups_stride: gl.constexpr,
    group_stride: gl.constexpr,
    topk: gl.constexpr,
    HIST_DERIVED_FIXUP_ONLY: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    count_layout: gl.constexpr = _vector_layout(2, gl.num_warps(), 1)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_logits = logits + row * logits_stride + row_start
    candidate_len = gl.maximum(row_end - row_start, 0)
    if candidate_len <= topk:
        base = (row * row_groups_stride + group) * group_stride
        gl.store(group_offsets + base, 0)
        gl.store(group_offsets + base + 1, 0)
        return
    vector_end = candidate_len & -4
    threshold = gl.load(prefixes + row).to(gl.uint32)
    if HIST_DERIVED_FIXUP_ONLY:
        if (row_start == 0) & (threshold != 0x007FFFFF):
            return
    counts = gl.zeros([2], gl.int32, layout=count_layout)

    for tile in range(group, tiles_per_row, groups_per_row):
        tile_start = tile * BLOCK_N
        offsets = tile_start + gl.arange(0, BLOCK_N, layout=layout)
        offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
        mask = offsets < candidate_len
        if tile_start + BLOCK_N <= candidate_len:
            values = gl.amd.cdna4.buffer_load(
                ptr=candidate_logits,
                offsets=offsets,
            )
        else:
            vector_mask = gl.max_constancy(offsets < vector_end, 4)
            vector_values = gl.amd.cdna4.buffer_load(
                ptr=candidate_logits,
                offsets=offsets,
                mask=vector_mask,
                other=-float("inf"),
            )
            tail_mask = (offsets >= vector_end) & mask
            tail_values = gl.load(
                candidate_logits + offsets,
                mask=tail_mask,
                other=-float("inf"),
            )
            values = gl.where(tail_mask, tail_values, vector_values)
        keys = _fp32_to_ordered_key(values)
        greater = mask & (keys > threshold)
        equal = mask & (keys == threshold)
        count_bin = gl.where(greater, 0, 1).to(gl.int32)
        counts += gl.histogram(
            count_bin,
            2,
            mask=greater | equal,
            layout=count_layout,
        )

    base = (row * row_groups_stride + group) * group_stride
    count_offsets = gl.arange(0, 2, layout=count_layout)
    gl.store(group_offsets + base + count_offsets, counts)


@gluon.jit
def _dsa_radix_group_prefix_kernel(
    group_offsets,
    groups_per_row: gl.constexpr,
    row_groups_stride: gl.constexpr,
    group_stride: gl.constexpr,
    BLOCK_GROUPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_GROUPS, gl.num_warps(), LOAD_ELEMS)
    group = gl.arange(0, BLOCK_GROUPS, layout=layout)
    mask = group < groups_per_row
    base = row * row_groups_stride * group_stride
    greater = gl.load(
        group_offsets + base + group * group_stride,
        mask=mask,
        other=0,
    )
    equal = gl.load(
        group_offsets + base + group * group_stride + 1,
        mask=mask,
        other=0,
    )
    greater_offset = gl.associative_scan(greater, 0, _topk_add) - greater
    equal_offset = gl.associative_scan(equal, 0, _topk_add) - equal
    gl.store(
        group_offsets + base + group * group_stride,
        greater_offset,
        mask=mask,
    )
    gl.store(
        group_offsets + base + group * group_stride + 1,
        equal_offset,
        mask=mask,
    )


@gluon.jit
def _dsa_prefill_radix_deterministic_scatter_kernel(
    logits,
    prefixes,
    remaining,
    group_offsets,
    row_starts,
    row_ends,
    out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    n_cols: gl.constexpr,
    topk: gl.constexpr,
    tiles_per_row: gl.constexpr,
    groups_per_row: gl.constexpr,
    row_groups_stride: gl.constexpr,
    group_stride: gl.constexpr,
    LOCAL_GROUP_PREFIX: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    BLOCK_GROUPS: gl.constexpr,
    GROUP_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    group_layout: gl.constexpr = _vector_layout(
        BLOCK_GROUPS, gl.num_warps(), GROUP_LOAD_ELEMS
    )
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_logits = logits + row * logits_stride + row_start
    candidate_len = gl.maximum(row_end - row_start, 0)
    if candidate_len <= topk:
        return
    vector_end = candidate_len & -4
    threshold = gl.load(prefixes + row).to(gl.uint32)
    keep_equal = gl.load(remaining + row).to(gl.int32)
    count_greater = topk - keep_equal
    if LOCAL_GROUP_PREFIX:
        group_ids = gl.arange(0, BLOCK_GROUPS, layout=group_layout)
        prior_group = group_ids < group
        row_group_base = row * row_groups_stride * group_stride
        greater_counts = gl.load(
            group_offsets + row_group_base + group_ids * group_stride,
            mask=(group_ids < groups_per_row) & prior_group,
            other=0,
        )
        equal_counts = gl.load(
            group_offsets + row_group_base + group_ids * group_stride + 1,
            mask=(group_ids < groups_per_row) & prior_group,
            other=0,
        )
        greater_base = gl.sum(greater_counts, axis=0).to(gl.int32)
        equal_base = gl.sum(equal_counts, axis=0).to(gl.int32)
    else:
        group_base = (row * row_groups_stride + group) * group_stride
        greater_base = gl.load(group_offsets + group_base).to(gl.int32)
        equal_base = gl.load(group_offsets + group_base + 1).to(gl.int32)
    greater_cursor = 0
    equal_cursor = 0

    for tile in range(group, tiles_per_row, groups_per_row):
        tile_start = tile * BLOCK_N
        offsets = tile_start + gl.arange(0, BLOCK_N, layout=layout)
        offsets = gl.max_contiguous(gl.multiple_of(offsets.to(gl.int32), 4), 4)
        mask = offsets < candidate_len
        if tile_start + BLOCK_N <= candidate_len:
            values = gl.amd.cdna4.buffer_load(
                ptr=candidate_logits,
                offsets=offsets,
            )
        else:
            vector_mask = gl.max_constancy(offsets < vector_end, 4)
            vector_values = gl.amd.cdna4.buffer_load(
                ptr=candidate_logits,
                offsets=offsets,
                mask=vector_mask,
                other=-float("inf"),
            )
            tail_mask = (offsets >= vector_end) & mask
            tail_values = gl.load(
                candidate_logits + offsets,
                mask=tail_mask,
                other=-float("inf"),
            )
            values = gl.where(tail_mask, tail_values, vector_values)
        keys = _fp32_to_ordered_key(values)
        greater = mask & (keys > threshold)
        equal = mask & (keys == threshold)

        greater_i32 = greater.to(gl.int32)
        equal_i32 = equal.to(gl.int32)
        tile_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
        tile_equal = gl.sum(equal_i32, axis=0).to(gl.int32)
        greater_pos = (
            greater_base
            + greater_cursor
            + gl.associative_scan(greater_i32, 0, _topk_add)
            - 1
        )
        equal_pos = (
            count_greater
            + equal_base
            + equal_cursor
            + gl.associative_scan(equal_i32, 0, _topk_add)
            - 1
        )
        greater_write = greater & (greater_pos < topk)
        equal_write = (
            equal & (equal_pos < topk) & (equal_pos < count_greater + keep_equal)
        )
        local = row_start + offsets.to(gl.int32)
        gl.store(out + row * out_stride + greater_pos, local, mask=greater_write)
        gl.store(out + row * out_stride + equal_pos, local, mask=equal_write)
        greater_cursor += tile_greater
        equal_cursor += tile_equal


def _load_elems(block: int, num_warps: int) -> int:
    return max(1, triton.cdiv(int(block), 64 * int(num_warps)))


def _contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _to_contiguous(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.to(device=device, dtype=dtype)
    return _contiguous(tensor)


def _validate_topk(topk: int) -> None:
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk & (topk - 1):
        raise ValueError(f"DSA Gluon top-k requires power-of-two topk, got {topk}")


def _use_radix_topk(cols: int) -> bool:
    return int(cols) >= _RADIX_TOPK_MIN_COLS


@cache
def _device_compute_units(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _persistent_prefill_groups(
    rows: int,
    cols: int,
    topk: int,
    device: torch.device,
) -> int | None:
    if (
        rows < _PERSISTENT_PREFILL_MIN_ROWS
        or topk != _PERSISTENT_PREFILL_TOPK
        or cols < _PERSISTENT_PREFILL_MIN_COLS
        or cols > _PERSISTENT_PREFILL_MAX_COLS
    ):
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    max_groups = _device_compute_units(device_index) // rows
    target_groups = 2 if cols < _PERSISTENT_PREFILL_FOUR_GROUP_MIN_COLS else 4
    if max_groups >= target_groups:
        return target_groups
    if target_groups == 4 and max_groups >= 2:
        return 2
    return None


def _persistent_decode_groups(
    rows: int,
    cols: int,
    topk: int,
    device: torch.device,
) -> int | None:
    if (
        rows <= 0
        or topk != _PERSISTENT_PREFILL_TOPK
        or cols <= _PERSISTENT_DECODE_MIN_COLS
        or cols > _PERSISTENT_DECODE_MAX_COLS
    ):
        return None
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    groups = min(
        _PERSISTENT_DECODE_MAX_GROUPS,
        _device_compute_units(device_index) // rows,
    )
    return groups if groups >= 2 else None


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _persistent_topk_workspace(
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Return zero-once scratch owned by the current stream and size bucket."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
    row_bucket = _next_power_of_two(rows)
    key = (device_index, stream_id, row_bucket)
    capturing = torch.cuda.is_current_stream_capturing()
    with _persistent_topk_workspace_lock:
        workspace = _persistent_topk_workspace_cache.get(key)
        if workspace is not None:
            _persistent_topk_workspace_cache.move_to_end(key)
            if capturing:
                _persistent_topk_graph_workspace_keys.add(key)
    if workspace is not None:
        return workspace

    # Retaining a stream-local allocation keeps graph pointers alive and avoids
    # cross-stream races. The kernel restores every touched word to zero.
    workspace = (
        torch.zeros(
            (
                row_bucket,
                _PERSISTENT_PREFILL_NUM_PASSES,
                _PERSISTENT_PREFILL_NUM_BUCKETS,
            ),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (
                row_bucket,
                _PERSISTENT_PREFILL_NUM_PASSES,
                _PERSISTENT_PREFILL_COUNTER_STRIDE,
            ),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (row_bucket, _PERSISTENT_PREFILL_COUNTER_STRIDE),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (row_bucket, _PERSISTENT_PREFILL_COUNTER_STRIDE),
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(
            (row_bucket, 2, _PERSISTENT_PREFILL_COUNTER_STRIDE),
            dtype=torch.int32,
            device=device,
        ),
    )
    with _persistent_topk_workspace_lock:
        existing = _persistent_topk_workspace_cache.get(key)
        if existing is not None:
            _persistent_topk_workspace_cache.move_to_end(key)
            if capturing:
                _persistent_topk_graph_workspace_keys.add(key)
            return existing

        if len(_persistent_topk_workspace_cache) >= (
            _PERSISTENT_PREFILL_WORKSPACE_CACHE_MAXSIZE
        ):
            evict_key = next(
                (
                    cached_key
                    for cached_key in _persistent_topk_workspace_cache
                    if cached_key not in _persistent_topk_graph_workspace_keys
                ),
                None,
            )
            if evict_key is None:
                # Allocations made during capture belong to the graph's memory
                # pool, so an uncached workspace remains valid for replay.
                return workspace
            del _persistent_topk_workspace_cache[evict_key]

        _persistent_topk_workspace_cache[key] = workspace
        if capturing:
            _persistent_topk_graph_workspace_keys.add(key)
        return workspace


def _dsa_persistent_radix_topk(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    page_size: int,
    q_len_per_req: int,
    is_decode: bool,
    topk: int,
    groups: int,
    workspace: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    histograms, arrivals, pass_done, reset_arrivals, output_counters = workspace
    block_table_stride = block_table.stride(0) if is_decode else 0
    block_table_cols = block_table.shape[1] if is_decode else 0
    kernel_args = (
        logits,
        histograms,
        arrivals,
        pass_done,
        reset_arrivals,
        output_counters,
        block_table,
        row_starts,
        row_ends,
        out,
        lens_out,
        logits.stride(0),
        block_table_stride,
        out.stride(0),
        block_table_cols,
        cols,
        page_size,
        q_len_per_req,
        is_decode,
        groups,
        topk,
        _PERSISTENT_PREFILL_BLOCK_N,
        _PERSISTENT_PREFILL_NUM_BUCKETS,
        _PERSISTENT_PREFILL_NUM_PASSES,
        _PERSISTENT_PREFILL_COUNTER_STRIDE,
    )
    specialization_key = kernel_args[16:]
    grid = (rows, groups, 1)
    dispatch_key = (grid, specialization_key)
    compiled_plan = _get_cached_compiled_runner_plan(
        _persistent_runner_plans,
        dispatch_key,
    )
    driver = triton.runtime.driver.active
    if (
        compiled_plan is not None
        and compiled_plan.pointers_match(kernel_args)
        and _compiled_runner_plan_state_matches(
            compiled_plan,
            _dsa_persistent_radix_topk_kernel,
            driver,
        )
    ):
        compiled_plan.runner(*kernel_args)
    else:
        _launch_warmed_compiled_kernel(
            _dsa_persistent_radix_topk_kernel,
            grid,
            kernel_args,
            _PERSISTENT_RADIX_POINTER_DTYPES,
            specialization_key,
            dispatch_cache=_persistent_runner_plans,
            dispatch_key=dispatch_key,
            native_scalar_count=5,
            num_warps=_PERSISTENT_PREFILL_NUM_WARPS,
        )
    return out, lens_out


def _dsa_persistent_prefill_radix_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    groups: int,
    workspace: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _dsa_persistent_radix_topk(
        logits,
        row_starts,
        row_starts,
        row_ends,
        page_size=1,
        q_len_per_req=1,
        is_decode=False,
        topk=topk,
        groups=groups,
        workspace=workspace,
        out=out,
        lens_out=lens_out,
    )


def _dsa_persistent_decode_topk_slots(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    q_len_per_req: int,
    groups: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    workspace = _persistent_topk_workspace(logits.shape[0], logits.device)
    return _dsa_persistent_radix_topk(
        logits,
        block_table,
        seq_lens,
        seq_lens,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        is_decode=True,
        topk=topk,
        groups=groups,
        workspace=workspace,
        out=out,
        lens_out=lens_out,
    )


def _radix_groups_per_row(
    rows: int,
    tiles: int,
    device: torch.device,
    *,
    target_groups_per_cu: int | None = None,
) -> int:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compute_units = _device_compute_units(device_index)
    if target_groups_per_cu is None:
        target_groups_per_cu = _RADIX_TOPK_TARGET_GROUPS_PER_CU
    target_groups = compute_units * int(target_groups_per_cu)
    return min(int(tiles), max(1, triton.cdiv(target_groups, int(rows))))


def _dsa_radix_scratch(
    rows: int,
    cols: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    tiles = triton.cdiv(int(cols), _RADIX_TOPK_BLOCK_N)
    groups = _radix_groups_per_row(rows, tiles, device)
    hist = torch.empty((rows, groups, 16), dtype=torch.int32, device=device)
    prefixes = torch.empty((rows,), dtype=torch.int32, device=device)
    remaining = torch.empty((rows,), dtype=torch.int32, device=device)
    counters = torch.empty((rows, 2), dtype=torch.int32, device=device)
    block_groups = triton.next_power_of_2(groups)
    return hist, prefixes, remaining, counters, tiles, groups, block_groups


def _run_radix_prefix_passes(
    logits: torch.Tensor,
    hist: torch.Tensor,
    prefixes: torch.Tensor,
    remaining: torch.Tensor,
    *,
    rows: int,
    cols: int,
    tiles: int,
    groups: int,
    block_groups: int,
) -> None:
    hist_load_elems = _load_elems(_RADIX_TOPK_BLOCK_N, 8)
    update_load_elems = _load_elems(block_groups, 8)
    # Ordered FP32 keys have 8 nibbles; each pass fixes one more prefix nibble.
    for shift in range(28, -1, -4):
        if groups == tiles:
            _dsa_radix_hist_kernel[(rows, tiles)](
                logits,
                prefixes,
                hist,
                logits.stride(0),
                tiles,
                n_cols=cols,
                shift=shift,
                BLOCK_N=_RADIX_TOPK_BLOCK_N,
                LOAD_ELEMS=hist_load_elems,
                num_warps=8,
            )
        else:
            _dsa_radix_grouped_hist_kernel[(rows, groups)](
                logits,
                prefixes,
                hist,
                logits.stride(0),
                tiles,
                groups,
                n_cols=cols,
                shift=shift,
                BLOCK_N=_RADIX_TOPK_BLOCK_N,
                LOAD_ELEMS=hist_load_elems,
                num_warps=8,
            )
        _dsa_radix_update_kernel[(rows,)](
            prefixes,
            remaining,
            hist,
            hist_groups=groups,
            BLOCK_GROUPS=block_groups,
            LOAD_ELEMS=update_load_elems,
            num_warps=8,
        )


def _run_prefill_wide_radix_prefix_passes(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    hist: torch.Tensor,
    prefixes: torch.Tensor,
    remaining: torch.Tensor,
    *,
    rows: int,
    cols: int,
    tiles: int,
    groups: int,
    row_groups_stride: int,
    hist_group_stride: int,
    block_n: int,
    topk: int,
    q_len_per_req: int,
    is_decode: bool,
) -> None:
    for shift, radix_bits in _PREFILL_RADIX_SCHEDULE:
        num_buckets = 1 << radix_bits
        _dsa_prefill_wide_radix_grouped_hist_kernel[(rows, groups)](
            logits,
            prefixes,
            row_starts,
            row_ends,
            hist,
            logits.stride(0),
            tiles_per_row=tiles,
            groups_per_row=groups,
            row_groups_stride=row_groups_stride,
            hist_group_stride=hist_group_stride,
            n_cols=cols,
            shift=shift,
            topk=topk,
            q_len_per_req=q_len_per_req,
            IS_DECODE=is_decode,
            RADIX_BITS=radix_bits,
            NUM_BUCKETS=num_buckets,
            BLOCK_N=block_n,
            LOAD_ELEMS=_load_elems(block_n, 8),
            num_warps=8,
        )
        _dsa_prefill_wide_radix_update_kernel[(rows,)](
            prefixes,
            remaining,
            hist,
            groups_per_row=groups,
            row_groups_stride=row_groups_stride,
            hist_group_stride=hist_group_stride,
            RADIX_BITS=radix_bits,
            NUM_BUCKETS=num_buckets,
            LOAD_ELEMS=_load_elems(num_buckets, 8),
            num_warps=8,
        )


def _dsa_decode_radix_topk_slots(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    q_len_per_req: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    hist, prefixes, remaining, counters, tiles, groups, block_groups = (
        _dsa_radix_scratch(rows, cols, device=logits.device)
    )
    _dsa_decode_radix_init_kernel[(rows,)](
        seq_lens,
        out,
        lens_out,
        prefixes,
        remaining,
        counters,
        out.stride(0),
        topk=topk,
        q_len_per_req=q_len_per_req,
        TOPK_LOAD_ELEMS=_load_elems(topk, 8),
        num_warps=8,
    )
    _run_radix_prefix_passes(
        logits,
        hist,
        prefixes,
        remaining,
        rows=rows,
        cols=cols,
        tiles=tiles,
        groups=groups,
        block_groups=block_groups,
    )
    _dsa_decode_radix_scatter_slots_kernel[(rows, groups)](
        logits,
        prefixes,
        remaining,
        counters,
        block_table,
        seq_lens,
        out,
        logits.stride(0),
        block_table.stride(0),
        out.stride(0),
        block_table.shape[1],
        n_cols=cols,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        tiles_per_row=tiles,
        groups_per_row=groups,
        BLOCK_N=_RADIX_TOPK_BLOCK_N,
        LOAD_ELEMS=_load_elems(_RADIX_TOPK_BLOCK_N, 8),
        num_warps=8,
    )
    return out, lens_out


def _dsa_prefill_radix_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    tiles = triton.cdiv(int(cols), _RADIX_TOPK_BLOCK_N)
    groups = _radix_groups_per_row(
        rows,
        tiles,
        logits.device,
        target_groups_per_cu=_PREFILL_RADIX_SCATTER_TARGET_GROUPS_PER_CU,
    )
    hist_tiles = triton.cdiv(int(cols), _PREFILL_RADIX_BLOCK_N)
    hist_groups = _radix_groups_per_row(
        rows,
        hist_tiles,
        logits.device,
        target_groups_per_cu=_PREFILL_RADIX_HIST_TARGET_GROUPS_PER_CU,
    )
    # The histogram and scatter grids share one scratch buffer with a common row stride.
    workspace_groups = max(groups, hist_groups)
    block_groups = triton.next_power_of_2(groups)
    use_local_group_prefix = (
        cols >= _PREFILL_LOCAL_GROUP_PREFIX_MIN_COLS and groups <= 8
    )
    hist_group_stride = _PREFILL_RADIX_BUCKETS
    hist = torch.empty(
        (rows, workspace_groups, hist_group_stride),
        dtype=torch.int32,
        device=logits.device,
    )
    prefixes = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    remaining = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    counters = torch.empty((rows, 2), dtype=torch.int32, device=logits.device)
    _dsa_prefill_radix_init_kernel[(rows,)](
        row_starts,
        row_ends,
        out,
        lens_out,
        prefixes,
        remaining,
        counters,
        out.stride(0),
        topk=topk,
        TOPK_LOAD_ELEMS=_load_elems(topk, 8),
        num_warps=8,
    )
    _run_prefill_wide_radix_prefix_passes(
        logits,
        row_starts,
        row_ends,
        hist,
        prefixes,
        remaining,
        rows=rows,
        cols=cols,
        tiles=hist_tiles,
        groups=hist_groups,
        row_groups_stride=workspace_groups,
        hist_group_stride=hist_group_stride,
        block_n=_PREFILL_RADIX_BLOCK_N,
        topk=topk,
        q_len_per_req=1,
        is_decode=False,
    )
    if groups < tiles:
        _dsa_prefill_radix_group_count_kernel[(rows, groups)](
            logits,
            prefixes,
            row_starts,
            row_ends,
            hist,
            logits.stride(0),
            n_cols=cols,
            tiles_per_row=tiles,
            groups_per_row=groups,
            row_groups_stride=workspace_groups,
            group_stride=hist_group_stride,
            topk=topk,
            HIST_DERIVED_FIXUP_ONLY=False,
            BLOCK_N=_RADIX_TOPK_BLOCK_N,
            LOAD_ELEMS=_load_elems(_RADIX_TOPK_BLOCK_N, 8),
            num_warps=8,
        )
        if not use_local_group_prefix:
            _dsa_radix_group_prefix_kernel[(rows,)](
                hist,
                groups_per_row=groups,
                row_groups_stride=workspace_groups,
                group_stride=hist_group_stride,
                BLOCK_GROUPS=block_groups,
                LOAD_ELEMS=_load_elems(block_groups, 1),
                num_warps=1,
            )
        _dsa_prefill_radix_deterministic_scatter_kernel[(rows, groups)](
            logits,
            prefixes,
            remaining,
            hist,
            row_starts,
            row_ends,
            out,
            logits.stride(0),
            out.stride(0),
            n_cols=cols,
            topk=topk,
            tiles_per_row=tiles,
            groups_per_row=groups,
            row_groups_stride=workspace_groups,
            group_stride=hist_group_stride,
            LOCAL_GROUP_PREFIX=use_local_group_prefix,
            BLOCK_N=_RADIX_TOPK_BLOCK_N,
            LOAD_ELEMS=_load_elems(_RADIX_TOPK_BLOCK_N, 8),
            BLOCK_GROUPS=block_groups,
            GROUP_LOAD_ELEMS=_load_elems(block_groups, 8),
            num_warps=8,
        )
    else:
        _dsa_prefill_radix_scatter_kernel[(rows, groups)](
            logits,
            prefixes,
            remaining,
            counters,
            row_starts,
            row_ends,
            out,
            logits.stride(0),
            out.stride(0),
            n_cols=cols,
            topk=topk,
            tiles_per_row=tiles,
            groups_per_row=groups,
            BLOCK_N=_RADIX_TOPK_BLOCK_N,
            LOAD_ELEMS=_load_elems(_RADIX_TOPK_BLOCK_N, 8),
            num_warps=8,
        )
    return out, lens_out


def _dsa_prefill_hist_derived_radix_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    block_n = _RADIX_TOPK_BLOCK_N
    tiles = triton.cdiv(int(cols), block_n)
    groups = _radix_groups_per_row(
        rows,
        tiles,
        logits.device,
        target_groups_per_cu=_PREFILL_RADIX_SCATTER_TARGET_GROUPS_PER_CU,
    )
    block_groups = triton.next_power_of_2(groups)
    use_local_group_prefix = (
        cols >= _PREFILL_LOCAL_GROUP_PREFIX_MIN_COLS and groups <= 8
    )
    hist_group_stride = _PREFILL_RADIX_BUCKETS
    hist = torch.empty(
        (rows, groups, hist_group_stride),
        dtype=torch.int32,
        device=logits.device,
    )
    group_offsets = torch.empty(
        (rows, groups, 2), dtype=torch.int32, device=logits.device
    )
    prefixes = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    remaining = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    counters = torch.empty((rows, 2), dtype=torch.int32, device=logits.device)

    _dsa_prefill_radix_init_kernel[(rows,)](
        row_starts,
        row_ends,
        out,
        lens_out,
        prefixes,
        remaining,
        counters,
        out.stride(0),
        topk=topk,
        TOPK_LOAD_ELEMS=_load_elems(topk, 8),
        num_warps=8,
    )
    for pass_index, (shift, radix_bits) in enumerate(_PREFILL_RADIX_SCHEDULE):
        num_buckets = 1 << radix_bits
        _dsa_prefill_wide_radix_grouped_hist_kernel[(rows, groups)](
            logits,
            prefixes,
            row_starts,
            row_ends,
            hist,
            logits.stride(0),
            tiles_per_row=tiles,
            groups_per_row=groups,
            row_groups_stride=groups,
            hist_group_stride=hist_group_stride,
            n_cols=cols,
            shift=shift,
            topk=topk,
            q_len_per_req=1,
            IS_DECODE=False,
            RADIX_BITS=radix_bits,
            NUM_BUCKETS=num_buckets,
            BLOCK_N=block_n,
            LOAD_ELEMS=_load_elems(block_n, 8),
            num_warps=8,
        )
        _dsa_prefill_wide_radix_update_kernel[(rows,)](
            prefixes,
            remaining,
            hist,
            groups_per_row=groups,
            row_groups_stride=groups,
            hist_group_stride=hist_group_stride,
            RADIX_BITS=radix_bits,
            NUM_BUCKETS=num_buckets,
            LOAD_ELEMS=_load_elems(num_buckets, 8),
            num_warps=8,
        )
        _dsa_prefill_derive_group_counts_from_hist_kernel[(rows, groups)](
            prefixes,
            hist,
            group_offsets,
            groups_per_row=groups,
            hist_group_stride=hist_group_stride,
            RADIX_BITS=radix_bits,
            NUM_BUCKETS=num_buckets,
            PASS_INDEX=pass_index,
            LOAD_ELEMS=_load_elems(num_buckets, 8),
            num_warps=8,
        )

    _dsa_prefill_radix_group_count_kernel[(rows, groups)](
        logits,
        prefixes,
        row_starts,
        row_ends,
        group_offsets,
        logits.stride(0),
        n_cols=cols,
        tiles_per_row=tiles,
        groups_per_row=groups,
        row_groups_stride=groups,
        group_stride=2,
        topk=topk,
        HIST_DERIVED_FIXUP_ONLY=True,
        BLOCK_N=block_n,
        LOAD_ELEMS=_load_elems(block_n, 8),
        num_warps=8,
    )
    if not use_local_group_prefix:
        _dsa_radix_group_prefix_kernel[(rows,)](
            group_offsets,
            groups_per_row=groups,
            row_groups_stride=groups,
            group_stride=2,
            BLOCK_GROUPS=block_groups,
            LOAD_ELEMS=_load_elems(block_groups, 1),
            num_warps=1,
        )
    _dsa_prefill_radix_deterministic_scatter_kernel[(rows, groups)](
        logits,
        prefixes,
        remaining,
        group_offsets,
        row_starts,
        row_ends,
        out,
        logits.stride(0),
        out.stride(0),
        n_cols=cols,
        topk=topk,
        tiles_per_row=tiles,
        groups_per_row=groups,
        row_groups_stride=groups,
        group_stride=2,
        LOCAL_GROUP_PREFIX=use_local_group_prefix,
        BLOCK_N=block_n,
        LOAD_ELEMS=_load_elems(block_n, 8),
        BLOCK_GROUPS=block_groups,
        GROUP_LOAD_ELEMS=_load_elems(block_groups, 8),
        num_warps=8,
    )
    return out, lens_out


def _dsa_decode_staged_wide_radix_topk_slots(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    q_len_per_req: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    hist_tiles = triton.cdiv(int(cols), _PREFILL_RADIX_BLOCK_N)
    hist_groups = _radix_groups_per_row(
        rows,
        hist_tiles,
        logits.device,
        target_groups_per_cu=_PREFILL_RADIX_HIST_TARGET_GROUPS_PER_CU,
    )
    scatter_tiles = triton.cdiv(int(cols), _RADIX_TOPK_BLOCK_N)
    scatter_groups = _radix_groups_per_row(
        rows,
        scatter_tiles,
        logits.device,
    )
    workspace_groups = max(hist_groups, scatter_groups)
    hist_group_stride = _PREFILL_RADIX_BUCKETS
    hist = torch.empty(
        (rows, workspace_groups, hist_group_stride),
        dtype=torch.int32,
        device=logits.device,
    )
    prefixes = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    remaining = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    counters = torch.empty((rows, 2), dtype=torch.int32, device=logits.device)

    _dsa_decode_radix_init_kernel[(rows,)](
        seq_lens,
        out,
        lens_out,
        prefixes,
        remaining,
        counters,
        out.stride(0),
        topk=topk,
        q_len_per_req=q_len_per_req,
        TOPK_LOAD_ELEMS=_load_elems(topk, 8),
        num_warps=8,
    )
    _run_prefill_wide_radix_prefix_passes(
        logits,
        seq_lens,
        seq_lens,
        hist,
        prefixes,
        remaining,
        rows=rows,
        cols=cols,
        tiles=hist_tiles,
        groups=hist_groups,
        row_groups_stride=workspace_groups,
        hist_group_stride=hist_group_stride,
        block_n=_PREFILL_RADIX_BLOCK_N,
        topk=topk,
        q_len_per_req=q_len_per_req,
        is_decode=True,
    )
    _dsa_decode_radix_scatter_slots_kernel[(rows, scatter_groups)](
        logits,
        prefixes,
        remaining,
        counters,
        block_table,
        seq_lens,
        out,
        logits.stride(0),
        block_table.stride(0),
        out.stride(0),
        block_table.shape[1],
        n_cols=cols,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        tiles_per_row=scatter_tiles,
        groups_per_row=scatter_groups,
        BLOCK_N=_RADIX_TOPK_BLOCK_N,
        LOAD_ELEMS=_load_elems(_RADIX_TOPK_BLOCK_N, 8),
        num_warps=8,
    )
    return out, lens_out


def _dsa_decode_topk_slots(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    q_len_per_req: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    if cols <= topk:
        if topk == 2048 and page_size == 64:
            kernel_args = (
                block_table,
                seq_lens,
                out,
                lens_out,
                block_table.stride(0),
                out.stride(0),
                block_table.shape[1],
                int(page_size),
                q_len_per_req,
            )
            specialization_key = kernel_args[7:]
            dispatch_key = (rows, specialization_key)
            compiled_plan = _get_cached_compiled_runner_plan(
                _trivial_decode_runner_plans,
                dispatch_key,
            )
            driver = triton.runtime.driver.active
            if (
                compiled_plan is not None
                and compiled_plan.pointers_match(kernel_args)
                and _compiled_runner_plan_state_matches(
                    compiled_plan,
                    _dsa_trivial_decode_topk2048_kernel,
                    driver,
                )
            ):
                compiled_plan.runner(*kernel_args)
            else:
                _launch_warmed_compiled_kernel(
                    _dsa_trivial_decode_topk2048_kernel,
                    (rows, 1, 1),
                    kernel_args,
                    _TRIVIAL_DECODE_POINTER_DTYPES,
                    specialization_key,
                    dispatch_cache=_trivial_decode_runner_plans,
                    dispatch_key=dispatch_key,
                    native_scalar_count=3,
                    num_warps=8,
                )
        else:
            _dsa_trivial_topk_kernel[(rows,)](
                block_table,
                seq_lens,
                seq_lens,
                seq_lens,
                out,
                lens_out,
                block_table.stride(0),
                out.stride(0),
                block_table.shape[1],
                page_size=int(page_size),
                topk=topk,
                q_len_per_req=q_len_per_req,
                IS_DECODE=True,
                TOPK_LOAD_ELEMS=_load_elems(topk, 8),
                num_warps=8,
            )
        return out, lens_out

    persistent_groups = _persistent_decode_groups(
        rows,
        cols,
        topk,
        logits.device,
    )
    if persistent_groups is not None:
        return _dsa_persistent_decode_topk_slots(
            logits,
            block_table,
            seq_lens,
            page_size=page_size,
            topk=topk,
            q_len_per_req=q_len_per_req,
            groups=persistent_groups,
            out=out,
            lens_out=lens_out,
        )

    if cols <= _ONEBLOCK_DECODE_RUNTIME_MAX_COLS:
        if (
            cols < _ONEBLOCK_DECODE_EARLY_STOP_MIN_COLS
            or cols > _ONEBLOCK_RADIX_MAX_COLS
        ):
            load_elems = (
                _ONEBLOCK_DECODE_SHORT_LOAD_ELEMS
                if cols < _ONEBLOCK_DECODE_EARLY_STOP_MIN_COLS
                else _ONEBLOCK_DECODE_LONG_LOAD_ELEMS
            )
            kernel_args = (
                logits,
                block_table,
                seq_lens,
                seq_lens,
                out,
                lens_out,
                logits.stride(0),
                block_table.stride(0),
                out.stride(0),
                block_table.shape[1],
                int(page_size),
                topk,
                q_len_per_req,
                True,
                False,
                _ONEBLOCK_RADIX_BUCKETS,
                _ONEBLOCK_DECODE_RADIX_BLOCK_N,
                load_elems,
            )
            specialization_key = kernel_args[10:]
            dispatch_key = (rows, specialization_key)
            compiled_plan = _get_cached_compiled_runner_plan(
                _runtime_decode_runner_plans,
                dispatch_key,
            )
            driver = triton.runtime.driver.active
            if (
                compiled_plan is not None
                and compiled_plan.pointers_match(kernel_args)
                and _compiled_runner_plan_state_matches(
                    compiled_plan,
                    _dsa_runtime_radix_topk_kernel,
                    driver,
                )
            ):
                compiled_plan.runner(*kernel_args)
            else:
                _launch_warmed_compiled_kernel(
                    _dsa_runtime_radix_topk_kernel,
                    (rows, 1, 1),
                    kernel_args,
                    _RUNTIME_RADIX_POINTER_DTYPES,
                    specialization_key,
                    dispatch_cache=_runtime_decode_runner_plans,
                    dispatch_key=dispatch_key,
                    native_scalar_count=4,
                    num_warps=16,
                )
        else:
            kernel_args = (
                logits,
                block_table,
                seq_lens,
                seq_lens,
                seq_lens,
                out,
                lens_out,
                logits.stride(0),
                block_table.stride(0),
                out.stride(0),
                block_table.shape[1],
                int(page_size),
                topk,
                q_len_per_req,
                True,
                _ONEBLOCK_RADIX_SCHEDULE[0],
                _ONEBLOCK_RADIX_SCHEDULE[1],
                _ONEBLOCK_RADIX_SCHEDULE[2],
                _ONEBLOCK_RADIX_BUCKETS,
                _ONEBLOCK_DECODE_RADIX_BLOCK_N,
                _ONEBLOCK_COMPACT_FINAL_BLOCK_N,
                False,
                True,
            )
            specialization_key = kernel_args[11:]
            _launch_warmed_compiled_kernel(
                _dsa_oneblock_manual_radix_topk_kernel,
                (rows, 1, 1),
                kernel_args,
                _MANUAL_RADIX_POINTER_DTYPES,
                specialization_key,
                dispatch_cache=_manual_decode_runner_plans,
                dispatch_key=(rows, specialization_key),
                native_scalar_count=4,
                num_warps=16,
            )
        return out, lens_out

    return _dsa_decode_staged_wide_radix_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=page_size,
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )


def _dsa_prefill_topk_indices(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    if cols <= topk:
        if topk == 2048:
            kernel_args = (
                row_starts,
                row_ends,
                out,
                lens_out,
                out.stride(0),
            )
            specialization_key = ()
            dispatch_key = (rows, specialization_key)
            compiled_plan = _get_cached_compiled_runner_plan(
                _trivial_prefill_runner_plans,
                dispatch_key,
            )
            driver = triton.runtime.driver.active
            if (
                compiled_plan is not None
                and compiled_plan.pointers_match(kernel_args)
                and _compiled_runner_plan_state_matches(
                    compiled_plan,
                    _dsa_trivial_prefill_topk2048_kernel,
                    driver,
                )
            ):
                compiled_plan.runner(*kernel_args)
            else:
                _launch_warmed_compiled_kernel(
                    _dsa_trivial_prefill_topk2048_kernel,
                    (rows, 1, 1),
                    kernel_args,
                    _TRIVIAL_DECODE_POINTER_DTYPES,
                    specialization_key,
                    dispatch_cache=_trivial_prefill_runner_plans,
                    dispatch_key=dispatch_key,
                    native_scalar_count=1,
                    num_warps=8,
                )
        else:
            _dsa_trivial_topk_kernel[(rows,)](
                row_starts,
                row_starts,
                row_starts,
                row_ends,
                out,
                lens_out,
                0,
                out.stride(0),
                0,
                page_size=1,
                topk=topk,
                q_len_per_req=1,
                IS_DECODE=False,
                TOPK_LOAD_ELEMS=_load_elems(topk, 8),
                num_warps=8,
            )
        return out, lens_out

    persistent_groups = _persistent_prefill_groups(
        rows,
        cols,
        topk,
        logits.device,
    )
    if persistent_groups is not None:
        workspace = _persistent_topk_workspace(
            rows,
            logits.device,
        )
        return _dsa_persistent_prefill_radix_topk(
            logits,
            row_starts,
            row_ends,
            topk=topk,
            groups=persistent_groups,
            workspace=workspace,
            out=out,
            lens_out=lens_out,
        )

    if cols <= _ONEBLOCK_RADIX_MAX_COLS:
        if cols < _ONEBLOCK_COMPACT_FINAL_MIN_COLS:
            kernel_args = (
                logits,
                row_starts,
                row_starts,
                row_ends,
                out,
                lens_out,
                logits.stride(0),
                0,
                out.stride(0),
                0,
                1,
                topk,
                1,
                False,
                False,
                _ONEBLOCK_RADIX_BUCKETS,
                _ONEBLOCK_PREFILL_RADIX_BLOCK_N,
                _load_elems(_ONEBLOCK_PREFILL_RADIX_BLOCK_N, 16),
            )
            specialization_key = kernel_args[10:]
            _launch_warmed_compiled_kernel(
                _dsa_runtime_radix_topk_kernel,
                (rows, 1, 1),
                kernel_args,
                _RUNTIME_RADIX_POINTER_DTYPES,
                specialization_key,
                dispatch_cache=_runtime_prefill_runner_plans,
                dispatch_key=(rows, specialization_key),
                native_scalar_count=4,
                num_warps=16,
            )
        else:
            kernel_args = (
                logits,
                row_starts,
                row_starts,
                row_starts,
                row_ends,
                out,
                lens_out,
                logits.stride(0),
                0,
                out.stride(0),
                0,
                1,
                topk,
                1,
                False,
                _ONEBLOCK_RADIX_SCHEDULE[0],
                _ONEBLOCK_RADIX_SCHEDULE[1],
                _ONEBLOCK_RADIX_SCHEDULE[2],
                _ONEBLOCK_RADIX_BUCKETS,
                _ONEBLOCK_PREFILL_RADIX_BLOCK_N,
                _ONEBLOCK_COMPACT_FINAL_BLOCK_N,
                True,
                False,
            )
            specialization_key = kernel_args[11:]
            _launch_warmed_compiled_kernel(
                _dsa_oneblock_manual_radix_topk_kernel,
                (rows, 1, 1),
                kernel_args,
                _MANUAL_RADIX_POINTER_DTYPES,
                specialization_key,
                dispatch_cache=_manual_prefill_runner_plans,
                dispatch_key=(rows, specialization_key),
                native_scalar_count=4,
                num_warps=16,
            )
        return out, lens_out

    if _PREFILL_RUNTIME_RADIX_MIN_COLS <= cols <= _PREFILL_RUNTIME_RADIX_MAX_COLS:
        kernel_args = (
            logits,
            row_starts,
            row_starts,
            row_ends,
            out,
            lens_out,
            logits.stride(0),
            0,
            out.stride(0),
            0,
            1,
            topk,
            1,
            False,
            True,
            _ONEBLOCK_RADIX_BUCKETS,
            _ONEBLOCK_PREFILL_RADIX_BLOCK_N,
            _load_elems(_ONEBLOCK_PREFILL_RADIX_BLOCK_N, 16),
        )
        specialization_key = kernel_args[10:]
        _launch_warmed_compiled_kernel(
            _dsa_runtime_radix_topk_kernel,
            (rows, 1, 1),
            kernel_args,
            _RUNTIME_RADIX_POINTER_DTYPES,
            specialization_key,
            dispatch_cache=_runtime_prefill_runner_plans,
            dispatch_key=(rows, specialization_key),
            native_scalar_count=4,
            num_warps=16,
        )
        return out, lens_out

    if cols >= _PREFILL_HIST_DERIVED_MIN_COLS:
        return _dsa_prefill_hist_derived_radix_topk(
            logits,
            row_starts,
            row_ends,
            topk=topk,
            out=out,
            lens_out=lens_out,
        )

    return _dsa_prefill_radix_topk(
        logits,
        row_starts,
        row_ends,
        topk=topk,
        out=out,
        lens_out=lens_out,
    )


def gluon_dsa_decode_topk_fp8_gfx950(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    index_k_cache: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _validate_topk(topk)
    if not 1 <= q_len_per_req <= 6:
        raise ValueError(f"q_len_per_req must be in [1, 6], got {q_len_per_req}")
    if index_k_cache is None:
        raise RuntimeError("Gluon DSA paged top-k requires packed FP8 index_k_cache")
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
    if seq_lens.dim() != 1:
        raise ValueError(
            f"seq_lens must be 1-D, got {tuple(seq_lens.shape)} for q={tuple(q.shape)}"
        )
    expected_tokens = int(seq_lens.numel()) * q_len_per_req
    if expected_tokens != q.shape[0]:
        raise ValueError(
            "q rows must equal seq_lens rows times q_len_per_req, got "
            f"q={tuple(q.shape)}, seq_lens={tuple(seq_lens.shape)}, "
            f"q_len_per_req={q_len_per_req}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < seq_lens.numel():
        raise ValueError(
            "block_table must have at least one row per request, got "
            f"block_table={tuple(block_table.shape)}, q={tuple(q.shape)}"
        )
    if q.shape[0] == 0:
        empty_out = (
            torch.empty((0, int(topk)), dtype=torch.int32, device=q.device)
            if out is None
            else out
        )
        empty_lens = (
            torch.empty((0,), dtype=torch.int32, device=q.device)
            if lens_out is None
            else lens_out
        )
        return empty_out, empty_lens
    q = _contiguous(q)
    index_k_cache = _contiguous(index_k_cache)
    weights = _contiguous(weights)
    seq_lens = _to_contiguous(seq_lens, device=q.device, dtype=torch.int32)
    block_table = _to_contiguous(block_table, device=q.device, dtype=torch.int32)
    max_seq_len = int(block_table.shape[1]) * int(page_size)
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    block_n = 32
    _dsa_decode_logits_fp8_kernel[(q.shape[0], triton.cdiv(max_seq_len, block_n))](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        block_table.stride(0),
        logits.stride(0),
        page_size=int(page_size),
        row_bytes=row_bytes,
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=float(softmax_scale),
        q_len_per_req=q_len_per_req,
        BLOCK_N=block_n,
        BLOCK_D=128,
        num_warps=4,
    )
    return _dsa_decode_topk_slots(
        logits,
        block_table,
        seq_lens,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )


def gluon_dsa_prefill_topk_fp8_gfx950(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _validate_topk(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Gluon DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError(
            "row_starts/row_ends must be [tokens], got "
            f"row_starts={tuple(row_starts.shape)}, row_ends={tuple(row_ends.shape)}, "
            f"q={tuple(q.shape)}"
        )
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out
    q = _contiguous(q)
    index_k_cache = _contiguous(index_k_cache)
    weights = _contiguous(weights)
    kv_workspace_slots = _to_contiguous(
        kv_workspace_slots, device=q.device, dtype=torch.int64
    )
    row_starts = _to_contiguous(row_starts, device=q.device, dtype=torch.int32)
    row_ends = _to_contiguous(row_ends, device=q.device, dtype=torch.int32)
    seq_len_sum = int(kv_workspace_slots.numel())
    if seq_len_sum == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out

    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(seq_len_sum, 1) * 4))
    block_n = 32
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, seq_len_sum), dtype=torch.float32, device=q.device
        )
        _dsa_prefill_logits_fp8_kernel[
            (end - start, triton.cdiv(seq_len_sum, block_n))
        ](
            q[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            logits.stride(0),
            seq_len_sum=seq_len_sum,
            page_size=int(page_size),
            row_bytes=row_bytes,
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            num_groups=q.shape[2] // 128,
            softmax_scale=float(softmax_scale),
            BLOCK_N=block_n,
            BLOCK_D=128,
            num_warps=4,
        )
        _dsa_prefill_topk_indices(
            logits,
            row_starts[start:end],
            row_ends[start:end],
            topk=topk,
            out=out[start:end],
            lens_out=lens_out[start:end],
        )
    return out, lens_out
