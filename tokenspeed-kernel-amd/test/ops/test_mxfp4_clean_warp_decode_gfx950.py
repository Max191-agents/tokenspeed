# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "AMD GFX950 is required for MXFP4 warp-decode tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950  # noqa: E402
from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (  # noqa: E402
    gluon_mxfp_dynamic_mxfp4_fused_moe,
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950 import (  # noqa: E402
    gluon_mxfp4_moe_decode,
    invoke_sigmoid_bias_topk_route_gluon,
    invoke_softmax_topk_route_gluon,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.moe_sorting import (  # noqa: E402
    gluon_moe_sorting,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.prefill_stage1 import (  # noqa: E402
    invoke_gluon_mxfp4_moe_stage1,
)
from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.scale import (  # noqa: E402
    gather_package_cdna4_scale,
)
from tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)


def _make_weights(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(910603)
    nibble_values = torch.tensor((0, 1, 2, 9, 10), device=device, dtype=torch.uint8)

    def packed(shape):
        lo = nibble_values[
            torch.randint(
                0, len(nibble_values), shape, device=device, generator=generator
            )
        ]
        hi = nibble_values[
            torch.randint(
                0, len(nibble_values), shape, device=device, generator=generator
            )
        ]
        return lo | (hi << 4)

    w13 = packed((num_experts, 2 * intermediate_size, hidden_size // 2))
    w2 = packed((num_experts, hidden_size, intermediate_size // 2))
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        127,
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        127,
        device=device,
        dtype=torch.uint8,
    )
    return w13, w13_scale, w2, w2_scale


def test_dynamic_mxfp4_precomputed_tiny_m_prefers_direct_over_mfma(
    monkeypatch: pytest.MonkeyPatch,
):
    # Below _PRECOMPUTED_MFMA_MIN_M the dispatch must try the direct decode
    # kernel and must not fall through to the precomputed-MFMA decode.
    hidden = torch.empty((1, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((1, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    topk_ids = torch.empty((1, 1), device="cuda", dtype=torch.int32)
    topk_weights = torch.empty((1, 1), device="cuda", dtype=torch.float32)
    direct_sentinel = torch.empty_like(hidden)
    direct_calls = 0
    mfma_calls = 0

    def fake_direct_decode(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        assert kwargs["precomputed_topk_ids"] is topk_ids
        assert kwargs["precomputed_topk_weights"] is topk_weights
        return direct_sentinel

    def fake_mfma_decode(*args, **kwargs):
        nonlocal mfma_calls
        mfma_calls += 1
        return torch.empty_like(hidden)

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_direct_mfma_decode",
        fake_direct_decode,
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_mfma_decode",
        fake_mfma_decode,
    )
    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_ids=topk_ids,
        precomputed_topk_weights=topk_weights,
    )

    assert out is direct_sentinel
    assert direct_calls == 1
    assert mfma_calls == 0


def test_dynamic_mxfp4_precomputed_default_uses_mfma_for_medium_m(
    monkeypatch: pytest.MonkeyPatch,
):
    hidden = torch.empty((4, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((4, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    topk_ids = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    topk_weights = torch.empty((4, 1), device="cuda", dtype=torch.float32)
    sentinel = torch.empty_like(hidden)
    mfma_calls = 0

    def fake_mfma_decode(*args, **kwargs):
        nonlocal mfma_calls
        mfma_calls += 1
        assert kwargs["precomputed_topk_ids"] is topk_ids
        assert kwargs["precomputed_topk_weights"] is topk_weights
        return sentinel

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_mfma_decode",
        fake_mfma_decode,
    )
    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_ids=topk_ids,
        precomputed_topk_weights=topk_weights,
    )

    assert out is sentinel
    assert mfma_calls == 1


def test_dynamic_mxfp4_package_prefill_runs_on_caller_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    # The package-prefill bridge uses the in-house ``gluon_moe_sorting``, which
    # runs on the caller's stream, so the bridge must execute entirely on that
    # stream with no cross-stream fence or ``record_stream`` fixup.
    hidden = torch.empty((128, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((128, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    caller_stream = torch.cuda.Stream()

    class Sentinel:
        def record_stream(self, stream):  # pragma: no cover - must not be called
            raise AssertionError(
                "package prefill must not need cross-stream record_stream"
            )

    sentinel = Sentinel()
    observed = {}

    def fake_package_prefill(*args, **kwargs):
        observed["stream"] = torch.cuda.current_stream()
        return sentinel

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_gluon_package_mxfp4_prefill",
        fake_package_prefill,
    )

    with torch.cuda.stream(caller_stream):
        out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
            hidden,
            router,
            dummy_w,
            dummy_w,
            w13_mx_scale=dummy_scale,
            w2_mx_scale=dummy_scale,
            top_k=1,
            correction_bias=None,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
        )

    assert out is sentinel
    # The bridge ran on the caller's stream, not the default stream.
    assert observed["stream"] == caller_stream
    assert observed["stream"] != torch.cuda.default_stream()


def test_package_exposes_prefill_stage_entry_points():
    from tokenspeed_kernel_amd.ops.moe import gluon_a4w4_gfx950 as package
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950 import (
        invoke_gluon_mxfp4_moe_stage1,
        invoke_gluon_mxfp4_moe_stage2_1x2,
    )

    assert not hasattr(fused_mxfp_gfx950, "_maybe_runtime_mxfp4_warp_decode")
    assert not hasattr(package, "invoke_stage1_warp_decode_gluon")
    assert not hasattr(package, "invoke_stage2_warp_decode_gluon")
    assert invoke_gluon_mxfp4_moe_stage1.__module__.endswith(".prefill_stage1")
    assert invoke_gluon_mxfp4_moe_stage2_1x2.__module__.endswith(".prefill_stage2")


def _make_preprocessed_layer(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: str,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.quant_config = type("QuantConfig", (), {})()
    layer.quant_config.use_dynamic_mxfp4_activations = True
    layer.w13_input_layout = "concatenated"
    layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale.clone(), requires_grad=False)
    layer.w13_weight_bias = torch.nn.Parameter(
        torch.zeros(num_experts, 2 * intermediate_size, device=device),
        requires_grad=False,
    )
    layer.w2_weight_bias = torch.nn.Parameter(
        torch.zeros(num_experts, hidden_size, device=device),
        requires_grad=False,
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, layer)
    return layer


def _precomputed_mfma_expected(
    hidden: torch.Tensor,
    router: torch.Tensor,
    layer: torch.nn.Module,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    # Force the MFMA decode kernel below its tuned batch-size gate so it can be
    # used as the bit-exact reference for the direct/route-owned decode tests.
    out = fused_mxfp_gfx950._maybe_precomputed_mxfp4_mfma_decode(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        w13_bias=None,
        w2_bias=None,
        out_dtype=torch.bfloat16,
        max_m=8,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
        min_m=1,
    )
    assert out is not None
    return out


def _softmax_topk_reference(
    router: torch.Tensor,
    topk: int,
    *,
    correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(router.float(), dim=-1)
    choice = scores
    if correction_bias is not None:
        choice = choice + correction_bias.float().unsqueeze(0)
    _, topk_ids = torch.topk(choice, k=topk, dim=-1, sorted=True)
    topk_weights = scores.gather(1, topk_ids)
    if normalize_topk_weights:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


def _sigmoid_bias_topk_reference(
    router: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    routed_scaling_factor: float = 1.0,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(router.float()).to(router.dtype)
    _, topk_ids = torch.topk(
        scores.float() + correction_bias.float().unsqueeze(0),
        k=topk,
        dim=-1,
        sorted=True,
    )
    topk_weights = scores.gather(1, topk_ids)
    if normalize_topk_weights:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * routed_scaling_factor
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_stable_topk_uses_smaller_index_for_exact_ties(dtype: torch.dtype):
    values = torch.tensor([[1.0, 3.0, 3.0, -2.0, 3.0, 0.5]], device="cuda", dtype=dtype)

    selected, ids = fused_mxfp_gfx950._stable_topk_smaller_index(values, 4)

    torch.testing.assert_close(
        ids,
        torch.tensor([[1, 2, 4, 0]], device="cuda", dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        selected,
        torch.tensor([[3.0, 3.0, 3.0, 1.0]], device="cuda", dtype=dtype),
        rtol=0,
        atol=0,
    )


def test_stable_topk_preserves_non_tied_values_and_order():
    generator = torch.Generator(device="cuda").manual_seed(20260712)
    values = torch.randn(
        (32, 384), device="cuda", dtype=torch.float32, generator=generator
    )
    expected_values, expected_ids = torch.topk(values, 8, dim=-1, sorted=True)

    actual_values, actual_ids = fused_mxfp_gfx950._stable_topk_smaller_index(
        values, 8, dim=-1, sorted=True
    )

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stable_topk_is_cuda_graph_capturable(dtype: torch.dtype):
    """Regression: stable top-k must not copy CPU->CUDA (illegal under capture).

    The prior form built the tie-break masks with ``raw.new_tensor(<int>)``,
    which materializes a CPU tensor and copies it to the GPU -- this crashed
    CUDA-graph capture in the runtime model tests.
    """
    values = torch.randn((8, 128), device="cuda", dtype=dtype)
    eager_v, eager_i = fused_mxfp_gfx950._stable_topk_smaller_index(
        values, 8, dim=-1, sorted=True
    )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fused_mxfp_gfx950._stable_topk_smaller_index(values, 8, dim=-1, sorted=True)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cap_v, cap_i = fused_mxfp_gfx950._stable_topk_smaller_index(
            values, 8, dim=-1, sorted=True
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(cap_i, eager_i, rtol=0, atol=0)
    torch.testing.assert_close(cap_v, eager_v, rtol=0, atol=0)


def test_biased_grouped_route_is_repeatable_with_bf16_ties():
    num_tokens = 512
    num_experts = 384
    topk = 8
    generator = torch.Generator(device="cuda").manual_seed(603)
    logits = torch.randn(
        (num_tokens, num_experts),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    correction_bias = torch.zeros((num_experts,), device="cuda", dtype=torch.bfloat16)

    expected_weights, expected_ids = fused_mxfp_gfx950._biased_grouped_topk_reference(
        logits,
        correction_bias,
        topk,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    for _ in range(4):
        actual_weights, actual_ids = fused_mxfp_gfx950._biased_grouped_topk_reference(
            logits,
            correction_bias,
            topk,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
        )
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_softmax_topk_route_gluon_matches_reference():
    device = "cuda"
    router = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75, -1.0, 0.5, -0.25, 1.25],
            [-0.75, 0.5, 1.5, -0.25, 0.0, 1.0, -1.5, 0.25],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    correction_bias = torch.tensor(
        [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3],
        device=device,
        dtype=torch.float32,
    )
    topk = 3
    expected_ids, expected_weights = _softmax_topk_reference(
        router,
        topk,
        correction_bias=correction_bias,
        routed_scaling_factor=1.75,
        normalize_topk_weights=True,
    )

    topk_ids, topk_weights = invoke_softmax_topk_route_gluon(
        router,
        topk,
        correction_bias=correction_bias,
        routed_scaling_factor=1.75,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, atol=5e-3, rtol=5e-3)


def test_sigmoid_bias_topk_route_gluon_matches_reference():
    device = "cuda"
    router = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75, -1.0, 0.5, -0.25, 1.25],
            [-0.75, 0.5, 1.5, -0.25, 0.0, 1.0, -1.5, 0.25],
        ],
        device=device,
        dtype=torch.float32,
    )
    correction_bias = torch.tensor(
        [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3],
        device=device,
        dtype=torch.float32,
    )
    topk = 3
    expected_ids, expected_weights = _sigmoid_bias_topk_reference(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )

    topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_sigmoid_bias_topk_route_gluon_matches_kimi_k3_shape(num_tokens: int):
    device = "cuda"
    num_experts = 896
    topk = 16
    generator = torch.Generator(device=device).manual_seed(990611)
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    correction_bias = (
        torch.randn(
            (num_experts,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    expected_ids, expected_weights = _sigmoid_bias_topk_reference(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )

    topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_sigmoid_bias_topk_route_gluon_kimi_k3_is_cuda_graph_capturable(
    num_tokens: int,
):
    generator = torch.Generator(device="cuda").manual_seed(20260720 + num_tokens)
    router = torch.randn(
        (num_tokens, 896),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    correction_bias = torch.randn(
        (896,), device="cuda", dtype=torch.float32, generator=generator
    )

    eager_ids, eager_weights = invoke_sigmoid_bias_topk_route_gluon(
        router,
        correction_bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ids, graph_weights = invoke_sigmoid_bias_topk_route_gluon(
            router,
            correction_bias,
            16,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_ids, eager_ids, rtol=0, atol=0)
    torch.testing.assert_close(graph_weights, eager_weights, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_precomputed_topk_fused_route_matches_route_from_topk(num_tokens: int):
    device = "cuda"
    num_experts = 16
    topk = 8
    topk_ids = (
        (torch.arange(num_tokens * topk, device=device, dtype=torch.int32) * 7 + 3)
        .reshape(num_tokens, topk)
        .remainder(num_experts)
    )
    topk_weights = torch.linspace(
        0.125,
        1.0,
        steps=num_tokens * topk,
        device=device,
        dtype=torch.float32,
    ).reshape(num_tokens, topk)

    actual = fused_mxfp_gfx950.gluon_precomputed_topk_fused_route(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=torch.bfloat16,
    )
    expected = fused_mxfp_gfx950._route_from_topk(
        topk_weights,
        topk_ids,
        num_experts,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    actual_ragged, actual_gather, actual_scatter, actual_gate = actual
    expected_ragged, expected_gather, expected_scatter, expected_gate = expected
    torch.testing.assert_close(actual_gather, expected_gather, rtol=0, atol=0)
    torch.testing.assert_close(actual_scatter, expected_scatter, rtol=0, atol=0)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_ragged.slice_sizes, expected_ragged.slice_sizes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_ragged.slice_offs, expected_ragged.slice_offs, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_ragged.block_offs_data,
        expected_ragged.block_offs_data,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_ragged.block_schedule_data,
        expected_ragged.block_schedule_data,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_package_mxfp4_decode_matches_reference_mfma_exact(num_tokens: int):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260710 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
    topk_ids = topk_ids.to(torch.int32)

    out = gluon_mxfp4_moe_decode(
        hidden,
        layer.w13_weight_triton_tensor,
        layer.w13_precision_config.b_mx_scale,
        layer.w2_weight_triton_tensor,
        layer.w2_precision_config.b_mx_scale,
        topk_ids,
        topk_weights,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [3, 4, 8])
def test_dynamic_mxfp4_dispatch_uses_mfma_with_topk(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    # At M in the precomputed-MFMA decode range the dispatch must route to the
    # MFMA decode kernel and match it bit-exactly. M=3 is the previously-gapped
    # interval (_DIRECT_DECODE_MAX_M, _PRECOMPUTED_MFMA_MIN_M): the apply
    # wrapper now forwards precomputed top-k for it too, so it must also match
    # the MFMA reference exactly rather than falling back to logit routing.
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260710)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
    topk_ids = topk_ids.to(torch.int32)
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    # the reference default runtime path consumes preshuffled Gluon-dot tensors.
    # The tiny-M decode path must accept those same runtime objects directly.
    assert getattr(layer.w13_weight_triton_tensor, "is_shuffled_for_gluon_dot", False)
    assert getattr(layer.w2_weight_triton_tensor, "is_shuffled_for_gluon_dot", False)
    assert not hasattr(layer.w13_weight_triton_tensor, "gluon_decode_clean_weight")
    assert not hasattr(layer.w13_weight_triton_tensor, "gluon_decode_clean_scale")
    assert not hasattr(layer.w2_weight_triton_tensor, "gluon_decode_clean_weight")
    assert not hasattr(layer.w2_weight_triton_tensor, "gluon_decode_clean_scale")

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_dynamic_mxfp4_direct_precomputed_matches_mfma_exact(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260713 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
    topk_ids = topk_ids.to(torch.int32)
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_dynamic_mxfp4_route_owned_default_falls_back_when_direct_unsupported(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((num_tokens, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    intermediate = torch.empty_like(hidden)
    sentinel = torch.empty_like(hidden)
    route_calls = 0
    quantize_calls = 0
    matmul_calls = 0

    def fake_route(*args, **kwargs):
        nonlocal route_calls
        route_calls += 1
        return None, None, None, None

    def fake_quantize(x, *args, **kwargs):
        nonlocal quantize_calls
        quantize_calls += 1
        return x, dummy_scale

    def fake_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        # gemm1 fuses the intermediate requant (out_quant_format="mxfp4") and
        # returns (intermediate, gemm2_scale); gemm2 returns the final output.
        return (intermediate, dummy_scale) if matmul_calls == 1 else sentinel

    monkeypatch.setattr(fused_mxfp_gfx950, "_dynamic_mxfp4_route", fake_route)
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_quantize_mxfp4_activation",
        fake_quantize,
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "gluon_mxfp_ragged_matmul", fake_matmul)

    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
    )

    assert out is sentinel
    assert route_calls == 1
    # Only the hidden state is quantized explicitly; the intermediate requant is
    # fused into the gemm1 ragged matmul (out_quant_format="mxfp4").
    assert quantize_calls == 1
    assert matmul_calls == 2


def test_dynamic_mxfp4_generic_path_consumes_precomputed_topk(
    monkeypatch: pytest.MonkeyPatch,
):
    """Large-M fallback must not silently recompute routing from logits."""

    hidden = torch.empty((32, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((32, 4), device="cuda", dtype=torch.float32)
    topk_weights = torch.ones((32, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((32, 1), device="cuda", dtype=torch.int32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    intermediate = torch.empty_like(hidden)
    sentinel = torch.empty_like(hidden)
    route_calls = 0
    matmul_calls = 0

    def fake_route_from_topk(weights, ids, *, num_experts, dtype):
        nonlocal route_calls
        route_calls += 1
        assert weights.data_ptr() == topk_weights.data_ptr()
        assert ids.data_ptr() == topk_ids.data_ptr()
        assert num_experts == router.shape[1]
        assert dtype == router.dtype
        return None, None, None, None

    def fail_dynamic_route(*args, **kwargs):
        raise AssertionError("precomputed top-k was ignored")

    def fake_quantize(x, *args, **kwargs):
        return x, dummy_scale

    def fake_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        # gemm1 fuses the intermediate requant and returns (intermediate, scale).
        return (intermediate, dummy_scale) if matmul_calls == 1 else sentinel

    monkeypatch.setattr(fused_mxfp_gfx950, "_route_from_topk", fake_route_from_topk)
    monkeypatch.setattr(fused_mxfp_gfx950, "_dynamic_mxfp4_route", fail_dynamic_route)
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_quantize_mxfp4_activation",
        fake_quantize,
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "gluon_mxfp_ragged_matmul", fake_matmul)

    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
    )

    assert out is sentinel
    assert route_calls == 1
    assert matmul_calls == 2


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8])
def test_dynamic_mxfp4_route_owned_softmax_mfma_decode(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260711 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids, topk_weights = invoke_softmax_topk_route_gluon(router, topk)
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8])
def test_dynamic_mxfp4_route_owned_kimi_sigmoid_mfma_decode(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    scale = 2.827
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260712 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    correction_bias = torch.tensor(
        [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.05, -0.05],
        device=device,
        dtype=torch.float32,
    )
    topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_gluon(
        router,
        correction_bias,
        topk,
        routed_scaling_factor=scale,
        normalize_topk_weights=True,
    )
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    out = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        layer.w13_weight_triton_tensor,
        layer.w2_weight_triton_tensor,
        w13_mx_scale=layer.w13_precision_config.b_mx_scale,
        w2_mx_scale=layer.w2_precision_config.b_mx_scale,
        top_k=topk,
        correction_bias=correction_bias,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=scale,
        normalize_topk_weights=True,
        routing_method_type=2,
        w13_bias=None,
        w2_bias=None,
    )
    torch.cuda.synchronize()

    expected = _precomputed_mfma_expected(
        hidden,
        router,
        layer,
        topk_ids,
        topk_weights,
        topk=topk,
    )
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Precomputed entry point (``gluon_mxfp_precomputed_mxfp4_fused_moe``) dispatch.
#
# This is the entry consumed by ``gluon_mxfp4_precomputed_moe_apply`` (the
# ``routing_mode="precomputed_topk"`` registered kernel). These tests pin down
# its *current* dispatch behavior so we can validate the observation that,
# unlike ``gluon_mxfp_dynamic_mxfp4_fused_moe``, it does NOT route to the decode
# or package-prefill fast paths: every batch size falls through to the generic
# ragged path. If we later wire the fast paths into this entry, these tests are
# the ones that must change.
# ---------------------------------------------------------------------------


def _spy_precomputed_entry_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, int], object]:
    """Install spies on every dispatch helper the precomputed entry could hit.

    Returns ``(counters, sentinel)`` where ``counters`` maps each helper to its
    call count and ``sentinel`` is the object the stubbed ragged path returns.
    The generic ragged path is stubbed so no real kernel launches are needed;
    the decode / package-prefill helpers raise if the precomputed entry ever
    reaches them.
    """

    counters: dict[str, int] = {
        "package_prefill": 0,
        "direct_decode": 0,
        "mfma_decode": 0,
        "fused_route": 0,
        "route_from_topk": 0,
        "ragged": 0,
    }

    def fail_package_prefill(*args, **kwargs):
        counters["package_prefill"] += 1
        raise AssertionError(
            "precomputed entry unexpectedly reached the package-prefill path"
        )

    def fail_direct_decode(*args, **kwargs):
        counters["direct_decode"] += 1
        raise AssertionError(
            "precomputed entry unexpectedly reached the direct MFMA decode path"
        )

    def fail_mfma_decode(*args, **kwargs):
        counters["mfma_decode"] += 1
        raise AssertionError(
            "precomputed entry unexpectedly reached the precomputed-MFMA decode path"
        )

    def fake_fused_route(*args, **kwargs):
        counters["fused_route"] += 1
        return None, None, None, None

    def fake_route_from_topk(*args, **kwargs):
        counters["route_from_topk"] += 1
        return None, None, None, None

    sentinel = object()

    def fake_ragged(*args, **kwargs):
        counters["ragged"] += 1
        return sentinel

    monkeypatch.setattr(
        fused_mxfp_gfx950, "_maybe_gluon_package_mxfp4_prefill", fail_package_prefill
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_direct_mfma_decode",
        fail_direct_decode,
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_precomputed_mxfp4_mfma_decode",
        fail_mfma_decode,
    )
    monkeypatch.setattr(
        fused_mxfp_gfx950, "gluon_precomputed_topk_fused_route", fake_fused_route
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "_route_from_topk", fake_route_from_topk)
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_gluon_mxfp_dynamic_mxfp4_fused_moe_from_route",
        fake_ragged,
    )
    return counters, sentinel


def _make_dummy_weights_1x4(device: str = "cuda"):
    # Rank-3 expert weight tensor so ``_extract_gluon_raw_w_unshuffled`` accepts
    # it; the ragged path is stubbed so contents never get read.
    w = torch.empty((1, 4, 4), device=device, dtype=torch.uint8)
    scale = torch.empty((1, 4, 1), device=device, dtype=torch.uint8)
    return w, scale


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_precomputed_entry_decode_sizes_use_ragged_not_fast_paths(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Decode-sized batches must NOT reach decode/package-prefill from this entry.

    This documents the gap versus ``gluon_mxfp_dynamic_mxfp4_fused_moe``: the
    precomputed entry ignores the tuned decode kernels and always builds ragged
    metadata + runs the generic matmul.
    """
    counters, sentinel = _spy_precomputed_entry_dispatch(monkeypatch)
    w, scale = _make_dummy_weights_1x4()
    topk_weights = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.int32)
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)

    out = gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden,
        topk_weights,
        topk_ids,
        w,
        w,
        w13_mx_scale=scale,
        w2_mx_scale=scale,
    )

    assert out is sentinel
    assert counters["ragged"] == 1
    # The fast paths are never consulted.
    assert counters["package_prefill"] == 0
    assert counters["direct_decode"] == 0
    assert counters["mfma_decode"] == 0
    # Small-M uses the single-kernel fused route; either way we routed exactly
    # once and never recomputed from logits.
    assert counters["fused_route"] + counters["route_from_topk"] == 1


def test_precomputed_entry_prefill_size_uses_ragged_not_package_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    """Prefill-sized batches must NOT reach the package-prefill path either."""
    counters, sentinel = _spy_precomputed_entry_dispatch(monkeypatch)
    w, scale = _make_dummy_weights_1x4()
    num_tokens = 128
    topk_weights = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.int32)
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)

    out = gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden,
        topk_weights,
        topk_ids,
        w,
        w,
        w13_mx_scale=scale,
        w2_mx_scale=scale,
    )

    assert out is sentinel
    assert counters["ragged"] == 1
    assert counters["package_prefill"] == 0
    assert counters["direct_decode"] == 0
    assert counters["mfma_decode"] == 0
    # Large M builds ragged metadata via the generic host route helper.
    assert counters["route_from_topk"] == 1
    assert counters["fused_route"] == 0


def test_precomputed_entry_small_m_uses_fused_route(
    monkeypatch: pytest.MonkeyPatch,
):
    """Small-M precomputed entry uses the single-kernel fused route helper."""
    counters, _sentinel = _spy_precomputed_entry_dispatch(monkeypatch)
    w, scale = _make_dummy_weights_1x4()
    # M=2, top_k=1 -> M < SMALLM_MAX_M and M*top_k <= GLUON_ROUTE_MAX_G.
    topk_weights = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((2, 1), device="cuda", dtype=torch.int32)
    hidden = torch.empty((2, 8), device="cuda", dtype=torch.bfloat16)

    gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden,
        topk_weights,
        topk_ids,
        w,
        w,
        w13_mx_scale=scale,
        w2_mx_scale=scale,
    )

    assert counters["fused_route"] == 1
    assert counters["route_from_topk"] == 0


def test_precomputed_entry_rejects_missing_or_mismatched_topk():
    """Shape/None validation on the precomputed entry."""
    w, scale = _make_dummy_weights_1x4()
    hidden = torch.empty((2, 8), device="cuda", dtype=torch.bfloat16)

    # rank-1 topk_ids is rejected.
    with pytest.raises(ValueError, match="rank-2"):
        gluon_mxfp_precomputed_mxfp4_fused_moe(
            hidden,
            torch.ones((2,), device="cuda", dtype=torch.float32),
            torch.zeros((2,), device="cuda", dtype=torch.int32),
            w,
            w,
            w13_mx_scale=scale,
            w2_mx_scale=scale,
        )

    # mismatched weights/ids shapes are rejected.
    with pytest.raises(ValueError, match="same shape"):
        gluon_mxfp_precomputed_mxfp4_fused_moe(
            hidden,
            torch.ones((2, 2), device="cuda", dtype=torch.float32),
            torch.zeros((2, 1), device="cuda", dtype=torch.int32),
            w,
            w,
            w13_mx_scale=scale,
            w2_mx_scale=scale,
        )


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8])
def test_kernel_routing_decode_reaches_route_owned_up_to_decode_max_m(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Kernel-routing (no precomputed top-k) must hit route-owned decode M<=8.

    This pins the dispatch cap lift: without precomputed top-k, batches up to
    ``_DECODE_MAX_M`` must be handed to ``_maybe_route_owned_mxfp4_mfma_decode``
    (which routes in-kernel and dispatches to the tuned decode kernels) rather
    than falling through to the generic ragged GEMM. Previously only M <=
    ``_ROUTE_OWNED_DECODE_MAX_M`` (=2) reached it, so Kimi-style M=8 decode
    silently used the slow ragged path.
    """
    assert num_tokens <= fused_mxfp_gfx950._DECODE_MAX_M
    hidden = torch.empty((num_tokens, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.empty((num_tokens, 4), device="cuda", dtype=torch.float32)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)
    sentinel = torch.empty_like(hidden)
    route_owned_calls = 0
    captured = {}

    def fake_route_owned(*args, **kwargs):
        nonlocal route_owned_calls
        route_owned_calls += 1
        captured["max_m"] = kwargs.get("max_m")
        captured["allow_generic_fallback"] = kwargs.get("allow_generic_fallback")
        return sentinel

    def fail_route(*args, **kwargs):
        raise AssertionError(
            "kernel-routing decode fell through to the ragged reference path"
        )

    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_route_owned_mxfp4_mfma_decode",
        fake_route_owned,
    )
    monkeypatch.setattr(fused_mxfp_gfx950, "_dynamic_mxfp4_route", fail_route)

    out = fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=1,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
    )

    assert out is sentinel
    assert route_owned_calls == 1
    # The cap lift forwards the full decode ceiling and lets route-owned reach
    # the precomputed-MFMA decode kernel for the M=3..8 range.
    assert captured["max_m"] == fused_mxfp_gfx950._DECODE_MAX_M
    assert captured["allow_generic_fallback"] is True


@pytest.mark.parametrize("num_tokens", [9, 16, 33, 64])
def test_package_stage1_token_order_scale_matches_sorted_scale(
    num_tokens: int,
):
    """Stage 1 must produce the same result without a scale-gather launch."""
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.moe_sorting import (
        gluon_moe_sorting,
    )
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.prefill_stage1 import (
        invoke_gluon_mxfp4_moe_stage1,
    )
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.scale import (
        gather_package_cdna4_scale,
    )

    hidden_size = 1024
    intermediate_size = 512
    num_experts = 32
    topk = 8
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260724 + num_tokens)
    hidden = torch.randn(
        (num_tokens, hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    router = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(
        torch.softmax(router.float(), dim=-1),
        topk,
    )
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = gluon_moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        hidden_size,
        torch.bfloat16,
        128,
    )
    q_hidden, q_hidden_scale = fused_mxfp_gfx950._quantize_mxfp4_activation(hidden)
    sorted_scale = gather_package_cdna4_scale(
        q_hidden_scale,
        sorted_ids,
        source_rows=num_tokens,
        cols=hidden_size,
        top_k=topk,
        flatten_topk=False,
    )
    package_w13 = layer.w13_weight_triton_tensor.gluon_package_prefill_weight.view(
        torch.uint8
    )
    package_w13_scale = layer.w13_weight_triton_tensor.gluon_package_prefill_scale.view(
        torch.uint8
    )
    expected = torch.empty(
        (num_tokens, topk, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )
    actual = torch.empty_like(expected)

    common_args = (
        q_hidden,
        package_w13,
        None,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
    )
    common_kwargs = {
        "topk": topk,
        "w1_scale": package_w13_scale,
        "sorted_weights": None,
        "b_preshuffled": True,
        "b_gdot128": True,
    }
    invoke_gluon_mxfp4_moe_stage1(
        *common_args,
        expected,
        a1_scale=sorted_scale,
        **common_kwargs,
    )
    invoke_gluon_mxfp4_moe_stage1(
        *common_args,
        actual,
        a1_scale=q_hidden_scale,
        a1_scale_is_sorted=False,
        **common_kwargs,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_package_stage1_token_order_scale_tracks_routes_during_graph_replay():
    """Graph replay must use the current route to address token-order scales."""
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.moe_sorting import (
        gluon_moe_sorting,
    )
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.prefill_stage1 import (
        invoke_gluon_mxfp4_moe_stage1,
    )

    num_tokens = 16
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 32
    topk = 8
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260726)
    row_scale = torch.arange(
        1, num_tokens + 1, device=device, dtype=torch.float32
    ).unsqueeze(1)
    hidden_a = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * row_scale
    ).to(torch.bfloat16)
    hidden_b = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * row_scale.flip(0)
    ).to(torch.bfloat16)
    token = torch.arange(num_tokens, device=device, dtype=torch.int32).unsqueeze(1)
    slot = torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0)
    topk_ids_a = ((token + slot) % num_experts).contiguous()
    topk_ids_b = ((token * 5 + slot * 3 + 7) % num_experts).contiguous()
    topk_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32
    )

    def sort_routes(topk_ids):
        sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = gluon_moe_sorting(
            topk_ids,
            topk_weights,
            num_experts,
            hidden_size,
            torch.bfloat16,
            128,
        )
        return sorted_ids, sorted_expert_ids, num_valid_ids

    route_a = sort_routes(topk_ids_a)
    route_b = sort_routes(topk_ids_b)
    q_hidden_a, q_scale_a = fused_mxfp_gfx950._quantize_mxfp4_activation(hidden_a)
    q_hidden_b, q_scale_b = fused_mxfp_gfx950._quantize_mxfp4_activation(hidden_b)
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    package_w13 = layer.w13_weight_triton_tensor.gluon_package_prefill_weight.view(
        torch.uint8
    )
    package_scale = layer.w13_weight_triton_tensor.gluon_package_prefill_scale.view(
        torch.uint8
    )
    captured_q_hidden = q_hidden_a.clone()
    captured_q_scale = q_scale_a.clone()
    captured_route = tuple(value.clone() for value in route_a)
    actual = torch.empty(
        (num_tokens, topk, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )
    expected = torch.empty_like(actual)

    def run_stage1(q_hidden, q_scale, route, out):
        invoke_gluon_mxfp4_moe_stage1(
            q_hidden,
            package_w13,
            None,
            *route,
            out,
            topk,
            w1_scale=package_scale,
            a1_scale=q_scale,
            sorted_weights=None,
            b_preshuffled=True,
            b_gdot128=True,
            a1_scale_is_sorted=False,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run_stage1(captured_q_hidden, captured_q_scale, captured_route, actual)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_stage1(captured_q_hidden, captured_q_scale, captured_route, actual)

    captured_q_hidden.copy_(q_hidden_b)
    captured_q_scale.copy_(q_scale_b)
    for captured, updated in zip(captured_route, route_b, strict=True):
        captured.copy_(updated)
    run_stage1(q_hidden_b, q_scale_b, route_b, expected)
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(q_scale_a, q_scale_b)
    assert not torch.equal(route_a[0], route_b[0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_package_stage1_rejects_partial_k_tile():
    from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.prefill_stage1 import (
        invoke_gluon_mxfp4_moe_stage1,
    )

    device = "cuda"
    num_tokens = 16
    hidden_size = 288
    intermediate_size = 128
    num_experts = 1
    topk = 1
    k_scale = hidden_size // 32

    with pytest.raises(ValueError, match="K divisible.*256"):
        invoke_gluon_mxfp4_moe_stage1(
            torch.empty(
                (num_tokens, hidden_size // 2), device=device, dtype=torch.uint8
            ),
            torch.empty(
                (num_experts, 2 * intermediate_size, hidden_size // 2),
                device=device,
                dtype=torch.uint8,
            ),
            None,
            torch.empty((128,), device=device, dtype=torch.int32),
            torch.empty((1,), device=device, dtype=torch.int32),
            torch.empty((2,), device=device, dtype=torch.int32),
            torch.empty(
                (num_tokens, topk, intermediate_size),
                device=device,
                dtype=torch.bfloat16,
            ),
            topk,
            w1_scale=torch.empty(
                (num_experts, 2 * intermediate_size, k_scale),
                device=device,
                dtype=torch.uint8,
            ),
            a1_scale=torch.empty((128, k_scale), device=device, dtype=torch.uint8),
            b_preshuffled=True,
            b_gdot128=True,
        )


@pytest.mark.parametrize("num_tokens", [16, 64, 256])
def test_package_prefill_stage2_bm128_matches_ragged_reference(
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Package prefill (flat stage2 BLOCK_M=128) must match the ragged reference.

    The package-prefill path (stage1/stage2 kernels, stage2 down-proj at the
    tuned flat BLOCK_M=128) must be bit-exact against the generic ragged
    reference (``_gluon_mxfp_dynamic_mxfp4_fused_moe_from_route`` reached by
    disabling package prefill). This pins the correctness of the BLOCK_M=128
    tuning change: it is 10-17% faster on Kimi with no gpt-oss regression, and
    here we confirm it does not change the MoE result.
    """
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 16
    topk = 4
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(20260716 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts), device=device, dtype=torch.bfloat16, generator=gen
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    def run():
        return fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
            hidden,
            router,
            layer.w13_weight_triton_tensor,
            layer.w2_weight_triton_tensor,
            w13_mx_scale=layer.w13_precision_config.b_mx_scale,
            w2_mx_scale=layer.w2_precision_config.b_mx_scale,
            top_k=topk,
            correction_bias=None,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            normalize_topk_weights=True,
            routing_method_type=0,
            out_dtype=torch.bfloat16,
            precomputed_topk_weights=topk_weights,
            precomputed_topk_ids=topk_ids,
        )

    # Reference: disable package prefill so the dispatch falls to the generic
    # ragged path (the trusted MXFP4 MoE reference).
    monkeypatch.setattr(
        fused_mxfp_gfx950,
        "_maybe_gluon_package_mxfp4_prefill",
        lambda *args, **kwargs: None,
    )
    ref = run()
    torch.cuda.synchronize()

    # Package prefill (flat stage2 BLOCK_M=128).
    monkeypatch.undo()
    out = run()
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "num_tokens, num_experts, hidden_size, concentrated_routes",
    [
        pytest.param(16, 32, 1024, False, id="padded-quarters"),
        pytest.param(128, 32, 1024, True, id="all-four-quarters"),
        pytest.param(16, 384, 7168, False, id="kimi-e384-top8"),
    ],
)
def test_package_stage1_quantized_epilogue_matches_separate_quantizer(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    concentrated_routes: bool,
):
    intermediate_size = 512
    topk = 8
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(
        20260724 + num_tokens + num_experts
    )
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    if concentrated_routes:
        # Every selected expert receives 128 routes, so each of the four
        # 32-row epilogue groups contains real tokens.
        topk_ids = (
            torch.arange(topk, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(num_tokens, -1)
            .contiguous()
        )
        topk_weights = torch.full(
            (num_tokens, topk),
            1.0 / topk,
            dtype=torch.float32,
            device=device,
        )
    else:
        router = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(router.float(), dim=-1), topk)
        topk_ids = topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_weights.to(torch.float32).contiguous()
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    del w13, w13_scale, w2, w2_scale
    package_weight = layer.w13_weight_triton_tensor.gluon_package_prefill_weight
    package_scale = layer.w13_weight_triton_tensor.gluon_package_prefill_scale

    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = gluon_moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        hidden_size,
        torch.bfloat16,
        128,
    )
    q_hidden, q_hidden_scale = fused_mxfp_gfx950._quantize_mxfp4_activation(hidden)

    inter = torch.empty(
        (num_tokens, topk, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    invoke_gluon_mxfp4_moe_stage1(
        q_hidden,
        package_weight.view(torch.uint8),
        None,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        inter,
        topk,
        w1_scale=package_scale.view(torch.uint8),
        a1_scale=q_hidden_scale,
        sorted_weights=None,
        b_preshuffled=True,
        b_gdot128=True,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
        a1_scale_is_sorted=False,
    )
    expected_q, token_order_scale = fused_mxfp_gfx950._quantize_mxfp4_activation(
        inter.view(num_tokens * topk, intermediate_size)
    )
    expected_scale = gather_package_cdna4_scale(
        token_order_scale,
        sorted_ids,
        source_rows=num_tokens * topk,
        cols=intermediate_size,
        top_k=topk,
        flatten_topk=True,
    )

    actual_q = torch.empty_like(expected_q)
    actual_scale = torch.empty_like(expected_scale)
    invoke_gluon_mxfp4_moe_stage1(
        q_hidden,
        package_weight.view(torch.uint8),
        None,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        actual_q,
        topk,
        w1_scale=package_scale.view(torch.uint8),
        a1_scale=q_hidden_scale,
        sorted_weights=None,
        b_preshuffled=True,
        b_gdot128=True,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
        a1_scale_is_sorted=False,
        dst_type=torch.uint8,
        out_scale=actual_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    valid_rows = int(num_valid_ids[0].item())
    valid_scale_bytes = valid_rows * (intermediate_size // 32)
    torch.testing.assert_close(
        actual_scale.view(-1)[:valid_scale_bytes],
        expected_scale.view(-1)[:valid_scale_bytes],
        rtol=0,
        atol=0,
    )


def test_package_stage1_quantized_epilogue_graph_replay_reads_new_routing():
    num_tokens = 128
    num_experts = 32
    hidden_size = 1024
    intermediate_size = 512
    topk = 8
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260725)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.01
    ).contiguous()
    weights = torch.full(
        (num_tokens, topk),
        1.0 / topk,
        dtype=torch.float32,
        device=device,
    )
    sparse_ids = (
        torch.arange(num_tokens * topk, dtype=torch.int32, device=device)
        .view(num_tokens, topk)
        .remainder(num_experts)
    )
    concentrated_ids = (
        torch.arange(topk, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
        .contiguous()
    )

    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    del w13, w13_scale, w2, w2_scale
    package_weight = layer.w13_weight_triton_tensor.gluon_package_prefill_weight
    package_scale = layer.w13_weight_triton_tensor.gluon_package_prefill_scale
    q_hidden, q_hidden_scale = fused_mxfp_gfx950._quantize_mxfp4_activation(hidden)

    def sorted_routing(topk_ids):
        sorted_ids, _, sorted_experts, num_valid, _ = gluon_moe_sorting(
            topk_ids,
            weights,
            num_experts,
            hidden_size,
            torch.bfloat16,
            128,
        )
        return sorted_ids, sorted_experts, num_valid

    sparse = sorted_routing(sparse_ids)
    concentrated = sorted_routing(concentrated_ids)
    empty_weights = torch.empty(0, dtype=torch.float32, device=device)

    expected_inter = torch.empty(
        (num_tokens, topk, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    invoke_gluon_mxfp4_moe_stage1(
        q_hidden,
        package_weight.view(torch.uint8),
        None,
        concentrated[0],
        concentrated[1],
        concentrated[2],
        expected_inter,
        topk,
        w1_scale=package_scale.view(torch.uint8),
        a1_scale=q_hidden_scale,
        sorted_weights=empty_weights,
        b_preshuffled=True,
        b_gdot128=True,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
        a1_scale_is_sorted=False,
    )
    expected_q, token_order_scale = fused_mxfp_gfx950._quantize_mxfp4_activation(
        expected_inter.view(num_tokens * topk, intermediate_size)
    )
    expected_scale = gather_package_cdna4_scale(
        token_order_scale,
        concentrated[0],
        source_rows=num_tokens * topk,
        cols=intermediate_size,
        top_k=topk,
        flatten_topk=True,
    )

    static_sorted_ids = sparse[0].clone()
    static_sorted_experts = sparse[1].clone()
    static_num_valid = sparse[2].clone()
    actual_q = torch.empty_like(expected_q)
    actual_scale = torch.empty_like(expected_scale)

    def run_quantized_stage1():
        invoke_gluon_mxfp4_moe_stage1(
            q_hidden,
            package_weight.view(torch.uint8),
            None,
            static_sorted_ids,
            static_sorted_experts,
            static_num_valid,
            actual_q,
            topk,
            w1_scale=package_scale.view(torch.uint8),
            a1_scale=q_hidden_scale,
            sorted_weights=empty_weights,
            b_preshuffled=True,
            b_gdot128=True,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            swiglu_beta=1.0,
            a1_scale_is_sorted=False,
            dst_type=torch.uint8,
            out_scale=actual_scale,
        )

    run_quantized_stage1()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_quantized_stage1()

    static_sorted_ids.copy_(concentrated[0])
    static_sorted_experts.copy_(concentrated[1])
    static_num_valid.copy_(concentrated[2])
    actual_q.fill_(0xFF)
    actual_scale.fill_(0)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    valid_rows = int(concentrated[2][0].item())
    valid_scale_bytes = valid_rows * (intermediate_size // 32)
    torch.testing.assert_close(
        actual_scale.view(-1)[:valid_scale_bytes],
        expected_scale.view(-1)[:valid_scale_bytes],
        rtol=0,
        atol=0,
    )


def test_package_stage1_quantized_epilogue_rejects_i_r_128():
    hidden = torch.empty((1, 128), dtype=torch.uint8, device="cuda")
    weight = torch.empty((1, 256, 128), dtype=torch.uint8, device="cuda")
    weight_scale = torch.empty((1, 256, 8), dtype=torch.uint8, device="cuda")
    activation_scale = torch.empty((128, 8), dtype=torch.uint8, device="cuda")
    sorted_ids = torch.empty(128, dtype=torch.int32, device="cuda")
    sorted_experts = torch.empty(1, dtype=torch.int32, device="cuda")
    num_valid = torch.empty(2, dtype=torch.int32, device="cuda")
    out = torch.empty((1, 64), dtype=torch.uint8, device="cuda")
    out_scale = torch.empty((128, 4), dtype=torch.uint8, device="cuda")

    with pytest.raises(AssertionError, match="I_r must be divisible by 256"):
        invoke_gluon_mxfp4_moe_stage1(
            hidden,
            weight,
            None,
            sorted_ids,
            sorted_experts,
            num_valid,
            out,
            1,
            w1_scale=weight_scale,
            a1_scale=activation_scale,
            b_preshuffled=True,
            dst_type=torch.uint8,
            out_scale=out_scale,
        )


def test_package_prefill_falls_back_for_i_r_128():
    num_tokens = 16
    num_experts = 4
    hidden_size = 256
    intermediate_size = 128
    topk = 2
    device = "cuda"
    hidden = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    router = torch.empty((num_tokens, num_experts), dtype=torch.bfloat16, device=device)
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device=device)
    topk_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, dtype=torch.float32, device=device
    )
    w13_weight = torch.empty(1, dtype=torch.uint8, device=device)
    w2_weight = torch.empty(1, dtype=torch.uint8, device=device)
    w13_weight.gluon_package_prefill_weight = torch.empty(
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_weight.gluon_package_prefill_scale = torch.empty(
        1, dtype=torch.uint8, device=device
    )
    w2_weight.gluon_package_prefill_weight = torch.empty(
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_weight.gluon_package_prefill_scale = torch.empty(
        1, dtype=torch.uint8, device=device
    )

    out = fused_mxfp_gfx950._maybe_gluon_package_mxfp4_prefill(
        hidden,
        router,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_weight.gluon_package_prefill_scale,
        w2_mx_scale=w2_weight.gluon_package_prefill_scale,
        top_k=topk,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        routing_method_type=0,
        precomputed_topk_weights=topk_weights,
        precomputed_topk_ids=topk_ids,
        out_dtype=torch.bfloat16,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
    )

    assert out is None


def test_package_stage1_quantized_epilogue_dst_type_matches_storage():
    hidden = torch.empty((1, 128), dtype=torch.uint8, device="cuda")
    weight = torch.empty((1, 512, 128), dtype=torch.uint8, device="cuda")
    weight_scale = torch.empty((1, 512, 8), dtype=torch.uint8, device="cuda")
    activation_scale = torch.empty((128, 8), dtype=torch.uint8, device="cuda")
    sorted_ids = torch.empty(128, dtype=torch.int32, device="cuda")
    sorted_experts = torch.empty(1, dtype=torch.int32, device="cuda")
    num_valid = torch.empty(2, dtype=torch.int32, device="cuda")
    out = torch.empty((1, 128), dtype=torch.uint8, device="cuda")
    out_scale = torch.empty((128, 8), dtype=torch.uint8, device="cuda")

    with pytest.raises(NotImplementedError, match="torch.uint8"):
        invoke_gluon_mxfp4_moe_stage1(
            hidden,
            weight,
            None,
            sorted_ids,
            sorted_experts,
            num_valid,
            out,
            1,
            w1_scale=weight_scale,
            a1_scale=activation_scale,
            b_preshuffled=True,
            dst_type=torch.bfloat16,
            out_scale=out_scale,
        )


@pytest.mark.parametrize(
    "normalize, rsf, expect_bail",
    [
        (False, 2.5, True),  # divergent: grouped-one-group, unnormalized, rsf!=1
        (True, 2.5, False),  # normalized -> gluon kernel matches generic
        (False, 1.0, False),  # rsf==1 -> scaling is a no-op, matches generic
    ],
)
def test_route_owned_bails_on_unnormalized_grouped_scaling(
    normalize: bool,
    rsf: float,
    expect_bail: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Route-owned decode must defer the grouped-one-group unnormalized-scaling
    case to the generic path.

    For n_group == topk_group == 1 with no correction bias, the generic path
    (default_grouped_route -> _grouped_topk_reference) does NOT apply
    routed_scaling_factor when weights are unnormalized
    (scale_when_unnormalized=False), but invoke_softmax_topk_route_gluon always
    multiplies by ROUTED_SCALING_FACTOR. Route-owned decode must return None
    (bail to generic) in that config, and must NOT bail when the config is safe
    (normalized, or routed_scaling_factor == 1).
    """
    n_tokens = 8
    hidden = torch.empty((n_tokens, 8), device="cuda", dtype=torch.bfloat16)
    router = torch.randn((n_tokens, 8), device="cuda", dtype=torch.bfloat16)
    dummy_w = torch.empty((1, 4, 4), device="cuda", dtype=torch.uint8)
    dummy_scale = torch.empty((1, 4, 1), device="cuda", dtype=torch.uint8)

    route_called = 0

    def spy_softmax_route(*args, **kwargs):
        nonlocal route_called
        route_called += 1
        # Return shape-correct dummies so the (unused for this assertion)
        # downstream decode returns None on the dummy weights.
        ids = torch.zeros((n_tokens, 2), device="cuda", dtype=torch.int32)
        wts = torch.ones((n_tokens, 2), device="cuda", dtype=torch.float32)
        return ids, wts

    import tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.routing as routing_mod

    monkeypatch.setattr(
        routing_mod, "invoke_softmax_topk_route_gluon", spy_softmax_route
    )

    out = fused_mxfp_gfx950._maybe_route_owned_mxfp4_mfma_decode(
        hidden,
        router,
        dummy_w,
        dummy_w,
        w13_mx_scale=dummy_scale,
        w2_mx_scale=dummy_scale,
        top_k=2,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=rsf,
        normalize_topk_weights=normalize,
        routing_method_type=0,
        w13_bias=None,
        w2_bias=None,
        out_dtype=torch.bfloat16,
        max_m=fused_mxfp_gfx950._DECODE_MAX_M,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
        allow_generic_fallback=False,
    )

    if expect_bail:
        # Guard fired before routing: no gluon route call, returns None.
        assert out is None
        assert route_called == 0
    else:
        # Safe config: the guard did not fire, so the gluon route ran (the
        # decode then returns None on the dummy weights, which is fine here --
        # we only assert the guard let it through to routing).
        assert route_called == 1


@pytest.mark.parametrize("num_tokens", [4, 8])
def test_route_owned_grouped_unnormalized_matches_generic(num_tokens: int):
    """End-to-end: the divergent config must match the generic path exactly.

    With the guard in place, the grouped-one-group unnormalized-scaling config
    is deferred to the generic route, so gluon_mxfp_dynamic_mxfp4_fused_moe must
    produce the same result as the generic ragged path (decode disabled).
    """
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    topk = 2
    rsf = 2.5
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(20260716 + num_tokens)
    hidden = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.01
    ).contiguous()
    router = torch.randn(
        (num_tokens, num_experts), device=device, dtype=torch.bfloat16, generator=gen
    )
    w13, w13_scale, w2, w2_scale = _make_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    layer = _make_preprocessed_layer(
        w13,
        w13_scale,
        w2,
        w2_scale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )

    def run():
        return fused_mxfp_gfx950.gluon_mxfp_dynamic_mxfp4_fused_moe(
            hidden,
            router,
            layer.w13_weight_triton_tensor,
            layer.w2_weight_triton_tensor,
            w13_mx_scale=layer.w13_precision_config.b_mx_scale,
            w2_mx_scale=layer.w2_precision_config.b_mx_scale,
            top_k=topk,
            correction_bias=None,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=rsf,
            normalize_topk_weights=False,
            routing_method_type=0,
            w13_bias=None,
            w2_bias=None,
        )

    # Reference: force the fully-generic path (no decode fast path).
    import unittest.mock as _mock

    with _mock.patch.object(
        fused_mxfp_gfx950,
        "_maybe_route_owned_mxfp4_mfma_decode",
        lambda *a, **k: None,
    ):
        ref = run()
        torch.cuda.synchronize()

    # Real path: the guard should defer this config to the same generic route.
    out = run()
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), rtol=0.0, atol=0.0)
