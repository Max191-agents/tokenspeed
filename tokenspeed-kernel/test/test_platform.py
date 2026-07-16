from types import SimpleNamespace

import pytest

from tokenspeed_kernel import platform


@pytest.fixture(autouse=True)
def clear_hip_runtime_cache():
    platform._get_hip_runtime.cache_clear()
    yield
    platform._get_hip_runtime.cache_clear()


class _FakeFunction:
    argtypes = None
    restype = None


def _fake_hip_runtime():
    return SimpleNamespace(
        hipHostGetDevicePointer=_FakeFunction(),
        hipGetErrorString=_FakeFunction(),
    )


def test_get_hip_runtime_prefers_torch_dependency_scope(monkeypatch):
    runtime = _fake_hip_runtime()
    calls = []

    def fake_cdll(candidate):
        calls.append(candidate)
        return runtime

    monkeypatch.setattr(platform.ctypes, "CDLL", fake_cdll)

    assert platform._get_hip_runtime() is runtime
    assert calls == [str(platform.Path(platform.torch._C.__file__).resolve())]
    assert runtime.hipHostGetDevicePointer.argtypes == [
        platform.ctypes.POINTER(platform.ctypes.c_void_p),
        platform.ctypes.c_void_p,
        platform.ctypes.c_uint,
    ]
    assert runtime.hipHostGetDevicePointer.restype is platform.ctypes.c_int
    assert runtime.hipGetErrorString.argtypes == [platform.ctypes.c_int]
    assert runtime.hipGetErrorString.restype is platform.ctypes.c_char_p


def test_get_hip_runtime_skips_scope_without_hip_symbols(monkeypatch):
    runtime = _fake_hip_runtime()
    calls = []

    def fake_cdll(candidate):
        calls.append(candidate)
        if len(calls) == 1:
            return SimpleNamespace()
        return runtime

    monkeypatch.setattr(platform.ctypes, "CDLL", fake_cdll)

    assert platform._get_hip_runtime() is runtime
    assert calls == [
        str(platform.Path(platform.torch._C.__file__).resolve()),
        None,
    ]


def test_get_hip_runtime_does_not_load_another_runtime(monkeypatch):
    calls = []

    def fake_cdll(candidate):
        calls.append(candidate)
        return SimpleNamespace()

    monkeypatch.setattr(platform.ctypes, "CDLL", fake_cdll)

    with pytest.raises(RuntimeError, match="Failed to load libamdhip64.so"):
        platform._get_hip_runtime()
    assert calls == [
        str(platform.Path(platform.torch._C.__file__).resolve()),
        None,
    ]
