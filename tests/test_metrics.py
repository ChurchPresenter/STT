"""Unit tests for stt.metrics — health-dashboard derivation helpers."""

import stt.metrics as metrics
from stt.metrics import (
    STATUS_DEGRADED,
    STATUS_ERROR,
    STATUS_HEALTHY,
    STATUS_UNKNOWN,
    format_uptime,
    fraction_status,
    rtf_status,
    sample_system_resources,
    segments_per_minute,
    update_perf_ema,
)


# --- update_perf_ema ---------------------------------------------------------

def test_update_perf_ema_seeds_from_first_sample():
    state = update_perf_ema(None, inference_ms=200.0, audio_seconds=2.0)
    assert state["infer_ms_ema"] == 200.0
    # RTF = 0.2s / 2.0s = 0.1
    assert state["rtf_ema"] == 0.1
    assert state["segments_total"] == 1


def test_update_perf_ema_accumulates_and_smooths():
    s1 = update_perf_ema(None, inference_ms=100.0, audio_seconds=1.0)
    s2 = update_perf_ema(s1, inference_ms=300.0, audio_seconds=1.0, alpha=0.5)
    # inference EMA: 0.5*300 + 0.5*100 = 200
    assert s2["infer_ms_ema"] == 200.0
    # rtf samples: 0.1 then 0.3 -> 0.5*0.3 + 0.5*0.1 = 0.2
    assert s2["rtf_ema"] == 0.2
    assert s2["segments_total"] == 2


def test_update_perf_ema_zero_audio_counts_segment_but_keeps_rtf():
    s1 = update_perf_ema(None, inference_ms=100.0, audio_seconds=1.0)
    s2 = update_perf_ema(s1, inference_ms=500.0, audio_seconds=0.0)
    assert s2["segments_total"] == 2
    # RTF untouched (still the prior single-sample value 0.1)
    assert s2["rtf_ema"] == s1["rtf_ema"]
    # inference EMA still moves
    assert s2["infer_ms_ema"] != s1["infer_ms_ema"]


def test_update_perf_ema_first_sample_zero_audio_leaves_rtf_none():
    state = update_perf_ema(None, inference_ms=120.0, audio_seconds=0.0)
    assert state["rtf_ema"] is None
    assert state["segments_total"] == 1


# --- segments_per_minute -----------------------------------------------------

def test_segments_per_minute_basic():
    # 30 segments over 60 seconds -> 30/min
    assert segments_per_minute(30, first_ts=1000.0, now_ts=1060.0) == 30.0


def test_segments_per_minute_no_window_returns_none():
    assert segments_per_minute(5, first_ts=None, now_ts=1000.0) is None
    assert segments_per_minute(5, first_ts=1000.0, now_ts=1000.0) is None
    assert segments_per_minute(5, first_ts=1000.0, now_ts=999.0) is None


# --- rtf_status --------------------------------------------------------------

def test_rtf_status_thresholds():
    assert rtf_status(None) == STATUS_UNKNOWN
    assert rtf_status(0.5) == STATUS_HEALTHY
    assert rtf_status(1.0) == STATUS_HEALTHY   # boundary is inclusive-healthy
    assert rtf_status(1.2) == STATUS_DEGRADED
    assert rtf_status(1.5) == STATUS_DEGRADED
    assert rtf_status(2.0) == STATUS_ERROR


# --- fraction_status ---------------------------------------------------------

def test_fraction_status_thresholds():
    assert fraction_status(None, 100) == STATUS_UNKNOWN
    assert fraction_status(50, None) == STATUS_UNKNOWN
    assert fraction_status(50, 0) == STATUS_UNKNOWN
    assert fraction_status(50, 100) == STATUS_HEALTHY
    assert fraction_status(85, 100) == STATUS_DEGRADED
    assert fraction_status(99, 100) == STATUS_ERROR


def test_fraction_status_custom_bounds():
    assert fraction_status(6, 100, degraded_above=0.05, error_above=0.1) == STATUS_DEGRADED
    assert fraction_status(20, 100, degraded_above=0.05, error_above=0.1) == STATUS_ERROR


# --- format_uptime -----------------------------------------------------------

def test_format_uptime_variants():
    assert format_uptime(None) == "—"
    assert format_uptime(-1) == "—"
    assert format_uptime(5) == "5s"
    assert format_uptime(65) == "1m 5s"
    assert format_uptime(3661) == "1h 1m"
    assert format_uptime(90061) == "1d 1h 1m"


# --- sample_system_resources -------------------------------------------------

def test_sample_system_resources_all_none_when_deps_absent(monkeypatch):
    # Force every optional import to fail, and nvidia-smi to be absent.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("psutil", "torch"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(metrics.shutil, "which", lambda _exe: None)

    result = sample_system_resources()
    assert result == {
        "cpu_pct": None,
        "ram_used_bytes": None,
        "ram_total_bytes": None,
        "gpu_util_pct": None,
        "vram_used_bytes": None,
        "gpu_kind": None,
    }


def test_sample_system_resources_never_raises():
    # With whatever is (or isn't) installed in the test env, it must return a
    # dict with the full key set and never raise.
    result = sample_system_resources()
    assert set(result) == {
        "cpu_pct", "ram_used_bytes", "ram_total_bytes",
        "gpu_util_pct", "vram_used_bytes", "gpu_kind",
    }


def test_sample_system_resources_detects_mps(monkeypatch):
    # Fake an Apple-Silicon torch: no CUDA, MPS available with allocated memory.
    import builtins
    import types

    fake_torch = types.SimpleNamespace()
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True))
    fake_torch.mps = types.SimpleNamespace(current_allocated_memory=lambda: 2048.0)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        if name == "psutil":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(metrics.shutil, "which", lambda _exe: None)

    result = sample_system_resources()
    assert result["gpu_kind"] == "mps"
    assert result["vram_used_bytes"] == 2048.0
    assert result["gpu_util_pct"] is None  # no utilisation counter on MPS


def test_sample_system_resources_cuda_uses_device_wide_memory(monkeypatch):
    # Fake a CUDA torch where the process's PyTorch allocator holds nothing
    # (memory_reserved == 0) but the device is 8 GiB total / 2 GiB free — as
    # happens when faster-whisper/CTranslate2 owns the VRAM. Used must reflect
    # the device-wide figure (6 GiB), not the process-scoped 0.
    import builtins
    import types

    gib = 1024 ** 3
    fake_torch = types.SimpleNamespace()
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda _dev: (2 * gib, 8 * gib),
        memory_reserved=lambda _dev: 0,
        utilization=lambda _dev: 99,
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        if name == "psutil":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(metrics.shutil, "which", lambda _exe: None)

    result = sample_system_resources()
    assert result["gpu_kind"] == "cuda"
    assert result["vram_used_bytes"] == float(6 * gib)
    assert result["gpu_util_pct"] == 99.0


def test_sample_system_resources_cuda_falls_back_when_mem_get_info_unavailable(monkeypatch):
    # Old torch without mem_get_info: fall back to the process-scoped figure.
    import builtins
    import types

    def _raise(_dev):
        raise AttributeError("mem_get_info")

    fake_torch = types.SimpleNamespace()
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=_raise,
        memory_reserved=lambda _dev: 4096.0,
        utilization=lambda _dev: 50,
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        if name == "psutil":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(metrics.shutil, "which", lambda _exe: None)

    result = sample_system_resources()
    assert result["gpu_kind"] == "cuda"
    assert result["vram_used_bytes"] == 4096.0


def test_sample_system_resources_uses_nvidia_smi_fallback(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("psutil", "torch"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        metrics, "_nvidia_smi_query",
        lambda: {"gpu_util_pct": 42.0, "vram_used_bytes": 1024.0},
    )
    result = sample_system_resources()
    assert result["gpu_util_pct"] == 42.0
    assert result["vram_used_bytes"] == 1024.0
