"""Health / metrics helpers for the /health dashboard.

Pure derivation math plus an optional live system-resource sampler. Extracted
from the monolith so the logic (real-time factor, throughput, status
thresholds, uptime formatting) is importable and unit-tested without the
monolith's import-time side effects.

The module imports clean with stdlib only: ``psutil`` and ``torch`` are heavy,
optional runtime deps imported lazily inside ``sample_system_resources`` behind
``try/except ImportError``. When they are absent every resource field degrades
to ``None`` rather than raising, so CI (which installs no ML deps) and hosts
without ``psutil`` still import and exercise this module.

All thresholds live here — not in the template — so the colour a metric gets is
decided in one tested place. Statuses use the four-value vocabulary the
dashboard paints: ``healthy`` / ``degraded`` / ``error`` (plus ``unknown`` for
a missing value).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, Optional

# Status vocabulary — maps 1:1 onto DESIGN.md's status colours
# (green / amber / red, with grey for "no data").
STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_ERROR = "error"
STATUS_UNKNOWN = "unknown"

# Real-time factor: transcribe_seconds / audio_seconds. <1 keeps up with live
# audio; >1 falls behind. These bounds drive the headline transcription colour.
RTF_DEGRADED_ABOVE = 1.0
RTF_ERROR_ABOVE = 1.5

# A default EMA smoothing factor: weight of the newest sample. 0.3 reacts within
# a handful of segments without letting a single slow chunk spike the display.
DEFAULT_ALPHA = 0.3


def _ema(prev: Optional[float], sample: float, alpha: float) -> float:
    """Exponential moving average. Seeds with the first sample when no prior."""
    if prev is None:
        return sample
    return alpha * sample + (1.0 - alpha) * prev


def update_perf_ema(
    prev: Optional[Dict[str, Any]],
    inference_ms: float,
    audio_seconds: float,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Fold one transcribe timing into the running performance state.

    ``prev`` is the prior state dict (or ``None`` to start fresh),
    ``inference_ms`` how long the transcribe call took, and ``audio_seconds``
    the duration of audio it processed. Returns a new plain dict with the
    updated inference-ms EMA, real-time-factor EMA, and total segment count.

    A non-positive ``audio_seconds`` (unknown/empty chunk) still counts the
    segment and updates the inference EMA but leaves the RTF EMA untouched,
    since RTF would be undefined.
    """
    prev = prev or {}
    infer_ema = _ema(prev.get("infer_ms_ema"), float(inference_ms), alpha)

    rtf_ema = prev.get("rtf_ema")
    if audio_seconds > 0:
        rtf_sample = (inference_ms / 1000.0) / audio_seconds
        rtf_ema = _ema(rtf_ema, rtf_sample, alpha)

    return {
        "infer_ms_ema": round(infer_ema, 1),
        "rtf_ema": None if rtf_ema is None else round(rtf_ema, 3),
        "segments_total": int(prev.get("segments_total", 0)) + 1,
    }


def segments_per_minute(
    segments_total: int, first_ts: Optional[float], now_ts: float
) -> Optional[float]:
    """Segments processed per minute since ``first_ts``.

    Returns ``None`` when there is no elapsed window yet (no first timestamp,
    or ``now_ts`` not after it) so the caller can show "n/a" instead of a
    divide-by-zero.
    """
    if not first_ts or now_ts <= first_ts:
        return None
    minutes = (now_ts - first_ts) / 60.0
    if minutes <= 0:
        return None
    return round(segments_total / minutes, 1)


def rtf_status(rtf: Optional[float]) -> str:
    """Colour status for a real-time factor value."""
    if rtf is None:
        return STATUS_UNKNOWN
    if rtf > RTF_ERROR_ABOVE:
        return STATUS_ERROR
    if rtf > RTF_DEGRADED_ABOVE:
        return STATUS_DEGRADED
    return STATUS_HEALTHY


def fraction_status(
    used: Optional[float],
    total: Optional[float],
    degraded_above: float = 0.8,
    error_above: float = 0.95,
) -> str:
    """Colour status for a used/total ratio (RAM, VRAM, disk, error-rate).

    ``degraded_above``/``error_above`` are fractions of ``total``. Returns
    ``unknown`` when either operand is missing or ``total`` is non-positive.
    """
    if used is None or total is None or total <= 0:
        return STATUS_UNKNOWN
    ratio = used / total
    if ratio > error_above:
        return STATUS_ERROR
    if ratio > degraded_above:
        return STATUS_DEGRADED
    return STATUS_HEALTHY


def format_uptime(seconds: Optional[float]) -> str:
    """Human-readable uptime, e.g. ``3d 4h 12m`` or ``5m 3s``."""
    if seconds is None or seconds < 0:
        return "—"
    secs = int(seconds)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _nvidia_smi_query() -> Optional[Dict[str, Optional[float]]]:
    """GPU utilisation + used VRAM (bytes) via ``nvidia-smi``, or ``None``.

    Fallback for hosts where ``torch`` is not importable but an NVIDIA driver
    is present. Mirrors the CSV query style used by the monolith's VRAM probe.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout.strip().splitlines()
        if not out:
            return None
        util_str, used_mib_str = (p.strip() for p in out[0].split(","))
        return {
            "gpu_util_pct": float(util_str),
            "vram_used_bytes": float(used_mib_str) * 1024 * 1024,
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def sample_system_resources() -> Dict[str, Optional[float]]:
    """Live resource utilisation snapshot; every field degrades to ``None``.

    - CPU percent and used/total RAM come from ``psutil`` when installed.
    - GPU utilisation and used VRAM come from ``torch.cuda`` when torch is
      already usable, else from ``nvidia-smi``.

    Never raises: a missing dependency or a probe failure yields ``None`` for
    that field so the dashboard shows "n/a" and stays up.
    """
    result: Dict[str, Optional[float]] = {
        "cpu_pct": None,
        "ram_used_bytes": None,
        "ram_total_bytes": None,
        "gpu_util_pct": None,
        "vram_used_bytes": None,
    }

    try:
        import psutil  # type: ignore

        result["cpu_pct"] = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        result["ram_used_bytes"] = float(vm.used)
        result["ram_total_bytes"] = float(vm.total)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            result["vram_used_bytes"] = float(torch.cuda.memory_reserved(0))
            try:
                util = torch.cuda.utilization(0)  # needs pynvml; optional
                result["gpu_util_pct"] = float(util)
            except Exception:
                pass
    except Exception:
        pass

    # If torch could not supply GPU numbers, try nvidia-smi as a last resort.
    if result["vram_used_bytes"] is None or result["gpu_util_pct"] is None:
        smi = _nvidia_smi_query()
        if smi:
            if result["gpu_util_pct"] is None:
                result["gpu_util_pct"] = smi.get("gpu_util_pct")
            if result["vram_used_bytes"] is None:
                result["vram_used_bytes"] = smi.get("vram_used_bytes")

    return result
