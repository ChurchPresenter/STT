"""Model memory footprint estimation + system-requirements warnings.

Pure config-and-hardware math: given the live config, a probed hardware
snapshot, and the models directory, work out what the configured models need
and produce human-readable warnings when the machine falls short. The hardware
probing itself (RAM/VRAM/cores) stays in the monolith and is passed in as `hw`,
and the models directory is passed as `models_dir`, so this module reads no
monolith globals and imports clean with stdlib only.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

BASELINE_RAM_GB = 4.0   # OS + Python/torch runtime + Flask + audio buffers
GPU_HOST_RAM_GB = 1.0   # host-side RAM per GPU-resident model (weight load/copy, tokenizer)
BASE_DISK_GB = 5.0      # logs, session databases, temp audio, headroom

MODEL_MEMORY_ESTIMATES: Dict[str, Dict[str, Dict[str, float]]] = {
    "whisper": {
        "tiny":   {"ram": 0.6,  "vram": 0.6,  "disk": 0.075},
        "base":   {"ram": 0.8,  "vram": 0.8,  "disk": 0.15},
        "small":  {"ram": 2.5,  "vram": 2.0,  "disk": 0.5},
        "medium": {"ram": 5.0,  "vram": 5.0,  "disk": 1.5},
        "large":  {"ram": 10.0, "vram": 10.0, "disk": 3.0},
        "turbo":  {"ram": 6.0,  "vram": 6.0,  "disk": 1.6},
    },
    "faster-whisper": {  # CTranslate2: int8 on CPU, fp16 on GPU
        "tiny":   {"ram": 0.3, "vram": 0.5, "disk": 0.075},
        "base":   {"ram": 0.4, "vram": 0.6, "disk": 0.15},
        "small":  {"ram": 1.2, "vram": 1.5, "disk": 0.5},
        "medium": {"ram": 2.5, "vram": 3.0, "disk": 1.5},
        "large":  {"ram": 4.5, "vram": 4.5, "disk": 3.0},
        "turbo":  {"ram": 2.5, "vram": 3.0, "disk": 1.6},
    },
    "nllb": {
        "nllb-200-distilled-600M": {"ram": 3.0,  "vram": 2.0,  "disk": 1.2},
        "nllb-200-distilled-1.3B": {"ram": 6.5,  "vram": 3.5,  "disk": 2.6},
        "nllb-200-1.3B":           {"ram": 6.5,  "vram": 3.5,  "disk": 5.2},
        "nllb-200-3.3B":           {"ram": 15.0, "vram": 8.0,  "disk": 13.0},
        "nllb-moe-54b":            {"ram": 60.0, "vram": 30.0, "disk": 17.0},
    },
}


def _normalize_whisper_size(name: Any) -> Optional[str]:
    """Map a configured Whisper model name to a MODEL_MEMORY_ESTIMATES size key.

    Strips the .en suffix, folds large-v1/v2/v3 into "large" and the turbo /
    distilled variants into "turbo". Unknown names -> None (contributes nothing).
    """
    if not isinstance(name, str) or not name.strip():
        return None
    n = name.strip().lower()
    if n.endswith(".en"):
        n = n[:-3]
    if n in ("turbo", "large-v3-turbo", "distil-large-v3"):
        return "turbo"
    if n in ("large", "large-v1", "large-v2", "large-v3"):
        return "large"
    if n in ("tiny", "base", "small", "medium"):
        return n
    return None


def estimate_memory_requirements(
    cfg: Dict[str, Any],
    models_dir: str,
    *,
    gpu_available: bool = False,
    unified: bool = False,
) -> Dict[str, Any]:
    """Pure config math: what the configured models need to run.

    Returns {ram_gb, vram_gb, disk_gb, min_cores, tier, parts}; parts is a
    per-model breakdown for building warning messages. Unknown model names or
    malformed config contribute nothing (fail open). `gpu_available` decides
    whether use_gpu flags put a model on the GPU (vram) or the CPU (ram).
    `unified` (Apple Silicon) means CPU and GPU share one pool, so there's no
    separate host-side copy — the per-model GPU_HOST_RAM_GB is skipped.
    """
    ram_gb = BASELINE_RAM_GB
    vram_gb = 0.0
    disk_gb = BASE_DISK_GB
    parts: List[Dict[str, Any]] = [{"label": "app/OS baseline", "ram_gb": BASELINE_RAM_GB, "vram_gb": 0.0}]
    seen = set()

    def _add(label: str, table: str, key: str, use_gpu: Any) -> None:
        nonlocal ram_gb, vram_gb, disk_gb
        est = MODEL_MEMORY_ESTIMATES.get(table, {}).get(key)
        if not est:
            return
        on_gpu = bool(use_gpu) and gpu_available
        dedupe = (table, key, on_gpu)
        if dedupe in seen:
            return
        seen.add(dedupe)
        part = {"label": label + (" on GPU" if on_gpu else " on CPU"), "ram_gb": 0.0, "vram_gb": 0.0}
        if on_gpu:
            vram_gb += est["vram"]
            part["vram_gb"] = est["vram"]
            if not unified:  # discrete GPU keeps a host-side copy; unified doesn't
                ram_gb += GPU_HOST_RAM_GB
                part["ram_gb"] = GPU_HOST_RAM_GB
        else:
            ram_gb += est["ram"]
            part["ram_gb"] = est["ram"]
        disk_gb += est["disk"]
        parts.append(part)

    def _add_whisper(label: str, model_cfg: Any, backend: Any, use_gpu: Any) -> None:
        if not isinstance(model_cfg, dict) or model_cfg.get("type", "whisper") != "whisper":
            return  # huggingface/custom models: no estimate, fail open
        name = model_cfg.get("whisper", {}).get("model")
        size = _normalize_whisper_size(name)
        if size is None:
            return
        table = "faster-whisper" if backend == "faster-whisper" else "whisper"
        _add(f"{label} '{name}' via {table}", table, size, use_gpu)

    main_backend = "whisper"
    try:
        main_backend = cfg.get("model", {}).get("backend") or "whisper"
        _add_whisper("Whisper", cfg.get("model", {}),
                     main_backend, cfg.get("performance", {}).get("use_gpu"))
    except Exception:
        pass
    try:
        ft = cfg.get("file_transcription", {})
        ft_backend = ft.get("model", {}).get("backend") or main_backend
        _add_whisper("file-transcription Whisper", ft.get("model", {}),
                     ft_backend, ft.get("use_gpu"))
    except Exception:
        pass

    local_translation = False
    try:
        _lt = cfg.get("live_translation", {})
        _remote = _lt.get("remote", {})
        _enabled_local = bool(_lt.get("enabled")) and not (_remote.get("enabled") and _remote.get("endpoint"))
        # Only count local translation toward the requirement when it's actually
        # set up — enabled AND its model downloaded. Translation is on by default
        # in config, so counting it before a model exists over-warned ("minimum 8
        # for transcription + translation") on machines not doing translation.
        _model_full = str(_lt.get("translation_model", ""))
        _model_dir = os.path.join(models_dir, _model_full.replace("/", "--")) if _model_full else ""
        _model_present = bool(_model_dir) and os.path.isdir(_model_dir)
        local_translation = _enabled_local and _model_present
        if local_translation:
            model_id = _model_full.split("/")[-1]
            _add(f"NLLB '{model_id}'", "nllb", model_id, _lt.get("use_gpu"))
    except Exception:
        pass

    if local_translation:
        min_cores, tier = 8, "transcription + translation"
    else:
        min_cores, tier = 6, "transcription"
    return {"ram_gb": ram_gb, "vram_gb": vram_gb, "disk_gb": disk_gb,
            "min_cores": min_cores, "tier": tier, "parts": parts}


def _largest_fitting_whisper(avail_gb: float, on_gpu: bool = False, table: str = "faster-whisper") -> Optional[str]:
    """Largest Whisper size whose footprint (OS/runtime baseline + the model)
    fits in avail_gb, or None if not even 'tiny' fits. Uses the faster-whisper
    (CTranslate2 int8) estimates by default — the recommended, lighter backend —
    so 'what fits' reflects the best-case setup."""
    key = "vram" if on_gpu else "ram"
    base = 0.0 if on_gpu else BASELINE_RAM_GB  # unified: model shares the pool, no host copy
    for size in ("large", "turbo", "medium", "small", "base", "tiny"):
        est = MODEL_MEMORY_ESTIMATES.get(table, {}).get(size)
        if est and base + est.get(key, 0) <= avail_gb:
            return size
    return None


def _memory_advice(avail_gb: float, on_gpu: bool = False) -> str:
    """A short 'here's what fits' hint appended to a memory-shortfall warning,
    steering toward faster-whisper (much lighter than openai-whisper)."""
    fit = _largest_fitting_whisper(avail_gb, on_gpu=on_gpu)
    if fit:
        return f" With faster-whisper (lighter), the largest model that fits is '{fit}'."
    key = "vram" if on_gpu else "ram"
    base = 0.0 if on_gpu else BASELINE_RAM_GB
    model = MODEL_MEMORY_ESTIMATES["faster-whisper"]["tiny"][key]
    total = base + model
    # Name the model's own size, not the total — the total (app baseline +
    # model) is what exceeds the machine, but 'tiny' itself is tiny.
    return (f" Even the lightest setup — STT (~{base:.0f} GB) plus faster-whisper "
            f"'tiny' (~{model:.1f} GB) = ~{total:.1f} GB — is too big; a machine "
            f"with more memory is needed.")


def _format_parts(parts: List[Dict[str, Any]], key: str) -> str:
    """'Whisper 'small' via faster-whisper on CPU: ~1.2 GB; app/OS baseline: ~4 GB'."""
    out = []
    for p in parts:
        if p.get(key):
            out.append(f"{p['label']}: ~{p[key]:g} GB")
    return "; ".join(out)


def check_system_requirements(cfg: Dict[str, Any], hw: Dict[str, Any], models_dir: str) -> List[str]:
    """Human-readable warnings when the hardware falls short of what the
    currently configured models need; [] when OK.

    cfg (live config), hw (cached hardware probe) and models_dir are injected by
    the caller. Each check fails open (unknown -> no warning) and must never
    block startup.
    """
    try:
        gpu_available = bool(hw.get("has_cuda") or hw.get("apple_silicon"))
        need = estimate_memory_requirements(cfg, models_dir, gpu_available=gpu_available,
                                            unified=bool(hw.get("apple_silicon")))
    except Exception:
        return []

    found: List[str] = []
    try:
        cores = hw.get("cpu_cores") or 0
        if 0 < cores < need["min_cores"]:
            found.append(f"{cores} CPU cores (minimum {need['min_cores']} for {need['tier']})")
    except Exception:
        pass
    try:
        ram = hw.get("ram_bytes") or 0
        if hw.get("apple_silicon"):
            # Unified memory: CPU and GPU draw from the same pool.
            need_gb = need["ram_gb"] + need["vram_gb"]
            avail = ram / 1024**3
            if 0 < ram < (need_gb - 0.5) * 1024**3:  # installed RAM reports slightly less
                # Severity scales with the shortfall: a big overshoot won't just
                # be slow, the model likely won't load and the app can crash.
                sev = "the model likely won't load and STT may crash" if need_gb > avail * 1.5 \
                    else "transcription may be slow or unstable"
                found.append(f"STT plus the configured models need ~{need_gb:.1f} GB (incl. ~{BASELINE_RAM_GB:.0f} GB app/OS baseline) "
                             f"of the {avail:.1f} GB unified memory on this Apple Silicon Mac "
                             f"(CPU and GPU share one pool) — {sev}.{_memory_advice(avail, on_gpu=False)}")
        else:
            avail = ram / 1024**3
            if 0 < ram < (need["ram_gb"] - 0.5) * 1024**3:
                sev = "models likely won't load and STT may crash" if need["ram_gb"] > avail * 1.5 \
                    else "transcription may be slow or unstable"
                found.append(f"STT plus the configured models need ~{need['ram_gb']:.1f} GB RAM ({_format_parts(need['parts'], 'ram_gb')}) "
                             f"but this PC has {avail:.1f} GB — {sev}.{_memory_advice(avail, on_gpu=False)}")
    except Exception:
        pass
    try:
        vram = hw.get("vram_bytes")
        if (hw.get("has_cuda") and not hw.get("apple_silicon") and vram
                and need["vram_gb"] > 0 and vram < (need["vram_gb"] - 0.5) * 1024**3):
            found.append(f"Configured models need ~{need['vram_gb']:.1f} GB VRAM ({_format_parts(need['parts'], 'vram_gb')}) "
                         f"but the GPU has {vram / 1024**3:.1f} GB — models may fail to load or fall back to CPU")
    except Exception:
        pass
    try:
        free = hw.get("disk_free_bytes") or 0
        if 0 < free < need["disk_gb"] * 1024**3:
            found.append(f"{free / 1024**3:.1f} GB free disk (~{need['disk_gb']:.0f} GB needed for STT + the configured models)")
    except Exception:
        pass
    return found
