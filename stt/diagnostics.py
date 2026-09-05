"""The operator-facing diagnostic report: what a support thread actually needs.

Issue #8 took a multi-day round trip to answer questions a machine could have
answered instantly — which model is selected, whether its files are actually
complete, how much RAM the box has, where startup stopped. This assembles that
into one document the operator can read, then attach to an issue themselves.

**It is never uploaded.** The report is built on request and handed back for the
operator to inspect and send on. A support bundle that posts itself is a support
bundle nobody can consent to.

Three rules shape everything here, because the material is unusually sensitive:

**Captions never appear.** ``logs/stt.log`` is the worker's raw stdout, and a
great many of its lines carry verbatim congregation speech — every translation
debug line, every SRT and sermon line. So the log filter is an **allowlist of
tags** (``LOG_TAGS``), not a denylist: a tag nobody has vetted is dropped. A new
``print`` in the monolith therefore fails closed, the way stt/demo_guard.py
treats a new route.

**Secrets never appear.** Likewise ``CONFIG_FIELDS`` is an allowlist of config
paths. config.json holds pair tokens, API keys and SMB credentials, and a
denylist would leak the first setting someone adds next to them.

**Names never appear.** Install paths contain the operator's username, so
everything is passed through the same ``redact_home_paths`` the crash reporter
uses.

Stdlib-only and free of IO: the caller reads the files, probes the hardware and
lists the directories, then passes plain data in. That keeps the redaction rules
— the part that actually matters — unit-testable without a machine to break.
"""

from __future__ import annotations

import re
from typing import (Any, Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple, Union)

from stt.crash_reports import redact_home_paths
from stt.model_disk import is_weight_file
from stt.model_files import REQUIRED_FASTER_WHISPER, describe_missing

#: Log tags whose lines are structurally operational and cannot carry caption
#: text. Deliberately excludes everything on the text path — LIVE-TRANSLATION,
#: TRANS-DBG, SRT, SRT-TRANSLATION, SERMON, LLM-TRANSLATE, LLM-LOCAL,
#: TRANSLATION, REMOTE_TRANSLATE, HTML, TTS, SESSION-META — each of which prints
#: what somebody said. When in doubt a tag stays out: a missing line costs a
#: follow-up question, a leaked line costs a congregation's privacy.
LOG_TAGS = frozenset({
    "AUDIO", "AUTH", "AUTO-UPDATE", "CALIBRATION", "CLEANUP", "CONFIG",
    "DB", "DB-CLEANUP", "DEBUG-TS-CMD", "DEBUG-TS-RESTART", "DEBUG-TS-START",
    "DEBUG-TS-STDERR", "DEBUG-TS-STOP", "DEBUG-TS-TIMEOUT", "DOWNLOAD", "ERROR",
    "EXECUTE", "FATAL", "FFMPEG", "INFO", "INIT", "LIVEMAP", "MIGRATION", "OK",
    "PAIR", "PANNS", "PERMS", "RESET-SESSION", "RESTART", "SENTRY", "SERVICE-PHASE",
    "SHUTDOWN", "START", "STOP", "TUNNEL", "VAD", "WARN", "WARNING", "WORKER",
})

#: Lines with no tag at all that are still worth keeping, matched on their exact
#: shape. A worker traceback is the single most useful thing in the file and has
#: no tag, so it is matched explicitly rather than by loosening the tag rule.
_UNTAGGED_KEEP = (
    re.compile(r"^\s*Process Process-\d+:"),
    re.compile(r"^\s*Traceback \(most recent call last\):"),
    re.compile(r'^\s*File "[^"]+", line \d+'),
    re.compile(r"^\s*[A-Za-z_.]*(Error|Exception|Interrupt|Exit)\b"),
    re.compile(r"^\s*Model loaded\.\s*$"),
)

_TAG_RE = re.compile(r"^\s*(?:\[[\d:. -]+\]\s*)?\[([A-Z0-9_-]+)\]")

#: Longest line kept. Operational lines are short; a very long one is a sign that
#: something unexpected (a payload, a dump) is riding along, so it is truncated.
_MAX_LINE = 400

#: Config paths worth reporting, as dotted paths. Everything else — tokens, keys,
#: endpoints, credentials, filesystem paths — is omitted by construction.
CONFIG_FIELDS: Tuple[str, ...] = (
    "transcription.backend",
    "transcription.model",
    "transcription.compute_type",
    "transcription.language",
    "transcription.use_gpu",
    "faster_whisper.model",
    "faster_whisper.compute_type",
    "whisper.model",
    "audio.default_microphone",
    "audio.energy_threshold",
    "audio.phrase_timeout",
    "audio.record_timeout",
    "audio.use_vad",
    "audio.vad_threshold",
    "audio.autostart",
    "translation.enabled",
    "translation.method",
    "translation.model",
    "translation.source_language",
    "translation.target_languages",
    "tts.enabled",
    "tts.backend",
    "service_phase.enabled",
    "service_phase.profile",
    "crash_reporting.sentry_enabled",
)

#: Files a loader needs, per model family. Each entry is a filename, or a tuple
#: of interchangeable filenames.
#:
#: faster-whisper's list is **imported, not restated**. This module previously
#: kept its own copy demanding ``vocabulary.json`` exactly, while the library
#: globs ``vocabulary.*`` and real Systran repos ship ``vocabulary.txt`` — so a
#: perfectly good model was reported broken, telling an operator to delete
#: something that worked. Two modules answering "is this loadable?" is the bug;
#: stt/model_files.py owns the answer and this one asks it.
#:
#: A GGUF has no companions at all — the single file is the whole model — so it
#: maps to an empty tuple rather than being left out, which would silently give
#: it the default family's requirements and report every LLM as broken.
REQUIRED_COMPANIONS: Dict[str, Tuple[Union[str, Tuple[str, ...]], ...]] = {
    "faster-whisper": tuple(
        entry if isinstance(entry, str) else tuple(entry)
        for entry in REQUIRED_FASTER_WHISPER
        if entry != "model.bin"  # the weights are judged separately, by size
    ),
    "nllb": ("config.json", "tokenizer.json"),
    "gguf": (),
    "whisper": (),
}

#: Approximate finished size of each faster-whisper weights file, in bytes, used
#: only to flag an implausibly small one. Deliberately loose: this catches a
#: transfer that died part-way, not a version that differs by a few percent.
WEIGHT_SIZE_HINTS: Dict[str, int] = {
    "tiny": 75_000_000,
    "base": 145_000_000,
    "small": 480_000_000,
    "medium": 1_500_000_000,
    "large-v1": 3_000_000_000,
    "large-v2": 3_000_000_000,
    "large-v3": 3_000_000_000,
    "large-v3-turbo": 1_600_000_000,
    "distil-large-v3": 1_500_000_000,
    "turbo": 1_600_000_000,
}

#: How short a weights file may be, as a fraction of its hint, before it is
#: called truncated.
_SIZE_FLOOR = 0.80


class ModelHealth(NamedTuple):
    """A verdict on one model directory on disk."""

    name: str
    ok: bool
    weight_bytes: int
    missing: Tuple[str, ...]  #: required companion files that are absent
    truncated: bool  #: weights present but implausibly small
    notes: Tuple[str, ...]  #: human-readable summary lines


def _lookup(config: Mapping[str, Any], dotted: str) -> Any:
    """Read a dotted path out of nested mappings, or None if any hop is missing."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def summarise_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The reportable settings, by allowlist.

    Absent keys are skipped rather than reported as null, so the summary shows
    what the install actually configures instead of a wall of nulls. Values are
    stringified through the redactor because a model name can be a path.
    """
    summary: Dict[str, Any] = {}
    for dotted in CONFIG_FIELDS:
        value = _lookup(config, dotted)
        if value is None:
            continue
        summary[dotted] = redact_home_paths(value) if isinstance(value, str) else value
    return summary


def filter_log_lines(lines: Iterable[str], *, limit: int = 400) -> List[str]:
    """Keep the last ``limit`` operational lines, dropping anything unvetted.

    Allowlist by tag, plus the handful of untagged shapes worth keeping (a
    traceback). Everything surviving is redacted and length-capped. The tail is
    what matters — a report is read backwards from the failure — so the limit is
    applied after filtering, from the end.
    """
    kept: List[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        match = _TAG_RE.match(line)
        if match:
            if match.group(1) not in LOG_TAGS:
                continue
        elif not any(pattern.match(line) for pattern in _UNTAGGED_KEEP):
            continue
        clean = redact_home_paths(line)
        if len(clean) > _MAX_LINE:
            clean = clean[:_MAX_LINE] + " …[truncated]"
        kept.append(clean)
    return kept[-limit:] if limit and limit > 0 else kept


def infer_family(dir_name: str, filenames: Sequence[str]) -> Optional[str]:
    """Which loader reads this directory — or None when it is not a model at all.

    ``models/`` is not exclusively model directories: it also holds caches
    (``.hf_cache``) and per-backend parents (``tts``) whose children are the real
    models. Judging those against a model's requirements reported healthy installs
    as broken, which is worse than saying nothing — an operator who is told their
    cache folder is corrupt will go and delete something that matters.

    A GGUF is checked first because such a directory holds nothing else: no
    config, no tokenizer, so every later test would misread it.
    """
    names = list(filenames)
    if any(n.endswith(".gguf") for n in names):
        return "gguf"
    if dir_name.startswith("faster-whisper-"):
        return "faster-whisper"
    if any(n.endswith(".pt") for n in names):
        return "whisper"
    if any(is_weight_file(n) for n in names) or "config.json" in names:
        return "nllb"
    return None


def missing_companions(
    present: Iterable[str],
    required: Sequence[Union[str, Sequence[str]]],
) -> Tuple[str, ...]:
    """Which ``required`` entries no file in ``present`` satisfies.

    An entry that is itself a sequence is a set of interchangeable names,
    satisfied by any one of them and reported as "a or b" — the same rule as
    ``model_files.missing_required``, which is what keeps the diagnostic report
    and the Model Manager from disagreeing about the same directory.
    """
    names = set(present)
    missing = []
    for entry in required:
        options = [entry] if isinstance(entry, str) else list(entry)
        if not any(option in names for option in options):
            missing.append(" or ".join(options))
    return tuple(missing)


def check_model_dir(
    dir_name: str,
    entries: Sequence[Tuple[str, int]],
    *,
    family: str = "faster-whisper",
    status: Optional[Any] = None,
) -> ModelHealth:
    """Judge one model directory from its ``(filename, size)`` listing.

    Answers the question #8 needed and nobody could ask remotely: is this model
    actually complete, or did the download die part-way and leave something the
    UI still lists as available?

    ``status`` is an optional ``model_files.DirStatus`` from the caller, which —
    unlike this module — is allowed to touch the disk and so can check the
    download manifest. When it is supplied it **wins**: it knows the size every
    file was supposed to be, where this function can only compare a weights file
    against a table of typical sizes. Passing it is how the report and the
    loader are kept from reaching different verdicts about one directory.
    """
    sizes = {name: size for name, size in entries}
    # Largest recognised weight file: a sharded layout has several, and the
    # biggest is the one whose size is worth judging.
    weights = sorted(
        ((n, sz) for n, sz in sizes.items() if is_weight_file(n)),
        key=lambda pair: pair[1], reverse=True,
    )
    weight_name, weight_bytes = weights[0] if weights else (None, 0)

    if status is not None:
        # The authoritative answer. Its "missing" already covers the weights, so
        # drop that entry here rather than reporting the same file twice.
        missing = tuple(m for m in status.missing if m != "model.bin")
        truncated = bool(status.missing) and weight_name is not None and not missing
    else:
        missing = missing_companions(sizes, REQUIRED_COMPANIONS.get(family, ()))
        truncated = False
        hint = _size_hint(dir_name, family)
        if weight_name is not None and hint and weight_bytes < hint * _SIZE_FLOOR:
            truncated = True

    notes: List[str] = []
    if weight_name is None:
        notes.append("No weights file — this model is not really downloaded.")
    else:
        notes.append(f"{weight_name}: {_human_bytes(weight_bytes)}")
        if truncated:
            notes.append(
                "Weights look truncated — the download did not finish. "
                "Open the Model Manager and press Repair on this model."
            )
    if missing:
        notes.append(
            "Missing " + describe_missing(missing)
            + " — the loader needs these; open the Model Manager and press Repair."
        )
    if weight_name is not None and not missing and not truncated:
        notes.append("Looks complete.")

    return ModelHealth(
        name=dir_name,
        ok=weight_name is not None and not missing and not truncated,
        weight_bytes=weight_bytes,
        missing=missing,
        truncated=truncated,
        notes=tuple(notes),
    )


def _size_hint(dir_name: str, family: str) -> int:
    """Expected weights size for ``dir_name``, or 0 when we have no opinion."""
    if family != "faster-whisper":
        return 0
    stem = dir_name[len("faster-whisper-"):] if dir_name.startswith("faster-whisper-") else dir_name
    return WEIGHT_SIZE_HINTS.get(stem.lower(), 0)


def _human_bytes(count: int) -> str:
    """Bytes as a short human string; the report is read by people, not parsers."""
    size = float(count)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def build_report(
    *,
    versions: Mapping[str, Any],
    platform: Mapping[str, Any],
    hardware: Mapping[str, Any],
    config: Mapping[str, Any],
    models: Sequence[ModelHealth],
    audio_devices: Sequence[str],
    transcription_state: Mapping[str, Any],
    log_lines: Sequence[str],
) -> Dict[str, Any]:
    """Assemble the report. Every section is already-redacted plain data.

    ``transcription_state`` is filtered to the fields that describe the run
    rather than copied wholesale, because it also carries the current caption.
    """
    return {
        "generated": "on request by the operator; not sent anywhere automatically",
        "versions": dict(versions),
        "platform": dict(platform),
        "hardware": dict(hardware),
        "settings": summarise_config(config),
        "models": [
            {
                "name": m.name,
                "ok": m.ok,
                "weight_bytes": m.weight_bytes,
                "missing": list(m.missing),
                "truncated": m.truncated,
                "notes": list(m.notes),
            }
            for m in models
        ],
        "audio_devices": [redact_home_paths(d) for d in audio_devices],
        "transcription_state": {
            key: transcription_state.get(key)
            for key in ("status", "running", "error", "message", "start_time")
            if key in transcription_state
        },
        "log_tail": list(log_lines),
    }


def format_report_text(report: Mapping[str, Any]) -> str:
    """The report as flat text, which is what gets pasted into an issue."""
    out: List[str] = ["STT diagnostic report", "=" * 21, ""]

    for title, key in (
        ("Versions", "versions"),
        ("Platform", "platform"),
        ("Hardware", "hardware"),
        ("Settings", "settings"),
    ):
        section = report.get(key) or {}
        if not section:
            continue
        out.append(f"## {title}")
        for name, value in section.items():
            out.append(f"  {name}: {value}")
        out.append("")

    models = report.get("models") or []
    if models:
        out.append("## Models on disk")
        for model in models:
            mark = "OK  " if model.get("ok") else "BAD "
            out.append(f"  [{mark}] {model.get('name')}")
            for note in model.get("notes", []):
                out.append(f"         {note}")
        out.append("")

    devices = report.get("audio_devices") or []
    if devices:
        out.append("## Audio devices")
        out.extend(f"  {d}" for d in devices)
        out.append("")

    state = report.get("transcription_state") or {}
    if state:
        out.append("## Transcription state")
        for name, value in state.items():
            out.append(f"  {name}: {value}")
        out.append("")

    log_tail = report.get("log_tail") or []
    if log_tail:
        out.append("## Log (operational lines only — captions are never included)")
        out.extend(f"  {line}" for line in log_tail)
        out.append("")

    return "\n".join(out)


def report_filename(now: str, *, prefix: str = "stt-diagnostic") -> str:
    """A filename safe on every platform: no colons, no spaces."""
    stamp = re.sub(r"[^0-9A-Za-z-]", "-", now).strip("-")
    return f"{prefix}-{stamp}.txt"

