"""What the demo answers instead of loading a model, downloading one, or translating.

The demo ships without torch, faster-whisper, transformers or PANNs, so the routes
that would import them must never run. Rather than editing eighty view functions, the
server consults this module before dispatch: a returned payload short-circuits the
request, and ``None`` lets the real view handle it. That ordering is the safety
property — the production view is not entered at all, so it cannot reach an ML import
or shell out to ``uv``.

The families that could load a model are **deny-by-default**: a route added under
``/api/models/`` next year is intercepted before anyone remembers this file exists,
and fails closed with a polite message instead of crashing a demo on someone's desk.

Responses come from ``stt.demo_fixtures``, recorded from a real server and scrubbed,
so the pages get shapes they actually understand. Downloads are simulated against an
injected clock: a visitor clicking "download" sees real progress and ends up with a
model that shows as installed, because a demo that refuses the first thing anyone
clicks demonstrates nothing.

Stdlib-only, pure apart from the state object handed in.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

from stt import demo_fixtures

Canned = Tuple[Dict[str, Any], int]

# Families that can reach a model, a download, or an inference engine. Everything
# under these is intercepted unless it appears in PASS_THROUGH.
INTERCEPT_PREFIXES: Tuple[str, ...] = (
    "/api/models/",
    "/api/tts/",
    "/api/translate/",
    "/api/llm/",
    "/api/calibration/",
    "/api/demo/",
    # Starting a tunnel downloads the cloudflared binary and publishes the demo — with
    # its authentication switched off — on a public URL. /api/tunnel/settings also
    # stores a `binary` path that is later executed, so this is code execution rather
    # than merely egress, and no network guard would catch it.
    "/api/tunnel/",
    # The client half of the paired-machine protocol. /api/translate/pair guards the
    # server half; these proxies are a different prefix and were reachable.
    "/api/remote-translation/",
    # Takes a share path and SMB credentials straight from the request body, with no
    # config gate, and mounts it: `net use` on Windows, `sudo mount -t cifs` on Linux.
    # Note this is file-MOVER; /api/file-manager/* is local and must keep working.
    "/api/file-mover/",
)

# Exact paths inside those families that the real server must still handle.
PASS_THROUGH: frozenset = frozenset()

# Intercepted even though they sit outside the prefixes above.
ALSO_INTERCEPT: frozenset = frozenset({
    "/api/translate",          # note: /api/translation/* is config-backed and passes through
    "/api/audio-devices",
    "/api/transcribe-file",
    "/api/restart",
    "/api/server/restart",
    "/api/server/update",
    "/api/sermon-summary/generate",
})

# Actions that genuinely cannot be demonstrated. Answered with success=False and an
# explanation at HTTP 200, so the page shows a toast rather than an error screen.
UNAVAILABLE: frozenset = frozenset({
    "/api/models/upload",
    "/api/models/upload-local",
    "/api/models/sync",
    "/api/models/refresh-whisper",
    "/api/models/refresh-faster-whisper",
    "/api/models/search",
    "/api/models/gguf-repo-files",
    "/api/transcribe-file",
    "/api/restart",
    "/api/server/restart",
    "/api/server/update",
})

# Anything under this prefix is machine-to-machine pairing with a second STT box.
PAIRING_PREFIX = "/api/translate/pair"

#: Families where the risk is not an ML import but leaving the machine: spawning a
#: downloaded binary, mounting a remote share, or acting as a client against a host
#: somebody else names. Nothing here may act, whatever the method — the generic
#: "accept a settings write" fallback below must not reach them.
#:
#: A read-only GET with a recorded response is still served, so the pages render and
#: the refusal is attached to the button rather than to the whole screen.
BLOCKED_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("/api/tunnel/",
     "Publishing the demo on a public address is not available in the demo."),
    ("/api/file-mover/",
     "Copying recordings to a network share is not available in the demo."),
    ("/api/remote-translation/",
     "Pairing with a second machine is not available in the demo."),
    # Risk here is subprocess, not egress: the report enumerates audio devices by
    # running ffmpeg. It also has nothing to describe — a demo has no models and
    # no real install — so it is refused rather than faked.
    ("/api/diagnostics/",
     "The diagnostic report is not available in the demo."),
)

# POST paths that begin a download, mapped to the catalogue key they install.
DOWNLOAD_PATHS: Tuple[str, ...] = (
    "/api/models/download",
    "/api/models/faster-whisper/download",
    "/api/models/translation/download",
    "/api/models/tts/download",
    "/api/models/panns/download",
)

# GET paths that report on one.
PROGRESS_PATHS: Tuple[str, ...] = (
    "/api/models/download-status",
    "/api/models/tts/download-progress",
    "/api/models/nllb-download-progress",
)

REMOVAL_PATHS: Tuple[str, ...] = (
    "/api/models/remove",
    "/api/models/remove-whisper",
    "/api/models/faster-whisper/remove",
    "/api/models/translation/remove",
    "/api/models/tts/remove",
)

DOWNLOAD_SECONDS = 12.0
DEFAULT_TOTAL_BYTES = 1_500_000_000

UNAVAILABLE_MESSAGE = "Not available in the demo."

# Hand-written rather than recorded: calibration reports the room a server is
# actually sitting in, and there is no room here. Plausible values for a quiet hall.
CALIBRATION_RESULTS: Dict[str, Any] = {
    "noise_floor_db": -52.4,
    "speech_peak_db": -14.8,
    "recommended_energy_threshold": 380,
    "recommended_silence_seconds": 0.9,
    "samples": {"noise": 412, "speech": 268, "silence": 173},
    "complete": True,
}

# What an end-of-service summary looks like. The real one is written by an LLM the
# demo does not ship.
SERMON_SUMMARY: Dict[str, Any] = {
    "title": "Sample service summary",
    "summary": (
        "The service opened with a welcome and a call to prayer, followed by "
        "congregational singing. The message centred on forgiveness and the cost of "
        "grace, drawing on the account of the crucifixion, and closed with a "
        "benediction and announcements for the coming week."
    ),
    "points": [
        "Welcome and opening prayer",
        "Congregational singing",
        "Message: the price of forgiveness",
        "Closing benediction and announcements",
    ],
}


class FakeJob:
    """A download that takes time and then finishes."""

    __slots__ = ("cancelled", "duration_s", "name", "started_at", "total_bytes")

    def __init__(self, name: str, started_at: float, total_bytes: int = DEFAULT_TOTAL_BYTES,
                 duration_s: float = DOWNLOAD_SECONDS) -> None:
        self.name = name
        self.started_at = started_at
        self.total_bytes = total_bytes
        self.duration_s = duration_s
        self.cancelled = False

    def progress(self, now: float) -> float:
        if self.cancelled:
            return 0.0
        if self.duration_s <= 0:
            return 1.0
        return min(max((now - self.started_at) / self.duration_s, 0.0), 1.0)

    def report(self, now: float) -> Dict[str, Any]:
        fraction = self.progress(now)
        downloaded = int(self.total_bytes * fraction)
        elapsed = max(now - self.started_at, 0.001)
        return {
            "success": True,
            "model": self.name,
            "name": self.name,
            "status": "cancelled" if self.cancelled
                      else ("completed" if fraction >= 1.0 else "downloading"),
            "downloading": not self.cancelled and fraction < 1.0,
            "progress": round(fraction * 100, 1),
            "percent": round(fraction * 100, 1),
            "downloaded_bytes": downloaded,
            "total_bytes": self.total_bytes,
            "speed_bps": int(downloaded / elapsed) if elapsed else 0,
            "eta_seconds": max(int(self.duration_s - (now - self.started_at)), 0),
            "error": None,
        }


class State:
    """What the visitor has changed since the demo started.

    Lives for the life of the process; the demo's data directory is rebuilt on every
    launch anyway, so nothing here needs to survive a restart.
    """

    def __init__(self, now: Optional[Callable[[], float]] = None,
                 translations: Optional[Mapping[str, str]] = None) -> None:
        self.now: Callable[[], float] = now or time.time
        self.installed: Set[str] = set()
        self.downloads: Dict[str, FakeJob] = {}
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self.translations: Dict[str, str] = dict(translations or {})


# --- helpers ---------------------------------------------------------------


def _unavailable(message: str = UNAVAILABLE_MESSAGE) -> Canned:
    return {"success": False, "error": message, "demo": True}, 200


def _ok(**payload: Any) -> Canned:
    body: Dict[str, Any] = {"success": True, "demo": True}
    body.update(payload)
    return body, 200


def is_intercepted(path: str) -> bool:
    """Whether this route is the demo's business at all."""
    if path in PASS_THROUGH:
        return False
    if path in ALSO_INTERCEPT:
        return True
    return any(path.startswith(prefix) for prefix in INTERCEPT_PREFIXES)


def _model_name(body: Mapping[str, Any], args: Mapping[str, str]) -> str:
    for key in ("model", "model_id", "name", "repo_id", "voice"):
        value = body.get(key) or args.get(key)
        if value:
            return str(value)
    return "model"


def _recorded(path: str, state: State) -> Optional[Dict[str, Any]]:
    """The scrubbed recording for this endpoint, with anything the visitor changed."""
    payload = demo_fixtures.RESPONSES.get(path)
    if payload is None:
        return None
    result = _deep_copy(payload)
    override = state.overrides.get(path)
    if override:
        result.update(_deep_copy(override))
    return result


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _with_installed(path: str, payload: Dict[str, Any], state: State) -> Dict[str, Any]:
    """Show a model the visitor "downloaded" as present in the local listing."""
    if not state.installed:
        return payload
    models = payload.get("models")
    if isinstance(models, list):
        known = {m.get("name") or m.get("id") for m in models if isinstance(m, dict)}
        for name in sorted(state.installed):
            if name not in known:
                models.append({"name": name, "id": name, "downloaded": True,
                               "installed": True, "size_bytes": DEFAULT_TOTAL_BYTES,
                               "source": "demo"})
        payload["count"] = len(models)
    return payload


# --- the entry point -------------------------------------------------------


def intercept(method: str, path: str, args: Mapping[str, str], body: Mapping[str, Any],
              state: State, player: Any = None) -> Optional[Canned]:
    """The demo's answer for this request, or None to let the real route run."""
    if not is_intercepted(path):
        return None

    method = method.upper()
    now = state.now()

    if path.startswith("/api/demo/"):
        return _demo_control(method, path, body, state, player)

    for prefix, message in BLOCKED_PREFIXES:
        if path.startswith(prefix):
            recorded = _recorded(path, state) if method == "GET" else None
            return (recorded, 200) if recorded is not None else _unavailable(message)

    if path in UNAVAILABLE:
        return _unavailable()

    if path.startswith(PAIRING_PREFIX):
        return _unavailable("Pairing a second machine is not available in the demo.")

    if path in DOWNLOAD_PATHS and method == "POST":
        return _start_download(_model_name(body, args), state, now)

    if path in PROGRESS_PATHS and method == "GET":
        return _download_report(state, now)

    if path == "/api/models/cancel-download" and method == "POST":
        return _cancel_download(state)

    if path in REMOVAL_PATHS and method == "POST":
        state.installed.discard(_model_name(body, args))
        return _ok(message="Removed.")

    if path == "/api/translate" and method == "POST":
        return _translate(body, state)

    if path == "/api/calibration/results" and method == "GET":
        return _ok(**CALIBRATION_RESULTS)

    if path == "/api/calibration/status" and method == "GET":
        return _ok(active=False, step=1, complete=True)

    if method == "POST" and path in ("/api/llm/summarize", "/api/sermon-summary/generate"):
        return _ok(**SERMON_SUMMARY)

    if method == "GET":
        recorded = _recorded(path, state)
        if recorded is not None:
            if path in ("/api/models/local", "/api/models/list", "/api/models/cached"):
                recorded = _with_installed(path, recorded, state)
            return recorded, 200
        # Deny by default: an endpoint added under an intercepted prefix must fail
        # closed rather than fall through to a view that may import torch.
        return _unavailable()

    # A write inside an intercepted family: accept it, and remember it so the
    # matching GET reflects what the visitor just did.
    if path in demo_fixtures.RESPONSES and isinstance(body, Mapping) and body:
        state.overrides.setdefault(path, {}).update(
            {k: v for k, v in body.items() if isinstance(k, str)})
    return _ok()


# --- families --------------------------------------------------------------


def _start_download(name: str, state: State, now: float) -> Canned:
    state.downloads[name] = FakeJob(name, started_at=now)
    return _ok(status="started", model=name,
               message=f"Downloading {name}...")


def _download_report(state: State, now: float) -> Canned:
    active = [job for job in state.downloads.values() if not job.cancelled]
    if not active:
        return {"success": True, "downloading": False, "status": "idle",
                "progress": 0, "demo": True}, 200
    job = active[-1]
    report = job.report(now)
    if report["status"] == "completed":
        state.installed.add(job.name)
    report["demo"] = True
    return report, 200


def _cancel_download(state: State) -> Canned:
    for job in state.downloads.values():
        job.cancelled = True
    return _ok(status="cancelled")


def _translate(body: Mapping[str, Any], state: State) -> Canned:
    """Translate by looking the line up in the service the demo is replaying.

    The recording already holds both sides of every caption, so a visitor who types
    one in gets the same translation the demo is displaying — real output from the
    real engine, just produced earlier.
    """
    text = str(body.get("text") or "").strip()
    if not text:
        return _unavailable("Nothing to translate.")
    translated = state.translations.get(text) or state.translations.get(text.lower())
    if translated:
        return _ok(translation=translated, translated_text=translated, text=translated,
                   source_language=body.get("source_language") or "auto",
                   target_language=body.get("target_language") or "en")
    return _unavailable(
        "The demo translates the phrases from the sample service. "
        "Try a line from the live transcript.")


# --- playback control ------------------------------------------------------


def _demo_control(method: str, path: str, body: Mapping[str, Any], state: State,
                  player: Any) -> Canned:
    if player is None:
        return _unavailable("Playback is not running.")

    if path == "/api/demo/status" and method == "GET":
        return _ok(running=bool(getattr(player, "running", False)),
                   elapsed_s=round(float(player.elapsed_s()), 2),
                   session=getattr(player, "db_path", None))

    if path == "/api/demo/control" and method == "POST":
        action = str(body.get("action") or "")
        if action == "play":
            if not player.running:
                player.begin_session()
        elif action == "pause":
            player.end_session()
        elif action == "restart":
            player.restart()
        elif action == "speed":
            try:
                player.set_speed(float(body.get("value") or 1.0))
            except (TypeError, ValueError):
                return _unavailable("Speed must be a number.")
        else:
            return _unavailable(f"Unknown action {action!r}.")
        return _ok(action=action)

    return _unavailable()
