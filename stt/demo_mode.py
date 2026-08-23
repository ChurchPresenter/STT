"""Is this process a demo, and what does a demo run against.

The demo is the shipped product with the transcription worker taken out: no torch,
no ffmpeg, no microphone, no models. It replays a recorded service into a real
session database so the rest of the server — the web layer, the phase detector, the
corrections path, the file manager — runs completely unchanged.

Stdlib-only and side-effect free at import (the monolith's ``APP_DIR`` is chosen from
:func:`prepare_data_dir` at import time, so nothing here may itself touch the disk
until it is called). Every path is passed in explicitly; nothing reads the monolith's
globals.

Two properties this module is responsible for:

* A demo never writes into a real install. :func:`prepare_data_dir` returns a root
  that is not ``~/.stt``, and the monolith redirects ``APP_DIR`` to it before it
  creates a single directory.
* A demo never plays a recording nobody chose. :func:`discover_session` prefers what
  the operator dropped next to the executable and only reads a real install's archive
  when explicitly asked.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import shutil
import socket
import threading
import webbrowser
from typing import Any, Dict, List, MutableMapping, Optional, Sequence

# Env var the PyInstaller runtime hook sets, so the shipped artifact is a demo by
# construction. Also the dev switch: STT_DEMO=1 .venv/bin/python3 speech_to_text.py
ENV_FLAG = "STT_DEMO"

# Where a demo keeps config, logs and the session databases it materialises. A
# sibling of ~/.stt rather than a child, so "delete the demo's data" can never be
# mistyped into deleting a real install's.
DATA_DIR_NAME = ".stt-demo"

# Folder the operator drops a recorded service into, both beside the executable and
# inside the demo data dir.
SESSIONS_DIR_NAME = "sessions"

# Bundled fallback, relative to BUNDLE_DIR.
BUNDLED_SESSION = os.path.join("demo", "demo.db")

# First port tried. Not 80 (needs admin) and not 8080 (a real install's default, which
# the demo must be able to run alongside).
DEFAULT_PORT = 8099
PORT_SEARCH_LIMIT = 12

_TRUE = {"1", "true", "yes", "on"}


def enabled(environ: Optional[MutableMapping[str, str]] = None) -> bool:
    """Whether this process is a demo."""
    env = os.environ if environ is None else environ
    return str(env.get(ENV_FLAG, "")).strip().lower() in _TRUE


# --- data directory --------------------------------------------------------


def data_dir(home: Optional[str] = None) -> str:
    """The demo's data root. Never ``~/.stt``."""
    return os.path.join(home or os.path.expanduser("~"), DATA_DIR_NAME)


def read_install_id(home: Optional[str] = None) -> Optional[str]:
    """The anonymous id a previous demo run recorded, if there was one.

    Read *before* :func:`prepare_data_dir` wipes the directory. Without carrying it
    across, the wipe would mint a new id on every launch and the collector would count
    one person opening the demo five times as five separate trials.
    """
    path = os.path.join(data_dir(home), "config", "config.json")
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return None
    value = (stored.get("analytics") or {}).get("install_id")
    return str(value) if value else None


def prepare_data_dir(bundle_dir: str, home: Optional[str] = None) -> str:
    """Create a clean demo data root seeded from the bundled templates, and return it.

    Wiped on every launch, which is what makes "the settings pages are fully editable"
    safe to offer: a visitor can change anything and the next run starts fresh. The
    wipe is scoped to the returned root and nothing above it.
    """
    root = data_dir(home)
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    for sub in ("", "config", "models", "logs", "_AUTOMATIC_BACKUP", SESSIONS_DIR_NAME):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    src_config = os.path.join(bundle_dir, "config")
    dst_config = os.path.join(root, "config")
    if os.path.isdir(src_config):
        for name in os.listdir(src_config):
            if name.endswith(".default.json"):
                shutil.copy2(os.path.join(src_config, name), os.path.join(dst_config, name))
    return root


# --- config overlay --------------------------------------------------------


def config_overlay(port: int) -> Dict[str, Any]:
    """Settings a demo must have, as a tree to deep-merge over config.default.json.

    Only keys that exist in the template appear here — a test asserts that, so this
    cannot silently drift into setting something the server never reads.
    """
    return {
        "web_server": {
            "port": port,
            # Reachable from the network, so the demo can be opened on a phone or a
            # second screen while someone drives it from a laptop. That is also why
            # the outbound guards in stt/demo_guard.py are not optional: with the
            # password off, every route is reachable by anyone on the network.
            "host": "0.0.0.0",
            "password_auth": {"enabled": False},
            "settings_ip_whitelist": [],
            "access_token": "",
        },
        # The visitor presses Start, exactly as an operator does.
        "audio": {"autostart": False},
        # Translations come from the recording's own translated_text column. The
        # monolith skips fresh translation, backfill and in-progress translation
        # outright in demo mode, so a drop-in session with untranslated rows cannot
        # reach a model, an LLM endpoint or a paired machine.
        "live_translation": {"enabled": True, "tts": {"enabled": False}},
        "service_phase": {"enabled": True},
        "sermon_summary": {"enabled": False},
        # A demo reports two things and nothing else: that it ran, and that it
        # crashed. Both are tagged as a demo at the source — the live-map ping sends
        # src="demo" and Sentry carries a "demo" tag — so the collector can count
        # them separately instead of mistaking a trial for an install.
        #
        # Everything else a demo could send is still shut off at the choke points in
        # stt/demo_guard.py. Telemetry being on is exactly why that list matters.
        "crash_reporting": {"enabled": True, "sentry_enabled": True},
        "auto_update": {"enabled": False},
    }


def apply_overlay(config: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``overlay`` into ``config``, overwriting leaves. Mutates and returns config."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            apply_overlay(config[key], value)
        else:
            config[key] = json.loads(json.dumps(value))  # deep copy via round-trip
    return config


def write_config(root: str, bundle_dir: str, port: int,
                 install_id: Optional[str] = None) -> str:
    """Seed the demo's live config from the template with the overlay applied.

    Written before the server loads its config, because the port it binds is decided
    here: a demo has to be able to run alongside a real install on the same machine.

    ``install_id`` carries the anonymous id forward across the wipe — see
    :func:`read_install_id`.
    """
    with open(os.path.join(bundle_dir, "config", "config.default.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    apply_overlay(config, config_overlay(port))
    if install_id:
        config.setdefault("analytics", {})["install_id"] = install_id
    dest = os.path.join(root, "config", "config.json")
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return dest


def missing_overlay_paths(overlay: Dict[str, Any], template: Dict[str, Any],
                          _prefix: str = "") -> List[str]:
    """Dotted paths the overlay sets that the template does not define.

    The overlay names settings the real server reads; if one is renamed upstream the
    demo would silently stop configuring it. This is what the drift test asserts on.
    """
    missing: List[str] = []
    for key, value in overlay.items():
        path = f"{_prefix}{key}"
        if key not in template:
            missing.append(path)
        elif isinstance(value, dict) and isinstance(template[key], dict):
            missing.extend(missing_overlay_paths(value, template[key], path + "."))
    return missing


# --- which recording to play ----------------------------------------------


def session_search_dirs(bundle_dir: str, exe_dir: Optional[str],
                        root: str, use_local_sessions: bool = False,
                        home: Optional[str] = None) -> List[str]:
    """Directories to look for a recorded service in, most-preferred first.

    A real install's archive is last and only when asked for: a demo that silently
    picked up whatever service happened to be on the machine would be a nasty
    surprise on a support call.
    """
    dirs: List[str] = []
    if exe_dir:
        dirs.append(os.path.join(exe_dir, SESSIONS_DIR_NAME))
    dirs.append(os.path.join(root, SESSIONS_DIR_NAME))
    if use_local_sessions:
        dirs.append(os.path.join(home or os.path.expanduser("~"), ".stt", "_AUTOMATIC_BACKUP"))
    dirs.append(os.path.join(bundle_dir, os.path.dirname(BUNDLED_SESSION)))
    return dirs


def find_sessions(dirs: Sequence[str]) -> List[str]:
    """Every .db under ``dirs``, in directory-preference order and newest-first within each.

    Sorted by filename because session names are ``%Y-%m-%d_%H%M%S`` — see
    :func:`stt.session_index.session_date`, which reads the same convention.
    """
    found: List[str] = []
    for directory in dirs:
        if not os.path.isdir(directory):
            continue
        matches = sorted(glob.glob(os.path.join(directory, "**", "*.db"), recursive=True),
                         key=lambda p: os.path.basename(p), reverse=True)
        found.extend(matches)
    return found


def discover_session(bundle_dir: str, exe_dir: Optional[str], root: str,
                     explicit: Optional[str] = None, use_local_sessions: bool = False,
                     home: Optional[str] = None) -> Optional[str]:
    """The recording to replay, or None if there is nothing playable anywhere.

    ``explicit`` (a --session argument) wins outright, including over the bundled
    fallback, so a demo can always be pointed at a specific file.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    dirs = session_search_dirs(bundle_dir, exe_dir, root, use_local_sessions, home)
    candidates = find_sessions(dirs)
    return candidates[0] if candidates else None


def executable_dir() -> Optional[str]:
    """The directory the shipped executable sits in, or None when running from source.

    On macOS the binary lives inside ``STT Demo.app/Contents/MacOS``; a signed bundle
    must not be written into, so the drop-in folder belongs beside the .app itself.
    """
    import sys

    if not getattr(sys, "frozen", False):
        return None
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    parts = exe_dir.split(os.sep)
    if len(parts) >= 3 and parts[-1] == "MacOS" and parts[-2] == "Contents":
        app_bundle = os.sep.join(parts[:-2])
        if app_bundle.endswith(".app"):
            return os.path.dirname(app_bundle)
    return exe_dir


# --- shared state without a Manager process --------------------------------


class LocalManager:
    """Stand-in for ``multiprocessing.Manager`` when everything runs in one process.

    The demo has no worker process, so the state the monolith shares between the web
    server and the worker only has to be shared between threads. Dropping the Manager
    removes a spawned child from the frozen bundle — which is the part of
    multiprocessing most likely to misbehave under PyInstaller — while leaving every
    ``transcription_state[...]`` call site in the monolith untouched.
    """

    def dict(self, initial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return dict(initial or {})

    def list(self, initial: Optional[Sequence[Any]] = None) -> List[Any]:
        return list(initial or [])


def local_queue(maxsize: int = 0) -> "queue.Queue[Any]":
    """A thread queue with the subset of the MPQueue API the monolith uses."""
    return queue.Queue(maxsize=maxsize)


# --- launching -------------------------------------------------------------


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether ``port`` can be bound right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(start: int = DEFAULT_PORT, limit: int = PORT_SEARCH_LIMIT,
              host: str = "127.0.0.1") -> int:
    """The first free port at or after ``start``.

    Falls back to ``start`` when every candidate is taken, so the caller still gets a
    number and the bind failure is reported by the server rather than here.
    """
    for port in range(start, start + limit):
        if port_is_free(port, host):
            return port
    return start


def lan_address() -> Optional[str]:
    """This machine's address on the local network, or None if it has none.

    Uses a connect-less UDP socket: it asks the routing table which interface would
    be used to reach the internet and reports that interface's address. No packet is
    sent, and nothing is resolved — a demo must not make a network request to work
    out how to describe itself.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1: routable-looking, never routed
        address = sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
    return None if not address or address.startswith("127.") else str(address)


def startup_banner(port: int, lan: Optional[str] = None) -> str:
    """What the console prints so whoever launched the demo knows how to reach it.

    Names the network address explicitly, and says the demo has no password: the
    combination is deliberate, and someone running it on a shared network should be
    told rather than have to infer it.
    """
    lines = ["", "  STT Demo", f"    On this machine:  http://127.0.0.1:{port}/"]
    if lan:
        lines.append(f"    On this network:  http://{lan}:{port}/")
        lines.append("    Anyone on this network can open it — the demo has no password.")
    lines.append("")
    return "\n".join(lines)


def open_browser_later(port: int, delay: float = 1.2, host: str = "127.0.0.1") -> None:
    """Open the demo in the default browser once the server has had time to bind.

    A downloadable demo that opens to nothing visible has already failed, so this is
    load-bearing rather than a nicety. Best-effort: a headless machine just won't.
    """
    def _open() -> None:
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    threading.Timer(delay, _open).start()
