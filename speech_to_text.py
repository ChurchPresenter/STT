import argparse
import os
import sys
import warnings
from typing import ClassVar

# Determine application directory (works for both dev and PyInstaller bundle).
# APP_DIR    = user data dir: config, models, logs, per-session DBs.
# BUNDLE_DIR = bundled read-only assets: templates, static, config.default.
# Every non-override run (frozen OR run-from-repo) uses ~/.stt, so the data dir
# is always the per-user, always-writable location and a run-from-repo server no
# longer writes data into the checkout. STT_DATA_DIR (set by the watchdog) still
# overrides, so a managed worker follows whatever path the watchdog chose.
_data_override = os.environ.get("STT_DATA_DIR")
_script_dir = os.path.dirname(os.path.abspath(__file__))
_is_frozen = getattr(sys, "frozen", False)
if _data_override:
    APP_DIR    = os.path.abspath(os.path.expanduser(_data_override))
    BUNDLE_DIR = _script_dir
else:
    APP_DIR    = os.path.join(os.path.expanduser("~"), ".stt")
    BUNDLE_DIR = sys._MEIPASS if _is_frozen else _script_dir

os.makedirs(APP_DIR, exist_ok=True)

MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Default base directory for database + audio backups (rooted in APP_DIR so compiled
# builds keep all data under ~/.stt instead of the launch directory).
BACKUP_DIR = os.path.join(APP_DIR, "_AUTOMATIC_BACKUP")


# Live (user-mutated, gitignored) config files live in APP_DIR/config next to
# their tracked *.default.json templates. Templates ship with the checkout;
# live files are seeded from them on first run and never touched by updates.
CONFIG_DIR = os.path.join(APP_DIR, "config")
os.makedirs(CONFIG_DIR, exist_ok=True)

# One-time layout migration: live config files used to sit in APP_DIR root.
# stt/watchdog.py performs the same move-if-absent migration; whichever
# process starts first wins, the other finds nothing left to move.
for _mig_name in ("config.json", "custom_dictionary.json", "word_highlighting.json",
                  "whisper_models.json", "faster_whisper_models.json"):
    _mig_old = os.path.join(APP_DIR, _mig_name)
    _mig_new = os.path.join(CONFIG_DIR, _mig_name)
    if os.path.exists(_mig_old) and not os.path.exists(_mig_new):
        try:
            import shutil as _mig_shutil
            _mig_shutil.move(_mig_old, _mig_new)
            print(f"[MIGRATE] {_mig_name} -> config/{_mig_name}")
        except OSError as _mig_err:
            print(f"[MIGRATE] Could not move {_mig_name}: {_mig_err}")


# Path-containment guards live in stt/paths.py (importable, unit-tested);
# safe_model_path is re-imported and safe_managed_path keeps its APP_DIR default.
from stt import paths as _paths
from stt.paths import safe_model_path  # noqa: F401
from stt.coercion import coerce_float, coerce_int
from stt.http_params import merge_request_params, parse_json_body as _parse_json_body
from stt.model_disk import dir_has_weights, dir_is_writable, has_weight_file, is_weight_file  # noqa: F401


def safe_managed_path(path, base_dir=None):
    """Resolve a file-manager path and return its realpath only if it stays
    inside base_dir (defaults to APP_DIR). See stt/paths.py."""
    return _paths.safe_managed_path(path, base_dir if base_dir is not None else APP_DIR)


def _seed_from_bundle(filename):
    """Return the live path of a config file (CONFIG_DIR/<filename>), seeding it
    from the bundled template (config/<stem>.default.json) on first run."""
    import shutil
    stem, ext = os.path.splitext(filename)
    dst = os.path.join(CONFIG_DIR, filename)
    src = os.path.join(BUNDLE_DIR, "config", f"{stem}.default{ext}")
    if not os.path.exists(dst) and os.path.exists(src):
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[INIT] Could not seed {filename} from bundle: {e}")
    return dst


def _chmod_quiet(path, mode):
    """Best-effort chmod; ignore failures (not owner, fs without unix perms, etc.)."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def make_db_world_readable(db_path):
    """Make a DB file and its WAL/SHM/journal sidecars world-readable (a+r, 0644),
    so databases written by the service (often as root) can be read by all users
    and downstream consumers."""
    if not db_path:
        return
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = db_path + suffix
        if os.path.exists(p):
            _chmod_quiet(p, 0o644)


def make_dirs_world_readable(leaf_dir, base_dir=None):
    """Make leaf_dir world-readable and traversable (a+rx, 0755). When base_dir is
    given and contains leaf_dir, every directory from base_dir down to leaf_dir is
    updated too, so DB files written inside are reachable by all users."""
    if not leaf_dir:
        return
    leaf = os.path.abspath(leaf_dir)
    if not base_dir:
        _chmod_quiet(leaf, 0o755)
        return
    base = os.path.abspath(base_dir)
    _chmod_quiet(base, 0o755)
    if leaf == base:
        return
    if leaf.startswith(base + os.sep):
        cur = base
        for part in os.path.relpath(leaf, base).split(os.sep):
            cur = os.path.join(cur, part)
            _chmod_quiet(cur, 0o755)
    else:
        _chmod_quiet(leaf, 0o755)


def make_tree_world_readable(root):
    """Recursively make every directory (a+rx, 0755) and file (a+r, 0644) under
    root readable by all users. Best-effort; skips entries it cannot chmod. Used to
    sweep the whole DB/backup folder, including files created during stop cleanup."""
    if not root or not os.path.isdir(root):
        return
    _chmod_quiet(root, 0o755)
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            _chmod_quiet(os.path.join(dirpath, d), 0o755)
        for f in filenames:
            _chmod_quiet(os.path.join(dirpath, f), 0o644)

# Suppress NNPACK warnings from PyTorch (harmless but spammy)
# These are C++ warnings so we need to disable at the PyTorch level
os.environ['NNPACK_DISABLE'] = '1'
os.environ['PYTORCH_NNPACK_WARN'] = '0'
# macOS: allow fork() after Objective-C threads are initialized (Flask, PyTorch, etc.)
# Without this, multiprocessing with fork start method crashes on macOS
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
warnings.filterwarnings('ignore', message='.*NNPACK.*')

# Set HuggingFace cache to local models directory BEFORE any HF imports
# This prevents models from being downloaded to ~/.cache/huggingface/hub
_models_cache_dir = os.path.join(MODELS_DIR, ".hf_cache")
os.makedirs(_models_cache_dir, exist_ok=True)
os.environ["HF_HUB_CACHE"] = _models_cache_dir
os.environ["HF_HOME"] = _models_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = _models_cache_dir

# TTS models directory (for piper models)
_tts_cache_dir = os.path.join(MODELS_DIR, "tts")
os.makedirs(_tts_cache_dir, exist_ok=True)

import sqlite3
import logging
import signal
import threading
import multiprocessing
from multiprocessing import Queue as MPQueue

import json
import re
import secrets
import shutil
import socket
import statistics
from pathlib import Path
from datetime import timedelta, datetime
from queue import Queue, Empty, Full
from time import sleep
import time
from sys import platform
from stt.file_mover import execute_file_move_now, execute_file_move

# Tracks the most recent file mover run so the UI can show live activity/status.
# Updated by both the automatic (transcription-stop) path and the manual trigger.
_file_mover_runtime_lock = threading.Lock()
file_mover_runtime = {
    "state": "idle",      # idle | running | success | error
    "trigger": None,      # "auto" | "manual"
    "started_at": None,   # ISO timestamp
    "finished_at": None,  # ISO timestamp
    "moved": 0,
    "failed": 0,
    "message": "",
}


def set_file_mover_running(trigger):
    """Mark the file mover as actively running (so the UI shows a live indicator)."""
    with _file_mover_runtime_lock:
        file_mover_runtime.update({
            "state": "running",
            "trigger": trigger,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "message": "File move in progress...",
        })


def set_file_mover_result(trigger, result):
    """Record the outcome of a completed file mover run."""
    with _file_mover_runtime_lock:
        file_mover_runtime.update({
            "state": "success" if result.get("success") else "error",
            "trigger": trigger,
            "finished_at": datetime.now().isoformat(),
            "moved": result.get("moved", 0),
            "failed": result.get("failed", 0),
            "message": result.get("message", ""),
        })


def get_file_mover_runtime():
    """Return a thread-safe copy of the current file mover runtime status."""
    with _file_mover_runtime_lock:
        return dict(file_mover_runtime)


import functools

from flask import Flask, render_template, jsonify, request, redirect, send_from_directory, make_response, g, Response, stream_with_context
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import numpy as np

# Heavy ML imports will be loaded lazily when needed
torch = None
whisper = None
AutoModelForSpeechSeq2Seq = None
AutoProcessor = None
AutoModelForCTC = None
Wav2Vec2Processor = None
pipeline = None
HfApi = None
model_info = None


def _lazy_import_ml_libraries():
    """Import heavy ML libraries only when needed"""
    global \
        torch, \
        whisper, \
        AutoModelForSpeechSeq2Seq, \
        AutoProcessor, \
        AutoModelForCTC, \
        Wav2Vec2Processor, \
        pipeline, \
        HfApi, \
        model_info

    if torch is None:
        import torch as _torch

        torch = _torch
        print("[INFO] PyTorch loaded")

    if whisper is None:
        import whisper as _whisper

        whisper = _whisper
        print("[INFO] Whisper loaded")

    if AutoModelForSpeechSeq2Seq is None:
        from transformers import (
            AutoModelForSpeechSeq2Seq as _AutoModelForSpeechSeq2Seq,
            AutoProcessor as _AutoProcessor,
            AutoModelForCTC as _AutoModelForCTC,
            Wav2Vec2Processor as _Wav2Vec2Processor,
            pipeline as _pipeline,
        )
        from huggingface_hub import HfApi as _HfApi, model_info as _model_info

        AutoModelForSpeechSeq2Seq = _AutoModelForSpeechSeq2Seq
        AutoProcessor = _AutoProcessor
        AutoModelForCTC = _AutoModelForCTC
        Wav2Vec2Processor = _Wav2Vec2Processor
        pipeline = _pipeline
        HfApi = _HfApi
        model_info = _model_info
        print("[INFO] Transformers and HuggingFace Hub loaded")


# Suppress pydub ffmpeg warning - ffmpeg is only needed for specific audio formats
# WAV files (which we use) don't require ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
# AudioSegment will be imported lazily when needed
AudioSegment = None


def _lazy_import_audio():
    """Import audio processing libraries only when needed"""
    global AudioSegment
    if AudioSegment is None:
        from pydub import AudioSegment as _AudioSegment

        AudioSegment = _AudioSegment
        print("[INFO] AudioSegment loaded")


import tempfile
import uuid
import random


# Whisper decoding parameters optimized for streaming (3s chunks)
LIVE_TRANSCRIPTION_PARAMS = {
    "beam_size": 3,  # Matches config.default.json — accuracy win over greedy, live loop tolerates it
    "best_of": 1,  # No sampling
    "temperature": 0.0,  # Deterministic
    "condition_on_previous_text": False,  # Chunks lack context
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}

# Whisper decoding parameters optimized for file chunks (30s)
FILE_TRANSCRIPTION_PARAMS = {
    "beam_size": 5,  # Better quality for longer audio
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),  # Fallback on failure
    "condition_on_previous_text": True,  # Chunks have context
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


# Configuration file management
# The canonical default/template config lives in config.default.json (bundled at
# BUNDLE_DIR when compiled). It is used only to seed a fresh config.json on first
# run, or to recover from a missing/corrupted config.json — see load_config().
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
CONFIG_TEMPLATE_FILE = os.path.join(BUNDLE_DIR, "config", "config.default.json")

# Serializes all writers to config.json so concurrent endpoint saves cannot
# interleave and corrupt the file.
_config_file_lock = threading.Lock()

# Guards in-memory read-modify-write of the global `config` dict at the hotspots
# (the deep-merge in update_config, save_config's serialization, and the
# snapshots pushed to the transcription worker) so a reader never observes a
# half-mutated config. Reentrant: nested save_config / snapshot calls on one
# thread must not deadlock. Acquire order is always _config_lock -> _config_file_lock.
_config_lock = threading.RLock()


def _config_snapshot():
    """A coherent deep copy of the global config, taken under _config_lock, for
    pushing to the hot-reload queue (a shallow copy would still share nested
    dicts and could be observed mid-mutation)."""
    import copy as _copy
    with _config_lock:
        return _copy.deepcopy(config)


# Config/validation/version helpers live in stt/config_utils.py (importable,
# unit-tested); names are re-imported here so call sites stay unchanged.
from stt import config_utils as _config_utils
from stt.config_utils import (  # noqa: F401
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    _atomic_write_json,
    _merge_missing_keys,
    validate_file,
    is_known_timezone as _is_known_timezone,
    resolve_timezone as _resolve_timezone,
    system_timezone as _system_timezone,
)


def save_config(config_to_save):
    """Save configuration to config.json atomically with error handling."""
    try:
        with _config_lock, _config_file_lock:
            _atomic_write_json(CONFIG_FILE, config_to_save)
        print(f"[OK] Configuration saved to '{CONFIG_FILE}'")
        # Single choke point for session provenance: any settings change that
        # reaches disk is recorded against the running session, whichever route
        # made it. Guarded because config persistence is critical and must not
        # fail on a provenance problem — including a NameError if some future
        # caller invokes save_config before this module finishes importing.
        try:
            _sync_session_meta_from_config()
        except Exception as e:
            print(f"[SESSION-META] WARNING: could not sync provenance after save: {e}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")
        return False


def _get_install_id():
    """Stable anonymous UUID for the live-map ping; generated once and persisted."""
    analytics = config.get("analytics", {})
    iid = (analytics.get("install_id") or "").strip()
    if not iid:
        iid = str(uuid.uuid4())
        analytics["install_id"] = iid
        config["analytics"] = analytics
        save_config(config)
    return iid


# Word highlighting uses a separate config file
WORD_HIGHLIGHTING_FILE = _seed_from_bundle("word_highlighting.json")


def load_word_highlighting():
    """Load word highlighting configuration from separate file"""
    if os.path.exists(WORD_HIGHLIGHTING_FILE):
        try:
            with open(WORD_HIGHLIGHTING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load word highlighting config: {e}")
    return {"enabled": True, "words": []}


def save_word_highlighting(data):
    """Save word highlighting configuration to separate file"""
    try:
        _atomic_write_json(WORD_HIGHLIGHTING_FILE, data, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save word highlighting config: {e}")
        return False


def _restore_config_from_template(reason=""):
    """Copy the bundled config.default.json over CONFIG_FILE. Returns True on success."""
    return _config_utils.restore_config_from_template(CONFIG_TEMPLATE_FILE, CONFIG_FILE, reason)

def load_config():
    """Load configuration from config.json.

    config.default.json (bundled, read from BUNDLE_DIR) is the canonical template.
    It seeds a fresh config.json on first run and is used to recover from a
    missing or corrupted config.json. On every load, settings present in the
    template but missing from config.json (added in later releases) are patched
    in and written back; user-set values are never overwritten."""
    # First run: no config.json yet -> seed from the bundled template.
    if not os.path.exists(CONFIG_FILE):
        if _restore_config_from_template("create config.json"):
            print(f"[OK] Created '{CONFIG_FILE}' from config.default.json")
            print("[NOTE] Edit this file to configure your settings.")
        else:
            raise FileNotFoundError(
                f"Neither '{CONFIG_FILE}' nor template '{CONFIG_TEMPLATE_FILE}' exist; cannot start."
            )

    try:
        # encoding= is load-bearing: config.json is seeded as a byte-copy of the
        # UTF-8 template, whose hallucination-phrase list is Cyrillic. Without it,
        # Windows reads cp1252 and dies on bytes cp1252 leaves undefined (0x81…),
        # which quarantined the freshly seeded config in an infinite crash loop
        # (seen in the field on 26.1.146).
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[OK] Loaded configuration from '{CONFIG_FILE}'")
    except OSError as e:
        # A read failure (permissions, I/O) is NOT corruption: the file's content
        # may be perfectly good, so displacing it and seeding defaults would
        # silently discard the user's settings. Fail loudly instead.
        raise RuntimeError(
            f"Cannot read '{CONFIG_FILE}': {e}. If the file is owned by another "
            f"user (e.g. root after running the server with sudo), fix it with: "
            f"sudo chmod 644 {CONFIG_FILE}"
        ) from e
    except Exception as e:
        # Corrupted config.json (readable but unparseable): back up the bad file,
        # then rewrite a fresh one from the template so the app can still start.
        print(f"[CONFIG] ERROR: could not parse '{CONFIG_FILE}': {e}")
        from datetime import datetime
        import shutil
        try:
            corrupt_path = f"{CONFIG_FILE}.corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(CONFIG_FILE, corrupt_path)
            print(f"[CONFIG] Backed up corrupt config to '{corrupt_path}'")
        except Exception as move_err:
            print(f"[CONFIG] WARNING: could not back up corrupt config: {move_err}")
        if not _restore_config_from_template("recover config.json"):
            raise
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[CONFIG] Recovered '{CONFIG_FILE}' from config.default.json")

    # Migrate old use_english_model format to new .en model names
    migrated = False
    for section in ["model", "file_transcription"]:
        if section in config:
            # Handle top-level model section
            if section == "model" and "whisper" in config[section]:
                whisper = config[section]["whisper"]
                if whisper.get("use_english_model", False):
                    model = whisper.get("model", "base")
                    if model in ["tiny", "base", "small", "medium"] and not model.endswith(".en"):
                        whisper["model"] = f"{model}.en"
                        print(f"[MIGRATION] Converted model '{model}' to '{model}.en'")
                        migrated = True
                # Remove the old flag
                if "use_english_model" in whisper:
                    whisper.pop("use_english_model")
                    migrated = True

            # Handle file_transcription model section
            elif section == "file_transcription" and "model" in config[section]:
                if "whisper" in config[section]["model"]:
                    whisper = config[section]["model"]["whisper"]
                    if whisper.get("use_english_model", False):
                        model = whisper.get("model", "base")
                        if model in ["tiny", "base", "small", "medium"] and not model.endswith(".en"):
                            whisper["model"] = f"{model}.en"
                            print(f"[MIGRATION] Converted file transcription model '{model}' to '{model}.en'")
                            migrated = True
                    # Remove the old flag
                    if "use_english_model" in whisper:
                        whisper.pop("use_english_model")
                        migrated = True

    # Save migrated config
    if migrated:
        try:
            with _config_file_lock:
                _atomic_write_json(CONFIG_FILE, config)
            print("[MIGRATION] Config file updated and saved")
        except Exception as e:
            print(f"[MIGRATION] Warning: Could not save migrated config: {e}")

    # Patch in settings added since this config.json was created (e.g. a new
    # auto_update block), so older installs pick up new defaults without manual
    # edits. Only missing keys are added; user-set values are never overwritten.
    try:
        with open(CONFIG_TEMPLATE_FILE, "r", encoding="utf-8") as f:
            template = json.load(f)
        if _merge_missing_keys(config, template):
            with _config_file_lock:
                _atomic_write_json(CONFIG_FILE, config)
            print("[CONFIG] Added new default settings missing from config.json")
    except Exception as e:
        print(f"[CONFIG] Warning: could not patch missing defaults: {e}")

    return config


# Load configuration
config = load_config()


# Project default DSN (ingest-only key). Overridable via crash_reporting.sentry_dsn;
# disable everything with crash_reporting.sentry_enabled = false.
_SENTRY_DEFAULT_DSN = "https://eff01fdec5e9330b80ffd96093038588@o4511050918723584.ingest.us.sentry.io/4511714251702272"


def _sentry_scrub_request(event, hint):
    """Drop the request body, query string and headers from every outgoing
    event. Transcript segments, glossary terms, file paths and the ?key=
    access token all travel in those, and none of them are ours to collect —
    the UI promises no transcription content is sent, and this is what makes
    that true. Stack traces, versions and OS context still go through."""
    request = event.get("request")
    if request:
        for key in ("data", "query_string", "cookies", "headers", "env"):
            request.pop(key, None)
    # ArgvIntegration attaches the command line, which carries the install path
    # (frequently including the operator's username).
    extra = event.get("extra")
    if extra:
        extra.pop("sys.argv", None)
    # Subprocess spans are named after the full command line, which for ffmpeg
    # carries the input device name and for media jobs the file path.
    for span in event.get("spans") or ():
        if span.get("op", "").startswith("subprocess"):
            span["description"] = (span.get("description", "").split() or ["subprocess"])[0]
            span.pop("data", None)
    return event


def _init_sentry():
    """Sentry error reporting, logs, tracing, and profiling — on by default,
    disabled via crash_reporting.sentry_enabled = false. Runs in the web
    process and again in the transcription worker (which re-imports this
    module), so both report. Never blocks boot.

    Deliberately configured to carry no user content: no request bodies, no
    frame locals, no PII (IP addresses/headers). See _sentry_scrub_request."""
    cr = config.get("crash_reporting", {})
    if not cr.get("sentry_enabled", True):
        return
    dsn = (cr.get("sentry_dsn", "") or "").strip() or _SENTRY_DEFAULT_DSN
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
        try:
            with open(os.path.join(BUNDLE_DIR, "VERSION"), encoding="utf-8") as _vf:
                release = "stt@" + _vf.read().strip()
        except OSError:
            release = None
        sentry_sdk.init(
            dsn=dsn,
            integrations=[FlaskIntegration()],
            release=release,
            # PII off: no client IP addresses, headers or cookies. Opt-in via a
            # new key, so installs carrying the old sentry_send_pii=true default
            # in their config.json are healed rather than grandfathered.
            send_default_pii=bool(cr.get("sentry_send_pii_optin", False)),
            # Request bodies can hold transcript text (/api/translate),
            # dictionary entries and file paths — never send them.
            max_request_body_size="never",
            # Frame locals routinely hold the text being transcribed/translated.
            include_local_variables=False,
            # Runs on errors and on sampled transactions alike.
            before_send=_sentry_scrub_request,
            before_send_transaction=_sentry_scrub_request,
            # Left unset the SDK fills this with socket.gethostname(), which on
            # a church PC is often the building or the operator's name.
            server_name="stt",
            # Forward logging-module records as Sentry Logs (watchdog uses
            # logging heavily; the server mostly print()s, which is not captured)
            enable_logs=bool(cr.get("sentry_enable_logs", True)),
            # NOT 1.0: the web UI polls ~2x/second, so full tracing would be
            # ~100k+ transactions per viewer per day
            traces_sample_rate=float(cr.get("sentry_traces_sample_rate", 0.1)),
            # Continuous profiling is OFF unless explicitly opted in: the SDK's
            # profiler thread (continuous_profiler._sample_stack/extract_stack)
            # repeatedly segfaulted the server while sampling other threads'
            # stacks — every faulthandler dump named it as the crashing thread.
            # Gated behind a new key (not the old sample-rate) so installs with
            # the old 1.0 default materialized in config.json are healed too.
            profile_session_sample_rate=(
                float(cr.get("sentry_profile_session_sample_rate", 1.0))
                if cr.get("sentry_profiling_enabled", False) else 0.0
            ),
            profile_lifecycle=cr.get("sentry_profile_lifecycle", "trace"),
        )
        sentry_sdk.set_tag("process", "server")
        _prof = "profiling" if cr.get("sentry_profiling_enabled", False) else "no profiling"
        print(f"[SENTRY] Error reporting enabled (logs + traces + {_prof})")
    except ImportError:
        print("[SENTRY] sentry-sdk is not installed — rerun the installer or: uv pip install 'sentry-sdk[flask]'")
    except Exception as e:
        print(f"[SENTRY] Init failed (continuing without): {e}")


_init_sentry()

# Create empty word highlighting file if it doesn't exist
if not os.path.exists(WORD_HIGHLIGHTING_FILE):
    print(f"[INIT] Creating empty {WORD_HIGHLIGHTING_FILE}")
    save_word_highlighting({"enabled": True, "words": []})

# Calibration mode state
calibration_mode = False
calibration_data = None

# Remote translation state
_pending_pair_requests = {}    # {ip: {"code": str, "expires": float, "attempts": int}}
_pending_pair_lock = threading.Lock()
PAIR_MAX_PENDING = 20          # cap unauthenticated pending requests
PAIR_MAX_ATTEMPTS = 5          # wrong-code guesses before the pending code is voided
_translation_clients = {}      # {ip: last_seen_timestamp}
_translation_clients_lock = threading.Lock()
_trusted_translation_clients = set(
    config.get("live_translation", {}).get("trusted_clients", [])
)
_session_glossary_override = None  # {"glossary": {...}} pushed by a paired Machine A for this session; None = use custom_dictionary.json

# What a paired Machine A has actually taken over on this machine, as opposed to
# what it could take over. A pairing on its own dictates nothing: A pushes a
# language only when it switches one, and a glossary only when one is edited.
# Reported by /api/translation/status so this machine's settings page locks the
# controls A really owns instead of every control it might — locking on the mere
# existence of a pairing shut the operator out of the LLM settings, which A never
# sends at all.
#
# The model is deliberately absent. An offload server runs its own engine and its
# own model, and A has no say in either: it asks for a translation and is told
# what produced it. One machine owning that choice is the whole point — two
# machines negotiating it was a picker on A listing B's files, a push route, a
# reload, and a lock group, all to arrive at a setting B already had.
_a_pushed = {"language": False, "glossary": False}


def _is_trusted_translation_client(ip):
    return ip in _trusted_translation_clients


def _add_trusted_client(ip, port=None):
    _trusted_translation_clients.add(ip)
    if "live_translation" not in config:
        config["live_translation"] = {}
    trusted = config["live_translation"].setdefault("trusted_clients", [])
    if ip not in trusted:
        trusted.append(ip)
    if port:
        _remember_client_port(ip, port, persist=False)
    save_config(config)


def _forget_client_port(ip):
    """Drop a client's stored UI port when it is unpaired."""
    _translation_client_ports.pop(ip, None)
    stored = config.get("live_translation", {}).get("trusted_client_ports")
    if isinstance(stored, dict) and stored.pop(ip, None) is not None:
        save_config(config)


def _remember_client_port(ip, port, persist=True):
    """Record where a paired client's own UI lives, so we can link back to it.

    Learned at pairing and refreshed by the heartbeat. Stored in its own config
    map rather than folded into trusted_clients, which is a plain list of IPs
    that older configs and _is_trusted_translation_client both rely on.
    """
    port = coerce_int(port, 0, lo=1, hi=65535)
    if not port:
        return
    _translation_client_ports[ip] = port
    stored = config.setdefault("live_translation", {}).setdefault("trusted_client_ports", {})
    if stored.get(ip) != port:
        stored[ip] = port
        if persist:
            save_config(config)


# {ip: port} for paired clients that have told us where their own UI lives. Kept
# beside _translation_clients rather than in it, because that maps ip -> last-seen
# timestamp and several readers do arithmetic on the value. Seeded from config so
# a paired client can be linked to before it has run a session since our restart.
_translation_client_ports = {
    str(ip): int(port)
    for ip, port in (config.get("live_translation", {}).get("trusted_client_ports", {}) or {}).items()
    if str(port).isdigit()
}


def _register_translation_client(ip, port=None):
    with _translation_clients_lock:
        _translation_clients[ip] = time.time()
        if port:
            _translation_client_ports[ip] = port

# Generate random password if not configured
password_auth_config = config.get("web_server", {}).get("password_auth", {})
if password_auth_config.get("enabled", False) and not password_auth_config.get("password", ""):
    import secrets
    import string
    # Generate a random 12-character password
    alphabet = string.ascii_letters + string.digits
    random_password = ''.join(secrets.choice(alphabet) for i in range(12))

    # Update config with generated password
    if "web_server" not in config:
        config["web_server"] = {}
    if "password_auth" not in config["web_server"]:
        config["web_server"]["password_auth"] = {}
    config["web_server"]["password_auth"]["password"] = random_password

    # Save config
    save_config(config)

    print("=" * 80)
    print("[AUTH] Password authentication enabled with auto-generated password:")
    print(f"[AUTH] Password: {random_password}")
    print("[AUTH] Save this password to access settings from non-whitelisted IPs.")
    print("[AUTH] You can change it in config.json under web_server.password_auth.password")
    print("=" * 80)


# Timezone Helper Function
def get_configured_timezone():
    """The timezone transcript rows are stamped in.

    Honours config["timezone"]; mode "auto" (the default) means the machine's own
    zone, anything else uses "value" as an IANA name. A name this machine cannot
    load falls back to the system zone with a warning rather than raising —
    timestamps are written on every row, so a typo in a setting must not be able to
    stop a service from recording.
    """
    tz, note = _resolve_timezone(config.get("timezone"), _system_timezone())
    if note:
        print(f"[TIMEZONE] {note}")
    return tz


# Load configured timezone. Read once at import: the value is stamped onto rows in
# the worker, which re-imports this module, so a change needs a restart to take
# effect on both sides — which is what the save endpoint tells the operator.
configured_timezone = get_configured_timezone()
print(f"[OK] Using timezone: {configured_timezone}")


# ====================================================================================
# NLLB Translation Support - Translate to 200+ languages
# ====================================================================================

# NLLB language tables + model catalog live in stt/nllb_catalog.py (importable, tested).
from stt.nllb_catalog import (  # noqa: F401
    NLLB_LANG_CODES,
    TRANSLATION_LANGUAGES,
    get_nllb_model_description,
    get_default_nllb_models,
    build_madlad_input,
    is_madlad_model,
    get_default_madlad_models,
    madlad_anti_repetition_defaults,
    supported_target,
    languages_for_method,
    resolve_translation_model_id as _resolve_translation_model_id,
)
from stt.ct2_translate import (  # noqa: F401
    resolve_compute_type as _ct2_resolve_compute_type,
    ct2_model_dir as _ct2_model_dir,
    nllb_ct2_target_prefix as _ct2_nllb_target_prefix,
    strip_target_prefix as _ct2_strip_target_prefix,
    nllb_source_tokens as _ct2_nllb_source_tokens,
    madlad_source_tokens as _ct2_madlad_source_tokens,
    decode_ct2_tokens as _ct2_decode_tokens,
    score_to_confidence as _ct2_score_to_confidence,
)
# Session provenance: which models/decode settings produced a given transcript.
from stt.llm_translate import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE as _DEFAULT_LLM_SYSTEM_PROMPT,
    build_chat_messages as _llm_chat_messages,
    build_chat_payload as _llm_chat_payload,
    build_system_prompt as _llm_system_prompt,
    check_translation as _llm_check,
    retry_system_prompt as _llm_retry_prompt,
    fit_context_prefix as _llm_fit_context,
    input_fits as _llm_input_fits,
    input_token_budget as _llm_input_budget,
    local_model_path as _llm_local_model_path,
    looks_like_reasoning_model as _llm_looks_like_reasoning,
    resolve_gpu_layers as _llm_resolve_gpu_layers,
    scan_gguf_models as _scan_gguf_models,
    extract_chat_text as _llm_extract_text,
    uses_local_llm as _uses_local_llm,
)
# Service-phase detection lives in stt/service_phase.py (importable, unit-tested);
# the monolith supplies the connection and the live config and does nothing else.
from stt.service_phase import (
    analyze as _service_phase_analyze,
    delete_correction as _service_phase_delete_correction,
    delete_correction_by_id as _service_phase_delete_correction_by_id,
    ensure_tables as _service_phase_ensure_tables,
    load_analysis as _service_phase_load,
    load_corrections as _service_phase_corrections,
    read_rows as _service_phase_rows,
    save_analysis as _service_phase_save,
    save_correction as _service_phase_save_correction,
    save_group_correction as _service_phase_save_group,
)
from stt.phase_rules import load_rules as _phase_rules_load
from stt.phase_learn import (
    apply_proposals as _phase_learn_apply,
    collect as _phase_learn_collect,
    propose_all as _phase_learn_propose,
)
from stt.db_maintenance import (
    checkpoint_and_release as _db_checkpoint_and_release,
    sweep_orphaned_sidecars as _db_sweep_sidecars,
    _iter_databases as _db_iter_databases,
)
from stt.session_meta import (
    append_changes as _session_meta_append,
    append_new_changes as _session_meta_append_new,
    build_session_meta as _build_session_meta,
    changed_keys as _session_meta_changed_keys,
    glossary_provenance as _glossary_provenance,
    is_offloaded as _translation_is_offloaded,
    uses_llm as _translation_uses_llm,
    latest_values as _session_meta_latest,
    load_session_meta as _load_session_meta,
    remote_provenance as _remote_provenance,
    write_missing as _session_meta_write_missing,
    write_session_meta as _write_session_meta,
    asr_row_label as _session_asr_row_label,
    mt_row_label as _session_mt_row_label,
    row_label_if_changed as _session_row_label_if_changed,
    MT_ENGINE_LLM,
    MT_ENGINE_NMT,
    MT_ENGINE_NONE,
    MT_ENGINE_REMOTE,
    MT_ENGINE_WHISPER,
)

# Default MADLAD-400 model when the translation engine is set to "madlad".
MADLAD_DEFAULT_MODEL = "google/madlad400-3b-mt"


def _local_fallback_ready():
    """Whether this machine could translate locally if the remote went away.

    Answers the question the "fall back to local translation" setting quietly
    assumes: is a local model actually on disk. Whisper-translate modes need no
    separate model — the ASR pass produces the translation — and an LLM session
    is ready when its GGUF is there. Checked on disk rather than by loading,
    because this runs on a settings page, not on the caption path.
    """
    lt = config.get("live_translation", {})
    method = (lt.get("translation_method") or "nllb").strip().lower()
    if method in ("whisper_translate", "whisper_forced_lang"):
        return True
    if method == "llm":
        llm_cfg = lt.get("llm") or {}
        if (llm_cfg.get("provider") or "endpoint").strip().lower() != "local":
            return bool((llm_cfg.get("endpoint") or "").strip())
        path = (llm_cfg.get("gguf_path") or "").strip() or _llm_local_model_path(
            MODELS_DIR, (llm_cfg.get("gguf_repo") or "").strip(),
            (llm_cfg.get("gguf_file") or "").strip())
        return bool(path and os.path.isfile(path))
    model_id = _resolve_live_translation_model_id(lt)
    if not model_id:
        return False
    return dir_has_weights(os.path.join(MODELS_DIR, model_id.replace("/", "--")))


def _resolve_live_translation_model_id(lt_cfg):
    """Effective live-translation model id for the configured engine.

    Thin wrapper over stt.nllb_catalog.resolve_translation_model_id so session
    provenance (stt/session_meta.py) records the model that actually loads
    rather than a possibly-stale configured string, with no chance of the two
    resolutions drifting apart.
    """
    return _resolve_translation_model_id(lt_cfg, MADLAD_DEFAULT_MODEL)


def _maybe_half_translation_model(model, device, use_fp16):
    """Convert the model to fp16 on GPU when requested (CUDA/MPS only — see
    _should_use_fp16). Returns (model, applied). Never raises: on failure it
    keeps the fp32 model so a bad fp16 path degrades instead of failing the load."""
    if not _should_use_fp16(use_fp16, device):
        return model, False
    try:
        return model.half(), True
    except Exception as e:
        print(f"[LIVE-TRANSLATION] fp16 conversion failed, staying fp32: {e}")
        return model, False


def _load_ct2_translator(hf_model_path, model_id, use_gpu, ct2_compute_type):
    """Load (converting once, cached) a CTranslate2 Translator for the model at
    hf_model_path. CT2 has no Metal backend, so Apple Silicon runs CPU int8 —
    which is the whole point (MADLAD-3B in ~3 GB). Sets _live_translation_device.
    Raises if the HF model isn't downloaded locally (conversion needs the weights)."""
    global _live_translation_device
    import ctranslate2
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    compute_type = _ct2_resolve_compute_type(ct2_compute_type or "auto", device)
    if not os.path.isdir(hf_model_path):
        raise RuntimeError(
            f"CTranslate2 backend needs '{model_id}' downloaded locally first "
            "(Model Manager). Cannot convert a hub-only model.")
    ct2_dir = _ct2_model_dir(hf_model_path, compute_type)
    if not os.path.isdir(ct2_dir):
        # One-time conversion from the official HF weights (transiently needs
        # ~model-size RAM). Cached beside the HF model, keyed by compute type.
        print(f"[CT2] Converting {model_id} -> {ct2_dir} ({compute_type})... one-time", flush=True)
        from ctranslate2.converters import TransformersConverter
        TransformersConverter(hf_model_path).convert(ct2_dir, quantization=compute_type, force=False)
    _lt = config.get("live_translation", {})
    intra = max(0, int(_lt.get("ct2_intra_threads", 4)))   # CPU compute threads (P-cores)
    inter = max(1, int(_lt.get("ct2_inter_threads", 1)))   # parallel batches
    translator = ctranslate2.Translator(ct2_dir, device=device, compute_type=compute_type,
                                        intra_threads=intra, inter_threads=inter)
    _live_translation_device = device
    print(f"[CT2] Translator loaded ({device}, {compute_type}, intra={intra} inter={inter})")
    return translator


def load_translation_model(use_gpu=True, model_id=None, use_fp16=False, use_ct2=False, ct2_compute_type="auto"):
    """
    Load a translation model (NLLB-200 or MADLAD-400).

    Args:
        use_gpu: Whether to use GPU acceleration
        model_id: HuggingFace model ID (e.g., "facebook/nllb-200-distilled-600M")
                  If None, defaults to facebook/nllb-200-distilled-600M
        use_fp16: Load in half precision on GPU (MPS/CUDA only; ignored on CPU)
        use_ct2: Use the CTranslate2 int8 backend (returns a Translator instead
                 of a transformers model; converts once from local HF weights)
        ct2_compute_type: CT2 quantization ("auto" -> int8 on CPU, int8_float16 on CUDA)

    Returns:
        Tuple of (model, tokenizer) — model is a transformers model or, with
        use_ct2, a ctranslate2.Translator.
    """
    _lazy_import_ml_libraries()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Use provided model_id or default
    if model_id is None:
        model_id = "facebook/nllb-200-distilled-600M"

    # Check ./models/ directory first for local copy
    local_dir_name = model_id.replace("/", "--")
    local_model_path = os.path.join(MODELS_DIR, local_dir_name)

    if os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"[INFO] Loading translation model from local: {model_path}")
    else:
        model_path = model_id
        print(f"[INFO] Loading translation model from HuggingFace: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    global _live_translation_is_ct2
    if use_ct2:
        translator = _load_ct2_translator(model_path, model_id, use_gpu, ct2_compute_type)
        _live_translation_is_ct2 = True
        return translator, tokenizer
    _live_translation_is_ct2 = False

    bin_path = os.path.join(model_path, "pytorch_model.bin") if os.path.isdir(model_path) else None
    if bin_path and os.path.exists(bin_path) and not os.path.exists(os.path.join(model_path, "model.safetensors")):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_config(cfg)
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    else:
        # Load straight to fp16 when the model is headed for a GPU and fp16 was
        # asked for, instead of loading fp32 and halving afterwards. That order
        # holds both copies at once: MADLAD-3B is ~11.8 GB of fp32 weights, so the
        # peak is ~17.7 GB — more than a 16 GB unified-memory Mac has, which put
        # MPS out of reach for that model even though its fp16 weights are ~5.9 GB.
        # low_cpu_mem_usage streams the checkpoint rather than building a second
        # full copy in RAM first.
        _load_dtype = _translation_load_dtype(
            use_fp16, use_gpu, torch.cuda.is_available(),
            bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        )
        _load_kwargs = {}
        if _load_dtype:
            _load_kwargs = {"torch_dtype": getattr(torch, _load_dtype), "low_cpu_mem_usage": True}
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **_load_kwargs)

    global _live_translation_device
    # _maybe_half_translation_model stays as the fallback: it is a no-op when the
    # weights already loaded as fp16, and still does the conversion for the
    # state-dict branch above (which cannot take a load dtype).
    if use_gpu and torch.cuda.is_available():
        model = model.to("cuda")
        _live_translation_device = "cuda"
        model, _fp16 = _maybe_half_translation_model(model, "cuda", use_fp16)
        print(f"[INFO] Translation model loaded on GPU (CUDA) (fp16={_fp16})")
    elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
        _live_translation_device = "mps"
        model, _fp16 = _maybe_half_translation_model(model, "mps", use_fp16)
        print(f"[INFO] Translation model loaded on GPU (MPS) (fp16={_fp16})")
    else:
        _live_translation_device = "cpu"
        if use_fp16:
            print("[LIVE-TRANSLATION] use_fp16 ignored on CPU")
        print("[INFO] Translation model loaded on CPU")
        if use_gpu:
            # GPU was requested but no accelerator was usable — translations will
            # be seconds-per-sentence instead of sub-second. This is the silent
            # failure mode behind "translations got slower" field reports (e.g. a
            # torch upgrade that dropped CUDA support for the installed driver).
            print(f"[WARNING] Translation model '{model_id}' requested GPU but is running on CPU — "
                  "expect slow translations. Check torch CUDA/MPS availability "
                  "(.venv python -c 'import torch; print(torch.cuda.is_available())').", flush=True)

    return model, tokenizer


# Glossary application and the TranslationCache live in stt/translation_utils.py
# (importable, unit-tested); the wrapper below resolves config/session state.
from stt.translation_utils import (
    TranslationCache,
    TextTranslationCache,
    apply_glossary as _apply_glossary_dict,
    should_cache_translation as _should_cache_translation,
    should_use_fp16 as _should_use_fp16,
    translation_load_dtype as _translation_load_dtype,
)


def _apply_glossary(text, source_lang, target_lang):
    """Apply NLLB glossary post-processing replacements from custom dictionary."""
    try:
        dict_config = config.get("custom_dictionary", {})
        override = _session_glossary_override
        if override is not None:
            # Offloaded session: honor the CLIENT's glossary-enabled flag when it
            # sent one, else fall back to this machine's own config.
            enabled = override.get("nllb_glossary_enabled")
            if enabled is None:
                enabled = dict_config.get("nllb_glossary_enabled", False)
            if not enabled:
                return text
            dictionary = override
        elif not dict_config.get("nllb_glossary_enabled", False):
            return text
        else:
            dict_file = dict_config.get("file", "custom_dictionary.json")
            if not os.path.isabs(dict_file):
                dict_file = os.path.join(CONFIG_DIR, dict_file)

            if not os.path.exists(dict_file):
                return text

            import json as _json
            with open(dict_file, "r", encoding="utf-8") as f:
                dictionary = _json.load(f)

        return _apply_glossary_dict(text, source_lang, target_lang, dictionary)
    except Exception as e:
        print(f"[WARNING] Glossary application failed: {e}")
        return text


def _session_meta_enabled(session_config=None):
    """Whether provenance recording is on (config/config.default.json: session_meta)."""
    return bool((session_config or config).get("session_meta", {}).get("enabled", True))


def _current_session_meta(session_config=None):
    """Provenance mapping for the running config, or {} when disabled.

    The ASR model loads at init step 3 and the database is created at step 4, so
    what actually loaded is already known here and belongs in the session-start
    values rather than in the change log. The translation model loads lazily
    (usually after the db exists), so its effective values are appended later by
    _record_session_meta_change().

    ``session_config`` describes a config other than this process's global one.
    The transcription worker outlives a Start/Stop cycle and is reused, so its
    module-level `config` dates from when the process spawned — it reloads a
    fresh copy into process_config at each session start (see thread1_function)
    and must pass it here. Reading the global instead recorded a session as
    translating locally with MADLAD while it was in fact offloading every caption
    to a paired remote, which is precisely the misattribution this table exists
    to prevent.
    """
    if not _session_meta_enabled(session_config):
        return {}
    meta = _build_session_meta(
        session_config or config, SERVER_VERSION, SERVER_COMMIT, SERVER_DESCRIBE,
        socket.gethostname(), MADLAD_DEFAULT_MODEL,
    )
    try:
        if transcription_state is not None:
            loaded = transcription_state.get("loaded_model")
            if loaded:
                meta["asr.effective.model"] = str(loaded)
            device = transcription_state.get("loaded_model_device")
            if device:
                meta["asr.effective.device"] = str(device)
    except Exception:
        pass  # provenance is best-effort; a missing state proxy is not an error
    try:
        # The glossary terms rewrite translated captions and live in a file, not in
        # config, so build_session_meta() cannot see them. Absent glossary.file
        # means this session applies no glossary at all (a local LLM session), and
        # there is nothing to describe. An override is a paired client's own table,
        # pushed for this session: the local file is then not what ran.
        if "glossary.file" in meta:
            override = _session_glossary_override
            meta.update(_glossary_provenance(
                override if override is not None else load_custom_dictionary(),
                "paired-client" if override is not None else "local"))
    except Exception as e:
        print(f"[SESSION-META] WARNING: could not read glossary provenance: {e}")
    return meta


# Last successful answer from the paired machine about what it translates with.
# Kept because every offloaded row records the model that produced it, and probing
# per caption would put a network round trip in front of every one.
_remote_effective = {}


def _fetch_remote_provenance():
    """Ask the paired remote what it is translating with. {} if unreachable.

    On an offloaded session the remote's model does the work, so without this the
    database records nothing about what actually translated — and with a blank
    remote model ("use Machine B's own model") this box can't infer it either.
    """
    endpoint = _get_remote_endpoint_safe()
    if not endpoint:
        return {}
    try:
        import requests as _req
        r = _req.get(endpoint + "/api/translation/status", timeout=5)
        provenance = _remote_provenance(r.json())
    except Exception as e:
        print(f"[SESSION-META] Could not read remote translation provenance: {e}")
        return {}
    if provenance:
        _remote_effective.update(provenance)
    return provenance


def _record_remote_provenance_async(db_path):
    """Probe the paired remote off-thread and record it as a session-start fact.

    Off-thread because a 5s network timeout must not delay the start of
    transcription; recorded with write_missing so it lands as a base key rather
    than a timestamped change — nothing changed, we only just found out.
    """
    if not (db_path and _session_meta_enabled()):
        return

    def _probe():
        remote = _fetch_remote_provenance()
        if remote and _session_meta_write_missing(db_path, remote):
            print(f"[DB] OK: Recorded remote translation provenance ({remote.get('mt.remote.effective.model', '?')})")

    threading.Thread(target=_probe, daemon=True).start()


def _reprobe_remote_provenance_async():
    """Re-probe the remote after an offload change and append what differs.

    A change here means a different box (or a different model on the same box) is
    now translating, so this genuinely is a mid-session change and belongs in the
    timeline rather than as a base key.
    """
    if not _session_meta_enabled():
        return
    active_db = transcription_state.get("db_name") if transcription_state else None
    if not active_db:
        return

    def _probe():
        remote = _fetch_remote_provenance()
        if remote:
            # Guarded append: this fires on every offload-touching save, and the
            # remote usually answers with what it answered last time. It used to
            # diff against the raw table, whose base key still holds the
            # session-start value, so an unchanged remote was re-recorded under a
            # fresh timestamp on every save.
            _session_meta_append_new(active_db, remote)

    threading.Thread(target=_probe, daemon=True).start()


def _record_session_meta_change(**values):
    """Append a mid-session change to the active session db.

    For runtime facts that aren't in config — which model actually loaded, on
    which device. Config-driven settings are handled wholesale by
    _sync_session_meta_from_config() and must not be recorded here too.

    Append-only: the base key keeps meaning "at session start", so a session
    that began in one configuration and changed reads differently from one that
    started in the final configuration. Never raises — a failed provenance write
    must not break a live language switch.

    Records nothing when the value already stands: the translation model reloads
    for reasons other than a change of model (an unload to free VRAM, a settings
    save that rebuilds it), and each reload used to append another identical
    mt.effective.model row.
    """
    if not _session_meta_enabled():
        return
    active_db = transcription_state.get("db_name") if transcription_state else None
    if not active_db or not values:
        return
    _session_meta_append_new(
        active_db, {k: "" if v is None else str(v) for k, v in values.items()})


# Regenerated on every call, so it can never be compared for equality.
_SESSION_META_VOLATILE = frozenset({"session.started_at"})


def _sync_session_meta_from_config():
    """Append any provenance-relevant setting that just changed in config.

    Hooked into save_config (and /api/config's direct write) rather than into
    each settings route. There are ~30 config writers, and hooking them one at a
    time is exactly how the transcription-language switch and the three
    hallucination-filter toggles went unrecorded — a session could flip the ASR
    language mid-service while asr.language still read as the starting value,
    contradicting the per-row source_language column. One choke point covers
    every present and future path by construction.

    Diffs against the latest recorded value (base key as superseded by any
    appended change), so a setting that already changed isn't re-appended on
    every subsequent save. Settings that appear for the first time become base
    keys — they were never anything else — while genuine changes append.
    """
    if not _session_meta_enabled():
        return
    active_db = transcription_state.get("db_name") if transcription_state else None
    if not active_db:
        return
    try:
        stored, read_error = _load_session_meta(active_db)
        if read_error or not stored:
            return  # no session provenance yet: nothing to diff against
        latest = _session_meta_latest(stored)
        current = _current_session_meta()
        changes, additions = {}, {}
        for key, value in current.items():
            if key in _SESSION_META_VOLATILE:
                continue
            if key not in latest:
                additions[key] = value
            elif latest[key] != value:
                changes[key] = value
        if additions:
            _session_meta_write_missing(active_db, additions)
        if changes:
            _session_meta_append(active_db, changes)
    except Exception as e:
        print(f"[SESSION-META] Could not sync settings change: {e}")


def _translate_text_ct2(text, source_lang, target_lang, translator, tokenizer,
                        return_confidence=False, num_alternatives=0, generation_params=None):
    """Translate one string via the CTranslate2 int8 backend.

    CT2 works on token strings: encode with the HF tokenizer, translate_batch,
    decode. NLLB forces the target via target_prefix (stripped on decode);
    MADLAD carries it as a "<2xx>" prefix in the source text. Mirrors the
    transformers path's return shape (text, or dict with confidence/alternatives).
    """
    _is_madlad = is_madlad_model(_live_translation_model_id or "")
    gp = generation_params or {}
    num_beams = max(1, int(gp.get("num_beams", 2)))
    no_repeat = int(gp.get("no_repeat_ngram_size", 0))
    rep_pen = float(gp.get("repetition_penalty", 1.0))
    if _is_madlad:
        # MADLAD can loop; apply safe anti-repetition defaults unless tuned.
        rep_pen, no_repeat = madlad_anti_repetition_defaults(rep_pen, no_repeat)

    want_extras = return_confidence or num_alternatives > 0
    num_hyp = min(num_alternatives + 1, 5) if num_alternatives > 0 else 1
    beam_size = max(num_beams, num_hyp)

    tgt_code = None
    if _is_madlad:
        source = _ct2_madlad_source_tokens(tokenizer, text, target_lang)
        target_prefix = None
    else:
        src_code = NLLB_LANG_CODES.get(source_lang, "eng_Latn")
        tgt_code = NLLB_LANG_CODES.get(target_lang, "eng_Latn")
        source = _ct2_nllb_source_tokens(tokenizer, text, src_code)
        target_prefix = _ct2_nllb_target_prefix(tgt_code)

    _kwargs = {"beam_size": beam_size, "num_hypotheses": num_hyp,
               "return_scores": want_extras, "max_decoding_length": 1024}
    if target_prefix is not None:
        _kwargs["target_prefix"] = target_prefix
    if no_repeat > 0:
        _kwargs["no_repeat_ngram_size"] = no_repeat
    if rep_pen != 1.0:
        _kwargs["repetition_penalty"] = rep_pen

    _tr_t0 = time.perf_counter()
    results = translator.translate_batch([source], **_kwargs)
    try:
        _record_local_translate_ms((time.perf_counter() - _tr_t0) * 1000.0)
    except Exception:
        pass

    res = results[0]
    hyps = res.hypotheses
    scores = getattr(res, "scores", None) or []

    def _decode(tok_list):
        toks = _ct2_strip_target_prefix(tok_list, tgt_code) if tgt_code else tok_list
        return _ct2_decode_tokens(tokenizer, toks)

    if not hyps:
        return {"text": text, "confidence": None, "alternatives": []} if want_extras else text

    best = _apply_glossary(_decode(hyps[0]), source_lang, target_lang)
    if not want_extras:
        return best
    confidence = _ct2_score_to_confidence(scores[0]) if scores else None
    alternatives = [_apply_glossary(_decode(h), source_lang, target_lang) for h in hyps[1:]]
    return {"text": best, "confidence": confidence, "alternatives": alternatives}


def translate_text(text, source_lang, target_lang, model, tokenizer, return_confidence=False,
                   num_alternatives=0, generation_params=None, model_id=None, is_ct2=None):
    """
    Translate text using NLLB-200

    Args:
        text: Text to translate
        source_lang: Source language ISO code (e.g., 'en', 'fr')
        target_lang: Target language ISO code
        model: Loaded NLLB model
        tokenizer: Loaded NLLB tokenizer
        return_confidence: If True, return (text, confidence) tuple
        num_alternatives: Number of alternative translations to return (0 = none)
        generation_params: Dict of generation parameters (num_beams, length_penalty, etc.)

    Returns:
        Translated text string, or dict with text/confidence/alternatives if extras requested
    """
    if not text or not text.strip():
        if return_confidence or num_alternatives > 0:
            return {"text": text, "confidence": None, "alternatives": []}
        return text

    # Which engine this call is for. model/tokenizer are parameters, but these two
    # facts were read from the globals describing the LIVE model — so a caller that
    # loaded its own model (batch file transcription does) got the live model's
    # engine applied to a different model: a MADLAD file model tokenized as NLLB
    # receives no target-language prefix at all, and a transformers model would be
    # sent down the CTranslate2 path. Callers with their own model pass these.
    _ct2 = _live_translation_is_ct2 if is_ct2 is None else is_ct2
    _model_id = _live_translation_model_id if model_id is None else model_id

    # CTranslate2 backend uses a different (token-string) API — route to it and
    # leave the transformers path below untouched.
    if _ct2:
        return _translate_text_ct2(text, source_lang, target_lang, model, tokenizer,
                                   return_confidence, num_alternatives, generation_params)

    # MADLAD encodes the target as a "<2xx>" prefix on the input (no source
    # language, no forced_bos); NLLB sets tokenizer.src_lang + a forced target
    # token. Everything after tokenization is shared.
    _is_madlad = is_madlad_model(_model_id or "")
    if _is_madlad:
        inputs = tokenizer(build_madlad_input(text, target_lang),
                           return_tensors="pt", padding=True, truncation=True, max_length=1024)
    else:
        # Convert ISO codes to NLLB codes and set the source language
        tokenizer.src_lang = NLLB_LANG_CODES.get(source_lang, "eng_Latn")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    # Move inputs to same device as model (CUDA, MPS, or CPU)
    _device = next(model.parameters()).device
    if _device.type != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    # Merge user generation params with defaults
    gp = generation_params or {}
    num_beams = gp.get("num_beams", 2)
    length_penalty = gp.get("length_penalty", 1.0)
    no_repeat_ngram_size = gp.get("no_repeat_ngram_size", 0)
    repetition_penalty = gp.get("repetition_penalty", 1.0)
    if _is_madlad:
        # MADLAD can loop; apply safe anti-repetition defaults unless tuned.
        repetition_penalty, no_repeat_ngram_size = madlad_anti_repetition_defaults(
            repetition_penalty, no_repeat_ngram_size)

    generate_kwargs = {
        "max_length": 1024,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "early_stopping": True,
    }
    if not _is_madlad:
        # NLLB forces the target-language BOS token — validate it is known.
        tgt_nllb = NLLB_LANG_CODES.get(target_lang, "eng_Latn")
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
        if forced_bos_id == tokenizer.unk_token_id:
            print(f"[LIVE-TRANSLATION WARNING] Unknown target language token: {tgt_nllb} for lang={target_lang}, falling back to eng_Latn")
            forced_bos_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        generate_kwargs["forced_bos_token_id"] = forced_bos_id

    # Only add these if non-default to avoid warnings
    if no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if repetition_penalty != 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    # Enable confidence scoring and/or alternatives
    if return_confidence or num_alternatives > 0:
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True
        if num_alternatives > 0:
            generate_kwargs["num_return_sequences"] = min(num_alternatives + 1, 5)
            generate_kwargs["num_beams"] = max(5, num_alternatives + 1)

    # Generate translation
    _tr_t0 = time.perf_counter()
    translated = model.generate(**inputs, **generate_kwargs)
    try:
        _record_local_translate_ms((time.perf_counter() - _tr_t0) * 1000.0)
    except Exception:
        pass

    if return_confidence or num_alternatives > 0:
        # Extract sequences and scores
        sequences = translated.sequences
        all_decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # Compute confidence from sequence scores if available
        confidence = None
        if hasattr(translated, 'sequences_scores') and translated.sequences_scores is not None:
            import torch
            confidence = float(torch.exp(translated.sequences_scores[0]).item())

        best_text = _apply_glossary(all_decoded[0], source_lang, target_lang)
        alternatives = [_apply_glossary(t, source_lang, target_lang) for t in all_decoded[1:]] if len(all_decoded) > 1 else []

        return {"text": best_text, "confidence": confidence, "alternatives": alternatives}

    # Simple mode - decode and return
    result_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return _apply_glossary(result_text, source_lang, target_lang)


def translate_segments(segments, source_lang, target_lang, model, tokenizer, progress_callback=None, generation_params=None, context_window=1, model_id=None, is_ct2=None):
    """
    Translate a list of transcription segments

    Args:
        segments: List of segment dicts with 'text', 'start', 'end' keys
        source_lang: Source language ISO code
        target_lang: Target language ISO code
        model: Loaded NLLB model
        tokenizer: Loaded NLLB tokenizer
        progress_callback: Optional callback function(percent, status) for progress updates
        generation_params: Dict of generation parameters (num_beams, length_penalty, etc.)
        context_window: Number of segments to combine for context (1 = no batching)

    Returns:
        List of translated segment dicts with same structure
    """
    translated_segments = []
    total = len(segments)
    context_window = max(1, context_window)

    for i, seg in enumerate(segments):
        translated_text = None
        if context_window > 1 and i > 0:
            # Translate (context + target) in one call, then extract the target's
            # portion by sentence-count alignment
            start_idx = max(0, i - (context_window - 1))
            context_text = " ".join(segments[j]["text"] for j in range(start_idx, i)).strip()
            if context_text:
                num_ctx_sentences = count_sentence_units(context_text)
                combined_source = context_text + " " + seg["text"]
                ctx_char_ratio = (len(context_text) + 1) / max(1, len(combined_source))
                combined_translated = translate_text(combined_source, source_lang, target_lang, model, tokenizer, generation_params=generation_params, model_id=model_id, is_ct2=is_ct2)
                translated_text = extract_context_translation(combined_translated, num_ctx_sentences, ctx_char_ratio)
        if not translated_text:
            # No context, or alignment failed - translate the segment alone
            translated_text = translate_text(seg["text"], source_lang, target_lang, model, tokenizer, generation_params=generation_params, model_id=model_id, is_ct2=is_ct2)
        translated_segments.append({
            "text": translated_text,
            "start": seg["start"],
            "end": seg["end"]
        })

        # Report progress
        if progress_callback and total > 0:
            percent = int(70 + (i + 1) / total * 20)  # 70-90% range
            progress_callback(percent, f"Translating... ({i + 1}/{total} segments)")

    return translated_segments


def _empty_device_cache():
    """Release cached accelerator memory back to the OS (CUDA and MPS).

    Call AFTER gc.collect(): the allocators can only release blocks whose
    tensors have been collected. On MPS this is what actually returns model
    memory to macOS — without it the process keeps the freed memory reserved
    and a reload allocates a second full copy (swap death on ARM Macs)."""
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _mps = getattr(torch, "mps", None)
    if _mps is not None and hasattr(_mps, "empty_cache") and torch.backends.mps.is_available():
        try:
            _mps.empty_cache()
        except Exception:
            pass


def _preload_cudnn(use_gpu):
    """Make cuDNN 9 loadable before faster_whisper/ctranslate2 is imported.

    ctranslate2 >= 4.5 dlopens cuDNN 9 itself and knows nothing about the
    venv's nvidia-cudnn wheel or torch's bundled copy, so resolve the lib
    directory from the running interpreter (works for any Python version/venv)
    instead of trusting the loader search path. No-op on macOS or CPU-only."""
    if not use_gpu or platform == "darwin":
        return
    import ctypes
    import importlib.util

    lib_dirs = []
    for mod in ("nvidia.cudnn", "torch"):
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, AttributeError, ValueError):
            spec = None
        for loc in (spec.submodule_search_locations or []) if spec else []:
            lib_dirs.append(os.path.join(loc, "lib"))
            if platform.startswith("win"):
                lib_dirs.append(os.path.join(loc, "bin"))

    if platform.startswith("win"):
        # cuDNN DLLs live in the wheel's bin/ (nvidia-cudnn) or torch/lib;
        # register them explicitly instead of relying on PATH.
        for d in lib_dirs:
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
        return

    cudnn_libs = [
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn.so.9",
    ]
    lib_dir = next((d for d in lib_dirs
                    if os.path.exists(os.path.join(d, "libcudnn.so.9"))), None)
    if not lib_dir:
        return
    for lib in cudnn_libs:
        lib_full_path = os.path.join(lib_dir, lib)
        if os.path.exists(lib_full_path):
            try:
                ctypes.CDLL(lib_full_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"Warning: Could not preload {lib}: {e}")


def cleanup_translation_model(model, tokenizer):
    """Clean up translation model to free memory"""
    import gc

    del model
    del tokenizer

    gc.collect()
    _empty_device_cache()
    print("[CLEANUP] Translation model unloaded")


# ====================================================================================
# Live Translation - Singleton Model Management and Caching
# ====================================================================================

# Global translation model state (persistent while live translation is enabled)
_live_translation_model = None
_live_translation_tokenizer = None
_live_translation_lock = threading.Lock()
_live_translation_model_loaded = False
_live_translation_model_loading = False  # Track when model is being loaded
_live_translation_model_id = None  # Track which model is loaded to detect config changes
_live_translation_device = None  # 'cuda' | 'mps' | 'cpu' once loaded — exposed in /api/translation/status
_live_translation_is_ct2 = False  # True when the loaded model is a CTranslate2 Translator
_live_translation_target_lang = None
# Set True by load/preload, False by unload. A queued unload re-checks this under
# the lock and aborts if a preload re-requested the model in the meantime, so a
# quick stop->start doesn't ack "already loaded" and then lose the model.
_live_translation_model_wanted = False

# EMA of local NLLB translate time (ms), surfaced on the health dashboard. Set in
# translate_text() around model.generate; read in get_translation_status().
_local_translate_ms_ema = None
_local_translate_ms_lock = threading.Lock()


def _record_local_translate_ms(elapsed_ms, alpha=0.3):
    """Fold one local translation timing into the running EMA (thread-safe)."""
    global _local_translate_ms_ema
    with _local_translate_ms_lock:
        prev = _local_translate_ms_ema
        _local_translate_ms_ema = elapsed_ms if prev is None else alpha * elapsed_ms + (1 - alpha) * prev


# EMA of the round-trip time (ms) for an OFFLOADED translation as seen by this
# machine (Machine A): network + serialization + Machine B's inference. Compared
# against Machine B's own local_translate_ms_ema, it separates network overhead
# from remote inference. Set in _translate_via_remote on success.
_remote_translate_ms_ema = None
_remote_translate_ms_lock = threading.Lock()


def _record_remote_translate_ms(elapsed_ms, alpha=0.3):
    """Fold one remote-translation round-trip timing into the EMA (thread-safe)."""
    global _remote_translate_ms_ema
    with _remote_translate_ms_lock:
        prev = _remote_translate_ms_ema
        _remote_translate_ms_ema = elapsed_ms if prev is None else alpha * elapsed_ms + (1 - alpha) * prev


def is_live_translation_ready():
    """True when a live translation can actually be produced right now: a remote
    endpoint is configured, the LLM is the engine and is up, or the local NLLB model
    has finished loading. Used to avoid persisting a warmup echo (the source text
    returned unchanged while the model is still loading)."""
    live_cfg = config.get("live_translation", {})
    remote_cfg = live_cfg.get("remote", {})
    if remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
        # A machine configured to offload can produce translations for its own
        # display regardless of whether it also hosts trusted clients. (Serving
        # a paired machine's request always translates locally — see
        # translate_remote's local_only=True — so this never causes chaining.)
        return True
    if live_cfg.get("translation_method") == "llm":
        # The LLM is the engine here; the NMT model is only its fallback, and
        # llm.fallback = "skip" means it is never loaded at all. Asking the NMT
        # flag in that configuration would report "not ready" for the whole
        # session, and every translated caption would be produced and then
        # silently dropped instead of saved.
        llm_cfg = live_cfg.get("llm") or {}
        if (llm_cfg.get("provider") or "endpoint").strip().lower() == "local":
            return is_local_llm_loaded()
        return bool((llm_cfg.get("endpoint") or "").strip() and (llm_cfg.get("model") or "").strip())
    return _live_translation_model_loaded


def _warmup_translation_model(model, tokenizer, device):
    """Run one throwaway translation so MPS/CUDA compile their generate kernels
    here rather than on the first real request (a multi-second cold-start spike,
    worst right after a restart / hourly auto-update). No-op on CPU (nothing to
    pre-compile) and when disabled via config. Never raises — a warmup failure
    must not block model availability."""
    try:
        if not config.get("live_translation", {}).get("warmup", True):
            return
        # CTranslate2 Translator has no .generate — warm it via its own path.
        if _live_translation_is_ct2:
            t0 = time.perf_counter()
            _translate_text_ct2("Hello.", "en", "es", model, tokenizer)
            print(f"[LIVE-TRANSLATION] Warmup (ct2/{device}) took {(time.perf_counter() - t0) * 1000:.0f}ms")
            return
        if device == "cpu":
            return
        num_beams = int((config.get("live_translation", {}).get("generation_params", {}) or {}).get("num_beams", 2))
        t0 = time.perf_counter()
        # Mirror the real path per engine so the same generate kernels compile.
        _is_madlad = is_madlad_model(_live_translation_model_id or "")
        if _is_madlad:
            inputs = tokenizer(build_madlad_input("Hello.", "es"), return_tensors="pt")
            _warm_kwargs = {}
        else:
            tokenizer.src_lang = "eng_Latn"
            inputs = tokenizer("Hello.", return_tensors="pt")
            _warm_kwargs = {"forced_bos_token_id": tokenizer.convert_tokens_to_ids("spa_Latn")}
        _dev = next(model.parameters()).device
        if _dev.type != "cpu":
            inputs = {k: v.to(_dev) for k, v in inputs.items()}
        model.generate(**inputs, max_length=8, num_beams=num_beams, **_warm_kwargs)
        print(f"[LIVE-TRANSLATION] Warmup ({device}) took {(time.perf_counter() - t0) * 1000:.0f}ms")
    except Exception as e:
        print(f"[LIVE-TRANSLATION] Warmup failed (non-fatal): {e}")


# Latched so "no fallback model configured" is said once, not once per caption:
# this runs on the caption path. Cleared as soon as a model id appears, so a
# later change back is announced again.
_no_fallback_model_logged = False


def get_live_translation_model(use_gpu=True, model_id=None):
    """Get or load the live translation model (singleton pattern).
    If model_id differs from the currently loaded model, unloads and reloads."""
    global _live_translation_model, _live_translation_tokenizer, _live_translation_model_loaded, _live_translation_model_loading, _live_translation_model_id, _live_translation_model_wanted

    global _no_fallback_model_logged

    with _live_translation_lock:
        # No model configured is a choice, not a failure: the operator selected
        # "None" so this machine keeps no fallback weights resident — on a
        # memory-bound box that is the room a larger LLM needs. Guarding here
        # rather than at each caller covers all five of them at once, including
        # /api/translate/preload, which an offload client calls on every service
        # start and which would otherwise load the model the choice was avoiding.
        #
        # Without this, an empty id reached load_translation_model(), where ""
        # joined to MODELS_DIR is a real directory — so it tried to load the
        # models folder as a model and raised, once per caption.
        if not (model_id or "").strip():
            if not _no_fallback_model_logged:
                print("[LIVE-TRANSLATION] no fallback model configured — captions this "
                      "engine declines will stay in the source language")
                _no_fallback_model_logged = True
            return None, None
        _no_fallback_model_logged = False

        # Don't load model if transcription is actively stopping (to prevent GPU memory leak)
        status = _ts_get("status", "")
        if _live_translation_model is None and status == "stopping":
            print("[LIVE-TRANSLATION] Skipping model load - transcription is stopping")
            return None, None

        _live_translation_model_wanted = True

        _lt_cfg = config.get("live_translation", {})
        _want_ct2 = bool(_lt_cfg.get("use_ctranslate2", False))

        # If model_id OR the inference backend changed, unload so it reloads correctly.
        if _live_translation_model is not None and (
                (model_id and _live_translation_model_id and model_id != _live_translation_model_id)
                or (_want_ct2 != _live_translation_is_ct2)):
            _why = "backend" if _want_ct2 != _live_translation_is_ct2 else "model"
            print(f"[LIVE-TRANSLATION] {_why} changed ({_live_translation_model_id}, ct2={_live_translation_is_ct2}) -> "
                  f"({model_id}, ct2={_want_ct2}), reloading...")
            import gc
            del _live_translation_model
            del _live_translation_tokenizer
            _live_translation_model = None
            _live_translation_tokenizer = None
            _live_translation_model_loaded = False
            _live_translation_model_id = None
            gc.collect()
            _empty_device_cache()
            get_server_text_cache().clear()  # results were from the old model

        if _live_translation_model is None:
            _live_translation_model_loading = True
            try:
                print(f"[LIVE-TRANSLATION] Loading live translation model: {model_id or 'default'} (ct2={_want_ct2})...")
                _live_translation_model, _live_translation_tokenizer = load_translation_model(
                    use_gpu=use_gpu,
                    model_id=model_id,
                    use_fp16=_lt_cfg.get("use_fp16", False),
                    use_ct2=_want_ct2,
                    ct2_compute_type=_lt_cfg.get("ct2_compute_type", "auto"),
                )
                _live_translation_model_loaded = True
                _live_translation_model_id = model_id
                print(f"[LIVE-TRANSLATION] Live translation model loaded: {model_id or 'default'}")
                # The weights that actually loaded, and the device they landed on.
                # Only knowable here: loading is lazy, so the session db exists first.
                _record_session_meta_change(**{
                    "mt.effective.model": model_id,
                    "mt.effective.device": _live_translation_device,
                })
                _warmup_translation_model(_live_translation_model, _live_translation_tokenizer, _live_translation_device)
            finally:
                _live_translation_model_loading = False
        return _live_translation_model, _live_translation_tokenizer


def unload_live_translation_model():
    """Unload the live translation model to free GPU memory"""
    global _live_translation_model, _live_translation_tokenizer, _live_translation_model_loaded, _live_translation_model_id, _live_translation_model_wanted, _live_translation_device
    import gc

    _live_translation_model_wanted = False
    with _live_translation_lock:
        if _live_translation_model_wanted:
            # A preload re-requested the model while this unload waited on the
            # lock (quick stop->start) — keep it loaded
            print("[LIVE-TRANSLATION] Unload cancelled: model re-requested")
            return
        if _live_translation_model is not None:
            print("[LIVE-TRANSLATION] Unloading live translation model...")
            del _live_translation_model
            del _live_translation_tokenizer
            _live_translation_model = None
            _live_translation_tokenizer = None
            _live_translation_model_loaded = False
            _live_translation_model_id = None
            _live_translation_device = None
            gc.collect()
            _empty_device_cache()
            # Drop server-side text cache: it may hold results from the model
            # we're unloading (a reload could change model/precision/output).
            get_server_text_cache().clear()
            print("[LIVE-TRANSLATION] Live translation model unloaded")


def is_live_translation_model_loaded():
    """Check if the live translation model is currently loaded"""
    return _live_translation_model_loaded


def is_live_translation_model_loading():
    """Check if the live translation model is currently being loaded"""
    return _live_translation_model_loading


# ====================================================================================
# TTS (Text-to-Speech) - Multi-backend: edge-tts (cloud) and piper (local)
# ====================================================================================

_tts_piper_model = None
_tts_lock = threading.Lock()
_tts_model_loaded = False
_tts_model_loading = False
_tts_sample_rate = 22050

# Edge-TTS voice cache (populated on first request)
_edge_tts_voices = None
_edge_tts_voices_lock = threading.Lock()


def _get_tts_backend():
    """Get the configured TTS backend ('edge' or 'piper')"""
    return config.get("live_translation", {}).get("tts", {}).get("backend", "edge")


def get_edge_tts_voices():
    """Get cached list of edge-tts voices. Returns list of dicts with Name, ShortName, Gender, Locale."""
    global _edge_tts_voices
    with _edge_tts_voices_lock:
        if _edge_tts_voices is not None:
            return _edge_tts_voices
    try:
        import edge_tts
    except ImportError:
        print("[TTS] edge-tts not installed. Install with: pip install edge-tts")
        return []
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        voices = loop.run_until_complete(edge_tts.list_voices())
        loop.close()
        with _edge_tts_voices_lock:
            _edge_tts_voices = voices
        print(f"[TTS] Loaded {len(voices)} edge-tts voices")
        return voices
    except Exception as e:
        print(f"[TTS] Failed to fetch edge-tts voices: {e}")
        return []


# Piper catalog + pure voice-selection matching live in stt/tts_catalog.py; the
# shims below inject the IO (live edge-tts voices, the downloaded-model check).
from stt.tts_catalog import (  # noqa: F401
    PIPER_MODELS_CATALOG as _PIPER_MODELS_CATALOG,
    edge_voice_for_lang,
    piper_model_for_lang,
)


def _pick_default_edge_voice(lang_code):
    """Default edge-tts voice ShortName for a language code, or None."""
    return edge_voice_for_lang(lang_code, get_edge_tts_voices())


def _pick_default_piper_model(lang_code):
    """A downloaded piper model id matching a language code, or None."""
    return piper_model_for_lang(lang_code, _PIPER_MODELS_CATALOG, _is_piper_model_downloaded)


def get_tts_model(use_gpu=False, model_name=None):
    """Load piper TTS model (singleton). For edge-tts, no model loading needed."""
    global _tts_piper_model, _tts_model_loaded, _tts_model_loading, _tts_sample_rate

    backend = _get_tts_backend()

    if backend == "edge":
        _tts_model_loaded = True
        return True  # edge-tts needs no model

    # Piper backend
    with _tts_lock:
        status = _ts_get("status", "")
        if _tts_piper_model is None and status == "stopping":
            print("[TTS] Skipping piper model load - transcription is stopping")
            return None

        if _tts_piper_model is None:
            _tts_model_loading = True
            try:
                from piper import PiperVoice
                tts_config = config.get("live_translation", {}).get("tts", {})
                if model_name is None:
                    model_name = tts_config.get("piper_model", "")

                if not model_name:
                    print("[TTS ERROR] No piper model configured")
                    return None

                model_path = os.path.join(_tts_cache_dir, "piper", model_name)
                onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")] if os.path.isdir(model_path) else []
                if not onnx_files:
                    print(f"[TTS ERROR] No .onnx model found in {model_path}")
                    return None

                onnx_path = os.path.join(model_path, onnx_files[0])
                json_path = onnx_path + ".json"

                print(f"[TTS] Loading piper model: {model_name}...")
                _tts_piper_model = PiperVoice.load(onnx_path, config_path=json_path if os.path.exists(json_path) else None)
                _tts_model_loaded = True
                _tts_sample_rate = _tts_piper_model.config.sample_rate if hasattr(_tts_piper_model, 'config') else 22050
                print(f"[TTS] Piper model loaded (sample_rate={_tts_sample_rate})")
            except Exception as e:
                print(f"[TTS ERROR] Failed to load piper model: {e}")
                _tts_piper_model = None
                _tts_model_loaded = False
            finally:
                _tts_model_loading = False
        return _tts_piper_model


def unload_tts_model():
    """Unload the piper TTS model to free memory"""
    global _tts_piper_model, _tts_model_loaded
    import gc

    with _tts_lock:
        if _tts_piper_model is not None:
            print("[TTS] Unloading piper model...")
            del _tts_piper_model
            _tts_piper_model = None
            _tts_model_loaded = False
            gc.collect()
            print("[TTS] Piper model unloaded")
        else:
            _tts_model_loaded = False


def is_tts_model_loaded():
    if _get_tts_backend() == "edge":
        return True  # edge-tts is always ready
    return _tts_model_loaded


def is_tts_model_loading():
    return _tts_model_loading


def _synthesize_edge_tts(text, voice=None, speed=1.0):
    """Synthesize speech using edge-tts (Microsoft Edge cloud TTS). Returns (mp3_bytes, sample_rate) or (None, None)."""
    try:
        import asyncio
        import edge_tts
        import io

        tts_config = config.get("live_translation", {}).get("tts", {})
        if voice is None:
            voice = tts_config.get("edge_voice", "en-US-AriaNeural")

        # edge-tts rate format: "+0%", "-50%", "+100%" etc
        rate_str = "+0%"
        if speed != 1.0:
            pct = int((speed - 1.0) * 100)
            rate_str = f"{pct:+d}%"

        async def _do_synth():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        loop = asyncio.new_event_loop()
        try:
            # Bound the network call: a hung connection would otherwise block
            # the TTS emit loop indefinitely
            mp3_bytes = loop.run_until_complete(asyncio.wait_for(_do_synth(), timeout=15))
        finally:
            loop.close()

        if not mp3_bytes:
            return None, None

        return mp3_bytes, 24000  # edge-tts outputs 24kHz mp3
    except Exception as e:
        print(f"[TTS ERROR] edge-tts synthesis failed: {e}")
        return None, None


def _synthesize_piper_tts(text, language="en"):
    """Synthesize speech using piper (local). Returns (wav_bytes, sample_rate) or (None, None)."""
    model = get_tts_model()
    if model is None:
        return None, None

    try:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(_tts_sample_rate)
            model.synthesize(text, wav_file)

        return buf.getvalue(), _tts_sample_rate
    except Exception as e:
        print(f"[TTS ERROR] Piper synthesis failed: {e}")
        return None, None


# EMA of TTS synthesis time (ms), surfaced on the health dashboard.
_tts_synth_ms_ema = None
_tts_synth_ms_lock = threading.Lock()


def _record_tts_ms(elapsed_ms, alpha=0.3):
    """Fold a TTS synth timing into the EMA (thread-safe)."""
    global _tts_synth_ms_ema
    with _tts_synth_ms_lock:
        prev = _tts_synth_ms_ema
        _tts_synth_ms_ema = elapsed_ms if prev is None else alpha * elapsed_ms + (1 - alpha) * prev


def synthesize_tts(text, language="en"):
    """Synthesize speech from text. Returns (audio_bytes, sample_rate) or (None, None).
    Audio format: mp3 for edge-tts, wav for piper.
    """
    tts_config = config.get("live_translation", {}).get("tts", {})
    speed = tts_config.get("speed", 1.0)
    backend = _get_tts_backend()

    _t0 = time.perf_counter()
    if backend == "edge":
        result = _synthesize_edge_tts(text, speed=speed)
    elif backend == "piper":
        result = _synthesize_piper_tts(text, language=language)
    else:
        print(f"[TTS ERROR] Unknown backend: {backend}")
        return None, None
    try:
        if result and result[0] is not None:
            _record_tts_ms((time.perf_counter() - _t0) * 1000.0)
    except Exception:
        pass
    return result


# Global translation cache instance
_translation_cache = TranslationCache()

# Server-side cache for offloaded /api/translate requests: keyed by
# (text, langs, num_beams) so repeated phrases skip model.generate. Distinct
# from _translation_cache (segment-id keyed). Sized from config on first use.
_server_text_cache = TextTranslationCache(
    max_size=int((config.get("live_translation", {}).get("remote", {}) or {}).get("server_cache_size", 512) or 512)
)


def get_translation_cache():
    """Get the global translation cache"""
    return _translation_cache


def get_server_text_cache():
    """Get the server-side text translation cache (for offloaded requests)."""
    return _server_text_cache


# ====================================================================================
# Model Factory - Supports multiple STT model types
# ====================================================================================


class ModelFactory:
    """Factory class for loading different types of speech-to-text models"""

    _model_cache: ClassVar[dict] = {}  # shared across instances by design
    _cache_lock = threading.Lock()

    @staticmethod
    def load_model(model_config, use_gpu=True):
        """
        Load a speech-to-text model based on configuration with caching

        Args:
            model_config: Dictionary with model configuration
            use_gpu: Whether to use GPU acceleration

        Returns:
            Tuple of (model, processor/tokenizer, model_type)
        """
        # Import ML libraries before using them
        _lazy_import_ml_libraries()

        model_type = model_config.get("type", "whisper")
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Create cache key
        cache_key = f"{model_type}_{model_config!s}_{device}"

        # Check cache first
        with ModelFactory._cache_lock:
            if cache_key in ModelFactory._model_cache:
                print(f"Using cached {model_type} model on {device}")
                return ModelFactory._model_cache[cache_key]

        print(f"Loading {model_type} model on {device}...")

        try:
            if model_type == "whisper":
                # Check for backend preference (faster-whisper or standard whisper)
                backend = model_config.get("backend")
                if backend == "faster-whisper":
                    print("Using faster-whisper backend (4-10x faster)")
                    model, processor, model_type_return = ModelFactory._load_faster_whisper(
                        model_config["whisper"], use_gpu
                    )
                else:
                    print("Using standard OpenAI Whisper backend")
                    model, processor, model_type_return = ModelFactory._load_whisper(
                        model_config["whisper"], use_gpu
                    )
            elif model_type == "huggingface":
                model, processor, model_type_return = ModelFactory._load_huggingface(
                    model_config["huggingface"], device
                )
            elif model_type == "custom":
                model, processor, model_type_return = ModelFactory._load_custom(
                    model_config["custom"], device
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Cache the loaded model
            with ModelFactory._cache_lock:
                ModelFactory._model_cache[cache_key] = (
                    model,
                    processor,
                    model_type_return,
                )

            return model, processor, model_type_return
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

    @staticmethod
    def cleanup_models():
        """Clean up all cached models to free memory"""
        import gc

        with ModelFactory._cache_lock:
            # First, copy items and clear cache to remove references
            cache_items = list(ModelFactory._model_cache.items())
            ModelFactory._model_cache.clear()

            # Now delete model objects from the copied list
            for cache_key, (model, processor, _) in cache_items:
                try:
                    # Move to CPU first if possible (frees GPU memory faster)
                    if hasattr(model, "cpu"):
                        model.cpu()
                    # Delete the actual model object
                    del model
                    if processor:
                        del processor
                except Exception as e:
                    print(f"[WARNING] Error cleaning up model {cache_key}: {e}")

            # Delete the list too
            del cache_items

        # Force garbage collection OUTSIDE the lock
        # This is CRITICAL for ctranslate2/faster-whisper to release GPU memory
        gc.collect()

        # Now clear the accelerator cache (CUDA or MPS)
        try:
            _empty_device_cache()
            print("[OK] All models cleaned up from memory, GPU cache cleared")
        except Exception as e:
            print(f"[OK] All models cleaned up from memory (GPU cleanup: {e})")

    @staticmethod
    def _load_whisper(whisper_config, device):
        """Load OpenAI Whisper model"""
        _lazy_import_ml_libraries()

        model_name = whisper_config.get("model", "base")

        print(f"Loading Whisper model: {model_name}")

        # Check if model exists in ./models directory first
        models_dir = MODELS_DIR
        whisper_model_dir = os.path.join(models_dir, f"whisper-{model_name}")

        # Determine download_root based on where model is located
        download_root = None
        if os.path.exists(whisper_model_dir):
            # Model exists in new ./models location
            download_root = whisper_model_dir
            print(f"Using Whisper model from: {whisper_model_dir}")
        else:
            # Fall back to checking old cache location
            whisper_cache_old = os.path.expanduser("~/.cache/whisper")
            model_file = f"{model_name}.pt"
            old_model_path = os.path.join(whisper_cache_old, model_file)
            if os.path.exists(old_model_path):
                print(f"Using Whisper model from cache: {whisper_cache_old}")
            else:
                raise FileNotFoundError(
                    f"Whisper model '{model_name}' is not downloaded. "
                    f"Please download it first from the Model Manager (Settings → Model Manager)."
                )

        # Load the model (model_name can include .en suffix for English-only variants)
        model = whisper.load_model(model_name, download_root=download_root)

        if device == "cuda":
            model = model.cuda()

        return model, None, "whisper"

    @staticmethod
    def _load_faster_whisper(whisper_config, use_gpu=True):
        """Load faster-whisper model (CTranslate2-based, 4-10x faster)"""
        # Must happen BEFORE importing faster_whisper/ctranslate2
        _preload_cudnn(use_gpu)

        from faster_whisper import WhisperModel

        model_name = whisper_config.get("model", "small")
        compute_type = whisper_config.get("compute_type", "auto")

        print(f"Loading faster-whisper model: {model_name}")

        # Auto-detect best compute type based on hardware
        if compute_type == "auto":
            if use_gpu and torch.cuda.is_available():
                # Check GPU compute capability for float16 support
                gpu_props = torch.cuda.get_device_properties(0)
                if gpu_props.major >= 7:
                    compute_type = "float16"
                else:
                    compute_type = "float32"
            else:
                compute_type = "int8"  # CPU optimized

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Check for local model first
        models_dir = MODELS_DIR
        local_model_path = os.path.join(models_dir, f"faster-whisper-{model_name}")

        if os.path.exists(local_model_path):
            model_path = local_model_path
            print(f"Using faster-whisper model from: {local_model_path}")
        else:
            raise FileNotFoundError(
                f"Faster-whisper model '{model_name}' is not downloaded. "
                f"Please download it first from the Model Manager (Settings → Model Manager)."
            )

        print(f"Device: {device}, Compute type: {compute_type}")

        model = WhisperModel(model_path, device=device, compute_type=compute_type)

        return model, None, "faster_whisper"

    @staticmethod
    def _load_huggingface(hf_config, device):
        """Load Hugging Face transformers model"""
        _lazy_import_ml_libraries()

        model_id = hf_config.get("model_id", "openai/whisper-tiny")
        use_flash_attention = hf_config.get("use_flash_attention", False)

        # Check if model exists locally in ./models directory
        models_dir = MODELS_DIR
        model_dir_name = model_id.replace("/", "--")
        local_model_path = os.path.join(models_dir, model_dir_name)

        # Use local path if it exists, otherwise tell user to download first
        if os.path.exists(local_model_path):
            model_path = local_model_path
            print(f"Loading Hugging Face model from local path: {local_model_path}")
        else:
            raise FileNotFoundError(
                f"HuggingFace model '{model_id}' is not downloaded. "
                f"Please download it first from the Model Manager (Settings → Model Manager)."
            )

        try:
            # Determine model architecture from model card
            info = model_info(model_id)
            pipeline_tag = info.pipeline_tag

            # Load based on architecture
            if (
                "whisper" in model_id.lower()
                or pipeline_tag == "automatic-speech-recognition"
            ):
                # Whisper-based models (including Distil-Whisper)
                torch_dtype = torch.float16 if device == "cuda" else torch.float32

                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                }

                if use_flash_attention and device == "cuda":
                    model_kwargs["attn_implementation"] = "flash_attention_2"

                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path, **model_kwargs
                )
                model.to(device)

                processor = AutoProcessor.from_pretrained(model_path)

                # Create pipeline for easier inference
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16,
                    torch_dtype=torch_dtype,
                    device=device,
                )

                return pipe, processor, "huggingface_whisper"

            elif "wav2vec2" in model_id.lower():
                # Wav2Vec2 models
                model = AutoModelForCTC.from_pretrained(model_path)
                model.to(device)
                processor = Wav2Vec2Processor.from_pretrained(model_path)

                return model, processor, "huggingface_wav2vec2"

            else:
                # Generic ASR model
                pipe = pipeline(
                    "automatic-speech-recognition", model=model_path, device=device
                )
                return pipe, None, "huggingface_generic"

        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            raise

    @staticmethod
    def _load_custom(custom_config, device):
        """Load custom model from local path"""
        _lazy_import_ml_libraries()

        model_path = custom_config.get("model_path", "")
        model_type = custom_config.get("model_type", "whisper")

        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Custom model path not found: {model_path}")

        print(f"Loading custom {model_type} model from: {model_path}")

        if model_type == "whisper":
            # If model_path is a directory (e.g., ./models/whisper-base),
            # check if it contains a .pt file
            if os.path.isdir(model_path):
                pt_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
                if pt_files:
                    # Use the directory as download_root and extract model name
                    model_file = pt_files[0]
                    model_name = model_file.replace(".pt", "").replace(".en", "")
                    model = whisper.load_model(model_name, download_root=model_path)
                else:
                    raise ValueError(f"No .pt files found in directory: {model_path}")
            else:
                # model_path points directly to a .pt file
                model = whisper.load_model(model_path)

            if device == "cuda":
                model = model.cuda()
            return model, None, "whisper"
        else:
            # Try loading as Hugging Face model
            pipe = pipeline(
                "automatic-speech-recognition", model=model_path, device=device
            )
            return pipe, None, "huggingface_generic"

    @staticmethod
    def transcribe(model, processor, model_type, audio_data, language="auto", whisper_params=None, return_segments=False):
        """
        Transcribe audio using the loaded model

        Args:
            model: The loaded model
            processor: The processor/tokenizer (if applicable)
            model_type: Type of model ('whisper', 'huggingface_whisper', etc.)
            audio_data: Audio data as numpy array
            language: Language code (default: 'en', 'auto' for auto-detection)
            whisper_params: Dict of Whisper decoding parameters (optional)
                          For Whisper models: beam_size, temperature, condition_on_previous_text, etc.
                          See LIVE_TRANSCRIPTION_PARAMS and FILE_TRANSCRIPTION_PARAMS constants
            return_segments: If True, return list of segments with timestamps instead of just text

        Returns:
            If return_segments=False: Transcription text (str)
            If return_segments=True: List of segment dicts with 'text', 'start', 'end' keys
        """
        try:
            if model_type == "whisper":
                # Original Whisper model (OpenAI whisper)
                # Build params dict with language and whisper_params
                params = {}
                if language != "auto":
                    params["language"] = language

                # OpenAI whisper supported parameters (transcribe-level + DecodingOptions)
                whisper_transcribe_params = {
                    "verbose", "temperature", "compression_ratio_threshold",
                    "logprob_threshold", "no_speech_threshold",
                    "condition_on_previous_text", "initial_prompt",
                    "word_timestamps", "prepend_punctuations", "append_punctuations",
                    "clip_timestamps", "hallucination_silence_threshold",
                    "carry_initial_prompt",
                    # DecodingOptions params
                    "task", "language", "sample_len", "best_of", "beam_size",
                    "patience", "length_penalty", "prefix", "suppress_tokens",
                    "suppress_blank", "without_timestamps", "max_initial_timestamp",
                    "fp16",
                }

                if whisper_params:
                    for k, v in whisper_params.items():
                        if k.startswith("_"):
                            continue
                        if k in whisper_transcribe_params:
                            params[k] = v

                result = model.transcribe(audio_data, **params)

                if return_segments:
                    # Return Whisper's native segments with timestamps
                    segments = result.get("segments", [])
                    return [{"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]} for seg in segments if seg["text"].strip()]
                return result["text"].strip()

            elif model_type == "faster_whisper":
                # faster-whisper model (CTranslate2-based)
                # Build params dict with language and whisper_params
                params = {
                    "vad_filter": False,  # External VAD already screens; internal VAD over-chunks long buffers
                }
                if language != "auto":
                    params["language"] = language

                # faster-whisper supported parameters (different from standard whisper)
                # NOTE: initial_prompt and hotwords are intentionally excluded.
                # Both are tokenized into the decoder prefix and consume decoder positions
                # (max 448 total). The full Bible-book hotwords list is ~227 tokens alone,
                # leaving only ~220 positions for actual transcription — not enough for
                # dense speech. Whisper's language setting is sufficient for recognition.
                faster_whisper_params = {
                    "beam_size", "best_of", "patience", "length_penalty",
                    "repetition_penalty", "no_repeat_ngram_size",
                    "temperature", "compression_ratio_threshold",
                    "log_prob_threshold", "no_speech_threshold",
                    "condition_on_previous_text",
                    "prefix", "suppress_blank", "suppress_tokens",
                    "without_timestamps", "max_initial_timestamp",
                    "word_timestamps", "prepend_punctuations",
                    "append_punctuations", "vad_filter", "vad_parameters",
                    "task",
                }

                # Parameter name mapping (whisper -> faster-whisper)
                param_mapping = {
                    "logprob_threshold": "log_prob_threshold",  # Different naming
                }

                if whisper_params:
                    for k, v in whisper_params.items():
                        if k.startswith("_"):
                            continue
                        # Map parameter name if needed
                        mapped_key = param_mapping.get(k, k)
                        # Only include if faster-whisper supports it
                        if mapped_key in faster_whisper_params:
                            params[mapped_key] = v

                # faster-whisper returns (segments_iterator, info)
                segments_iter, info = model.transcribe(audio_data, **params)

                # Convert iterator to list with standard format
                segments = []
                full_text = []
                for seg in segments_iter:
                    text = seg.text.strip()
                    if text:
                        seg_dict = {
                            "text": text,
                            "start": seg.start,
                            "end": seg.end,
                            "no_speech_prob": getattr(seg, "no_speech_prob", 0),
                            "avg_logprob": getattr(seg, "avg_logprob", 0),
                            # Detected source language (ISO code) so rows can record source_language
                            # even when audio.language is 'auto'. Same for every seg in the chunk.
                            "language": getattr(info, "language", None),
                        }
                        # Extract word-level confidence if available
                        if hasattr(seg, 'words') and seg.words:
                            seg_dict["words"] = [
                                {
                                    "word": w.word,
                                    "probability": getattr(w, "probability", None),
                                    "start": w.start,
                                    "end": w.end,
                                }
                                for w in seg.words
                            ]
                        segments.append(seg_dict)
                        full_text.append(text)

                if return_segments:
                    return segments
                return " ".join(full_text)

            elif model_type == "huggingface_whisper":
                # Hugging Face Whisper pipeline
                generate_kwargs = {}
                if language != "auto":
                    generate_kwargs["language"] = language

                # Map Whisper params to HuggingFace generate_kwargs
                if whisper_params:
                    if "beam_size" in whisper_params:
                        generate_kwargs["num_beams"] = whisper_params["beam_size"]
                    if "temperature" in whisper_params:
                        _hf_temp = whisper_params["temperature"]
                        if isinstance(_hf_temp, (list, tuple)):
                            # HF supports a fallback tuple in long-form generation
                            generate_kwargs["temperature"] = tuple(float(t) for t in _hf_temp)
                        elif isinstance(_hf_temp, (int, float)):
                            generate_kwargs["temperature"] = _hf_temp
                    # Quality thresholds: honored by HF Whisper long-form generation;
                    # stripped via the retry below when the installed transformers
                    # version doesn't accept them.
                    if "compression_ratio_threshold" in whisper_params:
                        generate_kwargs["compression_ratio_threshold"] = whisper_params["compression_ratio_threshold"]
                    if "logprob_threshold" in whisper_params:
                        generate_kwargs["logprob_threshold"] = whisper_params["logprob_threshold"]
                    if "no_speech_threshold" in whisper_params:
                        generate_kwargs["no_speech_threshold"] = whisper_params["no_speech_threshold"]
                    if "condition_on_previous_text" in whisper_params:
                        # HF names this condition_on_prev_tokens
                        generate_kwargs["condition_on_prev_tokens"] = whisper_params["condition_on_previous_text"]

                _hf_optional_keys = (
                    "compression_ratio_threshold", "logprob_threshold",
                    "no_speech_threshold", "condition_on_prev_tokens",
                )

                def _hf_pipeline_call(**call_kwargs):
                    try:
                        return model(audio_data, generate_kwargs=generate_kwargs, **call_kwargs) if generate_kwargs else model(audio_data, **call_kwargs)
                    except (TypeError, ValueError) as hf_err:
                        stripped = {k: v for k, v in generate_kwargs.items() if k not in _hf_optional_keys}
                        if len(stripped) == len(generate_kwargs):
                            raise
                        print(f"[WARNING] HF pipeline rejected quality thresholds ({hf_err}); retrying without them")
                        return model(audio_data, generate_kwargs=stripped, **call_kwargs) if stripped else model(audio_data, **call_kwargs)

                if return_segments:
                    # Request timestamps from HuggingFace pipeline
                    result = _hf_pipeline_call(return_timestamps=True)
                    chunks = result.get("chunks", [])
                    segments = []
                    for chunk in chunks:
                        text = chunk.get("text", "").strip()
                        timestamp = chunk.get("timestamp", (0, 0))
                        if text and timestamp:
                            start = timestamp[0] if timestamp[0] is not None else 0
                            end = timestamp[1] if timestamp[1] is not None else start
                            segments.append({"text": text, "start": start, "end": end})
                    return segments

                result = _hf_pipeline_call()
                return result["text"].strip()

            elif model_type == "huggingface_wav2vec2":
                # Wav2Vec2 model (doesn't support language parameter or timestamps)
                import torch

                inputs = processor(
                    audio_data, sampling_rate=16000, return_tensors="pt", padding=True
                )

                with torch.no_grad():
                    logits = model(inputs.input_values.to(model.device)).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                text = transcription[0].strip()

                if return_segments:
                    # Wav2Vec2 doesn't provide timestamps, return as single segment
                    return [{"text": text, "start": 0, "end": 0}] if text else []
                return text

            elif model_type == "huggingface_generic":
                # Generic Hugging Face pipeline
                # Try to pass language if supported, otherwise just use audio
                try:
                    if return_segments:
                        if language == "auto":
                            result = model(audio_data, return_timestamps=True)
                        else:
                            result = model(audio_data, return_timestamps=True, generate_kwargs={"language": language})
                        chunks = result.get("chunks", [])
                        segments = []
                        for chunk in chunks:
                            text = chunk.get("text", "").strip()
                            timestamp = chunk.get("timestamp", (0, 0))
                            if text and timestamp:
                                start = timestamp[0] if timestamp[0] is not None else 0
                                end = timestamp[1] if timestamp[1] is not None else start
                                segments.append({"text": text, "start": start, "end": end})
                        return segments

                    if language == "auto":
                        result = model(audio_data)
                    else:
                        result = model(
                            audio_data, generate_kwargs={"language": language}
                        )
                except (TypeError, ValueError, RuntimeError) as e:
                    # Fallback if language parameter not supported
                    print(f"[WARNING] Model language parameter failed ({e}), falling back to auto-detect")
                    result = model(audio_data)
                    if return_segments:
                        text = result["text"].strip()
                        return [{"text": text, "start": 0, "end": 0}] if text else []
                return result["text"].strip()

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            print(f"Transcription error: {e}")
            return [] if return_segments else ""


class WhisperLiveTranscriber:
    """
    Streaming transcription using whisper-live approach.

    Uses a rolling numpy buffer instead of dual confirmed/active buffers.
    Segments are finalized when the same output is repeated N times (same_output_threshold).

    Ported from: Whisper-Live-main/whisper_live/backend/base.py
    """

    RATE = 16000  # Sample rate in Hz

    def __init__(
        self,
        sample_rate=16000,
        same_output_threshold=7,
        no_speech_thresh=0.45,
        send_last_n_segments=10,
    ):
        """
        Initialize the transcriber.

        Args:
            sample_rate: Audio sample rate (default 16000)
            same_output_threshold: Number of repeated outputs before finalizing (default 7)
            no_speech_thresh: Threshold for filtering no-speech segments (default 0.45)
            send_last_n_segments: Number of recent segments to keep (default 10)
        """
        self.RATE = sample_rate
        self.same_output_threshold = same_output_threshold
        self.no_speech_thresh = no_speech_thresh
        self.send_last_n_segments = send_last_n_segments

        # Frame buffer (numpy array of float32 audio samples)
        self.frames_np = None
        self.frames_offset = 0.0  # Time offset when buffer was clipped
        self.timestamp_offset = 0.0  # Current transcription position

        # Segment tracking
        self.transcript = []  # List of completed segments
        self.current_out = ""  # Current incomplete output
        self.prev_out = ""  # Previous output for comparison
        self.same_output_count = 0
        self.end_time_for_same_output = None
        self._last_seg_confidence = {}  # Word-level confidence from last segment

        # Threading lock for buffer access
        self.lock = threading.Lock()

    def _is_similar_output(self, text1, text2, threshold=0.85):
        """
        Check if two outputs are similar enough to count as 'same'.

        Uses fuzzy matching because Whisper often returns slightly different text
        each iteration (e.g., "I'm going to" vs "I'm gonna").

        Args:
            text1: First text to compare
            text2: Second text to compare
            threshold: Similarity threshold (0.0-1.0), default 0.85

        Returns:
            bool: True if texts are similar enough
        """
        if not text1 or not text2:
            return False
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
        return ratio >= threshold

    def add_frames(self, audio_bytes, sample_width=2):
        """
        Add audio frames to the buffer.

        Converts bytes to float32 numpy array and appends to rolling buffer.
        If buffer exceeds 45 seconds, clips oldest 30 seconds.

        Args:
            audio_bytes: Raw audio bytes (int16 PCM)
            sample_width: Bytes per sample (default 2 for int16)
        """
        # Convert bytes to float32 numpy array normalized to [-1, 1]
        frame_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        with self.lock:
            # Clip buffer if it exceeds 45 seconds
            if self.frames_np is not None and self.frames_np.shape[0] > 45 * self.RATE:
                self.frames_offset += 30.0
                self.frames_np = self.frames_np[int(30 * self.RATE):]
                # Ensure timestamp_offset doesn't fall behind frames_offset
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset

            # Append frames
            if self.frames_np is None:
                self.frames_np = frame_np.copy()
            else:
                self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

    def get_audio_chunk_for_processing(self):
        """
        Get the next audio chunk to transcribe.

        Returns audio from timestamp_offset to end of buffer.

        Returns:
            tuple: (audio_np, duration) - audio as float32 numpy array and duration in seconds
        """
        with self.lock:
            if self.frames_np is None:
                return np.array([], dtype=np.float32), 0.0

            samples_to_skip = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            audio_chunk = self.frames_np[samples_to_skip:].copy()

        duration = audio_chunk.shape[0] / self.RATE if audio_chunk.shape[0] > 0 else 0.0
        return audio_chunk, duration

    def get_buffer_duration(self):
        """Get total duration of audio in buffer in seconds."""
        with self.lock:
            if self.frames_np is None:
                return 0.0
            return self.frames_np.shape[0] / self.RATE

    def update_segments(self, segments, duration):
        """
        Process segments from Whisper transcription using Whisper-Live's approach.

        Key insight: Whisper returns segments with timestamps. We finalize all
        segments except the last one immediately. The last segment stays as
        "in-progress" and only finalizes when it repeats (same_output_threshold).

        This prevents overlapping text because we use Whisper's segment boundaries
        instead of guessing where phrases end.

        Args:
            segments: List of dicts with 'text', 'start', 'end' keys
            duration: Duration of the audio chunk that was transcribed

        Returns:
            dict: {
                'completed_segments': list of newly completed segments,
                'current_text': current incomplete text (last segment),
                'is_finalized': whether last segment was just finalized
            }
        """
        result = {
            'completed_segments': [],
            'current_text': '',
            'is_finalized': False
        }

        if not segments:
            return result

        # FIX: Detect garbage output from Whisper (overwhelmed by too much audio)
        # When Whisper gets 30+ seconds, it often returns just '...' or empty text
        all_text = ' '.join(seg.get('text', '').strip() for seg in segments)
        if len(all_text) < 5 or all_text == '...' or all_text.strip() == '':
            # Whisper is overwhelmed - force buffer trim and skip garbage
            with self.lock:
                if self.frames_np is not None:
                    buffer_duration = self.frames_np.shape[0] / self.RATE
                    current_pos = self.timestamp_offset - self.frames_offset
                    chunk_to_process = buffer_duration - current_pos
                    if chunk_to_process > 10:
                        # Force advance to keep only 10 seconds
                        extra_advance = chunk_to_process - 10
                        self.timestamp_offset += extra_advance
            return result  # Skip processing garbage segments

        offset = None

        # Process all segments except the last one (finalize them immediately)
        # This is the key difference from our previous approach
        if len(segments) > 1:
            for seg in segments[:-1]:
                text = seg.get('text', '').strip()
                if not text:
                    continue

                start = self.timestamp_offset + seg.get('start', 0)
                end = self.timestamp_offset + min(duration, seg.get('end', duration))

                if start >= end:
                    continue

                completed = {
                    'start': start,
                    'end': end,
                    'text': text,
                    'completed': True,
                }
                # Pass through word-level confidence data if available
                if 'words' in seg:
                    completed['words'] = seg['words']
                if 'avg_logprob' in seg:
                    completed['avg_logprob'] = seg['avg_logprob']
                if 'no_speech_prob' in seg:
                    completed['no_speech_prob'] = seg['no_speech_prob']
                if seg.get('language'):
                    completed['language'] = seg['language']
                self.transcript.append(completed)
                result['completed_segments'].append(completed)
                # print(f"[SEGMENT] Finalized: '{text[:50]}...' ({seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s)" if len(text) > 50 else f"[SEGMENT] Finalized: '{text}'", flush=True)
                offset = min(duration, seg.get('end', duration))

        # Handle the last segment (in-progress until repeated)
        last_seg = segments[-1]
        self.current_out = last_seg.get('text', '').strip()
        # Store last segment's confidence data for finalization
        self._last_seg_confidence = {
            k: last_seg[k] for k in ('words', 'avg_logprob', 'no_speech_prob', 'language') if k in last_seg
        }
        result['current_text'] = self.current_out

        # Check if last segment is repeating (same_output_threshold logic)
        if self._is_similar_output(self.current_out, self.prev_out) and self.current_out:
            self.same_output_count += 1
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = last_seg.get('end', duration)

            # Debug logging for same_output tracking
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # Finalize last segment if repeated enough times
        if self.same_output_count >= self.same_output_threshold:
            if self.current_out:
                completed = {
                    'start': self.timestamp_offset,
                    'end': self.timestamp_offset + min(duration, self.end_time_for_same_output or duration),
                    'text': self.current_out,
                    'completed': True,
                }
                # Pass through word-level confidence from last segment
                if 'words' in last_seg:
                    completed['words'] = last_seg['words']
                if 'avg_logprob' in last_seg:
                    completed['avg_logprob'] = last_seg['avg_logprob']
                if 'no_speech_prob' in last_seg:
                    completed['no_speech_prob'] = last_seg['no_speech_prob']
                if last_seg.get('language'):
                    completed['language'] = last_seg['language']
                self.transcript.append(completed)
                result['completed_segments'].append(completed)
                result['is_finalized'] = True
                # FIX: Save the text that was just finalized so phrase_complete knows not to re-process
                result['just_finalized_text'] = self.current_out
                print(f"[SAME_OUTPUT] Finalized last segment: '{self.current_out[:50]}...'" if len(self.current_out) > 50 else f"[SAME_OUTPUT] Finalized: '{self.current_out}'", flush=True)
                offset = min(duration, self.end_time_for_same_output or duration)

            # Reset
            self.current_out = ''
            self.prev_out = ''
            self.same_output_count = 0
            self.end_time_for_same_output = None
            result['current_text'] = ''
        else:
            self.prev_out = self.current_out

        # Advance timestamp_offset by the end of finalized segments
        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        # PROACTIVE FIX: ALWAYS check buffer size and limit it, not just when offset is None
        # This prevents the buffer from slowly growing over time even when segments are finalizing
        with self.lock:
            if self.frames_np is not None:
                buffer_duration = self.frames_np.shape[0] / self.RATE
                current_pos = self.timestamp_offset - self.frames_offset
                chunk_to_process = buffer_duration - current_pos
                # If chunk exceeds 20 seconds, force advance to keep ~15 seconds
                # This is more aggressive than before to prevent Whisper from being overwhelmed
                if chunk_to_process > 20:
                    extra_advance = chunk_to_process - 15  # Keep 15 seconds
                    self.timestamp_offset += extra_advance

        return result

    def force_finalize(self):
        """
        Force finalize current text (e.g., on phrase timeout / silence detection).

        Returns:
            dict: Segment if there was text to finalize, None otherwise
        """
        if not self.current_out:
            return None

        # Get current duration from buffer
        _, duration = self.get_audio_chunk_for_processing()

        segment = {
            'start': self.timestamp_offset,
            'end': self.timestamp_offset + duration,
            'text': self.current_out,
            'completed': True,
        }
        # Include stored confidence data from last segment
        if hasattr(self, '_last_seg_confidence') and self._last_seg_confidence:
            segment.update(self._last_seg_confidence)
        self.transcript.append(segment)

        # Update timestamp offset
        with self.lock:
            self.timestamp_offset += duration

        # Reset
        self.current_out = ""
        self.prev_out = ""
        self.same_output_count = 0
        self.end_time_for_same_output = None

        return segment

    def get_recent_segments(self):
        """Get the most recent completed segments."""
        return self.transcript[-self.send_last_n_segments:] if self.transcript else []

    def get_all_text(self):
        """Get all transcribed text concatenated."""
        return " ".join(seg['text'] for seg in self.transcript if seg.get('text'))

    def reset(self):
        """Reset transcriber state for new session."""
        with self.lock:
            self.frames_np = None
            self.frames_offset = 0.0
            self.timestamp_offset = 0.0
        self.transcript = []
        self.current_out = ""
        self.prev_out = ""
        self.same_output_count = 0
        self.end_time_for_same_output = None


# Use 'spawn' everywhere, not just where it is the platform default.
#
# The parent process loads the live-translation model, and when that runs on a local
# GPU it initialises a CUDA context. A *forked* child inherits that context, and CUDA
# forbids re-initialising it — so the transcription worker dies the moment it touches
# the GPU:
#
#     RuntimeError: Cannot re-initialize CUDA in forked subprocess.
#                   To use CUDA with multiprocessing, you must use the 'spawn' start method
#
# That makes "Restart Transcription" impossible on Linux+CUDA once translation has
# loaded, which is exactly the configuration that runs translation on the same box.
# macOS has always defaulted to spawn and this module is written for that path (see
# the note below about the child re-importing and receiving its state as arguments),
# so forcing it on Linux aligns the two rather than introducing an untried mode.
#
# Must happen before the Manager and Queues below: they inherit the active context,
# and mixing contexts is unsupported.
if multiprocessing.current_process().name == "MainProcess":
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
            print("[INIT] multiprocessing start method set to 'spawn' (CUDA-safe)")
    except RuntimeError as e:
        # Already started something; leave the existing method rather than crash.
        print(f"[INIT] WARNING: could not set spawn start method: {e}")

# Create shared state only in the main process.
# On macOS, 'spawn' is the default start method (safe — avoids ObjC/fork crashes after
# PyTorch/Whisper initialize the Objective-C runtime with background threads).
# On Linux, 'fork' was the default; the block above now forces spawn there too.
# With spawn (macOS), the child re-imports this module and must NOT recreate the Manager
# (it would fail before bootstrap completes). Instead, the child receives these objects
# as pickled arguments to thread1_function and assigns them to module globals there.
if multiprocessing.current_process().name == 'MainProcess':
    mp_manager = multiprocessing.Manager()

    # Create multiprocessing Queue for config updates (hot-reload)
    config_queue = MPQueue()

    # Create multiprocessing Queue for control commands (start/stop)
    control_queue = MPQueue()

    # Create multiprocessing Queue for streaming audio to web clients
    audio_stream_queue = MPQueue(maxsize=10)

    # Global transcription state - use Manager.dict() for cross-process sharing
    transcription_state = mp_manager.dict(
        {
            "running": False,
            "status": "stopped",
            "message": "Transcription not started",
            "error": None,  # Error message if status == "error"
            "db_name": None,  # Shared database name for cross-process access
            "session_id": None,  # Stable per-session id (.db filename stem); cross-process
            "audio_level": 0,  # Audio level for histogram (0-100)
            "audio_db": -60,  # Audio level in decibels
            "audio_energy": 0,  # Raw audio energy (RMS)
            "start_time": 0,  # epoch seconds when transcription became active; 0 = not running
            "live_text": "",  # Live preview text (not yet saved to DB)
            "live_start": 0,  # Start time of the live preview within the session
            "live_end": 0,  # End time of the live preview within the session
            "live_word_confidences": [],  # Word-level confidence for the live preview
            "loaded_model": "",  # Name of the actual model that was loaded
            "audio_stream_enabled": False,  # Whether to stream audio to web clients
            "audio_type": None,  # "Speaking", "Music", or "Quiet" — PANNs detection (no_speech_prob fallback)
            "detection_mode": None,  # "panns" (tagger live) or "energy" (fallback) — which detector is actually running
            "loaded_model_device": None,  # "cuda" / "mps" / "cpu" the ASR model landed on
            "model_load_ms": None,  # How long the ASR model took to load (ms)
            "infer_ms_ema": None,  # EMA of per-chunk transcribe time (ms) — health dashboard
            "rtf_ema": None,  # EMA real-time factor (transcribe_s / audio_s); <1 keeps up
            "segments_total": 0,  # Chunks transcribed this session (throughput numerator)
            "segments_per_min": None,  # Throughput over the session window
            "rows_saved": 0,  # Finalized transcript lines saved to the session DB
            "queue_depth": None,  # audio_stream_queue depth, when readable
        }
    )

    # Shared calibration state for cross-process communication
    calibration_state = mp_manager.dict(
        {
            "active": False,
            "step": 1,  # 1 = noise floor calibration, 2 = speech calibration
            "step1_complete": False,
            "start_time": 0,
            "duration": 15,  # 15 seconds per step (30 total)
            "speech_samples": 0,
            "noise_samples": 0,
            "silence_samples": 0,
        }
    )

    # Shared calibration data storage (Manager lists for cross-process)
    calibration_data_shared = mp_manager.dict(
        {
            "speech_samples": mp_manager.list(),
            "noise_samples": mp_manager.list(),
            "silence_durations": mp_manager.list(),
            "energy_levels": mp_manager.list(),
            "vad_probabilities": mp_manager.list(),
        }
    )

    # Step 1 calibration data (noise floor only)
    calibration_step1_data = mp_manager.dict(
        {
            "noise_energies": mp_manager.list(),
            "avg_noise": 0.0,
            "max_noise": 0.0,
        }
    )
else:
    # Spawned worker process: shared objects will be received as function arguments
    # and assigned to these globals at the top of thread1_function.
    mp_manager = None
    config_queue = None
    control_queue = None
    transcription_state = None
    calibration_state = None
    calibration_data_shared = None
    calibration_step1_data = None

# Global reference to transcription process for restart functionality
transcription_process = None

# Set once the server begins shutting down / restarting. The transcription_state
# Manager proxy dies when the Manager process is torn down (execv restart, signal,
# auto-update), after which any proxy access raises BrokenPipeError/EOFError/
# ConnectionError. Background emit threads check this and exit cleanly instead of
# crashing with an unhandled error.
_server_shutting_down = threading.Event()


# Every way a dead Manager proxy can fail: broken pipe/EOF mid-call,
# ConnectionRefusedError to the AF_UNIX listener, FileNotFoundError when the
# socket file is gone, AttributeError/TypeError when the proxy itself is None.
_TS_PROXY_ERRORS = (BrokenPipeError, EOFError, ConnectionError, FileNotFoundError, AttributeError, TypeError)


def _ts_get(key, default=None):
    """Read from the transcription_state Manager proxy, tolerating a torn-down
    Manager during shutdown. On a proxy disconnect, flag shutdown and return the
    default instead of raising an unhandled BrokenPipeError."""
    try:
        return transcription_state.get(key, default)
    except _TS_PROXY_ERRORS:
        _server_shutting_down.set()
        return default


def _ts_snapshot():
    """dict(transcription_state), tolerating a torn-down Manager during a
    shutdown/restart (e.g. the auto-update window). Returns a benign
    'restarting' state on proxy disconnect instead of raising an unhandled 500
    from the status routes."""
    try:
        return dict(transcription_state)
    except _TS_PROXY_ERRORS:
        _server_shutting_down.set()
        return {"running": False, "status": "restarting", "message": "Server is restarting"}

# Database cache for performance
_db_cache = {
    "last_entries": [],
    "last_fetch_time": 0,
    "cache_duration": 1.0,  # Cache for 1 second
}

# Thread locks for synchronization
_db_lock = threading.Lock()
_cache_lock = threading.Lock()
_transcription_state_lock = threading.Lock()
_transcription_start_lock = threading.Lock()  # Guards worker-process creation in the start route
_audio_queue_lock = threading.Lock()
# Generate the current date and time as a string
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M_%A")
current_year = datetime.now().strftime("%Y")
current_month = datetime.now().strftime("%Y-%m")

# Database will be created lazily when transcription starts
db_name = None  # Will be set when database is initialized
db_initialized = False
live_session_id = None  # Stable per-session id (the .db filename stem); set in initialize_database


# ============== PANNs audio tagger (music / speech detection) ==============
# Default checkpoint location (kept under APP_DIR so compiled builds stay self-contained)
PANNS_CHECKPOINT = os.path.join(APP_DIR, "panns_data", "Cnn14_mAP=0.431.pth")
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
# AudioSet class labels. panns_inference loads these at IMPORT TIME from a hardcoded
# ~/panns_data/class_labels_indices.csv and (over plain http) tries to wget them if
# absent. When that fetch is blocked/offline it leaves an empty file, which is never
# retried -> labels=[] -> classes_num=0 -> the 527-class CNN14 checkpoint fails to load
# -> music detection silently falls back to energy-only ("Speaking"/"Quiet", never
# "Music"). We ship the CSV under APP_DIR and place a valid copy before importing the
# library (with an https fallback) so detection works self-contained / offline.
PANNS_LABELS_FILENAME = "class_labels_indices.csv"
PANNS_LABELS_BUNDLED = os.path.join(APP_DIR, "panns_data", PANNS_LABELS_FILENAME)
PANNS_LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
_PANNS_LABELS_MIN_BYTES = 1024  # a valid 527-row CSV is ~14 KB; smaller == missing/poisoned
_panns_labels_ready = False  # only need to repair the on-disk CSV once per process
_audio_tagger = None
_audio_tagger_failed_key = None  # (device, ckpt) that hit a real load error — don't retry it
_audio_tagger_key = None  # (device, ckpt) currently loaded
_panns_label_idx = None  # (music_idx_list, speech_idx_list)
_panns_missing_logged = False  # avoid spamming the "checkpoint missing" log


def panns_checkpoint_path(cfg=None):
    """Resolve the CNN14 checkpoint path (config override or the default location)."""
    cfg = cfg if cfg is not None else config.get("speech_type_detection", {})
    custom = (cfg.get("checkpoint_path", "") or "").strip()
    return custom or PANNS_CHECKPOINT


def panns_labels_home_path():
    """The path panns_inference.config hardcodes for the AudioSet label CSV."""
    return os.path.join(os.path.expanduser("~"), "panns_data", PANNS_LABELS_FILENAME)


def ensure_panns_labels_csv():
    """Guarantee a valid class_labels_indices.csv exists where panns_inference expects
    it, BEFORE the library is imported (it hardcodes ~/panns_data at import time).
    All PANNs data lives in APP_DIR/panns_data; ~/panns_data is kept as a symlink
    into it so nothing real is stored outside the app folder. Best-effort: never
    raises."""
    global _panns_labels_ready
    if _panns_labels_ready:
        return
    try:
        home_csv = panns_labels_home_path()
        home_dir = os.path.dirname(home_csv)
        app_dir = os.path.dirname(PANNS_LABELS_BUNDLED)

        # The app-folder copy is the real storage — make sure it's valid first
        # (https download; the library itself only tries plain http, often blocked).
        if not (os.path.exists(PANNS_LABELS_BUNDLED) and os.path.getsize(PANNS_LABELS_BUNDLED) >= _PANNS_LABELS_MIN_BYTES):
            os.makedirs(app_dir, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(PANNS_LABELS_URL, PANNS_LABELS_BUNDLED)
            print(f"[PANNS] Downloaded AudioSet labels -> {PANNS_LABELS_BUNDLED}")
        if not (os.path.exists(PANNS_LABELS_BUNDLED) and os.path.getsize(PANNS_LABELS_BUNDLED) >= _PANNS_LABELS_MIN_BYTES):
            print("[PANNS] AudioSet labels unavailable; music detection will be unavailable")
            return

        # Point ~/panns_data at the app folder. Migrate a stale real directory
        # only when it holds nothing but the label CSV (never delete unknown data).
        if os.path.islink(home_dir):
            if os.path.realpath(home_dir) != os.path.realpath(app_dir):
                os.unlink(home_dir)
        elif os.path.isdir(home_dir):
            if all(name == PANNS_LABELS_FILENAME for name in os.listdir(home_dir)):
                shutil.rmtree(home_dir)
        if not os.path.lexists(home_dir):
            try:
                os.symlink(app_dir, home_dir)
                print(f"[PANNS] Linked {home_dir} -> {app_dir}")
            except OSError:
                pass

        # Fallback when the symlink couldn't be made (or a real dir with other
        # content remains): copy the CSV like before.
        if not (os.path.exists(home_csv) and os.path.getsize(home_csv) >= _PANNS_LABELS_MIN_BYTES):
            os.makedirs(home_dir, exist_ok=True)
            shutil.copyfile(PANNS_LABELS_BUNDLED, home_csv)
            print(f"[PANNS] Installed AudioSet labels from bundled copy -> {home_csv}")

        if os.path.exists(home_csv) and os.path.getsize(home_csv) >= _PANNS_LABELS_MIN_BYTES:
            _panns_labels_ready = True
        else:
            print("[PANNS] AudioSet labels still missing after repair attempt; music detection will be unavailable")
    except Exception as e:
        print(f"[PANNS] Could not install AudioSet labels: {e}")


def panns_package_installed():
    try:
        import importlib.util
        return importlib.util.find_spec("panns_inference") is not None
    except Exception:
        return False


def get_audio_tagger(cfg):
    """Lazy-load the PANNs CNN14 tagger for the given speech_type_detection cfg.
    Reloads if device/checkpoint changed. Returns None if unavailable (missing
    package or checkpoint). MUST be called off the audio-drain path (it can block
    for seconds on first load)."""
    global _audio_tagger, _audio_tagger_failed_key, _audio_tagger_key, _panns_label_idx, _panns_missing_logged
    # Make sure the AudioSet label CSV is valid before panns_inference is imported
    # below (it loads labels once at import time). Without this the checkpoint fails
    # to load and detection silently degrades to energy-only.
    ensure_panns_labels_csv()
    ckpt = panns_checkpoint_path(cfg)
    device = cfg.get("device", "cpu") or "cpu"
    key = (device, ckpt)
    if _audio_tagger is not None and _audio_tagger_key == key:
        return _audio_tagger
    if _audio_tagger_failed_key == key:
        return None
    # Missing checkpoint is *transient* (a download in the main process may produce
    # it later) — recheck cheaply each call instead of failing permanently.
    if not os.path.exists(ckpt):
        if not _panns_missing_logged:
            print(f"[PANNS] Checkpoint not found at {ckpt}; music detection falls back to energy-based until it's downloaded")
            _panns_missing_logged = True
        return None
    # Config changed (device/checkpoint) or first load: drop any stale model.
    _audio_tagger = None
    _audio_tagger_key = None
    try:
        from panns_inference import AudioTagging, labels
        tagger = AudioTagging(checkpoint_path=ckpt, device=device)
        music_idx = [i for i, l in enumerate(labels) if l in ("Music", "Singing")]
        _panns_label_idx = (music_idx, None)
        _audio_tagger = tagger
        _audio_tagger_key = key
        _audio_tagger_failed_key = None
        _panns_missing_logged = False
        print(f"[PANNS] Audio tagger loaded on {device} from {ckpt}")
        return _audio_tagger
    except Exception as e:
        _audio_tagger_failed_key = key
        print(f"[PANNS] Could not load audio tagger: {e}; falling back to energy-based")
        return None


def unload_audio_tagger():
    """Release the tagger (called when transcription stops/unloads to free VRAM).
    The detector reloads it lazily if detection is used again."""
    global _audio_tagger, _audio_tagger_key, _audio_tagger_failed_key, _panns_missing_logged
    _audio_tagger = None
    _audio_tagger_key = None
    _audio_tagger_failed_key = None
    _panns_missing_logged = False


def compute_music_prob(audio_np, sr, cfg):
    """Return (music_prob, dominant_tag) for an audio buffer using PANNs, or
    (None, None) when the tagger is unavailable. audio_np: float32 mono in [-1, 1]."""
    tagger = get_audio_tagger(cfg)
    if tagger is None or audio_np is None or len(audio_np) == 0:
        return None, None
    try:
        from panns_inference import labels
        wav = np.asarray(audio_np, dtype=np.float32)
        if sr != 32000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
        clipwise, _ = tagger.inference(wav[None, :])
        clip = clipwise[0]
        music_idx, _ = _panns_label_idx
        music_prob = float(max((clip[i] for i in music_idx), default=0.0))
        return music_prob, labels[int(np.argmax(clip))]
    except Exception as e:
        print(f"[PANNS] inference error: {e}")
        return None, None


# Audio-type labels and word-attribution logic live in stt/segments.py
# (importable, unit-tested); names are re-imported so call sites stay unchanged.
from stt import segments as _segments
from stt.segments import (  # noqa: F401
    attribute_words_to_sentences,
    panns_label_from_prob,
    words_json_or_none,
    words_to_session_ms,
)


def classify_audio_type(audio_db, cfg=None):
    """Energy-based fallback label (no PANNs): audible => Speaking, else Quiet.
    We never claim Music without the PANNs detector."""
    return _segments.classify_audio_type(audio_db, cfg if cfg is not None else config.get("speech_type_detection", {}))


class MusicDetector:
    """Runs PANNs inference on a dedicated daemon thread so the audio-drain loop
    never blocks. Latest-buffer / drop-stale: submit() just hands off the newest
    buffer; the thread loads the model (off the hot path), throttles, smooths, and
    writes music_prob / audio_tag / audio_type into the shared transcription state.
    Reads speech_type_detection live from process_config each iteration -> hot-reload."""

    def __init__(self):
        self.cfg_root = {}      # live process_config (refreshed on each submit)
        self.state = None       # shared transcription_state
        self._buf = None
        self._sr = 16000
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._history = []      # instance-local smoothing buffer (no cross-thread race)
        self._last_ts = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="panns-detector")
        self._thread.start()

    def submit(self, process_config, state, audio_np, sr):
        """Non-blocking hand-off of the latest raw (pre-VAD) buffer."""
        self.cfg_root = process_config
        self.state = state
        with self._lock:
            self._buf = audio_np
            self._sr = sr
        self._event.set()

    def _run(self):
        while True:
            self._event.wait(timeout=1.0)
            self._event.clear()
            cfg = (self.cfg_root or {}).get("speech_type_detection", {})
            if not cfg.get("enabled", True) or cfg.get("method", "panns") != "panns":
                self._history.clear()
                # Clear stale PANNs state so the live monitor doesn't keep showing
                # the last music label after detection is disabled mid-session.
                st = self.state
                if st is not None:
                    st["detection_mode"] = "energy"
                    if st.get("music_prob") is not None:
                        st["music_prob"] = None
                        st["audio_tag"] = None
                        st["audio_type"] = None
                continue
            now = time.time()
            if now - self._last_ts < 0.4:  # throttle to ~2-3 runs/sec
                continue
            with self._lock:
                buf = self._buf
                sr = self._sr
                self._buf = None  # consume: don't re-run on a stale buffer once audio stops
            if buf is None:
                continue
            self._last_ts = now
            try:
                music_prob, tag = compute_music_prob(buf, sr, cfg)
            except Exception as e:
                print(f"[PANNS] detector error: {e}")
                continue
            if music_prob is None:
                # PANNs enabled but the tagger is unavailable (missing/failed load):
                # finalized_audio_type falls back to the energy-based label.
                st = self.state
                if st is not None:
                    st["detection_mode"] = "energy"
                continue
            window = max(1, int(cfg.get("smoothing_window", 4) or 1))
            self._history.append(float(music_prob))
            del self._history[:-window]
            smoothed = sum(self._history) / len(self._history)
            st = self.state
            if st is not None:
                st["detection_mode"] = "panns"
                st["music_prob"] = music_prob
                st["audio_tag"] = tag
                # Live TYPE for the monitor, even when this audio isn't transcribed.
                st["audio_type"] = panns_label_from_prob(smoothed, st.get("audio_db"), cfg)


_music_detector = None


def submit_music_detection(process_config, state, audio_np, sr):
    """Hand the latest pre-VAD buffer to the background detector (non-blocking).
    Creates/starts the detector thread on first use."""
    global _music_detector
    if _music_detector is None:
        _music_detector = MusicDetector()
    _music_detector.submit(process_config, state, audio_np, sr)


def finalized_audio_type(process_config, state):
    """Label for a finalized (transcribed) segment: the detector's live PANNs
    label when active, else the energy-based fallback."""
    cfg = process_config.get("speech_type_detection", {})
    if (cfg.get("enabled", True) and cfg.get("method", "panns") == "panns"
            and state.get("music_prob") is not None):
        return state.get("audio_type") or "Speaking"
    return classify_audio_type(state.get("audio_db"), cfg)


def _set_asr_row_stamp(db_connection, label):
    """Make new rows carry ``label`` as asr_model, or nothing when it is None.

    A trigger rather than an addition to the INSERT statements: there are nineteen
    of those with nineteen different column lists — the segment batch, the phrase
    timeout, the stop flush, and a denied variant of each — and an approach that has
    to touch all nineteen is one that silently misses the twentieth when it is
    added. A trigger cannot be bypassed by a writer that does not know about it.

    Called with None at session start, so rows stay NULL while the session's own
    model is transcribing and the value lives once in session_meta. Called with a
    label after a hot reload, so every row from that point says what changed. The
    boundary between the two is then visible in the rows themselves.
    """
    cursor = db_connection.cursor()
    cursor.execute("DROP TRIGGER IF EXISTS stamp_asr_model")
    if label:
        cursor.execute(
            "CREATE TRIGGER stamp_asr_model AFTER INSERT ON transcriptions"
            " WHEN NEW.asr_model IS NULL"
            " BEGIN UPDATE transcriptions SET asr_model = '%s' WHERE id = NEW.id; END"
            % label.replace("'", "''")
        )
    db_connection.commit()


def initialize_database(session_config=None):
    """Initialize database only when transcription starts (lazy loading)

    ``session_config`` is the config this session actually runs on. The worker
    process is reused across Start/Stop cycles and reloads config from disk at
    each session start, so its module-level `config` is whatever was on disk when
    the process spawned — every setting read here must come from the session's
    own config or the database is created (and described) from a stale one.
    """
    global db_name, db_initialized, live_session_id

    if db_initialized:
        return db_name

    cfg = session_config or config

    # Get custom database path from config or use default
    custom_db_path = cfg.get("database", {}).get("path", "").strip()
    path_format = cfg.get("database", {}).get("path_format", "").strip() or "%Y/%m"
    now = datetime.now()
    formatted_path = now.strftime(path_format)

    if custom_db_path:
        # Use custom base path + path_format subdirectory
        folder_name = os.path.join(custom_db_path, formatted_path)
        print(f"[OK] Using custom database path: {folder_name}")
    else:
        # Use default base path (under APP_DIR) + path_format subdirectory so compiled
        # builds keep the DB in ~/.stt instead of the launch directory.
        folder_name = os.path.join(BACKUP_DIR, formatted_path)
        print(f"[OK] Using default database path: {folder_name}")

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    # Make the DB directory tree readable/traversable by all users (consumers read these)
    make_dirs_world_readable(folder_name, custom_db_path or BACKUP_DIR)

    # Create database file path with configurable format (using Python strftime format)
    filename_format = cfg.get("database", {}).get(
        "filename_format", ""
    ).strip() or "%Y-%m-%d_%H%M%S"

    # Validate that format includes time component for unique per-session databases
    if not any(time_fmt in filename_format for time_fmt in ["%H", "%M", "%S"]):
        print(f"[WARNING] Database filename_format '{filename_format}' does not include time component.")
        print("[WARNING] This may cause sessions on the same day to share a database.")
        print("[WARNING] Using default format: %Y-%m-%d_%H%M%S for unique per-session databases.")
        filename_format = "%Y-%m-%d_%H%M%S"

    now = datetime.now()

    # Use strftime directly with user's format
    formatted_filename = now.strftime(filename_format)

    # Get custom filename prefix or use default
    filename_prefix = cfg.get("database", {}).get("filename_prefix", "").strip()
    if filename_prefix:
        db_name = os.path.join(
            folder_name, f"{formatted_filename}_{filename_prefix}.db"
        )
        print(
            f"[OK] Using custom database filename: {formatted_filename}_{filename_prefix}.db"
        )
    else:
        db_name = os.path.join(folder_name, f"{formatted_filename}.db")
        print(
            f"[OK] Using default database filename: {formatted_filename}.db"
        )

    print(f"[OK] Initializing database: {db_name}")

    try:
        # Use context manager for database connection
        with sqlite3.connect(db_name) as db_connection:
            db_cursor = db_connection.cursor()

            # Enable WAL mode for better concurrent read/write performance
            # WAL (Write-Ahead Logging) allows simultaneous reads while writing
            db_cursor.execute("PRAGMA journal_mode=WAL")
            db_cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still safe

            # Create the table if it doesn't exist (with start_time, end_time for temporal ordering)
            db_cursor.execute(
                """CREATE TABLE IF NOT EXISTS transcriptions (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, text TEXT, start_time REAL DEFAULT 0, end_time REAL DEFAULT 0)"""
            )

            # Create index on timestamp for faster ORDER BY queries
            db_cursor.execute(
                """CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp DESC)"""
            )

            # Migration: Check if id column exists, add it if missing
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "id" not in columns:
                print("[DB] Migrating database: adding id column...")
                # SQLite doesn't support ALTER TABLE ADD COLUMN with PRIMARY KEY
                # So we need to recreate the table (wrapped in transaction for safety)
                try:
                    db_cursor.execute("BEGIN")
                    db_cursor.execute(
                        """CREATE TABLE transcriptions_new (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, text TEXT)"""
                    )
                    db_cursor.execute(
                        """INSERT INTO transcriptions_new (timestamp, text) SELECT timestamp, text FROM transcriptions"""
                    )
                    db_cursor.execute("""DROP TABLE transcriptions""")
                    db_cursor.execute(
                        """ALTER TABLE transcriptions_new RENAME TO transcriptions"""
                    )
                    # Recreate index
                    db_cursor.execute(
                        """CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp DESC)"""
                    )
                    db_connection.commit()
                    print("[DB] OK: Migration complete")
                except Exception:
                    db_connection.rollback()
                    raise

            # Migration: Check if start_time/end_time columns exist, add them if missing
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "start_time" not in columns:
                print("[DB] Migrating database: adding start_time and end_time columns...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN start_time REAL DEFAULT 0")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN end_time REAL DEFAULT 0")
                db_connection.commit()
                print("[DB] OK: Migration complete (added temporal columns)")

            # Migration: Add corrections-related columns
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "confidence" not in columns:
                print("[DB] Migrating database: adding corrections columns (confidence, original_text, corrected_by, needs_review)...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN confidence REAL DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN original_text TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN corrected_by TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN needs_review INTEGER DEFAULT 0")
                db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_needs_review ON transcriptions(needs_review) WHERE needs_review = 1")
                db_connection.commit()
                print("[DB] OK: Migration complete (added corrections columns)")

            if "translated_text" not in columns:
                print("[DB] Migrating database: adding translation columns (translated_text, translation_language)...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN translated_text TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN translation_language TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added translation columns)")

            if "speech_type" not in columns:
                print("[DB] Migrating database: adding speech_type column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN speech_type TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added speech_type column)")

            if "audio_tag" not in columns:
                print("[DB] Migrating database: adding audio_tag column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN audio_tag TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added audio_tag column)")

            if "music_prob" not in columns:
                print("[DB] Migrating database: adding music_prob column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN music_prob REAL DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added music_prob column)")

            if "denied" not in columns:
                # `denied` is a UI visibility/hide flag (0 = visible, 1 = hidden from
                # the transcript view); toggled by handle_set_segment_denied.
                print("[DB] Migrating database: adding denied column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN denied INTEGER DEFAULT 0")
                db_connection.commit()
                print("[DB] OK: Migration complete (added denied column)")

            # Schema v2: additive columns for the downstream consumer (epoch-ms ordering,
            # per-word timing/confidence, partial/final flag, source language, segment pairing).
            # All nullable / defaulted so old .db files and readers keep working unchanged.
            db_cursor.execute("PRAGMA table_info(transcriptions)")
            columns = [row[1] for row in db_cursor.fetchall()]
            if "ts_ms" not in columns:
                print("[DB] Migrating database: adding ts_ms column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN ts_ms INTEGER DEFAULT NULL")
                db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts_ms ON transcriptions(ts_ms)")
                db_connection.commit()
                print("[DB] OK: Migration complete (added ts_ms column)")
            if "words_json" not in columns:
                print("[DB] Migrating database: adding words_json column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN words_json TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added words_json column)")
            if "is_final" not in columns:
                print("[DB] Migrating database: adding is_final column...")
                # 1 = finalized (all existing rows are finals); 0 = partial hypothesis
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN is_final INTEGER DEFAULT 1")
                db_connection.commit()
                print("[DB] OK: Migration complete (added is_final column)")
            if "partial_seq" not in columns:
                print("[DB] Migrating database: adding partial_seq column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN partial_seq INTEGER DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added partial_seq column)")
            if "source_language" not in columns:
                print("[DB] Migrating database: adding source_language column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN source_language TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added source_language column)")
            if "segment_id" not in columns:
                print("[DB] Migrating database: adding segment_id column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN segment_id TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added segment_id column)")
            if "words_source" not in columns:
                # ASR backend that produced the row (faster_whisper/whisper/...). Makes a
                # NULL words_json interpretable: 'whisper' emits no per-word data, so NULL
                # is expected there, vs 'faster_whisper' where NULL would be unexpected.
                print("[DB] Migrating database: adding words_source column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN words_source TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added words_source column)")
            if "session_id" not in columns:
                # Stable per-session id (the .db filename stem) on every row, so the
                # consumer can anchor socket<->db and group rows by session.
                print("[DB] Migrating database: adding session_id column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN session_id TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added session_id column)")
            if "denied_reason" not in columns:
                # Why the row was denied: 'hallucination', 'cjk', 'cjk_shadow', 'short', 'dup'.
                # NULL means the row is a normal visible segment.
                print("[DB] Migrating database: adding denied_reason column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN denied_reason TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added denied_reason column)")
            if "marked" not in columns:
                # Manual operator bookmark (0 = normal, 1 = marked). Set from the
                # corrections page so a segment can be found again later; unrelated
                # to needs_review (the low-confidence review queue).
                print("[DB] Migrating database: adding marked column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN marked INTEGER DEFAULT 0")
                db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_marked ON transcriptions(marked) WHERE marked = 1")
                db_connection.commit()
                print("[DB] OK: Migration complete (added marked column)")
            if "translation_ts_ms" not in columns:
                # Epoch-ms when translated_text was written. Translation arrives via a
                # later async UPDATE, so the row's ts_ms alone can't reproduce the
                # source-vs-translation timing a live consumer experienced.
                print("[DB] Migrating database: adding translation_ts_ms column...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN translation_ts_ms INTEGER DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added translation_ts_ms column)")
            if "asr_model" not in columns:
                # What produced this row, per row rather than per session. The session
                # records what was *configured*; a caption records what actually ran,
                # and the two part company exactly where it matters. A session set to
                # translate with the LLM still contains NMT rows wherever the LLM
                # declined a caption, and nothing distinguished them — so measuring an
                # LLM change against a past service compared it against the other
                # model's output on precisely the rows the two disagree about.
                print("[DB] Migrating database: adding model provenance columns...")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN asr_model TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN mt_engine TEXT DEFAULT NULL")
                db_cursor.execute("ALTER TABLE transcriptions ADD COLUMN mt_model TEXT DEFAULT NULL")
                db_connection.commit()
                print("[DB] OK: Migration complete (added asr_model, mt_engine, mt_model columns)")

            # A row's asr_model is left NULL while the session's own model is the one
            # transcribing, because session_meta already records that and repeating it
            # per row is the same string written a thousand times (160 KB on a 3,500-row
            # service). It is stamped only after a hot reload changes the model, so a
            # value means "not what the session started with" — see
            # _set_asr_row_stamp(), which installs the trigger at that point.
            try:
                _set_asr_row_stamp(db_connection, None)
            except Exception as _asr_err:
                print(f"[DB] WARNING: could not prepare the ASR row stamp ({_asr_err})")

            # Service-phase tables. Created here, inside the init transaction, so the
            # detector's tick (which runs in the web process, on its own connection)
            # never races a CREATE against the writer. Empty tables are harmless on a
            # session where the feature is disabled.
            try:
                _service_phase_ensure_tables(db_connection)
            except Exception as _sp_err:
                print(f"[DB] WARNING: service phase tables unavailable ({_sp_err})")

            # Insert a blank first entry with default values
            default_timestamp = " "
            default_text = " "
            db_cursor.execute(
                "INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)",
                (default_timestamp, default_text),
            )
            db_connection.commit()

        db_initialized = True
        # Make the DB file (and any WAL/SHM sidecars) readable by all users
        make_db_world_readable(db_name)

        # Record which models and decode settings produced this session, so a
        # transcript reviewed weeks later can be attributed to the transcription
        # or the translation stage instead of being unattributable once the server
        # is restarted or retuned. Written after the init transaction closes so
        # there is only ever one writer, and non-fatal by contract: a session must
        # start even when provenance can't be recorded.
        # The baseline every row is measured against. Set unconditionally, not only
        # when session_meta is enabled: it decides whether a row repeats the model
        # name or stays NULL, and a session with provenance turned off must still
        # produce rows that a reader can attribute — there, nothing is redundant,
        # so every row carries the label.
        _set_mt_baseline_label(
            _session_mt_row_label(
                cfg.get("live_translation", {}),
                MT_ENGINE_REMOTE if _translation_is_offloaded(cfg.get("live_translation", {}))
                else (MT_ENGINE_LLM if _translation_uses_llm(cfg.get("live_translation", {}))
                      else MT_ENGINE_NMT),
                remote_status=_remote_effective_status(),
                model=_resolve_live_translation_model_id(cfg.get("live_translation", {})))
            if _session_meta_enabled(cfg) else "")

        if _session_meta_enabled(cfg):
            _session_provenance = _current_session_meta(cfg)
            if _write_session_meta(db_name, _session_provenance):
                print(f"[DB] OK: Recorded session provenance ({len(_session_provenance)} settings)")
            # When translation is offloaded, the remote's model is the one that
            # translates — fetch it over the network off-thread so a slow or
            # unreachable remote can't delay the start of transcription.
            if _translation_is_offloaded(cfg.get("live_translation", {})):
                _record_remote_provenance_async(db_name)
        # Stable per-session id = the .db filename stem (e.g. 2026-06-22_183007).
        # Stored on every row and emitted top-level on every socket payload so the
        # consumer can anchor socket<->db by exact match; re-derived each session.
        live_session_id = os.path.splitext(os.path.basename(db_name))[0]
        # Store database name + session id in shared state for web server access
        transcription_state["db_name"] = db_name
        transcription_state["session_id"] = live_session_id
        print("[OK] Database initialized successfully")

        return db_name
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        # Clean up database file if initialization failed
        if db_name and os.path.exists(db_name):
            try:
                os.unlink(db_name)
            except OSError:
                pass
        raise


# File Transcription Helper Functions


def extract_audio_from_file(file_path):
    """
    Extract audio from video/audio file using pydub.
    Converts to WAV 16kHz mono for transcription.

    Args:
        file_path: Path to audio/video file

    Returns:
        Path to converted WAV file
    """
    temp_wav = None
    try:
        # Get file extension
        ext = os.path.splitext(file_path)[1].lower().replace(".", "")

        # Load file
        _lazy_import_audio()
        if ext in SUPPORTED_VIDEO_FORMATS + SUPPORTED_AUDIO_FORMATS:
            audio = AudioSegment.from_file(file_path, format=ext)
        else:
            audio = AudioSegment.from_file(file_path)

        # Convert to WAV 16kHz mono
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)

        # Save as WAV with proper cleanup
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()  # Close the file handle before pydub writes to it
        audio.export(temp_wav.name, format="wav")

        return temp_wav.name

    except Exception as e:
        # Clean up temp file if it exists
        if temp_wav and os.path.exists(temp_wav.name):
            try:
                os.unlink(temp_wav.name)
            except OSError:
                pass
        raise Exception(f"Failed to extract audio: {e!s}") from e


# Formatting/export helpers live in stt/formatting.py (importable, unit-tested);
# names are re-imported here so existing call sites stay unchanged.
from stt import formatting as _formatting
from stt.formatting import (  # noqa: F401
    apply_word_highlighting_server,
    convert_db_to_translation_srt,
    format_file_size,
    format_timestamp_srt,
    format_timestamp_vtt,
    format_transcription,
)


def _word_highlighting_config_path():
    return os.path.join(CONFIG_DIR, "word_highlighting.json")


def convert_db_to_srt(db_path):
    """Convert a Transcriptions.db to SRT (plus HTML export if enabled in settings)."""
    # Reload config to get fresh settings (global config may be stale)
    html_enabled = load_config().get("database", {}).get("html_enabled", True)
    return _formatting.convert_db_to_srt(db_path, html_enabled=html_enabled, highlight_config_path=_word_highlighting_config_path())


def convert_db_to_html(db_path):
    """Convert a Transcriptions.db to HTML with word highlighting."""
    return _formatting.convert_db_to_html(db_path, highlight_config_path=_word_highlighting_config_path())


app = Flask(__name__,
            template_folder=os.path.join(BUNDLE_DIR, "templates"),
            static_folder=os.path.join(BUNDLE_DIR, "static"))
app.config["SECRET_KEY"] = os.environ.get("STT_SECRET_KEY") or secrets.token_urlsafe(32)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4GB cap on uploads (media files are large)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # Auto-reload templates when they change
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Disable caching for static files
socketio = SocketIO(app, async_mode="threading", static_url_path="/static", static_folder=os.path.join(BUNDLE_DIR, "static"), ping_timeout=120, ping_interval=25)

# --- Access log ------------------------------------------------------------
# Records every HTTP request (tagged web vs api), WebSocket connect/disconnect,
# and SocketIO action, into a size-capped SQLite table under APP_DIR/logs.
# Viewed at /logs (page) and /api/logs (JSON). Thin wrapper over stt.request_log.
from stt import request_log as _request_log  # noqa: E402
from stt import metrics as _metrics  # noqa: E402
from stt import audio_file as _audio_file  # noqa: E402
from stt.hypothesis_buffer import LocalAgreementBuffer  # noqa: E402

try:
    os.makedirs(os.path.join(APP_DIR, "logs"), exist_ok=True)
    request_logger = _request_log.RequestLog(
        os.path.join(APP_DIR, "logs", "access_log.db"),
        max_rows=int((config.get("access_log", {}) or {}).get("max_rows", 50000) or 50000),
    )
except Exception:
    # A logging store failure must never stop the server from booting.
    request_logger = None


def _access_log_enabled():
    """Whether request logging is currently on (default True). Read fresh so a
    config change via /api/config takes effect without a restart."""
    try:
        return bool((config.get("access_log", {}) or {}).get("enabled", True))
    except Exception:
        return False


def _access_log_skip_polling():
    """Whether to drop dashboard-polling requests before they are written.

    The log already knows these paths are noise — /api/logs filters them out via
    POLLING_LOG_PATHS whenever hide_polling is set, which the logs page sends by default.
    Writing and fsyncing a row only to hide it again is pure cost: an operator with the
    health and live-settings pages open generates around 150 rows a minute, and they
    displace the real requests inside the 50,000-row cap. Set false to log them anyway
    when diagnosing the polling itself. Read fresh so a config change takes effect
    without a restart."""
    try:
        return bool((config.get("access_log", {}) or {}).get("skip_polling_paths", True))
    except Exception:
        return True


def _record_socket_event(event_name, kind):
    """Log a SocketIO connect/disconnect or action. Best-effort, never raises."""
    if request_logger is None or not _access_log_enabled():
        return
    try:
        sid = getattr(request, "sid", None)
        request_logger.log(
            source=_request_log.SOURCE_SOCKET,
            kind=kind,
            path=event_name,
            ip=getattr(request, "remote_addr", None),
            user_agent=request.headers.get("User-Agent") if request else None,
            detail=(f"sid={sid}" if sid else None),
        )
    except Exception:
        pass


# Wrap socketio.on so every @socketio.on(...) handler registered below is
# transparently logged — connect/disconnect as connections, everything else as
# actions. Installed before any handler is defined so all of them are covered.
_socketio_on_orig = socketio.on


def _logging_socketio_on(message, namespace=None):
    decorator = _socketio_on_orig(message, namespace=namespace)
    _kind = _request_log.KIND_CONNECTION if message in ("connect", "disconnect") else _request_log.KIND_ACTION

    def register(handler):
        @functools.wraps(handler)
        def instrumented(*args, **kwargs):
            _record_socket_event(message, _kind)
            return handler(*args, **kwargs)
        return decorator(instrumented)
    return register


socketio.on = _logging_socketio_on
# ---------------------------------------------------------------------------

# When this (web) process started — used for the Server Settings uptime display.
# Resets on restart/redeploy/crash-recovery, reflecting the live server-process lifetime.
SERVER_START_TIME = time.time()

# Running version (read once at startup; a self-update restarts the process, so a
# fresh run picks up the new VERSION). Same source as the Sentry release tag.
try:
    with open(os.path.join(BUNDLE_DIR, "VERSION"), encoding="utf-8") as _vf:
        SERVER_VERSION = _vf.read().strip()
except OSError:
    SERVER_VERSION = "unknown"

# Exact git commit of this checkout (empty for frozen/installer builds with no .git).
# Recorded in the live-map ping alongside SERVER_VERSION; recomputed on self-update re-exec.
try:
    from stt.self_update import git_commit as _git_commit
    SERVER_COMMIT = _git_commit(BUNDLE_DIR)
except Exception:
    SERVER_COMMIT = ""

# Human version of the running checkout, e.g. '26.1.2-9-gc588d29' (tag + commits
# since + commit). The VERSION file only changes on releases, so on a git deploy
# it understates what's actually running; empty for frozen builds.
try:
    from stt.self_update import git_describe as _git_describe
    SERVER_DESCRIBE = _git_describe(BUNDLE_DIR)
except Exception:
    SERVER_DESCRIBE = ""


def _compute_display_version():
    """Single monotonic version string for scripts and the UI (see stt/config_utils.py)."""
    return _config_utils.compute_display_version(SERVER_DESCRIBE, SERVER_COMMIT, SERVER_VERSION)


SERVER_DISPLAY_VERSION = _compute_display_version()


# --- System requirements (informational warning shown in the web UI header) ---
# Requirements are estimated from the configured models (live Whisper, file
# transcription Whisper, local NLLB) rather than static tiers, and compared
# against detected hardware: RAM, CUDA VRAM, and Apple Silicon unified memory
# (where CPU and GPU share one pool). Every probe and lookup fails open
# (unknown -> no warning) and must never block startup.

# Model-memory tables + pure estimation/requirements logic live in
# stt/model_memory.py (importable, tested). Hardware probing (below) stays here
# and is injected via the shims further down.
from stt.model_memory import (  # noqa: F401
    MODEL_MEMORY_ESTIMATES,
    estimate_memory_requirements,
    check_system_requirements,
)


def _get_total_ram_bytes():
    """Total physical RAM in bytes, stdlib only; 0 when unknown (no false warning)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return 0
        if sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip()) if r.returncode == 0 else 0
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def _probe_vram_bytes():
    """Total VRAM of CUDA device 0 in bytes; None when unknown.

    Never triggers the heavy torch import itself: uses torch only when it is
    already in sys.modules, otherwise falls back to nvidia-smi (stdlib-only,
    works at startup before the lazy ML import runs).
    """
    try:
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.get_device_properties(0).total_memory)
            return None  # torch is loaded and says no CUDA — trust it
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and r.stdout.strip():
            return int(float(r.stdout.strip().splitlines()[0])) * 1024**2
    except Exception:
        pass
    return None


_HW_PROBE_CACHE = None
_HW_PROBE_LOCK = threading.Lock()


def _probe_hardware():
    """Cached hardware snapshot; every field fails open (0 / None / False).

    VRAM is re-probed on later calls if it was unknown and torch has since
    been imported (the lazy ML import can happen after startup).
    """
    global _HW_PROBE_CACHE
    with _HW_PROBE_LOCK:
        hw = _HW_PROBE_CACHE
        if hw is None:
            try:
                import platform as _platform  # module-level `platform` is sys.platform
                apple_silicon = sys.platform == "darwin" and _platform.machine() == "arm64"
            except Exception:
                apple_silicon = False
            vram = _probe_vram_bytes()
            hw = {
                "ram_bytes": _get_total_ram_bytes(),
                "cpu_cores": os.cpu_count() or 0,
                "vram_bytes": vram,
                "has_cuda": vram is not None,
                "apple_silicon": apple_silicon,
            }
            _HW_PROBE_CACHE = hw
        elif hw["vram_bytes"] is None and "torch" in sys.modules:
            vram = _probe_vram_bytes()
            hw["vram_bytes"] = vram
            hw["has_cuda"] = vram is not None
    try:
        import shutil
        hw["disk_free_bytes"] = shutil.disk_usage(APP_DIR).free  # cheap; keep current
    except Exception:
        hw["disk_free_bytes"] = 0
    return hw


def _estimate_memory_requirements(cfg, gpu_available=False, unified=False):
    """Thin shim: pure math lives in stt.model_memory; inject MODELS_DIR."""
    return estimate_memory_requirements(cfg, MODELS_DIR, gpu_available=gpu_available, unified=unified)


def _check_system_requirements(cfg=None, hw=None):
    """Thin shim: default cfg/hw to the live config + cached probe (both IO),
    then delegate to the pure stt.model_memory.check_system_requirements."""
    if cfg is None:
        cfg = config
    if hw is None:
        hw = _probe_hardware()
    return check_system_requirements(cfg, hw, MODELS_DIR)


SYSTEM_REQUIREMENTS_WARNINGS = _check_system_requirements()
if SYSTEM_REQUIREMENTS_WARNINGS:
    print("[SYSREQ] Below minimum system requirements: "
          + "; ".join(SYSTEM_REQUIREMENTS_WARNINGS))

app_logger = logging.getLogger(__name__)  # Use your module name here
socket_io_logger = logging.getLogger("socketio")

# Set log levels as needed
app_logger.setLevel(logging.DEBUG)
socket_io_logger.setLevel(logging.WARNING)

# Disable Flask's built-in logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


class _SuppressBenignWSGINoise(logging.Filter):
    """Drop the harmless 'write() before start_response' AssertionError that the
    werkzeug dev server logs for Socket.IO polling/transport requests under
    async_mode='threading' (the request finishes without the normal WSGI
    response path). The connection still works; this is cosmetic noise. All
    other werkzeug errors pass through unchanged."""

    def filter(self, record):
        try:
            if "write() before start_response" in record.getMessage():
                return False
        except Exception:
            pass
        return True


log.addFilter(_SuppressBenignWSGINoise())

# Password-based authentication sessions
# Format: {session_token: {"ip": client_ip, "expires": datetime}}
auth_sessions = {}
auth_sessions_lock = threading.Lock()

# Per-IP login throttle: after LOGIN_MAX_FAILURES wrong passwords, that IP is
# locked out for LOGIN_LOCKOUT_SECONDS. Bounds online brute-forcing of the
# configured password. Format: {ip: {"failures": int, "locked_until": datetime}}
_login_attempts = {}
_login_attempts_lock = threading.Lock()
LOGIN_MAX_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 30

def generate_session_token():
    """Generate a secure random session token"""
    import secrets
    return secrets.token_urlsafe(32)

def cleanup_expired_sessions():
    """Remove expired sessions from the auth_sessions dict"""
    now = datetime.now()
    with auth_sessions_lock:
        expired = [token for token, data in auth_sessions.items() if data["expires"] < now]
        for token in expired:
            del auth_sessions[token]
        if expired:
            print(f"[AUTH] Cleaned up {len(expired)} expired sessions")


@app.route("/")
def index():
    # Check if URL has any parameters
    if not request.args:
        # No parameters provided: redirect to the active URL-builder profile (if any)
        profiles, active = get_url_builder_profiles()
        params = next((p["params"] for p in profiles if p["name"] == active), None)
        if params:
            from flask import redirect, url_for
            return redirect(url_for('index', **params))

    response = make_response(render_template("index.html"))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route("/profile/<name>")
def profile_view(name):
    """Display the page using a named profile's settings, e.g. /profile/lower3rd.
    Case-insensitive; unknown names fall back to the root view."""
    from flask import redirect, url_for
    profiles, _ = get_url_builder_profiles()
    target = (name or "").strip().lower()
    params = next((p["params"] for p in profiles if p["name"].strip().lower() == target), None)
    if params is None:
        return redirect("/")
    return redirect(url_for("index", **params))


@app.route("/favicon.ico")
def favicon():
    # Return a custom response or an empty response
    return "", 204


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors by redirecting to appropriate page"""
    if check_ip_whitelist():
        return redirect("/live-settings")
    else:
        return redirect("/")


def _control_params(keep_blank=False):
    """Request parameters for an endpoint, from wherever the client sent them.

    Thin wrapper over stt.http_params.merge_request_params: a show-control system
    may send a JSON body, a form body, or a query string, and rejecting two of the
    three looks to the operator like a button that did nothing. Precedence is
    JSON > form > query.

    ``keep_blank=False`` (switch endpoints) treats a blank value as "not sent", so
    a surface posting a fixed field set on every press can't blank a live setting.
    ``keep_blank=True`` (settings endpoints) keeps blanks, because an empty string
    is how those clear a field.

    Do NOT use this on a route that merges the body wholesale into config — it
    unions the query string in, and a URL can carry ?key=<access_token>, which
    would be persisted as a config key. /api/config reads its JSON directly for
    exactly that reason.
    """
    body = request.get_json(silent=True)
    if body is None and not request.form:
        # Well-formed JSON under the wrong Content-Type: Flask's get_json returns
        # None and request.form stays empty (it only parses form content types),
        # so the body is invisible and the route would 400 as though nothing was
        # sent. Read it off the raw bytes as a last resort.
        try:
            body = _parse_json_body(request.get_data(as_text=True))
        except Exception:
            body = None
    return merge_request_params(body, request.form, request.args, keep_blank=keep_blank)


def check_ip_whitelist():
    """Check if the client IP is in the whitelist or has a valid password session"""
    import ipaddress

    # First check if password authentication is enabled
    password_auth_config = config.get("web_server", {}).get("password_auth", {})
    password_auth_enabled = password_auth_config.get("enabled", False)
    access_token = str(config.get("web_server", {}).get("access_token", "") or "")

    # A request carrying ?key=<access_token> is granted directly, so a
    # non-whitelisted device (or an OBS/display source that can't use the login
    # page) can embed auth in the URL. _mint_access_token_cookie (after_request)
    # then sets a session cookie so the rest of that browser session works
    # without re-passing the key. Empty token = disabled.
    if access_token:
        provided_key = request.args.get("key", "")
        if provided_key and secrets.compare_digest(provided_key, access_token):
            return True

    # Check for a valid session token (from password login OR a minted
    # access-token session) via cookie or Authorization header.
    if password_auth_enabled or access_token:
        # Check cookie first
        session_token = request.cookies.get("auth_session")

        # Fallback to Authorization header
        if not session_token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                session_token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate session token
        if session_token:
            cleanup_expired_sessions()  # Clean up expired sessions
            with auth_sessions_lock:
                session_data = auth_sessions.get(session_token)
                if session_data:
                    # Check if session is still valid
                    if session_data["expires"] > datetime.now():
                        # Check if IP matches (prevent session hijacking)
                        if session_data["ip"] == request.remote_addr:
                            return True
                        else:
                            print(f"[AUTH WARNING] Session token used from different IP: {request.remote_addr} != {session_data['ip']}")
                    else:
                        # Session expired, remove it
                        del auth_sessions[session_token]

    whitelist = config.get("web_server", {}).get("settings_ip_whitelist", [])

    # If whitelist is empty, allow all
    if not whitelist:
        return True

    client_ip = request.remote_addr

    # Always allow localhost variations
    localhost_ips = ["127.0.0.1", "::1", "localhost"]
    if client_ip in localhost_ips:
        return True

    # Check whitelist (supports both exact IPs and CIDR ranges)
    try:
        client_ip_obj = ipaddress.ip_address(client_ip)

        for entry in whitelist:
            entry = entry.strip()
            if not entry or entry.startswith("#"):  # Skip comments
                continue

            try:
                # Check if entry is a CIDR range (contains /)
                if "/" in entry:
                    network = ipaddress.ip_network(entry, strict=False)
                    if client_ip_obj in network:
                        return True
                else:
                    # Exact IP match
                    if client_ip == entry:
                        return True
            except ValueError:
                # Invalid entry, skip it
                print(f"[WARNING] Invalid IP whitelist entry: {entry}")
                continue

        return False
    except ValueError:
        # Invalid client IP format
        print(f"[WARNING] Invalid client IP format: {client_ip}")
        return False


@app.before_request
def _access_log_start_timer():
    """Stamp a monotonic start time so the after_request hook can report the
    request duration. Best-effort — never breaks a request."""
    try:
        g._access_log_t0 = time.perf_counter()
    except Exception:
        pass


def _note_access_detail(detail):
    """Attach a short summary to this request's access-log row.

    Routes opt in and pass only a curated description of what they did — never the
    raw parameter map, which can contain ?key=<access_token> picked up from the
    query string. That is the same reason the hook below drops query strings.

    Exists so a control-surface press is self-explaining afterwards: the row alone
    shows whether the request arrived, what it asked for, and whether anything
    actually changed. Best-effort — never breaks a request."""
    try:
        g._access_log_detail = str(detail)[:300]
    except Exception:
        pass


@app.after_request
def _access_log_record(response):
    """Log the request (tagged web vs api). Skips static assets and socket.io
    transport frames — WebSocket traffic is logged via the socket handlers.
    Best-effort — never breaks a response.

    The query string is deliberately dropped: it can carry ?key=<access_token>,
    which must not be written to the log. Routes that want context in the row
    call _note_access_detail() with a curated summary instead."""
    try:
        if request_logger is not None and _access_log_enabled():
            path = request.path or ""
            if not (path.startswith("/static/") or path.startswith("/socket.io/")
                    or (_access_log_skip_polling() and path in POLLING_LOG_PATHS)):
                t0 = getattr(g, "_access_log_t0", None)
                duration_ms = round((time.perf_counter() - t0) * 1000.0, 2) if t0 is not None else None
                request_logger.log(
                    source=_request_log.classify_source(path),
                    kind=_request_log.KIND_HTTP,
                    method=request.method,
                    path=path,
                    status=response.status_code,
                    ip=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                    duration_ms=duration_ms,
                    detail=getattr(g, "_access_log_detail", None),
                )
    except Exception:
        pass  # request logging must never break a response
    return response


@app.after_request
def _mint_access_token_cookie(response):
    """Turn a valid ?key=<access_token> request into a browser session.

    A display/OBS source (or any device) can embed ?key=<token> in the URL to
    get in; this mints an IP-bound auth_session cookie on that response so the
    page's subsequent requests (assets, socket.io, navigation) authenticate
    without re-passing the key. Best-effort — never breaks a response."""
    try:
        access_token = str(config.get("web_server", {}).get("access_token", "") or "")
        if not access_token or request.cookies.get("auth_session"):
            return response
        provided_key = request.args.get("key", "")
        if not (provided_key and secrets.compare_digest(provided_key, access_token)):
            return response
        timeout_minutes = int(config.get("web_server", {}).get("password_auth", {})
                              .get("session_timeout_minutes", 60) or 60)
        token = generate_session_token()
        with auth_sessions_lock:
            auth_sessions[token] = {
                "ip": request.remote_addr,
                "expires": datetime.now() + timedelta(minutes=timeout_minutes),
                "created": datetime.now(),
            }
        response.set_cookie("auth_session", token, max_age=timeout_minutes * 60,
                            httponly=True, samesite="Strict")
    except Exception:
        pass  # auth-cookie minting must never break a response
    return response


@app.route("/api/auth/login", methods=["POST"])
def password_login():
    """Authenticate with password and create a temporary session"""
    try:
        password_auth_config = config.get("web_server", {}).get("password_auth", {})

        # Check if password auth is enabled
        if not password_auth_config.get("enabled", False):
            return jsonify({"success": False, "error": "Password authentication is disabled"}), 403

        # Reject early if this IP is currently locked out
        client_ip = request.remote_addr
        with _login_attempts_lock:
            entry = _login_attempts.get(client_ip)
            if entry and entry.get("locked_until") and entry["locked_until"] > datetime.now():
                retry_after = int((entry["locked_until"] - datetime.now()).total_seconds()) + 1
                return jsonify({
                    "success": False,
                    "error": "Too many failed attempts. Try again later.",
                }), 429, {"Retry-After": str(retry_after)}

        # Get password from request
        data = request.get_json()
        provided_password = data.get("password", "")

        if not provided_password:
            return jsonify({"success": False, "error": "Password required"}), 400

        # Get configured password
        configured_password = password_auth_config.get("password", "")

        # If no password configured, reject
        if not configured_password:
            return jsonify({"success": False, "error": "No password configured on server"}), 500

        # Verify password (constant-time to avoid leaking length/prefix via timing)
        if not secrets.compare_digest(str(provided_password), str(configured_password)):
            print(f"[AUTH] Failed login attempt from {client_ip}")
            with _login_attempts_lock:
                rec = _login_attempts.setdefault(client_ip, {"failures": 0, "locked_until": None})
                rec["failures"] += 1
                if rec["failures"] >= LOGIN_MAX_FAILURES:
                    rec["locked_until"] = datetime.now() + timedelta(seconds=LOGIN_LOCKOUT_SECONDS)
                    rec["failures"] = 0
            return jsonify({"success": False, "error": "Invalid password"}), 401

        # Password correct — clear any failure record for this IP
        with _login_attempts_lock:
            _login_attempts.pop(client_ip, None)

        # Password correct - create session
        session_token = generate_session_token()
        timeout_minutes = password_auth_config.get("session_timeout_minutes", 60)
        expires = datetime.now() + timedelta(minutes=timeout_minutes)

        with auth_sessions_lock:
            auth_sessions[session_token] = {
                "ip": request.remote_addr,
                "expires": expires,
                "created": datetime.now()
            }

        print(f"[AUTH] Successful login from {request.remote_addr}, session expires in {timeout_minutes} minutes")

        # Return session token
        response = jsonify({
            "success": True,
            "session_token": session_token,
            "expires": expires.isoformat(),
            "timeout_minutes": timeout_minutes
        })

        # Set cookie for browser-based access
        response.set_cookie(
            "auth_session",
            session_token,
            max_age=timeout_minutes * 60,  # in seconds
            httponly=True,  # Prevent JavaScript access
            samesite="Strict"  # CSRF protection
        )

        return response

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/logout", methods=["POST"])
def password_logout():
    """Logout and invalidate the current session"""
    try:
        # Get session token
        session_token = request.cookies.get("auth_session")
        if not session_token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                session_token = auth_header[7:]

        if session_token:
            with auth_sessions_lock:
                if session_token in auth_sessions:
                    del auth_sessions[session_token]
                    print(f"[AUTH] Logged out session from {request.remote_addr}")

        response = jsonify({"success": True, "message": "Logged out successfully"})

        # Clear cookie
        response.set_cookie("auth_session", "", max_age=0)

        return response

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Check if the current session is authenticated"""
    try:
        is_authenticated = check_ip_whitelist()

        # Get session info if authenticated via password
        session_info = None
        if is_authenticated:
            session_token = request.cookies.get("auth_session")
            if not session_token:
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    session_token = auth_header[7:]

            if session_token:
                with auth_sessions_lock:
                    session_data = auth_sessions.get(session_token)
                    if session_data:
                        session_info = {
                            "authenticated_via": "password",
                            "expires": session_data["expires"].isoformat(),
                            "created": session_data["created"].isoformat()
                        }

        if not session_info and is_authenticated:
            session_info = {
                "authenticated_via": "ip_whitelist"
            }

        # Determine redirect URL based on authentication
        redirect_url = "/live-settings" if is_authenticated else "/url-builder"

        return jsonify({
            "success": True,
            "authenticated": is_authenticated,
            "session": session_info,
            "ip": request.remote_addr,
            "redirect_url": redirect_url
        })

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/live-settings")
def settings_page():
    """Render the live settings page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403

    return render_template("live-settings.html")


@app.route("/service-phase")
def service_phase_page():
    """Render the service phase detection / review page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403

    return render_template("service-phase.html")


def _service_phase_resolve_db():
    """(path, error_response) for the running session.

    Live only, deliberately. These routes take no path from the caller, so there is no
    path to confine and no way to reach a file outside the running session.
    """
    live = _service_phase_session_db()
    if not live or not os.path.exists(live):
        return None, (jsonify({"success": False, "error": "No session is running"}), 404)
    return live, None


@app.route("/api/service-phase")
def get_service_phase():
    """Detected phases for the running session.

    ``?recompute=1`` re-runs the detector over the session's rows instead of reading the
    output the tick saved, without writing anything. That is what makes the logic
    improvable without waiting: change a threshold, reload, compare against what the
    operator is watching happen.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    db_path, err = _service_phase_resolve_db()
    if err:
        return err

    cfg = _service_phase_config()
    recompute = request.args.get("recompute") in ("1", "true", "yes")
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            if recompute:
                result = _service_phase_analyze(
                    _service_phase_rows(conn), cfg,
                    first_sunday=_service_phase_first_sunday(db_path),
                    rules=_service_phase_rules())
            else:
                result = _service_phase_load(conn)
            corrections = _service_phase_corrections(conn)
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "success": True,
        "session_id": os.path.basename(db_path),
        "recomputed": recompute,
        "enabled": bool(cfg.get("enabled", True)),
        # The page renumbers the songs itself when a correction moves the opening, so it
        # needs the same threshold the detector used to decide what counts as a song.
        "songs_min_minutes": int(cfg.get("songs_min_minutes", 3)),
        "first_sunday": _service_phase_first_sunday(db_path),
        "current": result.get("current"),
        "blocks": result.get("blocks", []),
        "bins": result.get("bins", []),
        "classes": result.get("classes", ""),
        "spans": result.get("spans", []),
        "corrections": corrections,
    })


@app.route("/api/service-phase/correct", methods=["POST"])
def save_service_phase_correction():
    """Record an operator correction. Never touched by the detector's own rewrites."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    data = _control_params(keep_blank=True)
    db_path, err = _service_phase_resolve_db()
    if err:
        return err

    block_index = data.get("block_index")
    block_index = None if block_index in (None, "", "null") else coerce_int(block_index, 0, lo=0, hi=10000)
    label = (data.get("label") or "").strip()[:120]
    kind = (data.get("kind") or "").strip()[:8]
    note = (data.get("note") or "").strip()[:500]
    if not label and not kind:
        return jsonify({"success": False, "error": "A correction needs a label or a kind."}), 400

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            row_id = _service_phase_save_correction(
                conn, block_index, kind=kind or None, label=label or None,
                start_ms=coerce_int(data.get("start_ms"), 0, lo=0, hi=2 ** 62) or None,
                end_ms=coerce_int(data.get("end_ms"), 0, lo=0, hi=2 ** 62) or None,
                note=note, corrected_at=datetime.now().isoformat(timespec="seconds"))
            corrections = _service_phase_corrections(conn)
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({"success": True, "id": row_id, "corrections": corrections})


@app.route("/api/service-phase/uncorrect", methods=["POST"])
def delete_service_phase_correction():
    """Withdraw an operator correction, handing the block back to the detector."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    data = _control_params(keep_blank=True)
    db_path, err = _service_phase_resolve_db()
    if err:
        return err

    # A group spans several blocks and so has no block_index; the page undoes it by the row
    # id it already holds. One or the other, never both.
    row_id = data.get("id")
    block_index = data.get("block_index")
    if row_id not in (None, "", "null"):
        row_id = coerce_int(row_id, 0, lo=1, hi=2 ** 62)
        block_index = None
    elif block_index not in (None, "", "null"):
        row_id = None
        block_index = coerce_int(block_index, 0, lo=0, hi=10000)
    else:
        return jsonify({"success": False, "error": "An undo needs a block index or a row id."}), 400

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            if row_id is not None:
                removed = _service_phase_delete_correction_by_id(conn, row_id)
            else:
                removed = _service_phase_delete_correction(conn, block_index)
            corrections = _service_phase_corrections(conn)
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({"success": True, "removed": removed, "corrections": corrections})


@app.route("/api/service-phase/group", methods=["POST"])
def group_service_phase_blocks():
    """Record several detected blocks as one phase, spanning their whole time range."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    data = _control_params(keep_blank=True)
    db_path, err = _service_phase_resolve_db()
    if err:
        return err

    start_ms = coerce_int(data.get("start_ms"), 0, lo=0, hi=2 ** 62)
    end_ms = coerce_int(data.get("end_ms"), 0, lo=0, hi=2 ** 62)
    label = (data.get("label") or "").strip()[:120]
    kind = (data.get("kind") or "").strip()[:8]
    note = (data.get("note") or "").strip()[:500]
    if not label:
        return jsonify({"success": False, "error": "A group needs a name."}), 400
    if not start_ms or end_ms <= start_ms:
        return jsonify({"success": False, "error": "A group needs a time range."}), 400

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            row_id = _service_phase_save_group(
                conn, start_ms, end_ms, kind=kind or None, label=label,
                note=note, corrected_at=datetime.now().isoformat(timespec="seconds"))
            corrections = _service_phase_corrections(conn)
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({"success": True, "id": row_id, "corrections": corrections})


def _service_phase_learn_scan(limit=200):
    """Corrected phases across the archive, newest sessions first.

    Read-only and opened one at a time: the archive is on the same disk a service may be
    recording to, and a learner is never worth competing with the writer for it.
    """
    paths = sorted(_db_iter_databases(_sidecar_sweep_dirs()), reverse=True)[:limit]
    phases = []
    for path in paths:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            phases.extend(_phase_learn_collect([(os.path.basename(path), conn)], with_text=True))
        except Exception:
            continue   # one unreadable session must not stop the sweep
        finally:
            conn.close()
    return phases


@app.route("/api/service-phase/learn", methods=["POST"])
def learn_service_phase_settings():
    """What this installation's own corrected services say the settings should be."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        with open(CONFIG_TEMPLATE_FILE, "r", encoding="utf-8") as f:
            baseline = (json.load(f).get("service_phase") or {})
    except Exception:
        baseline = {}

    try:
        result = _phase_learn_propose(_service_phase_learn_scan(), _service_phase_config(),
                                      baseline)
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500

    result["success"] = True
    return jsonify(result)


@app.route("/api/service-phase/learn/apply", methods=["POST"])
def apply_service_phase_settings():
    """Take the proposals the operator ticked. Nothing is ever applied without this call."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    data = _control_params(keep_blank=True)
    keys = data.get("keys")
    if not isinstance(keys, list) or not keys:
        return jsonify({"success": False, "error": "Nothing selected to apply."}), 400

    try:
        with open(CONFIG_TEMPLATE_FILE, "r", encoding="utf-8") as f:
            baseline = (json.load(f).get("service_phase") or {})
    except Exception:
        baseline = {}

    try:
        # Re-derived here rather than taken from the request: a proposal is a claim about
        # the archive, and the archive is what should decide it, not a posted number.
        fresh = _phase_learn_propose(_service_phase_learn_scan(), _service_phase_config(),
                                     baseline)
        updated = _phase_learn_apply(_service_phase_config(), fresh["proposals"], keys)
        config["service_phase"] = updated
        save_config(config)
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({"success": True, "applied": keys, "service_phase": updated})


@app.route("/corrections")
def corrections_page():
    """Render the corrections page for editing transcriptions in real-time"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403

    return render_template("corrections.html")


@app.route("/url-builder")
def url_builder_page():
    """Render the URL builder page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("url-builder.html")


@app.route("/server-settings")
def server_settings_page():
    """Render the server settings page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("server-settings.html")


@app.route("/translation")
def translation_settings_page():
    """Render the live translation settings page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("translation.html")


@app.route("/word-highlighting")
def word_highlighting_page():
    """Render the word highlighting page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("word-highlighting.html")


@app.route("/logs")
def logs_page():
    """Render the access-log viewer page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("logs.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """API endpoint to get current configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        return jsonify({"success": True, "config": config})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# High-frequency endpoints the web UI polls on a timer — they flood the access
# log and drown out meaningful traffic, so the log viewer hides them by default.
POLLING_LOG_PATHS = [
    "/api/logs",
    "/api/health",
    "/api/health/remote",
    "/api/transcription/status",
    "/api/server/time",
    "/api/system/requirements",
    "/api/translation/status",
    "/api/tts/status",
]


@app.route("/api/logs", methods=["GET"])
def api_logs():
    """Return recent access-log entries (newest first), with optional filters:
    ?source=web|api|socket, ?kind=http|connection|action, ?search=<substr>,
    ?hide_polling=1 (drop high-frequency dashboard polling), ?limit=<n>."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    if request_logger is None:
        return jsonify({"success": True, "entries": [], "total": 0, "enabled": False})
    try:
        limit = int(request.args.get("limit", 200))
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, 2000))
    try:
        hide_polling = request.args.get("hide_polling") in ("1", "true", "yes")
        entries = request_logger.query(
            source=(request.args.get("source") or None),
            kind=(request.args.get("kind") or None),
            search=(request.args.get("search") or None),
            exclude_paths=POLLING_LOG_PATHS if hide_polling else None,
            limit=limit,
        )
        return jsonify({
            "success": True,
            "entries": entries,
            "total": request_logger.count(),
            "enabled": _access_log_enabled(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    """Delete all access-log entries."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        if request_logger is not None:
            request_logger.clear()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def update_config():
    """API endpoint to update configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        new_config = request.get_json()

        if not new_config:
            return jsonify(
                {"success": False, "error": "No configuration data provided"}
            ), 400

        # Deep merge to preserve fields not sent from frontend (like audio.backend)
        def deep_merge(base, updates):
            """Recursively merge updates into base, preserving existing fields"""
            for key, value in updates.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        # Snapshot restart-relevant values BEFORE merging — deep_merge mutates
        # `config` in place, so reading them afterwards (or re-loading the file
        # once it's written) would always match the new values and the
        # "restart required" prompt would never fire. These settings are not
        # hot-reloadable (model/VAD are bound at worker init).
        _prev_whisper_model = config.get("whisper", {}).get("model")
        _prev_vad_enabled = config.get("vad", {}).get("enabled")
        # Snapshot the target language too, so a change made via the generic
        # /api/config editor (not the Translations tab) still reaches a paired
        # offload server. The model is not snapshotted: an offload server runs its
        # own model and this machine no longer has an opinion about it.
        _lt_prev = config.get("live_translation", {})
        _prev_lt_target = _lt_prev.get("target_language")
        # Provenance snapshot before the merge, for the same reason: deep_merge
        # mutates config in place, so a post-merge read can't see what changed.
        # This is the hot-reload path (config edited outside the Translations tab),
        # so without it a mid-session retune would leave no record.
        _meta_before = _current_session_meta()

        # Merge new config into existing config (preserves backend and other fields)
        with _config_lock:
            config = deep_merge(config, new_config)

        # Live transcription requires temperature 0: nonzero output varies between
        # re-transcription passes, so same_output_threshold finalization never
        # triggers and no rows are ever saved.
        live_temp_clamped = False
        _live_decode = config.get("whisper_decoding", {}).get("live_transcription")
        if isinstance(_live_decode, dict) and "temperature" in _live_decode:
            _temp = _live_decode["temperature"]
            if isinstance(_temp, (list, tuple)) or (_temp or 0) != 0:
                _live_decode["temperature"] = 0.0
                live_temp_clamped = True

        # If the audio device selection changed, also persist a stable name-based
        # identifier so the correct card can be re-found after ALSA index reshuffles
        # across reboots (USB vs onboard/GPU HDA enumeration order is not stable).
        try:
            new_mic = new_config.get("audio", {}).get("default_microphone")
            if new_mic:
                # Durable "the user actively chose an input" flag. Value alone
                # can't signal this: the Default device saves as "default"
                # (same as the initial value), and on macOS there's no ALSA
                # card_id to populate default_microphone_name. The flag lets the
                # setup checklist tick off even when Default is selected.
                config.setdefault("audio", {})["microphone_selected"] = True
            if new_mic and os.path.isfile(new_mic):
                # A "Test Audio File" selection is a file path, not hardware. Clear the
                # stale stable-name so it doesn't keep resolving to a real device.
                config.setdefault("audio", {})["default_microphone_name"] = ""
            elif new_mic:
                from stt.audio_capture import list_audio_devices
                markers = config.get("audio", {}).get("deprioritize_device_markers", [])
                devices = list_audio_devices(deprioritize_markers=markers)
                matched = next((d for d in devices if d.get("name") == new_mic), None)
                if matched and matched.get("card_id"):
                    config.setdefault("audio", {})["default_microphone_name"] = matched["card_id"]
        except Exception as e:
            app_logger.warning(f"Could not derive stable microphone name: {e}")

        # Write to config file
        with _config_file_lock:
            _atomic_write_json(CONFIG_FILE, config)

        # Send config update through queue for hot-reload
        try:
            config_queue.put({"type": "config_update", "config": _config_snapshot()})
        except (OSError, ValueError):
            pass  # Queue might be full or process not ready

        # Propagate live-translation model/language changes to a paired offload
        # server — these can't ride the per-request payload, so without this the
        # remote keeps its old model/precision/voice after a /api/config edit.
        try:
            _lt_now = config.get("live_translation", {})
            _new_target = _lt_now.get("target_language")
            if _new_target and _new_target != _prev_lt_target \
                    and supported_target(_new_target, _lt_now.get("translation_method", "nllb")):
                # The helper needs config to hold the OLD language so its old!=new
                # side-effects fire; deep_merge already wrote the new value.
                config["live_translation"]["target_language"] = _prev_lt_target
                _apply_translation_language_switch(_new_target)
        except Exception as e:
            print(f"[REMOTE] Could not propagate live_translation change to remote: {e}")

        # This route writes config directly rather than through save_config, so
        # the provenance choke point has to be invoked explicitly here.
        _sync_session_meta_from_config()
        _meta_changes = _session_meta_changed_keys(_meta_before, _current_session_meta())
        if any(k.startswith(("mt.offloaded", "mt.remote.")) for k in _meta_changes):
            _reprobe_remote_provenance_async()

        # Determine which settings need restart (compare pre-merge snapshot
        # against the now-merged config)
        needs_restart = False
        restart_reason = []

        if _prev_whisper_model != config.get("whisper", {}).get("model"):
            needs_restart = True
            restart_reason.append("Whisper model changed")

        if _prev_vad_enabled != config.get("vad", {}).get("enabled"):
            needs_restart = True
            restart_reason.append("VAD enabled/disabled")

        message = "Configuration updated successfully!"
        if live_temp_clamped:
            message += " Live temperature forced to 0 — nonzero values prevent segment finalization."
        if needs_restart:
            message += ' Some changes require restarting the transcription process. Use the "Restart Transcription" button.'
        else:
            message += " Changes will be applied automatically within a few seconds."

        return jsonify(
            {
                "success": True,
                "message": message,
                "config": config,
                "needs_restart": needs_restart,
                "restart_reason": restart_reason,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config/reset", methods=["POST"])
def reset_config():
    """API endpoint to reset configuration to defaults with optional backup"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        # Check if backup is requested
        request_data = request.get_json() or {}
        create_backup = request_data.get("create_backup", False)
        backup_path = None

        # Create backup if requested
        if create_backup and os.path.exists(CONFIG_FILE):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{CONFIG_FILE}.backup.{timestamp}"

            try:
                import shutil
                shutil.copy2(CONFIG_FILE, backup_path)
                print(f"[OK] Config backup created: {backup_path}")
            except Exception as backup_error:
                print(f"[WARNING] Failed to create backup: {backup_error}")
                # Continue with reset even if backup fails
                backup_path = None

        # Reset to defaults: reseed config.json from the canonical template, then
        # reload it (same path as first-run init).
        if not _restore_config_from_template("reset config to defaults"):
            return jsonify({"success": False, "error": "Default config template is missing; cannot reset."}), 500
        config = load_config()

        # Send config update through queue
        try:
            config_queue.put({"type": "config_update", "config": _config_snapshot()})
        except (OSError, ValueError):
            pass

        response_data = {
            "success": True,
            "message": 'Configuration reset to defaults. Use "Restart Transcription" button to apply changes.',
            "config": config,
            "needs_restart": True,
        }

        if backup_path:
            response_data["backup_path"] = backup_path

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# analyze_calibration_data lives in stt/calibration.py (importable, tested;
# numpy percentile replaced by a stdlib linear-interpolation equivalent).
from stt.calibration import analyze_calibration_data  # noqa: F401


@app.route("/api/calibration/start", methods=["POST"])
def start_calibration():
    """Start calibration mode to analyze environment and suggest optimal settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state, transcription_state

    try:
        # Get duration and skip_step1 from request
        request_data = request.get_json() or {}
        # Clamp to a numeric range: this flows to the transcription worker
        # (elapsed >= duration, duration * 2); a non-numeric value would crash it.
        duration = coerce_int(request_data.get("duration"), 15, lo=3, hi=120)
        skip_step1 = request_data.get("skip_step1", False)

        # Auto-start transcription if not running (calibration needs audio data)
        if not transcription_state["running"]:
            print("[CALIBRATION] Auto-starting transcription for calibration...", flush=True)
            control_queue.put({"command": "start"})
            transcription_state["status"] = "starting"
            transcription_state["message"] = "Starting for calibration..."

            # Wait for model to be fully loaded (status changes to "running")
            max_wait = 60  # Maximum 60 seconds for model loading
            waited = 0
            while transcription_state["status"] != "running" and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
                if transcription_state["status"] == "error":
                    return jsonify({"success": False, "error": transcription_state.get("error", "Model loading failed")}), 500

            if transcription_state["status"] != "running":
                return jsonify({"success": False, "error": "Timeout waiting for model to load"}), 500

            print(f"[CALIBRATION] Model ready after {waited}s", flush=True)

        # Clear shared calibration data
        calibration_data_shared["speech_samples"][:] = []
        calibration_data_shared["noise_samples"][:] = []
        calibration_data_shared["silence_durations"][:] = []
        calibration_data_shared["energy_levels"][:] = []
        calibration_data_shared["vad_probabilities"][:] = []

        # Clear step 1 data
        calibration_step1_data["noise_energies"][:] = []
        calibration_step1_data["avg_noise"] = 0.0
        calibration_step1_data["max_noise"] = 0.0

        # Initialize shared calibration state for two-step process
        calibration_state["active"] = True
        calibration_state["start_time"] = time.time()
        calibration_state["duration"] = duration
        calibration_state["speech_samples"] = 0
        calibration_state["noise_samples"] = 0
        calibration_state["silence_samples"] = 0

        if skip_step1:
            # Skip directly to Step 2
            calibration_state["step"] = 2
            calibration_state["step1_complete"] = True
            # Use current energy threshold from config as noise baseline
            # This preserves user's existing tuning
            current_threshold = config.get("audio", {}).get("energy_threshold", 300)
            # Use inverse of suggestion formula (suggested = avg_noise * 2.0)
            # So avg_noise = current_threshold / 2.0 to maintain round-trip consistency
            calibration_step1_data["avg_noise"] = float(current_threshold / 2.0)
            calibration_step1_data["max_noise"] = float(current_threshold)
            print(f"[CALIBRATION] Skipping Step 1, starting at Step 2 (speech only) - using current threshold {current_threshold} as baseline", flush=True)
        else:
            # Normal two-step process
            calibration_state["step"] = 1
            calibration_state["step1_complete"] = False
            print(f"[CALIBRATION] Started two-step calibration - {duration}s per step ({duration * 2}s total)", flush=True)

        # Send calibration command to transcription process via control queue
        control_queue.put({
            "command": "start_calibration",
            "duration": duration
        })

        return jsonify({
            "success": True,
            "message": f"Two-step calibration started ({duration}s per step)",
            "duration": duration,
            "total_duration": duration * 2
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/calibration/status", methods=["GET"])
def calibration_status():
    """Get current calibration progress (two-step process)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state

    if not calibration_state["active"]:
        return jsonify({"calibrating": False, "active": False})

    elapsed = time.time() - calibration_state["start_time"]
    duration = calibration_state.get("duration", 15)
    current_step = calibration_state.get("step", 1)
    progress = min(100, int((elapsed / duration) * 100))

    return jsonify({
        "calibrating": calibration_state["active"],
        "active": calibration_state["active"],
        "step": current_step,
        "step1_complete": calibration_state.get("step1_complete", False),
        "progress": progress,
        "elapsed": round(elapsed, 1),
        "duration": duration,
        "samples_collected": {
            "noise": calibration_state["noise_samples"],
            "speech": calibration_state["speech_samples"],
            "silence": calibration_state["silence_samples"],
        }
    })


@app.route("/api/calibration/continue", methods=["POST"])
def continue_calibration():
    """Continue calibration from Step 1 to Step 2"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state

    if not calibration_state.get("step1_complete", False):
        return jsonify({"success": False, "error": "Step 1 not complete yet"}), 400

    if calibration_state.get("step", 1) != 1:
        return jsonify({"success": False, "error": "Already on Step 2 or not calibrating"}), 400

    # Transition to step 2
    calibration_state["step"] = 2
    calibration_state["start_time"] = time.time()
    calibration_state["reset_timer"] = True  # Signal transcription process to reset local timer

    print("[CALIBRATION] Transitioning to Step 2 - user clicked Start Step 2", flush=True)

    return jsonify({"success": True, "message": "Starting Step 2"})


@app.route("/api/calibration/results", methods=["GET"])
def calibration_results():
    """Get calibration results and suggested settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global calibration_state, calibration_data_shared

    if calibration_state["active"]:
        return jsonify({"success": False, "error": "Calibration still in progress"}), 400

    # Convert shared data to regular dict for analysis
    calibration_data = {
        "speech_samples": list(calibration_data_shared["speech_samples"]),
        "noise_samples": list(calibration_data_shared["noise_samples"]),
        "silence_durations": list(calibration_data_shared["silence_durations"]),
        "energy_levels": list(calibration_data_shared["energy_levels"]),
        "vad_probabilities": list(calibration_data_shared["vad_probabilities"]),
    }

    if not calibration_data.get("energy_levels"):
        return jsonify({"success": False, "error": "No calibration data available"}), 400

    try:
        # Analyze collected data
        results = analyze_calibration_data(calibration_data)

        return jsonify({
            "success": True,
            "current_settings": {
                "energy_threshold": config.get("audio", {}).get("energy_threshold", 3500),
                "phrase_timeout": config.get("audio", {}).get("phrase_timeout", 2),
                "active_window_duration": config.get("audio", {}).get("active_window_duration", 5.0),
                "confirmation_delay": config.get("audio", {}).get("confirmation_delay", 1.5),
                "stride_length": config.get("audio", {}).get("stride_length", 2.0),
                "vad_enabled": config.get("vad", {}).get("enabled", True),
                "vad_threshold": config.get("vad", {}).get("threshold", 0.5),
            },
            "suggested_settings": results["suggestions"],
            "analysis": results["analysis"],
            "confidence": results["confidence"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/word-highlighting/words", methods=["GET"])
def get_highlighted_words():
    """API endpoint to get list of highlighted words (accessible to all for index page)"""
    # No IP whitelist check - this needs to be accessible to all users viewing the index page
    data = load_word_highlighting()
    return jsonify({"success": True, "enabled": data.get("enabled", True), "words": data.get("words", []), "disabled_colors": data.get("disabled_colors", [])})


@app.route("/api/word-highlighting/words", methods=["POST"])
def add_highlighted_word():
    """Add a new highlighted word
    Example: POST /api/word-highlighting/words {"word": "hello", "color": "#ff0000", "case_sensitive": false}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    req_data = _control_params(keep_blank=True)
    word = req_data.get("word", "").strip()
    color = req_data.get("color", "#ffff00")  # Default yellow
    case_sensitive = req_data.get("case_sensitive", False)
    is_regex = req_data.get("is_regex", False)

    if not word:
        return jsonify({"success": False, "error": "Word is required"}), 400

    # Load from separate file
    wh_data = load_word_highlighting()
    words = wh_data.get("words", [])

    # Check if word already exists
    for existing_word in words:
        if existing_word.get("word") == word:
            return jsonify({"success": False, "error": "Word already exists"}), 400

    # Add new word
    new_word = {"word": word, "color": color, "case_sensitive": case_sensitive, "is_regex": is_regex}
    words.append(new_word)
    wh_data["words"] = words

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "word": new_word})


@app.route("/api/word-highlighting/words/<int:index>", methods=["DELETE"])
def delete_highlighted_word(index):
    """Delete a highlighted word by index
    Example: DELETE /api/word-highlighting/words/0"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    wh_data = load_word_highlighting()
    words = wh_data.get("words", [])

    if index < 0 or index >= len(words):
        return jsonify({"success": False, "error": "Invalid index"}), 400

    deleted_word = words.pop(index)
    wh_data["words"] = words

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "deleted": deleted_word})


@app.route("/api/word-highlighting/words/<int:index>", methods=["PUT"])
def update_highlighted_word(index):
    """Update a highlighted word by index
    Example: PUT /api/word-highlighting/words/0 {"word": "test", "color": "#ff0000"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    req_data = _control_params(keep_blank=True)
    wh_data = load_word_highlighting()
    words = wh_data.get("words", [])

    if index < 0 or index >= len(words):
        return jsonify({"success": False, "error": "Invalid index"}), 400

    # Update word properties
    if "word" in req_data:
        words[index]["word"] = req_data["word"].strip()
    if "color" in req_data:
        words[index]["color"] = req_data["color"]
    if "case_sensitive" in req_data:
        words[index]["case_sensitive"] = req_data["case_sensitive"]
    if "is_regex" in req_data:
        words[index]["is_regex"] = req_data["is_regex"]

    wh_data["words"] = words

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "word": words[index]})


@app.route("/api/word-highlighting/toggle", methods=["POST"])
def toggle_word_highlighting():
    """API endpoint to toggle word highlighting on/off"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    wh_data = load_word_highlighting()
    current = wh_data.get("enabled", True)
    wh_data["enabled"] = not current

    # Save to separate file
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": wh_data.get("disabled_colors", [])
    })

    return jsonify({"success": True, "enabled": wh_data["enabled"]})


@app.route("/api/word-highlighting/toggle-color", methods=["POST"])
def toggle_color_group():
    """Toggle a color group on/off (disable highlighting without deleting)
    Example: POST /api/word-highlighting/toggle-color {"color": "#ff0000"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    req_data = _control_params(keep_blank=True)
    color = req_data.get("color", "").strip().lower()

    if not color:
        return jsonify({"success": False, "error": "Color is required"}), 400

    wh_data = load_word_highlighting()
    disabled_colors = wh_data.get("disabled_colors", [])

    # Toggle the color in disabled list
    if color in disabled_colors:
        disabled_colors.remove(color)
        is_disabled = False
    else:
        disabled_colors.append(color)
        is_disabled = True

    wh_data["disabled_colors"] = disabled_colors
    save_word_highlighting(wh_data)

    # Broadcast update to all connected clients
    socketio.emit("word_highlighting_update", {
        "enabled": wh_data.get("enabled", True),
        "words": wh_data.get("words", []),
        "disabled_colors": disabled_colors
    })

    return jsonify({"success": True, "color": color, "disabled": is_disabled})


# Hallucination Filter API Endpoints


@app.route("/api/hallucination-filter/toggle", methods=["POST"])
def toggle_hallucination_filter():
    """API endpoint to toggle hallucination filter on/off"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        if "hallucination_filter" not in config:
            config["hallucination_filter"] = {"enabled": True, "phrases": []}

        current = config["hallucination_filter"].get("enabled", True)
        config["hallucination_filter"]["enabled"] = not current
        save_config(config)

        return jsonify({"success": True, "enabled": config["hallucination_filter"]["enabled"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hallucination-filter/cjk-toggle", methods=["POST"])
def toggle_cjk_filter():
    """API endpoint to toggle CJK character hallucination filter on/off"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        if "hallucination_filter" not in config:
            config["hallucination_filter"] = {"enabled": True, "phrases": [], "cjk_filter_enabled": True}

        current = config["hallucination_filter"].get("cjk_filter_enabled", True)
        config["hallucination_filter"]["cjk_filter_enabled"] = not current
        save_config(config)

        return jsonify({"success": True, "cjk_filter_enabled": config["hallucination_filter"]["cjk_filter_enabled"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hallucination-filter/phrases", methods=["POST"])
def update_hallucination_phrases():
    """API endpoint to update hallucination filter phrases"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json()
        phrases = data.get("phrases", [])

        if "hallucination_filter" not in config:
            config["hallucination_filter"] = {"enabled": True, "phrases": []}

        config["hallucination_filter"]["phrases"] = phrases
        save_config(config)

        return jsonify({"success": True, "phrases": phrases})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profanity-filter/toggle", methods=["POST"])
def toggle_profanity_filter():
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    global config
    try:
        if "profanity_filter" not in config:
            config["profanity_filter"] = {"enabled": False, "words": []}
        current = config["profanity_filter"].get("enabled", False)
        config["profanity_filter"]["enabled"] = not current
        save_config(config)
        return jsonify({"success": True, "enabled": config["profanity_filter"]["enabled"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profanity-filter/words", methods=["POST"])
def update_profanity_words():
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    global config
    try:
        data = request.get_json()
        words = data.get("words", [])
        if "profanity_filter" not in config:
            config["profanity_filter"] = {"enabled": False, "words": []}
        config["profanity_filter"]["words"] = words
        save_config(config)
        return jsonify({"success": True, "words": words})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def get_url_builder_profiles():
    """Return (profiles, active) for the URL builder, migrating the legacy
    single-blob `url_builder_defaults` into a named-profile list on first access.

    profiles: list of {"name": str, "params": {url param dict}}
    active:   name of the profile that drives the root "/" redirect ("" = none)
    """
    profiles = config.get("url_builder_profiles")
    if profiles is None:
        legacy = config.get("url_builder_defaults")
        if legacy:
            profiles = [{"name": "Default", "params": legacy}]
            config["url_builder_active"] = "Default"
        else:
            profiles = []
            config.setdefault("url_builder_active", "")
        config["url_builder_profiles"] = profiles
    return profiles, config.get("url_builder_active", "")


@app.route("/api/url-builder/profiles", methods=["GET"])
def get_url_builder_profiles_endpoint():
    """List all saved URL builder profiles and which one is active (root)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    profiles, active = get_url_builder_profiles()
    return jsonify({"success": True, "profiles": profiles, "active": active})


@app.route("/api/url-builder/profiles", methods=["POST"])
def save_url_builder_profile():
    """Create or update (upsert by name) a URL builder profile."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json() or {}
        name = (data.get("name") or "").strip()
        params = data.get("params") or {}
        if not name:
            return jsonify({"success": False, "error": "Profile name is required"}), 400
        if not all(c.isalnum() or c in "-_" for c in name):
            return jsonify({"success": False, "error": "Profile names can only use letters, numbers, - and _ (no spaces) for clean /profile URLs"}), 400

        profiles, _ = get_url_builder_profiles()
        for p in profiles:
            if p["name"] == name:
                p["params"] = params
                break
        else:
            profiles.append({"name": name, "params": params})

        config["url_builder_profiles"] = profiles
        save_config(config)
        return jsonify({"success": True, "profiles": profiles, "active": config.get("url_builder_active", "")})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/url-builder/profiles/activate", methods=["POST"])
def activate_url_builder_profile():
    """Set which profile drives the root "/" redirect."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json() or {}
        name = (data.get("name") or "").strip()
        profiles, _ = get_url_builder_profiles()
        if not any(p["name"] == name for p in profiles):
            return jsonify({"success": False, "error": "Profile not found"}), 404

        config["url_builder_active"] = name
        save_config(config)
        return jsonify({"success": True, "active": name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/url-builder/profiles/<name>", methods=["DELETE"])
def delete_url_builder_profile(name):
    """Delete a profile by name; clears active if it was the root profile."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        profiles, active = get_url_builder_profiles()
        new_profiles = [p for p in profiles if p["name"] != name]
        if len(new_profiles) == len(profiles):
            return jsonify({"success": False, "error": "Profile not found"}), 404

        config["url_builder_profiles"] = new_profiles
        if active == name:
            config["url_builder_active"] = ""
        save_config(config)
        return jsonify({"success": True, "profiles": new_profiles, "active": config.get("url_builder_active", "")})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Backward-compat shims: map the old single-blob "defaults" API onto the active profile.
@app.route("/api/url-builder/defaults", methods=["POST"])
def save_url_builder_defaults():
    """Legacy endpoint: upsert a 'Default' profile from the posted params and activate it."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    try:
        data = request.get_json() or {}
        profiles, _ = get_url_builder_profiles()
        for p in profiles:
            if p["name"] == "Default":
                p["params"] = data
                break
        else:
            profiles.append({"name": "Default", "params": data})
        config["url_builder_profiles"] = profiles
        config["url_builder_active"] = "Default"
        save_config(config)
        return jsonify({"success": True, "message": "Default settings saved"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/url-builder/defaults", methods=["GET"])
def get_url_builder_defaults():
    """Legacy endpoint: return the active profile's params as the saved defaults."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    profiles, active = get_url_builder_profiles()
    params = next((p["params"] for p in profiles if p["name"] == active), {})
    return jsonify({"success": True, "defaults": params})


# File Transcription Settings Endpoints


@app.route("/api/file-transcription/settings", methods=["GET"])
def get_file_transcription_settings():
    """API endpoint to get file transcription settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    ft_config = config.get("file_transcription", {})

    return jsonify(
        {
            "success": True,
            "settings": {
                "use_gpu": ft_config.get("use_gpu", True),
                "language": ft_config.get("language", "auto"),
                "translate_enabled": ft_config.get("translate_enabled", False),
                "translate_to": ft_config.get("translate_to", "en"),
                "translation_model": ft_config.get("translation_model", "facebook/nllb-200-distilled-600M"),
                "translation_method": config.get("live_translation", {}).get("translation_method", "nllb"),
                "model": {
                    "type": ft_config.get("model", {}).get("type", "whisper"),
                    "whisper": {
                        "model": ft_config.get("model", {})
                        .get("whisper", {})
                        .get("model", "base"),
                    },
                    "huggingface": {
                        "model_id": ft_config.get("model", {})
                        .get("huggingface", {})
                        .get("model_id", "openai/whisper-base"),
                        "use_flash_attention": ft_config.get("model", {})
                        .get("huggingface", {})
                        .get("use_flash_attention", False),
                    },
                    "custom": {
                        "model_path": ft_config.get("model", {})
                        .get("custom", {})
                        .get("model_path", ""),
                        "model_type": ft_config.get("model", {})
                        .get("custom", {})
                        .get("model_type", "whisper"),
                    },
                },
            },
        }
    )


@app.route("/api/file-transcription/settings", methods=["POST"])
def save_file_transcription_settings():
    """API endpoint to save file transcription settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = _control_params(keep_blank=True)

    if "file_transcription" not in config:
        config["file_transcription"] = {}

    # Update settings
    if "use_gpu" in data:
        config["file_transcription"]["use_gpu"] = data["use_gpu"]
    if "language" in data:
        config["file_transcription"]["language"] = data["language"]
    if "translate_enabled" in data:
        config["file_transcription"]["translate_enabled"] = data["translate_enabled"]
    if "translate_to" in data:
        config["file_transcription"]["translate_to"] = data["translate_to"]
    if "translation_model" in data:
        config["file_transcription"]["translation_model"] = data["translation_model"]

    # Update model settings
    if "model" in data:
        if "model" not in config["file_transcription"]:
            config["file_transcription"]["model"] = {}

        if "type" in data["model"]:
            config["file_transcription"]["model"]["type"] = data["model"]["type"]

        if "whisper" in data["model"]:
            if "whisper" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["whisper"] = {}
            config["file_transcription"]["model"]["whisper"].update(
                data["model"]["whisper"]
            )

        if "huggingface" in data["model"]:
            if "huggingface" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["huggingface"] = {}
            config["file_transcription"]["model"]["huggingface"].update(
                data["model"]["huggingface"]
            )

        if "custom" in data["model"]:
            if "custom" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["custom"] = {}
            config["file_transcription"]["model"]["custom"].update(
                data["model"]["custom"]
            )

    # Save to file
    save_config(config)

    return jsonify(
        {"success": True, "message": "File transcription settings saved successfully"}
    )


# Timezone Settings Endpoints


@app.route("/api/timezone/settings", methods=["GET"])
def get_timezone_settings():
    """API endpoint to get timezone settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    tz_config = config.get("timezone", {"mode": "auto", "value": ""})

    return jsonify(
        {
            "success": True,
            "settings": {
                "mode": tz_config.get("mode", "auto"),
                "value": tz_config.get("value", ""),
            },
            # What the server is actually stamping rows with right now, which differs
            # from the setting until a restart.
            "effective": str(configured_timezone),
        }
    )


@app.route("/api/timezone/settings", methods=["POST"])
def save_timezone_settings():
    """API endpoint to save timezone settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = _control_params(keep_blank=True)

    if "timezone" not in config:
        config["timezone"] = {}

    mode = str(data.get("mode", config["timezone"].get("mode", "auto")) or "auto").strip().lower()
    value = str(data.get("value", config["timezone"].get("value", "")) or "").strip()

    # Reject an unloadable zone here rather than accepting it and silently falling back
    # to the system zone at the next restart, which is what used to happen.
    if mode != "auto" and not _is_known_timezone(value):
        return jsonify({
            "success": False,
            "error": (f"Unknown timezone {value!r}. Use an IANA name such as "
                      f"'America/New_York' or 'Europe/Kyiv', or set mode to 'auto' "
                      f"to follow the machine's own zone."),
        }), 400

    config["timezone"]["mode"] = mode
    config["timezone"]["value"] = "" if mode == "auto" else value

    # Save to file
    save_config(config)

    _effective, _note = _resolve_timezone(config["timezone"], _system_timezone())
    return jsonify(
        {
            "success": True,
            "effective": str(_effective),
            "message": ("Timezone saved. Restart the server to apply it — rows already "
                        "written keep the timezone they were stamped with."),
        }
    )


def _selected_model_downloaded(cfg):
    """True when the *currently selected* transcription model is present on disk.

    Mirrors the resolution in ModelFactory._load_* so this is "downloaded AND
    set" — a different downloaded model does not satisfy it. The worker loads
    exactly this model when transcription starts, so it's the real precondition.
    """
    model_cfg = cfg.get("model", {})
    mtype = model_cfg.get("type", "whisper")
    if mtype == "whisper":
        name = model_cfg.get("whisper", {}).get("model", "small")
        backend = model_cfg.get("backend", "whisper")
        if backend == "faster-whisper":
            return os.path.isdir(os.path.join(MODELS_DIR, f"faster-whisper-{name}"))
        # OpenAI whisper: new ./models location or legacy ~/.cache/whisper/{name}.pt
        if os.path.isdir(os.path.join(MODELS_DIR, f"whisper-{name}")):
            return True
        return os.path.exists(os.path.expanduser(f"~/.cache/whisper/{name}.pt"))
    if mtype == "huggingface":
        model_id = model_cfg.get("huggingface", {}).get("model_id", "openai/whisper-tiny")
        return os.path.isdir(os.path.join(MODELS_DIR, model_id.replace("/", "--")))
    if mtype == "custom":
        return os.path.exists(model_cfg.get("custom", {}).get("model_path", ""))
    return False


def _mic_explicitly_selected(cfg):
    """True when an audio input is configured.

    The system default ('default') counts: on a single-mic machine it is the
    only option in the dropdown, so requiring an explicit non-default pick was a
    dead-end (you literally can't choose anything else). Only a truly empty
    value — or having none of the audio keys — reads as 'no mic configured'.
    """
    audio = cfg.get("audio", {})
    return bool(audio.get("default_microphone")) or \
        bool(audio.get("default_microphone_name")) or \
        bool(audio.get("microphone_selected"))


def _setup_status():
    """Readiness of the two prerequisites for Start (drives the checklist +
    the greyed-out Start button on the web UI and the watchdog GUI)."""
    model_ready = _selected_model_downloaded(config)
    mic_ready = _mic_explicitly_selected(config)
    return {
        "model_ready": model_ready,
        "mic_ready": mic_ready,
        "ready": model_ready and mic_ready,
        "model_hint": "" if model_ready else "Download the selected model in the Model Manager.",
        "mic_hint": "" if mic_ready else "Select a microphone in Live Settings.",
    }


def _disk_usage_percent():
    """Disk-usage percent of the app volume for the live page's disk-full
    banner. Served on the *public* /api/transcription/status poll so any viewer
    (including non-whitelisted displays) gets the warning without calling the
    IP-gated /api/disk-space endpoint. Percent only — no paths/bytes, which stay
    behind auth. Returns None on any failure so the banner just stays hidden."""
    try:
        du = shutil.disk_usage(APP_DIR)
        if du.total > 0:
            return round(du.used / du.total * 100, 2)
    except Exception:
        pass
    return None


@app.route("/api/system/requirements", methods=["GET"])
def get_system_requirements():
    """Whether this machine can run the configured models (drives the header banner).

    Recomputed per request so the banner follows config changes; only the
    hardware probe is cached.
    """
    warns = _check_system_requirements()
    # Prepend first-run setup gaps so the header banner nudges users on every
    # page until transcription can actually start (model downloaded + mic chosen).
    setup = _setup_status()
    setup_warns = []
    if not setup["model_ready"]:
        setup_warns.append("No transcription model downloaded — open Model Manager to download the selected model.")
    if not setup["mic_ready"]:
        setup_warns.append("No microphone selected — choose one in Live Settings.")
    warns = setup_warns + warns
    # Runtime check, not a hardware estimate: the loaded NLLB model actually
    # landed on CPU despite use_gpu — translations run seconds-per-sentence and
    # nothing else surfaces it (field reports arrive as "translation got slow").
    trans_cfg = config.get("live_translation", {})
    if (_live_translation_device == "cpu" and trans_cfg.get("use_gpu", True)
            and trans_cfg.get("translation_method", "nllb") == "nllb"):
        warns = [*warns, "Translation model is running on CPU (GPU requested but unavailable) — expect slow translations. Check GPU drivers / torch install."]
    details = {}
    try:
        hw = _probe_hardware()
        need = _estimate_memory_requirements(config, gpu_available=bool(hw.get("has_cuda") or hw.get("apple_silicon")),
                                             unified=bool(hw.get("apple_silicon")))
        details = {
            "ram_gb_needed": round(need["ram_gb"], 1),
            "vram_gb_needed": round(need["vram_gb"], 1),
            "disk_gb_needed": round(need["disk_gb"], 1),
            "ram_gb_found": round((hw.get("ram_bytes") or 0) / 1024**3, 1),
            "vram_gb_found": round(hw["vram_bytes"] / 1024**3, 1) if hw.get("vram_bytes") else None,
            "apple_silicon": bool(hw.get("apple_silicon")),
            "has_cuda": bool(hw.get("has_cuda")),
        }
    except Exception:
        pass
    return jsonify({
        "meets_requirements": not warns,
        "warnings": warns,
        "details": details,
    })


@app.route("/api/server/time", methods=["GET"])
def get_server_time():
    """API endpoint to get server's current time for timezone comparison"""
    now = datetime.now()
    return jsonify({
        "success": True,
        "timestamp": now.timestamp(),
        "formatted": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
        "timezone": str(now.astimezone().tzinfo),
        "iso": now.isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "uptime_seconds": round(time.time() - SERVER_START_TIME),
        "version": SERVER_VERSION,
        "commit": SERVER_COMMIT,
        "describe": SERVER_DESCRIBE,
        "display_version": SERVER_DISPLAY_VERSION
    })


def _safe(getter, default=None):
    """Call a status getter and return a plain JSON-serializable value.

    Some "getters" are actually Flask route handlers that return a ``Response``
    (from ``jsonify``) or a ``(Response, status)`` tuple rather than a dict, so
    unwrap those to their JSON body — embedding a Response in the health payload
    would make the final ``jsonify`` fail. Any error is swallowed so one bad
    section never fails the whole /api/health aggregation."""
    try:
        result = getter()
    except Exception:
        return default
    try:
        if isinstance(result, tuple):
            result = result[0]
        if hasattr(result, "get_json"):
            return result.get_json(silent=True) or default
        return result
    except Exception:
        return default


@app.route("/health")
def health_page():
    """Render the health / metrics dashboard page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("health.html")


@app.route("/api/health", methods=["GET"])
def get_health():
    """Aggregate live health/metrics from the existing status getters plus the
    new performance and system-resource instrumentation into one payload the
    dashboard polls. Each numeric metric carries a server-computed status so the
    client only paints colours."""
    # A paired translation peer (Machine A) may read this box's health so it can
    # show it as "Machine B" — the same trust the /api/translate endpoints use.
    if not (check_ip_whitelist() or _is_trusted_translation_client(request.remote_addr)):
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        ts = _ts_snapshot()
        hw = _safe(_probe_hardware, {}) or {}
        sysres = _safe(_metrics.sample_system_resources, {}) or {}

        # --- transcription ---
        # Live-perf metrics (RTF, inference, throughput, queue) are only
        # meaningful while transcribing. When stopped, report them as null /
        # unknown instead of the last session's stale EMA, so the dots don't
        # stay green on an idle box.
        running = ts.get("running", False)
        rtf = ts.get("rtf_ema") if running else None
        _start = ts.get("start_time") or 0
        _session_seconds = round(time.time() - _start) if (running and _start) else None
        transcription = {
            "session_id": ts.get("session_id"),
            "db_name": ts.get("db_name"),
            "session_seconds": _session_seconds,
            "session_display": _metrics.format_uptime(_session_seconds) if _session_seconds is not None else None,
            "rows_saved": ts.get("rows_saved", 0),
            "running": running,
            "status": ts.get("status", "stopped"),
            "message": ts.get("message", ""),
            "loaded_model": ts.get("loaded_model") or None,
            "loaded_model_device": ts.get("loaded_model_device"),
            "model_load_ms": ts.get("model_load_ms"),
            "audio_level": ts.get("audio_level", 0) if running else 0,
            "audio_db": ts.get("audio_db"),
            "audio_type": ts.get("audio_type") if running else None,
            "detection_mode": ts.get("detection_mode") if running else None,
            "audio_tag": ts.get("audio_tag") if running else None,      # PANNs top CNN14 class
            "music_prob": ts.get("music_prob") if running else None,    # 0-1 music score
            "rtf_ema": rtf,
            "rtf_status": _metrics.rtf_status(rtf) if running else "unknown",
            "infer_ms_ema": ts.get("infer_ms_ema") if running else None,
            "segments_total": ts.get("segments_total", 0),
            "segments_per_min": ts.get("segments_per_min") if running else None,
            "queue_depth": ts.get("queue_depth") if running else None,
        }

        # --- system resources (live used vs. static totals) ---
        ram_used = sysres.get("ram_used_bytes")
        ram_total = sysres.get("ram_total_bytes") or hw.get("ram_bytes")
        vram_used = sysres.get("vram_used_bytes")
        vram_total = hw.get("vram_bytes")
        cpu_pct = sysres.get("cpu_pct")
        # Disk: total/used/free so the UI can show GB + percent with a status colour.
        _disk_total = _disk_used = _disk_free = None
        try:
            import shutil as _shutil
            _du = _shutil.disk_usage(APP_DIR)
            _disk_total, _disk_used, _disk_free = _du.total, _du.used, _du.free
        except Exception:
            _disk_free = hw.get("disk_free_bytes")
        system = {
            "cpu_pct": cpu_pct,
            "cpu_cores": hw.get("cpu_cores"),
            "cpu_status": _metrics.fraction_status(cpu_pct, 100.0),
            "ram_used_bytes": ram_used,
            "ram_total_bytes": ram_total,
            "ram_status": _metrics.fraction_status(ram_used, ram_total),
            "gpu_util_pct": sysres.get("gpu_util_pct"),
            "gpu_kind": sysres.get("gpu_kind"),  # "cuda" | "mps" | None
            "has_cuda": hw.get("has_cuda", False),
            "vram_used_bytes": vram_used,
            "vram_total_bytes": vram_total,
            "vram_status": _metrics.fraction_status(vram_used, vram_total),
            "disk_free_bytes": _disk_free,
            "disk_used_bytes": _disk_used,
            "disk_total_bytes": _disk_total,
            "disk_percent": round(100 * _disk_used / _disk_total, 1) if (_disk_used and _disk_total) else None,
            "disk_status": _metrics.fraction_status(_disk_used, _disk_total),
            "swap_used_bytes": sysres.get("swap_used_bytes"),
            "swap_total_bytes": sysres.get("swap_total_bytes"),
            "swap_status": _metrics.fraction_status(
                sysres.get("swap_used_bytes"), sysres.get("swap_total_bytes"),
                degraded_above=0.25, error_above=0.6),
            "proc_rss_bytes": sysres.get("proc_rss_bytes"),
            "apple_silicon": hw.get("apple_silicon", False),
        }

        # --- requests (access log) ---
        requests_stats = None
        if _access_log_enabled() and request_logger is not None:
            requests_stats = _safe(lambda: request_logger.stats(300))
            if requests_stats is not None:
                # Status reflects server faults (5xx) only — 4xx are client/auth
                # errors and don't mean the server is unhealthy. Any 5xx in the
                # window is worth amber; a sustained rate goes red.
                requests_stats["error_status"] = _metrics.fraction_status(
                    requests_stats.get("server_error_count"),
                    max(1, requests_stats.get("requests", 0)),
                    degraded_above=0.0, error_above=0.05,
                )

        # --- audio detectors (music/speech tagging + VAD) ---
        _panns = _safe(panns_status, {}) or {}
        _vad = _safe(silero_vad_status, {}) or {}
        detectors = {
            "detection_mode": ts.get("detection_mode"),
            "panns_available": _panns.get("downloaded"),
            "panns_message": _panns.get("message"),
            "vad_available": _vad.get("downloaded"),
            "vad_message": _vad.get("message"),
        }

        health = {
            "server": {
                "uptime_seconds": round(time.time() - SERVER_START_TIME),
                "uptime_display": _metrics.format_uptime(time.time() - SERVER_START_TIME),
                "version": SERVER_VERSION,
                "commit": SERVER_COMMIT,
                "display_version": SERVER_DISPLAY_VERSION,
            },
            "transcription": transcription,
            "detectors": detectors,
            "translation": _safe(get_translation_status, {}),
            "tts": _safe(get_tts_status, {}),
            "file_mover": _safe(get_file_mover_runtime, {}),
            "system": system,
            "requests": requests_stats,
        }
        return jsonify({"success": True, "health": health})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health/remote", methods=["GET"])
def get_health_remote():
    """Proxy the remote translation machine's ("Machine B") /api/health so the
    local dashboard can show its metrics too. Kept separate from /api/health so
    a slow or unreachable remote never stalls the local poll. IP-trusted on the
    remote side, so no auth token is forwarded."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    endpoint = _safe(_get_remote_endpoint_safe)
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured",
                        "configured": False}), 200
    import requests as _req
    try:
        r = _req.get(endpoint.rstrip("/") + "/api/health", timeout=3)
        try:
            data = r.json()
        except ValueError:
            return jsonify({"success": False, "configured": True, "endpoint": endpoint,
                            "error": "Invalid response from remote"}), 200
        # Tag the endpoint on so the UI can label the section.
        if isinstance(data, dict):
            data["endpoint"] = endpoint
            data["configured"] = True
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"success": False, "configured": True, "endpoint": endpoint,
                        "reachable": False, "error": str(e)}), 200


# Live Translation API Endpoints


@app.route("/api/translation/settings", methods=["GET"])
def get_translation_settings():
    """API endpoint to get live translation settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    trans_config = config.get("live_translation", {
        "enabled": False,
        "target_language": "en",
        "source_language": "auto",
        "translate_in_progress": False,
        "display_mode": "translated_only",
        "translation_model": "facebook/nllb-200-distilled-600M"
    })

    translation_count = config.get("corrections", {}).get("n_best_alternatives", {}).get("translation_count", 3)

    # Each engine supports a different language set (NLLB ~200, MADLAD ~400), so
    # expose both maps and let the picker swap to the active method's list.
    active_method = trans_config.get("translation_method", "nllb")
    languages_by_method = {
        "nllb": languages_for_method("nllb"),
        "madlad": languages_for_method("madlad"),
        "llm": languages_for_method("llm"),
    }

    return jsonify({
        "success": True,
        "settings": trans_config,
        "translation_count": translation_count,
        # Back-compat: the active engine's list (older callers read this directly).
        "available_languages": languages_by_method.get(active_method, languages_by_method["nllb"]),
        "available_languages_by_method": languages_by_method,
        "model_loaded": is_live_translation_model_loaded(),
        "cache_size": get_translation_cache().get_size()
    })


@app.route("/api/translation/settings", methods=["POST"])
def save_translation_settings():
    """API endpoint to save live translation settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = _control_params(keep_blank=True)

    if "live_translation" not in config:
        config["live_translation"] = {}

    # Provenance snapshot before any mutation, diffed at the end of this handler.
    # Diffing the whole mapping covers every recorded setting (engine, model,
    # context_window, generation params, CT2/fp16/GPU) without enumerating them
    # here, so a setting added to build_session_meta is tracked automatically.
    _meta_before = _current_session_meta()

    # Track if we need to handle model loading/unloading
    was_enabled = config.get("live_translation", {}).get("enabled", False)
    old_target_lang = config.get("live_translation", {}).get("target_language", "en")
    old_model = config.get("live_translation", {}).get("translation_model", "")
    old_use_gpu = config.get("live_translation", {}).get("use_gpu", True)
    old_method = config.get("live_translation", {}).get("translation_method", "nllb")
    old_use_fp16 = config.get("live_translation", {}).get("use_fp16", False)
    old_use_ct2 = config.get("live_translation", {}).get("use_ctranslate2", False)

    # Update settings. NOTE: target_language is deliberately NOT merged here — it
    # is applied below via _apply_translation_language_switch, which needs config
    # to still hold the OLD language so its old!=new side-effects (TTS, remote
    # push, client reset) fire.
    for key in ["enabled", "source_language", "translate_in_progress",
                "display_mode", "translation_model", "use_gpu", "translation_method"]:
        if key in data:
            config["live_translation"][key] = data[key]

    # fp16 is applied at model load, so a change requires a reload (handled below).
    if "use_fp16" in data:
        config["live_translation"]["use_fp16"] = bool(data["use_fp16"])

    # CTranslate2 backend + quantization are applied at model load (reload below).
    if "use_ctranslate2" in data:
        config["live_translation"]["use_ctranslate2"] = bool(data["use_ctranslate2"])
    if "ct2_compute_type" in data:
        config["live_translation"]["ct2_compute_type"] = str(data["ct2_compute_type"])

    # Clamp to match the UI slider (1-5); larger windows approach NLLB's 1024-token truncation
    if "context_window" in data:
        try:
            config["live_translation"]["context_window"] = max(1, min(5, int(data["context_window"])))
        except (TypeError, ValueError):
            pass

    # Save generation parameters
    if "generation_params" in data:
        gp = data["generation_params"]
        config["live_translation"]["generation_params"] = {
            "num_beams": max(1, min(20, int(gp.get("num_beams", 2)))),
            "length_penalty": max(0.1, min(3.0, float(gp.get("length_penalty", 1.0)))),
            "no_repeat_ngram_size": max(0, min(10, int(gp.get("no_repeat_ngram_size", 0)))),
            "repetition_penalty": max(0.5, min(3.0, float(gp.get("repetition_penalty", 1.0)))),
        }

    # Save remote translation endpoint config (preserve keys the UI doesn't send,
    # e.g. server_cache_*).
    if "remote" in data:
        _remote = dict(config.get("live_translation", {}).get("remote", {}) or {})
        _remote["enabled"] = bool(data["remote"].get("enabled", False))
        _remote["endpoint"] = str(data["remote"].get("endpoint", ""))
        _remote["fallback"] = "local" if data["remote"].get("fallback") == "local" else "skip"
        # Which model Machine B should run (chosen from B's downloaded models).
        _remote["model"] = str(data["remote"].get("model", _remote.get("model", "")) or "")
        # Whether to push glossary/dictionary edits to B immediately.
        _remote["sync_dictionary_on_edit"] = bool(
            data["remote"].get("sync_dictionary_on_edit", _remote.get("sync_dictionary_on_edit", True)))
        config["live_translation"]["remote"] = _remote

    # Save LLM translation config (same preserve-what-the-UI-didn't-send shape as
    # remote above, so a form that omits the GGUF fields can't wipe them).
    if "llm" in data:
        _llm = dict(config.get("live_translation", {}).get("llm", {}) or {})
        _sent = data["llm"] if isinstance(data["llm"], dict) else {}
        _provider = str(_sent.get("provider", _llm.get("provider", "endpoint")) or "endpoint").lower()
        _llm["provider"] = "local" if _provider == "local" else "endpoint"
        for _key in ("endpoint", "model", "api_key", "system_prompt",
                     "gguf_repo", "gguf_file", "gguf_path"):
            if _key in _sent:
                _llm[_key] = str(_sent.get(_key) or "")
        # n_gpu_layers is "auto" or an int, so it is kept as a string-or-number
        # rather than coerced — resolve_gpu_layers() interprets it at load time.
        if "n_gpu_layers" in _sent:
            _llm["n_gpu_layers"] = _sent["n_gpu_layers"] if _sent["n_gpu_layers"] not in ("", None) else "auto"
        if "max_tokens" in _sent:
            _llm["max_tokens"] = coerce_int(_sent.get("max_tokens"), 160, lo=16, hi=1024)
        if "n_ctx" in _sent:
            _llm["n_ctx"] = coerce_int(_sent.get("n_ctx"), 2048, lo=_LLM_MIN_N_CTX, hi=32768)
        if "timeout_ms" in _sent:
            _llm["timeout_ms"] = coerce_int(_sent.get("timeout_ms"), 8000, lo=500, hi=120000)
        if "warmup_timeout_ms" in _sent:
            _llm["warmup_timeout_ms"] = coerce_int(_sent.get("warmup_timeout_ms"), 180000, lo=1000, hi=900000)
        if "keep_alive" in _sent:
            _llm["keep_alive"] = _sent["keep_alive"]
        _llm_before = dict(config.get("live_translation", {}).get("llm", {}) or {})
        config["live_translation"]["llm"] = _llm
        # Only a change that affects which weights are resident forces a reload —
        # the in-process load costs seconds (minutes on a cold CUDA box), and the
        # system prompt is applied per caption, so re-reading it is free.
        _reload_keys = ("provider", "gguf_repo", "gguf_file", "gguf_path",
                        "n_gpu_layers", "n_ctx")
        if any(_llm_before.get(k) != _llm.get(k) for k in _reload_keys):
            try:
                unload_local_llm()
            except Exception:
                pass  # a failed unload must not fail the save

    # Save translation alternatives count to corrections config
    if "translation_count" in data:
        config.setdefault("corrections", {}).setdefault("n_best_alternatives", {})["translation_count"] = coerce_int(data.get("translation_count"), 3, lo=1, hi=10)

    save_config(config)

    # Push to config queue for hot-reload (so transcription subprocess picks up translation_method changes)
    if config_queue:
        try:
            config_queue.put({"type": "config_update", "config": _config_snapshot()})
        except (OSError, ValueError):
            pass

    # Handle model loading/unloading based on enabled state
    now_enabled = config["live_translation"].get("enabled", False)
    new_model = config["live_translation"].get("translation_model", "")
    new_use_gpu = config["live_translation"].get("use_gpu", True)
    new_method = config["live_translation"].get("translation_method", "nllb")
    new_use_fp16 = config["live_translation"].get("use_fp16", False)
    new_use_ct2 = config["live_translation"].get("use_ctranslate2", False)

    # fp16 and the CT2 backend are applied at model load, so a change needs a reload.
    model_changed = (old_model != new_model or old_use_gpu != new_use_gpu
                     or old_use_fp16 != new_use_fp16 or old_use_ct2 != new_use_ct2)
    method_changed = old_method != new_method
    using_whisper = new_method in ("whisper_translate", "whisper_forced_lang")

    if not now_enabled and was_enabled:
        # Translation just disabled - unload model
        threading.Thread(target=unload_live_translation_model, daemon=True).start()
    elif using_whisper and not (old_method in ("whisper_translate", "whisper_forced_lang")):
        # Switched to Whisper method — unload NLLB model (not needed)
        threading.Thread(target=unload_live_translation_model, daemon=True).start()
    elif now_enabled and not using_whisper and (not was_enabled or model_changed or method_changed):
        # Using NLLB: translation just enabled, model/GPU/method changed - reload model
        # Skip eager loading if this machine serves remote clients (Machine B) —
        # model will be loaded when Machine A starts transcription via /api/translate/preload
        _remote = config.get("live_translation", {}).get("remote", {}) or {}
        _offload_no_local = (bool(_remote.get("enabled") and _remote.get("endpoint"))
                             and _remote.get("fallback", "skip") == "skip")
        if _trusted_translation_clients:
            if was_enabled and (model_changed or method_changed):
                # Model changed — unload old one, new one loads on next request/preload
                threading.Thread(target=unload_live_translation_model, daemon=True).start()
        elif _offload_no_local:
            # Offloading with fallback=skip: this machine never runs a local NLLB
            # model (Machine B does the translating), so don't download/load one —
            # this is what avoids downloading the model on both machines.
            if was_enabled:
                threading.Thread(target=unload_live_translation_model, daemon=True).start()
        else:
            def reload_translation_model():
                if was_enabled:
                    unload_live_translation_model()
                use_gpu = config.get("live_translation", {}).get("use_gpu", config.get("performance", {}).get("use_gpu", True))
                model_id = config["live_translation"].get("translation_model")
                get_live_translation_model(use_gpu, model_id)
            threading.Thread(target=reload_translation_model, daemon=True).start()

    # Clear cache on model or method change (stale tokenizer/model).
    # Don't clear on language change — stale-lang fallback keeps old translations.
    if model_changed or method_changed:
        get_translation_cache().clear()

    # Target-language change: route through the shared helper (same as
    # /api/language) so TTS voice, the client display reset, AND a paired remote's
    # config all update — not just the translated text. Config still holds the OLD
    # language here (excluded from the merge loop above), so old!=new fires.
    if "target_language" in data and data["target_language"] != old_target_lang \
            and supported_target(data["target_language"], new_method):
        _apply_translation_language_switch(data["target_language"])

    # Settings changes are recorded by save_config via
    # _sync_session_meta_from_config. What that can't know is what the REMOTE is
    # now running, so an offload change still triggers a re-probe.
    _meta_changes = _session_meta_changed_keys(_meta_before, _current_session_meta())
    if any(k.startswith(("mt.offloaded", "mt.remote.")) for k in _meta_changes):
        _reprobe_remote_provenance_async()

    return jsonify({
        "success": True,
        "message": "Translation settings saved. Changes take effect immediately."
    })


def _dictionary_sync_payload():
    """Payload for /api/translate/sync-dictionary: the custom dictionary plus the
    glossary-enabled flag (top-level, so old receivers that only read
    ``dictionary.glossary`` ignore it) — so the remote applies both."""
    return {
        "dictionary": load_custom_dictionary(),
        "nllb_glossary_enabled": bool(config.get("custom_dictionary", {}).get("nllb_glossary_enabled", False)),
    }


def _propagate_dictionary_to_remote():
    """Push the custom dictionary (+ glossary-enabled flag) to a paired remote so
    a mid-session glossary edit takes effect there too. Best-effort, off-thread;
    no-op when offload isn't enabled or ``sync_dictionary_on_edit`` is off."""
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if not (remote_cfg.get("enabled") and remote_cfg.get("endpoint")):
        return
    if not remote_cfg.get("sync_dictionary_on_edit", True):
        return
    payload = _dictionary_sync_payload()  # build on this thread (config is stable)

    def _push():
        try:
            ep = _get_remote_endpoint_safe()
            if not ep:
                return
            _get_remote_http_session().post(
                ep.rstrip("/") + "/api/translate/sync-dictionary",
                json=payload, timeout=5,
            )
        except Exception as e:
            print(f"[REMOTE] Could not propagate dictionary to remote: {e}")

    threading.Thread(target=_push, daemon=True).start()


def _remote_heartbeat_loop():
    """On a machine that offloads translation, ping the paired remote server
    every ~20s while transcription is running, so the server knows this machine
    is live even during silent stretches with no translate traffic. Self-gates
    on offload config, so it's a cheap no-op on non-offloading machines."""
    while True:
        try:
            time.sleep(20)
            if not _ts_get("running", False):
                continue
            remote_cfg = config.get("live_translation", {}).get("remote", {})
            if not (remote_cfg.get("enabled") and remote_cfg.get("endpoint")):
                continue
            ep = _get_remote_endpoint_safe()
            if not ep:
                continue
            # Tell B which port this machine's own UI is on. B sees only our IP
            # (request.remote_addr), so without this it cannot offer a link back
            # to us and would have to guess port 80.
            _get_remote_http_session().post(
                ep.rstrip("/") + "/api/translate/heartbeat",
                json={"port": coerce_int(config.get("web_server", {}).get("port"), 8080,
                                         lo=1, hi=65535)},
                timeout=5)
        except Exception:
            pass  # best-effort; never let the heartbeat crash


def _apply_transcription_language_switch(new_language):
    """Set the live transcription language and hot-reload the transcription
    subprocess. Shared by /api/transcription/language and /api/language so the
    two can't drift. Returns the previous language."""
    global config
    old_language = config.get("audio", {}).get("language", "auto")
    if "audio" not in config:
        config["audio"] = {}
    config["audio"]["language"] = new_language
    save_config(config)
    if config_queue:
        try:
            config_queue.put({"type": "config_update", "config": _config_snapshot()})
        except (OSError, ValueError):
            pass
    # The switch is recorded in the session's provenance by save_config above,
    # via _sync_session_meta_from_config.
    #
    # Say plainly when nothing changed. A fixed-language button pressed while
    # already in that language is a legitimate no-op, and logging it as
    # "Hot-switched language: ru -> ru" reads as a switch that happened — which is
    # indistinguishable from a press that failed when you're reading the log
    # afterwards trying to tell those two apart.
    if old_language == new_language:
        print(f"[TRANSCRIPTION] Language already {new_language}, no change")
    else:
        print(f"[TRANSCRIPTION] Hot-switched language: {old_language} -> {new_language}")
    return old_language


def _apply_translation_language_switch(new_language):
    """Switch the live-translation target language and fan out every side effect:
    TTS voice/model swap, transcription-subprocess reload, remote Machine B
    propagation, and the language_switched client event. Shared by
    /api/translation/language and /api/language so both stay in lock-step —
    critically, so a paired Machine B is always notified. Caller must have
    validated ``new_language`` is in NLLB_LANG_CODES.

    Returns (old_language, new_tts_voice, backend)."""
    global config
    old_language = config.get("live_translation", {}).get("target_language", "en")
    if "live_translation" not in config:
        config["live_translation"] = {}
    config["live_translation"]["target_language"] = new_language

    # Auto-switch TTS voice/model to match the new language
    new_tts_voice = None
    tts_section = config["live_translation"].setdefault("tts", {})
    backend = tts_section.get("backend", "edge")

    if old_language != new_language:
        if backend == "edge":
            prefs = tts_section.setdefault("edge_voice_preferences", {})
            current_voice = tts_section.get("edge_voice", "")
            if current_voice:
                prefs[old_language] = current_voice
            new_tts_voice = prefs.get(new_language) or _pick_default_edge_voice(new_language)
            if new_tts_voice:
                tts_section["edge_voice"] = new_tts_voice
                print(f"[TTS] Auto-switched edge voice: {current_voice} -> {new_tts_voice}")

        elif backend == "piper":
            prefs = tts_section.setdefault("piper_model_preferences", {})
            current_model = tts_section.get("piper_model", "")
            if current_model:
                prefs[old_language] = current_model
            new_tts_voice = prefs.get(new_language) or _pick_default_piper_model(new_language)
            if new_tts_voice:
                tts_section["piper_model"] = new_tts_voice
                print(f"[TTS] Auto-switched piper model: {current_model} -> {new_tts_voice}")
                # Reload piper model in background
                def _reload():
                    unload_tts_model()
                    get_tts_model(model_name=new_tts_voice)
                threading.Thread(target=_reload, daemon=True).start()

    save_config(config)

    # Push to config queue so transcription subprocess picks up the new target language
    if config_queue:
        try:
            config_queue.put({"type": "config_update", "config": _config_snapshot()})
        except (OSError, ValueError):
            pass

    # Propagate language change to remote Machine B so its TTS/display/config also updates
    if old_language != new_language:
        _remote_ep = _get_remote_endpoint_safe()
        if _remote_ep:
            try:
                import requests as _req
                _req.post(
                    _remote_ep.rstrip("/") + "/api/translate/language",
                    json={"target_language": new_language},
                    timeout=5,
                )
            except Exception as _e:
                print(f"[HOT-SWITCH] Could not notify remote server of language change: {_e}")

    # The switch is recorded in the session's provenance by save_config above,
    # so a transcript that changed target language mid-service doesn't read as
    # though it were that language all along. The per-row translation_language
    # column shows which rows went where; provenance shows when and from what.

    # Don't clear cache — old segments keep their cached translations (stale-lang fallback).
    # Only new segments will be translated to the new language.
    language_name = TRANSLATION_LANGUAGES.get(new_language, new_language)
    # Distinguish a real switch from a no-op, for the same reason as the
    # transcription helper: otherwise the log can't tell "already there" from
    # "the press never took effect".
    if old_language == new_language:
        print(f"[LIVE-TRANSLATION] Language already {new_language} ({language_name}), no change")
    else:
        print(f"[LIVE-TRANSLATION] Hot-switched language: {old_language} -> {new_language} ({language_name})")

    # Notify clients so they can cleanly reset their display
    socketio.emit("language_switched", {
        "old_language": old_language,
        "new_language": new_language,
        "language_name": language_name,
    })

    return old_language, new_tts_voice, backend


@app.route("/api/translation/language", methods=["POST"])
def hot_switch_translation_language():
    """Hot-switch target language without restart - clears cache and re-translates.

    Accepts JSON body, form body, or query string (see stt/http_params.py)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = _control_params()
    new_language = data.get("target_language")

    if not new_language:
        _note_access_detail(
            f"rejected: no target_language field (json={len(request.get_json(silent=True) or {})} "
            f"form={len(request.form)} query={len(request.args)})"
        )
        return jsonify({"success": False, "error": "target_language required"}), 400

    _active_method = config.get("live_translation", {}).get("translation_method", "nllb")
    if not supported_target(new_language, _active_method):
        _note_access_detail(f"rejected: target_language={new_language} unsupported by {_active_method}")
        return jsonify({"success": False, "error": f"Invalid language: {new_language}"}), 400

    old_language, new_tts_voice, backend = _apply_translation_language_switch(new_language)
    _note_access_detail(
        f"translation {old_language}->{new_language} "
        f"{'changed' if old_language != new_language else 'unchanged'}")

    language_name = TRANSLATION_LANGUAGES.get(new_language, new_language)
    result = {
        "success": True,
        "changed": old_language != new_language,
        "message": (f"Switched to {language_name}. Translations will update shortly."
                    if old_language != new_language
                    else f"Already set to {language_name}; nothing changed."),
        "old_language": old_language,
        "new_language": new_language,
        "language_name": language_name,
    }
    if new_tts_voice:
        result["tts_voice"] = new_tts_voice
        result["tts_backend"] = backend
    return jsonify(result)


@app.route("/api/translation/status", methods=["GET"])
def get_translation_status():
    """Check if translation is active and model loaded"""
    trans_config = config.get("live_translation", {})
    caller_ip = request.remote_addr
    is_local = check_ip_whitelist()
    is_paired = _is_trusted_translation_client(caller_ip)

    # Collect active remote clients (last seen within 60s) and prune stale ones
    with _translation_clients_lock:
        now = time.time()
        active = {ip: ts for ip, ts in _translation_clients.items() if now - ts < 60}
        _translation_clients.clear()
        _translation_clients.update(active)

    remote_cfg = trans_config.get("remote", {})
    remote_active = bool(remote_cfg.get("enabled") and remote_cfg.get("endpoint"))

    # Show Whisper model when using Whisper translation methods
    _method = trans_config.get("translation_method", "nllb")
    _using_whisper = _method in ("whisper_translate", "whisper_forced_lang")
    # An LLM session must not be described by the NMT fields. A paired machine reads
    # this endpoint to record what actually translated its captions, and reporting
    # the standby NMT model there put "google/madlad400-3b-mt, device: cpu" into a
    # transcript whose captions came from a GGUF on the GPU.
    _llm_cfg = trans_config.get("llm") or {}
    _using_llm = _method == "llm"
    _llm_local = _uses_local_llm(trans_config)
    if _using_whisper:
        _status_model = "whisper/" + config.get("model", {}).get("whisper", {}).get("model", "whisper")
    elif _using_llm:
        _status_model = ((_llm_cfg.get("gguf_file") or _llm_cfg.get("gguf_repo"))
                         if _llm_local else _llm_cfg.get("model")) or "llm"
    else:
        _status_model = trans_config.get("translation_model", "facebook/nllb-200-distilled-600M")

    # Effective precision the local model actually loaded at (fp16/fp32), probed
    # off the live model; null when no local model is loaded (e.g. this box only
    # offloads). Distinct from the use_fp16 config flag (intent vs reality).
    _model_dtype = None
    try:
        if _live_translation_model is not None:
            _model_dtype = str(next(_live_translation_model.parameters()).dtype).replace("torch.", "")
    except Exception:
        pass

    result = {
        "success": True,
        "enabled": trans_config.get("enabled", False),
        "target_language": trans_config.get("target_language", "en"),
        "target_language_name": TRANSLATION_LANGUAGES.get(
            trans_config.get("target_language", "en"), "English"
        ),
        "model_loaded": (_local_llm is not None if _llm_local else True)
        if (remote_active or _using_whisper or _using_llm) else is_live_translation_model_loaded(),
        "model_loading": False if (remote_active or _using_whisper or _using_llm) else is_live_translation_model_loading(),
        "translation_model": _status_model,
        "translation_method": _method,
        "remote_active": remote_active,
        "remote_endpoint": remote_cfg.get("endpoint", "") if remote_active else "",
        "remote_reachable": _check_remote_reachable(_get_remote_endpoint_safe()) if remote_active else None,
        "remote_fallback": remote_cfg.get("fallback", "skip"),
        # Whether the "fall back to local translation" choice can actually be
        # honoured. It silently becomes "skip" when no local model is on disk,
        # and both look the same on screen: untranslated captions. Reported so
        # the page can say so before a service rather than during one. null when
        # the choice is not in play (not offloading, or set to skip).
        "local_fallback_ready": _local_fallback_ready() if (
            remote_active and remote_cfg.get("fallback", "skip") == "local") else None,
        "cache_size": get_translation_cache().get_size(),
        "is_transcription_running": transcription_state.get("running", False),
        # Device the local NLLB model actually landed on ('cuda'/'mps'/'cpu',
        # null until loaded). 'cpu' on a machine meant to accelerate is the
        # classic cause of seconds-per-sentence translations.
        "model_device": _llm_device_label() if _using_llm else (
            _live_translation_device if not (remote_active or _using_whisper) else None),
        # EMA of local NLLB per-translation latency (ms); null until a local
        # translation has run. High values flag the seconds-per-sentence problem.
        "local_translate_ms_ema": round(_local_translate_ms_ema, 1) if _local_translate_ms_ema is not None else None,
        # EMA of the offloaded round-trip latency (ms) this machine sees when it
        # sends a translation to the remote (network + remote inference); null
        # until an offloaded translation has run.
        "remote_translate_ms_ema": round(_remote_translate_ms_ema, 1) if _remote_translate_ms_ema is not None else None,
        # Server-side offload cache stats (size/hits/misses/hit_rate) — how much
        # the offloaded-translation cache is saving on this box.
        "server_cache": get_server_text_cache().get_stats(),
        # Precision: intended (config flag) vs actually loaded (probed above).
        "use_fp16": None if _using_llm else bool(trans_config.get("use_fp16", False)),
        "model_dtype": _model_dtype,
        # Inference backend: intended (config) vs actually loaded. A paired
        # Machine A reads these to show what this offload box will run with.
        "use_ctranslate2": None if _using_llm else bool(trans_config.get("use_ctranslate2", False)),
        "ct2_compute_type": None if _using_llm else trans_config.get("ct2_compute_type", "auto"),
        "is_ctranslate2": None if _using_llm else (
            bool(_live_translation_is_ct2) if not (remote_active or _using_whisper) else None),
        # What an LLM session is actually running, so a paired machine records the
        # model that translated rather than the NMT model standing by.
        "llm_provider": (_llm_cfg.get("provider") or "endpoint") if _using_llm else None,
        "llm_model": _status_model if _using_llm else None,
        "llm_endpoint": (_llm_cfg.get("endpoint") or "") if (_using_llm and not _llm_local) else None,
        # ...and *how* it runs it. On an offloaded session these settings live only
        # on this box, so without them the paired machine's transcript records which
        # model translated but nothing about the configuration that shaped every
        # caption — the prompt above all, which is the tuning surface. A service
        # recorded that way cannot be replayed against a changed setting later,
        # because nobody can say what the setting used to be.
        "llm_max_tokens": coerce_int(_llm_cfg.get("max_tokens"), 160, lo=16, hi=1024) if _using_llm else None,
        "llm_n_ctx": (coerce_int(_llm_cfg.get("n_ctx"), 2048, lo=_LLM_MIN_N_CTX, hi=32768)
                      if (_using_llm and _llm_local) else None),
        "llm_retry_on_reject": _llm_retry_enabled(_llm_cfg) if _using_llm else None,
        "llm_fallback": ((_llm_cfg.get("fallback") or "nmt").strip().lower()
                         if _using_llm else None),
        "llm_context_window": coerce_int(trans_config.get("context_window"), 1, lo=1, hi=10) if _using_llm else None,
        # The effective prompt, built exactly as the caption path builds it — not the
        # configured value, which is usually blank and says nothing about what was sent.
        "llm_system_prompt": (_llm_system_prompt(
            _llm_cfg.get("system_prompt") or _DEFAULT_LLM_SYSTEM_PROMPT,
            trans_config.get("target_language"), TRANSLATION_LANGUAGES) if _using_llm else None),
    }

    # Only expose sensitive info (clients, pairs) to local/whitelisted or paired callers
    if is_local or is_paired:
        pending = [
            {"ip": ip, "code": v["code"]}
            for ip, v in list(_pending_pair_requests.items())
            if time.time() < v["expires"]
        ]
        result["remote_clients"] = list(active.keys())
        # Same clients with last-seen age (seconds) — a paired A that heartbeats
        # while transcribing keeps a small age here even during silence.
        result["remote_clients_detail"] = [
            {"ip": ip, "age_s": round(now - ts),
             # None until the client has heartbeated; the UI falls back to 80.
             "port": _translation_client_ports.get(ip)}
            for ip, ts in active.items()]
        result["trusted_clients"] = list(_trusted_translation_clients)
        # Which settings a paired Machine A has actually taken over. Being paired
        # is not the same as being controlled — A pushes a model only when its
        # picker names one, and a language only when it switches one — so B's
        # settings page locks these and leaves everything else editable.
        result["a_pushed"] = sorted(k for k, v in _a_pushed.items() if v)
        # Where each paired client's own UI lives, so this machine can link back
        # to one that is paired but idle — the durable half of the port learned at
        # pairing. Only the paired IPs, so an unpaired stale entry cannot leak.
        result["trusted_client_ports"] = {
            ip: _translation_client_ports[ip]
            for ip in _trusted_translation_clients if ip in _translation_client_ports}
        result["pending_pairs"] = pending
    else:
        result["remote_clients"] = []
        result["remote_clients_detail"] = []
        result["trusted_clients"] = []
        result["a_pushed"] = []
        result["pending_pairs"] = []

    return jsonify(result)


@app.route("/api/translation/clear-cache", methods=["POST"])
def clear_translation_cache():
    """Clear the translation cache"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    get_translation_cache().clear()
    return jsonify({
        "success": True,
        "message": "Translation cache cleared"
    })


# Remote translation endpoints

@app.route("/api/translate", methods=["POST"])
def translate_remote():
    """Remote translation endpoint — called by a paired machine (Machine A).
    Body JSON: {text, source_lang, target_lang, return_extras, num_alternatives}
    Returns: {translated_text, confidence?, alternatives?}
    """
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"success": False, "error": "Not paired. Use the pairing flow in the Translations tab."}), 403

    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"success": True, "translated_text": "", "confidence": None, "alternatives": []}), 200

    _register_translation_client(client_ip)

    cfg = config.get("live_translation", {})
    # A caption is short; anything large here is a malformed or runaway payload. Reject
    # it before the model sees it and — just as important — before it becomes a
    # TextTranslationCache key, which holds the full text for up to server_cache_size
    # entries.
    _max_chars = coerce_int((cfg.get("remote", {}) or {}).get("max_text_chars"), 8000, lo=200, hi=1000000)
    if len(text) > _max_chars:
        return jsonify({"success": False,
                        "error": f"Text too long: {len(text)} chars exceeds the {_max_chars}-char limit."}), 413

    source_lang = data.get("source_lang", cfg.get("source_language", "auto"))
    target_lang = data.get("target_lang", cfg.get("target_language", "en"))
    return_extras = bool(data.get("return_extras", False))
    num_alternatives = coerce_int(data.get("num_alternatives"), 0, lo=0, hi=10)
    # Use generation_params from request (Machine A's settings) if provided
    generation_params = data.get("generation_params")

    # Server-side text cache: on the simple offload path (no extras) skip the
    # model for repeated identical requests. Only the hot path is cached —
    # extras/alternatives requests carry confidence data we don't want to stale.
    _cache_on = (not return_extras and num_alternatives == 0
                 and (cfg.get("remote", {}) or {}).get("server_cache_enabled", True))
    # Every generation param that changes the output is part of the cache key,
    # so e.g. a length_penalty change doesn't serve stale results.
    _gp = generation_params or cfg.get("generation_params", {}) or {}
    _cache_kw = {
        "length_penalty": coerce_float(_gp.get("length_penalty"), 1.0),
        "no_repeat_ngram_size": coerce_int(_gp.get("no_repeat_ngram_size"), 0),
        "repetition_penalty": coerce_float(_gp.get("repetition_penalty"), 1.0),
    }
    _num_beams = coerce_int(_gp.get("num_beams"), 2)
    if _cache_on:
        try:
            _hit = get_server_text_cache().get(text, source_lang, target_lang, _num_beams, **_cache_kw)
            if _hit is not None:
                return jsonify({"success": True, "translated_text": _hit.get("text", text),
                                "confidence": None, "alternatives": []})
        except Exception:
            pass  # cache must never break translation

    # local_only: we are the translation SERVER for this request — translate
    # locally and never re-offload, even if this machine is itself configured to
    # offload elsewhere (prevents chaining loops).
    result = translate_live_text(text, source_lang, target_lang,
                                 return_extras=return_extras,
                                 num_alternatives=num_alternatives,
                                 generation_params=generation_params,
                                 local_only=True)

    if return_extras and isinstance(result, dict):
        return jsonify({
            "success": True,
            "translated_text": result.get("text", text),
            "confidence": result.get("confidence"),
            "alternatives": result.get("alternatives", []),
        })

    translated = result if isinstance(result, str) else text
    # Skip caching a failed/echoed translation (would pin an untranslated answer).
    if _cache_on and isinstance(result, str) and _should_cache_translation(text, translated):
        try:
            get_server_text_cache().set(text, source_lang, target_lang, _num_beams, {"text": translated}, **_cache_kw)
        except Exception:
            pass
    return jsonify({"success": True, "translated_text": translated, "confidence": None, "alternatives": []})


@app.route("/api/translate/unload", methods=["POST"])
def translate_unload():
    """Remote unload — called by a paired Machine A to ask this machine to unload its translation model.
    Only unloads if no other trusted clients have been active in the last 60 seconds."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired"}), 403

    # Only unload if no other trusted clients are actively translating
    active_others = []
    now = time.time()
    with _translation_clients_lock:
        for ip, last_seen in _translation_clients.items():
            if ip != client_ip and (now - last_seen) < 60:
                active_others.append(ip)

    if active_others:
        return jsonify({"success": False, "reason": "Other clients still active", "active": len(active_others)})

    # Free whichever engine is actually resident. Checking only the NMT flag here
    # meant an LLM session reported "Model not loaded" and kept its GGUF for the
    # life of the process — Machine A logged that reply as a successful unload.
    unloaded = []
    if is_live_translation_model_loaded():
        unloaded.append(("translation model", unload_live_translation_model))
    if _uses_local_llm(config.get("live_translation", {})) and is_local_llm_loaded():
        unloaded.append(("LLM", unload_local_llm))

    if not unloaded:
        return jsonify({"success": True, "message": "Model not loaded"})

    import threading as _threading
    for _, _unload in unloaded:
        _threading.Thread(target=_unload, daemon=True).start()
    return jsonify({"success": True,
                    "message": "Unloading " + " and ".join(name for name, _ in unloaded),
                    "unloaded": [name for name, _ in unloaded]})


@app.route("/api/models/gguf-list", methods=["GET"])
def list_gguf_models():
    """Downloaded GGUF models for the LLM translation picker.

    ``runtime_available`` reports whether llama-cpp-python is importable, so the
    settings page can say plainly that the in-process provider cannot run here
    instead of letting a service discover it by falling back to the NMT model.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    return jsonify({
        "success": True,
        "runtime_available": local_llm_available(),
        "models": _scan_gguf_models(MODELS_DIR),
    })


@app.route("/api/models/gguf-repo-files", methods=["GET"])
def list_gguf_repo_files():
    """The .gguf files a HuggingFace repo publishes, with sizes, newest-first by name.

    A quantisation picker needs this because a GGUF repo carries a dozen variants of
    one model and only one of them should ever be downloaded.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    repo_id = (request.args.get("repo_id") or "").strip()
    if not repo_id:
        return jsonify({"success": False, "error": "repo_id required"}), 400

    try:
        from huggingface_hub import HfApi as _HfApi
        info = _HfApi().model_info(repo_id, files_metadata=True)
        files = [{"name": s.rfilename, "size_bytes": s.size or 0}
                 for s in (info.siblings or [])
                 if s.rfilename.lower().endswith(".gguf")]
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 502

    downloaded = set()
    try:
        local = os.path.join(MODELS_DIR, repo_id.replace("/", "--"))
        downloaded = {n for n in os.listdir(local) if n.lower().endswith(".gguf")}
    except OSError:
        pass
    for f in files:
        f["downloaded"] = f["name"] in downloaded

    return jsonify({"success": True, "repo_id": repo_id,
                    "files": sorted(files, key=lambda f: f["name"])})


@app.route("/api/translate/llm-prompt", methods=["GET"])
def get_llm_prompt():
    """The built-in prompt template, and the prompt actually sent to the model.

    Worth exposing because neither is visible in the settings field: an empty
    field means the built-in template is used, "{language}" is substituted, and
    the configured target language is appended in any case. An operator tuning
    terminology is editing one input to a prompt they could not otherwise read.

    ``target`` overrides the configured target language so the page can preview
    the effect of a language switch before saving it.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    lt = config.get("live_translation", {})
    llm_cfg = lt.get("llm") or {}
    target = (request.args.get("target") or lt.get("target_language") or "en").strip()
    # A prompt passed here is previewed without being saved, so the page can show
    # what an unsaved edit would send.
    custom = request.args.get("prompt")
    if custom is None:
        custom = llm_cfg.get("system_prompt") or ""

    return jsonify({
        "success": True,
        "default_template": _DEFAULT_LLM_SYSTEM_PROMPT,
        "is_custom": bool((custom or "").strip()),
        "target_language": target,
        "language_name": TRANSLATION_LANGUAGES.get(target, target),
        "effective": _llm_system_prompt(custom or _DEFAULT_LLM_SYSTEM_PROMPT,
                                        target, TRANSLATION_LANGUAGES),
    })


@app.route("/api/translate/llm-test", methods=["POST"])
def test_llm_translation():
    """Translate one fixed caption with the submitted (not yet saved) LLM settings.

    Translating something real is the only check worth offering: it catches an
    unreachable endpoint, a wrong model name, a missing GGUF — and the failure that
    actually cost a service, a reasoning model that answers every caption with
    "Okay, let's tackle this translation request…". No connectivity probe reveals
    that one, which is why looks_like_reasoning_model() is reported here.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = _control_params(keep_blank=True)
    llm_cfg = data.get("llm") if isinstance(data.get("llm"), dict) else data
    lt = config.get("live_translation", {})
    target_lang = str(data.get("target_language") or lt.get("target_language") or "en")

    # A cold in-process load is far slower than a caption's budget, and this is an
    # explicit, operator-initiated action — so it gets the warm-up timeout, not the
    # caption one.
    timeout = coerce_float(llm_cfg.get("warmup_timeout_ms"),
                           coerce_float(lt.get("llm", {}).get("warmup_timeout_ms"), 180000),
                           lo=1000, hi=900000) / 1000.0

    # Clear a previous failed load so a corrected setting can actually be retried.
    if (llm_cfg.get("provider") or "").strip().lower() == "local":
        unload_local_llm()

    source = "Да будет мир Твой, Господи, с нами всегда."
    started = time.time()
    try:
        clean, raw, extra = _translate_via_llm(source, "ru", target_lang,
                                               timeout_override=timeout,
                                               llm_cfg_override=llm_cfg,
                                               return_raw=True)
    except Exception as e:
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"})
    elapsed_ms = int((time.time() - started) * 1000)

    if raw is None:
        reason = extra if isinstance(extra, str) else None
        return jsonify({"success": False,
                        "error": reason or "No output — check the model, endpoint and file path.",
                        "elapsed_ms": elapsed_ms})

    return jsonify({
        "success": True,
        "source": source,
        "raw": raw,
        "translation": clean,
        "accepted": clean is not None,
        "reason": None if clean else "not a usable translation",
        "elapsed_ms": elapsed_ms,
        "reasoning_model": _llm_looks_like_reasoning(extra) if isinstance(extra, dict)
        else bool(_llm_looks_like_reasoning({"message": {"content": raw}})),
    })


@app.route("/api/translate/preload", methods=["POST"])
def translate_preload():
    """Remote preload — called by a paired Machine A when it starts transcription.
    Loads the translation model in the background so it's ready for requests."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired"}), 403

    # Mark wanted BEFORE the early returns: a queued unload (quick stop->start)
    # would otherwise remove the model right after we ack "already loaded"
    global _live_translation_model_wanted
    _live_translation_model_wanted = True

    if is_live_translation_model_loaded():
        return jsonify({"success": True, "message": "Model already loaded"})

    if is_live_translation_model_loading():
        return jsonify({"success": True, "message": "Model already loading"})

    cfg = config.get("live_translation", {})
    if not cfg.get("enabled", False):
        return jsonify({"success": False, "message": "Translation not enabled on this machine"})

    use_gpu = cfg.get("use_gpu", True)
    model_id = cfg.get("translation_model")

    def _preload():
        print(f"[PRELOAD] Loading translation model for remote client {client_ip}...")
        get_live_translation_model(use_gpu, model_id)
        print("[PRELOAD] Translation model loaded and ready")

    import threading
    threading.Thread(target=_preload, daemon=True).start()
    return jsonify({"success": True, "message": "Loading translation model"})


@app.route("/api/translate/sync-dictionary", methods=["POST"])
def translate_sync_dictionary():
    """Remote dictionary sync — called by a paired Machine A to push its
    glossary for this session only. Held in memory; never written to this
    machine's own custom_dictionary.json, and cleared when the pairing ends."""
    if not _is_trusted_translation_client(request.remote_addr):
        return jsonify({"error": "Not paired"}), 403
    global _session_glossary_override
    data = request.get_json() or {}
    _a_pushed["glossary"] = True
    _session_glossary_override = {
        "glossary": data.get("dictionary", {}).get("glossary", {}),
        # Client's glossary-enabled flag (None from an older client → fall back
        # to this machine's own config in _apply_glossary).
        "nllb_glossary_enabled": data.get("nllb_glossary_enabled"),
    }
    return jsonify({"success": True})


@app.route("/api/translate/pair/request", methods=["POST"])
def translation_pair_request():
    """Machine A calls this to initiate pairing. Machine B shows the code."""
    client_ip = request.remote_addr
    code = str(random.randint(100000, 999999))
    now = time.time()
    with _pending_pair_lock:
        # Drop expired entries, then bound total pending requests so an
        # unauthenticated caller can't grow the dict without limit.
        for ip in [ip for ip, v in _pending_pair_requests.items() if v["expires"] < now]:
            del _pending_pair_requests[ip]
        if client_ip not in _pending_pair_requests and len(_pending_pair_requests) >= PAIR_MAX_PENDING:
            return jsonify({"error": "Too many pending pair requests, try again later"}), 429
        _pending_pair_requests[client_ip] = {
            "code": code, "expires": now + 300, "attempts": 0,
            # Where A's own UI lives, so B can link back to it without waiting
            # for a session (the heartbeat only runs while A transcribes).
            "port": coerce_int((request.get_json(silent=True) or {}).get("port"), 0,
                               lo=0, hi=65535) or None,
        }
    socketio.emit("translation_pair_request", {"ip": client_ip, "code": code})
    return jsonify({"status": "pending", "message": "Check the Translations tab on Machine B for the 6-digit code"})


@app.route("/api/translate/pair/confirm", methods=["POST"])
def translation_pair_confirm():
    """Machine A calls this with the code displayed on Machine B."""
    client_ip = request.remote_addr
    data = request.get_json() or {}
    code = str(data.get("code", "")).strip()
    with _pending_pair_lock:
        pending = _pending_pair_requests.get(client_ip)
        if not pending or time.time() > pending["expires"]:
            _pending_pair_requests.pop(client_ip, None)
            return jsonify({"error": "Invalid or expired code"}), 400
        # Constant-time compare; void the pending code after too many misses so
        # the 10^6 space can't be brute-forced inside the 300 s window.
        if not secrets.compare_digest(pending["code"], code):
            pending["attempts"] += 1
            if pending["attempts"] >= PAIR_MAX_ATTEMPTS:
                del _pending_pair_requests[client_ip]
                return jsonify({"error": "Too many incorrect codes, pairing cancelled"}), 429
            return jsonify({"error": "Invalid or expired code"}), 400
        _pair_port = coerce_int(data.get("port"), 0, lo=0, hi=65535) or pending.get("port")
        del _pending_pair_requests[client_ip]
    _add_trusted_client(client_ip, _pair_port)
    socketio.emit("translation_pair_confirmed", {"ip": client_ip})
    return jsonify({"status": "paired"})


@app.route("/api/translate/pair/respond", methods=["POST"])
def translation_pair_respond():
    """Machine B's Allow/Deny buttons call this."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    data = request.get_json() or {}
    client_ip = data.get("ip", "")
    allow = bool(data.get("allow", False))
    if allow and client_ip:
        _add_trusted_client(client_ip,
                            (_pending_pair_requests.get(client_ip) or {}).get("port"))
        _pending_pair_requests.pop(client_ip, None)
        socketio.emit("translation_pair_confirmed", {"ip": client_ip})
    else:
        _pending_pair_requests.pop(client_ip, None)
        socketio.emit("translation_pair_denied", {"ip": client_ip})
    return jsonify({"success": True})


@app.route("/api/translate/pair/status", methods=["GET"])
def translation_pair_status():
    """Machine A polls this to check if it is paired."""
    client_ip = request.remote_addr
    return jsonify({"paired": _is_trusted_translation_client(client_ip)})


@app.route("/api/translate/pair/unpair", methods=["POST"])
def translation_unpair():
    """Remove a trusted client IP from the paired list."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    data = request.get_json() or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"success": False, "error": "ip required"}), 400
    _trusted_translation_clients.discard(ip)
    trusted = config.get("live_translation", {}).get("trusted_clients", [])
    if ip in trusted:
        trusted.remove(ip)
    _forget_client_port(ip)
    save_config(config)
    if not _trusted_translation_clients:
        global _session_glossary_override
        _session_glossary_override = None
        # Nobody is paired, so nothing here is A's any more.
        _a_pushed.update(language=False, glossary=False)
    socketio.emit("translation_pair_denied", {"ip": ip})
    return jsonify({"success": True})


@app.route("/api/translate/heartbeat", methods=["POST"])
def translate_remote_heartbeat():
    """A paired Machine A pings this while its transcription is running, so this
    offload server knows A is live even during silent stretches (no translate
    traffic). Refreshes the client's last-seen so it shows in remote_clients."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired"}), 403
    _hb_port = coerce_int((request.get_json(silent=True) or {}).get("port"), 0,
                          lo=0, hi=65535)
    _register_translation_client(client_ip, _hb_port or None)
    if _hb_port:
        # A moved to a different port, or we learned it for the first time.
        _remember_client_port(client_ip, _hb_port)
    return jsonify({"success": True})


@app.route("/api/translate/language", methods=["POST"])
def translate_remote_language():
    """Paired Machine A calls this to switch Machine B's target translation language."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"error": "Not paired"}), 403

    global config
    data = request.get_json() or {}
    new_language = data.get("target_language")
    _active_method = config.get("live_translation", {}).get("translation_method", "nllb")
    if not new_language or not supported_target(new_language, _active_method):
        return jsonify({"error": "Invalid language"}), 400

    old_language = config.get("live_translation", {}).get("target_language", "en")
    if "live_translation" not in config:
        config["live_translation"] = {}
    config["live_translation"]["target_language"] = new_language
    # The target language is A's from here: it switches languages mid-service,
    # and B's own setting would be overwritten again on the next switch.
    _a_pushed["language"] = True

    # Auto-switch TTS voice/model on Machine B to match the new language
    new_tts_voice = None
    tts_section = config["live_translation"].setdefault("tts", {})
    backend = tts_section.get("backend", "edge")
    if old_language != new_language:
        if backend == "edge":
            prefs = tts_section.setdefault("edge_voice_preferences", {})
            current_voice = tts_section.get("edge_voice", "")
            if current_voice:
                prefs[old_language] = current_voice
            new_tts_voice = prefs.get(new_language) or _pick_default_edge_voice(new_language)
            if new_tts_voice:
                tts_section["edge_voice"] = new_tts_voice
        elif backend == "piper":
            prefs = tts_section.setdefault("piper_model_preferences", {})
            current_model = tts_section.get("piper_model", "")
            if current_model:
                prefs[old_language] = current_model
            new_tts_voice = prefs.get(new_language) or _pick_default_piper_model(new_language)
            if new_tts_voice:
                tts_section["piper_model"] = new_tts_voice
                def _reload():
                    unload_tts_model()
                    get_tts_model(model_name=new_tts_voice)
                threading.Thread(target=_reload, daemon=True).start()

    save_config(config)
    language_name = TRANSLATION_LANGUAGES.get(new_language, new_language)
    print(f"[LIVE-TRANSLATION] Remote hot-switch: {old_language} -> {new_language} ({language_name})")
    socketio.emit("language_switched", {
        "old_language": old_language,
        "new_language": new_language,
        "language_name": language_name,
    })
    return jsonify({"success": True, "language_name": language_name})


@app.route("/api/translate/pair/unpair-me", methods=["POST"])
def translation_unpair_me():
    """A paired Machine A calls this to remove itself from Machine B's trusted list."""
    client_ip = request.remote_addr
    if not _is_trusted_translation_client(client_ip):
        return jsonify({"success": False, "error": "Not paired"}), 403
    _trusted_translation_clients.discard(client_ip)
    trusted = config.get("live_translation", {}).get("trusted_clients", [])
    if client_ip in trusted:
        trusted.remove(client_ip)
    _forget_client_port(client_ip)
    save_config(config)
    # Unload model if no other clients remain
    if not _trusted_translation_clients:
        global _session_glossary_override
        _session_glossary_override = None
        _a_pushed.update(language=False, glossary=False)
        if is_live_translation_model_loaded():
            import threading
            threading.Thread(target=unload_live_translation_model, daemon=True).start()
    socketio.emit("translation_pair_denied", {"ip": client_ip})
    return jsonify({"success": True, "message": f"Unpaired {client_ip}"})


# =============================================================================
# TTS (Text-to-Speech) API Endpoints
# =============================================================================

@app.route("/api/tts/status", methods=["GET"])
def get_tts_status():
    """Get TTS status for both edge-tts and piper backends"""
    tts_config = config.get("live_translation", {}).get("tts", {})
    backend = _get_tts_backend()

    result = {
        "success": True,
        "enabled": tts_config.get("enabled", False),
        "backend": backend,
        "model_loaded": is_tts_model_loaded(),
        "model_loading": is_tts_model_loading(),
        "downloading": _tts_download_status.get("status") == "downloading",
        "speed": tts_config.get("speed", 1.0),
        # EMA of TTS synthesis time (ms); null until TTS has run.
        "synth_ms_ema": round(_tts_synth_ms_ema, 1) if _tts_synth_ms_ema is not None else None,
    }

    if backend == "edge":
        result["edge_voice"] = tts_config.get("edge_voice", "en-US-AriaNeural")
        result["edge_available"] = True
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            result["edge_available"] = False
    elif backend == "piper":
        result["piper_model"] = tts_config.get("piper_model", "")
        result["piper_available"] = True
        try:
            import piper  # noqa: F401
        except ImportError:
            result["piper_available"] = False

    return jsonify(result)


@app.route("/api/tts/voices", methods=["GET"])
def get_tts_voices():
    """Get available edge-tts voices, optionally filtered by language"""
    lang_filter = request.args.get("language", "").strip().lower()
    try:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return jsonify({"success": False, "error": "edge-tts not installed. Run: pip install edge-tts"}), 500

        voices = get_edge_tts_voices()
        if not voices:
            return jsonify({"success": True, "voices": [], "error": "Could not fetch voices (network issue?)"})

        result = []
        for v in voices:
            locale = v.get("Locale", "").lower()
            if lang_filter and not locale.startswith(lang_filter):
                continue
            result.append({
                "id": v.get("ShortName", ""),
                "name": v.get("FriendlyName", v.get("ShortName", "")),
                "gender": v.get("Gender", ""),
                "locale": v.get("Locale", ""),
            })
        return jsonify({"success": True, "voices": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tts/settings", methods=["POST"])
def save_tts_settings():
    """Save TTS settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config
    data = _control_params(keep_blank=True)

    if "live_translation" not in config:
        config["live_translation"] = {}
    if "tts" not in config["live_translation"]:
        config["live_translation"]["tts"] = {}

    old_enabled = config["live_translation"]["tts"].get("enabled", False)
    old_backend = config["live_translation"]["tts"].get("backend", "edge")
    old_piper_model = config["live_translation"]["tts"].get("piper_model", "")

    allowed_keys = ["enabled", "backend", "edge_voice", "piper_model", "speed"]
    for key in allowed_keys:
        if key in data:
            config["live_translation"]["tts"][key] = data[key]

    # Save manual voice/model selection as per-language preference
    target_lang = config.get("live_translation", {}).get("target_language", "en")
    if data.get("edge_voice"):
        prefs = config["live_translation"]["tts"].setdefault("edge_voice_preferences", {})
        prefs[target_lang] = data["edge_voice"]
    if data.get("piper_model"):
        prefs = config["live_translation"]["tts"].setdefault("piper_model_preferences", {})
        prefs[target_lang] = data["piper_model"]

    save_config(config)

    now_enabled = config["live_translation"]["tts"].get("enabled", False)
    new_backend = config["live_translation"]["tts"].get("backend", "edge")
    new_piper_model = config["live_translation"]["tts"].get("piper_model", "")

    # Handle piper model loading/unloading
    if new_backend == "piper":
        if not now_enabled and old_enabled:
            threading.Thread(target=unload_tts_model, daemon=True).start()
        elif now_enabled and (old_backend != "piper" or old_piper_model != new_piper_model):
            def reload_piper():
                unload_tts_model()
                get_tts_model(model_name=new_piper_model)
            threading.Thread(target=reload_piper, daemon=True).start()
    elif old_backend == "piper":
        # Switched away from piper, unload
        threading.Thread(target=unload_tts_model, daemon=True).start()

    return jsonify({
        "success": True,
        "message": "TTS settings saved."
    })


# ─── Piper TTS model management ─────────────────────────────────────────────

_tts_download_status = {"status": "idle", "model": "", "error": ""}


def _get_piper_model_dir(model_id):
    """Get the directory for a piper model, or None if model_id would escape the
    piper cache dir (model_id is request input on the download/remove routes)."""
    return safe_model_path(os.path.join(_tts_cache_dir, "piper"), model_id)


def _is_piper_model_downloaded(model_id):
    """Check if a piper model is downloaded"""
    model_dir = _get_piper_model_dir(model_id)
    if model_dir and os.path.isdir(model_dir):
        return any(f.endswith(".onnx") for f in os.listdir(model_dir))
    return False


@app.route("/api/models/tts-list", methods=["GET"])
def list_tts_models_catalog():
    """List TTS models (piper) with download status"""
    try:
        models = []
        for m in _PIPER_MODELS_CATALOG:
            entry = dict(m)
            entry["downloaded"] = _is_piper_model_downloaded(m["id"])
            models.append(entry)
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/tts/download", methods=["POST"])
def download_tts_model():
    """Download a piper TTS model from HuggingFace"""
    global _tts_download_status
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json() or {}
    model_name = data.get("model_name", "").strip()
    if not model_name:
        return jsonify({"success": False, "error": "model_name required"}), 400

    # Reject any name that would resolve outside the piper cache dir before it
    # reaches os.makedirs / download.
    if _get_piper_model_dir(model_name) is None:
        return jsonify({"success": False, "error": "Invalid model name"}), 400

    # Parse model ID: e.g., "en_US-lessac-medium" -> lang "en_US", name "lessac", quality "medium"
    parts = model_name.split("-")
    if len(parts) < 3:
        return jsonify({"success": False, "error": f"Invalid piper model ID format: {model_name}"}), 400

    lang_code, voice_name, quality = parts[0], parts[1], parts[2]

    # HuggingFace piper voices URL
    lang_family = lang_code.split("_")[0]  # "en_US" -> "en"
    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{voice_name}/{quality}"
    onnx_url = f"{base_url}/{model_name}.onnx"
    json_url = f"{base_url}/{model_name}.onnx.json"

    # Best-effort total size for a real progress percentage (json config is negligible)
    total_size = None
    try:
        import urllib.request
        req = urllib.request.Request(onnx_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            total_size = int(resp.headers.get("Content-Length") or 0) or None
    except Exception as e:
        print(f"[TTS] Could not get size of {model_name}: {e}")

    download_key = f"tts-{model_name}"
    if not try_register_download(download_key, total=total_size):
        return jsonify({"success": False, "error": "Download already in progress"}), 409

    def _do_download():
        global _tts_download_status
        _tts_download_status = {"status": "downloading", "model": model_name, "error": ""}
        model_dir = _get_piper_model_dir(model_name)
        try:
            os.makedirs(model_dir, exist_ok=True)
            start_download_monitor(download_key, model_dir, total=total_size)

            print(f"[TTS] Downloading piper model: {model_name}")
            for url, filename in ((onnx_url, f"{model_name}.onnx"),
                                  (json_url, f"{model_name}.onnx.json")):
                print(f"[TTS]   {url}")
                outcome = download_url_to_file(
                    url, os.path.join(model_dir, filename),
                    cancel_check=lambda: download_key in cancelled_downloads,
                )
                if outcome == "cancelled":
                    print(f"[TTS] Download cancelled: {model_name}")
                    # Piper models live under models/tts/piper/, which the generic
                    # cancel-route cleanup doesn't know about — clean up here
                    shutil.rmtree(model_dir, ignore_errors=True)
                    _tts_download_status = {"status": "failed", "model": model_name, "error": "Cancelled"}
                    finish_download(download_key, cancelled=True)
                    return

            print(f"[TTS] Piper model downloaded: {model_name}")
            _tts_download_status = {"status": "completed", "model": model_name, "error": ""}
            finish_download(download_key)
        except Exception as e:
            print(f"[TTS ERROR] Download failed: {e}")
            _tts_download_status = {"status": "failed", "model": model_name, "error": str(e)}
            finish_download(download_key, error=e)

    threading.Thread(target=_do_download, daemon=True).start()
    return jsonify({"success": True, "message": f"Downloading {model_name}..."})


@app.route("/api/models/tts/download-progress", methods=["GET"])
def tts_download_progress():
    """Get TTS model download progress"""
    return jsonify(_tts_download_status)


@app.route("/api/models/tts/remove", methods=["POST"])
def remove_tts_model():
    """Remove a downloaded piper TTS model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json() or {}
    model_name = data.get("model_name", "").strip()
    if not model_name:
        return jsonify({"success": False, "error": "model_name required"}), 400

    # Unload if this is the currently loaded model
    tts_config = config.get("live_translation", {}).get("tts", {})
    if tts_config.get("piper_model") == model_name and is_tts_model_loaded():
        unload_tts_model()

    try:
        model_dir = _get_piper_model_dir(model_name)
        if model_dir is None:
            return jsonify({"success": False, "error": "Invalid model name"}), 400
        if os.path.isdir(model_dir):
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)
            return jsonify({"success": True, "message": f"Removed {model_name}"})
        return jsonify({"success": False, "error": "Model not found on disk"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tts/models", methods=["GET"])
def list_tts_models():
    """List available TTS voices/models for the active backend"""
    backend = _get_tts_backend()
    if backend == "edge":
        # Return edge-tts voices
        lang_filter = request.args.get("language", "").strip().lower()
        voices = get_edge_tts_voices()
        result = []
        for v in voices:
            locale = v.get("Locale", "").lower()
            if lang_filter and not locale.startswith(lang_filter):
                continue
            result.append(v.get("ShortName", ""))
        return jsonify({"success": True, "models": result})
    else:
        # Return piper models
        models = [m["id"] for m in _PIPER_MODELS_CATALOG if _is_piper_model_downloaded(m["id"])]
        return jsonify({"success": True, "models": models})


# Proxy endpoints — forward browser requests to Machine B server-side (avoids CORS)

class _RemoteEndpointError(Exception):
    pass


class _RemoteTranslateError(Exception):
    pass


def _probe_remote_port(base_url):
    """Try common ports to find which one the STT app is listening on."""
    from urllib.parse import urlparse
    import requests as _req
    parsed = urlparse(base_url)
    hostname = parsed.hostname
    for port in [80, 8080, 443, 5000, 8000]:
        scheme = "https" if port == 443 else "http"
        try:
            r = _req.get(f"{scheme}://{hostname}:{port}/api/translation/status", timeout=2)
            if r.status_code == 200:
                return f"{scheme}://{hostname}:{port}"
        except Exception:
            continue
    return None


_remote_reachable_cache = {}  # {endpoint: (reachable: bool, checked_at: float)}
_remote_reachable_cache_lock = threading.Lock()

_remote_http_session = None
_remote_http_session_lock = threading.Lock()


def _get_remote_http_session():
    """Shared pooled requests.Session for the offload path (reachability GET +
    translate POST), so connections are kept alive/reused instead of a fresh
    TCP handshake per request. requests.Session is safe for concurrent use; the
    lock only guards one-time construction. max_retries=0 keeps fail-fast so the
    existing fallback logic fires immediately on a dead peer."""
    global _remote_http_session
    if _remote_http_session is None:
        with _remote_http_session_lock:
            if _remote_http_session is None:
                import requests
                from requests.adapters import HTTPAdapter
                s = requests.Session()
                adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _remote_http_session = s
    return _remote_http_session


def _check_remote_reachable(endpoint, timeout=1.5, ttl=5.0):
    """Cheap, cached liveness check for a paired remote translation endpoint."""
    if not endpoint:
        return False
    now = time.time()
    with _remote_reachable_cache_lock:
        cached = _remote_reachable_cache.get(endpoint)
        if cached and (now - cached[1]) < ttl:
            return cached[0]
    try:
        r = _get_remote_http_session().get(endpoint.rstrip("/") + "/api/translation/status", timeout=timeout)
        reachable = r.status_code == 200
    except Exception:
        reachable = False
    with _remote_reachable_cache_lock:
        _remote_reachable_cache[endpoint] = (reachable, now)
    return reachable


def _get_remote_endpoint():
    from urllib.parse import urlparse
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if not (remote_cfg.get("enabled") and remote_cfg.get("endpoint")):
        return None
    ep = remote_cfg["endpoint"].strip().rstrip("/")
    if not ep:
        return None
    if "://" not in ep:
        ep = "http://" + ep
    # If no port specified, probe for it and save the result
    if not urlparse(ep).port:
        found = _probe_remote_port(ep)
        if found:
            remote_cfg["endpoint"] = found
            save_config(config)
            ep = found
        else:
            host = urlparse(ep).hostname
            raise _RemoteEndpointError(
                f"Could not find STT server on {host} — try specifying the port manually (e.g. {host}:8080)"
            )
    return ep


def _get_remote_endpoint_safe():
    """Return the remote endpoint URL, or None if not configured or unreachable."""
    try:
        return _get_remote_endpoint()
    except _RemoteEndpointError:
        return None


@app.route("/api/remote-translation/status", methods=["GET"])
def proxy_remote_translation_status():
    """Proxy: fetch Machine B's translation status for display on Machine A's UI."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.get(endpoint + "/api/translation/status", timeout=5)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/pair/request", methods=["POST"])
def proxy_pair_request():
    """Proxy: send pairing request from Machine A's server to Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/request",
                      json={"port": coerce_int(config.get("web_server", {}).get("port"),
                                               8080, lo=1, hi=65535)}, timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/remote-translation/pair/confirm", methods=["POST"])
def proxy_pair_confirm():
    """Proxy: send pairing confirmation from Machine A's server to Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        # Tell B where this machine's own UI lives while we have its attention:
        # after this exchange B only ever sees our IP, and a durable port means it
        # can link back to us without waiting for a session to start.
        _body = dict(request.get_json() or {})
        _body.setdefault("port", coerce_int(config.get("web_server", {}).get("port"),
                                            8080, lo=1, hi=65535))
        r = _req.post(endpoint + "/api/translate/pair/confirm", json=_body, timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/remote-translation/pair/status", methods=["GET"])
def proxy_pair_status():
    """Proxy: check if Machine A is paired with Machine B (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"paired": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"paired": False}), 200
    import requests as _req
    try:
        r = _req.get(endpoint + "/api/translate/pair/status", timeout=5)
        try:
            data = r.json()
        except ValueError:
            data = {"paired": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"paired": False, "error": str(e)}), 200


@app.route("/api/remote-translation/unload", methods=["POST"])
def proxy_translate_unload():
    """Proxy: tell Machine B to unload its translation model (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/unload", timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/preload", methods=["POST"])
def proxy_translate_preload():
    """Proxy: tell Machine B to preload its translation model (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/preload", timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/sync-dictionary", methods=["POST"])
def proxy_sync_dictionary():
    """Proxy: push this machine's custom dictionary to the paired Machine B
    for the current session (avoids CORS)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/sync-dictionary",
                      json=_dictionary_sync_payload(), timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/remote-translation/unpair", methods=["POST"])
def proxy_translate_unpair():
    """Proxy: tell Machine B to remove this machine from its trusted list."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    try:
        endpoint = _get_remote_endpoint()
    except _RemoteEndpointError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    if not endpoint:
        return jsonify({"success": False, "error": "No remote endpoint configured"}), 400
    import requests as _req
    try:
        r = _req.post(endpoint + "/api/translate/pair/unpair-me", timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {"success": False, "error": "Invalid JSON from remote"}
        return jsonify(data), r.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


# Transcription Language Endpoints


@app.route("/api/transcription/language", methods=["GET"])
def get_transcription_language():
    """Get current transcription language
    Example: GET /api/transcription/language"""
    language = config.get("audio", {}).get("language", "auto")
    return jsonify({
        "success": True,
        "language": language
    })


@app.route("/api/transcription/language", methods=["POST"])
def hot_switch_transcription_language():
    """Hot-switch transcription language without restart

    Accepts the parameter as a JSON body, form-encoded body, or query string, so a
    control surface that doesn't set Content-Type isn't rejected as though it sent
    nothing (see stt/http_params.py).
    Example: POST /api/transcription/language {"language": "en"}
    Example: POST /api/transcription/language?language=auto"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = _control_params()
    new_language = data.get("language")

    if not new_language:
        _note_access_detail(
            f"rejected: no language field (json={len(request.get_json(silent=True) or {})} "
            f"form={len(request.form)} query={len(request.args)})"
        )
        return jsonify({"success": False, "error": "language required"}), 400

    old_language = _apply_transcription_language_switch(new_language)
    changed = old_language != new_language
    _note_access_detail(
        f"transcription {old_language}->{new_language} {'changed' if changed else 'unchanged'}")

    return jsonify({
        "success": True,
        "changed": changed,
        "message": (f"Transcription language switched to {new_language}. Takes effect on next audio chunk."
                    if changed else f"Transcription language was already {new_language}; nothing changed."),
        "old_language": old_language,
        "new_language": new_language
    })


@app.route("/api/language", methods=["GET"])
def get_all_languages():
    """Get current transcription and translation languages
    Example: GET /api/language"""
    transcription_lang = config.get("audio", {}).get("language", "auto")
    translation_lang = config.get("live_translation", {}).get("target_language", "en")
    translation_name = TRANSLATION_LANGUAGES.get(translation_lang, translation_lang)

    return jsonify({
        "success": True,
        "transcription": transcription_lang,
        "translation": translation_lang,
        "translation_name": translation_name
    })


@app.route("/api/language", methods=["POST"])
def hot_switch_all_languages():
    """Hot-switch both transcription and translation languages.

    Accepts JSON body, form body, or query string (see stt/http_params.py), and
    reports per-field whether the value actually changed — a surface that sends a
    fixed language on every press needs to distinguish "already there" from
    "didn't work".
    Example: POST /api/language {"transcription": "en", "translation": "es"}
    Example: POST /api/language?transcription=auto&translation=fr"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = _control_params()

    transcription_lang = data.get("transcription")
    translation_lang = data.get("translation")

    if not transcription_lang and not translation_lang:
        # Record which sources were present but empty-handed. A surface whose body
        # didn't arrive as expected looks identical to one that sent nothing at
        # all; the counts distinguish them. Counts only — the values can include
        # ?key=<access_token> from the query string.
        _note_access_detail(
            "rejected: no language fields "
            f"(json={len(request.get_json(silent=True) or {})} "
            f"form={len(request.form)} query={len(request.args)})"
        )
        return jsonify({"success": False, "error": "At least one of 'transcription' or 'translation' required"}), 400

    # Validate the translation language before mutating anything, so a bad value
    # can't leave transcription switched but translation rejected.
    _active_method = config.get("live_translation", {}).get("translation_method", "nllb")
    if translation_lang and not supported_target(translation_lang, _active_method):
        _note_access_detail(f"rejected: translation={translation_lang} unsupported by {_active_method}")
        return jsonify({"success": False, "error": f"Invalid translation language: {translation_lang}"}), 400

    results = {}

    # Delegate to the same helpers the dedicated single-language routes use, so
    # every side effect (TTS voice, subprocess reload, remote Machine B
    # propagation, client event) fires identically no matter which route is hit.
    if transcription_lang:
        old_trans = _apply_transcription_language_switch(transcription_lang)
        results["transcription"] = {
            "old": old_trans,
            "new": transcription_lang,
            "changed": old_trans != transcription_lang,
        }

    if translation_lang:
        old_target, _tts_voice, _backend = _apply_translation_language_switch(translation_lang)
        results["translation"] = {
            "old": old_target,
            "new": translation_lang,
            "language_name": TRANSLATION_LANGUAGES.get(translation_lang, translation_lang),
            "changed": old_target != translation_lang,
        }

    # Top-level `changed` so a surface can tell "applied" from "already there"
    # without inspecting each field. Both are successes; only a non-2xx is not.
    any_changed = any(r.get("changed") for r in results.values())
    _note_access_detail("; ".join(
        f"{field} {r['old']}->{r['new']} {'changed' if r.get('changed') else 'unchanged'}"
        for field, r in results.items()
    ) or "no fields applied")
    return jsonify({
        "success": True,
        "changed": any_changed,
        "message": ("Language settings updated. Changes take effect immediately."
                    if any_changed else "Already set to the requested languages; nothing changed."),
        "changes": results
    })


# File Transcription Endpoints


@app.route("/file")
def upload_page():
    """Render file transcription page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("file.html")


@app.route("/model-manager")
def model_manager_page():
    """Render model manager page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("model-manager.html")


@app.route("/file-manager")
def file_manager_page():
    """Render file manager page"""
    if not check_ip_whitelist():
        return render_template("auth-required.html"), 403
    return render_template("file-manager.html")


@app.route("/api/file-transcription-settings", methods=["GET", "POST"])
def file_transcription_settings_endpoint():
    """Get or update file transcription settings"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global config

    if request.method == "GET":
        # Return current file transcription settings
        ft_config = config.get("file_transcription", {})
        model_config = ft_config.get("model", {})

        settings = {
            "model_type": model_config.get("type", "whisper"),
            "whisper_model": model_config.get("whisper", {}).get("model", "base"),
            "hf_model": model_config.get("huggingface", {}).get(
                "model_id", "openai/whisper-base"
            ),
            "language": ft_config.get("language", "auto"),
            "use_gpu": ft_config.get("use_gpu", True),
            "use_flash_attention": model_config.get("huggingface", {}).get(
                "use_flash_attention", False
            ),
        }

        return jsonify({"success": True, "settings": settings})

    elif request.method == "POST":
        # Update file transcription settings
        try:
            new_settings = _control_params(keep_blank=True)

            # Update config
            if "file_transcription" not in config:
                config["file_transcription"] = {}

            if "model" not in config["file_transcription"]:
                config["file_transcription"]["model"] = {}

            # Update model type
            if "model_type" in new_settings:
                config["file_transcription"]["model"]["type"] = new_settings[
                    "model_type"
                ]

            # Update Whisper settings
            if "whisper" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["whisper"] = {}

            if "whisper_model" in new_settings:
                config["file_transcription"]["model"]["whisper"]["model"] = (
                    new_settings["whisper_model"]
                )

            # Update HuggingFace settings
            if "huggingface" not in config["file_transcription"]["model"]:
                config["file_transcription"]["model"]["huggingface"] = {}

            if "hf_model" in new_settings:
                config["file_transcription"]["model"]["huggingface"]["model_id"] = (
                    new_settings["hf_model"]
                )

            if "use_flash_attention" in new_settings:
                config["file_transcription"]["model"]["huggingface"][
                    "use_flash_attention"
                ] = new_settings["use_flash_attention"]

            # Update other settings
            if "language" in new_settings:
                config["file_transcription"]["language"] = new_settings["language"]

            if "use_gpu" in new_settings:
                config["file_transcription"]["use_gpu"] = new_settings["use_gpu"]

            # Save config to file
            try:
                with _config_file_lock:
                    _atomic_write_json(CONFIG_FILE, config)
                print(
                    "[OK] File transcription settings updated and saved to config.json"
                )
            except Exception as e:
                print(f"[WARNING] Failed to save config to file: {e}")

            return jsonify(
                {"success": True, "message": "Settings updated successfully"}
            )

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcribe-file", methods=["POST"])
def transcribe_file_endpoint():
    """Handle file upload and start transcription in background thread"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    temp_upload = None
    try:
        # Get uploaded file
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        output_format = request.form.get("format", "txt")
        if output_format not in ("txt", "srt", "vtt", "json"):
            return jsonify({"success": False, "error": f"Invalid format: {output_format}"}), 400
        language = request.form.get("language", "auto")  # Get language from upload form
        translate_to = request.form.get("translate_to", "")  # Optional translation target

        # Validate file
        is_valid, error_msg = validate_file(file)
        if not is_valid:
            return jsonify({"success": False, "error": error_msg}), 400

        # Save uploaded file temporarily with proper cleanup
        temp_upload = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        )
        file.save(temp_upload.name)
        temp_upload.close()

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Start transcription in background thread with proper error handling
        def safe_transcription():
            try:
                process_file_transcription(
                    temp_upload.name, output_format, session_id, file.filename, language, translate_to
                )
            except Exception as e:
                print(f"[ERROR] Background transcription failed: {e}")
                socketio.emit(
                    "file_error",
                    {
                        "session_id": session_id,
                        "error": f"Transcription failed: {e!s}",
                    },
                )

        thread = threading.Thread(target=safe_transcription)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "message": "Transcription started",
            }
        )

    except Exception as e:
        # Clean up temp file if it exists
        if temp_upload and os.path.exists(temp_upload.name):
            try:
                os.unlink(temp_upload.name)
            except OSError:
                pass
        return jsonify({"success": False, "error": str(e)}), 500


def process_file_transcription(file_path, output_format, session_id, filename, language=None, translate_to=None):
    """Process file transcription in background thread with proper resource cleanup

    Args:
        file_path: Path to the uploaded file
        output_format: Output format (txt, srt, vtt, json)
        session_id: Unique session ID for progress tracking
        filename: Original filename
        language: Source language code (or 'auto')
        translate_to: Target language code for translation (optional)
    """
    import gc
    wav_path = None
    model = None
    processor = None
    translation_model = None
    translation_tokenizer = None

    try:
        # Send initial progress
        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 5, "status": "Extracting audio..."},
        )

        # Extract/convert audio to WAV
        wav_path = extract_audio_from_file(file_path)

        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 10, "status": "Loading audio..."},
        )

        # Load entire audio file for transcription (no chunking - let Whisper handle segmentation)
        import librosa
        audio_data, sr = librosa.load(wav_path, sr=16000)
        audio_duration = len(audio_data) / sr

        socketio.emit(
            "file_progress",
            {
                "session_id": session_id,
                "percent": 20,
                "status": f"Loading model... (audio: {int(audio_duration // 60)}m {int(audio_duration % 60)}s)",
            },
        )

        # Load model using file transcription settings (or fall back to main config)
        ft_config = config.get("file_transcription", {})
        ft_model_config = ft_config.get("model", config["model"])
        ft_use_gpu = ft_config.get(
            "use_gpu", config.get("performance", {}).get("use_gpu", True)
        )
        # Use language from upload request, or fall back to config, or finally default to auto
        ft_language = language if language is not None else ft_config.get("language", "auto")

        model, processor, model_type = ModelFactory.load_model(
            ft_model_config, ft_use_gpu
        )

        socketio.emit(
            "file_progress",
            {
                "session_id": session_id,
                "percent": 30,
                "status": "Transcribing audio (this may take a while)...",
            },
        )

        # Transcribe entire audio file at once - Whisper handles segmentation naturally
        whisper_params = config.get("whisper_decoding", {}).get(
            "file_transcription", FILE_TRANSCRIPTION_PARAMS
        )

        segments = ModelFactory.transcribe(
            model, processor, model_type, audio_data,
            language=ft_language, whisper_params=whisper_params,
            return_segments=True
        )
        segments = [dict(s, text=apply_profanity_filter(s.get("text", ""))) for s in segments]

        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 55, "status": f"Found {len(segments)} segments..."},
        )

        # Initialize translated_segments as None (will be populated if translation is requested)
        translated_segments = None
        target_language_name = None

        # Handle translation if requested
        _ft_method = config.get("live_translation", {}).get("translation_method", "nllb")
        if translate_to and translate_to.strip() and supported_target(translate_to, _ft_method):
            source_lang = ft_language if ft_language != "auto" else "en"

            # Check translation method
            ft_translation_method = config.get("live_translation", {}).get("translation_method", "nllb")
            remote_cfg = config.get("live_translation", {}).get("remote", {})

            if ft_translation_method in ("whisper_translate", "whisper_forced_lang") and model is not None:
                # Whisper-based translation: run a second pass on the same audio with translation params
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 60, "status": "Translating with Whisper (pass 2)..."},
                )

                pass2_params = dict(whisper_params)
                pass2_language = ft_language
                if ft_translation_method == "whisper_translate" and translate_to == "en":
                    pass2_params["task"] = "translate"
                elif ft_translation_method == "whisper_forced_lang":
                    pass2_language = translate_to

                pass2_segments = ModelFactory.transcribe(
                    model, processor, model_type, audio_data,
                    language=pass2_language, whisper_params=pass2_params,
                    return_segments=True
                )

                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 85, "status": f"Whisper translation: {len(pass2_segments)} segments..."},
                )

                # Build translated_segments by matching pass 1 and pass 2 results
                translated_segments = []
                for i, seg in enumerate(segments):
                    translated_seg = dict(seg)
                    if i < len(pass2_segments):
                        translated_seg["translated_text"] = pass2_segments[i].get("text", "").strip()
                    else:
                        # More segments in pass 1 than pass 2 — use last pass 2 text or original
                        translated_seg["translated_text"] = pass2_segments[-1].get("text", "").strip() if pass2_segments else seg.get("text", "")
                    translated_segments.append(translated_seg)

                # If pass 2 had more segments, append remaining translated text to last segment
                if len(pass2_segments) > len(segments) and translated_segments:
                    extra_text = " ".join(s.get("text", "").strip() for s in pass2_segments[len(segments):] if s.get("text", "").strip())
                    if extra_text:
                        translated_segments[-1]["translated_text"] += " " + extra_text

            elif remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
                # Remote path: send each segment to Machine B, no local model load needed
                try:
                    _file_remote_ep = _get_remote_endpoint()
                except _RemoteEndpointError as e:
                    print(f"[FILE_TRANSLATE] Endpoint error: {e}")
                    _file_remote_ep = None
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 65, "status": "Translating via remote server..."},
                )
                translated_segments = []
                total = len(segments)
                for i, seg in enumerate(segments):
                    text = seg.get("text", "").strip()
                    translated_text = _translate_via_remote(text, source_lang, translate_to, _file_remote_ep) if (text and _file_remote_ep) else text
                    translated_seg = dict(seg)
                    translated_seg["translated_text"] = translated_text
                    translated_segments.append(translated_seg)
                    if total > 0:
                        pct = 65 + int(30 * (i + 1) / total)
                        socketio.emit("file_progress", {"session_id": session_id, "percent": pct, "status": f"Translating... {i+1}/{total}"})
            elif ft_translation_method == "llm":
                # The LLM translates the file too. Without this branch the method
                # was silently ignored here: a box configured for LLM translation
                # loaded the NMT model for every batch file and translated with
                # that instead, which is the opposite of what the settings said.
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 60, "status": "Unloading transcription model..."},
                )
                if model:
                    del model
                    model = None
                if processor:
                    del processor
                    processor = None
                ModelFactory.cleanup_models()
                gc.collect()
                _empty_device_cache()

                translated_segments = []
                _declined = []
                total = len(segments)
                for i, seg in enumerate(segments):
                    text = (seg.get("text") or "").strip()
                    out = _translate_via_llm(text, source_lang, translate_to) if text else ""
                    translated_seg = dict(seg)
                    if out is None:
                        # Remember it and fall back in one pass below, rather than
                        # loading the NMT model for the first declined segment and
                        # holding both models for the rest of the file.
                        _declined.append(i)
                        translated_seg["translated_text"] = text
                    else:
                        translated_seg["translated_text"] = out
                    translated_segments.append(translated_seg)
                    if total > 0:
                        pct = 60 + int(30 * (i + 1) / total)
                        socketio.emit("file_progress", {"session_id": session_id, "percent": pct,
                                                        "status": f"Translating (LLM)... {i+1}/{total}"})

                if _declined:
                    # Same contract as a live caption: anything the LLM could not
                    # translate usably goes to the NMT model rather than being left
                    # in the source language without saying so.
                    print(f"[FILE-TRANSLATE] LLM declined {len(_declined)}/{total} segments; "
                          "using the NMT model for those")
                    socketio.emit("file_progress", {"session_id": session_id, "percent": 92,
                                                    "status": f"Retranslating {len(_declined)} segment(s) with the NMT model..."})
                    try:
                        _fb_id = ft_config.get("translation_model", "facebook/nllb-200-distilled-600M")
                        _fb_model, _fb_tok = load_translation_model(ft_use_gpu, model_id=_fb_id)
                        for i in _declined:
                            _src = (segments[i].get("text") or "").strip()
                            if _src:
                                translated_segments[i]["translated_text"] = translate_text(
                                    _src, source_lang, translate_to, _fb_model, _fb_tok,
                                    model_id=_fb_id)
                        cleanup_translation_model(_fb_model, _fb_tok)
                    except Exception as e:
                        print(f"[FILE-TRANSLATE] NMT fallback failed: {e}")
            else:
                # Local path: unload transcription model to free VRAM, load NLLB locally
                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 60, "status": "Unloading transcription model..."},
                )

                # CRITICAL: Unload transcription model BEFORE loading translation model
                # This frees GPU memory for the translation model
                if model:
                    del model
                    model = None
                if processor:
                    del processor
                    processor = None

                ModelFactory.cleanup_models()
                gc.collect()
                _empty_device_cache()
                print("[CLEANUP] Transcription model unloaded before translation")

                socketio.emit(
                    "file_progress",
                    {"session_id": session_id, "percent": 65, "status": "Loading translation model..."},
                )

                # Load translation model (use configured model or default)
                translation_model_id = ft_config.get("translation_model", "facebook/nllb-200-distilled-600M")
                translation_model, translation_tokenizer = load_translation_model(ft_use_gpu, model_id=translation_model_id)

                # Create progress callback for translation
                def translation_progress(percent, status):
                    socketio.emit(
                        "file_progress",
                        {"session_id": session_id, "percent": percent, "status": status},
                    )

                # Get generation params from live_translation config (shared settings)
                ft_gen_params = config.get("live_translation", {}).get("generation_params", {})
                ft_context_window = config.get("live_translation", {}).get("context_window", 1)

                translated_segments = translate_segments(
                    segments, source_lang, translate_to,
                    translation_model, translation_tokenizer,
                    progress_callback=translation_progress,
                    generation_params=ft_gen_params,
                    context_window=ft_context_window,
                    # This model was loaded here, not by the live pipeline — say so,
                    # or it is tokenized as whatever the live model happens to be.
                    model_id=translation_model_id,
                    is_ct2=False,
                )

                # Cleanup translation model
                if translation_model:
                    del translation_model
                    translation_model = None
                if translation_tokenizer:
                    del translation_tokenizer
                    translation_tokenizer = None
                gc.collect()
                _empty_device_cache()

            target_language_name = TRANSLATION_LANGUAGES.get(translate_to, translate_to)
            print(f"[INFO] Translation complete: {len(translated_segments)} segments translated to {target_language_name}")
            print("[CLEANUP] Translation model unloaded")

        # Send segments for client-side formatting
        socketio.emit(
            "file_progress",
            {"session_id": session_id, "percent": 95, "status": "Preparing results..."},
        )

        # Build completion data
        completion_data = {
            "session_id": session_id,
            "segments": segments,  # Original transcription segments
            "format": output_format,
            "duration": segments[-1]["end"] if segments else 0,
            "total_segments": len(segments),
            "filename": filename,
            "source_language": ft_language,
        }

        # Add translation data if available
        if translated_segments:
            completion_data["translated_segments"] = translated_segments
            completion_data["target_language"] = translate_to
            completion_data["target_language_name"] = target_language_name

        # Send completion with segments array for client-side format switching
        socketio.emit("file_complete", completion_data)

    except Exception as e:
        socketio.emit("file_error", {"session_id": session_id, "error": str(e)})
    finally:
        # Cleanup resources
        try:
            # Clean up models
            if model:
                del model
            if processor:
                del processor
            if translation_model:
                del translation_model
            if translation_tokenizer:
                del translation_tokenizer

            # Clear model cache to prevent blocking live transcription
            ModelFactory.cleanup_models()
            print("[CLEANUP] Model cache cleared after file transcription")

            # Accelerator cleanup (CUDA or MPS)
            _empty_device_cache()
            print("[CLEANUP] GPU cache cleared")

            gc.collect()

            # Clean up temp files
            if os.path.exists(file_path):
                os.unlink(file_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as cleanup_error:
            print(f"[WARNING] Error during cleanup: {cleanup_error}")


@app.route("/api/restart", methods=["POST"])
def restart_transcription():
    """API endpoint to restart the transcription process only (not the whole server).

    Use this for:
    - Restarting transcription after model changes
    - Resetting audio capture without losing web server connection

    For full server restart (config changes, port changes), use /api/server/restart instead.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_process
    try:
        if transcription_process is not None and transcription_process.is_alive():
            # Send stop command
            control_queue.put({"command": "stop"})
            sleep(2)  # Wait for graceful shutdown

            # Terminate if still alive
            if transcription_process.is_alive():
                transcription_process.terminate()
                transcription_process.join(timeout=5)

                # Force kill if still alive
                if transcription_process.is_alive():
                    transcription_process.kill()
                    transcription_process.join(timeout=2)

            # Start new process
            transcription_process = multiprocessing.Process(
                target=thread1_function,
                args=(transcription_state, control_queue, config_queue,
                      calibration_state, calibration_data_shared, calibration_step1_data,
                      audio_stream_queue)
            )
            transcription_process.start()

            # CRITICAL: Update global reference for signal handler
            globals()["thread1"] = transcription_process

            control_queue.put({"command": "start"})

            return jsonify(
                {
                    "success": True,
                    "message": "Transcription process restarted successfully!",
                }
            )
        else:
            return jsonify(
                {"success": False, "error": "Transcription process is not running"}
            ), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _terminate_child_processes():
    """Best-effort teardown of every multiprocessing child before an execv.

    Children forked after the web server bound its socket inherit the bound
    fd; any that survive an in-place execv keep the port held and the fresh
    process dies with EADDRINUSE.
    """
    try:
        mp_manager.shutdown()
    except Exception:
        pass
    for child in multiprocessing.active_children():
        try:
            child.terminate()
            child.join(timeout=3)
            if child.is_alive():
                child.kill()
                child.join(timeout=1)
        except Exception as e:
            print(f"[RESTART] Error terminating child PID={child.pid}: {e}")


def perform_server_restart():
    """Restart the whole server process.

    Prefers supervisor-managed restarts: under systemd, `systemctl restart`
    is atomic and the unit's stop script cleans up straggler processes that
    may still hold the bound server socket. Falls back to the restart
    scripts, then to an in-place execv as a last resort.
    """
    import subprocess

    # Signal emit threads to stop touching the Manager proxy before we tear it down.
    _server_shutting_down.set()

    if sys.platform.startswith('win'):
        # Windows: use restart_server.bat to cleanly stop and restart
        script_dir = APP_DIR
        restart_bat = os.path.join(script_dir, "restart_server.bat")
        if os.path.exists(restart_bat):
            print("[RESTART] Calling restart_server.bat...")
            subprocess.Popen(
                ["cmd.exe", "/c", restart_bat],
                cwd=script_dir,
                # NEW_PROCESS_GROUP: survive this process's exit. NO_WINDOW: the
                # server runs windowless — cmd would flash a console otherwise.
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            )
        else:
            # Fallback: spawn new process directly
            print("[RESTART] restart_server.bat not found, spawning directly...")
            subprocess.Popen(
                [sys.executable, *sys.argv],
                cwd=script_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        sleep(1)
        os._exit(0)
        return

    # Use systemctl restart if running as a systemd service
    # This is atomic - systemd handles stop+start without race conditions
    under_systemd = False
    for service_name in ["stt-watchdog", "stt-server", "stt"]:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "--quiet", service_name],
                capture_output=True,
            )
        except OSError:
            break  # no systemctl on this system (e.g. macOS)
        if result.returncode == 0:
            under_systemd = True
            print(f"[RESTART] Restarting via systemctl ({service_name})...")
            # Run synchronously so a failure (e.g. "Interactive authentication
            # required" when the unit runs as an unprivileged user) is detected
            # and we can fall back. On success systemd SIGTERMs this process
            # while we wait, so a zero exit is never actually observed here.
            for cmd in (["systemctl", "restart", service_name],
                        ["sudo", "-n", "systemctl", "restart", service_name]):
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                except Exception as e:
                    print(f"[RESTART] {' '.join(cmd[:2])} error: {e}")
                    continue
                if r.returncode == 0:
                    sleep(30)  # systemd is stopping us; don't race it with a fallback
                    return
                print(f"[RESTART] {' '.join(cmd[:2])} failed: {(r.stderr or r.stdout).strip()}")
            print("[RESTART] systemctl needs privileges this unit doesn't have; using in-place restart")
            break

    # Not under systemd: prefer restart_server.sh, which fully replaces the
    # process tree. Under systemd we must NOT use it — it would spawn a server
    # systemd doesn't manage — so an in-place execv keeps the unit's PID lineage.
    script_dir = APP_DIR
    restart_script = os.path.join(script_dir, "restart_server.sh")

    if not under_systemd and os.path.exists(restart_script):
        print("[RESTART] Calling restart_server.sh...")
        subprocess.Popen(
            ["bash", restart_script],
            cwd=script_dir,
            start_new_session=True,
        )
        sleep(2)
        os._exit(0)
    else:
        print("[RESTART] Restarting in place via execv...")
        # Children hold the bound server socket across an execv; reap them
        # first or the re-exec'd server can't rebind its port.
        _terminate_child_processes()
        sys.stdout.flush()
        sys.stderr.flush()
        # werkzeug marks its listen socket inheritable (run_simple calls
        # set_inheritable(True) for reloader fd-passing even with the reloader
        # off), so it would survive the execv and the fresh process could never
        # rebind. Close every fd above stderr; nothing may outlive the exec.
        os.environ.pop("WERKZEUG_SERVER_FD", None)
        try:
            import resource
            maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            if maxfd == resource.RLIM_INFINITY or maxfd > (1 << 20):
                maxfd = 1 << 20
        except Exception:
            maxfd = 65536
        os.closerange(3, maxfd)
        os.execv(sys.executable, [sys.executable, *sys.argv])


@app.route("/api/server/restart", methods=["POST"])
def restart_server():
    """API endpoint to restart the entire server (full application restart).

    Use this for:
    - Applying config changes that require full restart (port, host, etc.)
    - Loading new dependencies or major updates

    For just restarting transcription (model changes), use /api/restart instead.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Return success response before restarting
        response = jsonify(
            {
                "success": True,
                "message": "Server is restarting... Please wait 10-15 seconds and refresh the page.",
            }
        )

        # Schedule restart after response is sent
        def do_restart():
            sleep(1)  # Wait for response to be sent
            print("[RESTART] Server restart requested via API")
            perform_server_restart()

        restart_thread = threading.Thread(target=do_restart, daemon=True)
        restart_thread.start()

        return response

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/server/update", methods=["POST"])
def update_server():
    """Pull the latest code (git fast-forward) and, if it advanced, restart to
    apply it — the web-UI equivalent of update_server.sh. Reuses the same
    non-destructive self-update the nightly auto-update uses."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    # Frozen/watchdog installs update via releases, not a git pull.
    if _is_watchdog_managed():
        return jsonify({"success": False,
                        "message": "This install is managed by the watchdog; it updates via releases, not a git pull."})

    # Don't yank code + restart out from under a live session.
    if _ts_get("running"):
        return jsonify({"success": False,
                        "message": "Transcription is running — stop it before updating."})

    try:
        from stt.self_update import git_self_update
        updated, reason = git_self_update(BUNDLE_DIR)

        if not updated:
            messages = {
                "up-to-date": "Already up to date — no new code to apply.",
                "dirty-worktree": "Local uncommitted changes are present; not updating (nothing was changed).",
                "not-fast-forwardable": "The checkout has diverged or unpushed commits; not updating (nothing was changed).",
                "not-a-git-checkout": "This is not a git checkout, so it can't self-update.",
                "error": "Update check failed — see the server log.",
            }
            return jsonify({"success": True, "updated": False,
                            "message": messages.get(reason, f"No update applied ({reason}).")})

        # Advanced to new code — restart to load it (same pattern as /restart).
        def do_update_restart():
            sleep(1)  # let the response flush first
            print("[UPDATE] Update pulled via API; restarting to apply")
            perform_server_restart()

        threading.Thread(target=do_update_restart, daemon=True).start()
        return jsonify({"success": True, "updated": True,
                        "message": "Updated to the latest code. Restarting… wait 10-15 seconds and refresh."})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/server/access-token", methods=["POST"])
def generate_access_token():
    """Return a fresh random URL access token for the settings UI to save into
    web_server.access_token. Does not persist it — the normal settings save does."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    return jsonify({"success": True, "token": secrets.token_urlsafe(24)})


@app.route("/api/disk-space", methods=["GET"])
def get_disk_space():
    """API endpoint to get disk space information (cross-platform)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Get the current working directory path
        current_path = APP_DIR

        # Get disk usage statistics using shutil (works on both Windows and Linux)
        disk_usage = shutil.disk_usage(current_path)

        # Calculate values in bytes
        total_bytes = disk_usage.total
        used_bytes = disk_usage.used
        free_bytes = disk_usage.free

        # Calculate percentage used
        percent_used = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0

        # Helper function to format bytes to human-readable format
        def format_bytes(bytes_value):
            """Convert bytes to human-readable format"""
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if bytes_value < 1024.0:
                    return f"{bytes_value:.2f} {unit}"
                bytes_value /= 1024.0
            return f"{bytes_value:.2f} PB"

        return jsonify(
            {
                "success": True,
                "disk_space": {
                    "total": total_bytes,
                    "used": used_bytes,
                    "free": free_bytes,
                    "percent_used": round(percent_used, 2),
                    "total_formatted": format_bytes(total_bytes),
                    "used_formatted": format_bytes(used_bytes),
                    "free_formatted": format_bytes(free_bytes),
                    "path": current_path,
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/endpoints", methods=["GET"])
def get_api_endpoints():
    """API endpoint to list all available API endpoints (auto-generated from Flask routes)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        endpoints = []
        for rule in app.url_map.iter_rules():
            # Skip static files and internal endpoints
            if rule.endpoint == "static" or rule.rule.startswith("/static"):
                continue
            # Get methods, excluding HEAD and OPTIONS
            methods = list(rule.methods - {"HEAD", "OPTIONS"})
            if methods:  # Only include if there are actual methods
                # Get description and examples from docstring
                description = ""
                examples = []
                view_func = app.view_functions.get(rule.endpoint)
                auth_required = False
                if view_func and view_func.__doc__:
                    doc_lines = view_func.__doc__.strip().split('\n')
                    # First line is description
                    description = doc_lines[0].strip()
                    # Lines starting with "Example:" are examples
                    for line in doc_lines[1:]:
                        line = line.strip()
                        if line.startswith("Example:"):
                            examples.append(line[8:].strip())
                if view_func:
                    try:
                        import inspect
                        src = inspect.getsource(view_func)
                        auth_required = "check_ip_whitelist()" in src
                    except Exception:
                        pass

                endpoints.append({
                    "path": rule.rule,
                    "methods": sorted(methods),
                    "description": description,
                    "examples": examples,
                    "auth_required": auth_required,
                })

        # Sort by path for consistent ordering
        endpoints.sort(key=lambda x: x["path"])

        return jsonify({"success": True, "endpoints": endpoints})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/browse", methods=["GET"])
def browse_files():
    """API endpoint to browse files and directories"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        # Get query parameters
        path = request.args.get("path", APP_DIR)
        show_hidden = request.args.get("show_hidden", "false").lower() == "true"

        # Confine browsing to APP_DIR (resolves symlinks so links inside the
        # tree can't be used to escape it).
        path = safe_managed_path(path)
        if path is None:
            return jsonify({"success": False, "error": "Access denied"}), 403

        # Security check: ensure path is not trying to escape
        if not os.path.exists(path):
            return jsonify({"success": False, "error": "Path does not exist"}), 404

        if not os.path.isdir(path):
            return jsonify({"success": False, "error": "Path is not a directory"}), 400

        # Get parent directory — never above the confinement root
        app_root = os.path.realpath(APP_DIR)
        parent_dir = os.path.dirname(path) if path != app_root else None

        # Get hidden items from config
        hidden_items = config.get("file_manager", {}).get("hidden_items", [])
        working_dir = APP_DIR

        # Default-clean root view: at the app root only allowlisted items are
        # shown (backups by default) so end users don't wade through app
        # internals. "Show hidden items" reveals everything; deeper folders
        # browse normally; empty list disables the allowlist.
        root_visible = list(config.get("file_manager", {}).get("root_visible_items", ["_AUTOMATIC_BACKUP"]))
        _custom_db = (config.get("database", {}).get("path", "") or "").strip()
        if _custom_db:
            _db_abs = os.path.abspath(_custom_db if os.path.isabs(_custom_db) else os.path.join(APP_DIR, _custom_db))
            if _db_abs.startswith(os.path.abspath(APP_DIR) + os.sep):
                # A custom backup location inside the app dir stays visible
                root_visible.append(os.path.relpath(_db_abs, APP_DIR).split(os.sep)[0])
        limit_to_visible = (not show_hidden and root_visible
                            and os.path.abspath(path) == os.path.abspath(APP_DIR))

        # List directory contents
        items = []
        try:
            for item_name in os.listdir(path):
                item_path = os.path.join(path, item_name)

                # Skip dotfiles unless show_hidden is True
                if not show_hidden and item_name.startswith("."):
                    continue

                # Skip __pycache__ directories unless show_hidden is True
                if not show_hidden and item_name == "__pycache__":
                    continue

                # At the app root, show only allowlisted items unless show_hidden
                if limit_to_visible and item_name not in root_visible:
                    continue

                # Skip items in hidden list unless show_hidden is True
                if not show_hidden:
                    try:
                        rel_path = os.path.relpath(item_path, working_dir)
                    except ValueError:
                        rel_path = item_path

                    if rel_path in hidden_items:
                        continue

                try:
                    stat_info = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)

                    # Format file size
                    size = stat_info.st_size
                    size_formatted = format_file_size(size) if not is_dir else "-"

                    # Format modification time
                    modified = datetime.fromtimestamp(stat_info.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Check if item is in hidden list
                    try:
                        rel_path = os.path.relpath(item_path, working_dir)
                    except ValueError:
                        rel_path = item_path
                    is_hidden = rel_path in hidden_items

                    items.append(
                        {
                            "name": item_name,
                            "path": item_path,
                            "type": "directory" if is_dir else "file",
                            "size": size if not is_dir else 0,
                            "size_formatted": size_formatted,
                            "modified": modified,
                            "extension": os.path.splitext(item_name)[1]
                            if not is_dir
                            else "",
                            "is_hidden": is_hidden,
                        }
                    )
                except (PermissionError, OSError):
                    # Skip items we can't access
                    continue
        except PermissionError:
            return jsonify({"success": False, "error": "Permission denied"}), 403

        # Sort: directories first, then files, alphabetically
        items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

        return jsonify(
            {
                "success": True,
                "current_path": path,
                "parent_path": parent_dir,
                "items": items,
                "show_hidden": show_hidden,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/file-manager/delete", methods=["POST"])
def delete_file():
    """API endpoint to delete a file or directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        data = request.get_json()
        path = data.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Confine to APP_DIR (resolves symlinks)
        path = safe_managed_path(path)
        if path is None:
            return jsonify({"success": False, "error": "Access denied"}), 403

        # Security check: prevent deletion of critical files
        if path == os.path.realpath(APP_DIR) or path in [
            os.path.realpath(os.path.join(APP_DIR, "speech_to_text.py")),
            os.path.realpath(CONFIG_FILE),
        ]:
            return jsonify(
                {
                    "success": False,
                    "error": "Cannot delete critical files or current directory",
                }
            ), 403

        if not os.path.exists(path):
            return jsonify({"success": False, "error": "Path does not exist"}), 404

        # Delete file or directory
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

        return jsonify({"success": True, "message": "Deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/rename", methods=["POST"])
def rename_file():
    """API endpoint to rename a file or directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        data = request.get_json()
        old_path = data.get("old_path")
        new_name = data.get("new_name")

        if not old_path or not new_name:
            return jsonify(
                {"success": False, "error": "Old path and new name are required"}
            ), 400

        # Confine source to APP_DIR
        old_path = safe_managed_path(old_path)
        if old_path is None:
            return jsonify({"success": False, "error": "Access denied"}), 403

        if not os.path.exists(old_path):
            return jsonify({"success": False, "error": "Path does not exist"}), 404

        # Create new path — reject a new_name that contains path separators or
        # otherwise resolves outside the source's directory.
        parent_dir = os.path.dirname(old_path)
        new_path = safe_managed_path(os.path.join(parent_dir, new_name), base_dir=parent_dir)
        if new_path is None or os.path.dirname(new_path) != parent_dir:
            return jsonify({"success": False, "error": "Invalid new name"}), 400

        if os.path.exists(new_path):
            return jsonify(
                {
                    "success": False,
                    "error": "A file or directory with that name already exists",
                }
            ), 400

        # Rename
        os.rename(old_path, new_path)

        return jsonify(
            {"success": True, "message": "Renamed successfully", "new_path": new_path}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/create-folder", methods=["POST"])
def create_folder():
    """API endpoint to create a new folder"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        data = request.get_json()
        parent_path = data.get("parent_path")
        folder_name = data.get("folder_name")

        if not parent_path or not folder_name:
            return jsonify(
                {"success": False, "error": "Parent path and folder name are required"}
            ), 400

        # Confine parent to APP_DIR
        parent_path = safe_managed_path(parent_path)
        if parent_path is None:
            return jsonify({"success": False, "error": "Access denied"}), 403

        if not os.path.exists(parent_path):
            return jsonify(
                {"success": False, "error": "Parent directory does not exist"}
            ), 404

        if not os.path.isdir(parent_path):
            return jsonify(
                {"success": False, "error": "Parent path is not a directory"}
            ), 400

        # Create new folder path — folder_name must stay directly under parent
        new_folder_path = safe_managed_path(os.path.join(parent_path, folder_name), base_dir=parent_path)
        if new_folder_path is None or os.path.dirname(new_folder_path) != parent_path:
            return jsonify({"success": False, "error": "Invalid folder name"}), 400

        if os.path.exists(new_folder_path):
            return jsonify(
                {
                    "success": False,
                    "error": "A file or directory with that name already exists",
                }
            ), 400

        # Create folder
        os.makedirs(new_folder_path)

        return jsonify(
            {
                "success": True,
                "message": "Folder created successfully",
                "path": new_folder_path,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/cleanup-sidecars", methods=["POST"])
def file_manager_cleanup_sidecars():
    """Retire -wal/-shm files left beside finished session databases.

    Runs automatically at startup; exposed here so an operator who notices them
    can clear them without waiting for a restart, and see what happened.
    """
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    result = sweep_db_sidecars()
    return jsonify({"success": True, **result})


@app.route("/api/file-manager/hidden-items", methods=["GET"])
def get_hidden_items():
    """API endpoint to get list of hidden files/folders"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        hidden_items = config.get("file_manager", {}).get("hidden_items", [])
        return jsonify({"success": True, "hidden_items": hidden_items})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/hide", methods=["POST"])
def hide_item():
    """API endpoint to hide a specific file or folder"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        global config
        data = _control_params(keep_blank=True)
        path = data.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Confine to the managed tree (rejects ../ and symlink escapes) — same
        # guard the sibling file-manager handlers use — then store relative to APP_DIR.
        resolved = safe_managed_path(path)
        if resolved is None:
            return jsonify({"success": False, "error": "Access denied"}), 403
        try:
            rel_path = os.path.relpath(resolved, APP_DIR)
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            rel_path = resolved

        # Get current hidden items
        if "file_manager" not in config:
            config["file_manager"] = {}

        hidden_items = config["file_manager"].get("hidden_items", [])

        # Add if not already hidden
        if rel_path not in hidden_items:
            hidden_items.append(rel_path)
            config["file_manager"]["hidden_items"] = hidden_items
            save_config(config)

        return jsonify(
            {
                "success": True,
                "message": "Item hidden successfully",
                "hidden_items": hidden_items,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/unhide", methods=["POST"])
def unhide_item():
    """API endpoint to unhide a specific file or folder"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        global config
        data = _control_params(keep_blank=True)
        path = data.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Confine to the managed tree, then store relative to APP_DIR.
        resolved = safe_managed_path(path)
        if resolved is None:
            return jsonify({"success": False, "error": "Access denied"}), 403
        try:
            rel_path = os.path.relpath(resolved, APP_DIR)
        except ValueError:
            rel_path = resolved

        # Get current hidden items
        if "file_manager" not in config:
            config["file_manager"] = {}

        hidden_items = config["file_manager"].get("hidden_items", [])

        # Remove if hidden
        if rel_path in hidden_items:
            hidden_items.remove(rel_path)
            config["file_manager"]["hidden_items"] = hidden_items
            save_config(config)

        return jsonify(
            {
                "success": True,
                "message": "Item unhidden successfully",
                "hidden_items": hidden_items,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-manager/download", methods=["GET"])
def download_file():
    """API endpoint to download a file"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    try:
        path = request.args.get("path")

        if not path:
            return jsonify({"success": False, "error": "Path is required"}), 400

        # Security: confine to APP_DIR. safe_managed_path uses commonpath +
        # realpath, so a sibling dir sharing the prefix (e.g. STT_secrets) and
        # symlink escapes are both rejected — unlike a bare startswith check.
        abs_path = safe_managed_path(path)
        if abs_path is None:
            return jsonify({"success": False, "error": "Access denied"}), 403

        # Check if file exists
        if not os.path.exists(abs_path):
            return jsonify({"success": False, "error": "File not found"}), 404

        # Check if it's a file (not a directory)
        if not os.path.isfile(abs_path):
            return jsonify({"success": False, "error": "Not a file"}), 400

        # Send the file
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename, as_attachment=True)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Output formats offered for on-the-fly conversion (fixed allowlist → no shell
# injection): fmt -> (ffmpeg -f value, file extension, mime type).
_CONVERT_FORMATS = {
    "mp3": ("mp3", "mp3", "audio/mpeg"),
    "wav": ("wav", "wav", "audio/wav"),
}


@app.route("/api/file-manager/convert-download", methods=["GET"])
def convert_download():
    """Transcode a media file (e.g. the MPEG-TS capture backup, which browsers
    can't play) to MP3/WAV and stream it as a download. ffmpeg runs on the
    confined input with a fixed-allowlist output format — no shell injection or
    traversal."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    path = request.args.get("path")
    if not path:
        return jsonify({"success": False, "error": "Path is required"}), 400

    fmt = (request.args.get("format") or "mp3").lower()
    if fmt not in _CONVERT_FORMATS:
        return jsonify({"success": False, "error": "Unsupported format"}), 400
    ff_fmt, ext, mimetype = _CONVERT_FORMATS[fmt]

    abs_path = safe_managed_path(path)  # commonpath + realpath confinement
    if abs_path is None:
        return jsonify({"success": False, "error": "Access denied"}), 403
    if not os.path.isfile(abs_path):
        return jsonify({"success": False, "error": "File not found"}), 404

    import shutil as _shutil
    import subprocess
    ffmpeg = _shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", abs_path, "-vn", "-f", ff_fmt, "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def _gen():
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            # Stop ffmpeg on client disconnect / completion so it never lingers.
            try:
                proc.kill()
            except Exception:
                pass

    stem = os.path.splitext(os.path.basename(abs_path))[0]
    return Response(
        stream_with_context(_gen()),
        mimetype=mimetype,
        headers={"Content-Disposition": f'attachment; filename="{stem}.{ext}"'},
    )


@app.route("/api/file-manager/preview-db", methods=["GET"])
def preview_db():
    """Preview a SQLite .db as a generic, schema-agnostic table dump: every
    column, every row (capped). Opens read-only so an actively-recording session
    DB is never locked or mutated. Deliberately unfiltered — denied/partial rows
    show too — and columns come from the live schema, so new migrations appear
    here with no code change."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    path = request.args.get("path")
    if not path:
        return jsonify({"success": False, "error": "Path is required"}), 400

    abs_path = safe_managed_path(path)  # commonpath + realpath confinement
    if abs_path is None:
        return jsonify({"success": False, "error": "Access denied"}), 403
    if not os.path.isfile(abs_path):
        return jsonify({"success": False, "error": "File not found"}), 404

    LIMIT_ROWS = 2000
    CELL_MAX = 300  # elide huge blobs (e.g. words_json) so the payload/table stay sane

    def _cell(v):
        if v is None:
            return ""
        s = v if isinstance(v, str) else str(v)
        return (s[:CELL_MAX] + "…") if len(s) > CELL_MAX else s

    try:
        with sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True) as conn:
            cur = conn.cursor()
            # Pick the table generically: prefer the session table, else the first
            # user table — so this also handles renames / non-transcription DBs.
            names = [r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
            if not names:
                return jsonify({"success": False, "error": "No tables in this database"}), 400
            table = "transcriptions" if "transcriptions" in names else names[0]
            quoted = '"' + table.replace('"', '""') + '"'

            total = cur.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            # Order by id when the table has one; otherwise leave natural order.
            has_id = any(c[1] == "id" for c in cur.execute(f"PRAGMA table_info({quoted})").fetchall())
            order = " ORDER BY id ASC" if has_id else ""
            cur.execute(f"SELECT * FROM {quoted}{order} LIMIT ?", (LIMIT_ROWS,))
            columns = [d[0] for d in cur.description]
            rows = [[_cell(v) for v in r] for r in cur.fetchall()]
    except sqlite3.Error:
        # Not a SQLite DB / unreadable / encrypted.
        return jsonify({"success": False, "error": "Could not read database"}), 400

    return jsonify({
        "success": True,
        "table": table,
        "columns": columns,
        "rows": rows,
        "total": total,
        "truncated": total > len(rows),
    })


@app.route("/api/file-manager/session-meta", methods=["GET"])
def session_meta_for_db():
    """Which models and settings produced a session, from its session_meta table.

    Opens read-only, so this is safe against a session that is still recording.
    Sessions recorded before provenance existed simply have no such table — that
    is normal and returns an empty mapping, not an error, so the UI can render an
    empty state instead of a failure. A database that could not be READ is
    reported separately: collapsing the two would make a healthy session look
    like an unrecorded one and send the reader after the wrong problem."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access denied"}), 403

    path = request.args.get("path")
    if not path:
        return jsonify({"success": False, "error": "Path is required"}), 400

    abs_path = safe_managed_path(path)  # commonpath + realpath confinement
    if abs_path is None:
        return jsonify({"success": False, "error": "Access denied"}), 403
    if not os.path.isfile(abs_path):
        return jsonify({"success": False, "error": "File not found"}), 404

    meta, read_error = _load_session_meta(abs_path)
    return jsonify({
        "success": True,
        "meta": meta,
        "recorded": bool(meta),
        "read_error": read_error,
    })


# File Mover Endpoints
@app.route("/api/file-mover/status", methods=["GET"])
def get_file_mover_status():
    """Get file mover status and configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        mover_config = config.get("file_manager", {}).get("file_mover", {})
        return jsonify({
            "success": True,
            "config": mover_config,
            "runtime": get_file_mover_runtime(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/configure", methods=["POST"])
def configure_file_mover():
    """Update file mover configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()

        if "file_manager" not in config:
            config["file_manager"] = {}
        if "file_mover" not in config["file_manager"]:
            config["file_manager"]["file_mover"] = {}

        # Update configuration
        mover_config = config["file_manager"]["file_mover"]

        if "enabled" in data:
            mover_config["enabled"] = bool(data["enabled"])
        if "move_on_transcription_stop" in data:
            mover_config["move_on_transcription_stop"] = bool(
                data["move_on_transcription_stop"]
            )
        if "destination_path" in data:
            mover_config["destination_path"] = data["destination_path"]
        if "smb_username" in data:
            mover_config["smb_username"] = data["smb_username"]
        if "smb_password" in data:
            mover_config["smb_password"] = data["smb_password"]
        if "smb_domain" in data:
            mover_config["smb_domain"] = data["smb_domain"]
        if "source_patterns" in data:
            mover_config["source_patterns"] = data["source_patterns"]
        if "delete_source" in data:
            mover_config["delete_source"] = bool(data["delete_source"])
        if "preserve_structure" in data:
            mover_config["preserve_structure"] = bool(data["preserve_structure"])
        if "retry_on_failure" in data:
            mover_config["retry_on_failure"] = bool(data["retry_on_failure"])

        save_config(config)

        return jsonify({"success": True, "message": "Configuration updated"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/test", methods=["POST"])
def test_file_mover_connection():
    """Test connection to destination"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        from stt.file_mover import test_destination_accessible

        data = request.get_json()
        dest_path = data.get("destination_path", "")
        username = data.get("smb_username", "")
        password = data.get("smb_password", "")
        domain = data.get("smb_domain", "")

        if not dest_path:
            return jsonify(
                {"success": False, "error": "No destination path provided"}
            ), 400

        accessible = test_destination_accessible(dest_path, username, password, domain)

        if accessible:
            return jsonify({"success": True, "message": "Destination is accessible"})
        else:
            return jsonify({"success": False, "error": "Destination is not accessible"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/trigger", methods=["POST"])
def trigger_file_mover_endpoint():
    """API endpoint to manually trigger file mover check"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Use the core file move function for consistency
        set_file_mover_running("manual")
        result = execute_file_move(lambda: config, APP_DIR)
        set_file_mover_result("manual", result)

        if not result['success']:
            return jsonify({
                "success": False,
                "error": result.get('message', 'Unknown error'),
                "errors": result.get('errors', [])
            }), 400

        return jsonify({
            "success": True,
            "moved": result['moved'],
            "failed": result['failed'],
            "errors": result.get('errors', []),
            "message": result['message'],
            "delete_source": result.get('delete_source', True)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/file-mover/browse-remote", methods=["POST"])
def browse_remote_destination():
    """Browse files in the remote SMB destination"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        from stt.file_mover import test_destination_accessible

        data = request.get_json()
        dest_path = data.get("destination_path", "").strip()
        username = data.get("smb_username", "").strip()
        password = data.get("smb_password", "")
        domain = data.get("smb_domain", "").strip()
        subpath = data.get("subpath", "").strip()

        if not dest_path:
            return jsonify(
                {"success": False, "error": "No destination path specified"}
            ), 400

        # Build full path with subpath
        if subpath:
            full_path = os.path.join(dest_path, subpath)
        else:
            full_path = dest_path

        # Test/mount the destination
        if not test_destination_accessible(dest_path, username, password, domain):
            return jsonify(
                {"success": False, "error": "Cannot access remote destination"}
            ), 400

        # On Linux, we need to use the mount point
        if platform == "linux":
            from stt.file_mover import is_smb_path

            if is_smb_path(dest_path):
                # Extract server and share to build mount point path
                unc_path = dest_path.replace("\\", "/")
                if not unc_path.startswith("//"):
                    unc_path = "//" + unc_path
                parts = unc_path.replace("//", "").split("/")
                if len(parts) >= 2:
                    server = parts[0]
                    share = parts[1]
                    if "darwin" in platform:
                        mount_point = f"/Volumes/{share}"
                    else:
                        mount_point = f"/mnt/{server}_{share}"

                    # Replace dest_path with mount point
                    remaining_path = "/".join(parts[2:]) if len(parts) > 2 else ""
                    if subpath:
                        full_path = os.path.join(mount_point, remaining_path, subpath)
                    elif remaining_path:
                        full_path = os.path.join(mount_point, remaining_path)
                    else:
                        full_path = mount_point

        # List files in the remote path
        items = []
        try:
            for item_name in os.listdir(full_path):
                item_path = os.path.join(full_path, item_name)

                try:
                    stat_info = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)

                    size = stat_info.st_size
                    size_formatted = format_file_size(size) if not is_dir else "-"
                    modified = datetime.fromtimestamp(stat_info.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    items.append(
                        {
                            "name": item_name,
                            "path": os.path.join(subpath, item_name)
                            if subpath
                            else item_name,
                            "type": "directory" if is_dir else "file",
                            "size": size if not is_dir else 0,
                            "size_formatted": size_formatted,
                            "modified": modified,
                            "extension": os.path.splitext(item_name)[1]
                            if not is_dir
                            else "",
                        }
                    )
                except (PermissionError, OSError):
                    continue
        except PermissionError:
            return jsonify({"success": False, "error": "Permission denied"}), 403

        # Sort: directories first, then files
        items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

        # Calculate parent path
        parent_path = os.path.dirname(subpath) if subpath else ""

        return jsonify(
            {
                "success": True,
                "current_path": subpath,
                "parent_path": parent_path,
                "items": items,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/start", methods=["POST"])
def start_transcription():
    """Start the transcription process
    Example: POST /api/transcription/start"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_process, transcription_state
    try:
        with _transcription_start_lock:
            worker_dead = transcription_process is None or not transcription_process.is_alive()

            if transcription_state["running"]:
                if worker_dead:
                    # Worker crashed mid-run and never reset the state — recover
                    # instead of telling the user "already running" forever
                    print("[START] State says running but worker is dead — resetting state")
                    transcription_state["running"] = False
                    transcription_state["status"] = "stopped"
                else:
                    return jsonify(
                        {"success": False, "error": "Transcription is already running"}
                    ), 400

            # Don't start if still stopping (unless the worker is gone)
            if transcription_state["status"] == "stopping" and not worker_dead:
                return jsonify(
                    {"success": False, "error": "Transcription is still stopping, please wait"}
                ), 400

            # Ensure we have a valid worker process
            # Worker stays alive between Start/Stop cycles, so we usually just reuse it
            if worker_dead:
                # Worker doesn't exist or crashed - create a new one
                # This should only happen on first start or if worker unexpectedly died
                print("[START] Worker process not running, creating new worker...")
                transcription_process = multiprocessing.Process(
                    target=thread1_function,
                    args=(transcription_state, control_queue, config_queue,
                          calibration_state, calibration_data_shared, calibration_step1_data,
                          audio_stream_queue)
                )
                transcription_process.start()
                globals()["thread1"] = transcription_process
            else:
                # Worker is alive, just reuse it (it's waiting in idle loop)
                print(f"[START] Reusing existing worker process PID={transcription_process.pid}")

        # Send start command through queue
        control_queue.put({"command": "start"})

        # Update state - don't set running=True yet, worker will do that after initialization
        transcription_state["status"] = "starting"
        transcription_state["message"] = (
            "Initializing audio interface and loading model..."
        )
        transcription_state["error"] = None  # Clear any previous errors

        # Anonymous live-map ping so ChurchPresenter can see where STT is used.
        # Location is derived (and fuzzed) server-side from the connection; we send
        # only version + os + a stable per-install id used to dedupe the map.
        install_id = _get_install_id()

        def _notify_livemap():
            try:
                ep = (config.get("analytics", {}).get("endpoint") or "").strip()
                if not ep:
                    return
                # Numeric part of the display version only (e.g. '26.1.22' from
                # '26.1.22-gc588d29') — the map server rejects anything beyond
                # dotted numerics, and the commit hash is sent separately below.
                # A self-update re-execs the process, so this is always current.
                version = (SERVER_DISPLAY_VERSION or "").split("-", 1)[0] or SERVER_VERSION or "unknown"
                os_name = {"darwin": "macos", "win32": "windows", "linux": "linux"}.get(sys.platform, "linux")
                _lt = config.get("live_translation", {})
                _remote = _lt.get("remote", {})
                offloaded = bool(_remote.get("enabled") and _remote.get("endpoint"))
                # Languages in use at transcription start ('auto' when auto-detected).
                transcribe_lang = (config.get("audio", {}).get("language") or "auto").strip() or "auto"
                translate_lang = ((_lt.get("target_language") or "").strip() or "unknown") if _lt.get("enabled") else "none"
                url = (f"{ep}?os={os_name}&version={version}"
                       f"&transcribe_lang={transcribe_lang}&translate_lang={translate_lang}")
                if SERVER_COMMIT:
                    url += f"&commit={SERVER_COMMIT}"
                if offloaded:
                    url += "&offloaded=1"
                import requests as _req
                _req.get(url, headers={"X-Install-Id": install_id}, timeout=10)
            except Exception as e:
                print(f"[START] Live-map ping failed: {e}")
        import threading
        threading.Thread(target=_notify_livemap, daemon=True).start()

        # Warm up the translation model so the first translated segment doesn't
        # pay the load cost. Remote setups preload on Machine B; otherwise, for a
        # local seq2seq model (not Whisper-based translation), preload here.
        trans_cfg = config.get("live_translation", {})
        remote_cfg = trans_cfg.get("remote", {})
        remote_active = bool(remote_cfg.get("enabled") and remote_cfg.get("endpoint"))
        if remote_active:
            def _notify_remote_preload():
                try:
                    ep = _get_remote_endpoint()
                    if not ep:
                        return
                    import requests as _req
                    r = _req.post(ep + "/api/translate/preload", timeout=10)
                    print(f"[START] Remote translation preload: {r.json()}")
                    r2 = _req.post(ep + "/api/translate/sync-dictionary",
                                   json=_dictionary_sync_payload(), timeout=10)
                    print(f"[START] Remote dictionary sync: {r2.json()}")
                except Exception as e:
                    print(f"[START] Remote translation preload/sync failed: {e}")
            import threading
            threading.Thread(target=_notify_remote_preload, daemon=True).start()
        elif trans_cfg.get("enabled") and trans_cfg.get("translation_method", "nllb") == "llm":
            # Warm the LLM, and deliberately NOT the NMT model behind it.
            #
            # Preloading the fallback costs ~3.5 GB of VRAM for a model we hope never to
            # use, and measured on a 10 GB card that starves the thing we do use: with
            # Whisper (4202 MiB) and the NMT model (3558 MiB) resident, 1976 MiB was
            # left, the LLM server could not fit a 5 GB model, and every caption timed
            # out into the fallback — the LLM never ran at all. The NMT model still
            # loads lazily the first time the LLM declines, so the fallback survives;
            # it just stops pre-paying for itself.
            def _warm_llm():
                try:
                    target = trans_cfg.get("target_language", "en")
                    warm_s = coerce_float(
                        (trans_cfg.get("llm") or {}).get("warmup_timeout_ms"), 180000,
                        lo=5000, hi=600000) / 1000.0
                    if _translate_via_llm("Здравствуйте.", "ru", target,
                                          timeout_override=warm_s) is not None:
                        print("[START] LLM translation warmed and pinned")
                    else:
                        print("[START] LLM warm-up returned no usable text; "
                              "captions will fall back to the NMT model")
                except Exception as e:
                    print(f"[START] LLM warm-up failed: {e}")
            import threading
            threading.Thread(target=_warm_llm, daemon=True).start()
        elif trans_cfg.get("enabled") and trans_cfg.get("translation_method", "nllb") not in (
            "whisper_translate", "whisper_forced_lang"
        ):
            def _preload_local_translation():
                try:
                    use_gpu = trans_cfg.get("use_gpu", True)
                    model_id = trans_cfg.get("translation_model")
                    get_live_translation_model(use_gpu, model_id=model_id)
                    print("[START] Local translation model preloaded")
                except Exception as e:
                    print(f"[START] Local translation preload failed: {e}")
            import threading
            threading.Thread(target=_preload_local_translation, daemon=True).start()

        return jsonify(
            {
                "success": True,
                "message": "Transcription starting...",
                "state": _ts_snapshot(),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/stop", methods=["POST"])
def stop_transcription():
    """Stop the transcription process
    Example: POST /api/transcription/stop"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_state, transcription_process
    try:
        # Serialize the guard + state transition with start_transcription so two
        # concurrent stops (or a stop racing a start) can't both pass the check.
        # The first stop flips status to "stopping", which makes any second stop
        # fail the guard below.
        with _transcription_start_lock:
            if not transcription_state["running"] and transcription_state["status"] != "starting":
                return jsonify(
                    {"success": False, "error": "Transcription is not running"}
                ), 400

            # Send stop command through queue
            control_queue.put({"command": "stop"})

            # Update state
            transcription_state["running"] = False
            transcription_state["status"] = "stopping"
            transcription_state["message"] = (
                "Stopping transcription, unloading model, closing connections..."
            )
            transcription_state["loaded_model"] = ""  # Clear loaded model name
            transcription_state["live_text"] = ""  # Clear in-progress text
            transcription_state["live_start"] = 0
            transcription_state["live_end"] = 0

        # Clear database cache immediately so clients get empty data
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        # Unload Live Translation model synchronously to free GPU memory
        # Do this BEFORE starting cleanup thread to avoid CUDA conflicts
        if is_live_translation_model_loaded():
            print("[STOP] Unloading Live Translation model...")
            unload_live_translation_model()
            print("[STOP] Live Translation model unloaded")

        # The in-process GGUF is a separate engine with a separate releaser, and it
        # holds as much memory as the NMT model does. Freeing only the NMT one left
        # it resident until the process restarted.
        if _uses_local_llm(config.get("live_translation", {})) and is_local_llm_loaded():
            print("[STOP] Unloading local LLM translation model...")
            unload_local_llm()

        # Tell remote Machine B to unload its translation model too
        remote_cfg = config.get("live_translation", {}).get("remote", {})
        if remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
            def _notify_remote_unload():
                try:
                    ep = _get_remote_endpoint()
                    if not ep:
                        return
                    import requests as _req
                    r = _req.post(ep + "/api/translate/unload", timeout=10)
                    print(f"[STOP] Remote translation unload: {r.json()}")
                except Exception as e:
                    print(f"[STOP] Remote translation unload failed: {e}")
            import threading
            threading.Thread(target=_notify_remote_unload, daemon=True).start()

        if is_tts_model_loaded():
            print("[STOP] Unloading TTS model...")
            unload_tts_model()
            print("[STOP] TTS model unloaded")

        # Background cleanup to send unload command and update status
        # NOTE: We keep the worker process alive! This avoids CUDA fork issues on subsequent starts.
        # The worker will unload its models (Whisper, VAD) when it receives the unload command,
        # which releases GPU memory. The worker stays in its idle loop, ready for the next Start.
        def cleanup_process():
            """Background cleanup to send unload command and update status"""
            import time
            _log_path = os.path.join(APP_DIR, "server.log")
            def log(msg):
                with open(_log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                    f.flush()

            log("[STOP-CLEANUP] Thread started")

            time.sleep(2)  # Wait for graceful shutdown of transcription loop

            # Send unload command to worker to release GPU memory
            # Worker stays alive to avoid CUDA fork issues on subsequent starts
            log("[STOP-CLEANUP] Sending unload command to worker...")
            try:
                control_queue.put({"command": "unload"})
                log("[STOP-CLEANUP] Unload command sent to worker")
            except Exception as e:
                log(f"[STOP-CLEANUP] Error sending unload command: {e}")

            with _transcription_state_lock:
                if transcription_state["status"] == "stopping":
                    transcription_state["status"] = "stopped"
                    transcription_state["message"] = "Transcription stopped"
                    # Drop any file-playback markers so a stale duration bar
                    # can't reappear on the next (possibly mic) session.
                    transcription_state["is_file_playback"] = False
                    transcription_state["playback_source"] = None
                    transcription_state["playback_duration"] = None

        # Run cleanup in background thread
        import threading
        _server_log = os.path.join(APP_DIR, "server.log")
        # Write directly to log file since stdout might be buffered
        with open(_server_log, "a", encoding="utf-8") as logf:
            logf.write(f"[STOP] Creating cleanup thread, transcription_process={transcription_process}\n")
            logf.flush()
        cleanup_thread = threading.Thread(target=cleanup_process, daemon=True)
        cleanup_thread.start()
        with open(_server_log, "a", encoding="utf-8") as logf:
            logf.write("[STOP] Cleanup thread started\n")
            logf.flush()

        return jsonify(
            {
                "success": True,
                "message": "Transcription stopping...",
                "state": _ts_snapshot(),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/status", methods=["GET"])
def get_transcription_status():
    """API endpoint to get transcription status"""
    return jsonify(
        {
            "success": True,
            "state": _ts_snapshot(),
            "setup": _setup_status(),
            "disk_percent": _disk_usage_percent(),
        }
    )


@app.route("/api/transcription/force-reset", methods=["POST"])
def force_reset_transcription():
    """API endpoint to force reset transcription state (emergency use)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global transcription_state, transcription_process

    try:
        print("[FORCE RESET] Forcing transcription state reset...")

        # Force kill the transcription process if it exists
        if transcription_process is not None and transcription_process.is_alive():
            print("[FORCE RESET] Terminating transcription process...")
            try:
                transcription_process.terminate()
                transcription_process.join(timeout=3)
            except (OSError, ProcessLookupError):
                pass

            # If still alive, force kill
            if transcription_process.is_alive():
                print("[FORCE RESET] Force killing transcription process...")
                try:
                    transcription_process.kill()
                    transcription_process.join(timeout=2)
                except (OSError, ProcessLookupError):
                    pass

        # Clear the control queue
        import queue as _queue_mod
        while not control_queue.empty():
            try:
                control_queue.get_nowait()
            except _queue_mod.Empty:
                break

        # Reset transcription state
        transcription_state["running"] = False
        transcription_state["status"] = "stopped"
        transcription_state["message"] = "Transcription forcefully reset"
        transcription_state["loaded_model"] = ""
        transcription_state["is_file_playback"] = False
        transcription_state["playback_source"] = None
        transcription_state["playback_duration"] = None
        transcription_state["live_text"] = ""
        transcription_state["live_start"] = 0
        transcription_state["live_end"] = 0
        transcription_state["db_name"] = None
        transcription_state["session_id"] = None

        # Clear database cache immediately so clients get empty data
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        # Restart the transcription process
        transcription_process = multiprocessing.Process(
            target=thread1_function,
            args=(transcription_state, control_queue, config_queue,
                  calibration_state, calibration_data_shared, calibration_step1_data,
                  audio_stream_queue)
        )
        transcription_process.start()

        # Update global reference for signal handler
        globals()["thread1"] = transcription_process

        print("[FORCE RESET] Transcription process reset complete")

        return jsonify(
            {
                "success": True,
                "message": "Transcription state forcefully reset. You can now start transcription again.",
                "state": _ts_snapshot(),
            }
        )
    except Exception as e:
        print(f"[FORCE RESET ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Real-Time Corrections API Endpoints
# =============================================================================


@app.route("/api/transcription/correct", methods=["POST"])
def correct_transcription():
    """Correct a transcription segment's text"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    segment_id = data.get("segment_id")
    new_text = data.get("new_text", "").strip()
    correction_type = data.get("correction_type", "manual")

    if segment_id is None or not new_text:
        return jsonify({"success": False, "error": "segment_id and new_text are required"}), 400

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return jsonify({"success": False, "error": "No active database"}), 400

    try:
        # 30s timeout + busy_timeout so a correction submitted during a heavy
        # transcription write burst waits for the worker's lock instead of
        # failing fast with "database is locked" (default timeout is 5s).
        with sqlite3.connect(current_db_name, timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            cursor = conn.cursor()
            # Store original text before correction (only on first correction)
            cursor.execute(
                """UPDATE transcriptions
                   SET original_text = COALESCE(original_text, text),
                       text = ?,
                       corrected_by = ?
                   WHERE id = ?""",
                (new_text, correction_type, segment_id),
            )
            conn.commit()

            if cursor.rowcount == 0:
                return jsonify({"success": False, "error": "Segment not found"}), 404

        # Invalidate DB cache so next emit picks up the change
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        # Invalidate translation cache for this segment
        cache = get_translation_cache()
        cache.invalidate(segment_id)

        # Re-translate if translation is active
        translated_text = None
        trans_config = config.get("live_translation", {})
        if trans_config.get("enabled", False):
            target_lang = trans_config.get("target_language", "en")
            source_lang = trans_config.get("source_language", "auto")
            if source_lang == "auto":
                source_lang = config.get("audio", {}).get("language", "en")
                if source_lang == "auto":
                    source_lang = "en"
            translated_text = translate_live_text(new_text, source_lang, target_lang)
            if translated_text:
                cache.set(segment_id, new_text, translated_text, target_lang)

        # Emit correction event to all clients
        socketio.emit("correction_applied", {
            "segment_id": segment_id,
            "new_text": new_text,
            "corrected_by": correction_type,
            "translated_text": translated_text,
        })

        return jsonify({"success": True, "segment_id": segment_id, "new_text": new_text, "translated_text": translated_text})

    except Exception as e:
        print(f"[CORRECTION ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/review-queue", methods=["GET"])
def get_review_queue():
    """Get segments that need review (low confidence)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return jsonify({"success": True, "segments": []})

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, timestamp, text, COALESCE(start_time, 0), COALESCE(end_time, 0), confidence
                   FROM transcriptions
                   WHERE needs_review = 1 AND COALESCE(denied, 0) = 0
                   AND COALESCE(is_final, 1) = 1
                   ORDER BY id DESC
                   LIMIT 50"""
            )
            rows = cursor.fetchall()

        segments = [
            {"id": r[0], "timestamp": r[1], "text": r[2], "start": r[3], "end": r[4], "confidence": r[5]}
            for r in rows
        ]
        return jsonify({"success": True, "segments": segments})

    except Exception as e:
        print(f"[REVIEW QUEUE ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcription/mark-reviewed", methods=["POST"])
def mark_reviewed():
    """Mark segments as reviewed (remove from review queue)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    segment_ids = data.get("segment_ids", [])
    if not segment_ids:
        return jsonify({"success": False, "error": "segment_ids required"}), 400

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return jsonify({"success": False, "error": "No active database"}), 400

    try:
        with sqlite3.connect(current_db_name, timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in segment_ids)
            cursor.execute(
                f"UPDATE transcriptions SET needs_review = 0 WHERE id IN ({placeholders})",
                segment_ids,
            )
            conn.commit()

        # Invalidate DB cache
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        return jsonify({"success": True, "updated": len(segment_ids)})

    except Exception as e:
        print(f"[MARK REVIEWED ERROR] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Custom Dictionary API Endpoints
# =============================================================================

_dictionary_cache = None
_dictionary_mtime = 0


def load_custom_dictionary():
    """Load custom dictionary from JSON file, with caching"""
    global _dictionary_cache, _dictionary_mtime

    dict_file = config.get("custom_dictionary", {}).get("file", "custom_dictionary.json")
    if not os.path.isabs(dict_file):
        dict_file = os.path.join(CONFIG_DIR, dict_file)

    try:
        if os.path.exists(dict_file):
            mtime = os.path.getmtime(dict_file)
            if _dictionary_cache is not None and mtime == _dictionary_mtime:
                return _dictionary_cache
            with open(dict_file, "r", encoding="utf-8") as f:
                import json as _json
                _dictionary_cache = _json.load(f)
                _dictionary_mtime = mtime
                return _dictionary_cache
        else:
            # Create default dictionary file if it doesn't exist
            default_dict = {"glossary": {}}
            import json as _json
            with open(dict_file, "w", encoding="utf-8") as f:
                _json.dump(default_dict, f, indent=2, ensure_ascii=False)
            _dictionary_cache = default_dict
            _dictionary_mtime = os.path.getmtime(dict_file)
            print(f"[DICTIONARY] Created default dictionary: {dict_file}")
            return default_dict
    except Exception as e:
        print(f"[DICTIONARY] Error loading dictionary: {e}")

    return {"glossary": {}}


def save_custom_dictionary(data):
    """Save custom dictionary to JSON file"""
    global _dictionary_cache, _dictionary_mtime

    dict_file = config.get("custom_dictionary", {}).get("file", "custom_dictionary.json")
    if not os.path.isabs(dict_file):
        dict_file = os.path.join(CONFIG_DIR, dict_file)

    try:
        import json as _json
        with open(dict_file, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, ensure_ascii=False)
        _dictionary_cache = data
        _dictionary_mtime = os.path.getmtime(dict_file)
        return True
    except Exception as e:
        print(f"[DICTIONARY] Error saving dictionary: {e}")
        return False


@app.route("/api/dictionary", methods=["GET"])
def get_dictionary():
    """Get the custom dictionary"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    return jsonify({"success": True, "dictionary": load_custom_dictionary()})


@app.route("/api/dictionary", methods=["POST"])
def update_dictionary():
    """Update the entire custom dictionary"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    dictionary = data.get("dictionary", {})
    if save_custom_dictionary(dictionary):
        _propagate_dictionary_to_remote()  # keep a paired offload server in sync
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Failed to save dictionary"}), 500


@app.route("/api/dictionary/glossary", methods=["POST"])
def update_glossary():
    """Add or update a glossary mapping"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    lang_pair = data.get("lang_pair", "").strip()  # e.g., "en_to_es"
    source_term = data.get("source_term", "").strip()
    target_term = data.get("target_term", "").strip()

    if not lang_pair or not source_term or not target_term:
        return jsonify({"success": False, "error": "lang_pair, source_term, and target_term are required"}), 400

    dictionary = load_custom_dictionary()
    dictionary.setdefault("glossary", {}).setdefault(lang_pair, {})[source_term] = target_term
    save_custom_dictionary(dictionary)
    _propagate_dictionary_to_remote()  # keep a paired offload server in sync

    return jsonify({"success": True, "glossary": dictionary["glossary"]})


@app.route("/api/dictionary/glossary", methods=["DELETE"])
def remove_glossary_entry():
    """Remove a glossary mapping"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    lang_pair = data.get("lang_pair", "").strip()
    source_term = data.get("source_term", "").strip()

    if not lang_pair or not source_term:
        return jsonify({"success": False, "error": "lang_pair and source_term are required"}), 400

    dictionary = load_custom_dictionary()
    glossary = dictionary.get("glossary", {}).get(lang_pair, {})
    if source_term in glossary:
        del glossary[source_term]
        save_custom_dictionary(dictionary)
        _propagate_dictionary_to_remote()  # keep a paired offload server in sync

    return jsonify({"success": True, "glossary": dictionary.get("glossary", {})})


@app.route("/api/corrections/settings", methods=["GET"])
def get_corrections_settings():
    """Get corrections configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403
    return jsonify({"success": True, "corrections": config.get("corrections", {})})


@app.route("/api/corrections/settings", methods=["POST"])
def update_corrections_settings():
    """Update corrections configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    corrections = data.get("corrections", {})
    config["corrections"] = {**config.get("corrections", {}), **corrections}
    save_config(config)
    return jsonify({"success": True, "corrections": config["corrections"]})


# =============================================================================
# Remote Mic API Endpoints
# =============================================================================

@app.route("/api/audio-devices", methods=["GET"])
def get_audio_devices():
    """API endpoint to get list of available audio input devices"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        from stt.audio_capture import list_audio_devices

        # Get backend from config
        audio_config = config.get("audio", {})
        backend = audio_config.get("backend", "ffmpeg")

        devices = []

        # Get devices using ffmpeg
        try:
            markers = audio_config.get("deprioritize_device_markers", [])
            devices = list_audio_devices(deprioritize_markers=markers)
            app_logger.info(f"Listed {len(devices)} devices using ffmpeg")

            # Normalize device format for UI compatibility
            normalized_devices = []
            for dev in devices:
                normalized_devices.append(
                    {
                        "index": dev.get("index", 0),
                        "name": dev.get(
                            "display_name", dev.get("name", "Unknown Device")
                        ),
                        "device_id": dev.get(
                            "name"
                        ),  # Actual device identifier for ffmpeg
                        "is_default": dev.get("is_default", False),
                    }
                )
            devices = normalized_devices

            # Prepend any .wav files sitting directly in APP_DIR as selectable test
            # sources (non-recursive — backups live in _AUTOMATIC_BACKUP/ subdirs and
            # are excluded so they don't flood the dropdown).
            import glob
            test_wavs = sorted(glob.glob(os.path.join(APP_DIR, "*.wav")))
            for i, wav_path in enumerate(test_wavs):
                devices.insert(0, {
                    "index": -1 - i,
                    "name": f"{os.path.basename(wav_path)} — Test Audio File",
                    "device_id": wav_path,
                    "is_default": False,
                })

        except Exception as e:
            app_logger.error(f"Error listing devices: {e}")
            devices = []

        # Ensure we have at least one device
        if not devices:
            devices = [
                {
                    "index": 0,
                    "name": "Default Microphone",
                    "device_id": "default",
                    "is_default": True,
                }
            ]

        return jsonify(
            {
                "success": True,
                "devices": devices,
                "default_index": 0,
                "backend": backend,
            }
        )

    except Exception:
        # Return fallback device
        return jsonify(
            {
                "success": True,
                "devices": [
                    {
                        "index": 0,
                        "name": "Default Microphone",
                        "is_default": True,
                    }
                ],
                "default_index": 0,
            }
        )


@app.route("/api/models/search", methods=["GET"])
def search_models():
    """Search for ASR models on Hugging Face"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        query = request.args.get("query", "whisper")
        limit = int(request.args.get("limit", 20))

        # Search Hugging Face for ASR models
        api = HfApi()
        models = api.list_models(
            filter="automatic-speech-recognition",
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit,
        )

        model_list = []
        for model in models:
            try:
                info = model_info(model.modelId)

                # Calculate approximate model size from safetensors files
                size_bytes = 0
                size_str = "Unknown"
                try:
                    if hasattr(info, "siblings") and info.siblings:
                        for sibling in info.siblings:
                            if hasattr(sibling, "rfilename") and hasattr(
                                sibling, "size"
                            ):
                                # Sum up sizes of model files
                                if any(
                                    ext in sibling.rfilename
                                    for ext in [".safetensors", ".bin", ".pt", ".onnx"]
                                ):
                                    size_bytes += sibling.size

                        # Convert to readable format
                        if size_bytes > 0:
                            size_mb = size_bytes / (1024 * 1024)
                            if size_mb > 1024:
                                size_str = f"{size_mb / 1024:.2f} GB"
                            else:
                                size_str = f"{size_mb:.0f} MB"
                except OSError:
                    pass

                model_list.append(
                    {
                        "id": model.modelId,
                        "downloads": model.downloads
                        if hasattr(model, "downloads")
                        else 0,
                        "likes": model.likes if hasattr(model, "likes") else 0,
                        "tags": model.tags if hasattr(model, "tags") else [],
                        "library": info.library_name
                        if hasattr(info, "library_name")
                        else "unknown",
                        "size": size_str,
                        "size_bytes": size_bytes,
                    }
                )
            except (KeyError, ValueError, OSError, AttributeError):
                model_list.append(
                    {
                        "id": model.modelId,
                        "downloads": model.downloads
                        if hasattr(model, "downloads")
                        else 0,
                        "likes": model.likes if hasattr(model, "likes") else 0,
                        "tags": model.tags if hasattr(model, "tags") else [],
                        "library": "unknown",
                        "size": "Unknown",
                        "size_bytes": 0,
                    }
                )

        return jsonify(
            {"success": True, "models": model_list, "count": len(model_list)}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@app.route("/api/models/popular", methods=["GET"])
def get_popular_models():
    """Get a curated list of popular ASR models"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    popular_models = {
        "Whisper (OpenAI)": [
            {
                "id": "openai/whisper-tiny",
                "description": "Fastest, smallest Whisper model (39M params)",
                "size": "39M",
            },
            {
                "id": "openai/whisper-base",
                "description": "Base Whisper model (74M params)",
                "size": "74M",
            },
            {
                "id": "openai/whisper-small",
                "description": "Small Whisper model (244M params)",
                "size": "244M",
            },
            {
                "id": "openai/whisper-medium",
                "description": "Medium Whisper model (769M params)",
                "size": "769M",
            },
            {
                "id": "openai/whisper-large-v3",
                "description": "Latest large Whisper model (1.5B params)",
                "size": "1.5B",
            },
        ],
        "Distil-Whisper (Faster Whisper)": [
            {
                "id": "distil-whisper/distil-small.en",
                "description": "6x faster than Whisper, English only (166M params)",
                "size": "166M",
            },
            {
                "id": "distil-whisper/distil-medium.en",
                "description": "6x faster than Whisper, English only (394M params)",
                "size": "394M",
            },
            {
                "id": "distil-whisper/distil-large-v2",
                "description": "6x faster than large-v2, multilingual (756M params)",
                "size": "756M",
            },
            {
                "id": "distil-whisper/distil-large-v3",
                "description": "Latest distilled large model, 6x faster (756M params)",
                "size": "756M",
            },
        ],
        "Wav2Vec2 (Meta)": [
            {
                "id": "facebook/wav2vec2-base-960h",
                "description": "Base Wav2Vec2 trained on LibriSpeech (95M params)",
                "size": "95M",
            },
            {
                "id": "facebook/wav2vec2-large-960h",
                "description": "Large Wav2Vec2 trained on LibriSpeech (317M params)",
                "size": "317M",
            },
            {
                "id": "facebook/wav2vec2-large-960h-lv60-self",
                "description": "Large Wav2Vec2 with self-training (317M params)",
                "size": "317M",
            },
        ],
        "Other Popular Models": [
            {
                "id": "facebook/s2t-small-librispeech-asr",
                "description": "Speech2Text small model",
                "size": "77M",
            },
            {
                "id": "nvidia/stt_en_conformer_ctc_large",
                "description": "NVIDIA Conformer-CTC model",
                "size": "120M",
            },
        ],
    }

    return jsonify({"success": True, "models": popular_models})


@app.route("/api/models/info", methods=["GET"])
def get_model_info():
    """Get detailed information about a specific model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        model_id = request.args.get("model_id")
        if not model_id:
            return jsonify({"success": False, "error": "model_id required"}), 400

        info = model_info(model_id)

        # Calculate model size from file siblings
        size_bytes = 0
        size_str = "Unknown"
        try:
            if hasattr(info, "siblings") and info.siblings:
                for sibling in info.siblings:
                    if hasattr(sibling, "rfilename") and hasattr(sibling, "size"):
                        # Sum up sizes of model files
                        if any(
                            ext in sibling.rfilename
                            for ext in [".safetensors", ".bin", ".pt", ".onnx"]
                        ):
                            size_bytes += sibling.size

                # Convert to readable format
                if size_bytes > 0:
                    size_mb = size_bytes / (1024 * 1024)
                    if size_mb > 1024:
                        size_str = f"{size_mb / 1024:.2f} GB"
                    else:
                        size_str = f"{size_mb:.0f} MB"
        except OSError:
            pass

        model_details = {
            "id": model_id,
            "author": info.author if hasattr(info, "author") else "unknown",
            "downloads": info.downloads if hasattr(info, "downloads") else 0,
            "likes": info.likes if hasattr(info, "likes") else 0,
            "tags": info.tags if hasattr(info, "tags") else [],
            "pipeline_tag": info.pipeline_tag
            if hasattr(info, "pipeline_tag")
            else "unknown",
            "library": info.library_name
            if hasattr(info, "library_name")
            else "unknown",
            "languages": [],
            "size": size_str,
            "size_bytes": size_bytes,
        }

        # Extract language tags
        if hasattr(info, "tags"):
            for tag in info.tags:
                if tag.startswith("language:"):
                    model_details["languages"].append(tag.replace("language:", ""))

        return jsonify({"success": True, "model": model_details})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Download progress tracking
DOWNLOAD_PROGRESS_FILE = os.path.join(APP_DIR, "download_progress.json")


# Download tracking lives in stt/downloads.py (importable, unit-tested); the
# module owns the shared state and these names are re-imported so call sites
# stay unchanged.
from stt import downloads as _downloads
from stt.downloads import (  # noqa: F401
    active_downloads,
    active_downloads_lock,
    cancelled_downloads,
    cleanup_stale_downloads,
    download_url_to_file,
    finish_download,
    load_download_progress,
    monitor_download_progress,
    save_download_progress,
    start_download_monitor,
    try_register_download,
)

_downloads.configure(DOWNLOAD_PROGRESS_FILE)


# Whisper model sizes in bytes (for progress tracking)
WHISPER_MODEL_SIZES = {
    "tiny.en": 75572083,
    "tiny": 75572083,
    "base.en": 145262807,
    "base": 145262807,
    "small.en": 483617219,
    "small": 483617219,
    "medium.en": 1528008539,
    "medium": 1528008539,
    "large-v1": 3087371615,
    "large-v2": 3087371615,
    "large-v3": 3087371615,
    "large": 3087371615,
    "large-v3-turbo": 1550580107,
    "turbo": 1550580107,
}

# Faster-Whisper models (CTranslate2 format, 4-10x faster)
FASTER_WHISPER_MODELS = {
    "tiny": {"repo": "Systran/faster-whisper-tiny", "size": "~75MB", "params": "39M", "lang": "Multilingual"},
    "tiny.en": {"repo": "Systran/faster-whisper-tiny.en", "size": "~75MB", "params": "39M", "lang": "English-only"},
    "base": {"repo": "Systran/faster-whisper-base", "size": "~145MB", "params": "74M", "lang": "Multilingual"},
    "base.en": {"repo": "Systran/faster-whisper-base.en", "size": "~145MB", "params": "74M", "lang": "English-only"},
    "small": {"repo": "Systran/faster-whisper-small", "size": "~465MB", "params": "244M", "lang": "Multilingual"},
    "small.en": {"repo": "Systran/faster-whisper-small.en", "size": "~465MB", "params": "244M", "lang": "English-only"},
    "medium": {"repo": "Systran/faster-whisper-medium", "size": "~1.5GB", "params": "769M", "lang": "Multilingual"},
    "medium.en": {"repo": "Systran/faster-whisper-medium.en", "size": "~1.5GB", "params": "769M", "lang": "English-only"},
    "large-v1": {"repo": "Systran/faster-whisper-large-v1", "size": "~3GB", "params": "1550M", "lang": "Multilingual"},
    "large-v2": {"repo": "Systran/faster-whisper-large-v2", "size": "~3GB", "params": "1550M", "lang": "Multilingual"},
    "large-v3": {"repo": "Systran/faster-whisper-large-v3", "size": "~3GB", "params": "1550M", "lang": "Multilingual"},
    "large-v3-turbo": {"repo": "Systran/faster-whisper-large-v3-turbo", "size": "~1.6GB", "params": "809M", "lang": "Multilingual"},
    "distil-large-v3": {"repo": "Systran/faster-distil-whisper-large-v3", "size": "~1.5GB", "params": "756M", "lang": "Multilingual"},
}

# Restore download state from the previous run.
_downloads.load_state()

# A "downloading" entry loaded from disk means the server died mid-download:
# the download thread is gone, so mark it failed instead of showing it for 24h
for _key, _info in active_downloads.items():
    if _info.get("status") == "downloading":
        _info["status"] = "failed"
        _info["error"] = "Interrupted by server restart"
        _info["last_update"] = time.time()
        print(f"[DOWNLOAD] Marked interrupted download as failed: {_key}")
save_download_progress()

# Clean up stale downloads on startup
cleanup_stale_downloads()


def download_hf_repo_files(repo_id, local_dir, download_key, log=print, include=None):
    """Download a HuggingFace repo's files with resume + cancellation.

    ``include`` restricts the download to matching filenames (fnmatch patterns, or
    exact names). Needed for GGUF repos, which publish a dozen quantisations of the
    same model in one repo — downloading them all would pull 40+ GB to obtain the one
    file the user picked.

    Returns "ok" or "cancelled"; raises on failure after retries."""
    import fnmatch

    from huggingface_hub import list_repo_files, hf_hub_url

    os.makedirs(local_dir, exist_ok=True)
    local_root = os.path.abspath(local_dir)
    files = list_repo_files(repo_id=repo_id)
    if include:
        patterns = [include] if isinstance(include, str) else list(include)
        selected = [f for f in files
                    if any(f == p or fnmatch.fnmatch(f, p) for p in patterns)]
        if not selected:
            raise ValueError(f"No file in {repo_id} matches {patterns}")
        log(f"[DOWNLOAD] {len(selected)} of {len(files)} files match {patterns}")
        files = selected
    log(f"[DOWNLOAD] Found {len(files)} files to download for {repo_id}")

    for idx, filename in enumerate(files):
        if download_key in cancelled_downloads:
            return "cancelled"

        dest_path = os.path.abspath(os.path.join(local_root, filename))
        if not dest_path.startswith(local_root + os.sep):
            raise ValueError(f"Unsafe filename in repo {repo_id}: {filename}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Skip if already downloaded and has content
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            log(f"[DOWNLOAD] Already exists: {filename}")
            continue

        log(f"[DOWNLOAD] Downloading file {idx + 1}/{len(files)}: {filename}")

        # File-count progress only when no byte total is known (a directory
        # size monitor provides byte-accurate progress otherwise)
        with active_downloads_lock:
            entry = active_downloads.get(download_key)
            if entry and entry.get("status") == "downloading" and not entry.get("total"):
                entry["percentage"] = int((idx / len(files)) * 100)
                entry["last_update"] = time.time()
        save_download_progress()

        url = hf_hub_url(repo_id=repo_id, filename=filename)
        outcome = download_url_to_file(
            url, dest_path,
            cancel_check=lambda: download_key in cancelled_downloads,
            log=log,
        )
        if outcome == "cancelled":
            return "cancelled"
        log(f"[OK] Downloaded: {filename}")

    return "ok"

# Start periodic cleanup thread
def periodic_cleanup():
    """Run cleanup every hour"""
    import time
    while True:
        time.sleep(3600)  # Every hour
        cleanup_stale_downloads()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()


def _models_not_writable_response():
    """Return a JSON error response when MODELS_DIR can't be written, else None.

    Model downloads run in a background thread; without this up-front check the
    UI shows a brief "downloading" and then the thread dies with an opaque
    errno 13, so it looks like nothing happened. Checking here lets every
    download route fail fast with a clear, actionable message instead."""
    if dir_is_writable(MODELS_DIR):
        return None
    return jsonify({
        "success": False,
        "error": (f"Cannot write to the models folder ({MODELS_DIR}). It is not "
                  f"writable by the user running the server — run the server as "
                  f"that folder's owner, or fix its ownership/permissions."),
    }), 503


@app.route("/api/models/download", methods=["POST"])
def download_model():
    """Download/cache a model (Hugging Face or Whisper)
    Example: POST /api/models/download {"model_type": "whisper", "model_name": "small"}"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    not_writable = _models_not_writable_response()
    if not_writable:
        return not_writable

    try:
        data = request.get_json()
        model_type = data.get("model_type")

        if model_type == "whisper":
            # Handle Whisper model download
            model_name = data.get("model_name")
            if not model_name:
                return jsonify(
                    {
                        "success": False,
                        "error": "model_name required for Whisper models",
                    }
                ), 400

            print(f"[DOWNLOAD] Downloading Whisper model: {model_name}")

            _lazy_import_ml_libraries()

            # Create custom download directory in ./models
            models_dir = MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)

            whisper_dir = os.path.join(models_dir, f"whisper-{model_name}")
            os.makedirs(whisper_dir, exist_ok=True)

            # Set environment variable to use custom download directory
            os.environ["WHISPER_CACHE"] = whisper_dir


            model_key = f"whisper-{model_name}"
            if not try_register_download(model_key, total=WHISPER_MODEL_SIZES.get(model_name)):
                return jsonify({"success": False, "error": "Download already in progress"}), 409

            try:
                start_download_monitor(
                    model_key, os.path.join(whisper_dir, f"{model_name}.pt")
                )

                # Download the model file without loading it into GPU memory
                # Using custom download function that computes SHA256 during download
                # This avoids the blocking post-download verification that reads the whole file
                from whisper import _MODELS
                import urllib.request
                import hashlib
                from tqdm import tqdm

                if model_name not in _MODELS:
                    raise ValueError(f"Unknown Whisper model: {model_name}. Available: {list(_MODELS.keys())}")

                url = _MODELS[model_name]
                expected_sha256 = url.split("/")[-2]
                download_target = os.path.join(whisper_dir, os.path.basename(url))

                # Check if already downloaded with correct checksum
                if os.path.isfile(download_target):
                    print(f"[INFO] File exists, verifying checksum: {download_target}")
                    sha256_hash = hashlib.sha256()
                    with open(download_target, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha256_hash.update(chunk)
                    if sha256_hash.hexdigest() == expected_sha256:
                        print("[OK] Existing file checksum matches")
                        model_path = download_target
                    else:
                        print("[WARN] Existing file checksum mismatch, re-downloading")
                        os.remove(download_target)
                        model_path = None
                else:
                    model_path = None

                if model_path is None:
                    # Download with streaming SHA256 computation
                    print(f"[INFO] Downloading Whisper model to {download_target}")
                    sha256_hash = hashlib.sha256()

                    download_cancelled = False
                    with urllib.request.urlopen(url, timeout=120) as source, open(download_target, "wb") as output:
                        total_size = int(source.info().get("Content-Length", 0))
                        with tqdm(total=total_size, ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
                            while True:
                                if model_key in cancelled_downloads:
                                    download_cancelled = True
                                    break
                                buffer = source.read(8192)
                                if not buffer:
                                    break
                                output.write(buffer)
                                sha256_hash.update(buffer)  # Compute SHA256 during download
                                pbar.update(len(buffer))

                    if download_cancelled:
                        print(f"[CANCELLED] Whisper download cancelled: {model_name}")
                        try:
                            os.remove(download_target)
                        except OSError:
                            pass
                        finish_download(model_key, cancelled=True)
                        return jsonify({"success": False, "message": "Download cancelled"})

                    # Verify checksum computed during download
                    computed_sha256 = sha256_hash.hexdigest()
                    if computed_sha256 != expected_sha256:
                        os.remove(download_target)
                        raise RuntimeError(
                            f"Model download failed: SHA256 mismatch. Expected {expected_sha256}, got {computed_sha256}"
                        )

                    model_path = download_target
                    print("[OK] Download complete, checksum verified")

                message = f"Whisper {model_name} model downloaded to {model_path}"

                print(f"[OK] {message}")

                finish_download(model_key)
            except Exception as e:
                print(f"[ERROR] Whisper model download failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                finish_download(model_key, error=e)
                raise

            return jsonify(
                {
                    "success": True,
                    "message": f"downloaded to {whisper_dir}",
                    "model_name": model_name,
                    "path": whisper_dir,
                }
            )

        else:
            # Handle HuggingFace model download (original logic)
            model_id = data.get("model_id")
            local_dir = data.get("local_dir")

            if not model_id:
                return jsonify({"success": False, "error": "model_id required"}), 400

            print(f"[DOWNLOAD] Downloading HuggingFace model: {model_id}")

            _lazy_import_ml_libraries()

            # If no local_dir specified, download to ./models directory
            if not local_dir:
                models_dir = MODELS_DIR
                os.makedirs(models_dir, exist_ok=True)

                # Use model name as directory (replace / with --)
                model_dir_name = model_id.replace("/", "--")
                local_dir = os.path.join(models_dir, model_dir_name)


            if not try_register_download(model_id):
                return jsonify({"success": False, "error": "Download already in progress"}), 409

            # A GGUF repo publishes a dozen quantisations of the same model, so the
            # caller names the one file it wants; without this the download pulls
            # 40+ GB to obtain the ~2-5 GB the operator picked.
            _include = data.get("include") or data.get("gguf_file") or None
            if isinstance(_include, str):
                _include = [_include]

            try:
                # Per-file download with resume + cancellation
                # (huggingface_hub's snapshot_download hangs on large files)
                outcome = download_hf_repo_files(model_id, local_dir, model_id,
                                                 include=_include)
                if outcome == "cancelled":
                    print(f"[CANCELLED] Download cancelled for {model_id}")
                    finish_download(model_id, cancelled=True)
                    return jsonify({"success": False, "message": "Download cancelled"})

                path = local_dir
                message = f"Model {model_id} downloaded to: {path}"

                print(f"[OK] {message}")

                finish_download(model_id)
            except Exception as e:
                finish_download(model_id, error=e)
                raise

            return jsonify(
                {
                    "success": True,
                    "message": message,
                    "model_id": model_id,
                    "path": path,
                }
            )

    except Exception as e:
        print(f"[ERROR] Error downloading model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/download-status", methods=["GET"])
def download_status():
    """Get download status with real-time progress tracking"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        with active_downloads_lock:
            download_list = [
                {
                    "model": name,
                    "downloaded": info.get("downloaded", 0),
                    "total": info.get("total"),
                    "percentage": info.get("percentage"),
                    "status": info.get("status", "downloading"),
                    "error": info.get("error"),
                    "completion_time": info.get("completion_time"),
                }
                for name, info in active_downloads.items()
            ]

        # Separate by status
        downloading = [d for d in download_list if d["status"] == "downloading"]
        completed = [d for d in download_list if d["status"] == "completed"]
        failed = [d for d in download_list if d["status"] == "failed"]

        return jsonify(
            {
                "status": "active" if downloading else "idle",
                "active_downloads": download_list,
                "downloading": downloading,
                "completed": completed,
                "failed": failed,
                "message": f"{len(downloading)} active, {len(completed)} completed, {len(failed)} failed"
                if download_list
                else "No active downloads",
            }
        )
    except Exception as e:
        print(f"[ERROR] Error getting download status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/models/cancel-download", methods=["POST"])
def cancel_download():
    """Cancel an active download and clean up"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    data = request.get_json()
    model_id = data.get("model_id") if data else None

    if not model_id:
        return jsonify({"success": False, "error": "model_id required"}), 400

    try:
        with active_downloads_lock:
            entry = active_downloads.get(model_id)
            was_downloading = entry is not None and entry.get("status") == "downloading"
            if was_downloading:
                # Signal the download thread to stop
                cancelled_downloads.add(model_id)
            if entry is not None:
                del active_downloads[model_id]

        # Save outside the lock to avoid deadlock (save_download_progress acquires the same lock)
        save_download_progress()

        if not was_downloading:
            # Completed/failed entry (or already gone): this is a dismiss, not a
            # cancel — never delete files for a download that isn't in flight
            return jsonify({"success": True, "message": f"Dismissed {model_id}"})

        # Clean up partial download directories/files so model shows as not downloaded
        # model_id already includes prefix (e.g., "whisper-small.en" or "faster-whisper-base.en")
        # For HuggingFace models like "facebook/nllb-200-distilled-600M", slashes become double dashes
        dir_name = model_id.replace("/", "--")
        # Containment check: model_id is user input and must resolve to a
        # subdirectory strictly inside MODELS_DIR (rmtree happens below)
        model_path = safe_model_path(MODELS_DIR, dir_name)
        if model_path is None:
            return jsonify({"success": False, "error": "Invalid model id"}), 400
        if os.path.exists(model_path):
            try:
                shutil.rmtree(model_path)
                print(f"[INFO] Cleaned up partial download: {model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up {model_path}: {e}")

        # For whisper .pt files in cache, extract base name (e.g., "whisper-small.en" -> "small.en")
        if model_id.startswith("whisper-") and not model_id.startswith("whisper-faster"):
            whisper_cache = os.path.expanduser("~/.cache/whisper")
            base_name = model_id[8:]  # Remove "whisper-" prefix to get "small.en"
            pt_file = os.path.join(whisper_cache, f"{base_name}.pt")
            if os.path.exists(pt_file):
                try:
                    os.remove(pt_file)
                    print(f"[INFO] Cleaned up partial download: {pt_file}")
                except Exception as e:
                    print(f"[WARNING] Failed to clean up {pt_file}: {e}")

        return jsonify({"success": True, "message": f"Cancelled {model_id}"})
    except Exception as e:
        print(f"[ERROR] Error cancelling download: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/upload", methods=["POST"])
def upload_model():
    """Upload a local model to Hugging Face"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        token = data.get("token")
        model_path = data.get("model_path")
        repo_id = data.get("repo_id")
        commit_message = data.get("commit_message", "Upload model")
        private = data.get("private", False)

        if not all([token, model_path, repo_id]):
            return jsonify(
                {"success": False, "error": "token, model_path, and repo_id required"}
            ), 400

        _lazy_import_ml_libraries()
        from huggingface_hub import HfApi
        from pathlib import Path

        model_path = Path(model_path)
        if not model_path.exists():
            return jsonify(
                {"success": False, "error": f"Model path not found: {model_path}"}
            ), 404

        api = HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type="model",
                exist_ok=True,
            )
            print(f"[OK] Repository ready: {repo_id}")
        except Exception as e:
            print(f"[WARNING] Repository creation: {e}")

        # Upload files
        if model_path.is_file():
            # Single file upload
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
            )
        else:
            # Directory upload
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
            )

        print(f"[OK] Model uploaded successfully to: https://huggingface.co/{repo_id}")

        return jsonify(
            {
                "success": True,
                "message": f"Model uploaded successfully to: https://huggingface.co/{repo_id}",
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
            }
        )

    except Exception as e:
        print(f"[ERROR] Error uploading model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/local", methods=["GET"])
def get_local_models():
    """Get list of local models in directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        directory = request.args.get("directory", "./models")
        models_path = Path(directory)

        if not models_path.exists():
            return jsonify(
                {
                    "success": True,
                    "models": [],
                    "message": f"Directory not found: {directory}",
                }
            )

        models = []
        for item in models_path.iterdir():
            if item.is_dir():
                # Check for model files
                model_files = (
                    list(item.glob("*.bin"))
                    + list(item.glob("*.safetensors"))
                    + list(item.glob("*.pt"))
                )
                if model_files:
                    size_mb = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
                    models.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "files": [f.name for f in model_files],
                            "size_mb": size_mb,
                        }
                    )

        return jsonify({"success": True, "models": models, "count": len(models)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@app.route("/api/models/cached", methods=["GET"])
def get_cached_models():
    """Get list of cached/downloaded models"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        import os

        # Get Hugging Face cache directory
        cache_dir = os.getenv(
            "HF_HOME",
            os.getenv("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface")),
        )
        models_dir = os.path.join(cache_dir, "hub")

        cached_models = []

        if os.path.exists(models_dir):
            # List all model directories
            for item in os.listdir(models_dir):
                if item.startswith("models--"):
                    # Extract model name from directory
                    model_name = item.replace("models--", "").replace("--", "/")
                    model_path = os.path.join(models_dir, item)

                    # Get directory size
                    total_size = 0
                    for dirpath, _dirnames, filenames in os.walk(model_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if os.path.exists(fp):
                                total_size += os.path.getsize(fp)

                    # Convert to readable format
                    size_mb = total_size / (1024 * 1024)
                    if size_mb > 1024:
                        size_str = f"{size_mb / 1024:.2f} GB"
                    else:
                        size_str = f"{size_mb:.2f} MB"

                    cached_models.append(
                        {
                            "id": model_name,
                            "size": size_str,
                            "size_bytes": total_size,
                            "path": model_path,
                        }
                    )

        # Also check for Whisper models
        whisper_cache = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(whisper_cache):
            for item in os.listdir(whisper_cache):
                if item.endswith(".pt"):
                    model_path = os.path.join(whisper_cache, item)
                    size_bytes = os.path.getsize(model_path)
                    size_mb = size_bytes / (1024 * 1024)
                    if size_mb > 1024:
                        size_str = f"{size_mb / 1024:.2f} GB"
                    else:
                        size_str = f"{size_mb:.2f} MB"

                    cached_models.append(
                        {
                            "id": f"whisper/{item.replace('.pt', '')}",
                            "size": size_str,
                            "size_bytes": size_bytes,
                            "path": model_path,
                        }
                    )

        return jsonify(
            {
                "success": True,
                "models": cached_models,
                "count": len(cached_models),
                "cache_dir": cache_dir,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


# Model Manager Endpoints
@app.route("/api/models/manager", methods=["GET"])
def get_model_manager():
    """Get centralized model configuration for both live and file transcription"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        global config

        # Get live transcription model config
        live_model_config = config.get("model", {})

        # Get file transcription model config
        ft_config = config.get("file_transcription", {})
        ft_model_config = ft_config.get("model", {})

        # If file transcription doesn't have model config, use live config as fallback
        if not ft_model_config:
            ft_model_config = live_model_config.copy()

        return jsonify(
            {
                "success": True,
                "live_model": {
                    "type": live_model_config.get("type", "whisper"),
                    "backend": live_model_config.get("backend", ""),
                    "whisper": {
                        "model": live_model_config.get("whisper", {}).get(
                            "model", "tiny"
                        ),
                    },
                    "huggingface": {
                        "model_id": live_model_config.get("huggingface", {}).get(
                            "model_id", "openai/whisper-base"
                        ),
                        "use_flash_attention": live_model_config.get(
                            "huggingface", {}
                        ).get("use_flash_attention", False),
                    },
                    "custom": {
                        "model_path": live_model_config.get("custom", {}).get(
                            "model_path", ""
                        ),
                        "model_type": live_model_config.get("custom", {}).get(
                            "model_type", "whisper"
                        ),
                    },
                },
                "file_transcription_model": {
                    "type": ft_model_config.get("type", "whisper"),
                    "backend": ft_model_config.get("backend", ""),
                    "whisper": {
                        "model": ft_model_config.get("whisper", {}).get(
                            "model", "base"
                        ),
                    },
                    "huggingface": {
                        "model_id": ft_model_config.get("huggingface", {}).get(
                            "model_id", "openai/whisper-base"
                        ),
                        "use_flash_attention": ft_model_config.get(
                            "huggingface", {}
                        ).get("use_flash_attention", False),
                    },
                    "custom": {
                        "model_path": ft_model_config.get("custom", {}).get(
                            "model_path", ""
                        ),
                        "model_type": ft_model_config.get("custom", {}).get(
                            "model_type", "whisper"
                        ),
                    },
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/manager", methods=["POST"])
def update_model_manager():
    """Update centralized model configuration"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        global config

        # Update live transcription model config
        if "live_model" in data:
            live_model = data["live_model"]
            config["model"] = {"type": live_model.get("type", "whisper")}

            if live_model.get("type") == "whisper":
                config["model"]["whisper"] = {
                    "model": live_model.get("whisper", {}).get("model", "tiny"),
                }
                # Always save backend setting (faster-whisper vs standard whisper)
                config["model"]["backend"] = live_model.get("backend")
            elif live_model.get("type") == "huggingface":
                config["model"]["huggingface"] = {
                    "model_id": live_model.get("huggingface", {}).get(
                        "model_id", "openai/whisper-base"
                    ),
                    "use_flash_attention": live_model.get("huggingface", {}).get(
                        "use_flash_attention", False
                    ),
                }
            elif live_model.get("type") == "custom":
                config["model"]["custom"] = {
                    "model_path": live_model.get("custom", {}).get("model_path", ""),
                    "model_type": live_model.get("custom", {}).get(
                        "model_type", "whisper"
                    ),
                }

        # Update file transcription model config
        if "file_transcription_model" in data:
            ft_model = data["file_transcription_model"]

            if "file_transcription" not in config:
                config["file_transcription"] = {}

            config["file_transcription"]["model"] = {
                "type": ft_model.get("type", "whisper")
            }

            if ft_model.get("type") == "whisper":
                config["file_transcription"]["model"]["whisper"] = {
                    "model": ft_model.get("whisper", {}).get("model", "base"),
                }
                # Always save backend setting (faster-whisper vs standard whisper)
                config["file_transcription"]["model"]["backend"] = ft_model.get("backend")
            elif ft_model.get("type") == "huggingface":
                config["file_transcription"]["model"]["huggingface"] = {
                    "model_id": ft_model.get("huggingface", {}).get(
                        "model_id", "openai/whisper-base"
                    ),
                    "use_flash_attention": ft_model.get("huggingface", {}).get(
                        "use_flash_attention", False
                    ),
                }
            elif ft_model.get("type") == "custom":
                config["file_transcription"]["model"]["custom"] = {
                    "model_path": ft_model.get("custom", {}).get("model_path", ""),
                    "model_type": ft_model.get("custom", {}).get(
                        "model_type", "whisper"
                    ),
                }

        # Save configuration
        save_config(config)

        return jsonify(
            {"success": True, "message": "Model configurations updated successfully"}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/sync", methods=["POST"])
def sync_model_configs():
    """Sync model configuration between live and file transcription"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        direction = data.get("direction", "live_to_file")  # or 'file_to_live'

        global config

        if direction == "live_to_file":
            # Copy live model config to file transcription
            live_model_config = config.get("model", {}).copy()

            if "file_transcription" not in config:
                config["file_transcription"] = {}

            config["file_transcription"]["model"] = live_model_config

            message = "Live model configuration copied to file transcription"

        elif direction == "file_to_live":
            # Copy file transcription model config to live
            ft_model_config = (
                config.get("file_transcription", {}).get("model", {}).copy()
            )

            if ft_model_config:
                config["model"] = ft_model_config
                message = "File transcription model configuration copied to live"
            else:
                return jsonify(
                    {
                        "success": False,
                        "error": "No file transcription model configuration found",
                    }
                )

        # Save configuration
        save_config(config)

        return jsonify({"success": True, "message": message})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Cache for discovered Whisper models (to avoid frequent checks)
_whisper_models_cache = None
_whisper_models_cache_time = None
WHISPER_CACHE_DURATION = 86400  # Cache for 24 hours (1 day)
WHISPER_MODELS_FILE = _seed_from_bundle("whisper_models.json")  # Persistent storage file


def load_whisper_models_from_file():
    """Load discovered Whisper models from local file"""
    try:
        if os.path.exists(WHISPER_MODELS_FILE):
            with open(WHISPER_MODELS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("models", {}), data.get("timestamp", 0)
    except Exception as e:
        print(f"[WARNING] Could not load Whisper models from file: {e}")
    return None, None


def save_whisper_models_to_file(models):
    """Save discovered Whisper models to local file"""
    try:
        import datetime

        data = {
            "models": models,
            "timestamp": time.time(),
            "last_updated": datetime.datetime.now().isoformat(),
        }
        with open(WHISPER_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Saved {len(models)} Whisper models to {WHISPER_MODELS_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save Whisper models to file: {e}")


def get_whisper_models_list():
    """Get list of available Whisper models with fallback to defaults"""
    global _whisper_models_cache, _whisper_models_cache_time

    # Check if memory cache is still valid
    if _whisper_models_cache and _whisper_models_cache_time:
        if (time.time() - _whisper_models_cache_time) < WHISPER_CACHE_DURATION:
            return _whisper_models_cache

    # Try to load from file
    file_models, file_timestamp = load_whisper_models_from_file()
    if file_models and file_timestamp:
        # Use file cache (even if expired, better than defaults)
        _whisper_models_cache = file_models
        _whisper_models_cache_time = file_timestamp
        print(
            f"[OK] Loaded {len(file_models)} Whisper models from {WHISPER_MODELS_FILE}"
        )
        return file_models

    # Default models (fallback)
    default_models = {
        "tiny": {
            "params": "39M",
            "size": "~75MB",
            "desc": "Fastest",
            "lang": "Multilingual",
        },
        "tiny.en": {
            "params": "39M",
            "size": "~75MB",
            "desc": "Fastest",
            "lang": "English-only",
        },
        "base": {
            "params": "74M",
            "size": "~142MB",
            "desc": "Balanced",
            "lang": "Multilingual",
        },
        "base.en": {
            "params": "74M",
            "size": "~142MB",
            "desc": "Balanced",
            "lang": "English-only",
        },
        "small": {
            "params": "244M",
            "size": "~466MB",
            "desc": "Good accuracy",
            "lang": "Multilingual",
        },
        "small.en": {
            "params": "244M",
            "size": "~466MB",
            "desc": "Good accuracy",
            "lang": "English-only",
        },
        "medium": {
            "params": "769M",
            "size": "~1.5GB",
            "desc": "Better accuracy",
            "lang": "Multilingual",
        },
        "medium.en": {
            "params": "769M",
            "size": "~1.5GB",
            "desc": "Better accuracy",
            "lang": "English-only",
        },
        "large": {
            "params": "1550M",
            "size": "~3GB",
            "desc": "Best accuracy",
            "lang": "Multilingual",
        },
        "large-v2": {
            "params": "1550M",
            "size": "~3GB",
            "desc": "Best accuracy v2",
            "lang": "Multilingual",
        },
        "large-v3": {
            "params": "1550M",
            "size": "~3GB",
            "desc": "Best accuracy v3",
            "lang": "Multilingual",
        },
    }

    # Update memory cache
    _whisper_models_cache = default_models
    _whisper_models_cache_time = time.time()

    # Save to file for persistence
    save_whisper_models_to_file(default_models)

    return default_models


# Cache for discovered Faster-Whisper models
_faster_whisper_models_cache = None
_faster_whisper_models_cache_time = None
FASTER_WHISPER_MODELS_FILE = _seed_from_bundle("faster_whisper_models.json")


def load_faster_whisper_models_from_file():
    """Load discovered Faster-Whisper models from local file"""
    try:
        if os.path.exists(FASTER_WHISPER_MODELS_FILE):
            with open(FASTER_WHISPER_MODELS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("models", {}), data.get("timestamp", 0)
    except Exception as e:
        print(f"[WARNING] Could not load Faster-Whisper models from file: {e}")
    return None, None


def save_faster_whisper_models_to_file(models):
    """Save discovered Faster-Whisper models to local file"""
    try:
        import datetime

        data = {
            "models": models,
            "timestamp": time.time(),
            "last_updated": datetime.datetime.now().isoformat(),
        }
        with open(FASTER_WHISPER_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Saved {len(models)} Faster-Whisper models to {FASTER_WHISPER_MODELS_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save Faster-Whisper models to file: {e}")


def get_faster_whisper_models_list():
    """Get list of available Faster-Whisper models with fallback to defaults"""
    global _faster_whisper_models_cache, _faster_whisper_models_cache_time

    # Check if memory cache is still valid
    if _faster_whisper_models_cache and _faster_whisper_models_cache_time:
        if (time.time() - _faster_whisper_models_cache_time) < WHISPER_CACHE_DURATION:
            return _faster_whisper_models_cache

    # Try to load from file
    file_models, file_timestamp = load_faster_whisper_models_from_file()
    if file_models and file_timestamp:
        _faster_whisper_models_cache = file_models
        _faster_whisper_models_cache_time = file_timestamp
        print(f"[OK] Loaded {len(file_models)} Faster-Whisper models from {FASTER_WHISPER_MODELS_FILE}")
        return file_models

    # Use hardcoded defaults as fallback
    _faster_whisper_models_cache = FASTER_WHISPER_MODELS
    _faster_whisper_models_cache_time = time.time()

    # Save to file for persistence
    save_faster_whisper_models_to_file(FASTER_WHISPER_MODELS)

    return FASTER_WHISPER_MODELS


@app.route("/api/models/refresh-faster-whisper", methods=["POST"])
def refresh_faster_whisper_models():
    """Discover available Faster-Whisper models from HuggingFace"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global _faster_whisper_models_cache, _faster_whisper_models_cache_time

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Search for Systran's faster-whisper models
        models = list(api.list_models(author="Systran", search="faster-whisper"))

        # Known model specifications for size/params estimation
        model_specs = {
            "tiny": {"size": "~75MB", "params": "39M"},
            "base": {"size": "~145MB", "params": "74M"},
            "small": {"size": "~465MB", "params": "244M"},
            "medium": {"size": "~1.5GB", "params": "769M"},
            "large": {"size": "~3GB", "params": "1550M"},
            "distil": {"size": "~1.5GB", "params": "756M"},
            "turbo": {"size": "~1.6GB", "params": "809M"},
        }

        discovered_models = {}
        old_models = _faster_whisper_models_cache or FASTER_WHISPER_MODELS

        for model in models:
            repo_id = model.id  # e.g., "Systran/faster-whisper-large-v3"
            # Accept both faster-whisper and faster-distil-whisper repos
            if not (repo_id.startswith("Systran/faster-whisper") or repo_id.startswith("Systran/faster-distil-whisper")):
                continue

            # Extract model name from repo_id
            if repo_id.startswith("Systran/faster-distil-whisper-"):
                model_name = "distil-" + repo_id.replace("Systran/faster-distil-whisper-", "")
            else:
                model_name = repo_id.replace("Systran/faster-whisper-", "")

            # Determine specs based on model name
            size = "~3GB"
            params = "1550M"
            lang = "Multilingual"

            for key, specs in model_specs.items():
                if key in model_name.lower():
                    size = specs["size"]
                    params = specs["params"]
                    break

            if ".en" in model_name:
                lang = "English-only"

            discovered_models[model_name] = {
                "repo": repo_id,
                "size": size,
                "params": params,
                "lang": lang,
            }

        # Find new and removed models
        old_names = set(old_models.keys())
        new_names = set(discovered_models.keys())
        added = new_names - old_names
        removed = old_names - new_names

        # Update cache
        _faster_whisper_models_cache = discovered_models
        _faster_whisper_models_cache_time = time.time()

        # Save to file
        save_faster_whisper_models_to_file(discovered_models)

        return jsonify({
            "success": True,
            "message": f"Found {len(discovered_models)} Faster-Whisper models",
            "count": len(discovered_models),
            "new_models": list(added),
            "removed_models": list(removed),
        })

    except Exception as e:
        print(f"[ERROR] Error refreshing Faster-Whisper models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/refresh-whisper", methods=["POST"])
def refresh_whisper_models():
    """Discover available Whisper models from the whisper package"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global _whisper_models_cache, _whisper_models_cache_time

    try:
        # Check whisper package for available models
        try:
            import whisper

            available_models = whisper.available_models()

            # Known model specifications
            model_specs = {
                "tiny": {
                    "params": "39M",
                    "size": "~75MB",
                    "desc": "Fastest",
                    "lang": "Multilingual",
                },
                "tiny.en": {
                    "params": "39M",
                    "size": "~75MB",
                    "desc": "Fastest",
                    "lang": "English-only",
                },
                "base": {
                    "params": "74M",
                    "size": "~142MB",
                    "desc": "Balanced",
                    "lang": "Multilingual",
                },
                "base.en": {
                    "params": "74M",
                    "size": "~142MB",
                    "desc": "Balanced",
                    "lang": "English-only",
                },
                "small": {
                    "params": "244M",
                    "size": "~466MB",
                    "desc": "Good accuracy",
                    "lang": "Multilingual",
                },
                "small.en": {
                    "params": "244M",
                    "size": "~466MB",
                    "desc": "Good accuracy",
                    "lang": "English-only",
                },
                "medium": {
                    "params": "769M",
                    "size": "~1.5GB",
                    "desc": "Better accuracy",
                    "lang": "Multilingual",
                },
                "medium.en": {
                    "params": "769M",
                    "size": "~1.5GB",
                    "desc": "Better accuracy",
                    "lang": "English-only",
                },
                "large": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy",
                    "lang": "Multilingual",
                },
                "large-v1": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy v1",
                    "lang": "Multilingual",
                },
                "large-v2": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy v2",
                    "lang": "Multilingual",
                },
                "large-v3": {
                    "params": "1550M",
                    "size": "~3GB",
                    "desc": "Best accuracy v3",
                    "lang": "Multilingual",
                },
                "large-v3-turbo": {
                    "params": "809M",
                    "size": "~1.6GB",
                    "desc": "Fast large model",
                    "lang": "Multilingual",
                },
                "turbo": {
                    "params": "809M",
                    "size": "~1.6GB",
                    "desc": "Fastest large model",
                    "lang": "Multilingual",
                },
            }

            # Get previous models to detect removals
            old_models = (
                set(_whisper_models_cache.keys()) if _whisper_models_cache else set()
            )

            # Build models dict from discovered models
            discovered_models = {}
            new_models_found = []

            for model_name in available_models:
                # Skip if already added
                if model_name in discovered_models:
                    continue

                # Use known specs or create generic entry for new models
                if model_name in model_specs:
                    discovered_models[model_name] = model_specs[model_name]
                else:
                    # New model found - determine if it's English-only based on .en suffix
                    is_english_only = model_name.endswith(".en")
                    discovered_models[model_name] = {
                        "params": "Unknown",
                        "size": "Unknown",
                        "desc": "New model",
                        "lang": "English-only" if is_english_only else "Multilingual",
                    }
                    # Only count as new if not in old cache
                    if model_name not in old_models:
                        new_models_found.append(model_name)

            # Detect removed models
            current_models = set(discovered_models.keys())
            removed_models = list(old_models - current_models)

            # Update memory cache
            _whisper_models_cache = discovered_models
            _whisper_models_cache_time = time.time()

            # Save to file for persistence across restarts
            save_whisper_models_to_file(discovered_models)

            print(
                f"[OK] Whisper models refreshed: {len(discovered_models)} total, {len(new_models_found)} new, {len(removed_models)} removed"
            )

            return jsonify(
                {
                    "success": True,
                    "message": f"Discovered {len(discovered_models)} Whisper models",
                    "total_models": len(discovered_models),
                    "models": list(discovered_models.keys()),
                    "new_models": new_models_found,
                    "removed_models": removed_models,
                }
            )

        except ImportError:
            return jsonify(
                {"success": False, "error": "Whisper package not installed"}
            ), 500

    except Exception as e:
        print(f"[ERROR] Error refreshing Whisper models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/remove-whisper", methods=["POST"])
def remove_whisper_model():
    """Remove a Whisper model from both old cache and new ./models directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400

        files_removed = []
        dirs_removed = []

        # Check old Whisper cache directory
        whisper_cache_old = os.path.expanduser("~/.cache/whisper")
        if os.path.exists(whisper_cache_old):
            for filename in os.listdir(whisper_cache_old):
                if filename.endswith(".pt"):
                    # Check if this file matches the model name
                    base_name = filename.replace(".pt", "").replace(".en", "")
                    if base_name == model_name:
                        file_path = os.path.join(whisper_cache_old, filename)
                        try:
                            os.remove(file_path)
                            files_removed.append(filename)
                            print(
                                f"[OK] Removed Whisper model file from cache: {filename}"
                            )
                        except Exception as e:
                            print(f"[ERROR] Failed to remove {filename}: {e}")

        # Check new ./models/whisper-* directory
        whisper_model_dir = safe_model_path(MODELS_DIR, f"whisper-{model_name}")
        if whisper_model_dir is None:
            return jsonify({"success": False, "error": "Invalid model name"}), 400

        if os.path.exists(whisper_model_dir):
            try:
                import shutil

                shutil.rmtree(whisper_model_dir)
                dirs_removed.append(f"whisper-{model_name}")
                print(f"[OK] Removed Whisper model directory: {whisper_model_dir}")
            except Exception as e:
                print(f"[ERROR] Failed to remove directory {whisper_model_dir}: {e}")

        if files_removed or dirs_removed:
            message_parts = []
            if files_removed:
                message_parts.append(f"{len(files_removed)} file(s) from cache")
            if dirs_removed:
                message_parts.append(f"{len(dirs_removed)} directory from models")

            return jsonify(
                {
                    "success": True,
                    "message": f'Successfully removed Whisper model "{model_name}" ({", ".join(message_parts)})',
                    "files_removed": files_removed,
                    "dirs_removed": dirs_removed,
                }
            )
        else:
            return jsonify(
                {
                    "success": False,
                    "error": f'No files or directories found for model "{model_name}"',
                }
            ), 404

    except Exception as e:
        print(f"[ERROR] Error removing Whisper model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/faster-whisper/list", methods=["GET"])
def list_faster_whisper_models():
    """List available faster-whisper models with download status"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        models_dir = MODELS_DIR
        models_list = []

        # Use dynamic model list instead of hardcoded
        available_models = get_faster_whisper_models_list()

        for model_name, details in available_models.items():
            model_path = os.path.join(models_dir, f"faster-whisper-{model_name}")
            # Check directory exists AND contains model weight files
            downloaded = dir_has_weights(model_path)

            models_list.append({
                "name": model_name,
                "repo": details["repo"],
                "size": details["size"],
                "params": details["params"],
                "lang": details["lang"],
                "downloaded": downloaded,
                "path": model_path if downloaded else None,
            })

        # Reverse order to match Whisper models (smallest first)
        models_list.reverse()

        return jsonify({
            "success": True,
            "models": models_list,
            "count": len(models_list),
        })

    except Exception as e:
        print(f"[ERROR] Error listing faster-whisper models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/faster-whisper/download", methods=["POST"])
def download_faster_whisper_model():
    """Download a faster-whisper model from HuggingFace"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    not_writable = _models_not_writable_response()
    if not_writable:
        return not_writable

    try:
        data = request.get_json()
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "model_name required"}), 400

        # Use dynamic model list instead of hardcoded
        available_models = get_faster_whisper_models_list()

        if model_name not in available_models:
            return jsonify({"success": False, "error": f"Unknown model: {model_name}. Try refreshing the model list."}), 400

        model_info = available_models[model_name]
        repo_id = model_info["repo"]

        print(f"[DOWNLOAD] Downloading faster-whisper model: {model_name} from {repo_id}")

        models_dir = MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        local_dir = os.path.join(models_dir, f"faster-whisper-{model_name}")

        # Best-effort total size so the UI can show a real percentage
        total_size = None
        try:
            from huggingface_hub import HfApi
            repo_info = HfApi().model_info(repo_id, files_metadata=True)
            total_size = sum(f.size or 0 for f in repo_info.siblings) or None
        except Exception as e:
            print(f"[WARNING] Could not get size of {repo_id}: {e}")

        download_key = f"faster-whisper-{model_name}"
        if not try_register_download(download_key, total=total_size):
            return jsonify({"success": False, "error": "Download already in progress"}), 409

        try:
            start_download_monitor(download_key, local_dir, total=total_size)

            # Per-file download with resume + cancellation (snapshot_download
            # can't be interrupted once started)
            outcome = download_hf_repo_files(repo_id, local_dir, download_key)
            if outcome == "cancelled":
                print(f"[CANCELLED] Download cancelled for {download_key}")
                finish_download(download_key, cancelled=True)
                return jsonify({"success": False, "message": "Download cancelled"})

            message = f"faster-whisper {model_name} downloaded to: {local_dir}"
            print(f"[OK] {message}")

            finish_download(download_key)

        except Exception as e:
            finish_download(download_key, error=e)
            raise

        return jsonify({
            "success": True,
            "message": message,
            "model_name": model_name,
            "path": local_dir,
        })

    except Exception as e:
        print(f"[ERROR] Error downloading faster-whisper model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/faster-whisper/remove", methods=["POST"])
def remove_faster_whisper_model():
    """Remove a downloaded faster-whisper model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_name = data.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "model_name required"}), 400

        model_path = safe_model_path(MODELS_DIR, f"faster-whisper-{model_name}")
        if model_path is None:
            return jsonify({"success": False, "error": "Invalid model name"}), 400

        if not os.path.exists(model_path):
            return jsonify({
                "success": False,
                "error": f"Model not found: faster-whisper-{model_name}",
            }), 404

        import shutil
        shutil.rmtree(model_path)
        print(f"[OK] Removed faster-whisper model: {model_path}")

        return jsonify({
            "success": True,
            "message": f"Successfully removed faster-whisper-{model_name}",
            "model_name": model_name,
        })

    except Exception as e:
        print(f"[ERROR] Error removing faster-whisper model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/list", methods=["GET"])
def list_models():
    """List available and downloaded models"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Get available Whisper models (from cache or defaults)
        all_whisper_models = get_whisper_models_list()

        downloaded_whisper = []

        # Check both old Whisper cache directory and new ./models location
        whisper_cache_old = os.path.expanduser("~/.cache/whisper")
        models_dir = MODELS_DIR

        # Check old cache location for backward compatibility
        if os.path.exists(whisper_cache_old):
            for item in os.listdir(whisper_cache_old):
                if item.endswith(".pt"):
                    # Extract model name from filename (e.g., 'base.pt' -> 'base')
                    model_name = item.replace(".pt", "")
                    # Handle .en variants
                    if model_name.endswith(".en"):
                        base_name = model_name.replace(".en", "")
                        if base_name not in downloaded_whisper:
                            downloaded_whisper.append(base_name)
                    else:
                        downloaded_whisper.append(model_name)

        # Check new ./models/whisper-* directories
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if item.startswith("whisper-"):
                    whisper_model_dir = os.path.join(models_dir, item)
                    if os.path.isdir(whisper_model_dir):
                        # Check if directory contains .pt files
                        for file in os.listdir(whisper_model_dir):
                            if file.endswith(".pt"):
                                # Extract model name from directory (e.g., 'whisper-base' -> 'base')
                                model_name = item.replace("whisper-", "")
                                if model_name not in downloaded_whisper:
                                    downloaded_whisper.append(model_name)
                                break

        # Create whisper models list with download status and details
        whisper_models = []
        for model_name, details in all_whisper_models.items():
            whisper_models.append(
                {
                    "name": model_name,
                    "downloaded": model_name in downloaded_whisper,
                    "params": details["params"],
                    "size": details["size"],
                    "desc": details["desc"],
                    "lang": details["lang"],
                }
            )

        # Get downloaded/custom models (this would scan a models directory)
        downloaded_models = []
        models_dir = MODELS_DIR
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, item)):
                    # Skip internal cache/data directories
                    if item.startswith(".") or item in ("tts", "piper"):
                        continue

                    # Only count directories that still hold a weight file: a
                    # partial delete (or an interrupted download) leaves the
                    # directory with just config/tokenizer files, and listing
                    # that as downloaded is exactly what mismatched nllb-status
                    # / faster-whisper/list (which both require a weight file).
                    if not dir_has_weights(os.path.join(models_dir, item)):
                        continue

                    # Detect if it's a HuggingFace model (contains --)
                    if "--" in item:
                        # HuggingFace model - convert back to original ID
                        model_id = item.replace("--", "/")
                        downloaded_models.append(
                            {
                                "name": model_id,
                                "type": "huggingface",
                                "path": os.path.join(models_dir, item),
                                "directory": item,
                            }
                        )
                    else:
                        # Local/uploaded model
                        downloaded_models.append(
                            {
                                "name": item,
                                "type": "local",
                                "path": os.path.join(models_dir, item),
                                "directory": item,
                            }
                        )

        # Add downloaded Piper TTS models
        for m in _PIPER_MODELS_CATALOG:
            if _is_piper_model_downloaded(m["id"]):
                downloaded_models.append(
                    {
                        "name": m["name"],
                        "type": "piper",
                        "path": _get_piper_model_dir(m["id"]),
                        "directory": m["id"],
                    }
                )

        return jsonify(
            {
                "success": True,
                "whisper_models": whisper_models,
                "downloaded_models": downloaded_models,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/remove", methods=["POST"])
def remove_model():
    """Remove a downloaded model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        data = request.get_json()
        model_name = data.get("model_name")
        model_type = data.get("model_type")  # 'whisper', 'huggingface', 'local'

        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"})

        if model_type == "whisper":
            # Can't remove built-in Whisper models
            return jsonify(
                {"success": False, "error": "Cannot remove built-in Whisper models"}
            )

        elif model_type == "huggingface":
            # Remove HuggingFace model directory
            # Convert model ID (org/name) to directory name (org--name)
            model_dir_name = model_name.replace("/", "--")
            model_path = safe_model_path(MODELS_DIR, model_dir_name)
            if model_path is None:
                return jsonify({"success": False, "error": "Invalid model name"}), 400

            if os.path.exists(model_path):
                import shutil

                shutil.rmtree(model_path)
                return jsonify(
                    {"success": True, "message": f"Successfully removed {model_name}"}
                )
            else:
                return jsonify(
                    {"success": False, "error": f"Model {model_name} not found"}
                )

        elif model_type == "local":
            # Remove local model directory
            model_path = safe_model_path(MODELS_DIR, model_name)
            if model_path is None:
                return jsonify({"success": False, "error": "Invalid model name"}), 400

            if os.path.exists(model_path):
                import shutil

                shutil.rmtree(model_path)
                return jsonify(
                    {"success": True, "message": f"Successfully removed {model_name}"}
                )
            else:
                return jsonify(
                    {"success": False, "error": f"Model {model_name} not found"}
                )

        else:
            return jsonify({"success": False, "error": "Invalid model type"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/nllb-status", methods=["GET"])
def nllb_status():
    """Check if NLLB translation model is downloaded"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        models_dir = MODELS_DIR

        # Check ALL translation model directories (NLLB + MADLAD), not just 600M
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if (item.startswith("facebook--nllb-") or item.startswith("google--madlad400-")) and os.path.isdir(os.path.join(models_dir, item)):
                    nllb_path = os.path.join(models_dir, item)
                    has_model = False
                    total_size = 0
                    for root, _dirs, files in os.walk(nllb_path):
                        for f in files:
                            file_path = os.path.join(root, f)
                            total_size += os.path.getsize(file_path)
                            if is_weight_file(f):
                                has_model = True

                    if has_model:
                        size_gb = total_size / (1024 * 1024 * 1024)
                        model_id = item.replace("--", "/")
                        return jsonify({
                            "success": True,
                            "downloaded": True,
                            "path": nllb_path,
                            "model_id": model_id,
                            "size": f"{size_gb:.2f} GB"
                        })

        # Also check HuggingFace cache as fallback
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M")
        if os.path.exists(hf_cache):
            # Check if download is complete by looking for model files in snapshots
            snapshots_dir = os.path.join(hf_cache, "snapshots")
            if os.path.exists(snapshots_dir):
                for snapshot in os.listdir(snapshots_dir):
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if os.path.isdir(snapshot_path) and has_weight_file(os.listdir(snapshot_path)):
                        # Model exists in cache, offer to move it
                        return jsonify({
                            "success": True,
                            "downloaded": False,
                            "in_cache": True,
                            "cache_path": hf_cache,
                            "message": "Model found in HuggingFace cache. Click download to move it to ./models/"
                        })

            # Partial download exists
            return jsonify({
                "success": True,
                "downloaded": False,
                "partial": True,
                "message": "Partial download found in cache. Click download to complete."
            })

        return jsonify({
            "success": True,
            "downloaded": False,
            "message": "NLLB model not downloaded"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Track NLLB download progress globally
nllb_download_progress = {"status": "idle", "progress": 0, "message": ""}


@app.route("/api/models/nllb-download-progress", methods=["GET"])
def nllb_download_progress_endpoint():
    """Get NLLB download progress"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global nllb_download_progress
    return jsonify({"success": True, **nllb_download_progress})


# Cache for NLLB models list
_nllb_models_cache = {"models": [], "last_updated": 0}


@app.route("/api/models/nllb-list", methods=["GET"])
def list_nllb_models():
    """List available NLLB translation models from HuggingFace"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    global _nllb_models_cache
    import time

    # Check if we should use cache (valid for 1 hour)
    refresh = request.args.get("refresh", "false").lower() == "true"
    cache_valid = (time.time() - _nllb_models_cache["last_updated"]) < 3600

    if cache_valid and _nllb_models_cache["models"] and not refresh:
        models = _nllb_models_cache["models"]
    else:
        # Fetch from HuggingFace API
        try:
            import requests

            # Search for NLLB models from Facebook
            response = requests.get(
                "https://huggingface.co/api/models",
                params={
                    "search": "nllb",
                    "author": "facebook",
                    "filter": "translation",
                    "limit": 50
                },
                timeout=10
            )

            if response.status_code == 200:
                hf_models = response.json()
                models = []

                for m in hf_models:
                    model_id = m.get("modelId", "")
                    # Only include NLLB models
                    if "nllb" in model_id.lower():
                        # Determine size from model name
                        size = "Unknown"
                        if "distilled-600M" in model_id:
                            size = "~1.2 GB"
                            size_order = 1
                        elif "distilled-1.3B" in model_id:
                            size = "~2.6 GB"
                            size_order = 2
                        elif "1.3B" in model_id:
                            size = "~5.2 GB"
                            size_order = 3
                        elif "3.3B" in model_id:
                            size = "~13 GB"
                            size_order = 4
                        elif "moe" in model_id.lower():
                            size = "~17 GB"
                            size_order = 5
                        else:
                            size_order = 10

                        models.append({
                            "model_id": model_id,
                            "name": model_id.split("/")[-1],
                            "size": size,
                            "size_order": size_order,
                            "downloads": m.get("downloads", 0),
                            "likes": m.get("likes", 0),
                            "description": get_nllb_model_description(model_id)
                        })

                # Sort by size order
                models.sort(key=lambda x: x["size_order"])

                # Update cache
                _nllb_models_cache = {"models": models, "last_updated": time.time()}
            else:
                # Fallback to known models if API fails
                models = get_default_nllb_models()

        except Exception as e:
            print(f"[WARN] Failed to fetch NLLB models from HuggingFace: {e}")
            models = get_default_nllb_models()

    # Check which models are downloaded
    models_dir = MODELS_DIR
    for model in models:
        dir_name = model["model_id"].replace("/", "--")
        model["downloaded"] = dir_has_weights(os.path.join(models_dir, dir_name))

    return jsonify({"success": True, "models": models})


@app.route("/api/models/madlad-list", methods=["GET"])
def list_madlad_models():
    """List available MADLAD-400 translation models (static catalog + downloaded
    flags). MADLAD has only a couple of relevant sizes, so no live HF search."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    models = get_default_madlad_models()
    models_dir = MODELS_DIR
    for model in models:
        dir_name = model["model_id"].replace("/", "--")
        model["downloaded"] = dir_has_weights(os.path.join(models_dir, dir_name))
    return jsonify({"success": True, "models": models})


# ============== Silero VAD Status ==============
# Note: Silero VAD is now handled via pip package (silero-vad>=4.0.0)
# No separate download needed - model is bundled with the package


@app.route("/api/models/silero-vad-status", methods=["GET"])
def silero_vad_status():
    """Check if Silero VAD is available (via pip package)"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Check if silero-vad pip package is installed
        import importlib.util
        if importlib.util.find_spec("silero_vad") is None:
            raise ImportError("silero_vad not installed")
        return jsonify({
            "success": True,
            "downloaded": True,
            "source": "pip package (silero-vad)",
            "message": "Silero VAD available via pip package"
        })
    except ImportError:
        return jsonify({
            "success": True,
            "downloaded": False,
            "message": "Install with: pip install silero-vad"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ============== PANNs music/speech detector status + download ==============
PANNS_CKPT_SIZE = 327_000_000  # ~312 MB CNN14 checkpoint (approx, for progress %)


@app.route("/api/models/panns-status", methods=["GET"])
def panns_status():
    """Report whether the PANNs package is installed and the checkpoint downloaded."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    installed = panns_package_installed()
    ckpt = panns_checkpoint_path()
    downloaded = os.path.exists(ckpt)
    # Self-heal the AudioSet label CSV so a missing/0-byte file doesn't silently
    # break detection; then report whether labels are present.
    ensure_panns_labels_csv()
    labels_csv = panns_labels_home_path()
    labels_ok = os.path.exists(labels_csv) and os.path.getsize(labels_csv) >= _PANNS_LABELS_MIN_BYTES
    if not installed:
        msg = "panns-inference not installed. Install with: pip install panns-inference"
    elif not downloaded:
        msg = "PANNs CNN14 checkpoint not downloaded."
    elif not labels_ok:
        msg = "PANNs AudioSet labels missing; detection will not classify music."
    else:
        msg = "PANNs music/speech detector ready."
    return jsonify({
        "success": True,
        "package_installed": installed,
        "downloaded": bool(installed and downloaded and labels_ok),
        "checkpoint_path": ckpt,
        "labels_present": bool(labels_ok),
        "message": msg,
    })


@app.route("/api/models/panns/download", methods=["POST"])
def download_panns_model():
    """Download the PANNs CNN14 checkpoint in the background (progress via the
    shared download tracker, same as other model downloads)."""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    if not panns_package_installed():
        return jsonify({"success": False, "error": "panns-inference is not installed. Install it first: pip install panns-inference"}), 400

    dest = panns_checkpoint_path()
    if os.path.exists(dest):
        ensure_panns_labels_csv()  # checkpoint present but labels may still be missing
        return jsonify({"success": True, "message": "Checkpoint already downloaded"})

    key = "panns_cnn14"
    if not try_register_download(key, total=PANNS_CKPT_SIZE):
        return jsonify({"success": False, "error": "Download already in progress"}), 409

    def worker():
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # The checkpoint is useless without the AudioSet label CSV, so make sure
            # it's in place too (the detector loads labels at import time).
            ensure_panns_labels_csv()
            start_download_monitor(key, dest, total=PANNS_CKPT_SIZE)
            result = download_url_to_file(
                PANNS_CHECKPOINT_URL, dest,
                cancel_check=lambda: key in cancelled_downloads,
            )
            if result == "cancelled":
                finish_download(key, cancelled=True)
                return
            finish_download(key)
            # No global reset needed: the detector runs in the worker process and
            # re-checks the checkpoint each tick, so it picks this up without a restart.
            print(f"[PANNS] Checkpoint downloaded to {dest}")
        except Exception as e:
            finish_download(key, error=e)
            print(f"[PANNS] Checkpoint download failed: {e}")

    threading.Thread(target=worker, daemon=True, name="dl-panns").start()
    return jsonify({"success": True, "message": "Download started"})


@app.route("/api/models/translation/download", methods=["POST"])
def download_translation_model():
    """Download any translation model to ./models/ directory"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    not_writable = _models_not_writable_response()
    if not_writable:
        return not_writable

    global nllb_download_progress

    try:
        import threading

        data = request.get_json()
        model_id = data.get("model_id", "facebook/nllb-200-distilled-600M")

        # The status shim below is a single global, so only one translation
        # download can run at a time
        if nllb_download_progress.get("status") in ["downloading", "starting"]:
            return jsonify({"success": False, "error": "Download already in progress"})

        # Best-effort total size so progress can be a real percentage
        expected_total = None
        try:
            from huggingface_hub import HfApi
            repo_info = HfApi().model_info(model_id, files_metadata=True)
            expected_total = sum(f.size or 0 for f in repo_info.siblings) or None
        except Exception as e:
            print(f"[WARNING] Could not get size of {model_id}: {e}")

        # Atomic per-model registration in the shared download tracker
        if not try_register_download(model_id, total=expected_total):
            return jsonify({"success": False, "error": "Download already in progress"}), 409

        def download_model():
            global nllb_download_progress
            import time
            import logging

            # Set up detailed logging
            log_file = os.path.join(APP_DIR, "logs", "translation_download.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Configure file handler for this download
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            dl_logger = logging.getLogger('translation_download')
            dl_logger.setLevel(logging.DEBUG)
            dl_logger.addHandler(file_handler)

            start_time = time.time()
            dl_logger.info("=" * 60)
            dl_logger.info(f"Starting download of {model_id}")
            dl_logger.info("=" * 60)

            try:
                models_dir = MODELS_DIR
                os.makedirs(models_dir, exist_ok=True)
                model_dir_name = model_id.replace("/", "--")
                model_path = os.path.join(models_dir, model_dir_name)

                dl_logger.info(f"Target directory: {model_path}")
                dl_logger.info(f"Checking if directory exists: {os.path.exists(model_path)}")

                nllb_download_progress = {"status": "downloading", "progress": 10, "message": f"Downloading {model_id}..."}

                # Start a background thread to monitor progress
                stop_monitor = threading.Event()

                def monitor_progress():
                    """Monitor download progress by checking file sizes"""
                    # Without this, the assignment below creates a local var and
                    # the UI never sees these updates (global doesn't inherit
                    # from the enclosing function's declaration)
                    global nllb_download_progress
                    last_size = 0
                    stall_count = 0
                    while not stop_monitor.is_set():
                        try:
                            # Calculate total size of model directory
                            total_size = 0
                            incomplete_files = []
                            for root, _dirs, files in os.walk(model_path):
                                for f in files:
                                    fp = os.path.join(root, f)
                                    try:
                                        size = os.path.getsize(fp)
                                        total_size += size
                                        if f.endswith('.incomplete'):
                                            incomplete_files.append((f, size))
                                    except OSError:
                                        pass

                            size_mb = total_size / (1024 * 1024)
                            speed = (total_size - last_size) / (1024 * 1024)  # MB in last second

                            # Log progress
                            if incomplete_files:
                                for fname, fsize in incomplete_files:
                                    dl_logger.debug(f"Incomplete file: {fname[:30]}... = {fsize / (1024*1024):.1f} MB")

                            dl_logger.info(f"Progress: {size_mb:.1f} MB downloaded, speed: {speed:.2f} MB/s")

                            # Update progress for UI
                            if expected_total:
                                progress = min(99, int((total_size / expected_total) * 100))
                            else:
                                # No known total: estimate against ~2.5GB (typical NLLB)
                                progress = min(85, int(10 + (size_mb / 2500) * 75))
                            nllb_download_progress = {
                                "status": "downloading",
                                "progress": progress,
                                "message": f"Downloading: {size_mb:.0f} MB ({speed:.1f} MB/s)"
                            }

                            # Mirror into the shared download tracker (main status endpoint)
                            with active_downloads_lock:
                                entry = active_downloads.get(model_id)
                                if entry and entry.get("status") == "downloading":
                                    entry["downloaded"] = total_size
                                    entry["last_update"] = time.time()
                                    if expected_total:
                                        entry["percentage"] = min(int((total_size / expected_total) * 100), 99)

                            # Detect stalls
                            if total_size == last_size and total_size > 0:
                                stall_count += 1
                                if stall_count >= 30:  # 30 seconds of no progress
                                    dl_logger.warning(f"Download appears stalled for {stall_count} seconds!")
                            else:
                                stall_count = 0

                            last_size = total_size
                        except Exception as e:
                            dl_logger.error(f"Monitor error: {e}")

                        time.sleep(1)

                monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
                monitor_thread.start()
                dl_logger.info("Started progress monitor thread")

                # Download files using wget for reliability (huggingface_hub hangs on large files)
                dl_logger.info("Fetching file list from HuggingFace...")
                try:
                    from huggingface_hub import list_repo_files, hf_hub_url

                    # Get list of files in the repo
                    files = list_repo_files(repo_id=model_id)
                    dl_logger.info(f"Found {len(files)} files to download: {files}")

                    # Download each file using wget
                    for idx, filename in enumerate(files):
                        dest_path = os.path.join(model_path, filename)
                        os.makedirs(os.path.dirname(dest_path) if os.path.dirname(dest_path) else model_path, exist_ok=True)

                        # Skip if already downloaded and has content
                        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                            dl_logger.info(f"Already exists: {filename}")
                            continue

                        dl_logger.info(f"Downloading file {idx+1}/{len(files)}: {filename}")
                        nllb_download_progress = {
                            "status": "downloading",
                            "progress": int(10 + (idx / len(files)) * 70),
                            "message": f"Downloading {filename}..."
                        }

                        # Get download URL from HuggingFace
                        url = hf_hub_url(repo_id=model_id, filename=filename)
                        dl_logger.info(f"URL: {url}")

                        # Download with resume + retry, checking cancellation mid-file
                        outcome = download_url_to_file(
                            url, dest_path,
                            cancel_check=lambda: model_id in cancelled_downloads,
                            log=dl_logger.info,
                        )
                        if outcome == "cancelled":
                            dl_logger.info(f"Download cancelled for {model_id}")
                            nllb_download_progress = {"status": "error", "progress": 0, "message": "Download cancelled"}
                            finish_download(model_id, cancelled=True)
                            return

                        dl_logger.info(f"Successfully downloaded: {filename}")

                    dl_logger.info("All files downloaded successfully")
                except Exception as download_error:
                    dl_logger.error(f"Download failed: {type(download_error).__name__}: {download_error}")
                    raise
                finally:
                    stop_monitor.set()
                    dl_logger.info("Stopped progress monitor")

                elapsed = time.time() - start_time
                dl_logger.info(f"Download phase completed in {elapsed:.1f} seconds")

                # Post-download: check for incomplete files and finalize them
                nllb_download_progress = {"status": "downloading", "progress": 90, "message": "Finalizing download..."}
                dl_logger.info("Checking for incomplete files...")

                cache_dir = os.path.join(model_path, ".cache")
                if os.path.exists(cache_dir):
                    dl_logger.info(f"Found cache directory: {cache_dir}")
                    # Look for incomplete files that are actually complete
                    for root, _dirs, files in os.walk(cache_dir):
                        for f in files:
                            if f.endswith(".incomplete"):
                                incomplete_path = os.path.join(root, f)
                                file_size = os.path.getsize(incomplete_path)
                                dl_logger.info(f"Found incomplete file: {f[:40]}... size={file_size / (1024*1024):.1f} MB")

                                # If file is large (>100MB), it's likely the model weights
                                if file_size > 100_000_000:
                                    # Check if pytorch_model.bin or model.safetensors exists
                                    model_bin = os.path.join(model_path, "pytorch_model.bin")
                                    model_safetensors = os.path.join(model_path, "model.safetensors")
                                    if not os.path.exists(model_bin) and not os.path.exists(model_safetensors):
                                        dl_logger.info("Copying incomplete file to pytorch_model.bin")
                                        import shutil
                                        shutil.copy2(incomplete_path, model_bin)
                                        dl_logger.info("Copy completed")
                    # Clean up cache
                    dl_logger.info("Cleaning up cache directory")
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                else:
                    dl_logger.info("No cache directory found (download completed normally)")

                # Verify final state
                final_files = os.listdir(model_path)
                dl_logger.info(f"Final files in model directory: {final_files}")

                total_elapsed = time.time() - start_time
                dl_logger.info(f"Download complete! Total time: {total_elapsed:.1f} seconds")
                dl_logger.info("=" * 60)

                nllb_download_progress = {"status": "complete", "progress": 100, "message": "Download complete!"}
                finish_download(model_id)

            except Exception as e:
                import traceback
                dl_logger.error(f"Download failed: {type(e).__name__}: {e}")
                dl_logger.error(traceback.format_exc())
                nllb_download_progress = {"status": "error", "progress": 0, "message": str(e)}
                finish_download(model_id, error=e)
            finally:
                dl_logger.removeHandler(file_handler)
                file_handler.close()

        nllb_download_progress = {"status": "starting", "progress": 0, "message": "Starting download..."}
        thread = threading.Thread(target=download_model)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "message": f"Download started for {model_id}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/translation/remove", methods=["POST"])
def remove_translation_model():
    """Remove a downloaded translation model"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        import shutil
        data = request.get_json()
        model_id = data.get("model_id")

        if not model_id:
            return jsonify({"success": False, "error": "model_id is required"})

        model_dir_name = model_id.replace("/", "--")
        model_path = safe_model_path(MODELS_DIR, model_dir_name)
        if model_path is None:
            return jsonify({"success": False, "error": "Invalid model id"}), 400

        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            return jsonify({"success": True, "message": f"Removed {model_id}"})
        else:
            return jsonify({"success": False, "error": f"Model {model_id} not found"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/models/upload-local", methods=["POST"])
def upload_local_model():
    """Upload a local model file or folder to the server"""
    if not check_ip_whitelist():
        return jsonify({"success": False, "error": "Access Denied"}), 403

    try:
        # Check if the post request has the file part
        if "files[]" not in request.files:
            return jsonify({"success": False, "error": "No files uploaded"}), 400

        files = request.files.getlist("files[]")
        model_name = request.form.get("model_name")

        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400

        # Reject path separators / traversal so the model dir stays inside MODELS_DIR
        if not re.fullmatch(r"[\w.\- ]+", model_name) or model_name.strip(". ") == "":
            return jsonify({"success": False, "error": "Invalid model name"}), 400

        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No files selected"}), 400

        # Create models directory if it doesn't exist
        models_dir = MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)

        # Create model directory
        model_path = os.path.join(models_dir, model_name)

        if os.path.exists(model_path):
            return jsonify(
                {
                    "success": False,
                    "error": f'Model "{model_name}" already exists. Please choose a different name or remove the existing model first.',
                }
            ), 400

        os.makedirs(model_path, exist_ok=True)

        # Save uploaded files
        saved_files = []
        for file in files:
            if file and file.filename:
                # Sanitize filename
                filename = os.path.basename(file.filename)
                file_path = os.path.join(model_path, filename)

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                file.save(file_path)
                saved_files.append(filename)

        print(f"[OK] Uploaded {len(saved_files)} files for model: {model_name}")

        return jsonify(
            {
                "success": True,
                "message": f'Successfully uploaded model "{model_name}" with {len(saved_files)} file(s)',
                "model_name": model_name,
                "files_count": len(saved_files),
            }
        )

    except Exception as e:
        print(f"[ERROR] Error uploading local model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def _socket_auth_ok():
    """Auth gate for mutating Socket.IO handlers. HTTP routes call
    check_ip_whitelist() inline; handlers historically skipped it, so an
    unauthenticated client could rewrite/delete transcription rows or persist
    config changes over the socket. Returns True when the client is allowed.
    request context (cookies/headers/remote_addr) is available in handlers
    under async_mode='threading'."""
    try:
        return check_ip_whitelist()
    except Exception:
        # Fail closed: if the gate can't be evaluated, deny the mutation.
        return False


@socketio.on("connect")
def handle_connect():
    # print('Client connected')
    emit("connected", {"data": "Connected to Alexs server"})


@socketio.on("disconnect")
def handle_disconnect():
    emit("connected", {"data": "Disconnected from Alexs server"})


@socketio.on("request_all_entries")
def handle_request_all_entries():
    """Send all historical transcription entries to the requesting client only"""
    entries = get_new_entries(limit_override=0)  # 0 = no limit
    segments = [
        {
            "id": e[0], "timestamp": e[1], "text": e[2], "start": e[3], "end": e[4],
            "completed": True,
            "needs_review": bool(e[6]) if len(e) > 6 and e[6] is not None else False,
            "speech_type": e[9] if len(e) > 9 else None,
            "denied": bool(e[10]) if len(e) > 10 and e[10] is not None else False,
            "denied_reason": e[11] if len(e) > 11 else None,
            "marked": bool(e[13]) if len(e) > 13 and e[13] is not None else False,
        }
        for e in entries
    ]
    _attach_segment_ids(segments)
    emit("transcription_update", {
        "segments": segments,
        "in_progress_segment": None,
        "entries": [(e[1], e[2]) for e in entries],
        "in_progress": "",
        "is_running": transcription_state.get("running", False),
        "session_id": transcription_state.get("session_id"),
    })


@socketio.on("request_all_translation_entries")
def handle_request_all_translation_entries():
    """Send all historical translation entries to the requesting client only"""
    trans_config = config.get("live_translation", {})
    if not trans_config.get("enabled", False):
        # Translation is off — tell the client so the translate view can say so
        emit("translation_update", {
            "segments": [],
            "in_progress": None,
            "target_language": trans_config.get("target_language", "en"),
            "source_language": trans_config.get("source_language", "auto"),
            "enabled": False,
            "is_running": transcription_state.get("running", False),
            "session_id": transcription_state.get("session_id"),
        })
        return

    target_lang = trans_config.get("target_language", "en")
    source_lang = trans_config.get("source_language", "auto")
    if source_lang == "auto":
        source_lang = config.get("audio", {}).get("language", "en")
        if source_lang == "auto":
            source_lang = "en"

    entries = get_new_entries(limit_override=0)  # 0 = no limit
    cache = get_translation_cache()
    translated_segments = []

    for entry in entries:
        seg_id = entry[0]
        original_text = entry[2]
        cached = cache.get(seg_id, original_text, target_lang)
        if cached:
            translated_text = cached
        elif len(entry) > 7 and entry[7]:
            # Cache cold (e.g. server restart): serve the persisted translation
            # and seed the cache, same as the emit loop's db_seed branch.
            translated_text = entry[7]
            db_lang = entry[8] if len(entry) > 8 and entry[8] else target_lang
            cache.set(seg_id, original_text, translated_text, db_lang)
        else:
            # No cached or persisted translation — never translate live here.
            # This handler runs synchronously on every client connect and has
            # no warmup gate: an unready model echoes the source text, which
            # would get cached and later mutate into the real translation (the
            # post-restart repeating-display bug). The emit loop picks up the
            # miss within a few cycles with its warmup gate, per-cycle budget,
            # and DB write-back.
            continue

        if is_whisper_hallucination(translated_text):
            continue

        translated_segments.append({
            "id": seg_id,
            "timestamp": entry[1],
            "original_text": original_text,
            "translated_text": translated_text,
            "start": entry[3],
            "end": entry[4],
            "completed": True,
            "denied": bool(entry[10]) if len(entry) > 10 and entry[10] is not None else False,
        })

    _attach_segment_ids(translated_segments)
    emit("translation_update", {
        "segments": translated_segments,
        "in_progress": None,
        "target_language": target_lang,
        "target_language_name": TRANSLATION_LANGUAGES.get(target_lang, target_lang),
        "source_language": source_lang,
        "enabled": True,
        "is_running": transcription_state.get("running", False),
        "session_id": transcription_state.get("session_id"),
    })


# =============================================================================
# Real-Time Corrections Socket.IO Handlers
# =============================================================================


@socketio.on("submit_correction")
def handle_submit_correction(data):
    """Handle correction submitted via Socket.IO"""
    if not _socket_auth_ok():
        emit("correction_error", {"error": "Access denied"})
        return
    if not data:
        return

    segment_id = data.get("segment_id")
    new_text = data.get("new_text", "").strip()
    correction_type = data.get("correction_type", "manual")

    if segment_id is None or not new_text:
        emit("correction_error", {"error": "segment_id and new_text are required"})
        return

    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        emit("correction_error", {"error": "No active database"})
        return

    try:
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE transcriptions
                   SET original_text = COALESCE(original_text, text),
                       text = ?,
                       corrected_by = ?
                   WHERE id = ?""",
                (new_text, correction_type, segment_id),
            )
            conn.commit()

        # Invalidate caches
        with _cache_lock:
            _db_cache["last_entries"] = []
            _db_cache["last_fetch_time"] = 0

        cache = get_translation_cache()
        cache.invalidate(segment_id)

        # Re-translate if needed
        translated_text = None
        trans_config = config.get("live_translation", {})
        if trans_config.get("enabled", False):
            target_lang = trans_config.get("target_language", "en")
            source_lang = trans_config.get("source_language", "auto")
            if source_lang == "auto":
                source_lang = config.get("audio", {}).get("language", "en")
                if source_lang == "auto":
                    source_lang = "en"
            translated_text = translate_live_text(new_text, source_lang, target_lang)
            if translated_text:
                cache.set(segment_id, new_text, translated_text, target_lang)

        # Broadcast to all clients
        socketio.emit("correction_applied", {
            "segment_id": segment_id,
            "new_text": new_text,
            "corrected_by": correction_type,
            "translated_text": translated_text,
        })

    except Exception as e:
        print(f"[CORRECTION SOCKET ERROR] {e}")
        emit("correction_error", {"error": str(e)})


@socketio.on("mark_reviewed")
def handle_mark_reviewed(data):
    """Mark segments as reviewed via Socket.IO"""
    if not _socket_auth_ok():
        return
    if not data:
        return

    segment_ids = data.get("segment_ids", [])
    if not segment_ids:
        return

    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                placeholders = ",".join("?" for _ in segment_ids)
                conn.execute(
                    f"UPDATE transcriptions SET needs_review = 0 WHERE id IN ({placeholders})",
                    segment_ids,
                )
                conn.commit()
            finally:
                conn.close()

        _invalidate_entries_cache()

    except Exception as e:
        print(f"[MARK REVIEWED SOCKET ERROR] {e}")


@socketio.on("submit_translation_correction")
def handle_translation_correction(data):
    """Handle correction of translated text — updates TranslationCache only"""
    if not _socket_auth_ok():
        emit("correction_error", {"error": "Access denied"})
        return
    if not data:
        return

    segment_id = data.get("segment_id")
    new_translated_text = data.get("new_translated_text", "").strip()

    if segment_id is None or not new_translated_text:
        emit("correction_error", {"error": "segment_id and new_translated_text are required"})
        return

    try:
        cache = get_translation_cache()
        # Get the current cache entry to preserve original text and target lang
        with cache._lock:
            entry = cache._cache.get(segment_id)
            if entry:
                entry['translated'] = new_translated_text
            else:
                # No cache entry — create one with the corrected text
                trans_config = config.get("live_translation", {})
                target_lang = trans_config.get("target_language", "en")
                cache._cache[segment_id] = {
                    'original': '',
                    'translated': new_translated_text,
                    'target_lang': target_lang,
                }

        # Broadcast to all clients
        socketio.emit("translation_correction_applied", {
            "segment_id": segment_id,
            "new_translated_text": new_translated_text,
        })

    except Exception as e:
        print(f"[TRANSLATION CORRECTION ERROR] {e}")
        emit("correction_error", {"error": str(e)})


@socketio.on("select_translation_alternative")
def handle_select_translation_alternative(data):
    """Handle selection of a translation alternative"""
    if not _socket_auth_ok():
        return
    if not data:
        return

    segment_id = data.get("segment_id")
    alternative_text = data.get("alternative_text", "").strip()

    if segment_id is None or not alternative_text:
        return

    try:
        cache = get_translation_cache()
        with cache._lock:
            entry = cache._cache.get(segment_id)
            if entry:
                entry['translated'] = alternative_text

        socketio.emit("translation_correction_applied", {
            "segment_id": segment_id,
            "new_translated_text": alternative_text,
        })

    except Exception as e:
        print(f"[TRANSLATION ALT SELECT ERROR] {e}")


# =============================================================================
# Audio Streaming Socket.IO Handlers
# =============================================================================

@socketio.on("join_audio_stream")
def handle_join_audio_stream():
    """Client wants to receive live microphone audio"""
    from flask_socketio import join_room
    join_room("audio_stream")
    transcription_state["audio_stream_enabled"] = True
    emit("audio_stream_info", {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16
    })


@socketio.on("leave_audio_stream")
def handle_leave_audio_stream():
    """Client no longer wants live microphone audio"""
    from flask_socketio import leave_room
    leave_room("audio_stream")


@socketio.on("join_tts_audio")
def handle_join_tts_audio():
    """Client wants to receive TTS audio for translated text"""
    from flask_socketio import join_room
    join_room("tts_audio")


@socketio.on("leave_tts_audio")
def handle_leave_tts_audio():
    """Client no longer wants TTS audio"""
    from flask_socketio import leave_room
    leave_room("tts_audio")


# =============================================================================
# Staging Buffer for Output Delay
# =============================================================================


# "Staged" segments are ordinary DB rows whose timestamp is younger than the
# configured delay — emit_new_entries tags them and withholds them from the
# live view. Approve/discard therefore operate on the DB rows directly.


def _invalidate_entries_cache():
    """Force the next emit cycle to re-read the DB."""
    with _cache_lock:
        _db_cache["last_entries"] = []
        _db_cache["last_fetch_time"] = 0


def _open_db_writer():
    """Open a short-lived writer connection to the active session database.

    The caller MUST hold ``_db_lock`` so these UI-triggered writes serialize
    against the transcription thread's persistent-connection writes instead of
    racing them. A long ``busy_timeout`` is also set as defense-in-depth so a
    concurrent writer waits rather than failing immediately with
    "database is locked". Returns ``None`` if there is no active DB.
    """
    current_db_name = transcription_state.get("db_name")
    if not current_db_name or not os.path.exists(current_db_name):
        return None
    conn = sqlite3.connect(current_db_name, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@socketio.on("toggle_delay")
def handle_toggle_delay(data):
    """Toggle the output delay on/off"""
    if not _socket_auth_ok():
        return
    if not data:
        return
    enabled = data.get("enabled", False)
    config.setdefault("corrections", {}).setdefault("output_delay", {})["enabled"] = enabled
    save_config(config)

    # Disabling needs no flush: rows are already in the DB, and the next
    # emit cycle publishes everything once the gate is off
    socketio.emit("delay_status", {"enabled": enabled})


@socketio.on("set_delay_seconds")
def handle_set_delay_seconds(data):
    """Update the output delay duration"""
    if not _socket_auth_ok():
        return
    if not data:
        return
    seconds = max(2, min(30, int(data.get("delay_seconds", 7))))
    config.setdefault("corrections", {}).setdefault("output_delay", {})["delay_seconds"] = seconds
    save_config(config)


def _backdate_staged_rows(seg_id=None):
    """Publish staged row(s) immediately by backdating their timestamp past
    the delay window. seg_id None = all rows still inside the window."""
    delay_seconds = config.get("corrections", {}).get("output_delay", {}).get("delay_seconds", 7)
    # Match the emit-time age check, which is taken in configured_timezone — the same
    # zone the rows were stamped in.
    backdated = (datetime.now(configured_timezone)
                 - timedelta(seconds=delay_seconds + 1)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                if seg_id is not None:
                    conn.execute("UPDATE transcriptions SET timestamp = ? WHERE id = ?", (backdated, int(seg_id)))
                else:
                    cutoff = (datetime.now(configured_timezone)
                              - timedelta(seconds=delay_seconds)).strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute("UPDATE transcriptions SET timestamp = ? WHERE timestamp > ?", (backdated, cutoff))
                conn.commit()
            finally:
                conn.close()
        _invalidate_entries_cache()
    except Exception as e:
        print(f"[STAGING] Approve failed: {e}")


@socketio.on("approve_staged")
def handle_approve_staged(data):
    """Approve one or all staged segments for immediate publishing"""
    if not _socket_auth_ok():
        return
    if not data:
        return
    if data.get("all", False):
        _backdate_staged_rows()
    else:
        staging_id = data.get("staging_id")
        if staging_id is not None:
            _backdate_staged_rows(staging_id)


@socketio.on("discard_staged")
def handle_discard_staged(data):
    """Discard a staged segment (delete the row before it publishes)"""
    if not _socket_auth_ok():
        return
    if not data:
        return
    staging_id = data.get("staging_id")
    if staging_id is None:
        return
    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                conn.execute("DELETE FROM transcriptions WHERE id = ?", (int(staging_id),))
                conn.commit()
            finally:
                conn.close()
        _invalidate_entries_cache()
    except Exception as e:
        print(f"[STAGING] Discard failed: {e}")


@socketio.on("set_segment_denied")
def handle_set_segment_denied(data):
    """Toggle the 'denied' flag on one or more segments.

    Denied segments are hidden from the output display (index.html filters them
    client-side) but kept in the DB and still shown — struck-through — on the
    corrections page so they can be restored. Broadcasts the new state to every
    connected client so open displays update live without a reload.
    """
    if not _socket_auth_ok():
        return
    if not data:
        return

    segment_ids = data.get("segment_ids", [])
    if not segment_ids:
        return
    denied_val = 1 if data.get("denied", True) else 0

    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                placeholders = ",".join("?" for _ in segment_ids)
                conn.execute(
                    f"UPDATE transcriptions SET denied = ? WHERE id IN ({placeholders})",
                    [denied_val, *segment_ids],
                )
                conn.commit()
            finally:
                conn.close()

        _invalidate_entries_cache()

        # Broadcast to all clients (no room) so open index/corrections pages react live
        socketio.emit("segment_denied", {"segment_ids": segment_ids, "denied": bool(denied_val)})
    except Exception as e:
        print(f"[DENY] set_segment_denied failed: {e}")


@socketio.on("set_segment_marked")
def handle_set_segment_marked(data):
    """Toggle the 'marked' bookmark flag on one or more segments.

    Marked segments are a manual operator bookmark set from the corrections
    page so a spot can be found again later (in the DB or the HTML export).
    Independent of needs_review. Broadcasts the new state so every open
    corrections page updates live.
    """
    if not _socket_auth_ok():
        return
    if not data:
        return

    segment_ids = data.get("segment_ids", [])
    if not segment_ids:
        return
    marked_val = 1 if data.get("marked", True) else 0

    try:
        with _db_lock:
            conn = _open_db_writer()
            if conn is None:
                return
            try:
                placeholders = ",".join("?" for _ in segment_ids)
                conn.execute(
                    f"UPDATE transcriptions SET marked = ? WHERE id IN ({placeholders})",
                    [marked_val, *segment_ids],
                )
                conn.commit()
            finally:
                conn.close()

        _invalidate_entries_cache()

        socketio.emit("segment_marked", {"segment_ids": segment_ids, "marked": bool(marked_val)})
    except Exception as e:
        print(f"[MARK] set_segment_marked failed: {e}")


def get_new_entries(limit_override=None):
    """Get recent transcriptions with caching and efficient querying

    Args:
        limit_override: Optional limit to override database.max_entries_to_send config

    The window holds the latest N *visible* rows — denied rows are excluded so they
    can't consume slots and push real lines off the display. Manual deny/restore is
    delivered live via the separate "segment_denied" broadcast, so callers no longer
    need to filter denied entries out of this result (entry[10] is always 0 here).
    """
    global _db_cache

    # Get database name from shared transcription state
    current_db_name = transcription_state.get("db_name")

    # Debug logging (uncomment to trace issues)
    # import sys; print(f"[GET_ENTRIES] db_name={current_db_name}, exists={os.path.exists(current_db_name) if current_db_name else 'N/A'}", file=sys.stderr, flush=True)

    # If database not initialized yet, return empty list
    if current_db_name is None:
        return []

    # Check if database file exists
    if not os.path.exists(current_db_name):
        return []

    current_time = time.time()

    # Check cache first (only use cache if no limit_override, since cache is shared)
    if limit_override is None:
        with _cache_lock:
            if (
                current_time - _db_cache["last_fetch_time"] < _db_cache["cache_duration"]
                and _db_cache["last_entries"]
            ):
                return _db_cache["last_entries"]

    try:
        limit = limit_override if limit_override is not None else config.get("database", {}).get("max_entries_to_send", 100)

        # Use context manager for database connection
        with sqlite3.connect(current_db_name) as conn:
            cursor = conn.cursor()
            if limit <= 0:
                # 0 or negative means no limit — return all visible entries.
                cursor.execute(
                    """
                    SELECT id, timestamp, text, COALESCE(start_time, 0) as start_time, COALESCE(end_time, 0) as end_time,
                           confidence, needs_review, translated_text, translation_language, speech_type,
                           COALESCE(denied, 0), denied_reason, music_prob, COALESCE(marked, 0)
                    FROM transcriptions
                    WHERE timestamp != '' AND TRIM(text) != ''
                    AND COALESCE(is_final, 1) = 1
                    AND COALESCE(denied, 0) = 0
                    ORDER BY id ASC
                """
                )
            else:
                # Window the LATEST N *visible* (non-denied) rows. Denied rows —
                # auto-filters (hallucination/cjk/short/dup/music) that are never
                # shown, plus manual toggles — must not consume window slots, or
                # they push older visible lines out and the display looks like it
                # "advanced to a new screen". They're excluded here; a manual
                # deny/restore is delivered live via the separate "segment_denied"
                # broadcast (handle_set_segment_denied), so hiding still works.
                cursor.execute(
                    """
                    SELECT id, timestamp, text, start_time, end_time, confidence, needs_review, translated_text, translation_language, speech_type, denied, denied_reason, music_prob, marked FROM (
                        SELECT id, timestamp, text, COALESCE(start_time, 0) as start_time, COALESCE(end_time, 0) as end_time,
                               confidence, needs_review, translated_text, translation_language, speech_type,
                               COALESCE(denied, 0) as denied, denied_reason, music_prob, COALESCE(marked, 0) as marked
                        FROM transcriptions
                        WHERE timestamp != '' AND TRIM(text) != ''
                        AND COALESCE(is_final, 1) = 1
                        AND COALESCE(denied, 0) = 0
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                """,
                    (limit,),
                )
            transcriptions = cursor.fetchall()

        # Update cache only if using default limit (not override)
        if limit_override is None:
            with _cache_lock:
                _db_cache["last_entries"] = transcriptions
                _db_cache["last_fetch_time"] = current_time

        return transcriptions
    except Exception as e:
        print(f"[ERROR] Failed to fetch transcriptions: {e}")
        return []


def _attach_segment_ids(segments):
    """Add segment_id (string form of the db id) to each emitted segment so the
    socket key matches the db.segment_id TEXT column exactly. None when there is no
    id (e.g. a not-yet-persisted in_progress segment)."""
    for s in segments:
        if isinstance(s, dict):
            sid = s.get("id")
            s["segment_id"] = str(sid) if sid is not None else None
    return segments


def _live_preview_suppressed(text):
    """Whether the in-progress (live) preview text should be hidden from the
    display. Applies the same criteria that deny finalized rows so the live
    line doesn't loop content the finalized view already filters out:
      - empty/whitespace,
      - a known Whisper hallucination (the "Продолжение следует…"-type stock
        phrases Whisper invents on music/silence), or
      - audio currently detected as Music while music transcription is disabled.
    """
    if not text or not text.strip():
        return True
    if is_whisper_hallucination(text):
        return True
    try:
        if (_ts_get("audio_type") == "Music"
                and not config.get("speech_type_detection", {}).get("transcribe_detected_music", False)):
            return True
    except Exception:
        pass
    return False


_service_phase_last_run = 0.0
_service_phase_state = {"current": None, "blocks": [], "session_id": None}


def _service_phase_session_db():
    """Path of the session database the detector should read, or None."""
    try:
        return _ts_get("db_name")
    except Exception:
        return None


def _service_phase_first_sunday(db_path):
    """Whether this session's date is a first Sunday — communion's usual slot.

    Derived from the session filename (``%Y-%m-%d_%H%M%S.db``) rather than today's date,
    so reviewing an old session doesn't get scored against the calendar it is read on.
    Only ever raises the confidence of a communion label; it never creates one, because
    the exceptions the operator named — Passover, Good Friday — are real and frequent.
    """
    try:
        stamp = os.path.basename(db_path or "")[:10]
        d = datetime.strptime(stamp, "%Y-%m-%d").date()
        return d.weekday() == 6 and d.day <= 7
    except Exception:
        return False


def _service_phase_config():
    return config.get("service_phase", {}) or {}


PHASE_RULES_FILE = os.path.join(CONFIG_DIR, "service_phases.json")
PHASE_RULES_TEMPLATE_FILE = os.path.join(BUNDLE_DIR, "config", "service_phases.default.json")


def _service_phase_rules():
    """The phase-naming rules, re-read each tick so an edit takes effect without a restart.

    Reading a small JSON file every 20 seconds is nothing next to re-deriving the session,
    and the alternative — caching — means an operator who fixes a rule mid-service has to
    restart the server to see it, on the one machine that must not be restarted mid-service.
    """
    try:
        return _phase_rules_load(PHASE_RULES_FILE, PHASE_RULES_TEMPLATE_FILE)
    except Exception:
        return []


def _service_phase_tick(is_running):
    """Re-run phase detection at most once per interval, then broadcast the result.

    Runs in the web process on its own short-lived connection, the same way
    /api/transcription/correct reaches a session database — the long-lived writer belongs
    to the transcription worker. The whole thing is best-effort: a caption must never be
    delayed or lost because a diagnostic could not classify the service.
    """
    global _service_phase_last_run
    cfg = _service_phase_config()
    if not cfg.get("enabled", True) or not is_running:
        return
    interval = coerce_float(cfg.get("interval_seconds"), 20, lo=2, hi=600)
    now = time.time()
    if now - _service_phase_last_run < interval:
        return
    _service_phase_last_run = now

    db_path = _service_phase_session_db()
    if not db_path or not os.path.exists(db_path):
        return
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            result = _service_phase_analyze(
                _service_phase_rows(conn), cfg,
                first_sunday=_service_phase_first_sunday(db_path),
                rules=_service_phase_rules())
            _service_phase_save(conn, result)
        finally:
            conn.close()
    except Exception as e:
        print(f"[SERVICE-PHASE] tick failed ({type(e).__name__}: {e})")
        return

    _service_phase_state["current"] = result.get("current")
    _service_phase_state["blocks"] = result.get("blocks", [])
    _service_phase_state["session_id"] = os.path.basename(db_path)
    try:
        socketio.emit("service_phase_update", {
            "current": result.get("current"),
            "blocks": result.get("blocks", []),
            "spans": result.get("spans", []),
            "session_id": os.path.basename(db_path),
        })
    except Exception:
        pass  # a diagnostic broadcast must never break the emit loop


def emit_new_entries():
    """Emit combined transcription updates and audio levels to web clients"""
    update_interval = config.get("web_server", {}).get("update_interval", 0.5)
    while True:
        if _server_shutting_down.is_set():
            return
        # Check if transcription is running - if not, send empty data to clear display
        is_running = _ts_get("running", False)

        if not is_running:
            # Send empty data when stopped so frontend clears the display
            entries = []
            in_progress = ""
            in_progress_start = 0
            in_progress_end = 0
        else:
            # Get finalized entries from database (now includes id, timestamp, text, start_time, end_time)
            entries = get_new_entries()
            # Get in-progress text (not yet saved to DB)
            in_progress = _ts_get("live_text", "")
            in_progress_start = _ts_get("live_start", 0)
            in_progress_end = _ts_get("live_end", 0)


        # Convert entries to segment format with temporal data
        # entries format: (id, timestamp, text, start_time, end_time, confidence, needs_review, translated_text, translation_language, speech_type, denied, denied_reason, music_prob, marked)
        segments = []
        for entry in entries:
            seg = {
                "id": entry[0],
                "timestamp": entry[1],
                "text": entry[2],
                "start": entry[3],
                "end": entry[4],
                "completed": True,
            }
            # Include confidence data if available (new columns may be None for old DBs)
            if len(entry) > 5:
                seg["confidence"] = entry[5]
                seg["needs_review"] = bool(entry[6]) if entry[6] is not None else False
            if len(entry) > 9:
                seg["speech_type"] = entry[9]
            if len(entry) > 10:
                seg["denied"] = bool(entry[10]) if entry[10] is not None else False
            if len(entry) > 11:
                seg["denied_reason"] = entry[11]
            if len(entry) > 12:
                seg["music_prob"] = entry[12]
            if len(entry) > 13:
                seg["marked"] = bool(entry[13])
            segments.append(seg)

        # Stable key matching db.segment_id (TEXT). Done before the output-delay split
        # below so live + staged segments (same dict refs) both carry it.
        _attach_segment_ids(segments)

        # Build in-progress segment if there's text (suppressing hallucinated /
        # music live text so the root display doesn't loop it during songs)
        in_progress_segment = None
        if not _live_preview_suppressed(in_progress):
            in_progress_segment = {
                "text": in_progress,
                "start": in_progress_start,
                "end": in_progress_end,
                "completed": False,
                "segment_id": None,  # not yet persisted — no db row/segment_id
            }
            # Include word-level confidence for in-progress text
            live_words = _ts_get("live_word_confidences")
            if live_words:
                in_progress_segment["word_confidences"] = list(live_words) if hasattr(live_words, '__iter__') else []

        # Split segments into live and staged based on output delay setting
        delay_config = config.get("corrections", {}).get("output_delay", {})
        delay_enabled = delay_config.get("enabled", False)
        delay_seconds = delay_config.get("delay_seconds", 7)
        staged_segments = []

        if delay_enabled and segments:
            # Rows are stamped with datetime.now(configured_timezone), so the age check
            # has to be taken in that same zone. Comparing a configured-zone string
            # against system-local now() puts every row either instantly live or
            # permanently staged, by exactly the offset between the two.
            now_ts = datetime.now(configured_timezone).replace(tzinfo=None)
            live_segments = []
            for seg in segments:
                # Parse segment timestamp to check age
                try:
                    seg_ts = datetime.strptime(seg["timestamp"], "%Y-%m-%d %H:%M:%S")
                    age = (now_ts - seg_ts).total_seconds()
                    if age < delay_seconds:
                        # Still in delay window — staged
                        seg["staged"] = True
                        seg["delay_remaining"] = max(0, delay_seconds - age)
                        staged_segments.append(seg)
                    else:
                        live_segments.append(seg)
                except (ValueError, KeyError):
                    live_segments.append(seg)
            segments = live_segments

        # Emit single unified transcription update with new format
        emit_data = {
            "segments": segments,  # [{id, timestamp, text, start, end, completed}, ...]
            "in_progress_segment": in_progress_segment,  # {text, start, end, completed} or null
            # Keep backward compatibility with old format
            "entries": [(e[1], e[2]) for e in entries],  # [(timestamp, text), ...]
            "in_progress": in_progress,  # Current incomplete text or ""
            "is_running": is_running,
            "audio_type": _ts_get("audio_type"),  # "Speaking", "Music", or "Quiet"
            "detection_mode": _ts_get("detection_mode"),  # "panns" or "energy"
            "session_id": _ts_get("session_id"),  # stable per-session anchor
        }
        if staged_segments:
            emit_data["staged_segments"] = staged_segments
        emit_data["delay_seconds"] = delay_seconds
        if delay_enabled:
            emit_data["delay_enabled"] = True
        socketio.emit("transcription_update", emit_data)

        # Emit audio level only if transcription is running
        is_running = _ts_get("running", False)
        if is_running:
            audio_level = _ts_get("audio_level")
            audio_db = _ts_get("audio_db")
            audio_energy = _ts_get("audio_energy")

            if audio_level is not None and audio_db is not None:
                try:
                    socketio.emit(
                        "audio_level",
                        {
                            "level": audio_level,
                            "db": audio_db,
                            "energy": audio_energy if audio_energy is not None else 0,
                            "audio_type": _ts_get("audio_type"),
                            "detection_mode": _ts_get("detection_mode"),
                            "audio_tag": _ts_get("audio_tag"),
                            "music_prob": _ts_get("music_prob"),
                        },
                    )
                except Exception as emit_error:
                    print(f"[AUDIO-DEBUG] {time.strftime('%H:%M:%S')} - EMIT FAILED: {emit_error}", flush=True)

        _service_phase_tick(is_running)

        try:
            socketio.sleep(update_interval)  # Emit updates based on config
        except Exception as sleep_error:
            print(f"[AUDIO-DEBUG] {time.strftime('%H:%M:%S')} - SLEEP FAILED: {sleep_error}", flush=True)


def _translation_debug_enabled():
    """Opt-in gate for the [TRANS-DBG] translation-loop trace. Read fresh each
    cycle (config is hot-reloaded), or forced on via STT_TRANSLATION_DEBUG."""
    try:
        if config.get("live_translation", {}).get("debug_logging", False):
            return True
    except Exception:
        pass
    return os.environ.get("STT_TRANSLATION_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _translate_via_remote(text, source_lang, target_lang, endpoint,
                          return_extras=False, num_alternatives=0, generation_params=None,
                          raise_on_error=False):
    """Send text to a remote machine's /api/translate endpoint."""
    try:
        payload = {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "return_extras": return_extras,
            "num_alternatives": num_alternatives,
        }
        if generation_params:
            payload["generation_params"] = generation_params
        _rt_t0 = time.perf_counter()
        resp = _get_remote_http_session().post(
            endpoint.rstrip("/") + "/api/translate",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        # Round-trip latency Machine A experiences for this offloaded translation
        # (network + serialization + Machine B inference). Only successful calls.
        try:
            _record_remote_translate_ms((time.perf_counter() - _rt_t0) * 1000.0)
        except Exception:
            pass
        if return_extras:
            return {
                "text": data.get("translated_text", text),
                "confidence": data.get("confidence"),
                "alternatives": data.get("alternatives", []),
            }
        return data.get("translated_text", text)
    except Exception as e:
        print(f"[REMOTE_TRANSLATE] Failed: {e}")
        if raise_on_error:
            raise _RemoteTranslateError(str(e)) from e
        if return_extras:
            return {"text": text, "confidence": None, "alternatives": []}
        return text


# The prompt template lives in stt/llm_translate.py, where it is parameterised
# by target language and unit-tested.


_local_llm = None
_local_llm_path = ""        # file the resident model was loaded from
_local_llm_failed = False   # a load that failed once; do not retry per caption
_local_llm_lock = threading.Lock()


def local_llm_available():
    """Whether the in-process GGUF runtime is installed.

    Probed rather than imported, and absent is not an error — the same treatment
    panns-inference gets, so an install without the optional dependency degrades to
    the NMT model instead of failing to start.
    """
    try:
        import importlib.util
        return importlib.util.find_spec("llama_cpp") is not None
    except Exception:
        return False


def get_local_llm(llm_cfg_override=None):
    """Singleton in-process GGUF model, or None if unavailable.

    This is the provider that needs no server, no second installer and no extra port:
    one pip dependency and one model file, which is what makes LLM translation
    workable on a fresh Windows/Linux/macOS install where nothing else is present.
    """
    global _local_llm, _local_llm_failed, _local_llm_path
    if not local_llm_available():
        print("[LLM-LOCAL] llama-cpp-python is not installed; "
              "install it or set live_translation.llm.provider to 'endpoint'")
        return None

    llm_cfg = llm_cfg_override if llm_cfg_override is not None else (
        config.get("live_translation", {}).get("llm") or {})
    gguf_repo = (llm_cfg.get("gguf_repo") or "").strip()
    gguf_file = (llm_cfg.get("gguf_file") or "").strip()
    path = (llm_cfg.get("gguf_path") or "").strip()
    if not path and gguf_repo and gguf_file:
        path = _llm_local_model_path(MODELS_DIR, gguf_repo, gguf_file)

    # A different model than the resident one (the settings page testing a fresh
    # choice) means the resident one is the wrong answer — swap rather than
    # silently reporting on a model the operator is no longer asking about.
    if _local_llm is not None and path and path != _local_llm_path:
        unload_local_llm()
    if _local_llm is not None:
        return _local_llm

    if not path or not os.path.isfile(path):
        print(f"[LLM-LOCAL] model file not found ({path or 'unset'}); "
              "download it in the Model Manager")
        _local_llm_failed = True
        return None

    if _local_llm_failed:
        # Remember a failed load. Retrying per caption turned one broken install into a
        # storm of identical multi-second failures, each delaying the caption that then
        # fell back anyway. Cleared by unload_local_llm() so a config fix can retry.
        return None

    with _local_llm_lock:
        if _local_llm is not None:
            return _local_llm
        try:
            # Touch torch.cuda BEFORE importing llama_cpp. The CUDA build links against
            # the CUDA runtime, which torch ships in site-packages/nvidia/ and only puts
            # within reach once its CUDA support initialises. Importing llama_cpp first
            # fails with "libcudart.so.12: cannot open shared object file" even though
            # the library is present. The module-level `torch` is None until the lazy
            # importer has run, and nothing else in this process has necessarily run it.
            _lazy_import_ml_libraries()
            has_gpu = bool(torch.cuda.is_available()
                           or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
            from llama_cpp import Llama
            n_gpu_layers = _llm_resolve_gpu_layers(llm_cfg.get("n_gpu_layers", "auto"), has_gpu)
            print(f"[LLM-LOCAL] loading {os.path.basename(path)} "
                  f"(n_gpu_layers={n_gpu_layers})...", flush=True)
            _local_llm = Llama(
                model_path=path,
                n_ctx=coerce_int(llm_cfg.get("n_ctx"), 2048, lo=_LLM_MIN_N_CTX, hi=32768),
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            _local_llm_path = path
            print(f"[LLM-LOCAL] loaded {os.path.basename(path)}")
        except Exception as e:
            print(f"[LLM-LOCAL] load failed: {e}")
            _local_llm_failed = True
            if "libcudart" in str(e) or "cudart64" in str(e):
                # The CUDA build of llama-cpp-python links against the CUDA runtime,
                # which torch already ships in site-packages/nvidia/. Importing torch
                # first (as this module does long before any translation) normally
                # makes it resolvable; seeing this means it did not.
                print("[LLM-LOCAL] the CUDA runtime was not found. torch bundles it in "
                      "site-packages/nvidia/cuda_runtime/lib — add that to "
                      "LD_LIBRARY_PATH, or reinstall llama-cpp-python without CUDA "
                      "to run on CPU.")
            _local_llm = None
    return _local_llm


def _llm_device_label():
    """Where the in-process GGUF is running: 'metal', 'cuda', 'cpu', or None.

    Reported so a paired machine records the device that actually translated. An
    endpoint provider returns None — the device belongs to the other machine and
    guessing it here would assert something this box cannot know.
    """
    llm_cfg = config.get("live_translation", {}).get("llm") or {}
    if (llm_cfg.get("provider") or "endpoint").strip().lower() != "local":
        return None
    if _local_llm is None:
        return None
    try:
        if _llm_resolve_gpu_layers(llm_cfg.get("n_gpu_layers", "auto"), True) == 0:
            return "cpu"
        _lazy_import_ml_libraries()
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "metal"
    except Exception:
        pass
    return "cpu"


def unload_local_llm():
    """Release the in-process model, and allow a later load to be retried."""
    global _local_llm, _local_llm_failed, _local_llm_path
    with _local_llm_lock:
        _local_llm_failed = False
        _local_llm_path = ""
        if _local_llm is not None:
            _local_llm = None
            import gc
            gc.collect()
            print("[LLM-LOCAL] model unloaded")


def is_local_llm_loaded():
    """Whether the in-process GGUF currently holds weights.

    The counterpart to is_live_translation_model_loaded() for the other engine.
    Callers that free memory must ask both — asking only the NMT one is what let
    the GGUF survive every stop.
    """
    return _local_llm is not None


def _llm_token_counter():
    """The loaded model's own tokenizer as a callable, or None to use the estimate.

    Exact beats the heuristic wherever it can be had: the estimate is deliberately
    pessimistic and would shed context the model actually had room for. Only the local
    provider can offer this — an endpoint's vocabulary lives on another machine.
    """
    llm = _local_llm
    if llm is None:
        return None

    def _count(s):
        return len(llm.tokenize(s.encode("utf-8")))

    return _count


def _llm_budget_for(llm_cfg, system_prompt, max_tokens):
    """User-text token budget for the configured model, and the counter used for it."""
    counter = _llm_token_counter()
    n_ctx = coerce_int(llm_cfg.get("n_ctx"), 2048, lo=_LLM_MIN_N_CTX, hi=32768)
    return _llm_input_budget(n_ctx, max_tokens, system_prompt, counter=counter), counter


def _translate_via_local_llm(text, system_prompt, max_tokens, llm_cfg_override=None):
    """One caption through the in-process GGUF model. Returns raw text or None.

    Timed into the same EMA the NMT paths feed: this is inference on this box, so
    the number means what it means for MADLAD, and the health dashboard would
    otherwise read "no local translations" on a machine doing nothing else. The
    endpoint provider is deliberately not folded in — its time is another
    machine's, and mixing the two would make the figure undiagnosable.
    """
    llm = get_local_llm(llm_cfg_override)
    if llm is None:
        return None
    _t0 = time.perf_counter()
    try:
        out = llm.create_chat_completion(
            messages=_llm_chat_messages(text, system_prompt),
            temperature=0.0,
            max_tokens=max_tokens,
        )
        try:
            _record_local_translate_ms((time.perf_counter() - _t0) * 1000.0)
        except Exception:
            pass  # a metric must never break a caption
        return _llm_extract_text(out)
    except Exception as e:
        print(f"[LLM-LOCAL] generation failed ({type(e).__name__}: {e})")
        return None


# Below this much remaining budget a second attempt cannot finish in time, so the
# caption goes to the NMT model instead of arriving after the speaker has moved on.
_LLM_RETRY_MIN_SECONDS = 1.5

# The smallest context window the shipped prompt can actually work in. It used to be
# 512, which stopped meaning anything once the prompt grew: the system prompt plus the
# output reservation left a budget of zero, so every caption declined for not fitting
# and LLM translation was silently off for anyone who had lowered the setting. A small
# window is meant to cost context, not captions. At 1024 the shipped prompt still
# leaves ~470 tokens, against a p99.9 caption of 40 words.
_LLM_MIN_N_CTX = 1024


def _llm_retry_enabled(llm_cfg):
    """Whether a rejected caption gets one corrective second attempt."""
    value = llm_cfg.get("retry_on_reject", True)
    return value if isinstance(value, bool) else str(value).strip().lower() not in ("0", "false", "no", "off")


def _llm_fallback_is_skip():
    """Whether a declined caption shows the source instead of loading the NMT model.

    Same vocabulary as remote.fallback, deliberately: "skip" shows the original text
    there too, and an operator who has met one of these should not have to learn a
    second word for it.
    """
    llm_cfg = config.get("live_translation", {}).get("llm") or {}
    return (llm_cfg.get("fallback") or "nmt").strip().lower() == "skip"


def _translate_via_llm(text, source_lang, target_lang, timeout_override=None,
                       llm_cfg_override=None, return_raw=False):
    """Translate one caption with an LLM. Returns the caption, or None to fall back.

    None means "use the NMT model instead". An LLM can return its own reasoning, a
    refusal, the source language untouched, or — measured over a real service — a
    scripture reference followed by the recited passage. A wrong caption in front of a
    congregation is worse than a slower one, so anything that fails validation is
    declined here and the caller falls through to the NMT path.

    ``endpoint`` is the full chat URL so one code path serves any OpenAI-compatible
    server: Ollama (``/api/chat``), llama-server, LM Studio, vLLM or a hosted API
    (``/v1/chat/completions``). Nothing here is Ollama-specific.
    """
    llm_cfg = llm_cfg_override if llm_cfg_override is not None else (
        config.get("live_translation", {}).get("llm") or {})
    # The configured target language must reach the prompt, not just the validator:
    # the wrong-script screen looks for Cyrillic, so an English answer to a Spanish
    # request passes it and the session captions the wrong language in silence.
    system_prompt = _llm_system_prompt(
        llm_cfg.get("system_prompt") or _DEFAULT_LLM_SYSTEM_PROMPT,
        target_lang, TRANSLATION_LANGUAGES)
    max_tokens = coerce_int(llm_cfg.get("max_tokens"), 160, lo=16, hi=1024)

    # provider "local" runs the model in-process from a GGUF: no server, no extra
    # installer, no port — which is what makes this workable on a fresh install that
    # has no inference runtime of its own.
    if (llm_cfg.get("provider") or "endpoint").strip().lower() == "local":
        # Too long for the context window is a decline, not an attempt. llama.cpp
        # raises once the prompt exceeds n_ctx, which reaches the caller as a generic
        # failure; measuring first turns that into the same orderly fallback every
        # other rejection takes. Only the local provider is checked — n_ctx sizes the
        # Llama() we construct here, and says nothing about an endpoint model's window.
        # Live captions never come near this (p99.9 is 40 words, and the run-on valve
        # caps a row well before context can stack); it is n_ctx set low, a batch
        # segment, or a paired machine's payload that gets here.
        _budget, _counter = _llm_budget_for(llm_cfg, system_prompt, max_tokens)
        if not _llm_input_fits(text, _budget, counter=_counter):
            print(f"[LLM-TRANSLATE] input exceeds the {_budget}-token context budget "
                  f"({len(text)} chars); using the NMT model")
            return (None, None, "input exceeds context budget") if return_raw else None
        raw = _translate_via_local_llm(text, system_prompt, max_tokens, llm_cfg_override)
        clean, reason = _llm_check(raw, text, target_lang)
        if clean is None:
            retry_prompt = _llm_retry_prompt(system_prompt, reason) if _llm_retry_enabled(llm_cfg) else None
            if retry_prompt is not None:
                print(f"[LLM-TRANSLATE] retrying once ({reason})")
                raw2 = _translate_via_local_llm(text, retry_prompt, max_tokens, llm_cfg_override)
                clean2, reason2 = _llm_check(raw2, text, target_lang)
                if clean2 is not None:
                    return (clean2, raw2, None) if return_raw else clean2
                print(f"[LLM-TRANSLATE] retry also rejected ({reason2}); using the NMT model")
        return (clean, raw, None) if return_raw else clean

    endpoint = (llm_cfg.get("endpoint") or "").strip()
    model = (llm_cfg.get("model") or "").strip()
    if not endpoint or not model:
        return (None, None, None) if return_raw else None

    payload = _llm_chat_payload(
        model, text, system_prompt,
        max_tokens=max_tokens,
        # keep_alive pins the model in the runtime. Not a nicety: an unpinned model
        # measured p90 4.89s against p50 0.29s purely from being unloaded between
        # captions. Servers that don't know the field ignore it.
        keep_alive=llm_cfg.get("keep_alive", -1),
    )
    headers = {"Content-Type": "application/json"}
    if (llm_cfg.get("api_key") or "").strip():
        headers["Authorization"] = f"Bearer {llm_cfg['api_key'].strip()}"
    # A cold model load takes far longer than a caption's budget, so the warm-up call
    # passes its own timeout. Without that the warm-up times out, the NMT fallback
    # loads instead, and it then occupies the VRAM the LLM needed — after which the
    # LLM can never fit, and every caption falls back forever.
    timeout = (timeout_override if timeout_override is not None
               else coerce_float(llm_cfg.get("timeout_ms"), 8000, lo=500, hi=60000) / 1000.0)

    _started_at = time.time()
    try:
        import requests as _req
        resp = _req.post(endpoint, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[LLM-TRANSLATE] call failed ({type(e).__name__}: {e})")
        return (None, None, f"{type(e).__name__}: {e}") if return_raw else None

    raw = _llm_extract_text(data)
    clean, reason = _llm_check(raw, text, target_lang)
    if clean is None:
        # The retry is bounded by whatever the first call left of the caption's budget,
        # never a fresh timeout: a caption that already spent its time is late, and a
        # second full-length attempt would make it later than it is worth showing.
        # Nothing here retries a timeout or a transport error — those return above.
        remaining = timeout - (time.time() - _started_at)
        retry_prompt = _llm_retry_prompt(system_prompt, reason) if _llm_retry_enabled(llm_cfg) else None
        if retry_prompt is not None and remaining >= _LLM_RETRY_MIN_SECONDS:
            print(f"[LLM-TRANSLATE] retrying once ({reason}, {remaining:.1f}s left)")
            payload["messages"] = _llm_chat_messages(text, retry_prompt)
            try:
                resp = _req.post(endpoint, json=payload, headers=headers, timeout=remaining)
                resp.raise_for_status()
                data2 = resp.json()
            except Exception as e:
                print(f"[LLM-TRANSLATE] retry failed ({type(e).__name__}: {e})")
            else:
                raw2 = _llm_extract_text(data2)
                clean2, reason2 = _llm_check(raw2, text, target_lang)
                if clean2 is not None:
                    return (clean2, raw2, data2) if return_raw else clean2
                print(f"[LLM-TRANSLATE] retry also rejected ({reason2}); using the NMT model")
    return (clean, raw, data) if return_raw else clean


# Which engine served the caption this thread just translated. Thread-local rather
# than a return value because the answer is wanted at the row's UPDATE, several call
# shapes away, and threading it through would mean changing every signature between
# here and there for a field almost none of them care about. The contract is narrow:
# translate_live_text sets it on every return path, and the caller reads it
# immediately afterwards, on the same thread, before translating anything else.
_mt_provenance = threading.local()


# What the session recorded as its translating model at start. A row repeats this
# only when it stops being true — see stt.session_meta.row_label_if_changed.
_mt_baseline_label = {"value": ""}


def _set_mt_baseline_label(label):
    """Remember the session's translating model, so rows can record only changes."""
    _mt_baseline_label["value"] = (label or "").strip()


def _record_mt_engine(engine, model=""):
    """Note the engine that produced the translation being returned.

    The engine is stored on every row: it genuinely varies caption to caption, since
    a rejected caption is translated by a different engine than the one beside it,
    and it costs nothing (three characters, absorbed by existing page slack). The
    model name is stored only when it differs from the session's baseline — the same
    string on every row measured 160 KB on one service, for something session_meta
    already holds.
    """
    _mt_provenance.engine = engine
    label = _session_mt_row_label(
        config.get("live_translation", {}), engine,
        remote_status=_remote_effective_status(), model=model)
    _mt_provenance.model = _session_row_label_if_changed(label, _mt_baseline_label["value"])
    return engine


def last_mt_provenance():
    """(engine, model) for this thread's most recent translation, or (None, None)."""
    return getattr(_mt_provenance, "engine", None), getattr(_mt_provenance, "model", None)


def _remote_effective_status():
    """What a paired machine last reported about the model it translates with."""
    model = _remote_effective.get("mt.remote.effective.model")
    return {"model": model} if model else None


def translate_live_text(text, source_lang, target_lang, return_extras=False, num_alternatives=0, generation_params=None, local_only=False):
    """Translate text for live display using the singleton model.

    local_only=True forces local translation and never offloads — used when
    SERVING a paired machine's /api/translate request, so a machine that is both
    an offload client and a translation server doesn't re-offload (chaining)."""
    # Get generation params: explicit param > config fallback
    gen_params = generation_params or config.get("live_translation", {}).get("generation_params", {})

    # Route to remote translation server if configured. Offloading a machine's own
    # display output is honored whenever remote.enabled+endpoint are set, even if
    # this machine also hosts trusted clients — chaining is prevented at the server
    # endpoint (local_only) rather than by globally disabling offload here.
    remote_cfg = config.get("live_translation", {}).get("remote", {})
    if not local_only and remote_cfg.get("enabled") and remote_cfg.get("endpoint"):
        remote_failed = False
        try:
            _remote_ep = _get_remote_endpoint()
        except _RemoteEndpointError as e:
            print(f"[REMOTE_TRANSLATE] Endpoint error: {e}")
            _remote_ep = None
            remote_failed = True
        _dbg = _translation_debug_enabled()
        if _remote_ep:
            if _check_remote_reachable(_remote_ep):
                try:
                    _res = _translate_via_remote(text, source_lang, target_lang, _remote_ep,
                                                 return_extras=return_extras, num_alternatives=num_alternatives,
                                                 generation_params=gen_params, raise_on_error=True)
                    if _dbg:
                        print(f"[TRANS-DBG] remote result=ok ep={_remote_ep} text='{text[:40]}'", flush=True)
                    _record_mt_engine(MT_ENGINE_REMOTE)
                    return _res
                except _RemoteTranslateError as e:
                    print(f"[REMOTE_TRANSLATE] Call failed: {e}")
                    if _dbg:
                        print(f"[TRANS-DBG] remote result=fail ep={_remote_ep} err={e} text='{text[:40]}'", flush=True)
                    remote_failed = True
            else:
                print(f"[REMOTE_TRANSLATE] {_remote_ep} unreachable")
                if _dbg:
                    print(f"[TRANS-DBG] remote result=unreachable ep={_remote_ep} text='{text[:40]}'", flush=True)
                remote_failed = True

        # Remote path failed — apply configured fallback ("skip" preserves the
        # untranslated text as-is; anything else falls through to local translation below)
        if remote_failed and remote_cfg.get("fallback", "skip") == "skip":
            if _dbg:
                print(f"[TRANS-DBG] remote result=fallback-skip (returning source) text='{text[:40]}'", flush=True)
            _record_mt_engine(MT_ENGINE_NONE)
            if return_extras:
                return {"text": text, "confidence": None, "alternatives": []}
            return text

    if not text or not text.strip():
        _record_mt_engine(MT_ENGINE_NONE)
        if return_extras:
            return {"text": "", "confidence": None, "alternatives": []}
        return ""

    # LLM path. Declining falls through to the NMT model below, so the fallback needs
    # no plumbing of its own — which also means translation_model must stay pointed at
    # a real NMT model even when translation_method is "llm", unless the operator has
    # set llm.fallback to "skip".
    if config.get("live_translation", {}).get("translation_method") == "llm":
        _llm_text = _translate_via_llm(text, source_lang, target_lang)
        if _llm_text is not None:
            _record_mt_engine(MT_ENGINE_LLM)
            if return_extras:
                return {"text": _llm_text, "confidence": None, "alternatives": []}
            return _llm_text
        if _llm_fallback_is_skip():
            # No NMT model is ever loaded in this mode, which is the point: on a
            # memory-bound box the NMT weights are several GB held to serve about
            # one caption in a hundred, and that is the room a larger LLM needs.
            # The cost is that those captions show the source text untranslated.
            print(f"[LLM-TRANSLATE] declined; showing the source (llm.fallback=skip) for: '{text[:48]}'")
            _record_mt_engine(MT_ENGINE_NONE)
            if return_extras:
                return {"text": text, "confidence": None, "alternatives": []}
            return text
        print(f"[LLM-TRANSLATE] declined; using the NMT model for: '{text[:48]}'")

    try:
        trans_use_gpu = config.get("live_translation", {}).get("use_gpu", True)
        trans_model_id = _resolve_live_translation_model_id(config.get("live_translation", {}))
        model, tokenizer = get_live_translation_model(trans_use_gpu, model_id=trans_model_id)
        if model is None:
            # The operator chose an option that cannot be honoured. Silence here
            # turns "fall back to local translation" into "skip translation"
            # without anyone being told, so the choice they made is not the
            # behaviour they get — and untranslated captions look identical
            # either way. Say which model was wanted; that is the fix.
            print(f"[LIVE-TRANSLATION] no local model available ({trans_model_id or 'unset'}) — "
                  f"caption left in the source language")
            _record_mt_engine(MT_ENGINE_NONE)
            if return_extras:
                return {"text": text, "confidence": None, "alternatives": []}
            return text

        result = translate_text(
            text, source_lang, target_lang, model, tokenizer,
            return_confidence=return_extras,
            num_alternatives=num_alternatives if return_extras else 0,
            generation_params=gen_params,
        )
        _record_mt_engine(MT_ENGINE_NMT, model=trans_model_id)
        return result
    except Exception as e:
        print(f"[LIVE-TRANSLATION ERROR] {e}")
        _record_mt_engine(MT_ENGINE_NONE)
        if return_extras:
            return {"text": text, "confidence": None, "alternatives": []}
        return text  # Return original on error


def emit_translated_entries():
    """Background task that emits translated transcription updates"""
    update_interval = config.get("web_server", {}).get("update_interval", 0.5)
    _translation_backlog_state = {"active": False}  # Log backlog transitions once, not per cycle
    _dual_config_warned = False  # One-shot warning for offload+trusted-client misconfig
    _dbg_pending_streak = {}  # seg_id -> consecutive cycles stuck in _pending_fresh (loop detector)

    while True:
        if _server_shutting_down.is_set():
            return
        trans_config = config.get("live_translation", {})
        dbg = _translation_debug_enabled()
        _dbg_branches = []  # (seg_id, branch) collected this cycle when dbg on

        # Surface a machine that is BOTH an offload client and a trusted-client
        # host — a leftover trusted client used to silently disable offload and
        # loop the display. Offload now wins for own output, but the leftover is
        # still worth flagging so the operator can unpair it.
        if not _dual_config_warned:
            _rc = trans_config.get("remote", {})
            if _rc.get("enabled") and _rc.get("endpoint") and _trusted_translation_clients:
                print(f"[TRANSLATION] Note: offload is enabled and this machine also has trusted "
                      f"clients {sorted(_trusted_translation_clients)}. Own output offloads to "
                      f"{_rc.get('endpoint')}; unpair stale clients on the Translations page if unintended.", flush=True)
                _dual_config_warned = True
        if not trans_config.get("enabled", False):
            # Translation is off — emit a disabled marker so the translate view
            # can show "Translation disabled" instead of being stuck on "Waiting..."
            socketio.emit("translation_update", {
                "segments": [],
                "in_progress": None,
                "target_language": trans_config.get("target_language", "en"),
                "source_language": trans_config.get("source_language", "auto"),
                "enabled": False,
                "is_running": _ts_get("running", False),
                "model_loaded": is_live_translation_ready(),
                "model_loading": _live_translation_model_loading,
                "session_id": _ts_get("session_id"),
            })
            # Sleep longer when translation is disabled
            socketio.sleep(update_interval * 2)
            continue

        try:
            is_running = _ts_get("running", False)
            if not is_running:
                # Send empty data when stopped
                socketio.emit("translation_update", {
                    "segments": [],
                    "in_progress": None,
                    "target_language": trans_config.get("target_language", "en"),
                    "source_language": trans_config.get("source_language", "auto"),
                    "enabled": True,
                    "is_running": False,
                    "model_loaded": is_live_translation_ready(),
                    "model_loading": _live_translation_model_loading,
                    "session_id": _ts_get("session_id"),
                })
                socketio.sleep(update_interval)
                continue

            target_lang = trans_config.get("target_language", "en")
            source_lang = trans_config.get("source_language", "auto")

            # Resolve "auto" source language to actual language
            if source_lang == "auto":
                source_lang = config.get("audio", {}).get("language", "en")
                if source_lang == "auto":
                    source_lang = "en"  # Default fallback

            # Get finalized entries from database (use translation-specific limit)
            translation_limit = trans_config.get("max_entries_to_send")
            entries = get_new_entries(limit_override=translation_limit)
            cache = get_translation_cache()
            translated_segments = []

            # Check if Whisper-based translation is active (translations already cached by transcription loop)
            _translation_method = trans_config.get("translation_method", "nllb")
            _whisper_translation_active = _translation_method in ("whisper_translate", "whisper_forced_lang")

            # Check if corrections features are enabled for translation confidence
            corrections_cfg = config.get("corrections", {})
            want_confidence = corrections_cfg.get("enabled", True) and corrections_cfg.get("confidence_highlighting", True)
            n_alternatives = corrections_cfg.get("n_best_alternatives", {}).get("translation_count", 3) if corrections_cfg.get("enabled", True) else 0

            # Max 5: beyond that the combined NLLB input approaches the 1024-token truncation
            context_window = max(1, min(5, int(trans_config.get("context_window", 1) or 1)))

            # The LLM has its own, much smaller ceiling: n_ctx. Where NLLB silently
            # truncates a too-long input, llama.cpp raises and the caption drops to the
            # NMT model — losing the engine the operator chose over a prefix that is
            # only there to help. So the prefix is sized to fit and the oldest entries
            # shed, which costs context instead of costing the LLM. Computed once per
            # cycle: the budget is the same for every segment in it.
            _llm_ctx_budget = None
            _llm_ctx_counter = None
            # _uses_local_llm, not just the method: n_ctx belongs to the GGUF this box
            # constructs. An endpoint model's window is the other machine's business.
            if context_window > 1 and _uses_local_llm(trans_config):
                try:
                    _cw_llm_cfg = trans_config.get("llm") or {}
                    _cw_max_tokens = coerce_int(_cw_llm_cfg.get("max_tokens"), 160, lo=16, hi=1024)
                    _cw_prompt = _llm_system_prompt(
                        _cw_llm_cfg.get("system_prompt") or _DEFAULT_LLM_SYSTEM_PROMPT,
                        target_lang, TRANSLATION_LANGUAGES)
                    _llm_ctx_budget, _llm_ctx_counter = _llm_budget_for(
                        _cw_llm_cfg, _cw_prompt, _cw_max_tokens)
                except Exception:
                    pass  # sizing is an optimisation; the backstop in _translate_via_llm still holds
            _llm_ctx_shrunk = False  # log a shrink once per cycle, not once per caption

            max_translations_per_cycle = 3  # Limit new translations per cycle so cached segments emit fast

            # Budget the cycle's fresh translations newest-first (with one slot
            # reserved for the oldest so the tail still clears). FIFO drain would
            # translate the segment a live consumer needs *last* during a backlog.
            _allowed_fresh = set()
            _backlogged = False
            _pending_fresh = []
            if not _whisper_translation_active:
                for _e in entries:
                    if _e[10]:
                        continue
                    if len(_e) > 7 and _e[7]:
                        continue  # translation already stored in DB
                    if cache.get(_e[0], _e[2], target_lang) or cache.get(_e[0], _e[2], target_lang, accept_stale_lang=True):
                        continue
                    _pending_fresh.append(_e[0])
                if len(_pending_fresh) > max_translations_per_cycle:
                    _backlogged = True
                    _allowed_fresh = set(_pending_fresh[-(max_translations_per_cycle - 1):]) | {_pending_fresh[0]}
                else:
                    _allowed_fresh = set(_pending_fresh)
                if _backlogged != _translation_backlog_state["active"]:
                    _translation_backlog_state["active"] = _backlogged
                    if _backlogged:
                        print(f"[TRANSLATION] Backlog: {len(_pending_fresh)} segments pending — draining newest-first, extras paused", flush=True)
                    else:
                        print("[TRANSLATION] Backlog cleared", flush=True)

            if dbg:
                # Loop detector: track how many consecutive cycles each segment
                # stays "fresh-pending" (never persisting). A segment stuck here
                # is the repeating-phrase signature.
                _pf = set(_pending_fresh)
                for _sid in list(_dbg_pending_streak):
                    if _sid not in _pf:
                        del _dbg_pending_streak[_sid]
                for _sid in _pf:
                    _dbg_pending_streak[_sid] = _dbg_pending_streak.get(_sid, 0) + 1
                _rc = trans_config.get("remote", {})
                print(f"[TRANS-DBG] cycle entries={len(entries)} pending_fresh={_pending_fresh} "
                      f"allowed={sorted(_allowed_fresh)} backlog={_backlogged} "
                      f"remote_enabled={bool(_rc.get('enabled') and _rc.get('endpoint'))} "
                      f"ready={is_live_translation_ready()} trusted={len(_trusted_translation_clients)} "
                      f"target={target_lang} whisper={_whisper_translation_active}", flush=True)
                for _sid, _k in _dbg_pending_streak.items():
                    if _k >= 3:
                        _txt = next((e[2] for e in entries if e[0] == _sid), "")
                        print(f"[TRANS-DBG] LOOP-SUSPECT seg {_sid} pending {_k} cycles — never persisting; text='{_txt[:60]}'", flush=True)

            # While backlogged, skip confidence/alternatives on fresh translations
            # to raise drain throughput (cached extras still display normally)
            _want_conf_cycle = want_confidence and not _backlogged
            _n_alt_cycle = 0 if _backlogged else n_alternatives
            for idx, entry in enumerate(e for e in entries if not e[10]):
                seg_id = entry[0]
                original_text = entry[2]

                # Whisper-based translation: translations saved to DB by subprocess
                # (subprocess cache is in separate memory, so read from DB instead)
                if _whisper_translation_active:
                    cached = cache.get(seg_id, "", target_lang)
                    if not cached:
                        cached = cache.get(seg_id, "", target_lang, accept_stale_lang=True)
                    if not cached and len(entry) > 7 and entry[7]:
                        # Read translation from DB (written by subprocess)
                        cached = entry[7]
                        cache.set(seg_id, "", cached, entry[8] or target_lang)
                    translated_text = cached if cached else original_text
                    extras = None
                    seg_data = {
                        "id": seg_id,
                        "timestamp": entry[1],
                        "original_text": original_text,
                        "translated_text": translated_text,
                        "start": entry[3],
                        "end": entry[4],
                        "completed": True,
                    }
                    if not is_whisper_hallucination(translated_text):
                        translated_segments.append(seg_data)
                    continue

                # Check cache first (exact language match)
                cached = cache.get(seg_id, original_text, target_lang)
                if cached:
                    translated_text = cached
                    # Get cached extras (confidence, alternatives)
                    extras = cache.get_extras(seg_id) if want_confidence else None
                    if dbg:
                        _dbg_branches.append((seg_id, "cache"))
                else:
                    # After a hot language switch, keep old translations for already-translated segments
                    # instead of retranslating everything — only new segments get the new language
                    stale_cached = cache.get(seg_id, original_text, target_lang, accept_stale_lang=True)
                    if stale_cached:
                        translated_text = stale_cached
                        extras = cache.get_extras(seg_id) if want_confidence else None
                        seg_data = {
                            "id": seg_id,
                            "timestamp": entry[1],
                            "original_text": original_text,
                            "translated_text": translated_text,
                            "start": entry[3],
                            "end": entry[4],
                            "completed": True,
                        }
                        if extras:
                            seg_data["confidence"] = extras.get("confidence")
                            seg_data["alternatives"] = extras.get("alternatives", [])
                        if not is_whisper_hallucination(translated_text):
                            translated_segments.append(seg_data)
                        if dbg:
                            _dbg_branches.append((seg_id, "stale"))
                        continue
                    # Cache cold (e.g. server restart): seed from DB if it has any translation
                    # and skip live retranslation, same as stale-lang cache hit.
                    if len(entry) > 7 and entry[7]:
                        db_translation = entry[7]
                        db_lang = entry[8] if len(entry) > 8 and entry[8] else target_lang
                        cache.set(seg_id, original_text, db_translation, db_lang)
                        if not is_whisper_hallucination(db_translation):
                            translated_segments.append({
                                "id": seg_id,
                                "timestamp": entry[1],
                                "original_text": original_text,
                                "translated_text": db_translation,
                                "start": entry[3],
                                "end": entry[4],
                                "completed": True,
                            })
                        if dbg:
                            _dbg_branches.append((seg_id, "db_seed"))
                        continue
                    # Over this cycle's translation budget — a later cycle picks it
                    # up (newest-first); skip emission until it's translated
                    if seg_id not in _allowed_fresh:
                        if dbg:
                            _dbg_branches.append((seg_id, "over_budget"))
                        continue

                    # Build context from preceding segments if context_window > 1.
                    # The combined (context + target) text is translated in one call, then the
                    # target's portion is extracted by sentence-count alignment. If alignment
                    # fails (translator merged sentences), fall back to translating the target
                    # alone - never emit the combined translation.
                    text_to_translate = original_text
                    num_ctx_sentences = 0
                    ctx_char_ratio = None
                    if context_window > 1 and idx > 0:
                        ctx_start = max(0, idx - (context_window - 1))
                        context_texts = [entries[j][2] for j in range(ctx_start, idx)]
                        if context_texts and _llm_ctx_budget is not None:
                            _fitted = _llm_fit_context(
                                context_texts, original_text, _llm_ctx_budget,
                                counter=_llm_ctx_counter)
                            if len(_fitted) != len(context_texts):
                                if not _llm_ctx_shrunk:
                                    print(f"[LLM-TRANSLATE] context trimmed to "
                                          f"{len(_fitted)}/{len(context_texts)} segments "
                                          f"to fit the {_llm_ctx_budget}-token budget")
                                    _llm_ctx_shrunk = True
                                context_texts = _fitted
                        if context_texts:
                            context_prefix = " ".join(context_texts)
                            num_ctx_sentences = count_sentence_units(context_prefix)
                            text_to_translate = context_prefix + " " + original_text
                            # Context share of the source — guides the proportional
                            # split when the translator merges sentences
                            ctx_char_ratio = (len(context_prefix) + 1) / max(1, len(text_to_translate))

                    # Translate with confidence/alternatives if corrections enabled
                    if _want_conf_cycle or _n_alt_cycle > 0:
                        result = translate_live_text(
                            text_to_translate, source_lang, target_lang,
                            return_extras=True, num_alternatives=_n_alt_cycle,
                        )
                        if num_ctx_sentences:
                            extracted = extract_context_translation(result.get("text", ""), num_ctx_sentences, ctx_char_ratio)
                            if extracted:
                                result["text"] = extracted
                                result["alternatives"] = [
                                    alt_extracted for alt_extracted in (
                                        extract_context_translation(a, num_ctx_sentences, ctx_char_ratio)
                                        for a in result.get("alternatives", [])
                                    ) if alt_extracted
                                ]
                            else:
                                # Alignment failed - retranslate without context
                                result = translate_live_text(
                                    original_text, source_lang, target_lang,
                                    return_extras=True, num_alternatives=_n_alt_cycle,
                                )
                        translated_text = result["text"]
                        extras = {"confidence": result.get("confidence"), "alternatives": result.get("alternatives", [])}
                    else:
                        translated_text = translate_live_text(text_to_translate, source_lang, target_lang)
                        if num_ctx_sentences and isinstance(translated_text, str):
                            extracted = extract_context_translation(translated_text, num_ctx_sentences, ctx_char_ratio)
                            translated_text = extracted if extracted else translate_live_text(original_text, source_lang, target_lang)
                        extras = None

                    # Warmup guard: while the local model is still loading, translate_live_text
                    # returns the source unchanged. Don't cache/persist that echo — leave the row
                    # NULL so it retries next cycle and translates correctly once the model is up.
                    if not is_live_translation_ready():
                        if dbg:
                            _dbg_branches.append((seg_id, "warmup_skip"))
                        continue
                    # Read before anything else translates on this thread: the record
                    # is the last call's, and the in-progress line below makes one.
                    _mt_engine, _mt_model = last_mt_provenance()

                    if extras is not None:
                        cache.set_with_extras(seg_id, original_text, translated_text, target_lang,
                                              confidence=extras.get("confidence"), alternatives=extras.get("alternatives", []))
                    else:
                        cache.set(seg_id, original_text, translated_text, target_lang)

                    # Save translation to database
                    try:
                        current_db = _ts_get("db_name")
                        if current_db and os.path.exists(current_db):
                            # timeout/busy_timeout match every other short-lived writer here
                            # (e.g. /api/transcription/correct). The transcription worker holds
                            # the session's long-lived connection, so a partial snapshot or a
                            # finalize batch can be mid-write when this lands; on the default
                            # 5s this raised "database is locked" instead of waiting, and a lost
                            # write leaves the row NULL to be retried every cycle thereafter.
                            with sqlite3.connect(current_db, timeout=30.0) as _tconn:
                                _tconn.execute("PRAGMA busy_timeout=30000")
                                _tconn.execute(
                                    "UPDATE transcriptions SET translated_text = ?, translation_language = ?,"
                                    " translation_ts_ms = ?, mt_engine = ?, mt_model = ? WHERE id = ?",
                                    (translated_text, target_lang, int(time.time() * 1000),
                                     _mt_engine, _mt_model, seg_id),
                                )
                                _tconn.commit()
                                if dbg:
                                    _row = _tconn.execute("SELECT translated_text FROM transcriptions WHERE id = ?", (seg_id,)).fetchone()
                                    _reread = _row[0] if _row else "<no row>"
                                    _dbg_branches.append((seg_id, "fresh(persist=ok)"))
                                    if not _reread:
                                        print(f"[TRANS-DBG] seg {seg_id} committed but re-read translated_text={_reread!r} (persist not sticking)", flush=True)
                        elif dbg:
                            _dbg_branches.append((seg_id, "fresh(persist=no-db)"))
                    except Exception as e:
                        # Translation still shows from cache, but won't survive a
                        # page reload — surface the reason instead of hiding it
                        print(f"[TRANSLATION] DB save failed for segment {seg_id}: {e}", flush=True)
                        if dbg:
                            _dbg_branches.append((seg_id, f"fresh(persist=FAIL:{e})"))

                # Skip known Whisper hallucinations in translated text
                if is_whisper_hallucination(translated_text):
                    print(f"[SKIP HALLUCINATION-TRANSLATION] '{translated_text[:40]}'", flush=True)
                    continue

                seg_data = {
                    "id": seg_id,
                    "timestamp": entry[1],
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "start": entry[3],
                    "end": entry[4],
                    "completed": True,
                }
                if extras:
                    seg_data["confidence"] = extras.get("confidence")
                    seg_data["alternatives"] = extras.get("alternatives", [])
                translated_segments.append(seg_data)

            # Always send in-progress text; is_translated tells the frontend whether
            # it's in the target language (so it can suppress source-language flash).
            in_progress_translation = None
            in_progress = _ts_get("live_text", "")
            # Suppress on the SOURCE text — a source-language hallucination that
            # gets translated first would otherwise slip past a check on the
            # translated string. Also covers the music-detected gate.
            if not _live_preview_suppressed(in_progress):
                should_translate_ip = trans_config.get("translate_in_progress", False) and not _whisper_translation_active
                if should_translate_ip:
                    translated_in_progress = translate_live_text(in_progress, source_lang, target_lang)
                else:
                    translated_in_progress = in_progress  # untranslated; frontend filters by is_translated
                if not is_whisper_hallucination(translated_in_progress):
                    in_progress_translation = {
                        "original_text": in_progress,
                        "translated_text": translated_in_progress,
                        "is_translated": should_translate_ip,
                        "start": _ts_get("live_start", 0),
                        "end": _ts_get("live_end", 0),
                        "completed": False,
                        "segment_id": None,  # not yet persisted — no db row/segment_id
                    }

            # Tag denied state per segment (so the output can hide denied rows and the
            # corrections page can show them struck-through). Built from the DB rows above.
            _denied_by_id = {
                e[0]: (bool(e[10]) if len(e) > 10 and e[10] is not None else False)
                for e in entries
            }
            for _seg in translated_segments:
                _seg["denied"] = _denied_by_id.get(_seg["id"], False)
            _attach_segment_ids(translated_segments)

            if dbg:
                _brs = " ".join(f"{_sid}={_b}" for _sid, _b in _dbg_branches) or "(none)"
                print(f"[TRANS-DBG] branches: {_brs}", flush=True)
                print(f"[TRANS-DBG] emit ids={[s['id'] for s in translated_segments]} "
                      f"count={len(translated_segments)} in_progress={in_progress_translation is not None}", flush=True)

            # Emit translation update
            socketio.emit("translation_update", {
                "segments": translated_segments,
                "in_progress": in_progress_translation,
                "target_language": target_lang,
                "target_language_name": TRANSLATION_LANGUAGES.get(target_lang, target_lang),
                "source_language": source_lang,
                "enabled": True,
                "is_running": is_running,
                "model_loaded": is_live_translation_ready(),
                "model_loading": _live_translation_model_loading,
                "session_id": _ts_get("session_id"),
            })

        except (BrokenPipeError, EOFError, ConnectionError):
            # Manager proxy gone — server restarting/shutting down. Exit quietly.
            _server_shutting_down.set()
            return
        except Exception as e:
            print(f"[LIVE-TRANSLATION EMIT ERROR] {e}")
            import traceback
            traceback.print_exc()

        socketio.sleep(update_interval)


# =============================================================================
# Audio Streaming Background Tasks
# =============================================================================

def emit_audio_stream():
    """Background task that streams live audio chunks to web clients"""
    while True:
        try:
            data = audio_stream_queue.get(timeout=0.5)
            socketio.emit("audio_chunk", data, room="audio_stream")
        except Empty:
            pass  # No audio queued within the timeout window
        except Exception as e:
            print(f"[AUDIO-STREAM] emit error: {e}", flush=True)


_tts_last_spoken_id = 0

def emit_tts_audio():
    """Background task that synthesizes speech from translated text and emits audio.
    Buffers segments until a sentence-ending punctuation is found so TTS speaks
    complete phrases rather than tiny fragments."""
    global _tts_last_spoken_id
    import base64

    _tts_buffer = []  # Buffered segments waiting for sentence end
    _tts_buffer_last_update = 0  # Timestamp of last buffer addition
    _tts_was_off = True  # Start as off so first enable skips existing segments

    while True:
        if _server_shutting_down.is_set():
            return
        tts_config = config.get("live_translation", {}).get("tts", {})
        trans_config = config.get("live_translation", {})

        if not tts_config.get("enabled", False) or not trans_config.get("enabled", False):
            _tts_buffer.clear()
            # Mark that TTS is off so we can skip existing segments when re-enabled
            _tts_was_off = True
            socketio.sleep(1)
            continue

        # When TTS is first enabled mid-session, skip to the latest segment
        # so we don't replay everything from the beginning
        if _tts_was_off:
            _tts_was_off = False
            max_id = get_translation_cache().max_segment_id()
            if max_id > _tts_last_spoken_id:
                _tts_last_spoken_id = max_id

        if not _ts_get("running", False):
            _tts_last_spoken_id = 0
            _tts_buffer.clear()
            socketio.sleep(1)
            continue

        try:
            target_lang = trans_config.get("target_language", "en")
            cache = get_translation_cache()

            # Get segments that have been translated but not yet spoken
            new_segments = []
            with cache._lock:
                for seg_id, entry in cache._cache.items():
                    if isinstance(seg_id, int) and seg_id > _tts_last_spoken_id:
                        translated = entry.get("translated", "")
                        if translated and translated.strip():
                            new_segments.append({
                                "id": seg_id,
                                "translated_text": translated,
                            })

            # Sort by ID to speak in order
            new_segments.sort(key=lambda s: s["id"])

            # Add new segments to buffer
            for segment in new_segments:
                _tts_buffer.append(segment)
                _tts_last_spoken_id = segment["id"]
                _tts_buffer_last_update = time.time()

            # Check if buffer has a complete phrase to speak
            # Speak when: buffer ends with sentence punctuation, or buffer has been waiting too long (flush timeout)
            if _tts_buffer:
                combined_text = " ".join(s["translated_text"] for s in _tts_buffer).strip()
                last_char = combined_text.rstrip()[-1] if combined_text.rstrip() else ""
                sentence_complete = last_char in ".!?;:。！？"
                flush_timeout = time.time() - _tts_buffer_last_update > 4.0  # Flush after 4s of no new segments

                if sentence_complete or flush_timeout:
                    # Check if TTS is still enabled
                    if not config.get("live_translation", {}).get("tts", {}).get("enabled", False):
                        _tts_buffer.clear()
                        continue

                    try:
                        audio_bytes, sample_rate = synthesize_tts(combined_text, language=target_lang)
                    except Exception as synth_err:
                        # Drop the buffered text: retrying the same input every
                        # cycle would loop forever on a persistent failure
                        print(f"[TTS] Synthesis failed, dropping buffered text: {synth_err}")
                        _tts_buffer.clear()
                        continue
                    if audio_bytes:
                        backend = _get_tts_backend()
                        audio_format = "mp3" if backend == "edge" else "wav"
                        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
                        socketio.emit("tts_audio", {
                            "segment_id": _tts_buffer[-1]["id"],
                            "audio": audio_b64,
                            "format": audio_format,
                            "sample_rate": sample_rate,
                            "text": combined_text,
                        }, room="tts_audio")

                    _tts_buffer.clear()

        except (BrokenPipeError, EOFError, ConnectionError):
            # Manager proxy gone — server restarting/shutting down. Exit quietly.
            _server_shutting_down.set()
            return
        except Exception as e:
            print(f"[TTS EMIT ERROR] {e}")

        socketio.sleep(0.5)


# Text-processing helpers live in stt/text_utils.py (importable, unit-tested);
# names are re-imported here so the pipeline call sites below stay unchanged.
from stt import text_utils as _text_utils
from stt.text_utils import (  # noqa: F401
    DEFAULT_WHISPER_HALLUCINATIONS,
    count_sentence_units,
    distribute_whisper_translation,
    extract_context_translation,
    filter_hallucinated_text,
    is_fuzzy_duplicate,
    normalize_for_hallucination_check,
    remove_overlapping_prefix,
    scope_whisper_translation,
    split_into_sentences,
)


def get_hallucination_phrases():
    """Get hallucination phrases from config, falling back to defaults."""
    return _text_utils.get_hallucination_phrases(config.get("hallucination_filter", {}))


def is_whisper_hallucination(text):
    """Check if text is a known Whisper hallucination (exact phrase match, case/punctuation insensitive)."""
    return _text_utils.is_whisper_hallucination(text, get_hallucination_phrases())


def apply_profanity_filter(text):
    """Replace broadcast-forbidden words with **** (or configured replacement)."""
    return _text_utils.apply_profanity_filter(text, config.get("profanity_filter", {}))


class _TimestampedStream:
    """Wraps a text stream so each output line is prefixed with a local time.
    Turns the app's bare '[TAG] msg' prints into greppable-by-time log lines
    without touching the ~500 print() call sites. Buffers across writes so a
    prefix only lands at a true line start."""

    def __init__(self, wrapped):
        self._w = wrapped
        self._at_line_start = True
        self.stt_timestamped = True  # marker so we never double-wrap

    def write(self, s):
        if not s:
            return 0
        n = len(s)
        # Some writers (Flask/click banner) push bytes at stdout; decode so the
        # str-based prefixing works, then let the underlying stream handle it.
        if isinstance(s, (bytes, bytearray)):
            enc = getattr(self._w, "encoding", None) or "utf-8"
            s = bytes(s).decode(enc, "replace")
        ts = time.strftime("[%H:%M:%S] ")
        prefix = ts if self._at_line_start else ""
        self._at_line_start = s.endswith("\n")
        body = s.replace("\n", "\n" + ts)
        if self._at_line_start and body.endswith(ts):
            body = body[: -len(ts)]  # don't prefix the not-yet-written next line
        try:
            self._w.write(prefix + body)
        except Exception:
            pass
        return n

    def flush(self):
        try:
            self._w.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._w, name)


_diag_installed = False
_faulthandler_fh = None  # kept open for the whole process so native crashes dump


def install_crash_diagnostics(role="main"):
    """Per-process observability: native-crash capture, uncaught-exception
    logging, and timestamped stdout/stderr. Idempotent and best-effort; safe to
    call in every process (fork children inherit an already-installed setup)."""
    global _diag_installed, _faulthandler_fh
    if _diag_installed:
        return
    _diag_installed = True

    logs_dir = os.path.join(APP_DIR, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except OSError:
        pass

    # Native crashes (CUDA/torch/native segfaults, aborts) -> C stack trace.
    # A dedicated kept-open handle guarantees capture even if stdout/stderr are
    # redirected or None (GUI builds).
    try:
        import faulthandler
        _faulthandler_fh = open(os.path.join(logs_dir, "faulthandler.log"),
                                "a", buffering=1, encoding="utf-8", errors="replace")
        faulthandler.enable(file=_faulthandler_fh, all_threads=True)
    except Exception:
        try:
            import faulthandler
            faulthandler.enable()  # fall back to stderr
        except Exception:
            pass

    # Timestamp stdout/stderr (covers every [TAG] print).
    for _name in ("stdout", "stderr"):
        _s = getattr(sys, _name, None)
        if _s is not None and not getattr(_s, "stt_timestamped", False):
            try:
                setattr(sys, _name, _TimestampedStream(_s))
            except Exception:
                pass

    # Uncaught exceptions (main + worker threads) -> full traceback before exit.
    import traceback as _tb

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        try:
            print(f"[FATAL] Uncaught exception in {role} process:", flush=True)
            _tb.print_exception(exc_type, exc, tb)
            sys.stderr.flush()
        except Exception:
            pass

    sys.excepthook = _excepthook

    try:
        def _thread_hook(args):
            if issubclass(args.exc_type, KeyboardInterrupt):
                return
            try:
                print(f"[FATAL] Uncaught exception in thread {args.thread.name}:", flush=True)
                _tb.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
                sys.stderr.flush()
            except Exception:
                pass
        threading.excepthook = _thread_hook
    except Exception:
        pass


def thread1_function(ts, cq, cfq, cal_state, cal_data, cal_step1, asq):
    """Main transcription process with start/stop support"""
    install_crash_diagnostics("worker")
    try:
        import sentry_sdk
        sentry_sdk.set_tag("process", "worker")  # init ran at module import if a DSN is configured
    except ImportError:
        pass
    # On macOS/Windows (spawn), shared state objects are passed as arguments because the spawned
    # child re-imports this module and cannot recreate them. Assign to module globals so
    # all existing code in this function works unchanged. On Linux (fork), the globals are
    # already set via fork, and these reassignments are a no-op.
    global transcription_state, control_queue, config_queue, audio_stream_queue
    global calibration_state, calibration_data_shared, calibration_step1_data
    transcription_state = ts
    control_queue = cq
    config_queue = cfq
    calibration_state = cal_state
    calibration_data_shared = cal_data
    calibration_step1_data = cal_step1
    audio_stream_queue = asq

    # Make every file/dir this subprocess creates readable by all users (a+r files,
    # a+rx dirs): the session DB, its WAL/SHM sidecars, SRT/HTML exports, audio
    # backups, and file-mover output — including everything written during stop
    # cleanup. This is a separate process from the web server, so config writes
    # (which may hold secrets) are unaffected.
    try:
        os.umask(0o022)
    except Exception:
        pass

    # Initialize state variables
    is_running = False
    audio_model = None
    processor = None
    model_type = None
    vad_model = None
    source = None
    recorder = None
    persistent_db_conn = None
    stop_listening = None
    session_audio_file = None
    session_audio_written = False
    calibration_mode = False
    calibration_data = None

    try:
        while True:
            # Check for control commands
            try:
                command = None
                try:
                    command = control_queue.get_nowait()
                except Empty:
                    pass  # No pending control command
                if command is not None:
                    if command["command"] == "start" and not is_running:
                        is_running = True
                        print("[WORKER] Starting transcription...", flush=True)
                    elif command["command"] == "start_calibration":
                        # Handle calibration start command - use local state
                        calibration_mode = True
                        calibration_data = {
                            "start_time": time.time(),
                            "duration": command.get("duration", 30),
                            "noise_samples": [],
                            "speech_samples": [],
                            "silence_durations": [],
                            "energy_levels": [],
                            "vad_probabilities": [],
                        }
                        print(f"[CALIBRATION-PROCESS] Calibration mode enabled in transcription process - duration: {calibration_data['duration']}s", flush=True)
                    elif command["command"] == "stop" and is_running:
                        print("[STOP] Stop command received, signaling main loop to exit...")
                        is_running = False  # Signal the main loop to stop
                        transcription_state["live_text"] = ""  # Clear live preview
                        # Abort any in-flight calibration; the worker survives
                        # stop/start, so stale flags would report "calibrating" forever
                        if calibration_mode or calibration_state.get("active"):
                            print("[STOP] Aborting in-flight calibration")
                            calibration_mode = False
                            calibration_state["active"] = False
                        print("[STOP] is_running set to False - main() will exit and cleanup")

                        # Stop audio source to unblock the main loop if it's waiting for audio
                        if source:
                            try:
                                print("[STOP] Stopping audio source to unblock main loop...")
                                if hasattr(source, "stop") and callable(source.stop):
                                    source.stop()
                                    print("[STOP] OK: Audio source stopped")
                                # Fallback: kill only THIS source's lingering ffmpeg.
                                # Scope the kill to our own process — never a
                                # system-wide "taskkill /IM ffmpeg.exe", which
                                # would also kill unrelated ffmpeg jobs (e.g. a
                                # concurrent file transcription).
                                import subprocess as sp
                                our_proc = getattr(source, "process", None)
                                our_pid = getattr(our_proc, "pid", None) if our_proc else None
                                try:
                                    if platform.startswith('win'):
                                        if our_pid:
                                            sp.run(["taskkill", "/F", "/T", "/PID", str(our_pid)],
                                                   capture_output=True, timeout=2)
                                            print(f"[STOP] OK: Killed ffmpeg PID {our_pid}")
                                        else:
                                            print("[STOP] No ffmpeg PID tracked; skipping kill to avoid taking down unrelated ffmpeg")
                                    elif getattr(source, "device_name", None):
                                        sp.run(["pkill", "-9", "-f", f"ffmpeg.*{re.escape(source.device_name)}"],
                                               capture_output=True, timeout=2)
                                        print(f"[STOP] OK: Sent kill for ffmpeg using {source.device_name}")
                                except Exception as pkill_err:
                                    print(f"[STOP] kill fallback failed: {pkill_err}")
                            except Exception as e:
                                print(f"[STOP] WARNING: Error stopping audio source: {e}")

                        # Let main() handle all the cleanup when it exits
                        print("[STOP] Waiting for main() to exit and cleanup...")
                        continue
                    elif command["command"] == "unload":
                        # Explicit unload command to release GPU memory without killing the process
                        print("[UNLOAD] Unload command received, cleaning up models...")
                        import gc
                        # Unload models to release GPU memory
                        if audio_model is not None:
                            try:
                                del audio_model
                            except (NameError, AttributeError):
                                pass
                            audio_model = None
                        if processor is not None:
                            try:
                                del processor
                            except (NameError, AttributeError):
                                pass
                            processor = None
                        if vad_model is not None:
                            try:
                                del vad_model
                            except (NameError, AttributeError):
                                pass
                            vad_model = None
                        model_type = None
                        # Release the PANNs audio tagger too
                        try:
                            unload_audio_tagger()
                        except Exception:
                            pass
                        # Clean up model cache
                        try:
                            ModelFactory.cleanup_models()
                        except Exception as e:
                            print(f"[UNLOAD] Warning: ModelFactory cleanup error: {e}")
                        gc.collect()
                        _empty_device_cache()
                        print("[UNLOAD] Models unloaded, GPU memory released")
            except Exception as e:
                print(f"[WARNING] Error processing control command: {e}", flush=True)
                # Continue processing even if queue operations fail

            # If not running, just wait
            if not is_running:
                sleep(0.3)
                continue

            # If running but not initialized, initialize now
            if is_running and audio_model is None:
                def main():
                    # Declare nonlocal variables to update outer scope
                    nonlocal \
                        is_running, \
                        audio_model, \
                        processor, \
                        model_type, \
                        vad_model, \
                        source, \
                        recorder, \
                        persistent_db_conn, \
                        stop_listening, \
                        session_audio_file, \
                        session_audio_written, \
                        calibration_mode, \
                        calibration_data

                    # Load fresh config for this process
                    process_config = load_config()
                    # In whisper_translate modes the translation is a second decode by
                    # the ASR model itself, so the model that produced a row's
                    # translated_text is the one that produced its text.
                    #
                    # _asr_baseline_label is what session_meta recorded at session start
                    # and never changes; _asr_session_label tracks the model actually in
                    # use, so a hot reload can be spotted by comparing the two.
                    _asr_baseline_label = _session_asr_row_label(process_config)
                    _asr_session_label = _asr_baseline_label
                    _whisper_mt_model = _asr_baseline_label
                    # Get defaults from config file
                    # Note: model config is at model.whisper, not whisper
                    whisper_config = process_config.get("model", {}).get("whisper", {})
                    audio_config = process_config.get("audio", {})
                    vad_config = process_config.get("vad", {})

                    parser = argparse.ArgumentParser(
                        description="Real-time Speech-to-Text with Whisper",
                        epilog="Note: Command-line arguments override config.json settings",
                    )
                    parser.add_argument(
                        "--model",
                        default=whisper_config.get("model", "tiny"),
                        help=f"Model to use (default from config: {whisper_config.get('model', 'tiny')})",
                        choices=[
                            "tiny",
                            "base",
                            "small",
                            "medium",
                            "large",
                            "large-v1",
                            "large-v2",
                        ],
                    )
                    parser.add_argument(
                        "--energy_threshold",
                        default=audio_config.get("energy_threshold", 3500),
                        help=f"Energy level for mic to detect (default from config: {audio_config.get('energy_threshold', 3500)})",
                        type=int,
                    )
                    parser.add_argument(
                        "--record_timeout",
                        default=audio_config.get("record_timeout", 3),
                        help=f"How real time the recording is in seconds (default from config: {audio_config.get('record_timeout', 3)})",
                        type=float,
                    )
                    parser.add_argument(
                        "--phrase_timeout",
                        default=audio_config.get("phrase_timeout", 2),
                        help=f"Silence duration before new line (default from config: {audio_config.get('phrase_timeout', 2)}s)",
                        type=float,
                    )
                    parser.add_argument(
                        "--use_vad",
                        default=vad_config.get("enabled", True),
                        action="store_true",
                        help=f"Enable Voice Activity Detection (default from config: {vad_config.get('enabled', True)})",
                    )
                    parser.add_argument(
                        "--disable_vad",
                        dest="use_vad",
                        action="store_false",
                        help="Disable Voice Activity Detection (use only energy-based detection)",
                    )
                    parser.add_argument(
                        "--vad_threshold",
                        default=vad_config.get("threshold", 0.5),
                        help=f"VAD confidence threshold 0.0-1.0 (default from config: {vad_config.get('threshold', 0.5)}). "
                        "Examples: 0.3=sensitive, 0.5=balanced, 0.7=strict, 0.9=very strict",
                        type=float,
                    )
                    parser.add_argument(
                        "--config",
                        default=CONFIG_FILE,
                        help=f"Path to config file (default: {CONFIG_FILE})",
                    )
                    parser.add_argument(
                        "--default_microphone",
                        default=audio_config.get("default_microphone", "default"),
                        help=f"Default microphone for ffmpeg (default from config: {audio_config.get('default_microphone', 'default')}). "
                        "Linux: 'default', 'plughw:0,0', etc. macOS: ':0', ':1', or device name. Run with 'list' to view available devices",
                        type=str,
                    )
                    # Use empty args list to prevent inheriting CLI args from parent process
                    args = parser.parse_args([])

                    # The last time a recording was retreived from the queue.
                    phrase_time = None

                    # Initialize WhisperLive-style transcriber (replaces dual-buffer approach)
                    same_output_threshold = audio_config.get("same_output_threshold", 7)
                    live_transcriber = WhisperLiveTranscriber(
                        sample_rate=16000,
                        same_output_threshold=same_output_threshold,
                    )
                    print(f"[INIT] WhisperLiveTranscriber initialized (same_output_threshold={same_output_threshold})")

                    # Preserved from old implementation
                    saved_sentences = []        # Sentences already saved to DB (for fuzzy duplicate detection)
                    fuzzy_threshold = audio_config.get("fuzzy_duplicate_threshold", 0.85)  # Similarity threshold for dedup
                    min_words_threshold = audio_config.get("min_words", 0)  # Minimum word count to save segment

                    # Pending-remainder buffer: hold incomplete sentence fragments across captures
                    # so only complete sentences become DB rows (= translation units)
                    pending_buffer_cfg = audio_config.get("pending_buffer", {})
                    pending_buffer_enabled = pending_buffer_cfg.get("enabled", True)
                    pending_max_words = pending_buffer_cfg.get("max_words", 30)
                    pending_max_age = pending_buffer_cfg.get("max_age_seconds", 10)
                    pending_remainder = ""       # Incomplete sentence fragment held back from DB
                    pending_remainder_since = None  # When the current fragment was first buffered
                    pending_remainder_meta = None   # (start_time, end_time, confidence) of fragment

                    # LocalAgreement stabilizer for the live in-progress line: only
                    # reveals a word once two consecutive hypotheses agree, so the
                    # displayed caption stops rewriting itself (jitter). Display-only;
                    # reset whenever the underlying segment finalizes.
                    _hyp_buffer = LocalAgreementBuffer()

                    # Partial-snapshot recording: persist throttled is_final=0 rows of the
                    # live in-progress text so offline replay can reproduce the timing a
                    # live consumer experienced (final rows alone can't).
                    _partials_db_cfg = process_config.get("database", {})
                    record_partials = _partials_db_cfg.get("record_partials", True)
                    partials_min_interval_ms = _partials_db_cfg.get("partials_min_interval_ms", 1000)
                    partials_store_words = _partials_db_cfg.get("partials_store_words", False)
                    last_partial_write_ms = 0
                    last_partial_text = ""        # Last snapshot text (skip unchanged)
                    current_partial_seq = 0       # Increments per snapshot, resets per segment
                    current_partial_row_ids = []  # Partial rows awaiting segment_id linkage

                    # Full session audio file path (append mode)
                    session_audio_file = None
                    session_audio_written = False
                    # Thread safe Queue for passing data from the threaded recording callback.
                    data_queue = Queue()
                    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
                    recorder = sr.Recognizer()

                    # Use energy threshold from config
                    recorder.energy_threshold = args.energy_threshold
                    print(f"[AUDIO] Energy threshold: {args.energy_threshold}")

                    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
                    recorder.dynamic_energy_threshold = False

                    # Initialize ffmpeg audio backend with multi-level fallback
                    source = None
                    from stt.audio_capture import create_compatible_audio_source

                    # A configured file path wins: when default_microphone points at an
                    # existing file (a "Test Audio File" selection), drive the pipeline from
                    # that file only. No mic fallback, so a bad/missing file errors visibly
                    # instead of silently reverting to hardware capture.
                    if args.default_microphone and os.path.isfile(args.default_microphone):
                        print(f"[AUDIO] File playback mode: {args.default_microphone}")
                        audio_devices_to_try = [args.default_microphone]
                    else:
                        # Resolve saved stable device name (e.g. "UR22mkII") against the CURRENT
                        # ALSA enumeration first, since plughw:N,0 indices are not stable across
                        # reboots (USB vs onboard/GPU HDA cards can enumerate in either order).
                        resolved_device = None
                        saved_mic_name = audio_config.get("default_microphone_name", "")
                        if saved_mic_name:
                            try:
                                from stt.audio_capture import list_audio_devices, FFmpegAudioCapture
                                markers = audio_config.get("deprioritize_device_markers", [])
                                current_devices = list_audio_devices(deprioritize_markers=markers)
                                matched = FFmpegAudioCapture.resolve_device_by_name(saved_mic_name, current_devices)
                                if matched:
                                    resolved_device = matched["name"]
                                    print(f"[AUDIO] Resolved saved device name '{saved_mic_name}' -> '{resolved_device}'")
                                else:
                                    print(f"[AUDIO] WARNING: Saved device name '{saved_mic_name}' not found in current devices; falling back")
                            except Exception as e:
                                print(f"[AUDIO] WARNING: Device name resolution failed: {e}")

                        # Try multiple audio devices in order of preference
                        audio_devices_to_try = [
                            resolved_device,           # Name-resolved device (correct card, current index)
                            args.default_microphone,  # Configured device (e.g., plughw:1,0)
                            "default",                 # System default
                            "plughw:0,0"              # First hardware device
                        ]

                    last_error = None
                    for device in audio_devices_to_try:
                        # Skip invalid entries
                        if not device or device == "list":
                            continue

                        try:
                            print(f"[INIT] Step 2/5: Trying audio device: {device}")
                            # Get backup settings from config for MPEG-TS backup
                            backup_cfg = process_config.get("audio_backup", {})
                            # Check if .ts backup is enabled (default True for backward compatibility)
                            ts_enabled = backup_cfg.get("ts_enabled", True)
                            ts_filename_format = backup_cfg.get("filename_format", "").strip() or "%Y-%m-%d_%H%M%S"
                            ts_filename_prefix = backup_cfg.get("filename_prefix", "")
                            # Build full backup directory path (same as .wav uses)
                            ts_base_dir = backup_cfg.get("base_directory", "").strip() or BACKUP_DIR
                            ts_path_format = backup_cfg.get("path_format", "").strip() or "%Y/%m"
                            ts_formatted_path = datetime.now().strftime(ts_path_format)
                            ts_backup_dir = os.path.join(ts_base_dir, ts_formatted_path) if ts_enabled else None
                            source = create_compatible_audio_source(
                                device_name=device,
                                sample_rate=16000,
                                backup_dir=ts_backup_dir,
                                filename_format=ts_filename_format,
                                filename_prefix=ts_filename_prefix,
                                ts_enabled=ts_enabled,
                            )
                            # Start the ffmpeg capture (it will populate the data_queue)
                            source.start()
                            print(f"[OK] Audio initialized successfully with device: {device}")
                            # File-playback mode (a "Test Audio File"): expose the
                            # source + its total length so the UI can show a
                            # duration trackbar. Playback is ffmpeg -re (real time),
                            # so session elapsed ~= playback position. A mic clears
                            # these so a prior file's bar can't linger.
                            if os.path.isfile(device):
                                _dur = _audio_file.wav_duration_seconds(device)
                                if _dur is None:
                                    try:
                                        import librosa
                                        _dur = float(librosa.get_duration(path=device))
                                    except Exception:
                                        _dur = None
                                transcription_state["is_file_playback"] = True
                                transcription_state["playback_source"] = os.path.basename(device)
                                transcription_state["playback_duration"] = _dur
                            else:
                                transcription_state["is_file_playback"] = False
                                transcription_state["playback_source"] = None
                                transcription_state["playback_duration"] = None
                            break  # Success! Exit the loop
                        except Exception as e:
                            print(f"[WARN] Audio device '{device}' failed: {e}")
                            last_error = e
                            # Clean up failed source
                            if source:
                                try:
                                    source.stop()
                                except Exception:
                                    pass
                            source = None
                            # Continue to next device

                    # If all devices failed, check if ANY audio devices exist
                    if not source:
                        error_msg = None
                        try:
                            import subprocess
                            print("[CHECK] Checking for available audio devices...")
                            if platform.startswith('win'):
                                # Windows: use PowerShell to check for audio devices
                                result = subprocess.run(
                                    ["powershell", "-Command", "Get-WmiObject Win32_SoundDevice | Select-Object Name"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                )
                                if not result.stdout.strip() or "name" not in result.stdout.lower():
                                    error_msg = "No audio devices found on system. Please connect a microphone."
                                else:
                                    error_msg = f"All audio devices failed to initialize. Last error: {last_error}"
                            else:
                                result = subprocess.run(
                                    ["arecord", "-l"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if "no soundcards found" in result.stderr.lower() or "no soundcards found" in result.stdout.lower():
                                    error_msg = "No audio devices found on system. Please connect a microphone."
                                else:
                                    error_msg = f"All audio devices failed to initialize. Last error: {last_error}"
                        except FileNotFoundError:
                            error_msg = f"Audio initialization failed: {last_error}. Unable to check for devices."
                        except Exception:
                            error_msg = f"Audio initialization failed: {last_error}"

                        print(f"[ERROR] {error_msg}")
                        import traceback
                        traceback.print_exc()

                        # Update state to notify UI
                        with _transcription_state_lock:
                            transcription_state["running"] = False
                            transcription_state["status"] = "error"
                            transcription_state["error"] = error_msg
                            transcription_state["message"] = "Audio initialization failed"
                        # Clear the worker-local run flag too, so the outer loop
                        # parks in its idle branch instead of re-entering main()
                        # in a hot loop (audio_model is still None here).
                        is_running = False
                        return


                    # Check if stop was requested during audio initialization
                    if not is_running:
                        print(
                            "[INFO] Stop requested during audio initialization, cleaning up..."
                        )
                        if source:
                            try:
                                if hasattr(source, "__del__"):
                                    source.__del__()
                                del source
                            except Exception:
                                pass
                        return  # Exit main() function early

                    # Load / Download model using ModelFactory
                    model_config = process_config.get("model", {})
                    use_gpu = process_config.get("performance", {}).get("use_gpu", True)

                    # For backward compatibility with old CLI args
                    if model_config.get("type") == "whisper":
                        # Update whisper config with CLI args if provided
                        model_config["whisper"]["model"] = args.model

                    print(
                        f"[INIT] Step 3/5: Loading model ({model_config.get('type', 'whisper')}, backend={model_config.get('backend', 'whisper')})..."
                    )

                    try:
                        _model_load_t0 = time.perf_counter()
                        audio_model, processor, model_type = ModelFactory.load_model(
                            model_config, use_gpu
                        )
                        _model_load_ms = round((time.perf_counter() - _model_load_t0) * 1000, 1)
                        print(f"[OK] Model loaded successfully: {model_config.get('type', 'whisper')}")

                        # Best-effort: which device did the ASR model actually land on?
                        _model_device = None
                        try:
                            _m = getattr(audio_model, "model", audio_model)
                            _dev = getattr(_m, "device", None)
                            if _dev is not None:
                                _model_device = str(_dev).split(":")[0]  # "cuda:0" -> "cuda"
                        except Exception:
                            _model_device = None

                        # Determine the actual loaded model name for display
                        loaded_model_name = ""
                        if model_config.get("type") == "whisper":
                            # Model name now includes .en suffix directly (e.g., "small.en")
                            model_name = model_config.get("whisper", {}).get("model", "base")
                            backend = model_config.get("backend")
                            prefix = "Faster Whisper" if backend == "faster-whisper" else "Whisper"
                            loaded_model_name = f"{prefix} {model_name}"
                        elif model_config.get("type") == "huggingface":
                            # Extract model name from HuggingFace model_id (e.g., "openai/whisper-large-v3" -> "whisper-large-v3")
                            model_id = model_config.get("huggingface", {}).get("model_id", "")
                            loaded_model_name = model_id.split("/")[-1] if model_id else "huggingface"
                        elif model_config.get("type") == "custom":
                            # Extract basename from custom model path
                            model_path = model_config.get("custom", {}).get("model_path", "")
                            loaded_model_name = os.path.basename(model_path) if model_path else "custom"

                        # Update transcription state with loaded model name
                        with _transcription_state_lock:
                            transcription_state["loaded_model"] = loaded_model_name
                            transcription_state["loaded_model_device"] = _model_device
                            transcription_state["model_load_ms"] = _model_load_ms

                    except Exception as e:
                        error_msg = f"Model loading failed: {e!s}"
                        print(f"[ERROR] {error_msg}")
                        import traceback
                        traceback.print_exc()

                        # Update state to notify UI
                        with _transcription_state_lock:
                            transcription_state["running"] = False
                            transcription_state["status"] = "error"
                            transcription_state["error"] = error_msg
                            transcription_state["message"] = "Model loading failed"

                        # Cleanup audio source if it was initialized
                        if source:
                            try:
                                source.stop()
                            except Exception:
                                pass

                        # Clear orphaned GPU memory from failed model load
                        try:
                            _empty_device_cache()
                            print("[CLEANUP] GPU cache cleared after failed model load")
                        except (RuntimeError, AttributeError):
                            pass

                        # Park the worker: without this the outer loop sees
                        # is_running=True + audio_model=None and re-enters main()
                        # immediately, spinning on the failing model load.
                        is_running = False
                        return

                    # Check if stop was requested during model loading
                    if not is_running:
                        print(
                            "[INFO] Stop requested during model loading, cleaning up..."
                        )
                        # Clear references BEFORE cleanup to allow garbage collection
                        audio_model = None
                        processor = None
                        model_type = None
                        ModelFactory.cleanup_models()
                        return  # Exit main() function early

                    # Load Silero VAD model (if enabled)
                    vad_model = None
                    vad_threshold = args.vad_threshold
                    if args.use_vad:
                        print("Loading VAD model...")
                        try:
                            from silero_vad import load_silero_vad
                            vad_model = load_silero_vad()
                            print(f"VAD enabled (silero-vad) with threshold: {vad_threshold}")
                        except ImportError:
                            print("[VAD] silero-vad package not installed. Install with: pip install silero-vad")
                            print("[VAD] Continuing without VAD (using energy-based detection only)")
                            vad_model = None
                        except Exception as e:
                            print(f"[VAD] Error loading silero-vad: {e}")
                            print("[VAD] Continuing without VAD (using energy-based detection only)")
                            vad_model = None
                    else:
                        print("VAD disabled - using energy-based detection only")

                    # Check if stop was requested during VAD loading
                    if not is_running:
                        print(
                            "[INFO] Stop requested during VAD loading, cleaning up..."
                        )
                        # Clear references BEFORE cleanup to allow garbage collection
                        if vad_model:
                            del vad_model
                        audio_model = None
                        processor = None
                        model_type = None
                        vad_model = None
                        ModelFactory.cleanup_models()
                        return  # Exit main() function early

                    record_timeout = args.record_timeout
                    phrase_timeout = args.phrase_timeout

                    # Initialize database (lazy loading - only when transcription starts)
                    print("[INIT] Step 4/5: Initializing database...")

                    try:
                        # process_config was just reloaded from disk for this
                        # session; the worker's module-level config is older.
                        db_path = initialize_database(process_config)

                        # Create persistent database connection for this process
                        # This avoids overhead of opening/closing connection on every transcription
                        persistent_db_conn = sqlite3.connect(
                            db_path, check_same_thread=False, timeout=30.0
                        )
                        persistent_db_cursor = persistent_db_conn.cursor()

                        # Enable WAL mode for this connection too
                        persistent_db_cursor.execute("PRAGMA journal_mode=WAL")
                        persistent_db_cursor.execute("PRAGMA synchronous=NORMAL")
                        persistent_db_cursor.execute(
                            "PRAGMA busy_timeout=30000"
                        )  # 30 second timeout
                        # WAL/SHM sidecars are (re)created here by this connection —
                        # keep them readable by all users alongside the .db file.
                        make_db_world_readable(db_path)

                        print(f"[OK] Database initialized: {db_path}")
                    except Exception as e:
                        error_msg = f"Database initialization failed: {e!s}"
                        print(f"[ERROR] {error_msg}")
                        import traceback
                        traceback.print_exc()

                        # Update state to notify UI
                        with _transcription_state_lock:
                            transcription_state["running"] = False
                            transcription_state["status"] = "error"
                            transcription_state["error"] = error_msg
                            transcription_state["message"] = "Database initialization failed"

                        # Cleanup resources
                        if source:
                            try:
                                source.stop()
                            except Exception:
                                pass
                        # Clear references BEFORE cleanup to allow garbage collection
                        audio_model = None
                        processor = None
                        model_type = None
                        vad_model = None
                        try:
                            ModelFactory.cleanup_models()
                        except Exception:
                            pass
                        # Park the worker (audio_model is None → outer loop would
                        # otherwise re-enter main() in a hot loop).
                        is_running = False
                        return

                    # Initialize full session audio file (if .wav backup is enabled)
                    backup_config = process_config.get("audio_backup", {})
                    # Support both old "enabled" key and new "wav_enabled" key for backward compatibility
                    wav_backup_enabled = backup_config.get("wav_enabled", backup_config.get("enabled", False))
                    if wav_backup_enabled:
                        try:
                            now = datetime.now(configured_timezone)
                            base_dir = backup_config.get(
                                "base_directory", ""
                            ).strip() or BACKUP_DIR
                            path_format = backup_config.get("path_format", "").strip() or "%Y/%m"
                            formatted_path = now.strftime(path_format)
                            full_dir_path = os.path.join(base_dir, formatted_path)
                            os.makedirs(full_dir_path, exist_ok=True)

                            filename_format = backup_config.get(
                                "filename_format", ""
                            ).strip() or "%Y-%m-%d_%H%M%S"
                            filename_prefix = backup_config.get("filename_prefix", "")
                            # Build filename: {timestamp}_{prefix}.wav or {timestamp}.wav
                            if filename_prefix:
                                session_filename = f"{now.strftime(filename_format)}_{filename_prefix}.wav"
                            else:
                                session_filename = f"{now.strftime(filename_format)}.wav"
                            session_audio_file = os.path.join(
                                full_dir_path, session_filename
                            )
                            print(
                                f"[BACKUP] Full session file initialized: {session_audio_file}"
                            )
                        except Exception as e:
                            print(
                                f"[WARNING] Failed to initialize session audio file: {e}"
                            )
                            session_audio_file = None

                    def has_speech(audio_bytes, sample_rate=16000):
                        """
                        Check if audio contains speech using hybrid two-stage filtering:
                        1. Energy threshold (always applied) - fast rejection of quiet audio
                        2. VAD (if enabled) - accurate speech detection for loud audio

                        Returns True if speech is detected, False otherwise.
                        """
                        # Stage 1: Energy Threshold Filter (ALWAYS applied)
                        try:
                            # Calculate raw energy using EXACT same method as visualization
                            # Normalize to -1.0 to 1.0 range BEFORE calculating RMS
                            audio_np = (
                                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                                / 32768.0
                            )
                            rms = np.sqrt(np.mean(audio_np ** 2))
                            raw_energy = float(rms * 32768.0)  # Convert back to raw energy

                            # Get energy threshold from config (default 3500)
                            energy_threshold = process_config.get("audio", {}).get("energy_threshold", 3500)

                            # Fast rejection: if audio is too quiet, skip transcription
                            if raw_energy < energy_threshold:
                                # print(f"[FILTER] Energy rejected: {raw_energy:.0f} < {energy_threshold}")
                                return False
                            # else:
                            #     print(f"[FILTER] Energy passed: {raw_energy:.0f} >= {energy_threshold}")
                        except Exception as e:
                            # If energy calculation fails, continue to VAD check
                            print(f"[WARNING] Energy threshold check failed: {e}, continuing to VAD")

                        # Stage 2: VAD Filter (only if enabled and audio passed energy check)
                        if vad_model is not None:
                            try:
                                # Convert raw audio bytes to numpy array
                                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                                # Silero-VAD requires fixed chunk sizes: 512 samples for 16kHz, 256 for 8kHz
                                chunk_size = 512 if sample_rate == 16000 else 256

                                # Process audio in correctly-sized chunks
                                for i in range(0, len(audio_np) - chunk_size + 1, chunk_size):
                                    chunk = audio_np[i:i + chunk_size]
                                    audio_tensor = torch.from_numpy(chunk)
                                    prob = vad_model(audio_tensor, sample_rate).item()
                                    if prob >= vad_threshold:
                                        return True

                                # No chunk exceeded the VAD threshold
                                return False
                            except Exception as e:
                                # If VAD fails, default to processing the audio (passed energy check)
                                print(f"[WARNING] VAD check failed: {e}, defaulting to process audio")
                                return True

                        # VAD is disabled, audio passed energy check, so process it
                        return True

                    def save_audio_backup(wav_data_bytes, backup_config):
                        """
                        Save raw unprocessed audio input to backup directory with configurable format.
                        This saves ALL audio input (speech, music, noise, silence) before VAD filtering.
                        Uses configurable path and filename formats like database.

                        Args:
                            wav_data_bytes: WAV audio data as bytes (raw unprocessed input)
                            backup_config: Configuration dict with 'wav_enabled', 'base_directory', 'path_format', 'filename_format', 'filename_prefix', and 'format' keys

                        Returns:
                            str: Path to saved file if successful, None otherwise
                        """
                        # Support both old "enabled" key and new "wav_enabled" key for backward compatibility
                        if not backup_config.get("wav_enabled", backup_config.get("enabled", False)):
                            return None

                        try:
                            # Get current time in configured timezone
                            now = datetime.now(configured_timezone)

                            # Use configurable base directory or default
                            base_dir = backup_config.get("base_directory", "").strip() or BACKUP_DIR

                            # Use configurable path format or default (using Python strftime format)
                            path_format = backup_config.get("path_format", "").strip() or "%Y/%m"

                            # Use strftime directly with user's format
                            formatted_path = now.strftime(path_format)

                            # Create full directory path
                            full_dir_path = os.path.join(base_dir, formatted_path)
                            os.makedirs(full_dir_path, exist_ok=True)

                            # Use configurable filename format (using Python strftime format)
                            filename_format = backup_config.get(
                                "filename_format", ""
                            ).strip() or "%Y-%m-%d_%H%M%S"
                            audio_format = backup_config.get("format", "wav")
                            filename_prefix = backup_config.get(
                                "filename_prefix", ""
                            ).strip()

                            # Use strftime directly with user's format
                            formatted_filename = now.strftime(filename_format)

                            # Add prefix and extension
                            if filename_prefix:
                                filename = f"{formatted_filename}_{filename_prefix}.{audio_format}"
                            else:
                                filename = f"{formatted_filename}.{audio_format}"

                            filepath = os.path.join(full_dir_path, filename)

                            # Save audio file
                            with open(filepath, "wb") as f:
                                f.write(wav_data_bytes)

                            return filepath
                        except Exception as e:
                            print(f"[WARNING] Failed to save audio backup: {e}")
                            return None

                        try:
                            # Get current time in configured timezone
                            now = datetime.now(configured_timezone)

                            # Create directory structure: base/YYYY/YYYY-MM/
                            # Use same default as database if not specified
                            base_dir = backup_config.get("base_directory", "").strip()
                            if not base_dir:
                                base_dir = BACKUP_DIR
                            year_dir = os.path.join(base_dir, now.strftime("%Y"))
                            month_dir = os.path.join(year_dir, now.strftime("%Y-%m"))

                            # Create directories if they don't exist
                            os.makedirs(month_dir, exist_ok=True)

                            # Create filename: YYYY-MM-DD-HHmmss.wav or with custom prefix
                            audio_format = backup_config.get("format", "wav")
                            filename_prefix = backup_config.get(
                                "filename_prefix", ""
                            ).strip()

                            if filename_prefix:
                                filename = now.strftime(
                                    f"%Y-%m-%d-%H%M%S_{filename_prefix}.{audio_format}"
                                )
                            else:
                                filename = now.strftime(
                                    f"%Y-%m-%d-%H%M%S.{audio_format}"
                                )

                            filepath = os.path.join(month_dir, filename)

                            # Save the audio file
                            with open(filepath, "wb") as f:
                                f.write(wav_data_bytes)

                            return filepath
                        except Exception as e:
                            print(f"[WARNING] Failed to save audio backup: {e}")
                            return None

                    # Scratch path for per-chunk audio. Use mkstemp + close so we
                    # don't leak the open file descriptor that NamedTemporaryFile
                    # would keep alive.
                    _temp_fd, temp_file = tempfile.mkstemp(suffix=".wav")
                    os.close(_temp_fd)

                    # ffmpeg backend doesn't need ambient noise adjustment
                    print(
                        "[AUDIO] Skipping ambient noise adjustment (ffmpeg handles this)"
                    )

                    # Check if stop was requested during audio setup
                    if not is_running:
                        print(
                            "[INFO] Stop requested during audio setup, cleaning up..."
                        )
                        if persistent_db_conn:
                            try:
                                persistent_db_conn.close()
                            except (sqlite3.Error, OSError):
                                pass
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                        # Clear references BEFORE cleanup to allow garbage collection
                        if vad_model:
                            try:
                                del vad_model
                            except (NameError, AttributeError):
                                pass
                        audio_model = None
                        processor = None
                        model_type = None
                        vad_model = None
                        ModelFactory.cleanup_models()
                        if source:
                            try:
                                if hasattr(source, "__del__"):
                                    source.__del__()
                                del source
                            except Exception:
                                pass
                        return  # Exit main() function early

                    audio_callback_count = [
                        0
                    ]  # Use list to allow modification in nested function

                    def record_callback(_, audio: sr.AudioData) -> None:
                        """
                        Threaded callback function to recieve audio data when recordings finish.
                        audio: An AudioData containing the recorded bytes.
                        """
                        # Grab the raw bytes and push it into the thread safe queue.
                        data = audio.get_raw_data()
                        data_queue.put(data)
                        audio_callback_count[0] += 1
                        if (
                            audio_callback_count[0] <= 5
                            or audio_callback_count[0] % 100 == 0
                        ):
                            print(
                                f"[AUDIO_CALLBACK] Called {audio_callback_count[0]} times, queue_size={data_queue.qsize()}, data_size={len(data)}"
                            )

                        # Calculate and update energy level for every callback
                        try:
                            import audioop

                            # Calculate RMS energy (raw value, not normalized)
                            energy = audioop.rms(data, source.SAMPLE_WIDTH)

                            # Store raw energy and dB in shared state
                            if energy > 0:
                                db = 20 * np.log10(energy / 32768.0)
                            else:
                                db = -60

                            level = max(0, min(100, (db + 60) * (100 / 60)))

                            transcription_state["audio_energy"] = (
                                energy  # Raw energy value
                            )
                            transcription_state["audio_level"] = level
                            transcription_state["audio_db"] = db
                        except Exception:
                            pass  # Don't break callback on error

                    # For ffmpeg, the capture is already running and filling source.data_queue
                    # We need to use that queue as data_queue
                    data_queue = source.data_queue
                    print(f"[AUDIO] OK: ffmpeg backend already capturing audio to queue (queue id={id(data_queue)})", flush=True)

                    # Cue the user that we're ready to go.
                    print("Model loaded.\n")

                    # Update transcription state to running - ALL initialization complete
                    print("[INIT] Step 5/5: Starting transcription loop...")
                    with _transcription_state_lock:
                        transcription_state["running"] = True
                        transcription_state["status"] = "running"
                        transcription_state["message"] = "Transcription is active and ready"
                        transcription_state["error"] = None
                        transcription_state["start_time"] = time.time()
                        # Zero the health-perf counters for the new session
                        transcription_state["infer_ms_ema"] = None
                        transcription_state["rtf_ema"] = None
                        transcription_state["segments_total"] = 0
                        transcription_state["segments_per_min"] = None
                        transcription_state["rows_saved"] = 0
                        transcription_state["queue_depth"] = None
                    print("[READY] Transcription system initialized successfully!")

                    # Health-dashboard performance accounting (worker-local; pushed
                    # to the shared state periodically so /api/health can read it).
                    _perf_state = None       # EMA dict from stt.metrics.update_perf_ema
                    _perf_first_ts = None    # epoch of the first transcribed chunk (throughput window)
                    _perf_last_push = 0.0    # last time we wrote perf fields to shared state
                    _eof_stop_deadline = None  # set when a played file hits EOF; grace to finalize the tail

                    while True:
                        try:
                            # Check if we should exit the loop
                            if not is_running:
                                print("[LOOP] is_running is False, exiting main loop")
                                break

                            # Check for stop commands and calibration commands with non-blocking operations
                            try:
                                while not control_queue.empty():
                                    command = control_queue.get_nowait()
                                    if command["command"] == "stop":
                                        print("[LOOP] Stop command received, exiting main loop")
                                        is_running = False
                                        break
                                    elif command["command"] == "start_calibration":
                                        # Handle calibration start command in inner loop - use local state
                                        calibration_mode = True
                                        calibration_data = {
                                            "start_time": time.time(),
                                            "duration": command.get("duration", 30),
                                            "noise_samples": [],
                                            "speech_samples": [],
                                            "silence_durations": [],
                                            "energy_levels": [],
                                            "vad_probabilities": [],
                                        }
                                        print(f"[CALIBRATION-PROCESS] Calibration mode enabled in transcription process - duration: {calibration_data['duration']}s", flush=True)
                            except Exception as e:
                                print(f"[WARNING] Error checking control queue: {e}")

                            # Check again after processing control queue
                            if not is_running:
                                break

                            # End of a played audio file: finalize the last phrase
                            # via the existing phrase-timeout path, then stop just
                            # like a user stop. A mic never sets playback_finished,
                            # so live capture is unaffected.
                            if (_eof_stop_deadline is None and source is not None
                                    and getattr(source, "playback_finished", None) is not None
                                    and source.playback_finished.is_set()):
                                print("[LOOP] Audio file reached end of stream — finalizing tail, then stopping")
                                if hasattr(source, "flush_buffer"):
                                    source.flush_buffer()
                                _eof_stop_deadline = time.time() + float(phrase_timeout) + 1.0
                            if _eof_stop_deadline is not None:
                                # Feed trailing silence so the drain/phrase-timeout
                                # path runs (it's gated on a non-empty queue) and
                                # finalizes the last phrase.
                                if data_queue.empty() and time.time() < _eof_stop_deadline:
                                    data_queue.put(b"\x00" * (source.chunk_size * source.SAMPLE_WIDTH))
                                if time.time() >= _eof_stop_deadline:
                                    print("[LOOP] End-of-file grace elapsed — stopping transcription")
                                    is_running = False
                                    break

                            # Check for config updates (hot-reload) with non-blocking operations
                            try:
                                while not config_queue.empty():
                                    config_update = config_queue.get_nowait()
                                    if config_update["type"] == "config_update":
                                        new_config = config_update["config"]

                                        # Update process_config to use new values throughout
                                        process_config.update(new_config)

                                        # Update hot-reloadable settings
                                        if "audio" in new_config:
                                            recorder.energy_threshold = new_config[
                                                "audio"
                                            ].get(
                                                "energy_threshold",
                                                recorder.energy_threshold,
                                            )
                                            record_timeout = new_config["audio"].get(
                                                "record_timeout", record_timeout
                                            )
                                            phrase_timeout = new_config["audio"].get(
                                                "phrase_timeout", phrase_timeout
                                            )
                                            if "pending_buffer" in new_config["audio"]:
                                                _pb_cfg = new_config["audio"]["pending_buffer"]
                                                pending_buffer_enabled = _pb_cfg.get("enabled", pending_buffer_enabled)
                                                pending_max_words = _pb_cfg.get("max_words", pending_max_words)
                                                pending_max_age = _pb_cfg.get("max_age_seconds", pending_max_age)

                                        if "vad" in new_config:
                                            vad_threshold = new_config["vad"].get(
                                                "threshold", vad_threshold
                                            )

                                        # A hot reload can change the transcribing model
                                        # itself. Rows are NULL while it matches what the
                                        # session recorded at start, so from here on they
                                        # must carry the new one — otherwise every row
                                        # after the change silently claims the old model.
                                        _asr_now = _session_asr_row_label(process_config)
                                        if _asr_now != _asr_session_label:
                                            print(f"[DB] ASR model changed mid-session: "
                                                  f"{_asr_session_label!r} -> {_asr_now!r}; "
                                                  f"stamping it on new rows")
                                            _asr_session_label = _asr_now
                                            _whisper_mt_model = _asr_now
                                            try:
                                                _set_asr_row_stamp(
                                                    persistent_db_conn,
                                                    _session_row_label_if_changed(
                                                        _asr_now, _asr_baseline_label))
                                            except Exception as _stamp_err:
                                                print(f"[DB] WARNING: could not update the "
                                                      f"ASR row stamp ({_stamp_err})")

                                        print(
                                            f"[OK] Config hot-reloaded: energy={recorder.energy_threshold}, vad_threshold={vad_threshold}, process_config updated"
                                        )
                            except Exception as e:
                                print(f"[WARNING] Error processing config update: {e}")
                                # Continue processing even if queue operations fail

                            # Get the current time in configured timezone
                            now = datetime.now(configured_timezone)

                            # Pull raw recorded audio from the queue.
                            if not data_queue.empty():
                                # Track accumulated new data for session file
                                accumulated_new_data = bytes()
                                # Use timeout to prevent deadlock
                                if _audio_queue_lock.acquire(timeout=5.0):
                                    try:
                                        while not data_queue.empty():
                                            data = data_queue.get()
                                            skip_transcription = False

                                            # Calibration mode: collect environmental data (two-step process)
                                            if calibration_mode and calibration_data:
                                                skip_transcription = True
                                                current_step = calibration_state.get("step", 1)

                                                # Check if we need to reset the local timer for Step 2
                                                if calibration_state.get("reset_timer", False):
                                                    calibration_data["start_time"] = time.time()
                                                    calibration_state["reset_timer"] = False
                                                    print(f"[CALIBRATION] Timer reset for Step 2 - new start_time: {calibration_data['start_time']}", flush=True)

                                                # CRITICAL FIX: If starting at Step 2 from the beginning (skip_step1),
                                                # reset the timer on the FIRST audio frame to avoid premature completion
                                                if current_step == 2 and not calibration_data.get("step2_timer_initialized", False):
                                                    calibration_data["start_time"] = time.time()
                                                    calibration_data["step2_timer_initialized"] = True
                                                    print(f"[CALIBRATION] Step 2 timer initialized on first frame - start_time: {calibration_data['start_time']}", flush=True)

                                                # print(f"[CALIBRATION-DEBUG] Processing audio data - Step {current_step}")
                                                try:
                                                    # Calculate energy level
                                                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                                                    raw_energy = np.sqrt(np.mean(audio_np**2)) * 32768
                                                    calibration_data["energy_levels"].append(raw_energy)
                                                    calibration_data_shared["energy_levels"].append(raw_energy)

                                                    if current_step == 1:
                                                        # STEP 1: Collect noise floor (NO speech detection)

                                                        # Check if step 1 is already complete (waiting for user to click "Start Step 2")
                                                        if not calibration_state.get("step1_complete", False):
                                                            sample_dict = {
                                                                "energy": raw_energy,
                                                                "timestamp": time.time(),
                                                            }

                                                            # Store in step 1 data and shared data
                                                            calibration_step1_data["noise_energies"].append(raw_energy)
                                                            calibration_data_shared["noise_samples"].append(sample_dict)
                                                            calibration_data["noise_samples"].append(sample_dict)
                                                            calibration_state["noise_samples"] = len(calibration_data["noise_samples"])

                                                            # Check if step 1 is complete
                                                            elapsed = time.time() - calibration_data["start_time"]
                                                            if elapsed >= calibration_data["duration"]:
                                                                # Calculate noise statistics for step 2
                                                                noise_list = list(calibration_step1_data["noise_energies"])
                                                                print(f"[CALIBRATION] Step 1 time elapsed: {elapsed:.1f}s >= {calibration_data['duration']}s, noise samples collected: {len(noise_list)}", flush=True)

                                                                if noise_list:
                                                                    calibration_step1_data["avg_noise"] = statistics.mean(noise_list)
                                                                    calibration_step1_data["max_noise"] = max(noise_list)
                                                                    print(f"[CALIBRATION] Step 1 complete - avg_noise: {calibration_step1_data['avg_noise']:.1f}, max_noise: {calibration_step1_data['max_noise']:.1f}", flush=True)
                                                                else:
                                                                    # No noise samples collected - use conservative defaults
                                                                    calibration_step1_data["avg_noise"] = 300.0
                                                                    calibration_step1_data["max_noise"] = 500.0
                                                                    print("[CALIBRATION] Step 1 complete - WARNING: No noise samples collected, using defaults", flush=True)

                                                                # Mark step 1 as complete but DON'T auto-transition
                                                                # Wait for user to click "Start Step 2" button
                                                                calibration_state["step1_complete"] = True
                                                                print("[CALIBRATION] Set step1_complete = True", flush=True)
                                                                # DON'T set step = 2 yet - user must manually continue

                                                    elif current_step == 2:
                                                        # STEP 2: Collect speech with temporary low threshold
                                                        # Use noise floor + 100 as temporary threshold
                                                        temp_threshold = int(calibration_step1_data.get("avg_noise", 300) + 100)

                                                        # Temporarily override energy threshold for this check
                                                        old_threshold = process_config["audio"]["energy_threshold"]
                                                        process_config["audio"]["energy_threshold"] = temp_threshold

                                                        is_speech = has_speech(data, source.SAMPLE_RATE)

                                                        # Restore original threshold
                                                        process_config["audio"]["energy_threshold"] = old_threshold

                                                        sample_dict = {
                                                            "energy": raw_energy,
                                                            "timestamp": time.time(),
                                                        }

                                                        if is_speech:
                                                            calibration_data["speech_samples"].append(sample_dict)
                                                            calibration_data_shared["speech_samples"].append(sample_dict)
                                                            calibration_state["speech_samples"] = len(calibration_data["speech_samples"])

                                                            # Track when speech ends for pause detection
                                                            calibration_data["last_speech_time"] = time.time()

                                                            # Measure VAD probability if VAD enabled
                                                            if vad_model is not None:
                                                                try:
                                                                    # silero-vad pip package - simple call
                                                                    audio_tensor = torch.from_numpy(audio_np).float()
                                                                    speech_prob = vad_model(audio_tensor, source.SAMPLE_RATE).item()
                                                                    calibration_data["vad_probabilities"].append(speech_prob)
                                                                    calibration_data_shared["vad_probabilities"].append(speech_prob)
                                                                except (RuntimeError, TypeError, ValueError):
                                                                    pass
                                                        else:
                                                            # Still collecting some noise during speech phase
                                                            calibration_data["noise_samples"].append(sample_dict)
                                                            calibration_data_shared["noise_samples"].append(sample_dict)
                                                            calibration_state["noise_samples"] = len(calibration_data["noise_samples"])

                                                            # Track silence durations between speech segments
                                                            # Only record if we've had speech before and this is a new silence period
                                                            if "last_speech_time" in calibration_data and "silence_start_time" not in calibration_data:
                                                                # Just transitioned from speech to silence
                                                                calibration_data["silence_start_time"] = time.time()
                                                            elif "last_speech_time" in calibration_data and "silence_start_time" in calibration_data:
                                                                # Already in silence, check if enough time passed to record it
                                                                current_silence = time.time() - calibration_data["silence_start_time"]
                                                                if current_silence > 0.3:  # Only record meaningful silences
                                                                    # Mark that we've recorded this silence period
                                                                    calibration_data["silence_start_time"] = time.time()

                                                        # If we just detected speech after silence, record the pause duration
                                                        if is_speech and "silence_start_time" in calibration_data:
                                                            silence_duration = time.time() - calibration_data["silence_start_time"]
                                                            if silence_duration > 0.3:  # Only record pauses > 0.3s
                                                                calibration_data["silence_durations"].append(silence_duration)
                                                                calibration_data_shared["silence_durations"].append(silence_duration)
                                                                calibration_state["silence_samples"] = len(calibration_data["silence_durations"])
                                                                print(f"[CALIBRATION] Detected pause: {silence_duration:.2f}s", flush=True)
                                                            # Clear silence tracking
                                                            del calibration_data["silence_start_time"]

                                                        # Check if step 2 is complete
                                                        elapsed = time.time() - calibration_data["start_time"]

                                                        # Debug logging every 5 seconds
                                                        if not hasattr(calibration_data, '_last_log_time'):
                                                            calibration_data['_last_log_time'] = 0
                                                        if elapsed - calibration_data.get('_last_log_time', 0) >= 5:
                                                            print(f"[CALIBRATION-TIMER] Step 2 - elapsed: {elapsed:.1f}s / {calibration_data['duration']}s", flush=True)
                                                            calibration_data['_last_log_time'] = elapsed

                                                        # Force completion if elapsed significantly exceeds duration (safety mechanism)
                                                        if elapsed >= calibration_data["duration"] or elapsed > (calibration_data["duration"] * 2):
                                                            if elapsed > (calibration_data["duration"] * 2):
                                                                print(f"[CALIBRATION] WARNING: Forced completion - elapsed {elapsed:.1f}s exceeds 2x duration {calibration_data['duration']}s", flush=True)
                                                            # Both steps complete - end calibration
                                                            calibration_mode = False
                                                            calibration_state["active"] = False
                                                            print(f"[CALIBRATION] Complete - {len(calibration_data['speech_samples'])} speech samples, {len(calibration_data['noise_samples'])} noise samples, elapsed: {elapsed:.1f}s", flush=True)

                                                except Exception as calib_error:
                                                    print(f"[CALIBRATION] Error: {calib_error}")

                                            # Always stream audio to web clients regardless of VAD/transcription state
                                            if transcription_state.get("audio_stream_enabled", False):
                                                try:
                                                    audio_stream_queue.put_nowait(data)
                                                except Full:
                                                    pass  # Queue full, drop chunk to prevent lag

                                            # Add audio to WhisperLive transcriber buffer (replaces old dual-buffer approach)
                                            if not skip_transcription:
                                                live_transcriber.add_frames(data)
                                                accumulated_new_data += data
                                    finally:
                                        _audio_queue_lock.release()
                                else:
                                    print("[WARN] Failed to acquire audio queue lock, skipping this iteration")

                                # Write accumulated audio data to session file IMMEDIATELY (before phrase logic)
                                if session_audio_file and accumulated_new_data:
                                    try:
                                        # For continuous append, we need to handle WAV format properly
                                        # First write: write full WAV header + data
                                        # Subsequent writes: append only PCM data, update header
                                        if not session_audio_written:
                                            # First write - create WAV file with header
                                            temp_audio = sr.AudioData(
                                                accumulated_new_data,
                                                source.SAMPLE_RATE,
                                                source.SAMPLE_WIDTH,
                                            )
                                            temp_wav = temp_audio.get_wav_data()
                                            with open(session_audio_file, "wb") as f:
                                                f.write(temp_wav)
                                            session_audio_written = True
                                            print(
                                                "[BACKUP] Started session audio file"
                                            )
                                        else:
                                            # Append PCM data only (skip WAV header)
                                            with open(session_audio_file, "ab") as f:
                                                f.write(accumulated_new_data)
                                    except Exception as e:
                                        print(
                                            f"[WARNING] Failed to append to session file: {e}"
                                        )

                                # Calculate and store audio level for volume meter
                                try:
                                    # Convert audio bytes to numpy array for level calculation
                                    audio_np = (
                                        np.frombuffer(
                                            accumulated_new_data, dtype=np.int16
                                        ).astype(np.float32)
                                        / 32768.0
                                    )

                                    # Calculate RMS (Root Mean Square) for audio level
                                    if len(audio_np) > 0:
                                        rms = np.sqrt(np.mean(audio_np**2))
                                    else:
                                        rms = 0

                                    # Convert to dB
                                    if rms > 0:
                                        db = 20 * np.log10(rms)
                                    else:
                                        db = -60  # Silence

                                    # Normalize to 0-100% for display (assuming -60dB to 0dB range)
                                    level = max(0, min(100, (db + 60) * (100 / 60)))

                                    # Convert RMS back to raw energy value
                                    raw_energy = float(rms * 32768.0)

                                    # Store in shared state (parent process will emit via Socket.IO)
                                    transcription_state["audio_level"] = level
                                    transcription_state["audio_db"] = db
                                    transcription_state["audio_energy"] = raw_energy

                                    # Hand the raw pre-VAD buffer to the background
                                    # PANNs detector (non-blocking; never stalls draining).
                                    submit_music_detection(process_config, transcription_state, audio_np, source.SAMPLE_RATE)
                                except Exception as e:
                                    print(f"[WARNING] Audio level calculation failed: {e}")

                                # Check if audio contains speech using VAD
                                speech_detected = has_speech(accumulated_new_data, source.SAMPLE_RATE)

                                # Let confidently-detected music through to transcription even
                                # when VAD would drop it. Music is always transcribed; the
                                # transcribe_detected_music toggle only controls whether its
                                # rows are visible or auto-denied ('music') at insert time.
                                _std_cfg = process_config.get("speech_type_detection", {})
                                if (not speech_detected
                                        and (transcription_state.get("music_prob") or 0.0)
                                            > _std_cfg.get("music_prob_threshold", 0.5)):
                                    speech_detected = True

                                # Check if phrase is complete (silence after speech)
                                phrase_complete = False
                                if not speech_detected:
                                    # No speech - check if we had speech recently and it's been quiet for phrase_timeout
                                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                                        phrase_complete = True
                                        print(f"[PHRASE_TIMEOUT] Silence for {phrase_timeout}s detected", flush=True)
                                        # Flush any partial audio buffer to ensure last words are transcribed immediately
                                        if hasattr(source, 'flush_buffer'):
                                            source.flush_buffer()
                                    # Skip transcription if no speech detected (but still check phrase_complete below)
                                    if not phrase_complete:
                                        sleep(0.25)
                                        continue
                                else:
                                    # Speech detected - update phrase_time
                                    phrase_time = now

                                # === WHISPER-LIVE TRANSCRIPTION APPROACH ===
                                # Get audio chunk from the rolling buffer
                                audio_chunk, chunk_duration = live_transcriber.get_audio_chunk_for_processing()

                                # Need at least 1 second of audio before transcribing
                                if chunk_duration < 1.0:
                                    sleep(0.1)
                                    continue

                                # Optional pre-ASR loudness normalization: one gain per pass,
                                # boost-only, applied to the transcription copy. The rolling
                                # buffer, energy gate, and calibration always see raw levels.
                                _norm_cfg = process_config.get("audio", {}).get("loudness_normalization", {})
                                if _norm_cfg.get("enabled", False) and audio_chunk is not None and len(audio_chunk) > 0:
                                    try:
                                        _rms = float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float64)))))
                                        if _rms > 1e-6:
                                            _target_rms = 10.0 ** (float(_norm_cfg.get("target_rms_dbfs", -20)) / 20.0)
                                            _gain = min(float(_norm_cfg.get("max_gain", 10.0)), _target_rms / _rms)
                                            if _gain > 1.01:
                                                _peak = float(np.max(np.abs(audio_chunk)))
                                                if _peak > 0:
                                                    _gain = min(_gain, 0.99 / _peak)  # never clip
                                                if _gain > 1.01:
                                                    audio_chunk = (audio_chunk * np.float32(_gain)).astype(np.float32)
                                    except Exception as _norm_err:
                                        print(f"[WARNING] Loudness normalization failed: {_norm_err}")

                                try:
                                    # Get language from config
                                    live_language = process_config.get("audio", {}).get("language", "auto")
                                    # Get whisper params from config or use defaults
                                    whisper_params = process_config.get("whisper_decoding", {}).get(
                                        "live_transcription", LIVE_TRANSCRIPTION_PARAMS
                                    )

                                    # Live requires temperature 0: nonzero output varies between
                                    # passes, same_output_threshold never triggers, and no rows
                                    # save. Covers hand-edited configs the save endpoint missed.
                                    _temp_val = whisper_params.get("temperature", 0)
                                    if isinstance(_temp_val, (list, tuple)) or (_temp_val or 0) != 0:
                                        whisper_params = dict(whisper_params)
                                        whisper_params["temperature"] = 0.0
                                        if not globals().get("_live_temp_clamp_warned"):
                                            globals()["_live_temp_clamp_warned"] = True
                                            print(f"[LIVE] temperature {_temp_val!r} forced to 0.0 — nonzero temperature prevents segment finalization (no rows would save)")

                                    # Enable word_timestamps for confidence highlighting if configured
                                    corrections_config = process_config.get("corrections", {})
                                    if corrections_config.get("confidence_highlighting", True) and corrections_config.get("enabled", True):
                                        whisper_params = dict(whisper_params)  # Copy to avoid mutating config
                                        whisper_params["word_timestamps"] = True

                                    # Cross-capture context: feed the tail of the finalized transcript
                                    # as initial_prompt so each capture knows what came before.
                                    # Never include pending_remainder — that audio is still being
                                    # re-transcribed from the rolling buffer and would get echoed.
                                    _ctx_prompt_added = False
                                    _prompt_before_ctx = None
                                    ctx_prompt_cfg = process_config.get("audio", {}).get("context_prompt", {})
                                    if ctx_prompt_cfg.get("enabled", True) and saved_sentences:
                                        ctx_max_chars = ctx_prompt_cfg.get("max_chars", 200)
                                        if ctx_max_chars > 0:
                                            prompt_tail = " ".join(saved_sentences[-5:])
                                            if len(prompt_tail) > ctx_max_chars:
                                                prompt_tail = prompt_tail[-ctx_max_chars:]
                                                _cut = prompt_tail.find(" ")
                                                # Drop the leading partial word; if the tail is one
                                                # unbroken run with no space, discard it entirely
                                                if _cut > 0:
                                                    prompt_tail = prompt_tail[_cut + 1:]
                                                elif _cut < 0:
                                                    prompt_tail = ""
                                            if prompt_tail:
                                                whisper_params = dict(whisper_params)
                                                existing = whisper_params.get("initial_prompt")
                                                _prompt_before_ctx = existing
                                                whisper_params["initial_prompt"] = (existing + " " + prompt_tail) if existing else prompt_tail
                                                _ctx_prompt_added = True


                                    # Transcribe the audio chunk and get segments with timestamps
                                    # This is key to avoiding overlaps - Whisper knows segment boundaries
                                    _infer_t0 = time.perf_counter()
                                    segments = ModelFactory.transcribe(
                                        audio_model,
                                        processor,
                                        model_type,
                                        audio_chunk,
                                        language=live_language,
                                        whisper_params=whisper_params,
                                        return_segments=True
                                    )
                                    # Health metrics: fold this chunk's transcribe time into the
                                    # running EMA / real-time-factor and push to shared state at
                                    # most ~1/s. Never let instrumentation break transcription.
                                    try:
                                        _infer_ms = (time.perf_counter() - _infer_t0) * 1000.0
                                        _now = time.time()
                                        if _perf_first_ts is None:
                                            _perf_first_ts = _now
                                        _perf_state = _metrics.update_perf_ema(
                                            _perf_state, _infer_ms, chunk_duration
                                        )
                                        if _now - _perf_last_push >= 1.0:
                                            _perf_last_push = _now
                                            try:
                                                _qd = audio_stream_queue.qsize()
                                            except (NotImplementedError, OSError):
                                                _qd = None
                                            transcription_state.update({
                                                "infer_ms_ema": _perf_state["infer_ms_ema"],
                                                "rtf_ema": _perf_state["rtf_ema"],
                                                "segments_total": _perf_state["segments_total"],
                                                "segments_per_min": _metrics.segments_per_minute(
                                                    _perf_state["segments_total"], _perf_first_ts, _now
                                                ),
                                                "rows_saved": len(saved_sentences),
                                                "queue_depth": _qd,
                                            })
                                    except Exception:
                                        pass

                                    # === Whisper Translation Pass (dual-pass) ===
                                    # If Whisper-based translation is active, run a second pass on the same audio
                                    _whisper_translated_text = None
                                    _pass2_timed = []  # (session_start, session_end, text) per pass-2 segment
                                    _trans_cfg = process_config.get("live_translation", {})
                                    _trans_method = _trans_cfg.get("translation_method", "nllb")
                                    _trans_enabled = _trans_cfg.get("enabled", False)
                                    if _trans_enabled and _trans_method in ("whisper_translate", "whisper_forced_lang") and segments:
                                        try:
                                            _target_lang = _trans_cfg.get("target_language", "en")
                                            _pass2_params = dict(whisper_params)  # Copy pass 1 params
                                            # Drop the source-language context tail for the translation
                                            # pass - it would bias output toward the source language
                                            if _ctx_prompt_added:
                                                if _prompt_before_ctx:
                                                    _pass2_params["initial_prompt"] = _prompt_before_ctx
                                                else:
                                                    _pass2_params.pop("initial_prompt", None)
                                            _pass2_language = live_language

                                            if _trans_method == "whisper_translate" and _target_lang == "en":
                                                _pass2_params["task"] = "translate"
                                            elif _trans_method == "whisper_forced_lang":
                                                _pass2_language = _target_lang

                                            _pass2_segments = ModelFactory.transcribe(
                                                audio_model, processor, model_type,
                                                audio_chunk,
                                                language=_pass2_language,
                                                whisper_params=_pass2_params,
                                                return_segments=True
                                            )
                                            if _pass2_segments:
                                                _whisper_translated_text = " ".join(
                                                    s.get("text", "").strip() for s in _pass2_segments if s.get("text", "").strip()
                                                )
                                                # Session-time pass-2 segments (timestamp_offset hasn't
                                                # advanced yet — update_segments runs after this) so the
                                                # translation can later be scoped to the batch's time span
                                                _pass2_offset = live_transcriber.timestamp_offset
                                                _pass2_timed = [
                                                    (
                                                        _pass2_offset + (s.get("start") or 0),
                                                        _pass2_offset + (s.get("end") or 0),
                                                        s.get("text", "").strip(),
                                                    )
                                                    for s in _pass2_segments if s.get("text", "").strip()
                                                ]
                                                if _whisper_translated_text:
                                                    print(f"[WHISPER-TRANSLATE] Pass 2: '{_whisper_translated_text[:80]}'", flush=True)
                                        except Exception as _wt_err:
                                            print(f"[WHISPER-TRANSLATE] Pass 2 error: {_wt_err}", flush=True)

                                    # Update segments using Whisper-Live's approach
                                    # This finalizes all segments except the last one immediately
                                    result = live_transcriber.update_segments(segments, chunk_duration)

                                    _cjk_filter_enabled = config.get("hallucination_filter", {}).get("cjk_filter_enabled", True)
                                    # Music rows are transcribed but auto-denied when the user
                                    # hasn't opted in to seeing lyrics (restorable in /corrections).
                                    # The threshold in effect is recorded in the deny reason
                                    # ('music:<thr>') so Music Sensitivity can be tuned against
                                    # each denied row's stored music_prob.
                                    _transcribe_music_enabled = process_config.get("speech_type_detection", {}).get("transcribe_detected_music", False)
                                    _music_deny_reason = f"music:{process_config.get('speech_type_detection', {}).get('music_prob_threshold', 0.5):g}"

                                    # Handle completed segments (save to DB)
                                    if result['completed_segments']:
                                        # A segment boundary finalized -> the live hypothesis
                                        # restarts on a fresh incomplete segment.
                                        _hyp_buffer.reset()
                                        confidence_threshold = process_config.get("corrections", {}).get("confidence_threshold", 0.7)
                                        # Collect the batch of completed segments into one text so
                                        # sentences spanning Whisper's segment boundaries stay intact
                                        batch_parts = []
                                        batch_start = None
                                        batch_end = 0
                                        batch_confidences = []
                                        for segment in result['completed_segments']:
                                            segment_text = segment.get('text', '').strip()
                                            # Compute average word confidence for this segment
                                            word_confidences = segment.get('words', [])
                                            if word_confidences:
                                                probs = [w.get('probability') for w in word_confidences if w.get('probability') is not None]
                                                if probs:
                                                    batch_confidences.append(sum(probs) / len(probs))
                                            if not segment_text:
                                                continue
                                            # Remove overlapping prefix from previous saved text
                                            if saved_sentences:
                                                segment_text = remove_overlapping_prefix(segment_text, saved_sentences[-1])
                                            # The rolling buffer can re-transcribe words already held in
                                            # the pending fragment - strip those too
                                            if segment_text and pending_remainder:
                                                segment_text = remove_overlapping_prefix(segment_text, pending_remainder)
                                            if not segment_text:
                                                continue  # Entire segment was overlap
                                            batch_parts.append(segment_text)
                                            if batch_start is None:
                                                batch_start = segment.get('start', 0)
                                            batch_end = segment.get('end', 0)

                                        if batch_parts:
                                            batch_text = " ".join(batch_parts)
                                            segment_start = batch_start if batch_start is not None else 0
                                            segment_end = batch_end
                                            segment_confidence = sum(batch_confidences) / len(batch_confidences) if batch_confidences else None
                                            segment_speech_type = finalized_audio_type(process_config, transcription_state)
                                            transcription_state['audio_type'] = segment_speech_type
                                            segment_audio_tag = transcription_state.get("audio_tag")
                                            segment_music_prob = transcription_state.get("music_prob")
                                            # Source language ISO code: configured value, or Whisper's
                                            # detected language when audio.language is 'auto'.
                                            _detected_lang = next((s.get('language') for s in result['completed_segments'] if s.get('language')), None)
                                            # Never NULL on a non-blank row: configured -> detected -> 'und' (ISO 639 undetermined)
                                            src_lang = (live_language if (live_language and live_language != "auto") else _detected_lang) or "und"
                                            # Prepend the fragment held from the previous capture so it
                                            # can complete its sentence
                                            if pending_remainder:
                                                batch_text = pending_remainder + " " + batch_text
                                                if pending_remainder_meta:
                                                    segment_start = pending_remainder_meta[0]
                                                    if segment_confidence is None:
                                                        segment_confidence = pending_remainder_meta[2]

                                            # Split into sentences
                                            sentences, remainder = split_into_sentences(batch_text)

                                            if pending_buffer_enabled:
                                                # Hold the incomplete remainder for the next capture
                                                # instead of saving a fragment row
                                                if remainder:
                                                    if sentences or pending_remainder_since is None:
                                                        # Fresh fragment (old one consumed or none existed)
                                                        pending_remainder_since = now
                                                        pending_remainder_meta = (batch_end if sentences else segment_start, batch_end, segment_confidence)
                                                    elif pending_remainder_meta:
                                                        # Fragment still growing - keep its start, extend its end
                                                        pending_remainder_meta = (pending_remainder_meta[0], batch_end, pending_remainder_meta[2])
                                                else:
                                                    pending_remainder_since = None
                                                    pending_remainder_meta = None
                                                pending_remainder = remainder
                                                remainder = ""  # Insert block below must not save the held fragment

                                            # Per-word timing+confidence for words_json, attributed to
                                            # each re-split sentence by max temporal overlap. Built from
                                            # this batch's words (already in hand); a sentence stitched
                                            # across chunks keeps only its in-chunk words, never wrong ones.
                                            _word_stream = words_to_session_ms(result['completed_segments'])
                                            _sentence_word_groups = attribute_words_to_sentences(_word_stream, len(sentences))
                                            _words_source = model_type

                                            # Save substantial sentences to DB
                                            MIN_WORDS = min_words_threshold
                                            _newly_inserted_ids = []  # Track IDs for Whisper translation caching
                                            _accepted_rows = []  # (row_id, text) of non-denied rows — whisper translation targets
                                            with _db_lock:
                                                try:
                                                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                                    ts_ms = int(now.timestamp() * 1000)
                                                    for _sidx, sentence in enumerate(sentences):
                                                        # CJK filter: applied here so we have both versions for shadow row
                                                        _cjk_deny = False
                                                        _cjk_shadow = None
                                                        if _cjk_filter_enabled:
                                                            _cjk_stripped = filter_hallucinated_text(sentence, live_language)
                                                            if not _cjk_stripped.strip():
                                                                _cjk_deny = True
                                                                print(f"[CJK→DENIED] '{sentence[:40]}'", flush=True)
                                                            elif _cjk_stripped != sentence:
                                                                _cjk_shadow = sentence  # original with CJK → shadow row
                                                                sentence = _cjk_stripped

                                                        _is_hallucination = is_whisper_hallucination(sentence)
                                                        if _is_hallucination:
                                                            print(f"[HALLUCINATION→DENIED] '{sentence[:40]}'", flush=True)

                                                        _music_deny = (not _transcribe_music_enabled) and segment_speech_type == "Music"
                                                        if _music_deny and not (_is_hallucination or _cjk_deny):
                                                            print(f"[MUSIC→DENIED] '{sentence[:40]}'", flush=True)
                                                        _denied = 1 if (_is_hallucination or _cjk_deny or _music_deny) else 0
                                                        _denied_reason = ('hallucination' if _is_hallucination else 'cjk' if _cjk_deny else _music_deny_reason) if _denied else None

                                                        # original_text = verbatim ASR before profanity normalization
                                                        # (words_json `w` tokens are the fullest-raw form, never filtered)
                                                        _verbatim = sentence
                                                        sentence = apply_profanity_filter(sentence)
                                                        word_count = len(sentence.split())
                                                        is_dup = is_fuzzy_duplicate(sentence, saved_sentences, fuzzy_threshold)
                                                        needs_review = 1 if (segment_confidence is not None and segment_confidence < confidence_threshold) else 0
                                                        _words_json = words_json_or_none(_sentence_word_groups[_sidx] if _sidx < len(_sentence_word_groups) else None)
                                                        if _denied or (word_count >= MIN_WORDS and not is_dup):
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                (timestamp, sentence, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim, _words_json, _words_source, live_session_id, _denied, _denied_reason),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            if not _denied:
                                                                _accepted_rows.append((_newly_inserted_ids[-1], sentence))
                                                                saved_sentences.append(sentence)
                                                                conf_str = f", conf={segment_confidence:.2f}" if segment_confidence is not None else ""
                                                                print(f"[DB INSERT] '{sentence[:50]}...'{conf_str}" if len(sentence) > 50 else f"[DB INSERT] '{sentence}'{conf_str}", flush=True)
                                                            # Shadow row: full original sentence with CJK preserved, denied
                                                            if _cjk_shadow:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                    (timestamp, _cjk_shadow, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _cjk_shadow, _words_json, _words_source, live_session_id),
                                                                )
                                                                _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[CJK SHADOW→DENIED] '{_cjk_shadow[:40]}'", flush=True)
                                                        elif word_count < MIN_WORDS:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                (timestamp, sentence, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim, _words_json, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[SHORT→DENIED] '{sentence}' ({word_count} words)", flush=True)
                                                        elif is_dup:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                (timestamp, sentence, segment_start, segment_end, segment_confidence, needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim, _words_json, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[DUP→DENIED] '{sentence[:40]}'", flush=True)
                                                    # Run-on safety valve: flush a held fragment that never
                                                    # completes a sentence (word cap or age cap exceeded)
                                                    if pending_buffer_enabled and pending_remainder and (
                                                        len(pending_remainder.split()) > pending_max_words
                                                        or (pending_remainder_since is not None and (now - pending_remainder_since).total_seconds() > pending_max_age)
                                                    ):
                                                        remainder = pending_remainder
                                                        if pending_remainder_meta:
                                                            segment_start = pending_remainder_meta[0]
                                                            segment_end = pending_remainder_meta[1]
                                                        pending_remainder = ""
                                                        pending_remainder_since = None
                                                        pending_remainder_meta = None
                                                        print(f"[PENDING FLUSH] Run-on fragment hit cap: '{remainder[:50]}'", flush=True)
                                                    # Save substantial remainder (run-on flush, or pending buffer disabled)
                                                    if remainder:
                                                        # CJK filter on remainder
                                                        _rem_cjk_deny = False
                                                        _rem_cjk_shadow = None
                                                        if _cjk_filter_enabled:
                                                            _rem_stripped = filter_hallucinated_text(remainder, live_language)
                                                            if not _rem_stripped.strip():
                                                                _rem_cjk_deny = True
                                                                print(f"[CJK REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                            elif _rem_stripped != remainder:
                                                                _rem_cjk_shadow = remainder
                                                                remainder = _rem_stripped
                                                        _rem_is_hallucination = is_whisper_hallucination(remainder)
                                                        if _rem_is_hallucination:
                                                            print(f"[HALLUCINATION REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                        _rem_music_deny = (not _transcribe_music_enabled) and segment_speech_type == "Music"
                                                        if _rem_music_deny and not (_rem_is_hallucination or _rem_cjk_deny):
                                                            print(f"[MUSIC REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                        _rem_denied = 1 if (_rem_is_hallucination or _rem_cjk_deny or _rem_music_deny) else 0
                                                        _rem_denied_reason = ('hallucination' if _rem_is_hallucination else 'cjk' if _rem_cjk_deny else _music_deny_reason) if _rem_denied else None
                                                        _verbatim_rem = remainder
                                                        remainder = apply_profanity_filter(remainder)
                                                        rem_word_count = len(remainder.split())
                                                        rem_is_dup = is_fuzzy_duplicate(remainder, saved_sentences, fuzzy_threshold)
                                                        rem_needs_review = 1 if (segment_confidence is not None and segment_confidence < confidence_threshold) else 0
                                                        if _rem_denied or (rem_word_count >= MIN_WORDS and not rem_is_dup):
                                                            # words_json NULL: the remainder is the trailing fragment, not one of
                                                            # the attributed `sentences` (and may be carried text with no words).
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                (timestamp, remainder, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim_rem, _words_source, live_session_id, _rem_denied, _rem_denied_reason),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            if not _rem_denied:
                                                                _accepted_rows.append((_newly_inserted_ids[-1], remainder))
                                                                saved_sentences.append(remainder)
                                                                print(f"[DB INSERT REMAINDER] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DB INSERT REMAINDER] '{remainder}'", flush=True)
                                                            if _rem_cjk_shadow:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                    (timestamp, _rem_cjk_shadow, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _rem_cjk_shadow, _words_source, live_session_id),
                                                                )
                                                                _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[CJK SHADOW REMAINDER→DENIED] '{_rem_cjk_shadow[:40]}'", flush=True)
                                                        elif rem_word_count < MIN_WORDS:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                (timestamp, remainder, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim_rem, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[SHORT REMAINDER→DENIED] '{remainder}' ({rem_word_count} words)", flush=True)
                                                        elif rem_is_dup:
                                                            persistent_db_cursor.execute(
                                                                "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, needs_review, speech_type, audio_tag, music_prob, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                (timestamp, remainder, segment_start, segment_end, segment_confidence, rem_needs_review, segment_speech_type, segment_audio_tag, segment_music_prob, ts_ms, src_lang, _verbatim_rem, _words_source, live_session_id),
                                                            )
                                                            _newly_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            print(f"[DUP REMAINDER→DENIED] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DUP REMAINDER→DENIED] '{remainder}'", flush=True)
                                                    # segment_id links a transcript row to its translation
                                                    # (same row here); populate it equal to the row id.
                                                    if _newly_inserted_ids:
                                                        persistent_db_cursor.executemany(
                                                            "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                                            [(str(_rid), _rid) for _rid in _newly_inserted_ids],
                                                        )
                                                        # Link this segment's partial snapshots to the first
                                                        # final row so replay can pair partials with their final
                                                        if current_partial_row_ids:
                                                            persistent_db_cursor.executemany(
                                                                "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                                                [(str(_newly_inserted_ids[0]), _pid) for _pid in current_partial_row_ids],
                                                            )
                                                    persistent_db_conn.commit()
                                                    # Start a fresh partial run for the next segment
                                                    current_partial_row_ids = []
                                                    current_partial_seq = 0
                                                    last_partial_text = ""
                                                    # Cache Whisper translation for accepted rows only, distributing
                                                    # the whole-chunk pass-2 translation across them instead of
                                                    # duplicating it on every row. Denied rows are never emitted
                                                    # as translations, so they get none.
                                                    if _whisper_translated_text and _accepted_rows:
                                                        _target_lang = process_config.get("live_translation", {}).get("target_language", "en")
                                                        _tcache = get_translation_cache()
                                                        # Scope pass-2 text to this batch's time span (drops the
                                                        # in-progress tail's translation); fall back to full text
                                                        _wt_text = scope_whisper_translation(_pass2_timed, batch_end) or _whisper_translated_text
                                                        _parts = distribute_whisper_translation(
                                                            _wt_text, [t for _, t in _accepted_rows]
                                                        )
                                                        for (_row_id, _), _part in zip(_accepted_rows, _parts):
                                                            if not _part:
                                                                continue
                                                            _tcache.set(_row_id, "", _part, _target_lang)
                                                            # Also save to DB translated_text column
                                                            persistent_db_cursor.execute(
                                                                "UPDATE transcriptions SET translated_text = ?, translation_language = ?,"
                                                                " translation_ts_ms = ?, mt_engine = ?, mt_model = ? WHERE id = ?",
                                                                (_part, _target_lang, int(time.time() * 1000),
                                                                 MT_ENGINE_WHISPER, _whisper_mt_model, _row_id),
                                                            )
                                                        persistent_db_conn.commit()
                                                    # Track saved_sentences and database row count
                                                    # Periodically verify database row count matches
                                                    if len(saved_sentences) % 10 == 0:
                                                        db_count = persistent_db_cursor.execute("SELECT COUNT(*) FROM transcriptions WHERE COALESCE(is_final, 1) = 1").fetchone()[0]
                                                        if db_count != len(saved_sentences) + 1:  # +1 for default first entry
                                                            pass  # Row count mismatch — non-critical
                                                    # print(f"[LOOP-DEBUG] {time.strftime('%H:%M:%S')} - DB commit done", flush=True)
                                                except Exception as db_error:
                                                    print(f"[ERROR] DB save failed: {db_error}")

                                            # print(f"[FINALIZED] '{batch_text[:60]}...'" if len(batch_text) > 60 else f"[FINALIZED] '{batch_text}'")

                                    # Handle phrase completion (silence timeout)
                                    if phrase_complete:
                                        # FIX: Check if update_segments already finalized via same_output
                                        # If so, don't double-process (it's already in completed_segments)
                                        just_finalized = result.get('just_finalized_text', '')
                                        if just_finalized:
                                            finalized_segment = None  # Already handled
                                        else:
                                            # FIX: Capture pending text BEFORE force_finalize (which clears current_out)
                                            # This handles the case where same_output finalization already cleared current_out
                                            pending_text = result.get('current_text', '').strip()

                                            # Force finalize any remaining text
                                            finalized_segment = live_transcriber.force_finalize()

                                            # FIX: If force_finalize returned nothing but we had pending text, create segment from it
                                            if finalized_segment is None and pending_text:
                                                finalized_segment = {
                                                    'text': pending_text,
                                                    'start': live_transcriber.timestamp_offset,
                                                    'end': live_transcriber.timestamp_offset + chunk_duration,
                                                    'completed': True
                                                }

                                        if finalized_segment or pending_remainder:
                                            segment_text = (finalized_segment or {}).get('text', '').strip()
                                            segment_start = (finalized_segment or {}).get('start', 0)
                                            segment_end = (finalized_segment or {}).get('end', 0)
                                            _phrase_words = (finalized_segment or {}).get('words') or []
                                            # Remove overlapping prefix from previous saved text
                                            if segment_text and saved_sentences:
                                                segment_text = remove_overlapping_prefix(segment_text, saved_sentences[-1])
                                            if segment_text and pending_remainder:
                                                segment_text = remove_overlapping_prefix(segment_text, pending_remainder)
                                            # Silence boundary: no more words are coming, so flush the
                                            # held fragment together with (or instead of) the new text
                                            if pending_remainder:
                                                segment_text = (pending_remainder + " " + segment_text).strip() if segment_text else pending_remainder
                                                if pending_remainder_meta:
                                                    segment_start = pending_remainder_meta[0]
                                                    if not segment_end:
                                                        segment_end = pending_remainder_meta[1]
                                                pending_remainder = ""
                                                pending_remainder_since = None
                                                pending_remainder_meta = None

                                            if segment_text:  # Check again after overlap removal
                                                sentences, remainder = split_into_sentences(segment_text)
                                                MIN_WORDS = min_words_threshold
                                                _phrase_speech_type = finalized_audio_type(process_config, transcription_state)
                                                transcription_state['audio_type'] = _phrase_speech_type
                                                _phrase_audio_tag = transcription_state.get("audio_tag")
                                                _phrase_music_prob = transcription_state.get("music_prob")
                                                # Segment-level confidence from per-word probabilities
                                                _phrase_threshold = process_config.get("corrections", {}).get("confidence_threshold", 0.7)
                                                _phrase_probs = [w.get('probability') for w in _phrase_words if w.get('probability') is not None]
                                                _phrase_conf = (sum(_phrase_probs) / len(_phrase_probs)) if _phrase_probs else None
                                                _phrase_needs_review = 1 if (_phrase_conf is not None and _phrase_conf < _phrase_threshold) else 0
                                                # Source language (configured, or Whisper-detected when 'auto')
                                                _phrase_src = (live_language if (live_language and live_language != "auto") else (finalized_segment or {}).get('language')) or "und"
                                                # Per-word words_json attributed to each sentence by max
                                                # temporal overlap (same approach as the batch path).
                                                _phrase_stream = words_to_session_ms([finalized_segment] if finalized_segment else [])
                                                _phrase_word_groups = attribute_words_to_sentences(_phrase_stream, len(sentences))
                                                _phrase_words_source = model_type
                                                _phrase_inserted_ids = []
                                                _phrase_accepted_rows = []  # (row_id, text) of non-denied rows — whisper translation targets
                                                with _db_lock:
                                                    try:
                                                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                                        ts_ms = int(now.timestamp() * 1000)
                                                        for _sidx, sentence in enumerate(sentences):
                                                            # CJK filter: applied here so we have both versions for shadow row
                                                            _cjk_deny = False
                                                            _cjk_shadow = None
                                                            if _cjk_filter_enabled:
                                                                _cjk_stripped = filter_hallucinated_text(sentence, live_language)
                                                                if not _cjk_stripped.strip():
                                                                    _cjk_deny = True
                                                                    print(f"[CJK→DENIED] '{sentence[:40]}'", flush=True)
                                                                elif _cjk_stripped != sentence:
                                                                    _cjk_shadow = sentence
                                                                    sentence = _cjk_stripped

                                                            _is_hallucination = is_whisper_hallucination(sentence)
                                                            if _is_hallucination:
                                                                print(f"[HALLUCINATION→DENIED] '{sentence[:40]}'", flush=True)

                                                            _music_deny = (not _transcribe_music_enabled) and _phrase_speech_type == "Music"
                                                            if _music_deny and not (_is_hallucination or _cjk_deny):
                                                                print(f"[MUSIC→DENIED] '{sentence[:40]}'", flush=True)
                                                            _denied = 1 if (_is_hallucination or _cjk_deny or _music_deny) else 0
                                                            _denied_reason = ('hallucination' if _is_hallucination else 'cjk' if _cjk_deny else _music_deny_reason) if _denied else None

                                                            _verbatim = sentence
                                                            sentence = apply_profanity_filter(sentence)
                                                            word_count = len(sentence.split())
                                                            is_dup = is_fuzzy_duplicate(sentence, saved_sentences, fuzzy_threshold)
                                                            _phrase_words_json = words_json_or_none(_phrase_word_groups[_sidx] if _sidx < len(_phrase_word_groups) else None)
                                                            if _denied or (word_count >= MIN_WORDS and not is_dup):
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                    (timestamp, sentence, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim, _phrase_words_json, _phrase_words_source, live_session_id, _denied, _denied_reason),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                if not _denied:
                                                                    _phrase_accepted_rows.append((_phrase_inserted_ids[-1], sentence))
                                                                    saved_sentences.append(sentence)
                                                                    print(f"[DB INSERT PHRASE] '{sentence[:50]}...'" if len(sentence) > 50 else f"[DB INSERT PHRASE] '{sentence}'", flush=True)
                                                                if _cjk_shadow:
                                                                    persistent_db_cursor.execute(
                                                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                        (timestamp, _cjk_shadow, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _cjk_shadow, _phrase_words_json, _phrase_words_source, live_session_id),
                                                                    )
                                                                    _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                    print(f"[CJK SHADOW→DENIED] '{_cjk_shadow[:40]}'", flush=True)
                                                            elif word_count < MIN_WORDS:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                    (timestamp, sentence, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim, _phrase_words_json, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[SHORT→DENIED] '{sentence}' ({word_count} words)", flush=True)
                                                            elif is_dup:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_json, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                    (timestamp, sentence, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim, _phrase_words_json, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[DUP→DENIED] '{sentence[:40]}'", flush=True)
                                                        # Also save substantial remainder from phrase_complete
                                                        if remainder:
                                                            _rem_cjk_deny = False
                                                            _rem_cjk_shadow = None
                                                            if _cjk_filter_enabled:
                                                                _rem_stripped = filter_hallucinated_text(remainder, live_language)
                                                                if not _rem_stripped.strip():
                                                                    _rem_cjk_deny = True
                                                                    print(f"[CJK REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                                elif _rem_stripped != remainder:
                                                                    _rem_cjk_shadow = remainder
                                                                    remainder = _rem_stripped
                                                            _rem_is_hallucination = is_whisper_hallucination(remainder)
                                                            if _rem_is_hallucination:
                                                                print(f"[HALLUCINATION REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                            _rem_music_deny = (not _transcribe_music_enabled) and _phrase_speech_type == "Music"
                                                            if _rem_music_deny and not (_rem_is_hallucination or _rem_cjk_deny):
                                                                print(f"[MUSIC REMAINDER→DENIED] '{remainder[:40]}'", flush=True)
                                                            _rem_denied = 1 if (_rem_is_hallucination or _rem_cjk_deny or _rem_music_deny) else 0
                                                            _rem_denied_reason = ('hallucination' if _rem_is_hallucination else 'cjk' if _rem_cjk_deny else _music_deny_reason) if _rem_denied else None
                                                            _verbatim_rem = remainder
                                                            remainder = apply_profanity_filter(remainder)
                                                            rem_word_count = len(remainder.split())
                                                            rem_is_dup = is_fuzzy_duplicate(remainder, saved_sentences, fuzzy_threshold)
                                                            if _rem_denied or (rem_word_count >= MIN_WORDS and not rem_is_dup):
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                                                    (timestamp, remainder, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim_rem, _phrase_words_source, live_session_id, _rem_denied, _rem_denied_reason),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                if not _rem_denied:
                                                                    _phrase_accepted_rows.append((_phrase_inserted_ids[-1], remainder))
                                                                    saved_sentences.append(remainder)
                                                                if _rem_cjk_shadow:
                                                                    persistent_db_cursor.execute(
                                                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                                                        (timestamp, _rem_cjk_shadow, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _rem_cjk_shadow, _phrase_words_source, live_session_id),
                                                                    )
                                                                    _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                            elif rem_word_count < MIN_WORDS:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'short')",
                                                                    (timestamp, remainder, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim_rem, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[SHORT REMAINDER→DENIED] '{remainder}' ({rem_word_count} words)", flush=True)
                                                            elif rem_is_dup:
                                                                persistent_db_cursor.execute(
                                                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, speech_type, audio_tag, music_prob, confidence, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'dup')",
                                                                    (timestamp, remainder, segment_start, segment_end, _phrase_speech_type, _phrase_audio_tag, _phrase_music_prob, _phrase_conf, _phrase_needs_review, ts_ms, _phrase_src, _verbatim_rem, _phrase_words_source, live_session_id),
                                                                )
                                                                _phrase_inserted_ids.append(persistent_db_cursor.lastrowid)
                                                                print(f"[DUP REMAINDER→DENIED] '{remainder[:50]}...'" if len(remainder) > 50 else f"[DUP REMAINDER→DENIED] '{remainder}'", flush=True)
                                                        if _phrase_inserted_ids:
                                                            persistent_db_cursor.executemany(
                                                                "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                                                [(str(_rid), _rid) for _rid in _phrase_inserted_ids],
                                                            )
                                                            # Link this segment's partial snapshots to the first
                                                            # final row so replay can pair partials with their final
                                                            if current_partial_row_ids:
                                                                persistent_db_cursor.executemany(
                                                                    "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                                                    [(str(_phrase_inserted_ids[0]), _pid) for _pid in current_partial_row_ids],
                                                                )
                                                        persistent_db_conn.commit()
                                                        # Start a fresh partial run for the next segment
                                                        current_partial_row_ids = []
                                                        current_partial_seq = 0
                                                        last_partial_text = ""
                                                        # Cache Whisper translation for accepted phrase rows only,
                                                        # distributing the whole-chunk pass-2 translation across
                                                        # them instead of duplicating it on every row.
                                                        if _whisper_translated_text and _phrase_accepted_rows:
                                                            _target_lang = process_config.get("live_translation", {}).get("target_language", "en")
                                                            _tcache = get_translation_cache()
                                                            # Scope pass-2 text to the finalized span; fall back to full text
                                                            _wt_text = scope_whisper_translation(_pass2_timed, segment_end) or _whisper_translated_text
                                                            _parts = distribute_whisper_translation(
                                                                _wt_text, [t for _, t in _phrase_accepted_rows]
                                                            )
                                                            for (_row_id, _), _part in zip(_phrase_accepted_rows, _parts):
                                                                if not _part:
                                                                    continue
                                                                _tcache.set(_row_id, "", _part, _target_lang)
                                                                persistent_db_cursor.execute(
                                                                    "UPDATE transcriptions SET translated_text = ?, translation_language = ?,"
                                                                    " translation_ts_ms = ?, mt_engine = ?, mt_model = ? WHERE id = ?",
                                                                    (_part, _target_lang, int(time.time() * 1000),
                                                                     MT_ENGINE_WHISPER, _whisper_mt_model, _row_id),
                                                                )
                                                            persistent_db_conn.commit()
                                                    except Exception as db_error:
                                                        print(f"[ERROR] phrase_complete DB save failed: {db_error}")

                                                # print(f"[PHRASE_COMPLETE] Finalized: '{segment_text[:40]}...'")

                                        # Clear live preview (single update: readers in the
                                        # Flask process never see a half-written generation)
                                        transcription_state.update({
                                            "live_text": "",
                                            "live_start": 0,
                                            "live_end": 0,
                                        })
                                        # Phrase finalized -> the live hypothesis restarts.
                                        _hyp_buffer.reset()
                                        # Reset phrase_time so next silence doesn't immediately re-trigger
                                        phrase_time = None

                                    # Update live preview with current incomplete text only
                                    # (finalized segments are already shown separately from the database)
                                    # Prepend any held fragment so it stays visible until its sentence completes
                                    _raw_current = result.get('current_text', '')
                                    # LocalAgreement: reveal only the stabilized prefix of the current
                                    # segment so the live line stops rewriting itself. Held-back tail
                                    # words appear when the phrase finalizes (via the DB path). The word
                                    # confidences are truncated to match the shown prefix.
                                    _stabilize = process_config.get("audio", {}).get("stabilize_live_text", True)
                                    _stable_current = _hyp_buffer.stabilize(_raw_current) if _stabilize else _raw_current
                                    _stable_word_count = len(_stable_current.split())
                                    current_text = _stable_current
                                    if pending_remainder:
                                        current_text = (pending_remainder + " " + current_text).strip()
                                    if current_text:
                                        # Single update so text/timing/confidence stay consistent
                                        # for readers in the Flask process
                                        _live_update = {
                                            "live_text": current_text,
                                            "live_start": live_transcriber.timestamp_offset,
                                            "live_end": live_transcriber.timestamp_offset + chunk_duration,
                                        }
                                        if hasattr(live_transcriber, '_last_seg_confidence'):
                                            _conf_words = live_transcriber._last_seg_confidence.get('words', [])
                                            # Keep confidences aligned to the (possibly truncated) shown words
                                            _live_update["live_word_confidences"] = (
                                                _conf_words[:_stable_word_count] if _stabilize else _conf_words)
                                        transcription_state.update(_live_update)

                                        # Throttled partial snapshot (is_final=0): the row's own
                                        # ts_ms is the arrival time — what makes replay faithful
                                        if record_partials and current_text != last_partial_text:
                                            _p_now_ms = int(time.time() * 1000)
                                            if _p_now_ms - last_partial_write_ms >= partials_min_interval_ms:
                                                try:
                                                    # Per-word timings are deliberately not stored on a
                                                    # partial: nothing reads words_json off an is_final=0
                                                    # row — not this server (every reference is an INSERT,
                                                    # there is no SELECT) and not the ChurchPresenter BLE
                                                    # replay, which reads id/ts_ms/text/translated_text/
                                                    # speech_type/segment_id/session_id/start_time/
                                                    # is_final/denied and nothing else. Each snapshot
                                                    # re-serialised the whole growing word list and the
                                                    # final row supersedes all of them: measured on a real
                                                    # 167-minute service that was 4.18 MB of a 9.1 MB
                                                    # database, 46% of the file, for data no reader has.
                                                    # Set database.partials_store_words to bring it back.
                                                    _p_words = (_live_update.get("live_word_confidences")
                                                                if partials_store_words else None)
                                                    with _db_lock:
                                                        persistent_db_cursor.execute(
                                                            "INSERT INTO transcriptions (timestamp, text, start_time, end_time, ts_ms, original_text, words_json, words_source, session_id, is_final, partial_seq, denied) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0)",
                                                            (
                                                                datetime.now(configured_timezone).strftime("%Y-%m-%d %H:%M:%S"),
                                                                current_text,
                                                                _live_update["live_start"],
                                                                _live_update["live_end"],
                                                                _p_now_ms,
                                                                current_text,
                                                                json.dumps(_p_words) if _p_words else None,
                                                                model_type,
                                                                live_session_id,
                                                                current_partial_seq,
                                                            ),
                                                        )
                                                        current_partial_row_ids.append(persistent_db_cursor.lastrowid)
                                                        persistent_db_conn.commit()
                                                    current_partial_seq += 1
                                                    last_partial_write_ms = _p_now_ms
                                                    last_partial_text = current_text
                                                except Exception as _p_err:
                                                    print(f"[PARTIAL] Snapshot write failed: {_p_err}", flush=True)

                                except Exception as transcribe_error:
                                    print(f"[ERROR] Transcription failed: {transcribe_error}")
                                    import traceback
                                    traceback.print_exc()

                                # Infinite loops are bad for processors, must sleep.
                                sleep(0.25)
                        except KeyboardInterrupt:
                            break

                    # Clean up resources before exiting
                    print("\nCleaning up resources...")

                    # Flush any held sentence fragment so it isn't lost on stop
                    if pending_remainder:
                        try:
                            # CJK filter on flush remainder
                            _flush_cjk_deny = False
                            _flush_cjk_shadow = None
                            try:
                                _flush_cjk_enabled = config.get("hallucination_filter", {}).get("cjk_filter_enabled", True)
                                _flush_ll = live_language
                            except NameError:
                                _flush_cjk_enabled = True
                                _flush_ll = None
                            if _flush_cjk_enabled:
                                _flush_stripped = filter_hallucinated_text(pending_remainder, _flush_ll)
                                if not _flush_stripped.strip():
                                    _flush_cjk_deny = True
                                    print(f"[CJK STOP-FLUSH→DENIED] '{pending_remainder[:40]}'", flush=True)
                                elif _flush_stripped != pending_remainder:
                                    _flush_cjk_shadow = pending_remainder
                                    pending_remainder = _flush_stripped
                            _flush_is_hallucination = is_whisper_hallucination(pending_remainder)
                            if _flush_is_hallucination:
                                print(f"[HALLUCINATION STOP-FLUSH→DENIED] '{pending_remainder[:40]}'", flush=True)
                            _flush_denied = 1 if (_flush_is_hallucination or _flush_cjk_deny) else 0
                            _flush_denied_reason = ('hallucination' if _flush_is_hallucination else 'cjk') if _flush_denied else None
                            _verbatim_flush = pending_remainder
                            _flush_text = apply_profanity_filter(pending_remainder)
                            _flush_word_count = len(_flush_text.split())
                            _flush_is_dup = is_fuzzy_duplicate(_flush_text, saved_sentences, fuzzy_threshold)
                            _flush_word_ok = _flush_word_count >= min_words_threshold and not _flush_is_dup
                            _flush_start, _flush_end, _flush_conf = pending_remainder_meta if pending_remainder_meta else (0, 0, None)
                            _flush_now = datetime.now(configured_timezone)
                            _flush_ts_ms = int(_flush_now.timestamp() * 1000)
                            _flush_speech_type = finalized_audio_type(process_config, transcription_state)
                            _flush_audio_tag = transcription_state.get("audio_tag")
                            _flush_music_prob = transcription_state.get("music_prob")
                            _flush_threshold = process_config.get("corrections", {}).get("confidence_threshold", 0.7)
                            _flush_needs_review = 1 if (_flush_conf is not None and _flush_conf < _flush_threshold) else 0
                            try:
                                _flush_ll = live_language
                            except NameError:
                                _flush_ll = None
                            _flush_src = _flush_ll if (_flush_ll and _flush_ll != "auto") else "und"
                            try:
                                _flush_words_source = model_type
                            except NameError:
                                _flush_words_source = None
                            _flush_std_cfg = process_config.get("speech_type_detection", {})
                            if not _flush_denied and _flush_speech_type == "Music" and not _flush_std_cfg.get("transcribe_detected_music", False):
                                _flush_denied = 1
                                _flush_denied_reason = f"music:{_flush_std_cfg.get('music_prob_threshold', 0.5):g}"
                                print(f"[MUSIC STOP-FLUSH→DENIED] '{_flush_text[:40]}'", flush=True)
                            if not _flush_denied and not _flush_word_ok:
                                _flush_denied = 1
                                _flush_denied_reason = 'short' if _flush_word_count < min_words_threshold else 'dup'
                            with _db_lock:
                                # words_json NULL: carried fragment, no per-word data retained.
                                persistent_db_cursor.execute(
                                    "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, speech_type, audio_tag, music_prob, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                                    (_flush_now.strftime("%Y-%m-%d %H:%M:%S"), _flush_text, _flush_start, _flush_end, _flush_conf, _flush_speech_type, _flush_audio_tag, _flush_music_prob, _flush_needs_review, _flush_ts_ms, _flush_src, _verbatim_flush, _flush_words_source, live_session_id, _flush_denied, _flush_denied_reason),
                                )
                                _flush_row_id = persistent_db_cursor.lastrowid
                                persistent_db_cursor.execute(
                                    "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                    (str(_flush_row_id), _flush_row_id),
                                )
                                # Link any remaining partial snapshots to the flushed row
                                if current_partial_row_ids:
                                    persistent_db_cursor.executemany(
                                        "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                        [(str(_flush_row_id), _pid) for _pid in current_partial_row_ids],
                                    )
                                    current_partial_row_ids = []
                                if _flush_cjk_shadow:
                                    persistent_db_cursor.execute(
                                        "INSERT INTO transcriptions (timestamp, text, start_time, end_time, confidence, speech_type, audio_tag, music_prob, needs_review, ts_ms, source_language, original_text, words_source, session_id, is_final, denied, denied_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 'cjk_shadow')",
                                        (_flush_now.strftime("%Y-%m-%d %H:%M:%S"), _flush_cjk_shadow, _flush_start, _flush_end, _flush_conf, _flush_speech_type, _flush_audio_tag, _flush_music_prob, _flush_needs_review, _flush_ts_ms, _flush_src, _flush_cjk_shadow, _flush_words_source, live_session_id),
                                    )
                                    _shadow_id = persistent_db_cursor.lastrowid
                                    persistent_db_cursor.execute(
                                        "UPDATE transcriptions SET segment_id = ? WHERE id = ?",
                                        (str(_shadow_id), _shadow_id),
                                    )
                                    print(f"[CJK SHADOW STOP-FLUSH→DENIED] '{_flush_cjk_shadow[:40]}'", flush=True)
                                persistent_db_conn.commit()
                            if not _flush_denied:
                                saved_sentences.append(_flush_text)
                                print(f"[DB INSERT STOP-FLUSH] '{_flush_text[:50]}'", flush=True)
                            else:
                                print(f"[STOP-FLUSH {_flush_denied_reason.upper()}→DENIED] '{_flush_text[:50]}'", flush=True)
                        except Exception as _flush_err:
                            print(f"[WARNING] Failed to flush pending fragment on stop: {_flush_err}")
                        pending_remainder = ""
                        pending_remainder_since = None
                        pending_remainder_meta = None

                    # Stop audio source FIRST to release audio device
                    if source:
                        try:
                            print("[CLEANUP] Stopping audio source...")
                            source.stop()
                            print("[CLEANUP] OK: Audio source stopped and ffmpeg terminated")
                        except Exception as e:
                            print(f"[CLEANUP] WARNING: Error stopping audio source: {e}")

                    # Fix WAV header for session audio file (update file size in header)
                    if session_audio_file and session_audio_written:
                        try:
                            # Read all data
                            with open(session_audio_file, "rb") as f:
                                data = f.read()

                            # Recreate proper WAV file with correct header
                            audio_data = sr.AudioData(
                                data[44:], source.SAMPLE_RATE, source.SAMPLE_WIDTH
                            )  # Skip old header
                            correct_wav = audio_data.get_wav_data()

                            with open(session_audio_file, "wb") as f:
                                f.write(correct_wav)

                            print(
                                f"[BACKUP] Full session audio finalized: {session_audio_file}"
                            )
                        except Exception as e:
                            print(
                                f"[WARNING] Failed to finalize session audio header: {e}"
                            )

                    print("[DB-CLEANUP] Starting database cleanup...", flush=True)
                    try:
                        if persistent_db_conn:
                            # Save db_path for SRT conversion and WAL/SHM cleanup later
                            saved_db_path = db_path
                            print(f"[DB-CLEANUP] saved_db_path = {saved_db_path}", flush=True)
                            print(f"[DB-CLEANUP] global db_name = {db_name}", flush=True)

                            # CRITICAL: Clear db_name from state BEFORE cleanup to prevent
                            # web server thread (emit_new_entries) from opening new connections
                            with _transcription_state_lock:
                                transcription_state["db_name"] = None
                                transcription_state["session_id"] = None
                            print("[DB-CLEANUP] Cleared db_name from state (prevents new connections)", flush=True)

                            # Wait for any in-flight emit_new_entries() iterations to complete
                            # emit_new_entries runs every 0.5 seconds, so 1 second should be safe
                            sleep(1)
                            print("[DB-CLEANUP] Waited for web server thread to release connections", flush=True)

                            # Checkpoint WAL to flush all changes to main database file
                            # TRUNCATE mode removes WAL file after checkpoint
                            try:
                                result = persistent_db_cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                                checkpoint_result = result.fetchone()
                                print(f"[DB-CLEANUP] WAL checkpoint completed: {checkpoint_result}", flush=True)
                            except Exception as checkpoint_error:
                                print(f"[DB-CLEANUP] WAL checkpoint failed: {checkpoint_error}", flush=True)

                            persistent_db_conn.close()
                            print("[DB-CLEANUP] Database connection closed", flush=True)

                            # NOTE: WAL/SHM file deletion is deferred until AFTER SRT conversion
                            # because SRT conversion opens a new connection which recreates these files
                            # The actual deletion happens after SRT conversion outside of main()
                        else:
                            print("[DB-CLEANUP] persistent_db_conn is None, skipping cleanup", flush=True)
                    except Exception as e:
                        print(f"[DB-CLEANUP] Error closing DB connection: {e}", flush=True)
                        import traceback
                        traceback.print_exc()

                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            print("[OK] Temp file removed")
                    except Exception as e:
                        print(f"[WARNING] Error removing temp file: {e}")

                    # Clear model references BEFORE cleanup to allow garbage collection
                    audio_model = None
                    processor = None
                    model_type = None
                    vad_model = None

                    # Clean up models
                    ModelFactory.cleanup_models()

            main()

            # Ensure audio source is cleaned up after main() exits
            if source:
                try:
                    print("[CLEANUP] Ensuring audio source cleanup after main() exit...")
                    source.stop()
                    print("[CLEANUP] OK: Audio source cleanup complete")
                except Exception as e:
                    print(f"[CLEANUP] Error in post-main cleanup: {e}")
                finally:
                    source = None

            # Reset model references so next Start can reinitialize
            audio_model = None
            processor = None
            model_type = None
            vad_model = None

            # After main() returns, check if we're not running anymore and update status
            if not is_running:
                # Reset database initialization flag for next session
                global db_initialized
                db_initialized = False

                # Use global db_name for SRT conversion (transcription_state["db_name"] was cleared earlier)
                session_db_name = db_name

                with _transcription_state_lock:
                    transcription_state["running"] = False
                    transcription_state["status"] = "stopped"
                    transcription_state["message"] = "Transcription stopped"
                    transcription_state["error"] = None
                    # Drop file-playback markers so the live-settings trackbar
                    # doesn't linger after an end-of-file (or any) stop.
                    transcription_state["is_file_playback"] = False
                    transcription_state["playback_source"] = None
                    transcription_state["playback_duration"] = None
                    # db_name already cleared in cleanup code above
                print("[INFO] Transcription stopped successfully")

                # Convert database to SRT before file mover runs
                if session_db_name:
                    # Check if SRT generation is enabled (reload config for fresh settings)
                    fresh_config = load_config()
                    srt_enabled = fresh_config.get("database", {}).get("srt_enabled", True)
                    html_enabled = fresh_config.get("database", {}).get("html_enabled", True)
                    if srt_enabled:
                        try:
                            print(f"[SRT] Converting session database to SRT: {session_db_name}")
                            srt_result = convert_db_to_srt(session_db_name)
                            if srt_result:
                                print("[SRT] Successfully created SRT file")
                            else:
                                print("[SRT] No SRT file created (no valid entries or error)")
                        except Exception as e:
                            print(f"[SRT] Error during SRT conversion: {e}")
                    else:
                        print("[SRT] SRT generation disabled in settings")
                        # Generate HTML separately if SRT is disabled but HTML is enabled
                        if html_enabled:
                            try:
                                print(f"[HTML] Generating HTML file: {session_db_name}")
                                convert_db_to_html(session_db_name)
                            except Exception as e:
                                print(f"[HTML] Error during HTML generation: {e}")

                    # Generate translation SRT if enabled
                    trans_srt_enabled = fresh_config.get("live_translation", {}).get("srt_enabled", True)
                    if trans_srt_enabled and fresh_config.get("live_translation", {}).get("enabled", False):
                        try:
                            print(f"[SRT-TRANSLATION] Converting translations to SRT: {session_db_name}")
                            trans_srt_result = convert_db_to_translation_srt(session_db_name)
                            if trans_srt_result:
                                print("[SRT-TRANSLATION] Successfully created translation SRT file")
                            else:
                                print("[SRT-TRANSLATION] No translation SRT created (no translated entries)")
                        except Exception as e:
                            print(f"[SRT-TRANSLATION] Error: {e}")

                    # NOW retire the WAL/SHM sidecars, after SRT conversion:
                    # that opens its own connection, which recreates them.
                    #
                    # This used to unlink both files directly. It was safe here
                    # because a TRUNCATE checkpoint had already run, but it is
                    # the wrong pattern to have lying around — deleting a WAL
                    # that has not been checkpointed discards committed rows.
                    # The shared helper folds the WAL in first and lets SQLite
                    # remove it, so no caller can copy the unsafe shape.
                    print("[WAL-CLEANUP] Retiring WAL/SHM files after SRT conversion...", flush=True)
                    if _db_checkpoint_and_release(session_db_name):
                        print("[WAL-CLEANUP] Sidecars retired", flush=True)
                    else:
                        print("[WAL-CLEANUP] Sidecars still present; the startup "
                              "sweep will retry", flush=True)

                # File mover is THE VERY LAST operation after everything is fully stopped
                # Wait 10 seconds to ensure all file handles are released
                try:
                    # Reload config to get latest settings (supports hot-reload)
                    current_config = load_config()
                    mover_config = current_config.get("file_manager", {}).get("file_mover", {})
                    if mover_config.get("move_on_transcription_stop", True):
                        print("[FILE MOVER] Waiting 10 seconds for all file handles to close...")
                        set_file_mover_running("auto")
                        sleep(10)
                        print("[FILE MOVER] Executing file move after final cleanup...")
                        result = execute_file_move_now(lambda cfg=current_config: cfg, APP_DIR)
                        set_file_mover_result("auto", result)
                        if result['success']:
                            print(f"[FILE MOVER] OK: Moved {result['moved']} files")
                            if result['failed'] > 0:
                                print(f"[FILE MOVER] ! {result['failed']} files failed")
                        else:
                            print(f"[FILE MOVER] FAIL: Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    print(f"[FILE MOVER] Error executing file mover: {e}")

                # Final safety net: make the whole DB/backup folder readable by all
                # users — covering every file produced during stop cleanup (SRT/HTML
                # exports, the checkpointed DB, audio) and any pre-existing files.
                try:
                    make_tree_world_readable(BACKUP_DIR)
                    _custom_db_base = (load_config().get("database", {}).get("path", "") or "").strip()
                    if _custom_db_base and os.path.abspath(_custom_db_base) != os.path.abspath(BACKUP_DIR):
                        make_tree_world_readable(_custom_db_base)
                    print("[PERMS] DB/backup folder made world-readable", flush=True)
                except Exception as _perm_err:
                    print(f"[PERMS] Failed to update DB folder permissions: {_perm_err}", flush=True)
    except KeyboardInterrupt:
        print("Thread 1 received KeyboardInterrupt")
        os._exit(0)


# How long the startup sweep waits before its one retry. Longer than the
# min_age guard below it, so a session interrupted by this very restart is old
# enough to retire by the time the second pass runs.
SIDECAR_SWEEP_MIN_AGE_S = 120
SIDECAR_SWEEP_RETRY_S = 300


def _sidecar_sweep_dirs():
    """Directories a session database can live in: the backup tree and any
    configured custom database path."""
    dirs = [BACKUP_DIR]
    custom = (config.get("database", {}).get("path") or "").strip()
    if custom:
        dirs.append(custom)
    return [d for d in dirs if d and os.path.isdir(d)]


def sweep_db_sidecars():
    """Retire -wal/-shm files left beside finished session databases.

    They survive when the *process* is stopped mid-session: the shutdown handler
    terminates the worker before it reaches its own end-of-session checkpoint,
    so the sidecars are never retired and nothing sweeps them afterwards. A
    service restart (an auto-update, say) landing mid-session is the usual cause,
    which is why it happens only sometimes.

    Never touches the live session, and never deletes a WAL — see
    stt/db_maintenance.py for why that distinction matters.
    """
    try:
        active = None
        try:
            active = transcription_state.get("db_name") if transcription_state else None
        except Exception:
            pass  # a torn-down state proxy is not a reason to skip housekeeping

        result = _db_sweep_sidecars(
            _sidecar_sweep_dirs(),
            skip_paths=[p for p in (active,) if p],
            # A session that stopped moments ago may still be having its SRT
            # written by another thread; leave it for the next sweep.
            min_age_s=SIDECAR_SWEEP_MIN_AGE_S,
        )
        if result["scanned"]:
            print(f"[WAL-SWEEP] {result['cleaned']} cleaned, "
                  f"{result['skipped_active']} active, {result['skipped_recent']} recent, "
                  f"{result['failed']} failed", flush=True)
        return result
    except Exception as e:
        print(f"[WAL-SWEEP] Sweep failed: {e}", flush=True)
        return {"scanned": 0, "cleaned": 0, "skipped_active": 0,
                "skipped_recent": 0, "failed": 0, "errors": [str(e)]}


def _sweep_db_sidecars_startup():
    """Sweep at startup, then once more after the age guard can have expired.

    A restart that interrupts a session leaves that session's sidecars behind,
    and the immediate sweep deliberately skips it as "too recent" — it may still
    be having its SRT written. Without a second pass those files would wait for
    the *next* restart, which on a machine that runs for days is no cure at all.
    One delayed retry closes that without a scheduler.
    """
    sweep_db_sidecars()
    sleep(SIDECAR_SWEEP_RETRY_S)
    sweep_db_sidecars()


def cleanup_old_partials():
    """Retention valve for partial snapshots: delete is_final=0 rows from session
    databases older than database.partials_retention_days (0 = keep forever).
    Final rows are never touched, so long-term archives shrink while recent
    sessions stay replay-complete."""
    try:
        db_cfg = load_config().get("database", {})
        retention_days = db_cfg.get("partials_retention_days", 0)
        if not retention_days or retention_days <= 0:
            return
        base_dirs = [BACKUP_DIR]
        custom_base = (db_cfg.get("path", "") or "").strip()
        if custom_base and os.path.abspath(custom_base) != os.path.abspath(BACKUP_DIR):
            base_dirs.append(custom_base)
        cutoff = time.time() - retention_days * 86400
        cleaned = 0
        for base in base_dirs:
            if not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                for fname in files:
                    if not fname.endswith(".db"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        if os.path.getmtime(fpath) >= cutoff:
                            continue
                        with sqlite3.connect(fpath) as _conn:
                            _cur = _conn.execute(
                                "DELETE FROM transcriptions WHERE COALESCE(is_final, 1) = 0"
                            )
                            if _cur.rowcount > 0:
                                _conn.commit()
                                cleaned += _cur.rowcount
                    except Exception:
                        continue  # Locked/foreign .db files are not ours to touch
        if cleaned:
            print(f"[PARTIALS] Retention cleanup removed {cleaned} partial rows from sessions older than {retention_days} days", flush=True)
    except Exception as _cleanup_err:
        print(f"[PARTIALS] Retention cleanup failed: {_cleanup_err}", flush=True)


def thread2_function():
    try:
        # Get web server config
        web_config = config.get("web_server", {})
        host = web_config.get("host", "0.0.0.0")
        port = web_config.get("port", 80)

        print(f"Starting web server on {host}:{port}")

        # Housekeeping off the boot path: strip expired partial rows from old sessions
        threading.Thread(target=cleanup_old_partials, daemon=True).start()

        # Retire sidecars the previous run could not: a process stopped
        # mid-session never reaches the worker's end-of-session checkpoint.
        threading.Thread(target=_sweep_db_sidecars_startup, daemon=True).start()

        # Start the background task for emitting transcriptions
        socketio.start_background_task(emit_new_entries)

        # Start the background task for emitting translations
        socketio.start_background_task(emit_translated_entries)

        # Start audio streaming background tasks
        socketio.start_background_task(emit_audio_stream)
        socketio.start_background_task(emit_tts_audio)

        # Use socketio.run() instead of app.run() for proper Socket.IO support.
        # Retry transient bind failures: a just-restarted process can race a
        # dying predecessor (or a child that inherited the bound socket) still
        # holding the port. Werkzeug turns EADDRINUSE into SystemExit, which
        # would otherwise kill only this thread and leave a zombie server that
        # systemd considers alive but that serves no web UI.
        attempts = 5
        for attempt in range(1, attempts + 1):
            try:
                socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
                return  # clean server shutdown
            except (SystemExit, OSError) as e:
                if attempt == attempts:
                    print(f"[FATAL] Web server could not start after {attempts} attempts: {e!r}")
                    print("[FATAL] Exiting so the supervisor can restart the process...")
                    sys.stdout.flush()  # os._exit skips stdio flushing
                    sys.stderr.flush()
                    os._exit(1)
                print(f"[WEB] Server failed to start ({e!r}); retrying in 3s ({attempt}/{attempts})...")
                sleep(3)
    except KeyboardInterrupt:
        print("Thread 2 received KeyboardInterrupt")
        os._exit(0)


def signal_handler(signum, frame):
    print("\n[SHUTDOWN] Interrupt signal received, stopping threads...")

    # Stop emit threads from touching the Manager proxy before we tear it down.
    _server_shutting_down.set()

    # Terminate the transcription process (multiprocessing.Process)
    try:
        if "thread1" in globals() and globals()["thread1"].is_alive():
            print("[SHUTDOWN] Terminating transcription process...")
            globals()["thread1"].terminate()
            globals()["thread1"].join(timeout=3)
            if globals()["thread1"].is_alive():
                print("[SHUTDOWN] Force killing transcription process...")
                globals()["thread1"].kill()
                globals()["thread1"].join(timeout=1)
            print("[SHUTDOWN] Transcription process terminated")
        else:
            print("[SHUTDOWN] Transcription process already stopped")
    except Exception as e:
        print(f"[SHUTDOWN] Error terminating transcription process: {e}")

    # Stop the web server thread
    try:
        if "thread2" in globals() and globals()["thread2"].is_alive():
            print("[SHUTDOWN] Stopping web server thread...")
            # Thread will stop when main exits
        else:
            print("[SHUTDOWN] Web server thread already stopped")
    except Exception as e:
        print(f"[SHUTDOWN] Error checking web server thread: {e}")

    print("[SHUTDOWN] Cleanup complete, exiting...")
    os._exit(0)


# Loopback port used only as a single-instance lock for the server process
# (never serves traffic). Distinct from the watchdog's own lock port (57337 in
# stt/watchdog.py) so the watchdog and its managed server don't collide.
_SERVER_LOCK_PORT = 57338
_server_lock_socket = None  # module global: keeps the bound socket alive for the process lifetime


def acquire_server_lock():
    """Single-instance guard for the server: bind a loopback socket.

    The bind is the lock — the OS releases it on any exit (including kill -9),
    so there is no stale-lock file to clean up. If the port is already bound
    another STT server owns this machine, so exit cleanly with status 0: the
    watchdog treats a 0 exit as an intentional stop and won't relaunch (see
    CrashRecoveryThread in stt/watchdog.py), so a redundant launch disappears
    quietly instead of thrashing against the live server's port 8080 bind.
    """
    global _server_lock_socket
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind(("127.0.0.1", _SERVER_LOCK_PORT))
    except OSError:
        print(f"[FATAL] Another STT server is already running (lock {_SERVER_LOCK_PORT}); exiting.")
        sys.stdout.flush()
        sys.exit(0)
    _server_lock_socket = sock  # keep reference alive; OS releases on process exit


def _is_watchdog_managed():
    """True when launched under the watchdog, which owns its own updater.

    The watchdog sets STT_MANAGED=1 (and STT_DATA_DIR); either signals managed
    mode, so a direct-run git-pull doesn't double up with the watchdog's updates.
    """
    return bool(os.environ.get("STT_MANAGED") or os.environ.get("STT_DATA_DIR"))


def _self_update_enabled():
    return bool(config.get("auto_update", {}).get("enabled", True))


def _self_update_allow_reset():
    """Recover a diverged checkout (force-pushed upstream) by resetting onto it."""
    return bool(config.get("auto_update", {}).get("reset_on_diverged_upstream", True))


def _restart_for_update():
    """Restart the server process to load the pulled update.

    Delegates to perform_server_restart(): under systemd that's an atomic
    `systemctl restart` whose stop script reaps every child; the execv
    fallback tears down multiprocessing children first, since any child
    forked after the web server bound its port keeps that socket alive
    across an in-place execv and the re-exec'd server dies with EADDRINUSE.
    """
    perform_server_restart()


def _run_startup_self_update():
    """One-shot git self-update before any worker/threads start (direct runs only)."""
    if _is_watchdog_managed() or not _self_update_enabled():
        return
    try:
        from stt.self_update import git_self_update, restart_via_execv
        updated, reason = git_self_update(BUNDLE_DIR, allow_reset=_self_update_allow_reset())
        if updated:
            print("[AUTO-UPDATE] Update pulled at startup; restarting...")
            restart_via_execv()  # no worker yet -> clean re-exec
        else:
            print(f"[AUTO-UPDATE] Startup update check: {reason}")
    except Exception as e:
        print(f"[AUTO-UPDATE] Startup self-update error: {e}")


def _self_update_loop():
    """Nightly git self-update at a fixed hour; only applies while idle.

    Runs at auto_update.update_hour (local, default 1 = 1am) rather than a flat
    interval, so a live box updates during off-hours instead of restarting every
    hour. The one-shot startup check still catches a box up at boot."""
    import time
    from datetime import datetime
    from stt.self_update import git_self_update, seconds_until_hour
    try:
        update_hour = int(config.get("auto_update", {}).get("update_hour", 1))
    except (TypeError, ValueError):
        update_hour = 1
    while True:
        delay = seconds_until_hour(datetime.now(), update_hour)
        print(f"[AUTO-UPDATE] Next update check at {update_hour:02d}:00 "
              f"(in {delay / 3600:.1f}h)")
        time.sleep(delay)
        if _server_shutting_down.is_set():
            return
        try:
            if _ts_get("running"):
                continue  # idle-gate: never restart mid-transcription
            updated, reason = git_self_update(BUNDLE_DIR, allow_reset=_self_update_allow_reset())
            if not updated and reason == "not-fast-forwardable":
                # Otherwise the only trace is one startup line, and a box can sit
                # frozen on old code for weeks with nothing in the log saying why.
                print("[AUTO-UPDATE] Checkout diverged from upstream; not updating. Set "
                      "auto_update.reset_on_diverged_upstream, or reset the checkout by hand.")
            if updated:
                # git_self_update does a network pull that can take many
                # seconds; re-check the idle-gate afterwards so a session that
                # started during the pull isn't force-restarted mid-service.
                # The pulled code stays on disk and applies on the next idle tick.
                if _ts_get("running"):
                    print("[AUTO-UPDATE] Update pulled but a session started during the pull; deferring restart")
                    continue
                print("[AUTO-UPDATE] Update pulled; restarting to apply...")
                _restart_for_update()
        except Exception as e:
            print(f"[AUTO-UPDATE] Periodic self-update error: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    install_crash_diagnostics("main")
    # Single-instance guard: bail out immediately if another server owns this
    # machine, before spawning the worker/threads or doing any startup work.
    # This makes "only one server at a time" hold regardless of launcher
    # (watchdog child, bare service, dev run, or a post-install race).
    acquire_server_lock()
    # Bound server.log at startup (small breadcrumb log; rotated across launches)
    try:
        _srv_log = os.path.join(APP_DIR, "server.log")
        if os.path.exists(_srv_log) and os.path.getsize(_srv_log) > 5_000_000:
            os.replace(_srv_log, _srv_log + ".1")
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

    # Windows cannot deliver SIGTERM across processes, so a watchdog-managed
    # worker used to be stopped with TerminateProcess — no cleanup, risking
    # mid-write cuts to the transcription DB and backup files. The watchdog
    # holds our stdin pipe and writes 'shutdown' to request the same graceful
    # teardown the signals trigger. EOF alone is deliberately ignored (a dead
    # watchdog must not stop the service; its replacement re-attaches).
    if _is_watchdog_managed():
        def _stdin_shutdown_watcher():
            try:
                for line in sys.stdin:
                    if line.strip() == "shutdown":
                        print("[SHUTDOWN] Graceful shutdown requested by watchdog")
                        signal_handler(signal.SIGTERM, None)  # exits the process
            except Exception:
                pass  # stdin closed/unreadable: channel unavailable, signals still apply
        threading.Thread(target=_stdin_shutdown_watcher, daemon=True,
                         name="watchdog-shutdown").start()

    # Direct-run auto-update: fast-forward the checkout before anything spins up.
    # No-op (and no restart) under the watchdog, when disabled, or when there's
    # nothing to pull / the tree isn't safely fast-forwardable.
    _run_startup_self_update()

    transcription_process = multiprocessing.Process(
        target=thread1_function,
        args=(transcription_state, control_queue, config_queue,
              calibration_state, calibration_data_shared, calibration_step1_data,
              audio_stream_queue)
    )
    # thread2 = multiprocessing.Process(target=thread2_function)
    # thread1 = threading.Thread(target=thread1_function)
    thread2 = threading.Thread(target=thread2_function)

    transcription_process.start()
    thread2.start()

    # Store references in module for restart endpoint and signal handler
    globals()["thread1"] = transcription_process
    globals()["thread2"] = thread2

    # Opt-in auto-start: begin live transcription immediately on launch.
    # The worker (already started above) picks this up from its idle loop,
    # lazily loads the model, and self-selects the default device — no UI,
    # client, or device index required. Mirrors the calibration start path.
    try:
        if load_config().get("audio", {}).get("autostart", False):
            print("[AUTOSTART] audio.autostart enabled; starting transcription")
            control_queue.put({"command": "start"})
            transcription_state["status"] = "starting"
    except Exception as e:
        print(f"[AUTOSTART] Failed to auto-start transcription: {e}")

    # Periodic direct-run auto-update (idle-gated). Started here in the main
    # process only, so the multiprocessing worker (which re-imports this module)
    # never spawns a duplicate updater.
    if not _is_watchdog_managed() and _self_update_enabled():
        threading.Thread(target=_self_update_loop, daemon=True, name="SelfUpdate").start()

    # Heartbeat to a paired offload server while transcription runs (main process
    # only — the worker re-imports this module but doesn't run __main__).
    threading.Thread(target=_remote_heartbeat_loop, daemon=True, name="RemoteHeartbeat").start()

    # Use a loop with timeout instead of blocking join
    # This makes the main process responsive to signals
    try:
        while transcription_process.is_alive() or thread2.is_alive():
            transcription_process.join(timeout=1.0)
            thread2.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\nMain process received KeyboardInterrupt, cleaning up...")
        signal_handler(signal.SIGINT, None)
