"""
PyInstaller spec for the STT demo.

Unlike watchdog.spec — which builds a bootstrapper that clones the app and provisions
a venv on the user's machine — this bundles the whole application, so the artifact
runs with nothing installed: no Python, no pip, no models, no ML libraries. It
replays a recorded service through the real UI (see stt/demo_playback.py) and answers
the model/TTS/translation routes from recorded fixtures (stt/demo_api.py).

The ML excludes below are the load-bearing part. They are what keeps the build at
~25MB and what proves the demo genuinely cannot reach torch: if a code path ever
started importing one at module scope, this build would fail rather than quietly ship
a 3GB bundle.

Build (from the repo root):
    STT_DEMO_DB=/path/to/session.db pyinstaller packaging/demo.spec
Or via build.py:
    python packaging/build.py --demo --session /path/to/session.db
"""

import os
import sys

block_cipher = None
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (PyInstaller global)

_icon_file = os.path.join(ROOT, "icon.icns" if IS_MACOS else "icon.ico")
_icon = _icon_file if os.path.exists(_icon_file) else None

try:
    with open(os.path.join(ROOT, "VERSION")) as _vf:
        _bundle_version = _vf.read().strip() or "0.0.0"
except OSError:
    _bundle_version = "0.0.0"

# The recording to ship. Deliberately not defaulted: a session database is
# congregation speech until someone has decided otherwise, so the build refuses to
# guess which one — and therefore cannot produce an artifact from a recording nobody
# chose. See stt/demo_scrub.py and the sessions/ drop-in folder for the alternatives.
_demo_db = os.environ.get("STT_DEMO_DB", "").strip()
if not _demo_db:
    raise SystemExit(
        "demo.spec: set STT_DEMO_DB to the session database to bundle.\n"
        "  Generate a synthetic one:  python scripts/make_demo_session.py --synthetic -o build/demo.db\n"
        "  Or scrub a real service:   python scripts/make_demo_session.py --from-session <db> ..."
    )
if not os.path.isfile(_demo_db):
    raise SystemExit(f"demo.spec: STT_DEMO_DB does not exist: {_demo_db}")

_datas = [
    (os.path.join(ROOT, "templates"), "templates"),
    (os.path.join(ROOT, "static"), "static"),
    (os.path.join(ROOT, "VERSION"), "."),
    (_demo_db, "demo"),
]
# Only the tracked templates — never a live config, which carries the build
# machine's settings and possibly its credentials.
_config_dir = os.path.join(ROOT, "config")
for _name in sorted(os.listdir(_config_dir)):
    if _name.endswith(".default.json"):
        _datas.append((os.path.join(_config_dir, _name), "config"))
if os.path.exists(_icon_file):
    _datas.append((_icon_file, "."))

a = Analysis(
    [os.path.join(ROOT, "speech_to_text.py")],
    pathex=[ROOT],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        # engineio picks its async driver by dynamic import, which PyInstaller's
        # static analysis cannot see. Without this the server starts and then fails
        # to accept a single WebSocket — the classic flask-socketio freeze bug.
        "engineio.async_drivers.threading",
        "simple_websocket",
        "wsproto",
        "bidict",
        "jinja2.ext",
        "werkzeug",
        # Imported by the monolith through a name PyInstaller cannot follow.
        "stt.demo_mode",
        "stt.demo_playback",
        "stt.demo_api",
        "stt.demo_fixtures",
        "stt.demo_redact",
        # The two things a demo reports. requests is what the live-map ping uses;
        # sentry_sdk loads its integrations dynamically, so PyInstaller's static
        # analysis needs them named (same list as watchdog.spec).
        "requests",
        "certifi",
        "sentry_sdk",
        "sentry_sdk.integrations.flask",
        "sentry_sdk.integrations.stdlib",
        "sentry_sdk.integrations.excepthook",
        "sentry_sdk.integrations.dedupe",
        "sentry_sdk.integrations.atexit",
        "sentry_sdk.integrations.modules",
        "sentry_sdk.integrations.logging",
        "sentry_sdk.integrations.threading",
        "sentry_sdk.integrations.argv",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPECPATH, "demo_rthook.py")],  # noqa: F821
    excludes=[
        "matplotlib",
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        # The whole point of the demo build. Every one of these is reached only from
        # the transcription worker or the TTS backends, neither of which a demo runs.
        "torch",
        "torchaudio",
        "transformers",
        "faster_whisper",
        "whisper",
        "openai_whisper",
        "huggingface_hub",
        "ctranslate2",
        "panns_inference",
        "torchlibrosa",
        "silero_vad",
        "speech_recognition",
        "edge_tts",
        "piper",
        "piper_tts",
        "supertonic",
        "onnxruntime",
        "llama_cpp",
        "numpy",
        "scipy",
        "librosa",
        "pandas",
        "soundfile",
        "pydub",
        "datasets",
        "accelerate",
        "tkinter",
        "PIL",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="STT-Demo",
    debug=False,
    strip=False,
    upx=True,
    # A double-clicked demo must not open a terminal window; on Linux it is run from
    # one anyway, and the console is where its log goes.
    console=not (IS_WINDOWS or IS_MACOS),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="STT-Demo",
)

if IS_MACOS:
    app = BUNDLE(
        coll,
        name="STT Demo.app",
        icon=_icon,
        # Its own identity, so installing the demo can never replace a real STT.
        bundle_identifier="com.stt.demo",
        info_plist={
            "CFBundleName": "STT Demo",
            "CFBundleDisplayName": "STT Demo",
            "CFBundleExecutable": "STT-Demo",
            "CFBundleShortVersionString": _bundle_version,
            "CFBundleVersion": _bundle_version,
            "NSHighResolutionCapable": True,
            # No NSMicrophoneUsageDescription on purpose: the demo never opens an
            # audio device, and asking for the microphone would be a lie.
            "LSUIElement": False,
        },
    )
