# STT — Agent Instructions

## Project overview

Real-time speech transcription platform powered by Faster-Whisper, with a web UI for live event operators (church services, lectures, meetings). Live microphone transcription streams over WebSocket; also supports batch file transcription, real-time translation (Facebook NLLB-200), text-to-speech (Edge-TTS cloud / Piper-TTS local), and PANNs-based music/speech detection.

Stack: Python 3.9+, Flask + Flask-SocketIO, SQLite (per-session databases), Jinja2 server-rendered templates. No frontend build step — `static/` contains vendored jQuery, socket.io, and Font Awesome.

## Commands

The project venv (`.venv/`, Python 3.9) is uv-managed and has no pip — use `uv pip install` or the venv's python directly.

```bash
.venv/bin/python3 speech_to_text.py      # Run the server (port 80) — ./start_server.sh / .bat wrappers exist
STT_DEMO=1 .venv/bin/python3 speech_to_text.py   # Run as the demo (see Demo build)
.venv/bin/python3 -m pytest              # Run tests (testpaths = tests/, config in pyproject.toml)
.venv/bin/python3 -m ruff check .        # Lint (line-length 200, target py39)
./install.sh                             # Install runtime deps (install.bat / install.ps1 on Windows)
uv pip install -r requirements-dev.txt   # Dev/test deps (pytest, ruff)
```

Ruff selects the hard-error rules (E9, F63, F7, F82, F811) plus `B` and `RUF`, and excludes `.venv`, `_AUTOMATIC_BACKUP`, `models`, `installer`.

### Demo build

A self-contained demo ships as one executable per OS (~29MB, ~15MB zipped) that runs
with no Python, no models and no ML libraries. It replays a recorded service into the
real UI, so Start/Stop and every page behave exactly as in production.

```bash
python scripts/make_demo_session.py --synthetic -o build/demo.db  # a written service
python packaging/build.py --demo --synthetic                      # build the artifact
STT_DEMO=1 STT_DEMO_DB=<session.db> .venv/bin/python3 speech_to_text.py   # run from source
```

`stt/demo_playback.py` replaces the transcription worker: it opens a real session
database and copies the recording's rows into it at their original pace, so the whole
read path (`get_new_entries`, the phase detector, corrections, the file manager) works
unmodified. `stt/demo_api.py` answers the model/TTS/translation routes from
`stt/demo_fixtures.py` — recorded from a real server by `scripts/record_demo_fixtures.py`
and scrubbed through `stt/demo_redact.py`. Those families are **deny-by-default**, so a
route added under `/api/models/` later fails closed rather than reaching an ML import.

The demo reports two things and nothing else: that it ran (a live-map ping carrying
`src=demo`) and that it crashed (a Sentry event tagged `demo`). Both are tagged at the
source so the collector counts a trial as a trial — `stt/worker/index.ts` in the website
repo keeps `demo` out of every install figure, and the anonymous id is carried across
the per-launch data wipe so one person opening the demo twice is not two trials.

Everything else it could send is shut off. The demo binds the network with no password,
so it must not be usable as somebody else's network client. `stt/demo_guard.py` holds
`CHOKE_POINTS` — the few monolith
functions every outbound request passes through — and `audit_choke_points` parses
`speech_to_text.py` to assert each still opens with an `if DEMO:` guard, the way
`stt/peer_auth_audit.py` enforces its own one-door rule. **A new way to reach the
network must be added to `CHOKE_POINTS`.** Routes whose risk is `subprocess` rather than
egress (`/api/tunnel/`, `/api/file-mover/`, `/api/remote-translation/`) are refused by
`demo_api.BLOCKED_PREFIXES` instead, since no network guard would catch them.

Demo mode is `STT_DEMO=1`, set in the shipped artifact by `packaging/demo_rthook.py`. It
redirects all data to `~/.stt-demo` (wiped each launch), so a demo can never write into a
real install. The build refuses to run without `STT_DEMO_DB`: a session database is
congregation speech until someone decides otherwise. Prefer `stt/demo_synth.py` for
anything published; `stt/demo_scrub.py` exists for a real recording, reduces risk, and
does not certify — read the `.review.txt` it writes.

## Architecture

The server is mostly a monolith — most changes land in `speech_to_text.py`.

| Path | Role |
|------|------|
| `speech_to_text.py` (~15,700 lines) | The entire server: Flask routes, SocketIO events, transcription pipeline, translation, TTS, SQLite storage, settings |
| `stt/audio_capture.py` | Microphone capture layer |
| `stt/watchdog.py` | Separate process manager: crash recovery, auto-update, headless mode (`--headless`) |
| `stt/file_mover.py` | SMB/NAS remote file delivery |
| `templates/` | Jinja2 pages: index, live-settings, model-manager, server-settings, translation, corrections, file-manager, word-highlighting, url-builder |
| `static/` | Vendored JS/CSS — no build step, no npm |
| `stt/model_files.py` | Download manifests, file verification, per-family "is this model loadable" |
| `stt/demo_*.py` | The shippable demo: playback engine, fake backends, egress guards, redaction, synthetic service generator |
| `scripts/` | Dev-only tools: fixture recorder, demo session builder |
| `tests/` | Pytest suite: download state, path safety, staging, text utils, watchdog update |
| `packaging/` | Binary build tooling (build.py, make_icon.py, watchdog.spec, demo.spec) — NOT `build/`, which PyInstaller uses as its workdir |
| `deploy/` | OS service templates: stt-watchdog.service (systemd), com.stt.watchdog.plist (launchd) |

## Configuration

- `config/config.default.json` — defaults/schema. **New settings need a default entry here.**
- `config/config.json` — the user's live runtime settings (gitignored, as are all non-.default files in `config/`); the server writes to it. Don't clobber it with defaults.

## Conventions

- **New logic goes in `stt/` modules, not speech_to_text.py.** The monolith cannot be imported by tests (import-time side effects), so any new pure logic — and logic being touched anyway — belongs in an importable `stt/` module with unit tests shipped in the same commit. `stt/` modules must import clean with stdlib only (CI installs no ML deps), take config/paths as explicit parameters (never read the monolith's globals), and be fully type-annotated (mypy enforces `disallow_untyped_defs` on the logic modules — see pyproject). The monolith re-imports the names via thin wrappers so call sites stay unchanged.
- **Tests**: no fixed-time sleeps (wait on events/poll with deadline), deterministic, `tmp_path` for filesystem/sqlite, assert behavior rather than incidental values. Coverage target is 85% per logic module; the `stt/` package total deliberately includes untested IO-heavy code and understates the logic layer — don't add coverage omits.
- **Service recordings never enter the repository.** A caption is verbatim congregation speech, often naming people present. Code is public — `stt/translation_replay.py` and its tests are in the repo and run everywhere. The material is not: session databases (`tests/fixtures/sessions/`), replay runs (`*.replay.json`), and the caption fixture (`tests/fixtures/real_captions.json`) are gitignored, and the cases needing them `pytest.skip` when absent so CI stays green. When a rule was found by a real caption, put a **constructed** caption of the same shape in the test and describe the real one in prose — the shape is what the test needs, and prose cannot be grepped back to a person.
- **Captions never leave the machine in a support report.** The operator-facing diagnostic report (`/api/diagnostics/report`, built by `stt/diagnostics.py`) is **allowlist-only** in both directions: `LOG_TAGS` names the log tags it may quote and `CONFIG_FIELDS` the settings it may show. `logs/stt.log` is the worker's raw stdout and many tags print verbatim congregation speech, so **a new `print("[TAG] ...")` is invisible to the report until someone adds its tag deliberately** — that is the intended failure mode, the same deny-by-default rule as `stt/demo_guard.py`. Never convert either list to a denylist. The report is never uploaded; it downloads for the operator to read and share themselves.
- **A download is staged, verified, and repairable.** Every transfer goes through
  `stt/downloads.py:download_url_to_file`, which writes `<name>.part` and only `os.replace()`s
  it into position once it matches the size (and, for LFS weights, the sha256) the Hub
  reported. Writing straight to the final name is what made an interrupted download
  indistinguishable from a complete one — it listed as downloaded, a re-download skipped it as
  "already exists", and the loader either threw in a native reader or hung fetching what was
  missing. So: **"already exists" must mean "exists and verifies"** (`stt/model_files.py`
  `files_needing_download`), and a completed download writes a `.stt-download.json` manifest
  that is what later tells a truncated file from a whole one. A *missing* manifest never means
  "not downloaded" — every install predating it has none.
- **A loader checks its files before it opens them.** `faster_whisper_status` requires
  `tokenizer.json`, not just a weight file: without it faster-whisper falls back to
  `tokenizers.Tokenizer.from_pretrained(...)`, a Rust builtin with its own HTTP client that
  honours neither `HF_HUB_OFFLINE` nor any timeout, so a stalled connection blocks for ever.
  `local_files_only=True` does not help — faster-whisper only forwards it to `download_model()`,
  which a local directory skips. Refusing to call the loader is the only fix.
- **Worker failures must report themselves.** The transcription worker is a `multiprocessing.Process`, and `BaseProcess._bootstrap` swallows its exceptions to stderr without ever calling `sys.excepthook` — so Sentry does not see them. Anything that can kill the worker belongs inside the `except Exception` in `thread1_function`, which routes through `_report_worker_crash` / `stt/worker_crash.py`. Never assign to `sys.excepthook` or `threading.excepthook` directly either; chain, or Sentry's hook is silently displaced.
- **Commits**: conventional-commit style with a scope, e.g. `fix(translation): …`, `feat(server): …`. **No AI attribution at all** — no `Co-authored-by` line in the message, and no `git notes` attribution either. Commits carry the work, not who typed it.
- **UI work**: follow the design tokens in `DESIGN.md` (colors, typography, spacing, component styles).
- **Do not touch** `_AUTOMATIC_BACKUP/`, `models/`, `panns_data/`, `logs/` — generated/runtime data.
