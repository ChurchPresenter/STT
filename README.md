# STT - Speech-To-Text

**Website:** [stt.churchpresenter.org](https://stt.churchpresenter.org)

Real-time speech transcription platform with a modern web interface, powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Designed for continuous transcription scenarios like church services, lectures, and meetings.

## Features

### Transcription
- **Real-time transcription** - Live microphone capture with sub-second latency via WebSocket (FFmpeg-based capture)
- **File transcription** - Upload and transcribe audio/video files in batch
- **Selectable engines** - Faster-Whisper (CTranslate2), OpenAI Whisper, and HuggingFace models (distil-whisper, wav2vec2)
- **Voice Activity Detection** - Silero VAD filters silence and noise before transcription
- **Speech/music/quiet detection** - PANNs-based three-way classification; detected music can be transcribed but auto-hidden, restorable from corrections
- **Microphone calibration** - Guided wizard measures ambient noise and suggests threshold settings
- **Accuracy tuning** - Loudness normalization, sentence-completion buffering, context prompting, and per-mode Whisper decoding parameters
- **Hallucination filter** - Removes Whisper phantom phrases ("thanks for watching", etc.)
- **Device auto-recovery** - Re-finds the microphone by card name after reboots or device-index changes

### Translation & speech
- **Translation** - Real-time translation to 200+ languages using Facebook NLLB-200 or Google MADLAD-400, selectable per install
- **CTranslate2 int8 backend** - Optional quantised inference for either translation engine, converted locally on first use (MADLAD-3B in ~3 GB). CUDA and CPU; no Apple Metal, so Apple Silicon runs CPU int8
- **Remote translation offload** - Pair with another STT machine and offload translation to it, with reachability checks and configurable fallback
- **Custom dictionary & glossary** - Domain-specific term corrections, forced translations, synced to the paired remote machine
- **Text-to-Speech** - Edge-TTS (cloud) and Piper-TTS (local) with auto voice switching per language and speed control

### Display & output
- **Display profiles** - Named, recallable output layouts built in `/url-builder` and served at `/profile/<name>`
- **Layout modes** - Translated-only, side-by-side, or stacked, with drip-feed word-by-word reveal
- **Browser audio streaming** - Remote viewers can listen to the live room microphone and TTS audio in their browser

### Review & content control
- **Corrections workflow** - Review queue with confidence scores, low-confidence word flagging, and alternative translations
- **Staged output delay** - Hold segments for N seconds to approve or discard before they go live
- **Word highlighting** - Mark and emphasize specific words or phrases in transcriptions
- **Profanity filter** - Masks configured words with `****` in output

### Storage & files
- **Database storage** - SQLite per-session with SRT subtitle and HTML export, plus optional partial-snapshot recording
- **Session provenance** - Each session database records the transcription and translation models, decode settings, and filters that produced it, including settings changed mid-session, so a transcript stays attributable long after the server is retuned
- **Audio backup** - WAV and MPEG-TS formats with power-fail-safe continuous backup
- **File manager** - Web-based browser for backups: rename, download, hide, bulk operations, type/day filters
- **Remote file delivery** - Automatic backup to SMB/NAS shares

### Operations
- **Model manager** - Browse, search, and download Whisper, NLLB, MADLAD, VAD, and PANNs models from Hugging Face, or upload local models
- **Unattended start** - `audio.autostart` begins live transcription at server launch with no UI interaction; combined with the watchdog, transcription resumes by itself after a crash or update
- **Security** - IP whitelist (CIDR), password authentication, session timeouts
- **Hardware acceleration** - NVIDIA CUDA with automatic detection. Apple Silicon MPS is used for translation and the OpenAI Whisper backend; the default faster-whisper backend has no Metal path and runs CPU int8 (see System Requirements)
- **Crash recovery & auto-update** - Watchdog process manager restarts STT on crashes; idle-gated updates with stable/beta channels
- **Server tools** - Uptime/version display, disk-space monitor, timezone settings, runtime language switching

## Quick Start

```bash
# Install dependencies
./install.sh

# Start the server
./start_server.sh        # Linux / macOS (start_server.bat on Windows)
```

Open http://localhost:8080 in your browser (port is configurable in `config/config.json`).

## First Time Setup

A new install ships inert: no model is selected and translation is off, so nothing is
downloaded until you choose it. Start is disabled until steps 1 and 2 are done, and the
page says which one is outstanding.

1. Go to `/model-manager` to download a Whisper model, then select it
2. Go to `/live-settings` to select your microphone and language
3. Start transcribing on the home page

Translation is optional and separate: enable it on `/translation`, where you also pick
the model or endpoint that does the translating.

## Running Headless (No GUI)

The **Watchdog** manages STT with crash recovery and auto-updates. Run it headless to keep STT running in the background without a desktop.

### Binary install (downloaded release)

| Platform | Command |
|----------|---------|
| Linux / macOS | `./STT-Watchdog --headless` |
| Windows | `STT-Watchdog.exe --headless` |

Or use the provided scripts which handle logging automatically:

```bash
# Linux / macOS
./start_watchdog.sh

# Windows (cmd)
start_watchdog.bat

# Windows (PowerShell)
.\start_watchdog.ps1
```

### Source install

```bash
python3 stt/watchdog.py --headless
```

### Persistent service (auto-start on boot)

**Linux (systemd)**

```bash
sudo cp deploy/stt-watchdog.service /etc/systemd/system/
# Edit the file to set User= and adjust paths if needed
sudo systemctl daemon-reload
sudo systemctl enable --now stt-watchdog
sudo journalctl -u stt-watchdog -f   # view logs
```

**macOS (LaunchAgent)**

```bash
cp deploy/com.stt.watchdog.plist ~/Library/LaunchAgents/
# Edit INSTALL_DIR placeholders to your actual install path
launchctl load ~/Library/LaunchAgents/com.stt.watchdog.plist
```

**Windows (Task Scheduler)**

Run once at startup via Task Scheduler:
- Action: `STT-Watchdog.exe --headless` (or `pythonw stt/watchdog.py --headless` for source)
- Trigger: At log on / At startup
- Settings: Run whether user is logged on or not

## Web Interface Pages

| Page | Description |
|------|-------------|
| `/` | Live transcription with real-time updates |
| `/file` | File upload and batch transcription |
| `/translation` | Translation settings, language pairs, TTS voice selection |
| `/corrections` | Review and edit transcription segments |
| `/service-phase` | Detected service part (songs / speaking / quiet) and operator review |
| `/word-highlighting` | Manage highlighted phrases |
| `/url-builder` | Build and save display profiles (fonts, colors, layout URL parameters) |
| `/live-settings` | Audio device, language, VAD settings |
| `/server-settings` | Network, database, backup configuration |
| `/model-manager` | Download and manage AI models |
| `/file-manager` | File browser and SMB/NAS settings |
| `/profile/<name>` | Named display profile output layout |

## Documentation

- [PIPELINE.md](PIPELINE.md) — how audio becomes a caption: the four processes, every stage and the number that governs it, and what happens when translation declines.
- [INSTALL.md](INSTALL.md) — detailed installation instructions, system requirements, and troubleshooting.
- [DESIGN.md](DESIGN.md) — colours, typography and component rules for the control UI.

## System Requirements

### Transcription only (minimum)
- **CPU:** 6 cores | **RAM:** 12 GB | **Storage:** 15 GB
- **Python:** 3.9 - 3.13
- **OS:** Linux, Windows, or macOS (Apple Silicon only — PyTorch no longer publishes Intel-Mac builds, so setup refuses an Intel Mac)

### Transcription + Translation (minimum)
- **CPU:** 8 cores | **RAM:** 16 GB | **Storage:** 25 GB
- The translation model adds several GB of RAM and disk — NLLB-600M is the lightest at ~1.2 GB, MADLAD-3B the heaviest commonly used at ~12 GB. Enabling the CTranslate2 int8 backend cuts the runtime footprint substantially (MADLAD-3B to ~3 GB).

### Example configurations

Actual requirements depend on which models you configure. A few representative setups to get a feel for the range:

| Setup | Whisper model | Translation | Est. memory | Suggested hardware |
|-------|---------------|-------------|-------------|--------------------|
| Light, CPU-only | `tiny` / `base` (faster-whisper) | off / remote | ~5 GB RAM | any modern 8 GB PC |
| Balanced, CPU-only | `small` (faster-whisper) | off / remote | ~5.5 GB RAM | 8 GB PC |
| Default install, CPU-only | `small` (openai-whisper) | NLLB-600M local | ~9.5 GB RAM | 12 GB PC |
| Accurate, NVIDIA GPU | `large-v3` (faster-whisper) | off / remote | ~4.5 GB VRAM + ~5 GB RAM | 6 GB GPU (RTX 2060 / 3050) |
| Full stack, NVIDIA GPU | `large-v3` (faster-whisper) | NLLB-1.3B on GPU | ~8 GB VRAM + ~6 GB RAM | 10-12 GB GPU (RTX 3060 12GB+) |
| Apple Silicon | `small` (CPU int8) | NLLB-600M on MPS | ~10 GB unified memory | M1 or later with 16 GB |
| MADLAD, CTranslate2 int8 | `small` (faster-whisper) | MADLAD-3B int8 | ~3 GB for translation + ~5.5 GB RAM | 16 GB PC or Mac (CPU int8 on Apple Silicon) |

Estimates include a ~4 GB app/OS baseline. The faster-whisper backend (int8) needs roughly half the memory of openai-whisper (fp32); Apple Silicon shares one memory pool between CPU and GPU. The web UI shows a warning banner whenever the machine falls short of what the currently configured models need.

### Acceleration (optional, recommended)
- **NVIDIA minimum:** 4GB+ VRAM (RTX 2060 / RTX 3050) — enough for transcription with small/medium models
- **NVIDIA recommended:** 10GB+ VRAM (RTX 3060 12GB, RTX 4070 or better) — large models and transcription + translation on GPU
- **CUDA:** 12.8 compatible drivers (R570+)
- **Apple Silicon:** M1 or later. MPS is detected and used automatically for **translation** and for the non-default `whisper` (OpenAI) backend. The default `faster-whisper` backend is CTranslate2, which has no Metal support, so **transcription runs on the CPU** in int8 — fast and memory-efficient, but not GPU-accelerated. A larger GPU on an M-series Mac will not speed up the default transcription path.

> The minimum tiers run CPU-only, which is significantly slower than GPU — larger models add noticeable transcription latency. Lower-spec hardware may still work depending on configuration (e.g. smaller Whisper models, reduced settings), at the cost of accuracy and/or speed. Offloading translation to a remote machine keeps the local requirements at the transcription-only tier.

## Configuration

Edit `config/config.json` or use the web interface. Key settings include:

- Model selection (Whisper variant, backend)
- Audio device selection (FFmpeg-based capture)
- Unattended transcription start (`audio.autostart`, see below)
- Voice Activity Detection (Silero VAD) threshold
- Database paths and naming format
- Audio backup paths and formats
- Translation model and glossary
- Network host, port, and security

### Unattended start (`audio.autostart`)

Two different things are called "auto-start", and they stack:

| | What it does | Where |
|---|---|---|
| **Service auto-start** | Launches the STT *app* when the machine boots or the user logs in | systemd / LaunchAgent / Task Scheduler — see [Persistent service](#persistent-service-auto-start-on-boot) |
| **`audio.autostart`** | Begins live *transcription* as soon as the app starts, without the UI Start button | `config/config.json`, or the toggle on `/server-settings` |

Set both and a booth PC captions from power-on with nobody touching it. `audio.autostart`
defaults to `false`, and changing it **applies on the next server start** — the key is read
once at launch, so a config save alone won't arm it.

Because the watchdog re-launches the server on crash and after updates, this also means
transcription resumes unattended mid-service. Note it starts a **new session** rather than
resuming the old one: a new database and `session_id`, so the transcript splits at the
restart point, and crash backoff plus device and model init costs 5–60 seconds.

Two failure modes to know before enabling it on a machine you won't be watching:

- **No retry if the audio device isn't ready.** Autostart fires the moment the server
  reaches its entry point, and the worker opens the device before loading the model. On a
  cold boot — especially with a USB interface — the card may not have enumerated yet. If
  device init fails, the worker parks idle and nothing tries again until the process
  restarts. The bundled systemd unit orders on `network.target` only, not on sound.
- **A missing microphone doesn't error, it falls back.** Device resolution tries the saved
  card name, then the configured device, then `default`, then `plughw:0,0`. If your
  configured input is absent the run can start on a different input and record a full
  session from it. Confirm the device by name after any hardware change.

## Privacy & Telemetry

- **Error reporting (Sentry)** - Crash reports, logs, and performance traces are sent to Sentry to help improve STT. Reports carry the error and its stack trace, the STT version, and OS/Python/CPU details. They deliberately carry **no** transcription or translation text, no audio, no request bodies, no frame locals, no IP addresses or headers, and no hostname — request data is stripped before send and PII is off by default (`sentry_send_pii_optin`). Disable entirely via the toggle on `/server-settings` or set `sentry_enabled: false` in `config/config.json` (applies on restart) — crash dumps are then kept locally in `logs/crashes/` only.

## Tech Stack

- **Backend:** Python 3.9+ with Flask and Flask-SocketIO
- **Speech Recognition:** Faster-Whisper (CTranslate2)
- **Translation:** Facebook NLLB-200 or Google MADLAD-400 via Hugging Face Transformers, with an optional CTranslate2 (int8) backend
- **TTS:** Edge-TTS and Piper-TTS
- **Audio:** FFmpeg for capture and processing, Silero VAD
- **ML:** PyTorch with CUDA 12.8 support
- **Frontend:** Bootstrap, jQuery, Socket.IO
- **Database:** SQLite (per-session)

## Cross-Platform Support (Source Install)

Installation scripts are provided for all platforms:

| Platform | Install | Start | Stop | Restart |
|----------|---------|-------|------|---------|
| Linux/macOS | `install.sh` | `start_server.sh` | `stop_server.sh` | `restart_server.sh` |
| Windows | `install.bat` / `install.ps1` | `start_server.bat` | `stop_server.bat` | `restart_server.bat` |

On Linux, the installer can also set up a **systemd service** for auto-start on boot.
