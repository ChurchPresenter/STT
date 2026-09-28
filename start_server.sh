#!/bin/bash

# Speech-to-Text Start Script (Linux & macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Show current version + update status (git) — like update_server.sh. The app
# applies any pending update itself on startup (server.log); this just surfaces it.
if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    echo -e "${GREEN}[GIT]${NC} $(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$SCRIPT_DIR" log --oneline -1)"
    UPSTREAM=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)
    if [ -n "$UPSTREAM" ]; then
        git -C "$SCRIPT_DIR" fetch --quiet 2>/dev/null
        BEHIND=$(git -C "$SCRIPT_DIR" rev-list --count "HEAD..$UPSTREAM" 2>/dev/null || echo 0)
        if [ "${BEHIND:-0}" -gt 0 ]; then
            echo -e "${YELLOW}[GIT]${NC} Update available: $BEHIND commit(s) behind $UPSTREAM — applied on startup"
        else
            echo -e "${GREEN}[GIT]${NC} Up to date with $UPSTREAM"
        fi
    fi
fi

# Determine Python binary
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

# The config the server reads, and the port it binds, from stt/server_port.py — the
# invoking user's under sudo, never root's (see restart_server.sh).
# -B: run as root, a helper must not leave root-owned __pycache__ in the checkout.
DATA_DIR=$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" -B -m stt.server_port --data-dir 2>/dev/null)
[ -n "$DATA_DIR" ] || DATA_DIR="${STT_DATA_DIR:-$HOME/.stt}"
PORT=$(STT_DATA_DIR="$DATA_DIR" PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" -B -m stt.server_port 2>/dev/null || echo 8080)

# Claim a port only once the server answers on it, rather than the one this script guessed.
report_started() {
    if command -v curl >/dev/null 2>&1; then
        for _ in $(seq 1 30); do
            if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/" 2>/dev/null; then
                echo -e "${GREEN}[OK]${NC} Server started ($1) on port $PORT"
                return 0
            fi
            sleep 1
        done
        echo -e "${YELLOW}[WARNING]${NC} Server started ($1), but nothing answered on port $PORT yet — check the log"
        return 0
    fi
    echo -e "${GREEN}[OK]${NC} Server started ($1); expected on port $PORT"
}

# Check if already running
if pgrep -f "speech_to_text\.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}[WARNING]${NC} Server is already running"
    echo "Use ./restart_server.sh to restart or ./stop_server.sh to stop"
    exit 1
fi

# Check if port is in use
if command -v fuser &> /dev/null && fuser "$PORT/tcp" 2>/dev/null; then
    echo -e "${RED}[ERROR]${NC} Port $PORT is already in use by another process"
    exit 1
elif command -v lsof &> /dev/null && lsof -i :"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} Port $PORT is already in use by another process"
    exit 1
fi

# ─── Optional dependency preflight ──────────────────────────────────
# requirements.txt deliberately omits a few large, rarely-used packages (see the
# commented block at its end), so the hash-gated sync in update_server.sh can never
# notice one *missing* — it installs what that file lists, and these are not in it.
# A rebuilt venv therefore dropped llama-cpp-python silently, and live translation
# degraded to handing every caption back untranslated with HTTP 200. This asks the
# question that sync cannot: does the venv have what the live config is asking it to
# do? Best-effort and always exits 0 — a missing optional package degrades one
# feature, a start script that refuses to start degrades everything.
# Set STT_SKIP_DEP_CHECK=1 to skip it.
if [ -z "$STT_SKIP_DEP_CHECK" ] && [ -f "$VENV_PYTHON" ]; then
    PYTHONPATH="$SCRIPT_DIR" "$VENV_PYTHON" -B -m stt.optional_deps --repo-dir "$SCRIPT_DIR" --data-dir "$DATA_DIR" 2>&1
fi

OS=$(uname -s)

if [ "$OS" = "Linux" ]; then
    # Linux: try systemd first
    for service_name in stt-server stt; do
        if systemctl list-unit-files "${service_name}.service" 2>/dev/null | grep -q "$service_name"; then
            echo "Starting via systemd ($service_name)..."
            sudo systemctl start "$service_name"
            sleep 2
            if systemctl is-active --quiet "$service_name"; then
                report_started "systemd: $service_name"
                echo "View logs: sudo journalctl -u $service_name -f"
                exit 0
            fi
        fi
    done
fi

# macOS launchd check
if [ "$OS" = "Darwin" ]; then
    if launchctl list com.stt.server &> /dev/null; then
        echo "Starting via launchd..."
        launchctl start com.stt.server
        sleep 2
        report_started "launchd: com.stt.server"
        echo "View logs: tail -f $SCRIPT_DIR/server.log"
        exit 0
    fi
fi

# Fallback: start manually
echo "Starting server on port $PORT..."
if [ "$PORT" -le 1024 ] && [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} Port $PORT requires root. Running with sudo..."
    # STT_DATA_DIR: a root server would otherwise read /root/.stt — a different config,
    # models and sessions, and so possibly not the port printed above.
    sudo env STT_DATA_DIR="$DATA_DIR" STT_MANAGED=0 nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &
else
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &
fi

sleep 3

if pgrep -f "speech_to_text\.py" > /dev/null; then
    report_started "manual"
    echo "View logs: tail -f $SCRIPT_DIR/server.log"
else
    echo -e "${RED}[ERROR]${NC} Server failed to start. Check server.log"
    exit 1
fi
