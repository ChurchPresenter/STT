#!/bin/bash

# Speech-to-Text Stop Script (Linux & macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running as root (Linux only — macOS doesn't need it for basic kill)
OS=$(uname -s)
if [ "$OS" = "Linux" ] && [ "$EUID" -ne 0 ]; then
    echo "Not running as root. Please run with: sudo ./stop_server.sh"
    exit 1
fi

echo "Stopping server..."

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
PYTHON_BIN=$([ -f "$VENV_PYTHON" ] && echo "$VENV_PYTHON" || echo "python3")
# The config the server reads, and the port it binds, from stt/server_port.py. This
# script must be root on Linux, so ~ here is /root: reading ~/.stt killed port 8080 on a
# box whose server, running as the invoking user, serves port 80.
DATA_DIR=$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" -m stt.server_port --data-dir 2>/dev/null)
[ -n "$DATA_DIR" ] || DATA_DIR="${STT_DATA_DIR:-$HOME/.stt}"
PORT=$(STT_DATA_DIR="$DATA_DIR" PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" -m stt.server_port 2>/dev/null || echo 8080)

# ─── Stop managed services ──────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    for service_name in stt-server stt; do
        if systemctl is-active --quiet "$service_name" 2>/dev/null; then
            echo "Stopping $service_name systemd service..."
            systemctl stop "$service_name"
        fi
    done
elif [ "$OS" = "Darwin" ]; then
    if launchctl list com.stt.server &> /dev/null; then
        echo "Stopping launchd service..."
        launchctl stop com.stt.server
    fi
fi

# ─── Kill port holder ───────────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    fuser -k "$PORT/tcp" 2>/dev/null
elif [ "$OS" = "Darwin" ]; then
    lsof -ti :"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null
fi

# ─── Kill speech_to_text processes ───────────────────────────────────
pkill -TERM -f "speech_to_text\.py" 2>/dev/null
sleep 1
pkill -9 -f "speech_to_text\.py" 2>/dev/null

# ─── Kill orphaned ffmpeg processes ──────────────────────────────────
if [ "$OS" = "Linux" ]; then
    pkill -TERM -f "ffmpeg.*alsa.*pipe:1" 2>/dev/null
    sleep 1
    pkill -9 -f "ffmpeg.*alsa.*pipe:1" 2>/dev/null
elif [ "$OS" = "Darwin" ]; then
    pkill -TERM -f "ffmpeg.*avfoundation" 2>/dev/null
    sleep 1
    pkill -9 -f "ffmpeg.*avfoundation" 2>/dev/null
fi

# ─── Hand the data dir back ─────────────────────────────────────────
# A server run as root leaves root-owned files in the invoking user's data dir, and the
# next non-root start cannot rewrite its own config.
if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ] && [ -d "$DATA_DIR" ]; then
    USER_HOME=$("$PYTHON_BIN" -c "import pwd,sys; print(pwd.getpwnam(sys.argv[1]).pw_dir)" "$SUDO_USER" 2>/dev/null)
    case "$DATA_DIR" in
        "$USER_HOME"/*) [ -n "$USER_HOME" ] && chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$DATA_DIR" 2>/dev/null ;;
    esac
fi

# ─── Verify ──────────────────────────────────────────────────────────
sleep 1
if pgrep -f "speech_to_text\.py" > /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Some processes still running:"
    ps aux | grep speech_to_text | grep -v grep
else
    echo -e "${GREEN}[OK]${NC} Server stopped"
fi
