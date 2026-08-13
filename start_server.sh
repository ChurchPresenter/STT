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

# Read port from config.json
# The live config lives in the data dir (STT_DATA_DIR, else ~/.stt), not in the
# checkout — config/ here holds only the shipped template, so reading it always threw
# and always fell back to 8080 regardless of the port the server is bound to.
PORT=$("$PYTHON_BIN" -c "import os,json; d=os.environ.get('STT_DATA_DIR') or os.path.join(os.path.expanduser('~'),'.stt'); print(json.load(open(os.path.join(d,'config','config.json'))).get('web_server',{}).get('port',8080))" 2>/dev/null || echo 8080)

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

OS=$(uname -s)

if [ "$OS" = "Linux" ]; then
    # Linux: try systemd first
    for service_name in stt-server stt; do
        if systemctl list-unit-files "${service_name}.service" 2>/dev/null | grep -q "$service_name"; then
            echo "Starting via systemd ($service_name)..."
            sudo systemctl start "$service_name"
            sleep 2
            if systemctl is-active --quiet "$service_name"; then
                echo -e "${GREEN}[OK]${NC} Server started (systemd: $service_name)"
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
        echo -e "${GREEN}[OK]${NC} Server started (launchd: com.stt.server)"
        echo "View logs: tail -f $SCRIPT_DIR/server.log"
        exit 0
    fi
fi

# Fallback: start manually
echo "Starting server on port $PORT..."
if [ "$PORT" -le 1024 ] && [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} Port $PORT requires root. Running with sudo..."
    sudo nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &
else
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &
fi

sleep 3

if pgrep -f "speech_to_text\.py" > /dev/null; then
    echo -e "${GREEN}[OK]${NC} Server started on port $PORT"
    echo "View logs: tail -f $SCRIPT_DIR/server.log"
else
    echo -e "${RED}[ERROR]${NC} Server failed to start. Check server.log"
    exit 1
fi
