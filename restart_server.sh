#!/bin/bash

# Speech-to-Text Restart Script (Linux & macOS)
# Called from server settings page via /api/server/restart

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

OS=$(uname -s)

# Check if running as root (Linux needs it for port 80 and systemctl)
if [ "$OS" = "Linux" ] && [ "$EUID" -ne 0 ]; then
    echo "Not running as root. Please run with: sudo -E ./restart_server.sh"
    echo "The -E flag preserves environment variables (needed for Python packages)"
    exit 1
fi

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
PYTHON_BIN=$([ -f "$VENV_PYTHON" ] && echo "$VENV_PYTHON" || echo "python3")

# The config the server reads, and the port it binds. Both come from stt/server_port.py:
# this script must be root on Linux, so ~ here is /root, and reading ~/.stt reported (and
# port-killed) 8080 on a box whose server, running as the invoking user, serves port 80.
DATA_DIR=$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" -m stt.server_port --data-dir 2>/dev/null)
[ -n "$DATA_DIR" ] || DATA_DIR="${STT_DATA_DIR:-$HOME/.stt}"
PORT=$(STT_DATA_DIR="$DATA_DIR" PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" -m stt.server_port 2>/dev/null || echo 8080)

# A server run as root writes root-owned files into the invoking user's data dir, and the
# next non-root start cannot rewrite its own config. Hand them back once it has stopped.
return_data_dir() {
    [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ] && [ -d "$DATA_DIR" ] || return 0
    local user_home
    user_home=$("$PYTHON_BIN" -c "import pwd,sys; print(pwd.getpwnam(sys.argv[1]).pw_dir)" "$SUDO_USER" 2>/dev/null) || return 0
    case "$DATA_DIR" in
        "$user_home"/*) chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" "$DATA_DIR" 2>/dev/null ;;
    esac
}

# Claim a port only once the server answers on it. Every branch below used to print the
# port it had guessed, whether or not anything was listening there.
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
    PYTHONPATH="$SCRIPT_DIR" "$VENV_PYTHON" -m stt.optional_deps --repo-dir "$SCRIPT_DIR" --data-dir "$DATA_DIR" 2>&1
fi

# ─── Fast path: launchd KeepAlive supervisor (macOS) ────────────────
# When the server is supervised by launchd with KeepAlive — a system daemon
# (/Library/LaunchDaemons/com.stt.server.plist) or a per-user gui agent — killing
# the worker IS a full restart: launchd respawns it with the current on-disk code.
# We must NOT also start a second instance. That was the double-launch bug: a
# *system* daemon isn't visible to a user-context `launchctl list`, so the script
# fell through to its nohup fallback and launched a rival that fought for the port.
if [ "$OS" = "Darwin" ] && { [ -f /Library/LaunchDaemons/com.stt.server.plist ] \
        || launchctl print "gui/$(id -u)/com.stt.server" >/dev/null 2>&1; }; then
    echo "Restarting via launchd (KeepAlive respawn)..."
    pkill -TERM -f "speech_to_text\.py" 2>/dev/null
    sleep 4
    for _ in $(seq 1 15); do
        if pgrep -f "speech_to_text\.py" >/dev/null 2>&1; then
            report_started "launchd respawn"
            exit 0
        fi
        sleep 1
    done
    echo -e "${RED}[ERROR]${NC} launchd did not respawn the server — check: launchctl print system/com.stt.server"
    exit 1
fi

echo "Stopping server..."

# ─── Stop managed services ──────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    systemctl stop stt-server 2>/dev/null
    systemctl stop stt 2>/dev/null
elif [ "$OS" = "Darwin" ]; then
    launchctl stop com.stt.server 2>/dev/null
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
sleep 1

# ─── Wait for clean shutdown ────────────────────────────────────────
RETRIES=0
while pgrep -f "speech_to_text\.py" > /dev/null 2>&1; do
    echo "Waiting for processes to stop..."
    pkill -9 -f "speech_to_text\.py" 2>/dev/null
    if [ "$OS" = "Linux" ]; then
        fuser -k "$PORT/tcp" 2>/dev/null
    elif [ "$OS" = "Darwin" ]; then
        lsof -ti :"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null
    fi
    sleep 1
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge 10 ]; then
        echo -e "${RED}[ERROR]${NC} Could not stop all processes after 10 attempts"
        break
    fi
done
echo "All server processes stopped"
return_data_dir
sleep 2

# ─── Start server ───────────────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    # Try systemd first
    for service_name in stt-server stt; do
        if systemctl list-unit-files "${service_name}.service" 2>/dev/null | grep -q "$service_name"; then
            echo "Starting via systemd ($service_name)..."
            systemctl start "$service_name"
            sleep 3
            if systemctl is-active --quiet "$service_name" 2>/dev/null; then
                report_started "$service_name"
                exit 0
            fi
        fi
    done
elif [ "$OS" = "Darwin" ]; then
    # Try launchd first
    if launchctl list com.stt.server &> /dev/null; then
        echo "Starting via launchd..."
        launchctl start com.stt.server
        sleep 3
        report_started "launchd"
        exit 0
    fi
fi

# Fallback: start manually
echo "No managed service found, starting manually..."
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

# STT_DATA_DIR: as root the server would otherwise read /root/.stt — a different config,
# models and sessions from the ones this script just resolved.
STT_DATA_DIR="$DATA_DIR" nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &

# ─── Verify ──────────────────────────────────────────────────────────
sleep 3
if pgrep -f "speech_to_text\.py" > /dev/null; then
    report_started "manual"
else
    echo -e "${RED}[ERROR]${NC} Failed to start server. Check server.log or journalctl -u stt-server"
    exit 1
fi
