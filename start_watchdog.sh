#!/bin/bash
# Start the STT Watchdog in headless mode.
# Works with both the compiled binary and Python source installs.
# The watchdog manages STT: starts it, restarts on crash, and auto-updates daily at 1am.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/STT-Watchdog"

# The watchdog logs here in both modes (stt/watchdog.py LOG_DIR), never the checkout.
LOG_DIR="$HOME/.stt/logs"
mkdir -p "$LOG_DIR"

# Prevent double-start: a running watchdog holds its single-instance lock on 57337. Asked
# with python, not nc, which many installs lack — without it the check never fired, and a
# second watchdog exited on the lock while this script printed "[OK] started".
if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',57337))==0 else 1)" 2>/dev/null; then
    echo "[INFO] Watchdog is already running."
    exit 0
fi

# stdout goes to its own file: the watchdog already writes watchdog.log itself, so
# appending stdout there doubled every line and kept writing to the file after rotation.

if [ -f "$BINARY" ]; then
    # ── Compiled binary ──────────────────────────────────────────────────────
    nohup "$BINARY" --headless >> "$LOG_DIR/watchdog.stdout.log" 2>&1 &
    echo "[OK] Watchdog started (PID $!) — binary mode"
    echo "     Logs: tail -f $LOG_DIR/watchdog.log"
    echo "     STT:  tail -f $LOG_DIR/stt.log"
else
    # ── Python source fallback ───────────────────────────────────────────────
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    [ ! -f "$PYTHON_BIN" ] && PYTHON_BIN="python3"
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/stt/watchdog.py" --headless >> "$LOG_DIR/watchdog.stdout.log" 2>&1 &
    echo "[OK] Watchdog started (PID $!) — Python source mode"
    echo "     Logs: tail -f $LOG_DIR/watchdog.log"
    echo "     STT:  tail -f $LOG_DIR/stt.log"
fi
