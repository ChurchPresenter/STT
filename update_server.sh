#!/bin/bash

# Speech-to-Text Update Script (Linux & macOS)
# Pull the latest code and restart the server to apply it NOW, instead of
# waiting for the nightly auto-update. Restart is delegated to restart_server.sh
# (no duplicated logic). Run as the checkout owner; on systemd Linux run with
# sudo, like restart_server.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "[UPDATE] Pulling latest code (git pull --ff-only)..."
if ! git -C "$SCRIPT_DIR" pull --ff-only; then
    echo -e "${RED}[ERROR]${NC} git pull --ff-only failed."
    echo "  The working tree is probably dirty or the branch has diverged/unpushed"
    echo "  commits. Commit/stash your changes (or push) and try again — nothing was"
    echo "  changed and the server was NOT restarted."
    exit 1
fi

echo -e "${GREEN}[UPDATE]${NC} Now at: $(git -C "$SCRIPT_DIR" log --oneline -1)"

# Sync dependencies when requirements.txt changed, so a pull that adds a new
# package (e.g. psutil) doesn't leave the venv behind. Hash-gated against the
# same marker the in-process auto-updater uses (stt/self_update.py), so the two
# never double-install. Best-effort: a failure here must not block the restart.
sync_deps() {
    local req="$SCRIPT_DIR/requirements.txt"
    local marker="$SCRIPT_DIR/.venv/.requirements-synced"
    [ -f "$req" ] || return 0

    local sha
    if command -v sha256sum >/dev/null 2>&1; then
        sha="$(sha256sum "$req" | cut -d' ' -f1)"
    elif command -v shasum >/dev/null 2>&1; then
        sha="$(shasum -a 256 "$req" | cut -d' ' -f1)"
    else
        echo -e "${YELLOW}[UPDATE]${NC} No sha256 tool; skipping dependency hash check."
        return 0
    fi

    if [ -f "$marker" ] && [ "$(tr -d '[:space:]' < "$marker")" = "$sha" ]; then
        return 0  # venv already matches requirements.txt
    fi

    local uv_bin
    uv_bin="$(command -v uv || true)"
    [ -z "$uv_bin" ] && [ -x "$HOME/.local/bin/uv" ] && uv_bin="$HOME/.local/bin/uv"
    [ -z "$uv_bin" ] && [ -x "$SCRIPT_DIR/.venv/bin/uv" ] && uv_bin="$SCRIPT_DIR/.venv/bin/uv"
    if [ -z "$uv_bin" ]; then
        echo -e "${YELLOW}[UPDATE]${NC} uv not found; skipping dependency sync (run install.sh to update deps)."
        return 0
    fi

    echo "[UPDATE] requirements.txt changed — syncing dependencies..."
    if "$uv_bin" pip install --python "$SCRIPT_DIR/.venv/bin/python3" -r "$req"; then
        echo "$sha" > "$marker" 2>/dev/null || true
        echo -e "${GREEN}[UPDATE]${NC} Dependencies synced."
    else
        echo -e "${YELLOW}[UPDATE]${NC} Dependency sync failed; restarting anyway (optional deps degrade gracefully)."
    fi
}
sync_deps

echo "[UPDATE] Restarting to apply..."
exec bash "$SCRIPT_DIR/restart_server.sh"
