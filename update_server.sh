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
echo "[UPDATE] Restarting to apply..."
exec bash "$SCRIPT_DIR/restart_server.sh"
