#!/usr/bin/env bash
#
# One-time relocation of a run-from-repo STT install's data into ~/.stt, to
# match the new code that resolves APP_DIR to ~/.stt for every non-override run.
#
# Designed for a disk-tight box: the big directories (models, session DBs) are
# MOVED, not copied — on the same filesystem that is an instant rename using
# zero extra space. The small live config is copied so the checkout keeps its
# tracked *.default.json templates (BUNDLE_DIR still seeds from <repo>/config).
#
# Run it AS THE SERVICE USER (e.g. ai) — NOT under sudo — so the relocated data
# stays owned by that user and the server can keep writing it. The two
# `systemctl` lines escalate with sudo on their own and will prompt for your
# password.
#
#   ssh ai@<host>
#   bash migrate_data_to_home_stt.sh
#
# Override paths if your layout differs:
#   OLD_DIR=/home/ai/STT NEW_DIR=/home/ai/.stt SERVICE=stt-server bash migrate_data_to_home_stt.sh
#
set -eu

OLD_DIR="${OLD_DIR:-$HOME/STT}"
NEW_DIR="${NEW_DIR:-$HOME/.stt}"
SERVICE="${SERVICE:-stt-server}"
STATUS_URL="${STATUS_URL:-http://127.0.0.1/api/transcription/status}"

echo "Old data dir : $OLD_DIR"
echo "New data dir : $NEW_DIR"
echo "Service      : $SERVICE"
echo

# --- guards -----------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
  echo "ABORT: run this as the service user (e.g. ai), not root/sudo — otherwise"
  echo "       the relocated data becomes root-owned and the server can't write it."
  exit 1
fi
if [ ! -d "$OLD_DIR/.git" ]; then
  echo "ABORT: $OLD_DIR is not a git checkout; check OLD_DIR."; exit 1
fi
if curl -s --max-time 5 "$STATUS_URL" | grep -q '"running": *true'; then
  echo "ABORT: a transcription is currently running — try again when it is stopped."
  exit 1
fi

# --- 1. stop the service ----------------------------------------------------
echo "[1/5] stopping $SERVICE (sudo will prompt)..."
sudo systemctl stop "$SERVICE"
sleep 1
if systemctl is-active --quiet "$SERVICE"; then
  echo "ABORT: $SERVICE is still active after stop."; exit 1
fi
echo "      stopped."

# From here the service is down; on any error, start it again before exiting.
trap 'echo "ERROR — restarting $SERVICE so the box is not left down."; sudo systemctl start "$SERVICE" || true' ERR

# --- 2. fast-forward the checkout ------------------------------------------
echo "[2/5] git pull --ff-only ..."
git -C "$OLD_DIR" pull --ff-only
echo "      now at: $(git -C "$OLD_DIR" log --oneline -1)"

# --- 3. move the big/data dirs (instant, same-filesystem rename) ------------
echo "[3/5] relocating data to $NEW_DIR ..."
mkdir -p "$NEW_DIR/config"
for d in models _AUTOMATIC_BACKUP panns_data logs; do
  if [ -e "$OLD_DIR/$d" ] && [ ! -e "$NEW_DIR/$d" ]; then
    mv "$OLD_DIR/$d" "$NEW_DIR/$d"; echo "      moved $d"
  fi
done
for f in download_progress.json server.log; do
  if [ -e "$OLD_DIR/$f" ] && [ ! -e "$NEW_DIR/$f" ]; then
    mv "$OLD_DIR/$f" "$NEW_DIR/$f"; echo "      moved $f"
  fi
done

# --- 4. copy live config; keep the checkout's tracked defaults --------------
echo "[4/5] copying live config (checkout keeps its *.default.json) ..."
cp -a "$OLD_DIR/config/." "$NEW_DIR/config/"
echo "manual move $(date -u +%FT%TZ) from $OLD_DIR" > "$NEW_DIR/.migrated_from_repo"

# --- 5. start the service ---------------------------------------------------
echo "[5/5] starting $SERVICE (sudo will prompt)..."
trap - ERR
sudo systemctl start "$SERVICE"
sleep 5
if systemctl is-active --quiet "$SERVICE"; then
  echo "      ACTIVE."
else
  echo "WARNING: $SERVICE did not report active — check: journalctl -u $SERVICE -n 50"
  exit 1
fi

echo
echo "Done. Data now under $NEW_DIR (moved, so the old space is already reclaimed;"
echo "only the tiny $OLD_DIR/config remains on purpose — it holds tracked defaults)."
echo "Verify:"
echo "  curl -s $STATUS_URL"
echo "  ls -la $NEW_DIR"
echo "  journalctl -u $SERVICE -n 30 --no-pager   # confirm no [MIGRATE]/errors on boot"
