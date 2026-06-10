#!/bin/bash
# Install the auto-push systemd --user timer (~3x/day). Run once, AFTER GitHub
# auth is configured (SSH key registered, or PAT in git credential store).
#   bash scripts/systemd/install_autopush.sh
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HOME/.config/systemd/user"
mkdir -p "$DST"
cp "$SRC/atlas-autopush.service" "$SRC/atlas-autopush.timer" "$DST/"

# Let the timer fire even when not logged in (best-effort; HPC login nodes vary).
loginctl enable-linger "$USER" 2>/dev/null || echo "[warn] enable-linger not permitted; timer runs only while logged in"

systemctl --user daemon-reload
systemctl --user enable --now atlas-autopush.timer
echo "=== installed. next runs: ==="
systemctl --user list-timers atlas-autopush.timer --no-pager || true
echo "Manual test: systemctl --user start atlas-autopush.service && tail tacq_data/logs/auto_push.log"
