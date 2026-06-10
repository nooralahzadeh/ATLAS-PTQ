#!/bin/bash
# Auto-snapshot + push of code/docs to the git remote — defense against another
# scratch incident. Stages ONLY code/doc paths (never large *.pt / data dumps,
# which .gitignore already excludes), commits if there are changes, then pushes.
#
# Run manually (bash scripts/auto_push.sh) or on a timer (systemd --user, see
# scripts/systemd/). Safe to run repeatedly: no-op when there is nothing new.
set -uo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
cd "$ROOT" || exit 1

LOG="$ROOT/tacq_data/logs/auto_push.log"
mkdir -p "$(dirname "$LOG")"
exec 9>"$ROOT/.auto_push.lock"
flock -n 9 || { echo "[$(date -u +%FT%TZ)] another auto_push is running; skip" >>"$LOG"; exit 0; }

say() { echo "[$(date -u +%FT%TZ)] $*" >>"$LOG"; }
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"

# Stage tracked modifications/deletions + small code/doc trees only.
git add -u
for p in scripts paper recovery/recon recovery/RECOVERY_STATUS.md \
         data/contrastive .gitignore .cursorrules .cursor; do
  [ -e "$p" ] && git add "$p" 2>/dev/null
done

if git diff --cached --quiet; then
  say "no staged code/doc changes; checking for unpushed commits"
else
  git commit -q -m "auto-snapshot $(date -u +%FT%TZ) @$(hostname -s)" \
    && say "committed snapshot" || say "commit failed"
fi

# Nothing to push?
if git rev-parse --verify -q "origin/$BRANCH" >/dev/null 2>&1 \
   && [ -z "$(git log --oneline "origin/$BRANCH..HEAD" 2>/dev/null)" ]; then
  say "nothing to push (up to date with origin/$BRANCH)"; exit 0
fi

# Integrate remote, then push (non-interactive; requires SSH key or stored token).
GIT_TERMINAL_PROMPT=0 git pull --rebase --autostash origin "$BRANCH" >>"$LOG" 2>&1 \
  || say "WARN: pull --rebase failed (resolve manually); attempting push anyway"
if GIT_TERMINAL_PROMPT=0 git push origin "$BRANCH" >>"$LOG" 2>&1; then
  say "pushed to origin/$BRANCH OK"
else
  say "ERROR: push failed — is GitHub auth configured? (SSH key or PAT)"
  exit 1
fi
