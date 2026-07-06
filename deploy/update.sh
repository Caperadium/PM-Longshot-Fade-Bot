#!/usr/bin/env bash
#
# update.sh — safe in-place update for the Fader Bot on a VPS.
#
# Stops the engine gracefully, backs up the DB, pulls new code, refreshes
# deps, runs the test suite, and only then restarts. Aborts (leaving the
# engine stopped) if tests fail, so broken code never touches live money.
#
# Usage (run as a user that can sudo systemctl and read the repo):
#   cd /opt/fader-bot
#   ./deploy/update.sh              # pull current branch, update, restart
#   ./deploy/update.sh v1.4.0       # checkout a tag/branch/ref, then update
#   SKIP_TESTS=1 ./deploy/update.sh # emergency: skip the pytest gate
#
# Assumptions (match deploy/VPS_DEPLOY.md):
#   - repo at /opt/fader-bot, venv at /opt/fader-bot/.venv
#   - systemd units fader-engine (+ optional fader-dashboard)
#   - runs the git/pip steps as the $BOT_USER account that owns the repo

set -euo pipefail

# --- config (override via env) ---------------------------------------------
REPO_DIR="${REPO_DIR:-/opt/fader-bot}"
VENV="${VENV:-$REPO_DIR/.venv}"
BOT_USER="${BOT_USER:-fader}"
DB="${DB:-$REPO_DIR/fader/fader.db}"
BACKUP_DIR="${BACKUP_DIR:-$REPO_DIR/backups}"
REQ="${REQ:-$REPO_DIR/fader/requirements.txt}"
ENGINE_SVC="${ENGINE_SVC:-fader-engine}"
DASH_SVC="${DASH_SVC:-fader-dashboard}"
REF="${1:-}"                        # optional git ref to checkout
SKIP_TESTS="${SKIP_TESTS:-0}"

log()  { printf '\033[1;34m[update]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n'  "$*" >&2; exit 1; }

# run a command as the repo-owning user
asbot() { sudo -u "$BOT_USER" "$@"; }

cd "$REPO_DIR" || die "REPO_DIR $REPO_DIR not found"

# --- 0. record current commit so we can roll back --------------------------
OLD_REF="$(asbot git rev-parse HEAD)"
log "current commit: $OLD_REF"

# --- 1. graceful stop ------------------------------------------------------
# SIGTERM triggers cancel_all_resting() + telegram stop alert; unit gives 30s.
log "stopping $ENGINE_SVC (graceful, cancels resting orders)..."
sudo systemctl stop "$ENGINE_SVC"

# --- 2. backup DB (WAL-safe hot backup) ------------------------------------
if [[ -f "$DB" ]]; then
  mkdir -p "$BACKUP_DIR"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  BK="$BACKUP_DIR/pre-update-$STAMP.db"
  log "backing up DB -> $BK"
  # .backup checkpoints WAL into a consistent single-file snapshot
  asbot sqlite3 "$DB" ".backup '$BK'" || die "DB backup failed; aborting (engine left stopped)"
else
  warn "no DB at $DB (fresh install?) — skipping backup"
fi

# --- 3. pull new code ------------------------------------------------------
log "fetching..."
asbot git fetch --all --tags --prune
if [[ -n "$REF" ]]; then
  log "checking out $REF"
  asbot git checkout "$REF"
  asbot git pull --ff-only origin "$REF" 2>/dev/null || true   # branch: fast-forward; tag: no-op
else
  BRANCH="$(asbot git rev-parse --abbrev-ref HEAD)"
  log "fast-forward pull on $BRANCH"
  asbot git pull --ff-only origin "$BRANCH" \
    || die "pull not fast-forward (local edits or diverged). Resolve manually, then re-run."
fi
NEW_REF="$(asbot git rev-parse HEAD)"
log "now at: $NEW_REF"

if [[ "$OLD_REF" == "$NEW_REF" ]]; then
  warn "no new commits — restarting on same code"
fi

# --- 4. refresh deps if requirements changed -------------------------------
if [[ "$OLD_REF" != "$NEW_REF" ]] && ! asbot git diff --quiet "$OLD_REF" "$NEW_REF" -- "$REQ"; then
  log "requirements.txt changed — installing deps"
  asbot "$VENV/bin/pip" install -r "$REQ"
else
  log "requirements unchanged — skipping pip"
fi

# --- 5. test gate ----------------------------------------------------------
if [[ "$SKIP_TESTS" == "1" ]]; then
  warn "SKIP_TESTS=1 — bypassing pytest gate"
else
  log "running test suite (must pass before restart)..."
  if ! asbot "$VENV/bin/python" -m pytest fader/tests/ -q; then
    warn "TESTS FAILED. Rolling code back to $OLD_REF."
    asbot git checkout "$OLD_REF"
    die "rolled back to old commit. Engine left stopped — fix and re-run, or start manually."
  fi
  log "tests passed"
fi

# --- 6. restart ------------------------------------------------------------
log "starting $ENGINE_SVC..."
sudo systemctl start "$ENGINE_SVC"
# restart dashboard only if it's installed/enabled
if systemctl list-unit-files | grep -q "^$DASH_SVC"; then
  log "restarting $DASH_SVC..."
  sudo systemctl restart "$DASH_SVC" || warn "$DASH_SVC restart failed (non-fatal)"
fi

# --- 7. health check -------------------------------------------------------
sleep 3
if systemctl is-active --quiet "$ENGINE_SVC"; then
  log "$ENGINE_SVC is active. Update complete."
  log "watch startup: journalctl -u $ENGINE_SVC -f"
  log "rollback if needed: (cd $REPO_DIR && sudo -u $BOT_USER git checkout $OLD_REF) && sudo systemctl restart $ENGINE_SVC"
else
  die "$ENGINE_SVC did NOT come up. Check: journalctl -u $ENGINE_SVC -n 50"
fi
