#!/usr/bin/env bash
set -euo pipefail
CODEX_ROOT="${CODEX_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/codex}"
echo "CODEX_ROOT=$CODEX_ROOT"
date -Is || true
hostname || true
if command -v tmux >/dev/null 2>&1; then
  tmux ls 2>/dev/null || true
fi
if [ -f "$CODEX_ROOT/state/autopilot.pid" ]; then
  pid="$(cat "$CODEX_ROOT/state/autopilot.pid" || true)"
  echo "autopilot.pid=$pid"
  [ -n "$pid" ] && ps -p "$pid" -o pid,ppid,stat,etime,cmd || true
fi
if [ -f "$CODEX_ROOT/state/watchdog.pid" ]; then
  pid="$(cat "$CODEX_ROOT/state/watchdog.pid" || true)"
  echo "watchdog.pid=$pid"
  [ -n "$pid" ] && ps -p "$pid" -o pid,ppid,stat,etime,cmd || true
fi
echo "recent logs:"
ls -lt "$CODEX_ROOT/logs" 2>/dev/null | head -20 || true
echo "state:"
ls -lt "$CODEX_ROOT/state" 2>/dev/null | head -20 || true
if [ -f "$CODEX_ROOT/README_STATUS.md" ]; then
  sed -n '1,120p' "$CODEX_ROOT/README_STATUS.md"
fi
