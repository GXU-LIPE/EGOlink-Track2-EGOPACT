#!/usr/bin/env bash
set -euo pipefail

CODEX_ROOT="${CODEX_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/codex}"
source "$CODEX_ROOT/scripts/track2_common.sh"

RESUME=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --full-matrix) FULL_MATRIX=1; shift ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

RUN_ID="${RUN_ID:-track2_$(date +%Y%m%d_%H%M%S)}"
FULL_MATRIX="${FULL_MATRIX:-0}"
LOG="$CODEX_ROOT/logs/autopilot_${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

log_msg "autopilot_start run_id=$RUN_ID resume=$RESUME dry_run=$DRY_RUN"
echo "$$" > "$CODEX_ROOT/state/autopilot.pid"
print_runtime_brief

PY="$(pick_python)"

if [ "$DRY_RUN" -eq 1 ]; then
  "$PY" "$CODEX_ROOT/scripts/track2_autopilot.py" --run-id "$RUN_ID" --resume --dry-run
elif [ "$FULL_MATRIX" -eq 1 ]; then
  "$PY" "$CODEX_ROOT/scripts/track2_autopilot.py" --run-id "$RUN_ID" --resume --full-matrix
else
  "$PY" "$CODEX_ROOT/scripts/track2_autopilot.py" --run-id "$RUN_ID" --resume
fi

log_msg "autopilot_done run_id=$RUN_ID"
