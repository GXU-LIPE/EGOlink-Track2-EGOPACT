#!/usr/bin/env bash
set -euo pipefail
CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
TS=$(date +%Y%m%d_%H%M%S)
SNAPROOT="$CODEX/official_snapshots"
SNAP="$SNAPROOT/egolink2026_$TS"
REPORT="$CODEX/reports/OFFICIAL_UPDATE_AUDIT_$TS.md"
FINAL_REPORT="$CODEX/reports/FINAL_STAGE_SUBMISSION_GUIDE_$TS.md"
TOP1_REPORT="$CODEX/reports/TOP1_READINESS_ANALYSIS_$TS.md"
mkdir -p "$CODEX/reports" "$CODEX/analysis" "$CODEX/patches" "$CODEX/backups" "$SNAPROOT"
LOG="$CODEX/logs/official_update_audit_$TS.log"
mkdir -p "$CODEX/logs"
{
  echo "[$(date -Is)] starting official audit"
  echo "host=$(hostname)"
  echo "ego=$EGO"
  echo "codex=$CODEX"
} > "$LOG"
if command -v git >/dev/null 2>&1; then
  git clone --depth 1 https://github.com/ego-link/egolink2026.git "$SNAP" >> "$LOG" 2>&1
else
  echo "git not found" >> "$LOG"
  exit 2
fi
python3 "$CODEX/scripts/track2_official_update_audit.py" \
  --ego "$EGO" \
  --snapshot "$SNAP/code/track2/EgoBench" \
  --codex "$CODEX" \
  --timestamp "$TS" \
  --report "$REPORT" \
  --final-report "$FINAL_REPORT" \
  --top1-report "$TOP1_REPORT" >> "$LOG" 2>&1
printf '%s\n' "$REPORT"
printf '%s\n' "$FINAL_REPORT"
printf '%s\n' "$TOP1_REPORT"
