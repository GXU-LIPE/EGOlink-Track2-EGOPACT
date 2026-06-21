#!/usr/bin/env bash
set -euo pipefail
CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
TS=$(date +%Y%m%d_%H%M%S)
SNAP="$CODEX/official_snapshots/raw_$TS/code/track2/EgoBench"
BASE="https://raw.githubusercontent.com/ego-link/egolink2026/main/code/track2/EgoBench"
LOG="$CODEX/logs/official_raw_snapshot_$TS.log"
mkdir -p "$SNAP" "$CODEX/logs" "$CODEX/reports" "$CODEX/analysis" "$CODEX/patches" "$CODEX/backups"
export HTTPS_PROXY=
export HTTP_PROXY=
export https_proxy=
export http_proxy=
export ALL_PROXY=
export all_proxy=
files=(
README.md
run_all_scenarios.sh
run/multi_agent.py
run/prompts.py
run/utils.py
analysis_scripts/evaluate_interaction.py
analysis_scripts/run_eval.sh
analysis_scripts/analyze_error_reasons.py
config/service_agent_config.py
config/user_agent_config.py
tools/retail/retail_db.py
tools/retail/retail_init.py
tools/retail/retail_tools.json
tools/order/order_db.py
tools/order/order_init.py
tools/order/order_tools.json
tools/restaurant/restaurant_db.py
tools/restaurant/restaurant_init.py
tools/restaurant/restaurant_tools.json
tools/kitchen/kitchen_db.py
tools/kitchen/kitchen_init.py
tools/kitchen/kitchen_tools.json
scenarios/final/retail6.json
scenarios/final/retail10.json
scenarios/final/kitchen4.json
scenarios/final/restaurant5.json
scenarios/final/order2.json
)
{
  echo "[$(date -Is)] raw snapshot start"
  echo "base=$BASE"
} > "$LOG"
for rel in "${files[@]}"; do
  mkdir -p "$SNAP/$(dirname "$rel")"
  echo "download $rel" >> "$LOG"
  if ! curl -fL --retry 3 --retry-delay 2 --max-time 90 "$BASE/$rel" -o "$SNAP/$rel" >> "$LOG" 2>&1; then
    echo "FAILED $rel" >> "$LOG"
  fi
done
python3 "$CODEX/scripts/track2_official_update_audit.py" \
  --ego "$EGO" \
  --snapshot "$SNAP" \
  --codex "$CODEX" \
  --timestamp "$TS" \
  --report "$CODEX/reports/OFFICIAL_UPDATE_AUDIT_$TS.md" \
  --final-report "$CODEX/reports/FINAL_STAGE_SUBMISSION_GUIDE_$TS.md" \
  --top1-report "$CODEX/reports/TOP1_READINESS_ANALYSIS_$TS.md" >> "$LOG" 2>&1
echo "$CODEX/reports/OFFICIAL_UPDATE_AUDIT_$TS.md"
echo "$CODEX/reports/FINAL_STAGE_SUBMISSION_GUIDE_$TS.md"
echo "$CODEX/reports/TOP1_READINESS_ANALYSIS_$TS.md"
