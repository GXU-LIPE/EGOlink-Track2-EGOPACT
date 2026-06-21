#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
cd "$CODEX"
mkdir -p logs state

TS="$(date +%Y%m%d_%H%M%S)"
GT_RUN_ID="v14_gt_distill_${TS}"
DIST_RUN_ID="v14_distilled_val41_${TS}"

python3 scripts/v14_gt_trajectory_distillation.py --run-id "$GT_RUN_ID" \
  > "logs/${GT_RUN_ID}.log" 2>&1 &
GT_PID=$!

TRACK2_V14_WORKERS="${TRACK2_V14_WORKERS:-10}" \
python3 scripts/run_v14_distilled_val41_parallel.py --run-id "$DIST_RUN_ID" --max-workers "${TRACK2_V14_WORKERS:-10}" \
  > "logs/${DIST_RUN_ID}.log" 2>&1 &
DIST_PID=$!

cat > state/latest_v14_launch.json <<JSON
{
  "started_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "gt_run_id": "$GT_RUN_ID",
  "gt_pid": $GT_PID,
  "gt_log": "$CODEX/logs/${GT_RUN_ID}.log",
  "distilled_run_id": "$DIST_RUN_ID",
  "distilled_pid": $DIST_PID,
  "distilled_log": "$CODEX/logs/${DIST_RUN_ID}.log",
  "workers": ${TRACK2_V14_WORKERS:-10},
  "final_run": false
}
JSON

echo "$GT_RUN_ID"
echo "$DIST_RUN_ID"
