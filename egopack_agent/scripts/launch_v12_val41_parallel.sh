#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
cd "$CODEX"

mkdir -p logs state
RUN_ID="V12_qwen3vl_prior_all_modules_val41_parallel_$(date +%Y%m%d_%H%M%S)"
LOG="logs/v12_val41_parallel_${RUN_ID}.log"
ln -sfn "$(basename "$LOG")" logs/v12_val41_parallel_latest.log

cat > state/v12_val41_parallel_launch_latest.json <<JSON
{
  "run_id": "$RUN_ID",
  "log": "$CODEX/$LOG",
  "started_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "workers": ${TRACK2_V12_VAL41_WORKERS:-10}
}
JSON

TRACK2_V12_VAL41_WORKERS="${TRACK2_V12_VAL41_WORKERS:-10}" \
nohup python3 scripts/run_v12_val41_parallel.py \
  --run-id "$RUN_ID" \
  --max-workers "${TRACK2_V12_VAL41_WORKERS:-10}" \
  > "$LOG" 2>&1 < /dev/null &

echo "$RUN_ID"
echo "$CODEX/$LOG"
