#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
run_id="V9_5_memory_deepseek_rerank_validation_A_medium_$(date +%Y%m%d_%H%M)"
log="logs/${run_id}.launcher.log"
mkdir -p logs
nohup python3 scripts/run_v8_validation.py \
  --stage validation_A_medium \
  --version V9_5_memory_deepseek_rerank \
  --run-id "$run_id" \
  --limit-per-scenario 30 \
  > "$log" 2>&1 &
echo "run_id=$run_id"
echo "pid=$!"
echo "log=$log"
