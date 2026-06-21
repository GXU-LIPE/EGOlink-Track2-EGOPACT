#!/usr/bin/env bash
set -u
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
mkdir -p logs state
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export TRACK2_V8_TASK_TIMEOUT=900
run_id="V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_0925"
log="logs/${run_id}.launcher.log"
nohup python3 scripts/run_v8_validation.py --stage validation_A_medium --version V9_4_5_visual_retrieval_fix --run-id "$run_id" --limit-per-scenario 30 > "$log" 2>&1 &
pid=$!
echo "$pid" > "state/${run_id}.pid"
echo "started pid=$pid log=$log run_id=$run_id"
