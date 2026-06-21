#!/usr/bin/env bash
set -u
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
mkdir -p logs
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export TRACK2_V8_TASK_TIMEOUT=900
run_id="v8_2_valA_small_20260617_continue"
log="logs/${run_id}.launcher.log"
nohup python3 scripts/run_v8_validation.py --stage validation_A_small --version V8_2_kitchen_helper --run-id "$run_id" --limit-per-scenario 5 > "$log" 2>&1 &
pid=$!
echo "$pid" > "state/${run_id}.pid"
echo "started pid=$pid log=$log run_id=$run_id"
