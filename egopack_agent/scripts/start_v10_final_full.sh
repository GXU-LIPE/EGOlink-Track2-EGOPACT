#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
mkdir -p state logs
nohup bash scripts/run_v10_final_full.sh > logs/v10_final_full_launcher_20260618_1940.log 2>&1 &
echo $! > state/v10_final_full_20260618_1940.pid
sleep 2
echo "started_pid=$(cat state/v10_final_full_20260618_1940.pid)"
tail -n 20 logs/v10_final_full_launcher_20260618_1940.log || true
