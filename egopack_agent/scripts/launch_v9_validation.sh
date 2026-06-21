#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
cd "$CODEX"

stage="${1:-validation_A_small}"
version="${2:-V9_1_scoring_prompt}"
limit="${3:-5}"
run_id="${4:-${version}_${stage}_$(date +%Y%m%d_%H%M%S)}"
log="logs/${run_id}.launcher.log"

mkdir -p logs state runs

# GPT-5.5 ai-pixel route works direct from the remote; stale local proxy breaks it.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

nohup python3 scripts/run_v8_validation.py \
  --stage "$stage" \
  --version "$version" \
  --run-id "$run_id" \
  --limit-per-scenario "$limit" \
  > "$log" 2>&1 &

pid=$!
echo "$pid" > "state/${run_id}.pid"
cat > "state/${run_id}.launch.json" <<JSON
{
  "run_id": "$run_id",
  "stage": "$stage",
  "version": "$version",
  "limit_per_scenario": "$limit",
  "pid": $pid,
  "log": "$log",
  "started_at": "$(date +%Y-%m-%dT%H:%M:%S%z)"
}
JSON

echo "RUN_ID=$run_id"
echo "PID=$pid"
echo "LOG=$log"
