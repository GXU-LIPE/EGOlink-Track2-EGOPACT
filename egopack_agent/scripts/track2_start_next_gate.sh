#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
cd "$CODEX"
mkdir -p logs state

RUN_VERSION="${1:-V6_1_5_kitchen_branch_repair}"
RUN_ID="${2:-gpt55_next_gate_$(date +%Y%m%d_%H%M%S)}"
GATE_SPECS="${3:-retail:9 kitchen:2 restaurant:4 order:1}"

TRACK2_RUN_VERSION="$RUN_VERSION" \
TRACK2_RUN_ID="$RUN_ID" \
TRACK2_GATE_SPECS="$GATE_SPECS" \
nohup bash scripts/run_gpt55_endpoint_gate.sh > logs/gpt55_next_gate_latest.log 2>&1 &

PID=$!
echo "$PID" > state/gpt55_next_gate.pid
cat > state/gpt55_next_gate_launch.json <<JSON
{
  "pid": $PID,
  "run_id": "$RUN_ID",
  "version": "$RUN_VERSION",
  "gate_specs": "$GATE_SPECS",
  "log": "logs/gpt55_next_gate_latest.log",
  "started_at": "$(date +%Y-%m-%dT%H:%M:%S%z)"
}
JSON
echo "$RUN_ID $PID"
