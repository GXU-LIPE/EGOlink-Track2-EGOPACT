#!/usr/bin/env bash
set -euo pipefail
CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
RUN_ID="${TRACK2_RUN_ID:-qwen_local_gate_$(date +%Y%m%d_%H%M%S)}"
export TRACK2_FINAL_COMPLIANT=1
export SERVICE_MODEL_BACKEND="${SERVICE_MODEL_BACKEND:-local_qwen_vllm}"
export SERVICE_MODEL_NAME="${SERVICE_MODEL_NAME:-Qwen2.5-VL-32B-Instruct}"
export SERVICE_MODEL_PATH="${SERVICE_MODEL_PATH:-/home/data-gxu/acm/egolink2026-main/code1/models/Qwen/Qwen2___5-VL-32B-Instruct}"
export SERVICE_MODEL_API_BASE="${SERVICE_MODEL_API_BASE:-http://127.0.0.1:8000/v1}"
export TRACK2_OUTPUT_MODEL_NAME="${TRACK2_OUTPUT_MODEL_NAME:-qwen25vl32b-local-guarded-$RUN_ID}"
export TRACK2_RUN_VERSION="${TRACK2_RUN_VERSION:-V2_5_local_qwen_guarded}"
export TRACK2_RUN_ID="$RUN_ID"
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"
cd "$EGO"
for spec in retail:9 kitchen:2 restaurant:4 order:1; do
  scenario="${spec%%:*}"
  number="${spec##*:}"
  python3 "$CODEX/runners/track2_multi_agent_plus.py" \
    --scenario "$scenario" --scenario_number "$number" \
    --service_model_name "$SERVICE_MODEL_NAME" \
    --multi_agent_user --summary_user --num_tasks 1
done
bash analysis_scripts/run_eval.sh --model_name "$TRACK2_OUTPUT_MODEL_NAME" --num_samples 1
