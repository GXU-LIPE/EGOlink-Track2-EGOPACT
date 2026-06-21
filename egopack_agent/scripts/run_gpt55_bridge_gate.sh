#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
PYTHON_BIN="${TRACK2_PYTHON:-python3}"
RUN_ID="${TRACK2_RUN_ID:-gpt55_bridge_gate_$(date +%Y%m%d_%H%M%S)}"
MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
BRIDGE_BASE="${TRACK2_LOCAL_BRIDGE_BASE:-http://127.0.0.1:17901/v1}"

mkdir -p "$CODEX"/{logs,reports,analysis,state,runs}
cd "$CODEX"

export CODEX_ROOT="$CODEX"
export EGO_ROOT="$EGO"
export SERVICE_MODEL_BACKEND=""
export TRACK2_USE_OPENAI_GPT55=0
export SERVICE_MODEL_NAME="$MODEL"
export SERVICE_MODEL_API_BASE="$BRIDGE_BASE"
export SERVICE_MODEL_API_KEY="local-bridge"
export USER_AGENT_API_BASE_URL="$BRIDGE_BASE"
export USER_AGENT_API_KEY="local-bridge"
export USER_MODEL_NAME="$MODEL"
export TRACK2_USER_USE_OPENAI=0
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_VISUAL_CACHE=1
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export TRACK2_USE_VIDEO=0
export TRACK2_MAX_TURNS="${TRACK2_MAX_TURNS:-8}"
export TRACK2_DEFAULT_MAX_TOKENS="${TRACK2_DEFAULT_MAX_TOKENS:-2048}"
export TRACK2_CONNECT_TIMEOUT=5
export TRACK2_READ_TIMEOUT="${TRACK2_READ_TIMEOUT:-220}"
export TRACK2_API_MAX_RETRIES="${TRACK2_API_MAX_RETRIES:-1}"
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"

if ! curl -sS --noproxy '*' --connect-timeout 5 --max-time 15 "$BRIDGE_BASE/../health" >/dev/null; then
  cat > "$CODEX/reports/02_gpt55_gate_summary_${RUN_ID}.md" <<MD
# GPT-5.5 Bridge Gate Summary

- generated_at: $(date +%Y-%m-%dT%H:%M:%S%z)
- run_id: $RUN_ID
- status: bridge_unreachable
- bridge_base: $BRIDGE_BASE
- note: local bridge or SSH reverse tunnel is not reachable from remote.
MD
  exit 2
fi

run_version() {
  local version="$1"
  export TRACK2_RUN_VERSION="$version"
  export TRACK2_RUN_ID="$RUN_ID"
  export TRACK2_OUTPUT_MODEL_NAME="${MODEL}-${version}-${RUN_ID}"
  mkdir -p "$CODEX/runs/$version/$RUN_ID/logs"
  cd "$EGO"
  for spec in retail:9 kitchen:2 restaurant:4 order:1; do
    scenario="${spec%%:*}"
    num="${spec##*:}"
    "$PYTHON_BIN" "$CODEX/runners/track2_multi_agent_plus.py" \
      --scenario "$scenario" --scenario_number "$num" \
      --service_model_name "$MODEL" \
      --multi_agent_user --summary_user --num_tasks 1 \
      > "$CODEX/runs/$version/$RUN_ID/logs/${scenario}${num}.log" 2>&1 || true
  done
  bash analysis_scripts/run_eval.sh --model_name "$TRACK2_OUTPUT_MODEL_NAME" --num_samples 1 \
    > "$CODEX/runs/$version/$RUN_ID/logs/eval.log" 2>&1 || true
}

run_version V6_1_gpt55_guarded_bridge

cd "$CODEX"
"$PYTHON_BIN" "$CODEX/scripts/track2_gpt55_collect_gate.py" --run-id "$RUN_ID" --model "$MODEL" || true

cat > "$CODEX/state/latest_gpt55_bridge_gate.json" <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "run_id": "$RUN_ID",
  "model": "$MODEL",
  "version": "V6_1_gpt55_guarded_bridge",
  "bridge_base": "$BRIDGE_BASE",
  "report": "reports/02_gpt55_gate_summary_${RUN_ID}.md"
}
JSON
