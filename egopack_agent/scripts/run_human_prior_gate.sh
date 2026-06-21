#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
PYTHON_BIN="${TRACK2_PYTHON:-python3}"
RUN_ID="${TRACK2_RUN_ID:-human_prior_gate_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$CODEX"/{logs,reports,analysis,state,runs,train_data}
cd "$CODEX"

if [[ -f "$CODEX/state/.openai_env" ]]; then
  # shellcheck disable=SC1091
  source "$CODEX/state/.openai_env"
fi

MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
BASE_URL="${TRACK2_OPENAI_BASE_URL:-https://ai-pixel.online/v1}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export CODEX_ROOT="$CODEX"
export EGO_ROOT="$EGO"
export SERVICE_MODEL_BACKEND="${SERVICE_MODEL_BACKEND:-openai_compatible_chat}"
export SERVICE_MODEL_NAME="$MODEL"
export SERVICE_MODEL_API_BASE="$BASE_URL"
export SERVICE_MODEL_API_KEY="${OPENAI_API_KEY:-}"
export USER_AGENT_API_BASE_URL="$BASE_URL"
export USER_AGENT_API_KEY="${OPENAI_API_KEY:-}"
export USER_MODEL_NAME="$MODEL"
export TRACK2_USER_USE_OPENAI=0
export TRACK2_USE_OPENAI_GPT55=0
export TRACK2_GPT55_STRUCTURED_OUTPUT=0
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_VISUAL_CACHE=1
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export TRACK2_USE_VIDEO=0
export TRACK2_ENABLE_HUMAN_PRIOR=1
export TRACK2_MAX_TURNS="${TRACK2_MAX_TURNS:-6}"
export TRACK2_DEFAULT_MAX_TOKENS="${TRACK2_DEFAULT_MAX_TOKENS:-2048}"
export TRACK2_CONNECT_TIMEOUT="${TRACK2_CONNECT_TIMEOUT:-10}"
export TRACK2_READ_TIMEOUT="${TRACK2_READ_TIMEOUT:-240}"
export TRACK2_API_MAX_RETRIES="${TRACK2_API_MAX_RETRIES:-1}"
export TRACK2_TEMPERATURE="${TRACK2_TEMPERATURE:-0.1}"
export TRACK2_RUN_VERSION="${TRACK2_RUN_VERSION:-V7_4_human_prior_full}"
export TRACK2_RUN_ID="$RUN_ID"
export TRACK2_OUTPUT_MODEL_NAME="${MODEL}-${TRACK2_RUN_VERSION}-${RUN_ID}"
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"

case "$TRACK2_RUN_VERSION" in
  V7_0*) export TRACK2_HUMAN_PRIOR_LEVEL=graph ;;
  V7_1*) export TRACK2_HUMAN_PRIOR_LEVEL=verifier ;;
  V7_2*) export TRACK2_HUMAN_PRIOR_LEVEL=counterfactual ;;
  V7_3*) export TRACK2_HUMAN_PRIOR_LEVEL=helpers ;;
  *) export TRACK2_HUMAN_PRIOR_LEVEL=full ;;
esac

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  cat > "$CODEX/reports/HUMAN_PRIOR_GATE_SUMMARY_${RUN_ID}.md" <<MD
# Human Prior Gate Summary

- generated_at: $(date +%Y-%m-%dT%H:%M:%S%z)
- run_id: $RUN_ID
- version: $TRACK2_RUN_VERSION
- status: not_run_key_missing
- key_present: no
- final_auto_submitted: no
MD
  exit 2
fi

mkdir -p "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs"
cd "$EGO"

GATE_SPECS="${TRACK2_GATE_SPECS:-retail:9 restaurant:4 order:1 kitchen:2}"
for spec in $GATE_SPECS; do
  scenario="${spec%%:*}"
  num="${spec##*:}"
  SCENARIO_TIMEOUT="${TRACK2_SCENARIO_TIMEOUT:-420}"
  timeout "$SCENARIO_TIMEOUT" "$PYTHON_BIN" -u "$CODEX/runners/track2_multi_agent_plus.py" \
    --scenario "$scenario" --scenario_number "$num" \
    --service_model_name "$MODEL" --num_tasks 1 \
    > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/${scenario}${num}.log" 2>&1 || true
done

cd "$EGO/analysis_scripts"
"$PYTHON_BIN" evaluate_interaction.py --model_name "$TRACK2_OUTPUT_MODEL_NAME" --num_samples 1 \
  > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/eval_python3.log" 2>&1 || true

cd "$CODEX"
"$PYTHON_BIN" "$CODEX/scripts/track2_gpt55_collect_gate.py" \
  --run-id "$RUN_ID" --model "$MODEL" --version "$TRACK2_RUN_VERSION" --no-update-best || true

cat > "$CODEX/state/latest_human_prior_gate.json" <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "run_id": "$RUN_ID",
  "model": "$MODEL",
  "version": "$TRACK2_RUN_VERSION",
  "base_url": "$BASE_URL",
  "key_present": true,
  "key_logged": false,
  "report": "reports/02_gpt55_gate_summary_${RUN_ID}.md",
  "final_auto_submitted": false
}
JSON
