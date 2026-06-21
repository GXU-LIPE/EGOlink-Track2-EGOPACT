#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
PYTHON_BIN="${TRACK2_PYTHON:-python3}"
RUN_ID="${TRACK2_RUN_ID:-v12_4gate_$(date +%Y%m%d_%H%M%S)}"
VERSION="V12_official_style_qwen3vl_memory"

mkdir -p "$CODEX"/{logs,reports,analysis,state,runs}
mkdir -p "$CODEX/runs/$VERSION/$RUN_ID/logs"
cd "$CODEX"
if [[ -f "$CODEX/state/.openai_env" ]]; then
  # shellcheck disable=SC1091
  source "$CODEX/state/.openai_env"
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
BASE_URL="${TRACK2_OPENAI_BASE_URL:-https://ai-pixel.online/v1}"

export CODEX_ROOT="$CODEX"
export EGO_ROOT="$EGO"
export SERVICE_MODEL_BACKEND=openai_compatible_chat
export SERVICE_MODEL_NAME="$MODEL"
export SERVICE_MODEL_API_BASE="$BASE_URL"
export SERVICE_MODEL_API_KEY="${OPENAI_API_KEY:-}"
export USER_AGENT_API_BASE_URL="$BASE_URL"
export USER_AGENT_API_KEY="${OPENAI_API_KEY:-}"
export USER_MODEL_NAME="$MODEL"
export TRACK2_USER_USE_OPENAI=0
export TRACK2_USE_OPENAI_GPT55=0
export TRACK2_GPT55_STRUCTURED_OUTPUT=0
export TRACK2_ENABLE_OFFICIAL_STYLE_PROMPT=1
export TRACK2_ENABLE_EVALUATOR_AWARENESS=1
export TRACK2_ENABLE_MEMORY_RETRIEVAL=1
export TRACK2_MEMORY_BANK_DIR="$CODEX/memory_bank_v10"
export TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER=1
export TRACK2_ENABLE_QWEN3VL_GROUNDING=1
export TRACK2_QWEN3VL_GROUNDING_DIR="$CODEX/visual_cache_v12/qwen3vl_grounding"
export TRACK2_ENABLE_RETAIL_CANDIDATE_NARROWER=1
export TRACK2_ENABLE_RETAIL_PROCESS_TRIMMER=1
export TRACK2_ENABLE_ORDER_PROCESS_SYNTHESIS=1
export TRACK2_ENABLE_ORDER_PROCESS_MEMORY=1
export TRACK2_ENABLE_V9_SOFT_GUARD=1
export TRACK2_ENABLE_MULTICANDIDATE_RERANK=1
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_VISUAL_CACHE=1
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export TRACK2_USE_VIDEO=0
export TRACK2_MAX_TURNS="${TRACK2_MAX_TURNS:-6}"
export TRACK2_DEFAULT_MAX_TOKENS="${TRACK2_DEFAULT_MAX_TOKENS:-2048}"
export TRACK2_CONNECT_TIMEOUT="${TRACK2_CONNECT_TIMEOUT:-10}"
export TRACK2_READ_TIMEOUT="${TRACK2_READ_TIMEOUT:-240}"
export TRACK2_API_MAX_RETRIES="${TRACK2_API_MAX_RETRIES:-1}"
export TRACK2_TEMPERATURE="${TRACK2_TEMPERATURE:-0.1}"
export TRACK2_RUN_VERSION="$VERSION"
export TRACK2_RUN_ID="$RUN_ID"
export TRACK2_OUTPUT_MODEL_NAME="${MODEL}-${VERSION}-${RUN_ID}"
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  cat > "$CODEX/reports/V12_4GATE_QWEN3VL_MEMORY_${RUN_ID}.md" <<MD
# V12 4-Gate Qwen3VL Memory

- status: not_run_key_missing
- run_id: $RUN_ID
- key_present: no
MD
  exit 2
fi

cd "$EGO"
GATE_SPECS="retail:9 restaurant:4 order:1 kitchen:2"
for spec in $GATE_SPECS; do
  scenario="${spec%%:*}"
  num="${spec##*:}"
  "$PYTHON_BIN" "$CODEX/runners/track2_multi_agent_plus.py" \
    --scenario "$scenario" --scenario_number "$num" \
    --service_model_name "$MODEL" \
    --num_tasks 1 \
    > "$CODEX/runs/$VERSION/$RUN_ID/logs/${scenario}${num}.log" 2>&1 || true
  sleep 1
done

cd "$EGO/analysis_scripts"
"$PYTHON_BIN" evaluate_interaction.py --model_name "$TRACK2_OUTPUT_MODEL_NAME" --num_samples 1 \
  > "$CODEX/runs/$VERSION/$RUN_ID/logs/eval_python3.log" 2>&1 || true

cd "$CODEX"
"$PYTHON_BIN" "$CODEX/scripts/collect_v12_4gate_report.py" --run-id "$RUN_ID" --version "$VERSION" --model "$MODEL" || true
cat > "$CODEX/state/latest_v12_4gate.json" <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "run_id": "$RUN_ID",
  "version": "$VERSION",
  "model": "$MODEL",
  "report": "reports/V12_4GATE_QWEN3VL_MEMORY_${RUN_ID}.md",
  "final_run": false,
  "v10_zip_overwritten": false
}
JSON
