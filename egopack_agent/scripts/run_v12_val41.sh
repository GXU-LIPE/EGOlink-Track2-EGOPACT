#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
cd "$CODEX"

if [[ -f state/.openai_env ]]; then
  # shellcheck disable=SC1091
  source state/.openai_env
fi
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
export NO_PROXY="ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

RUN_ID="${TRACK2_RUN_ID:-V12_official_style_qwen3vl_memory_validation_A_medium_$(date +%Y%m%d_%H%M)}"
VERSION="V12_official_style_qwen3vl_memory"
MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
BASE_URL="${TRACK2_OPENAI_BASE_URL:-https://ai-pixel.online/v1}"

export CODEX_ROOT="$CODEX"
export EGO_ROOT="$EGO"
export TRACK2_OPENAI_BASE_URL="$BASE_URL"
export SERVICE_MODEL_API_BASE="$BASE_URL"
export SERVICE_MODEL_API_KEY="${OPENAI_API_KEY:-}"
export USER_AGENT_API_BASE_URL="$BASE_URL"
export USER_AGENT_API_KEY="${OPENAI_API_KEY:-}"
export SERVICE_MODEL_BACKEND=openai_compatible_chat
export SERVICE_MODEL_NAME="$MODEL"
export USER_MODEL_NAME="$MODEL"
export TRACK2_USER_USE_OPENAI=0
export TRACK2_USE_OPENAI_GPT55=0
export TRACK2_GPT55_STRUCTURED_OUTPUT=0
export TRACK2_DEFAULT_MAX_TOKENS=2048
export TRACK2_CONNECT_TIMEOUT=10
export TRACK2_READ_TIMEOUT=240
export TRACK2_API_MAX_RETRIES=1
export TRACK2_TEMPERATURE=0.1
export TRACK2_MAX_TURNS=6
export TRACK2_V8_TASK_TIMEOUT=1800
export TRACK2_USE_VIDEO=0
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export TRACK2_ENABLE_VISUAL_CACHE=1
export TRACK2_ENABLE_OFFICIAL_STYLE_PROMPT=1
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_EVALUATOR_AWARENESS=1
export TRACK2_ENABLE_MEMORY_RETRIEVAL=1
export TRACK2_MEMORY_BANK_DIR="$CODEX/memory_bank_v10"
export TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER=1
export TRACK2_ENABLE_QWEN3VL_GROUNDING=1
export TRACK2_QWEN3VL_GROUNDING_DIR="$CODEX/visual_cache_v12/qwen3vl_grounding"
export TRACK2_ENABLE_RETAIL_NARROWER=1
export TRACK2_ENABLE_RETAIL_CANDIDATE_NARROWER=1
export TRACK2_ENABLE_RETAIL_PROCESS_TRIMMER=1
export TRACK2_ENABLE_ORDER_PROCESS_MEMORY=1
export TRACK2_ENABLE_ORDER_PROCESS_SYNTHESIS=1
export TRACK2_ENABLE_V9_SOFT_GUARD=1
export TRACK2_ENABLE_SOFT_GUARD=1
export TRACK2_ENABLE_MULTICANDIDATE=1
export TRACK2_ENABLE_MULTICANDIDATE_RERANK=1
export TRACK2_ENABLE_DEEPSEEK_CROSSCHECK=0
export TRACK2_USE_DEEPSEEK_CROSSCHECK=0
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"
export TRACK2_RUN_VERSION="$VERSION"
export TRACK2_RUN_ID="$RUN_ID"
export TRACK2_OUTPUT_MODEL_NAME="$MODEL-$VERSION-$RUN_ID"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  mkdir -p reports
  cat > "reports/V12_VAL41_${RUN_ID}_KEY_MISSING.md" <<MD
# V12 Val41

- status: not_run_key_missing
- key_present: no
- final_run: false
MD
  exit 2
fi

python3 scripts/run_v8_validation.py --stage validation_A_medium --version "$VERSION" --run-id "${RUN_ID}_dryrun" --model "$MODEL" --dry-run
python3 scripts/run_v8_validation.py --stage validation_A_medium --version "$VERSION" --run-id "$RUN_ID" --model "$MODEL"
python3 scripts/collect_v12_val41_report.py --run-id "$RUN_ID" --version "$VERSION" --model "$MODEL"

cat > state/latest_v12_val41.json <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "run_id": "$RUN_ID",
  "version": "$VERSION",
  "model": "$MODEL",
  "stage": "validation_A_medium",
  "report": "reports/V12_VAL41_QWEN3VL_MEMORY_${RUN_ID}.md",
  "final_run": false,
  "v10_zip_overwritten": false
}
JSON
