#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
source state/.openai_env
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
export NO_PROXY="ai-pixel.online,cf.ai-pixel.online,localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
export CODEX_ROOT=/home/data-gxu/acm/egolink2026-main/code/track2/codex
export EGO_ROOT=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
export TRACK2_OPENAI_BASE_URL="${TRACK2_OPENAI_BASE_URL:-https://ai-pixel.online/v1}"
export SERVICE_MODEL_API_BASE="$TRACK2_OPENAI_BASE_URL"
export SERVICE_MODEL_API_KEY="${OPENAI_API_KEY:-}"
export USER_AGENT_API_BASE_URL="$TRACK2_OPENAI_BASE_URL"
export USER_AGENT_API_KEY="${OPENAI_API_KEY:-}"
export SERVICE_MODEL_BACKEND=openai_compatible_chat
export SERVICE_MODEL_NAME=gpt-5.5
export USER_MODEL_NAME=gpt-5.5
export TRACK2_DEFAULT_MAX_TOKENS=2048
export TRACK2_CONNECT_TIMEOUT=10
export TRACK2_READ_TIMEOUT=240
export TRACK2_API_MAX_RETRIES=1
export TRACK2_TEMPERATURE=0.1
export TRACK2_MAX_TURNS=6
export TRACK2_USE_VIDEO=0
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export TRACK2_ENABLE_VISUAL_CACHE=0
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_EVALUATOR_AWARENESS=1
export TRACK2_ENABLE_MEMORY_RETRIEVAL=1
export TRACK2_MEMORY_BANK_DIR=memory_bank_v10
export TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER=1
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
export TRACK2_FINAL_EVAL=1
export PYTHONPATH="$CODEX_ROOT/wrappers:$CODEX_ROOT:${PYTHONPATH:-}"
TEAM_NAME="V10_full_memory_final_candidate_draft"
export TRACK2_RUN_VERSION=V10_full_memory_final_candidate_draft
export TRACK2_OUTPUT_MODEL_NAME="$TEAM_NAME"
mkdir -p logs state runs/V10_full_memory_final_candidate_draft
run_one() {
  local scen="$1"
  local num="$2"
  local tag="$3"
  export TRACK2_RUN_ID="V10_full_memory_final_candidate_draft_${tag}_20260618_2010"
  mkdir -p "runs/V10_full_memory_final_candidate_draft/${TRACK2_RUN_ID}"
  echo "[V10 parallel] start ${scen}${num} $(date)"
  (cd "$EGO_ROOT" && python3 "$CODEX_ROOT/runners/track2_multi_agent_plus.py" --scenario "$scen" --scenario_number "$num" --service_model_name gpt-5.5 --multi_agent_user --summary_user --num_tasks 0 --final_eval) > "logs/v10_final_parallel_${scen}${num}_20260618_2010.log" 2>&1
  echo "[V10 parallel] done ${scen}${num} $(date)"
}
run_one retail 10 retail10 &
p1=$!
run_one kitchen 4 kitchen4 &
p2=$!
wait "$p1" "$p2"
run_one restaurant 5 restaurant5 &
p3=$!
run_one order 2 order2 &
p4=$!
wait "$p3" "$p4"
echo "[V10 parallel] all remaining scenarios done $(date)"
