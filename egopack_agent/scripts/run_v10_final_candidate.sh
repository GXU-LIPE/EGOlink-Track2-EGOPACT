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
export SERVICE_MODEL_NAME="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
export USER_MODEL_NAME="$SERVICE_MODEL_NAME"
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
export PYTHONPATH="$CODEX_ROOT/wrappers:$CODEX_ROOT:${PYTHONPATH:-}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_ID="V10_full_memory_final_candidate_draft_${TS}"
export TRACK2_RUN_VERSION=V10_full_memory_final_candidate_draft
export TRACK2_RUN_ID="$RUN_ID"
TEAM_NAME="V10_full_memory_final_candidate_draft"
export TRACK2_OUTPUT_MODEL_NAME="$TEAM_NAME"
mkdir -p logs runs/V10_full_memory_final_candidate_draft/$RUN_ID
python3 scripts/v10_api_healthcheck.py | tee "logs/v10_api_healthcheck_${TS}.log"
python3 scripts/build_v10_full_memory_bank.py | tee "logs/v10_memory_bank_${TS}.log"
python3 scripts/run_v8_validation.py --stage validation_A_medium --version V10_full_memory_final_candidate_draft --run-id "${RUN_ID}_A_medium_sanity" --model gpt-5.5 | tee "logs/v10_A_medium_${TS}.log"
python3 scripts/v10_make_reports.py --run-id "$RUN_ID" --stage a_medium --a-medium-run "${RUN_ID}_A_medium_sanity"
# Final smoke: official final scenarios, 2 tasks each.
export TRACK2_FINAL_EVAL=1
for spec in retail:6 retail:10 kitchen:4 restaurant:5 order:2; do
  scen=${spec%%:*}; num=${spec##*:}
  (cd "$EGO_ROOT" && python3 "$CODEX_ROOT/runners/track2_multi_agent_plus.py" --scenario "$scen" --scenario_number "$num" --service_model_name gpt-5.5 --multi_agent_user --summary_user --num_tasks 2 --final_eval) | tee "logs/v10_final_smoke_${scen}${num}_${TS}.log"
done
python3 scripts/v10_make_reports.py --run-id "$RUN_ID" --stage final_smoke --team-name "$TEAM_NAME"
SMOKE_OK=$(python3 scripts/v10_make_reports.py --run-id "$RUN_ID" --stage smoke_ok --team-name "$TEAM_NAME")
if [[ "$SMOKE_OK" != "OK" ]]; then
  echo "Final smoke failed; not running final full."
  exit 3
fi
for spec in retail:6 retail:10 kitchen:4 restaurant:5 order:2; do
  scen=${spec%%:*}; num=${spec##*:}
  (cd "$EGO_ROOT" && python3 "$CODEX_ROOT/runners/track2_multi_agent_plus.py" --scenario "$scen" --scenario_number "$num" --service_model_name gpt-5.5 --multi_agent_user --summary_user --num_tasks 0 --final_eval) | tee "logs/v10_final_full_${scen}${num}_${TS}.log"
done
python3 scripts/v10_make_reports.py --run-id "$RUN_ID" --stage final_full --team-name "$TEAM_NAME"
python3 scripts/track2_pack_submission.py --team-name "$TEAM_NAME" --report-md "reports/V10_TECHNICAL_REPORT_DRAFT_${RUN_ID}.md" | tee "logs/v10_pack_${TS}.log"
python3 scripts/v10_make_reports.py --run-id "$RUN_ID" --stage package --team-name "$TEAM_NAME"
echo "RUN_ID=$RUN_ID"
