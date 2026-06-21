#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
PYTHON_BIN="${TRACK2_PYTHON:-python3}"
RUN_ID="${TRACK2_RUN_ID:-gpt55_endpoint_gate_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$CODEX"/{logs,reports,analysis,state,runs}
cd "$CODEX"

if [[ -f "$CODEX/state/.openai_env" ]]; then
  # shellcheck disable=SC1091
  source "$CODEX/state/.openai_env"
fi

MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"
BASE_URL="${TRACK2_OPENAI_BASE_URL:-https://ai-pixel.online/v1}"

# The remote host has stale localhost proxy variables in some sessions. The
# ai-pixel endpoint is directly reachable from the remote, so avoid proxy use.
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
export TRACK2_USE_OPENAI_GPT55="${TRACK2_USE_OPENAI_GPT55:-0}"
export TRACK2_GPT55_STRUCTURED_OUTPUT="${TRACK2_GPT55_STRUCTURED_OUTPUT:-0}"
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
export TRACK2_RUN_VERSION="${TRACK2_RUN_VERSION:-V6_1_3_gpt55_guarded_endpoint}"
export TRACK2_RUN_ID="$RUN_ID"
export TRACK2_OUTPUT_MODEL_NAME="${MODEL}-${TRACK2_RUN_VERSION}-${RUN_ID}"
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"

key_present=no
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  key_present=yes
fi

probe_log="$CODEX/logs/gpt55_endpoint_probe_${RUN_ID}.log"
if [[ "$key_present" != yes ]]; then
  cat > "$CODEX/reports/02_gpt55_gate_summary_${RUN_ID}.md" <<MD
# GPT-5.5 Endpoint Gate Summary

- generated_at: $(date +%Y-%m-%dT%H:%M:%S%z)
- run_id: $RUN_ID
- status: not_run_key_missing
- key_present: no
- final_submission: not submitted
MD
  exit 2
fi

curl -sS --connect-timeout 10 --max-time 40 "$BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  > "$probe_log" 2>&1 || true

mkdir -p "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs"
cd "$EGO"

GATE_SPECS="${TRACK2_GATE_SPECS:-retail:9 kitchen:2 restaurant:4 order:1}"
for spec in $GATE_SPECS; do
  scenario="${spec%%:*}"
  num="${spec##*:}"
  user_flags=()
  if [[ "${TRACK2_MULTI_AGENT_USER:-0}" == "1" ]]; then
    user_flags+=(--multi_agent_user)
  fi
  if [[ "${TRACK2_SUMMARY_USER:-0}" == "1" ]]; then
    user_flags+=(--summary_user)
  fi
  "$PYTHON_BIN" "$CODEX/runners/track2_multi_agent_plus.py" \
    --scenario "$scenario" --scenario_number "$num" \
    --service_model_name "$MODEL" \
    "${user_flags[@]}" --num_tasks 1 \
    > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/${scenario}${num}.log" 2>&1 || true
done

cd "$EGO/analysis_scripts"
"$PYTHON_BIN" evaluate_interaction.py --model_name "$TRACK2_OUTPUT_MODEL_NAME" --num_samples 1 \
  > "$CODEX/runs/$TRACK2_RUN_VERSION/$RUN_ID/logs/eval_python3.log" 2>&1 || true

cd "$CODEX"
"$PYTHON_BIN" "$CODEX/scripts/track2_gpt55_collect_gate.py" \
  --run-id "$RUN_ID" --model "$MODEL" --version "$TRACK2_RUN_VERSION" || true

cat > "$CODEX/state/latest_gpt55_endpoint_gate.json" <<JSON
{
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S%z)",
  "run_id": "$RUN_ID",
  "model": "$MODEL",
  "version": "$TRACK2_RUN_VERSION",
  "base_url": "$BASE_URL",
  "key_present": true,
  "key_logged": false,
  "report": "reports/02_gpt55_gate_summary_${RUN_ID}.md"
}
JSON
