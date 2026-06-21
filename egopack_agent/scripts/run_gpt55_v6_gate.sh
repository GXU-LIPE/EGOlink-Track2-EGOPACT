#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
PYTHON_BIN="${TRACK2_PYTHON:-python3}"
RUN_ID="${TRACK2_RUN_ID:-gpt55_v6_gate_$(date +%Y%m%d_%H%M%S)}"
MODEL="${TRACK2_OPENAI_MODEL:-gpt-5.5}"

mkdir -p "$CODEX"/{logs,reports,analysis,state,runs}
cd "$CODEX"

export CODEX_ROOT="$CODEX"
export EGO_ROOT="$EGO"
export SERVICE_MODEL_BACKEND=openai_gpt55
export SERVICE_MODEL_NAME="$MODEL"
export TRACK2_OPENAI_MODEL="$MODEL"
export TRACK2_OPENAI_REASONING_EFFORT="${TRACK2_OPENAI_REASONING_EFFORT:-medium}"
export TRACK2_OPENAI_TEXT_VERBOSITY="${TRACK2_OPENAI_TEXT_VERBOSITY:-low}"
export TRACK2_OPENAI_MAX_OUTPUT_TOKENS="${TRACK2_OPENAI_MAX_OUTPUT_TOKENS:-2048}"
export TRACK2_OPENAI_TIMEOUT="${TRACK2_OPENAI_TIMEOUT:-180}"
export TRACK2_OPENAI_MAX_RETRIES="${TRACK2_OPENAI_MAX_RETRIES:-2}"
export TRACK2_USE_OPENAI_GPT55=1
export TRACK2_ENABLE_DB_GUARD=1
export TRACK2_ENABLE_PLANNER=1
export TRACK2_ENABLE_SCENARIO_RULES=1
export TRACK2_ENABLE_VISUAL_CACHE=1
export TRACK2_TEXT_ONLY_VISUAL_CONTEXT=1
export TRACK2_USE_VIDEO=0
export TRACK2_MAX_TURNS="${TRACK2_MAX_TURNS:-10}"
export PYTHONPATH="$CODEX/wrappers:$CODEX:${PYTHONPATH:-}"

key_present=no
if [[ -n "${OPENAI_API_KEY:-}" ]]; then key_present=yes; fi

echo "Preparing visual cache..."
for spec in retail:9:1 kitchen:2:1 restaurant:4:1 order:1:1; do
  IFS=: read -r scenario num task <<<"$spec"
  "$PYTHON_BIN" "$CODEX/scripts/track2_extract_video_frames.py" --scenario "$scenario" --scenario-number "$num" --task-index "$task" > "$CODEX/logs/v6_extract_${scenario}${num}_${RUN_ID}.log" 2>&1 || true
  if [[ "$key_present" == yes && "${TRACK2_SKIP_VISUAL_STATE:-0}" != "1" ]]; then
    "$PYTHON_BIN" "$CODEX/scripts/track2_build_visual_state_gpt55.py" --scenario "$scenario" --scenario-number "$num" --task-index "$task" > "$CODEX/logs/v6_visual_state_${scenario}${num}_${RUN_ID}.log" 2>&1 || true
  fi
done

"$PYTHON_BIN" "$CODEX/scripts/track2_gpt55_adapter_smoke.py" > "$CODEX/logs/v6_adapter_smoke_${RUN_ID}.log" 2>&1 || true

if [[ "$key_present" != yes ]]; then
  cat > "$CODEX/reports/02_gpt55_gate_summary_${RUN_ID}.md" <<MD
# GPT-5.5 Gate Summary

- generated_at: $(date +%Y-%m-%dT%H:%M:%S%z)
- run_id: $RUN_ID
- status: not_run_openai_key_missing
- openai_key_present: no
- model: $MODEL
- note: Set OPENAI_API_KEY in the remote shell/session and rerun this script. The key was not written to any file.
MD
  exit 0
fi

run_version() {
  local version="$1"
  local send_contact="$2"
  local effort="$3"
  export TRACK2_RUN_VERSION="$version"
  export TRACK2_RUN_ID="$RUN_ID"
  export TRACK2_OUTPUT_MODEL_NAME="${MODEL}-${version}-${RUN_ID}"
  export TRACK2_GPT55_SEND_CONTACT_SHEET="$send_contact"
  export TRACK2_OPENAI_REASONING_EFFORT="$effort"
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

run_version V6_gpt55_direct 0 medium
run_version V6_1_gpt55_guarded 0 medium
run_version V6_2_gpt55_visual_state 0 medium
run_version V6_3_gpt55_visual_retry 1 high

cd "$CODEX"
"$PYTHON_BIN" "$CODEX/scripts/track2_gpt55_collect_gate.py" --run-id "$RUN_ID" --model "$MODEL" || true
