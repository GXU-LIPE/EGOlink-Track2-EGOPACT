#!/usr/bin/env bash
set -euo pipefail
CODEX_ROOT="${CODEX_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/codex}"
EGO_ROOT="${EGO_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench}"
source "$CODEX_ROOT/scripts/track2_common.sh"
NUM_TASKS=0
VERSION=${TRACK2_VERSION:-V1_format_schema_guard}
FINAL_EVAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_tasks) NUM_TASKS="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --final_eval) FINAL_EVAL=1; shift ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done
PY=$(pick_python)
cd "$EGO_ROOT"
if [[ "$FINAL_EVAL" == "1" ]]; then
  export TRACK2_FINAL_EVAL=1
  specs=(retail:6 retail:10 kitchen:4 restaurant:5 order:2)
else
  specs=(retail:1 retail:2 retail:3 retail:4 retail:5 retail:7 retail:8 retail:9 kitchen:1 kitchen:2 kitchen:3 restaurant:1 restaurant:2 restaurant:3 restaurant:4 order:1)
fi
for spec in "${specs[@]}"; do
  scenario="${spec%%:*}"; num="${spec##*:}"
  echo "Running $VERSION $scenario$num num_tasks=$NUM_TASKS final_eval=$FINAL_EVAL"
  extra_args=()
  if [[ "$FINAL_EVAL" == "1" ]]; then
    extra_args+=(--final_eval)
  fi
  "$PY" "$CODEX_ROOT/runners/track2_multi_agent_plus.py" --scenario "$scenario" --scenario_number "$num" --service_model_name "$SERVICE_MODEL_NAME" --multi_agent_user --summary_user --num_tasks "$NUM_TASKS" "${extra_args[@]}" || true
done
