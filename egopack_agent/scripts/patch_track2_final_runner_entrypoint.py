#!/usr/bin/env python3
from pathlib import Path
import time
p=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex/runners/run_all_scenarios_plus.sh')
s=p.read_text(encoding='utf-8')
s=s.replace('''NUM_TASKS=0
VERSION=${TRACK2_VERSION:-V1_format_schema_guard}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_tasks) NUM_TASKS="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done
PY=$(pick_python)
cd "$EGO_ROOT"
for spec in retail:1 retail:2 retail:3 retail:4 retail:5 retail:7 retail:8 retail:9 kitchen:1 kitchen:2 kitchen:3 restaurant:1 restaurant:2 restaurant:3 restaurant:4 order:1; do
  scenario="${spec%%:*}"; num="${spec##*:}"
  echo "Running $VERSION $scenario$num num_tasks=$NUM_TASKS"
  "$PY" "$CODEX_ROOT/runners/track2_multi_agent_plus.py" --scenario "$scenario" --scenario_number "$num" --service_model_name "$SERVICE_MODEL_NAME" --multi_agent_user --summary_user --num_tasks "$NUM_TASKS" || true
done
''','''NUM_TASKS=0
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
''')
p.write_text(s,encoding='utf-8')
ts=time.strftime('%Y%m%d_%H%M%S')
report=Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex/reports')/f'FINAL_RUNNER_ENTRYPOINT_PATCH_{ts}.md'
report.write_text('''# Final Runner Entrypoint Patch

- Added `--final_eval` support to `codex/runners/run_all_scenarios_plus.sh`.
- Final mode runs exactly: `retail6`, `retail10`, `kitchen4`, `restaurant5`, `order2`.
- Final mode exports `TRACK2_FINAL_EVAL=1` and passes `--final_eval` to `track2_multi_agent_plus.py`, enabling service-agent no-direct-final-JSON safeguards.
- Existing offline/dev scenario list is unchanged when `--final_eval` is not used.
- No final submission was made.
''',encoding='utf-8')
with Path('/home/data-gxu/acm/egolink2026-main/code/track2/codex/README_STATUS.md').open('a',encoding='utf-8') as f:
    f.write(f'\n## Final Runner Entrypoint Patch {ts}\n\n- Report: `{report}`\n- `runners/run_all_scenarios_plus.sh --final_eval` now runs official final scenarios with `TRACK2_FINAL_EVAL=1`.\n- No final submission was made.\n')
print(report)
