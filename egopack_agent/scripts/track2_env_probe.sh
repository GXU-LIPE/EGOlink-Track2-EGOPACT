#!/usr/bin/env bash
set -euo pipefail

EGO_ROOT="${EGO_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench}"
CODEX_ROOT="${CODEX_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/codex}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$CODEX_ROOT/logs/env_probe_${TS}.log"
REPORT="$CODEX_ROOT/reports/00_inventory_${TS}.md"
STATE="$CODEX_ROOT/state/inventory.json"

mkdir -p "$CODEX_ROOT"/{scripts,runners,wrappers,patches,backups,logs,reports,runs,analysis,visual_cache,submissions,state}
exec > >(tee -a "$LOG") 2>&1

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
export PYTHONPATH="/home/data-gxu/wjb19/depthplus/codex1:${CODEX_ROOT}:${EGO_ROOT}:${PYTHONPATH:-}"

section() {
  echo
  echo "## $1"
}

run_block() {
  local title="$1"
  shift
  {
    echo
    echo "### ${title}"
    echo
    echo '```text'
    "$@" 2>&1 || true
    echo '```'
  } >> "$REPORT"
}

{
  echo "# Track2 Inventory"
  echo
  echo "- timestamp: $(date -Is)"
  echo "- ego_root: $EGO_ROOT"
  echo "- codex_root: $CODEX_ROOT"
  echo "- log: $LOG"
} > "$REPORT"

section "Identity"
run_block "Identity" bash -lc 'whoami; hostname; date -Is; pwd; id'

section "Conda and Python"
run_block "Conda env list" bash -lc 'command -v conda || true; conda env list 2>/dev/null || true; command -v mamba || true; command -v micromamba || true'
run_block "Current Python" bash -lc 'command -v python || true; python --version 2>&1 || true; command -v python3 || true; python3 --version 2>&1 || true'
run_block "Key Python packages" bash -lc 'python -c "import sys, importlib.util; print(sys.executable); print(sys.version); mods=[\"openai\",\"requests\",\"pandas\",\"numpy\",\"PIL\",\"cv2\",\"torch\"]; [print(m, bool(importlib.util.find_spec(m))) for m in mods]" 2>&1 || true'
python3 "$CODEX_ROOT/scripts/track2_select_env.py" --output "$CODEX_ROOT/state/env_choice.sh" --json-output "$CODEX_ROOT/state/env_choice.json" || true
if [ -f "$CODEX_ROOT/state/env_choice.sh" ]; then
  # shellcheck disable=SC1090
  source "$CODEX_ROOT/state/env_choice.sh" || true
fi
run_block "Selected Python" bash -lc 'echo TRACK2_PYTHON=${TRACK2_PYTHON:-}; ${TRACK2_PYTHON:-python} -c "import sys, importlib.util; print(sys.executable); print(sys.version); mods=[\"openai\",\"requests\",\"pandas\",\"numpy\",\"PIL\",\"cv2\",\"torch\"]; [print(m, bool(importlib.util.find_spec(m))) for m in mods]" 2>&1 || true'

section "Track1 clues"
run_block "Track1 clues" bash -lc 'find /home/data-gxu/acm/egolink2026-main /home/data-gxu/acm -maxdepth 5 \( -iname "*track1*" -o -iname "*log*" -o -iname "*result*" -o -iname "run*.sh" -o -iname "*.yml" -o -iname "*.yaml" \) 2>/dev/null | head -250'

section "Track2 root"
run_block "EgoBench root listing" bash -lc "ls -la '$EGO_ROOT'; find '$EGO_ROOT' -maxdepth 2 -type f | sort | head -250"
run_block "Git state" bash -lc "cd '$EGO_ROOT' && git rev-parse HEAD 2>/dev/null || true; cd '$EGO_ROOT' && git status --short 2>/dev/null || true"
run_block "Important files" bash -lc "for f in README.md run_all_scenarios.sh run/multi_agent.py analysis_scripts/run_eval.sh config/user_agent_config.py config/service_agent_config.py requirements.txt; do echo ==== \$f; [ -f '$EGO_ROOT/'\$f ] && sed -n '1,220p' '$EGO_ROOT/'\$f | head -220 || echo missing; done"
run_block "Scenarios tools results" bash -lc "find '$EGO_ROOT/scenarios' -maxdepth 3 -type f 2>/dev/null | sort; find '$EGO_ROOT/tools' -maxdepth 3 -type f 2>/dev/null | sort | head -200; find '$EGO_ROOT/results' -maxdepth 3 -type f 2>/dev/null | sort | head -200; find '$EGO_ROOT/eval_result' -maxdepth 3 -type f 2>/dev/null | sort | head -200"
run_block "Videos" bash -lc "find '$EGO_ROOT/videos' -maxdepth 1 -type f 2>/dev/null | sort; command -v ffmpeg || true"

section "Final availability"
run_block "Final task files" bash -lc "for f in '$EGO_ROOT'/scenarios/final/retail6.json '$EGO_ROOT'/scenarios/final/retail10.json '$EGO_ROOT'/scenarios/final/kitchen4.json '$EGO_ROOT'/scenarios/final/restaurant5.json '$EGO_ROOT'/scenarios/final/order2.json; do [ -f \"\$f\" ] && echo present:\$f || echo missing:\$f; done"

section "GPU and disk"
run_block "GPU" bash -lc 'command -v nvidia-smi && nvidia-smi || echo no-nvidia-smi'
run_block "Disk" bash -lc "df -h '$CODEX_ROOT' '$EGO_ROOT' /home/data-gxu 2>/dev/null || df -h"
run_block "Existing Track2 processes" bash -lc 'ps -eo pid,ppid,stat,etime,cmd | grep -E "multi_agent|track2_autopilot|track2_watchdog|run_all_scenarios|evaluate_interaction" | grep -v grep || true'

python3 "$CODEX_ROOT/scripts/track2_inventory_json.py" --report "$REPORT" --output "$STATE" --ego-root "$EGO_ROOT" --codex-root "$CODEX_ROOT" || true
echo "inventory_report=$REPORT"
echo "inventory_state=$STATE"
