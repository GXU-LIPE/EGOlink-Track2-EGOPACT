#!/usr/bin/env bash
set -euo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

EGO_ROOT="${EGO_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench}"
CODEX_ROOT="${CODEX_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/codex}"

mkdir -p "$CODEX_ROOT"/{scripts,runners,wrappers,patches,backups,logs,reports,runs,analysis,visual_cache,submissions,state}

set +u
source ~/.bashrc >/dev/null 2>&1 || true
source ~/.codex_proxy_env >/dev/null 2>&1 || true
if [ -f "$CODEX_ROOT/state/secrets.env" ]; then
  source "$CODEX_ROOT/state/secrets.env" >/dev/null 2>&1 || true
fi
if [ -f "$CODEX_ROOT/state/env_choice.sh" ]; then
  source "$CODEX_ROOT/state/env_choice.sh" >/dev/null 2>&1 || true
fi
set -u

export PYTHONPATH="/home/data-gxu/wjb19/depthplus/codex1:${CODEX_ROOT}:${CODEX_ROOT}/wrappers:${EGO_ROOT}:${PYTHONPATH:-}"

log_msg() {
  local msg="$1"
  echo "[$(date -Is)] $msg"
}

pick_python() {
  if [ -n "${TRACK2_PYTHON:-}" ] && [ -x "${TRACK2_PYTHON:-}" ]; then
    echo "$TRACK2_PYTHON"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo python3
    return 0
  fi
  echo python
}

print_runtime_brief() {
  local py
  py="$(pick_python)"
  log_msg "python_path=$($py -c 'import sys; print(sys.executable)' 2>/dev/null || echo unknown)"
  log_msg "python_version=$($py --version 2>&1 || echo unknown)"
  $py - <<'PY' 2>/dev/null || true
import importlib.util
mods = ["openai", "requests", "pandas", "numpy", "PIL", "cv2", "torch"]
for m in mods:
    print(f"[runtime] package {m}: {bool(importlib.util.find_spec(m))}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  fi
}
