#!/usr/bin/env bash
set -euo pipefail

CODEX_ROOT="${CODEX_ROOT:-/home/data-gxu/acm/egolink2026-main/code/track2/codex}"
source "$CODEX_ROOT/scripts/track2_common.sh"

RESUME=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

LOG="$CODEX_ROOT/logs/watchdog_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "$$" > "$CODEX_ROOT/state/watchdog.pid"
log_msg "watchdog_start resume=$RESUME dry_run=$DRY_RUN"
PY="$(pick_python)"

heartbeat_once() {
  local ts hb status_json
  ts="$(date +%Y%m%d_%H%M%S)"
  hb="$CODEX_ROOT/reports/heartbeat_${ts}.md"
  status_json="$CODEX_ROOT/state/autopilot_status.json"
  {
    echo "# Track2 Heartbeat"
    echo
    echo "- timestamp: $(date -Is)"
    echo "- current_stage: $("$PY" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p,encoding="utf-8")) if __import__("os").path.exists(p) else {}; print(d.get("stage","unknown"))' "$status_json" 2>/dev/null || echo unknown)"
    echo "- current_version: $("$PY" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p,encoding="utf-8")) if __import__("os").path.exists(p) else {}; print(d.get("version","unknown"))' "$status_json" 2>/dev/null || echo unknown)"
    echo "- completed_tasks: $("$PY" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p,encoding="utf-8")) if __import__("os").path.exists(p) else {}; print(d.get("completed_tasks",0))' "$status_json" 2>/dev/null || echo 0)"
    echo "- best_joint_success: $("$PY" -c 'import json,sys,os; p=os.path.join(sys.argv[1],"best_version.json"); d=json.load(open(p,encoding="utf-8")) if os.path.exists(p) else {}; print(d.get("joint_success","unknown"))' "$CODEX_ROOT/state" 2>/dev/null || echo unknown)"
    echo "- gpu:"
    echo '```text'
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo no-nvidia-smi
    echo '```'
    echo "- disk:"
    echo '```text'
    df -h "$CODEX_ROOT" "$EGO_ROOT" 2>/dev/null || df -h
    echo '```'
    echo "- recent_errors:"
    echo '```text'
    tail -20 "$CODEX_ROOT/state/error_queue.jsonl" 2>/dev/null || true
    echo '```'
    echo "- need_human_attention: false"
  } > "$hb"
  log_msg "heartbeat_written $hb"
}

if [ "$DRY_RUN" -eq 1 ]; then
  heartbeat_once
  exit 0
fi

LAST_HB=0
while true; do
  now="$(date +%s)"
  if [ $((now - LAST_HB)) -ge 21600 ]; then
    heartbeat_once
    LAST_HB="$now"
  fi

  alive=0
  if [ -f "$CODEX_ROOT/state/autopilot.pid" ]; then
    pid="$(cat "$CODEX_ROOT/state/autopilot.pid" || true)"
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
      alive=1
    fi
  fi
  if [ "$alive" -eq 0 ]; then
    stage="$("$PY" -c 'import json,sys,os; p=os.path.join(sys.argv[1],"autopilot_status.json"); d=json.load(open(p,encoding="utf-8")) if os.path.exists(p) else {}; print(d.get("stage","unknown"))' "$CODEX_ROOT/state" 2>/dev/null || echo unknown)"
    case "$stage" in
      gate_complete*|NEED_HUMAN_ATTENTION)
        log_msg "autopilot_not_restarted stage=$stage"
        ;;
      *)
        log_msg "autopilot_not_alive_restart stage=$stage"
      RUN_ID="gate_restart_$(date +%Y%m%d_%H%M%S)"
      nohup bash "$CODEX_ROOT/scripts/track2_autopilot.sh" --resume --run-id "$RUN_ID" >> "$CODEX_ROOT/logs/watchdog_restart.log" 2>&1 &
      echo $! > "$CODEX_ROOT/state/autopilot.pid"
      echo "$RUN_ID" > "$CODEX_ROOT/state/current_gate_run_id.txt"
        ;;
    esac
  fi
  sleep 1800
done
