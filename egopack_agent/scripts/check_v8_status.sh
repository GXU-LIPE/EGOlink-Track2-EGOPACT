#!/usr/bin/env bash
set -u
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
run_id="${1:-v8_2_valA_small_20260617_continue}"
version="${2:-V8_2_kitchen_helper}"
pid_file="state/${run_id}.pid"
echo "run_id=${run_id}"
echo "version=${version}"
date '+time=%Y-%m-%d %H:%M:%S %z'
if [[ -f "${pid_file}" ]]; then
  pid="$(tr -d '\r\n' < "${pid_file}")"
  echo "pid=${pid}"
  ps -p "${pid}" -o pid,ppid,etime,stat,cmd || true
else
  echo "pid_file_missing=${pid_file}"
fi
echo "--- active track2 processes ---"
ps aux | grep -E 'run_v8_validation.py|track2_multi_agent_plus.py' | grep -v grep || true
echo "--- launcher log ---"
tail -n 80 "logs/${run_id}.launcher.log" 2>/dev/null || true
echo "--- run files ---"
find "runs/${version}/${run_id}" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -80 || true
echo "--- reports ---"
find reports -maxdepth 1 -type f -name "*${run_id}*.md" -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort || true
