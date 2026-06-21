#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
printf '%s\n' '--- pids ---'
ps -o pid,ppid,stat,etime,cmd -p 2566022 -p 2566026 2>/dev/null || true
pgrep -af 'run_v10_final_remaining_parallel.sh|track2_multi_agent_plus.py' || true
printf '%s\n' '--- scenario progress ---'
for tag in retail6 retail10 kitchen4 restaurant5 order2; do
  log="logs/v10_final_full_${tag}_20260618_1940.log"
  if [[ ! -f "$log" ]]; then log="logs/v10_final_parallel_${tag}_20260618_2010.log"; fi
  if [[ -f "$log" ]]; then
    printf '%s ' "$tag"
    grep -E "Scenario .*:" "$log" | tail -n 1 || true
    printf '  lines='; wc -l < "$log"
    printf '  hard_errors='; grep -icE 'traceback|exception|429|rate limit|readtimeout|connectionerror|api_error' "$log" || true
  else
    printf '%s no_log\n' "$tag"
  fi
done
printf '%s\n' '--- result counts ---'
cd /home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('results/V10_full_memory_final_candidate_draft')
print('root_exists', root.exists())
for p in sorted(root.glob('*_easy.json')):
    try:
        data=json.loads(p.read_text())
        print(p.name, len(data), p.stat().st_size)
    except Exception as e:
        print(p.name, type(e).__name__, p.stat().st_size)
PY
