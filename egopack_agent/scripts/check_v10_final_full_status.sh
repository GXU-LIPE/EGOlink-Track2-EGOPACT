#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
pid=""
if [[ -f state/v10_final_full_20260618_1940.pid ]]; then pid=$(cat state/v10_final_full_20260618_1940.pid); fi
echo "pid=${pid}"
if [[ -n "$pid" ]]; then ps -p "$pid" -o pid,etime,cmd || true; fi
echo "--- active ---"
ps aux | grep -E 'run_v10_final_full.sh|track2_multi_agent_plus.py' | grep -v grep || true
echo "--- launcher tail ---"
tail -n 40 logs/v10_final_full_launcher_20260618_1940.log 2>/dev/null || true
echo "--- result counts ---"
cd /home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
python3 - <<'PY'
import json, pathlib
root=pathlib.Path('results/V10_full_memory_final_candidate_draft')
print('root_exists', root.exists())
for p in sorted(root.glob('*_easy.json')):
    try:
        print(p.name, len(json.loads(p.read_text())))
    except Exception as e:
        print(p.name, type(e).__name__)
PY
