#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
echo "--- before kill ---"
ps aux | grep -E 'run_v10_final_full.sh|track2_multi_agent_plus.py --scenario .*--final_eval' | grep -v grep || true
pkill -f 'run_v10_final_full.sh' || true
pkill -f 'track2_multi_agent_plus.py --scenario .*--final_eval' || true
sleep 3
echo "--- after kill ---"
ps aux | grep -E 'run_v10_final_full.sh|track2_multi_agent_plus.py --scenario .*--final_eval' | grep -v grep || true
rm -rf /home/data-gxu/acm/egolink2026-main/code/track2/EgoBench/results/V10_full_memory_final_candidate_draft
mkdir -p /home/data-gxu/acm/egolink2026-main/code/track2/EgoBench/results/V10_full_memory_final_candidate_draft
rm -f state/v10_final_full_20260618_1940.pid
