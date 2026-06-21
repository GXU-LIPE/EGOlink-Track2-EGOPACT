#!/usr/bin/env bash
set -euo pipefail
cd /home/data-gxu/acm/egolink2026-main/code/track2/codex
TS=$(date +%Y%m%d_%H%M%S)
REPORT="reports/V9_SPLIT_HYGIENE_FIX_${TS}.md"
MAT="state/materialized_splits/validation_A_limit30"
{
  echo "# V9 Split Hygiene Fix ${TS}"
  echo
  echo "- protected_best: V6_1_3_gpt55_guarded_endpoint unchanged"
  echo "- materialized_split: \`${MAT}\`"
  echo "- strategy: freeze validation_A_medium from the known V9_4_5 41-task subset when available; later runs read codex/state/materialized_splits instead of mutable scenarios/final/*.json"
  echo "- final_submission: not submitted"
  echo
  if [[ -f "${MAT}/manifest.json" ]]; then
    python3 - <<'PY'
import json, pathlib, hashlib
p=pathlib.Path('state/materialized_splits/validation_A_limit30/manifest.json')
d=json.loads(p.read_text())
print('## Materialized Manifest')
print()
print(f"- source: `{d.get('source')}`")
print(f"- planned_task_count: {d.get('planned_task_count')}")
print(f"- skipped_invalid_indices: {len(d.get('skipped_invalid_indices') or [])}")
for item in d.get('skipped_invalid_indices') or []:
    print(f"- skipped `{item.get('uid')}`: idx={item.get('idx')} available_count={item.get('available_count')} reason={item.get('reason')}")
print()
print('## Files')
print()
for f in d.get('files', []):
    print(f"- `{f.get('file')}` tasks={f.get('task_count')} sha256={f.get('sha256')}")
PY
  else
    echo "- status: manifest not created yet; run dry-run first."
  fi
} > "$REPORT"
echo "$REPORT"
