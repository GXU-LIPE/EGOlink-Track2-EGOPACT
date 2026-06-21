#!/usr/bin/env bash
set -euo pipefail

CODEX=/home/data-gxu/acm/egolink2026-main/code/track2/codex
EGO=/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP="$CODEX/backups/$TS"
PATCH_DIR="$CODEX/patches"

mkdir -p "$BACKUP/wrappers/egobench_agent_plus" "$BACKUP/runners" "$BACKUP/scripts" "$BACKUP/EgoBench/scenarios/final" "$PATCH_DIR"

cp "$CODEX/wrappers/egobench_agent_plus/planner.py" "$BACKUP/wrappers/egobench_agent_plus/planner.py"
cp "$CODEX/wrappers/egobench_agent_plus/prompt_builder.py" "$BACKUP/wrappers/egobench_agent_plus/prompt_builder.py"
cp "$CODEX/wrappers/egobench_agent_plus/tool_validator.py" "$BACKUP/wrappers/egobench_agent_plus/tool_validator.py"
cp "$CODEX/wrappers/egobench_agent_plus/canonical_resolver.py" "$BACKUP/wrappers/egobench_agent_plus/canonical_resolver.py"
cp "$CODEX/wrappers/egobench_agent_plus/db_guard.py" "$BACKUP/wrappers/egobench_agent_plus/db_guard.py"
cp "$CODEX/runners/track2_multi_agent_plus.py" "$BACKUP/runners/track2_multi_agent_plus.py"
cp "$CODEX/scripts/run_gpt55_endpoint_gate.sh" "$BACKUP/scripts/run_gpt55_endpoint_gate.sh"
cp "$EGO/scenarios/final/order1.json" "$BACKUP/EgoBench/scenarios/final/order1.json"

python3 - <<'PY'
from pathlib import Path
path = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench/scenarios/final/order1.json")
text = path.read_text(encoding="utf-8")
old = '"set_meal_name": "Cold Cuts & Cheese Platte"'
new = '"set_meal_name": "Cold Cuts & Cheese Platter"'
if old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY

{
  diff -u "$BACKUP/EgoBench/scenarios/final/order1.json" "$EGO/scenarios/final/order1.json" || true
} > "$PATCH_DIR/order1_official_pr7_gt_typo_${TS}.diff"

echo "$TS" > "$CODEX/state/last_v614_backup.txt"
echo "$BACKUP"
