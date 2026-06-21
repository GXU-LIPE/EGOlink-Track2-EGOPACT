#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
latest = sorted((CODEX / "runs" / "V8_tmp_scenarios").glob("validation_A_medium_*/split_hygiene.json"))
if not latest:
    raise SystemExit("no split_hygiene.json found")
path = latest[-1]
data = json.loads(path.read_text(encoding="utf-8"))
ts = time.strftime("%Y%m%d_%H%M%S")
report = CODEX / "reports" / f"V9_SPLIT_HYGIENE_FIX_{ts}.md"
lines = [
    f"# V9_SPLIT_HYGIENE_FIX {ts}",
    "",
    "- stage: validation_A_medium dry-run",
    "- final_submission: not submitted",
    "- protected_best_updated: false",
    f"- hygiene_json: `{path}`",
    "",
    "## Result",
    "",
    f"- planned_task_count_after_filter: {data.get('planned_task_count', 0)}",
    f"- skipped_invalid_indices: {len(data.get('skipped_invalid_indices') or [])}",
    "",
    "## Skipped Indices",
    "",
]
for item in data.get("skipped_invalid_indices") or []:
    lines.append(f"- `{item.get('uid')}` skipped: idx={item.get('idx')}, available_count={item.get('available_count')}, reason={item.get('reason')}")
lines += [
    "",
    "## Planned Specs After Filter",
    "",
    "```json",
    json.dumps(data.get("planned_specs"), ensure_ascii=False, indent=2),
    "```",
    "",
    "## Notes",
    "",
    "- The frozen validation_A split is otherwise preserved.",
    "- Out-of-range indices are filtered before temp scenario JSON generation, so the runner will not crash after completing valid samples.",
]
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
