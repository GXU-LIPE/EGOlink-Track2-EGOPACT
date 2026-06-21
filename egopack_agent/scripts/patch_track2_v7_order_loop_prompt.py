#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch V7 order no-visual behavior and compute loop ledger."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
HC = CODEX / "wrappers" / "egobench_agent_plus" / "human_prior_controller.py"
DG = CODEX / "wrappers" / "egobench_agent_plus" / "db_guard.py"


def backup(path: Path) -> None:
    dst = CODEX / "backups" / f"v7_order_loop_prompt_{time.strftime('%Y%m%d_%H%M%S')}" / path.relative_to(CODEX)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def replace(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"pattern_not_found {path}")
        return False
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path}")
    return True


def main() -> int:
    replace(HC, '''- Do not modify an order before restaurant_name is pinned.
- For kitchen, get recipe ingredients once, choose the branch from evidence, avoid broad scans, and compute nutrition only from confirmed ingredients.
''', '''- Do not modify an order before restaurant_name is pinned.
- Order with missing contact sheet: do not ask the simulated user for visual dish/category names. Use benchmark image_description/task analysis/layout hint plus restaurant-pinned retrieval to choose a grounded candidate, then continue the process.
- If an order aggregate compute returns 0.0 for a nonempty order, do not repeat the same aggregate call. Switch to the missing remove/add/final-tax stage or finish with the best supported process.
- For kitchen, get recipe ingredients once, choose the branch from evidence, avoid broad scans, and compute nutrition only from confirmed ingredients.
''')

    replace(DG, '''        if is_final_compute_tool(name) and _looks_success(result):
            fp = json.dumps({"name": name, "params": params}, ensure_ascii=False, sort_keys=True)
            state.setdefault("compute_call_ledger", {})[fp] = {"turn": turn, "tool_name": name}
''', '''        if is_final_compute_tool(name):
            fp = json.dumps({"name": name, "params": params}, ensure_ascii=False, sort_keys=True)
            state.setdefault("compute_call_ledger", {})[fp] = {"turn": turn, "tool_name": name, "success": _looks_success(result), "result_preview": str(result)[:300]}
''')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
