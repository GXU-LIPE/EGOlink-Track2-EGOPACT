#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch V7 process graph to avoid unnecessary aggregate loops."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
HPG = CODEX / "wrappers" / "egobench_agent_plus" / "human_process_graph.py"
TAM = CODEX / "wrappers" / "egobench_agent_plus" / "tool_affordance_memory.py"


def backup(path: Path) -> None:
    dst = CODEX / "backups" / f"v7_needs_aggregate_{time.strftime('%Y%m%d_%H%M%S')}" / path.relative_to(CODEX)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def replace(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"pattern not found in {path}: {old[:60]!r}")
        return False
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path}")
    return True


def main() -> int:
    replace(HPG, '''def _text_needs_mutation(text: str) -> bool:
    t = str(text or "").lower()
    return bool(re.search(r"\\b(add|remove|replace|delete|order|cart|shopping list|menu|include|exclude)\\b", t))


def infer_process_state(scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
''', '''def _text_needs_mutation(text: str) -> bool:
    t = str(text or "").lower()
    return bool(re.search(r"\\b(add|remove|replace|delete|order|cart|shopping list|menu|include|exclude)\\b", t))


def _text_needs_aggregate(text: str) -> bool:
    t = str(text or "").lower()
    return bool(re.search(r"\\b(total|tax|payment|payable|price|cost|nutrition|nutritions|calorie|calories|protein|fat|carb|sodium|discount)\\b", t))


def infer_process_state(scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
''')
    replace(HPG, '''    if scenario == "order":
        if not pins.get("restaurant_name"):
''', '''    needs_aggregate = _text_needs_aggregate(instruction)
    if scenario == "order":
        if state.get("order_requested_final_aggregate"):
            needs_aggregate = True
        if not pins.get("restaurant_name"):
''')
    replace(HPG, '''        elif (coverage["has_add"] or coverage["has_remove"] or state.get("order_requested_final_aggregate")) and not coverage["has_aggregate"]:
            stage = "compute_tax_or_payment"
            missing.append("final_aggregate")
''', '''        elif (coverage["has_add"] or coverage["has_remove"] or state.get("order_requested_final_aggregate")) and needs_aggregate and not coverage["has_aggregate"]:
            stage = "compute_tax_or_payment"
            missing.append("final_aggregate")
''')
    replace(HPG, '''        elif (coverage["has_add"] or coverage["has_remove"]) and not coverage["has_aggregate"]:
            stage = "compute_total_or_nutrition"
''', '''        elif (coverage["has_add"] or coverage["has_remove"]) and needs_aggregate and not coverage["has_aggregate"]:
            stage = "compute_total_or_nutrition"
''')

    replace(TAM, '''    for name in sorted(schema):
        fam = affordance_for_tool(name).get("family")
        if fam in wanted:
            out.append(name)
        if len(out) >= limit:
            break
''', '''    disallowed_management = {"add_product", "delete_product", "update_product", "add_dish", "delete_dish", "update_dish", "add_set_meal", "delete_set_meal", "update_set_meal"}
    for name in sorted(schema):
        if name in disallowed_management:
            continue
        fam = affordance_for_tool(name).get("family")
        if fam in wanted:
            out.append(name)
        if len(out) >= limit:
            break
''')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
