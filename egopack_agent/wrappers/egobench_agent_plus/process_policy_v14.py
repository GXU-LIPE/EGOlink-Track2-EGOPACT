# -*- coding: utf-8 -*-
"""V14 process-policy prompt cards for Track2.

This module is intentionally prompt-side only unless explicitly enabled by
TRACK2_ENABLE_V14_PROCESS_POLICY=1. It does not read final metadata.
"""

import json
import os
from pathlib import Path


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def _read_jsonl(path: Path, limit: int = 8):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
        if len(rows) >= limit:
            break
    return rows


def _scenario_cards(path: Path, scenario: str, limit: int = 4):
    rows = []
    for row in _read_jsonl(path, limit=500):
        sc = str(row.get("scenario") or "")
        if sc in {"global", scenario, ""}:
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def infer_v14_task_type(goal: str, scenario: str) -> str:
    text = (goal or "").lower()
    if any(x in text for x in ["add", "remove", "delete", "update", "cart", "order", "menu", "shopping list"]):
        if any(x in text for x in ["if ", "otherwise", "else", "whether", "exceeds", "less than", "greater than"]):
            return "branch-then-mutation"
        return "cart/order/menu mutation"
    if any(x in text for x in ["total", "tax", "payment", "nutrition", "calories", "carbohydrate", "protein", "fat"]):
        return "aggregate-required"
    if any(x in text for x in ["highest", "lowest", "cheapest", "most", "least"]):
        return "ranking/filtering"
    if any(x in text for x in ["pointed", "visible", "menu", "image", "video", "looked at"]):
        return "visual-entity query"
    return "query-only"


def build_v14_process_policy_prompt(scenario: str) -> str:
    if os.environ.get("TRACK2_ENABLE_V14_PROCESS_POLICY") != "1":
        return ""
    bank = Path(os.environ.get("TRACK2_V14_MEMORY_BANK_DIR") or (CODEX_ROOT / "memory_bank_v14_gt_trajectory"))
    goal = os.environ.get("TRACK2_CURRENT_USER_GOAL", "")
    task_type = infer_v14_task_type(goal, scenario)

    skeletons = _read_jsonl(bank / "minimal_process_skeletons.jsonl", limit=20)
    skeleton = next((x for x in skeletons if x.get("task_type") == task_type), None)
    anti = _scenario_cards(bank / "anti_broad_scan_rules.jsonl", scenario, limit=4)
    slots = _read_jsonl(bank / "entity_slot_mapping_rules.jsonl", limit=6)
    seq = _scenario_cards(bank / "tool_sequence_templates.jsonl", scenario, limit=4)
    shortcuts = _read_jsonl(bank / "query_only_shortcuts.jsonl", limit=4)

    lines = [
        "[V14 GT-Trajectory Distilled Process Policy]",
        f"- inferred_task_type: {task_type}",
        "- This is a hard process policy for val41 experimentation, but it must not use final hidden metadata.",
        "- Prefer minimal GT-like tool skeletons over broad exploration.",
        "- Mutation before canonical entity resolution is disallowed.",
        "- End-of-task closure: if the user requested aggregate/tax/payment/nutrition/order/cart summary, call the minimal matching aggregate/summary tool before final answer.",
    ]
    if skeleton:
        lines.append("- minimal_skeleton: " + " -> ".join(str(x) for x in skeleton.get("steps", [])))
    if anti:
        lines.append("- anti_broad_scan_rules:")
        for item in anti:
            lines.append(f"  * {item.get('rule')}")
    if shortcuts and task_type == "query-only":
        lines.append("- query_only_shortcuts:")
        for item in shortcuts:
            lines.append(f"  * {item.get('rule')}")
    if seq:
        lines.append("- scenario GT-like tool sequence patterns:")
        for item in seq:
            names = item.get("tool_names") or []
            if names:
                lines.append(f"  * {item.get('task_type')}: " + " -> ".join(names[:8]))
    if slots:
        lines.append("- entity slot discipline:")
        for item in slots[:4]:
            tool = item.get("tool_name")
            ents = item.get("entity_params") or []
            if tool and ents:
                lines.append(f"  * {tool}: resolve/canonicalize {', '.join(ents)} before call")
    return "\n".join(lines)
