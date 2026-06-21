#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def risk_flags(instruction: str, scenario: str, v14_joint: bool, v22_joint: bool, visual_status: str) -> List[str]:
    text = norm(instruction)
    out = []
    if not v14_joint:
        out.append("v14_failed")
    if not v22_joint:
        out.append("v22_failed")
    if any(x in text for x in ("point", "shelf", "left", "right", "bottle", "dish", "menu", "pot", "tray", "box")):
        out.append("visual_entity_missing" if visual_status in {"missing", "grounding_failed"} else "visual_grounding_needed")
    if any(x in text for x in (" if ", "whether", "otherwise", "else", "tied", "tie")):
        out.append("branch_target_missing")
    if scenario in {"order", "restaurant"} and any(x in text for x in ("dish", "set meal", "menu", "category")):
        out.append("dish_setmeal_confusion")
    if scenario == "order" and "restaurant" in text:
        out.append("restaurant_pin_missing")
    if any(x in text for x in ("total", "payment", "tax", "nutrition", "summary")):
        out.append("missing_closure")
    if scenario == "retail" and any(x in text for x in ("price", "cheapest", "lowest", "highest", "discount")):
        out.append("broad_scan")
    return sorted(set(out))


def load_summary(path: Path) -> Dict[str, Dict[str, bool]]:
    # Not enough per-task info in state; use selection traces when available.
    out: Dict[str, Dict[str, bool]] = {}
    trace = CODEX / "analysis" / "v22_guarded_selection_trace.jsonl"
    if trace.exists():
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row.get('spec')}::{row.get('index')}"
            out[key] = {
                "v14_joint": bool(row.get("v14_score_post_eval", {}).get("joint")),
                "v22_joint": bool(row.get("selected_score_post_eval", {}).get("joint")),
            }
    return out


def qwen_status(spec: str, pos: int) -> str:
    p = CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos + 1}.json"
    data = read_json(p)
    if not isinstance(data, dict):
        return "missing"
    if data.get("grounding_failed") or not data.get("top_k_candidates"):
        return "grounding_failed"
    return data.get("status", "present")


def main() -> None:
    manifest = read_json(SPLIT_DIR / "manifest.json", {})
    traces = load_summary(CODEX / "state" / "latest_v22_guarded_val41_shadow.json")
    rows = []
    for scenario, number, _idxs in manifest.get("specs", []):
        spec = f"{scenario}{number}"
        items = read_json(SPLIT_DIR / f"{spec}.json", [])
        for pos, item in enumerate(items):
            idx = item.get("_v8_original_index", pos)
            key = f"{spec}::{idx}"
            status = qwen_status(spec, pos)
            v14_joint = traces.get(key, {}).get("v14_joint", False)
            v22_joint = traces.get(key, {}).get("v22_joint", False)
            rows.append({
                "spec": spec,
                "index": idx,
                "local_pos": pos,
                "scenario": scenario,
                "instruction": item.get("Instruction", ""),
                "dialogue_summary": "",
                "visual_card_status": status,
                "v14_joint": v14_joint,
                "v22_joint": v22_joint,
                "known_risk": risk_flags(item.get("Instruction", ""), scenario, v14_joint, v22_joint, status),
                "baseline_candidates_available": ["V14", "V22", "V19_CASE"],
            })
    out = CODEX / "analysis" / "v23_val41_task_inventory.jsonl"
    write_jsonl(out, rows)
    report = CODEX / "reports" / f"V23_TASK_INVENTORY_{time.strftime('%Y%m%d_%H%M%S')}.md"
    by_scenario: Dict[str, int] = {}
    for r in rows:
        by_scenario[r["scenario"]] = by_scenario.get(r["scenario"], 0) + 1
    lines = [
        "# V23 Task Inventory",
        "",
        f"- total_tasks: {len(rows)}",
        f"- by_scenario: `{json.dumps(by_scenario, ensure_ascii=False)}`",
        f"- output: `{out}`",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "",
        "| scenario | spec | index | v14 | v22 | visual | risks |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['spec']} | {r['index']} | {int(r['v14_joint'])} | {int(r['v22_joint'])} | {r['visual_card_status']} | {', '.join(r['known_risk'])} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = {"inventory": str(out), "report": str(report), "tasks": len(rows), "final_run": False, "uses_final_hidden_metadata": False}
    (CODEX / "state" / "latest_v23_inventory.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
