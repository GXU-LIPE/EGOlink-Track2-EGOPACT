#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
GT17 = CODEX / "gt_distill_v17"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("valid") is True:
            rows.append(row)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def bucket_key(row: Dict[str, Any]) -> str:
    return f"{row.get('scenario')}::{row.get('intent_type') or row.get('task_type')}"


def main() -> None:
    rules = read_jsonl(GT17 / "gpt55_distilled_rules.jsonl")
    skeleton = collections.defaultdict(list)
    slot_index = collections.defaultdict(list)
    order_pin = []
    branch_index = collections.defaultdict(list)
    closure_index = collections.defaultdict(list)
    anti_index = collections.defaultdict(list)
    scenario_counts = collections.Counter()
    intent_counts = collections.Counter()
    for row in rules:
        scenario = row.get("scenario") or "unknown"
        intent = row.get("intent_type") or row.get("task_type") or "unknown"
        scenario_counts[scenario] += 1
        intent_counts[f"{scenario}::{intent}"] += 1
        key = bucket_key(row)
        skel = row.get("minimal_tool_skeleton") or []
        if skel:
            skeleton[key].append({"sample_id": row.get("sample_id"), "stages": skel, "rule": row.get("generalizable_rule", "")[:300]})
        slots = row.get("required_slots") or {}
        slot_rules = row.get("slot_resolution_rules") or []
        slot_index[key].append({"sample_id": row.get("sample_id"), "required_slots": slots, "slot_resolution_rules": slot_rules[:8]})
        if scenario == "order":
            order_pin.append({"sample_id": row.get("sample_id"), "required_slots": slots, "slot_resolution_rules": slot_rules[:8], "branch": row.get("branch_compiler_rules") or []})
        if row.get("branch_compiler_rules"):
            branch_index[key].append({"sample_id": row.get("sample_id"), "rules": row.get("branch_compiler_rules")[:5]})
        if row.get("closure_rules"):
            closure_index[key].append({"sample_id": row.get("sample_id"), "rules": row.get("closure_rules")[:8]})
        for anti in row.get("anti_patterns") or []:
            anti_index[scenario].append({"sample_id": row.get("sample_id"), "anti_pattern": anti})

    def trim_dict(d, limit=60):
        return {k: v[:limit] for k, v in sorted(d.items())}

    outputs = {
        "tool_skeleton_index.json": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_valid_rules": len(rules),
            "index": trim_dict(skeleton),
        },
        "slot_resolver_index.json": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_valid_rules": len(rules),
            "index": trim_dict(slot_index),
        },
        "order_restaurant_pin_index.json": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_valid_rules": len(rules),
            "rules": order_pin[:120],
        },
        "branch_compiler_index.json": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_valid_rules": len(rules),
            "index": trim_dict(branch_index),
        },
        "closure_repair_index.json": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_valid_rules": len(rules),
            "index": trim_dict(closure_index),
        },
        "anti_broad_scan_index.json": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_valid_rules": len(rules),
            "index": trim_dict(anti_index, limit=200),
        },
    }
    for name, obj in outputs.items():
        write_json(GT17 / name, obj)
    manifest = {
        "status": "PASS" if len(rules) >= 600 else "FAIL",
        "valid_rules": len(rules),
        "scenario_counts": dict(scenario_counts),
        "intent_counts": dict(intent_counts),
        "outputs": {name: str(GT17 / name) for name in outputs},
        "uses_final_hidden_metadata": False,
        "uses_val41_label": False,
    }
    report = CODEX / "reports" / f"V17_COMPILER_BUILD_{time.strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# V17 Compiler Build",
        "",
        f"- status: {manifest['status']}",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_label: false",
        f"- valid_rules: {len(rules)}",
        "",
        "## Scenario Counts",
    ]
    for k, v in sorted(scenario_counts.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Outputs"]
    for name in outputs:
        lines.append(f"- `gt_distill_v17/{name}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["report"] = str(report)
    write_json(GT17 / "compiler_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
