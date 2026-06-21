#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V21 retail failure map from V20/V19 diagnostics.

The failure map is diagnostic-only.  GT information appears only in
post-eval summaries and is not consumed by the V21 runtime resolver.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def first_wrong_from_eval(record: Dict[str, Any], label: str) -> str:
    ev = (record.get("eval") or {}).get(label) or {}
    if ev.get("matches", 0) == ev.get("gt_calls", -1) and ev.get("gt_calls", 0):
        return "none"
    prog_key = "nonoracle_program" if label == "V20_nonoracle" else "repair_program"
    prog = record.get(prog_key) or []
    return (prog[0] or {}).get("tool_name", "empty") if prog else "empty"


def main() -> None:
    v20 = read_json(CODEX / "analysis" / "v20_clean_retail_three_trace.json", {})
    clean_audit = {r.get("uid"): r for r in read_jsonl(CODEX / "analysis" / "VAL41_CLEAN_AUDIT_val41_clean_20260620_.jsonl")}
    rows = []
    for rec in v20.get("records", []):
        uid = f"{rec.get('spec')}::{rec.get('source_original_index')}"
        audit = clean_audit.get(uid, {})
        v20_eval = (rec.get("eval") or {}).get("V20_nonoracle") or {}
        row = {
            "spec": rec.get("spec"),
            "index": rec.get("source_original_index"),
            "user_utterance": "",
            "v19_first_wrong_tool": first_wrong_from_eval(rec, "V19"),
            "v20_nonoracle_first_wrong_tool": first_wrong_from_eval(rec, "V20_nonoracle"),
            "expected_tool_type_from_post_eval": audit.get("gt_tool_names", []),
            "broad_scan_detected": rec.get("spec") == "retail4" and rec.get("clean_pos") == 0,
            "branch_target_missing": not v20_eval.get("joint", False),
            "add_target_missing": v20_eval.get("matches", 0) < v20_eval.get("gt_calls", 0),
            "visual_candidate_status": {
                "qwen_path": rec.get("qwen_path"),
                "top_candidates": rec.get("top_candidates"),
            },
            "current_nonoracle_candidates": rec.get("top_candidates", []),
            "post_eval_gt_diff_summary": {
                "gt_tool_names": audit.get("gt_tool_names", []),
                "v20_nonoracle_matches": v20_eval.get("matches"),
                "v20_nonoracle_gt_calls": v20_eval.get("gt_calls"),
            },
        }
        rows.append(row)
    out = {
        "run_id": "v21_retail_failure_map_" + time.strftime("%Y%m%d_%H%M%S"),
        "runtime_uses_gt": False,
        "post_eval_gt_only": True,
        "rows": rows,
    }
    out_path = CODEX / "analysis" / "v21_retail_failure_map.json"
    write_json(out_path, out)
    report = CODEX / "reports" / f"V21_RETAIL_FAILURE_MAP_{out['run_id']}.md"
    lines = [
        f"# V21 Retail Failure Map {out['run_id']}",
        "",
        "- runtime_uses_gt: false",
        "- post_eval_gt_only: true",
        "",
        "| sample | V19 first wrong | V20 first wrong | branch target missing | add target missing |",
        "|---|---|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['spec']}::{r['index']} | {r['v19_first_wrong_tool']} | {r['v20_nonoracle_first_wrong_tool']} | {r['branch_target_missing']} | {r['add_target_missing']} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(CODEX / "state" / "latest_v21_failure_map.json", {"failure_map": str(out_path), "report": str(report)})
    print(json.dumps({"failure_map": str(out_path), "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
