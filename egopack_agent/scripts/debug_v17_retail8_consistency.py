#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check internal consistency of the single retail8 debug sample."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
RUN_ID = "v17_smoke5_20260620_1140"
SAMPLE = CODEX / "state" / "materialized_splits" / f"v17_smoke5_{RUN_ID}" / "retail8.json"


def ids(text: str):
    return sorted(set(re.findall(r"\buser_\d+\b", text or "")))


def products_from_gt(gt):
    out = []
    for call in gt or []:
        params = call.get("parameters") or {}
        if params.get("product_name"):
            out.append(params.get("product_name"))
        for item in params.get("products") or []:
            if isinstance(item, dict) and item.get("product_name"):
                out.append(item.get("product_name"))
    return sorted(set(out))


def main():
    row = json.loads(SAMPLE.read_text(encoding="utf-8"))[0]
    instruction = row.get("Instruction") or ""
    analysis = row.get("analysis") or ""
    image_desc = row.get("image_description") or ""
    gt = row.get("ground_truth") or []
    gt_text = json.dumps(gt, ensure_ascii=False)
    payload = {
        "sample_path": str(SAMPLE),
        "task_id": row.get("task_id"),
        "_v8_original_index": row.get("_v8_original_index"),
        "instruction_user_ids": ids(instruction),
        "analysis_user_ids": ids(analysis),
        "gt_user_ids": ids(gt_text),
        "instruction_mentions": {
            "price_condition": "price" in instruction.lower(),
            "high_sugar": "high sugar" in instruction.lower(),
            "sour_type": "sour" in instruction.lower(),
            "total_payment": "total amount payable" in instruction.lower() or "payment" in instruction.lower(),
            "total_nutrition": "nutrition" in instruction.lower() or "protein" in instruction.lower(),
        },
        "analysis_mentions": {
            "fat_condition": "fat content" in analysis.lower(),
            "origin_france": "france" in analysis.lower(),
            "milk": "milk" in analysis.lower(),
            "total_protein": "protein" in analysis.lower(),
            "total_payment": "payable" in analysis.lower() or "payment" in analysis.lower(),
        },
        "gt_tool_names": [c.get("tool_name") for c in gt],
        "gt_products": products_from_gt(gt),
        "key": row.get("key"),
        "value": row.get("value"),
        "image_description": image_desc,
    }
    failures = []
    if payload["instruction_user_ids"] and payload["gt_user_ids"] and payload["instruction_user_ids"] != payload["gt_user_ids"]:
        failures.append("instruction_user_id_vs_gt_user_id_mismatch")
    if payload["instruction_mentions"]["price_condition"] and payload["analysis_mentions"]["fat_condition"]:
        failures.append("instruction_vs_analysis_branch_condition_mismatch")
    if payload["instruction_mentions"]["sour_type"] and "find_products_by_taste" not in payload["gt_tool_names"]:
        failures.append("instruction_requests_sour_search_but_gt_has_no_taste_search")
    if payload["instruction_mentions"]["total_payment"] and "compute_total_payment" not in payload["gt_tool_names"]:
        failures.append("instruction_requests_payment_but_gt_missing_payment")
    if payload["analysis_mentions"]["total_protein"] and "compute_total_nutrition" not in payload["gt_tool_names"]:
        failures.append("analysis_requests_protein_but_gt_missing_nutrition")
    payload["consistency_failures"] = failures
    payload["consistent_enough_for_chain_debug"] = not failures

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = CODEX / "analysis" / f"V17_RETAIL8_SAMPLE_CONSISTENCY_{ts}.json"
    out_md = CODEX / "reports" / f"V17_RETAIL8_SAMPLE_CONSISTENCY_{ts}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# V17 Retail8 Sample Consistency {ts}",
        "",
        f"- sample_path: `{SAMPLE}`",
        f"- task_id: {payload['task_id']}",
        f"- original_index: {payload['_v8_original_index']}",
        f"- consistent_enough_for_chain_debug: {str(payload['consistent_enough_for_chain_debug']).lower()}",
        "",
        "## Failures",
        "",
    ]
    lines += [f"- {x}" for x in failures] or ["- none"]
    lines += [
        "",
        "## Evidence",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Conclusion",
        "",
        "This retail8 row is not a valid single-sample closed-loop debugger target because instruction, analysis, and ground_truth do not describe the same task/user. Fix split/materialization hygiene first, then rerun the chain debugger on a consistent sample.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_md), "json": str(out_json), "failures": failures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
