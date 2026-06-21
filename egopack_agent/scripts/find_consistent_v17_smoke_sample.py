#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find a consistent single sample among V17 smoke5 materialized rows."""

from __future__ import annotations

import json
import re
from pathlib import Path

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
RUN_ID = "v17_smoke5_20260620_1140"
SMOKE_DIR = CODEX / "state" / "materialized_splits" / f"v17_smoke5_{RUN_ID}"


def ids(text: str):
    return sorted(set(re.findall(r"\buser_\d+\b", text or "")))


def flags(text: str):
    t = (text or "").lower()
    return {
        "price": "price" in t,
        "sugar": "sugar" in t,
        "sour": "sour" in t,
        "fat": "fat" in t,
        "france": "france" in t,
        "milk": "milk" in t,
        "protein": "protein" in t,
        "payment": "payment" in t or "payable" in t or "amount" in t,
        "nutrition": "nutrition" in t or "protein" in t or "fat" in t or "sugar" in t,
    }


def score(row):
    inst = row.get("Instruction") or row.get("instruction") or ""
    analysis = row.get("analysis") or ""
    gt = row.get("ground_truth") or []
    gt_text = json.dumps(gt, ensure_ascii=False)
    inst_ids = ids(inst)
    gt_ids = ids(gt_text)
    f_inst = flags(inst)
    f_analysis = flags(analysis)
    tool_names = [c.get("tool_name") for c in gt if isinstance(c, dict)]
    failures = []
    if inst_ids and gt_ids and inst_ids != gt_ids:
        failures.append("user_id_mismatch")
    if f_inst["sour"] and not any("taste" in str(x) for x in tool_names):
        failures.append("sour_requested_gt_no_taste_tool")
    if f_inst["payment"] and "compute_total_payment" not in tool_names:
        failures.append("payment_requested_gt_missing")
    if f_inst["nutrition"] and "compute_total_nutrition" not in tool_names:
        failures.append("nutrition_requested_gt_missing")
    if f_inst["price"] and f_analysis["fat"] and not f_inst["fat"]:
        failures.append("instruction_analysis_condition_mismatch")
    return {
        "task_id": row.get("task_id"),
        "orig": row.get("_v8_original_index"),
        "instruction_ids": inst_ids,
        "gt_ids": gt_ids,
        "tool_names": tool_names,
        "failures": failures,
        "usable": not failures,
        "instruction_preview": inst[:220],
        "analysis_preview": analysis[:220],
    }


def main():
    rows = []
    for p in sorted(SMOKE_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            continue
        item = score(data[0])
        item["spec"] = p.stem
        item["path"] = str(p)
        rows.append(item)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
