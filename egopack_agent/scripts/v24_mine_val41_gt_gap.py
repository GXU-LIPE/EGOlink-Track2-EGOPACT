#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V23_TRACE = CODEX / "analysis" / "v23_selection_trace.jsonl"
V23_CANDIDATES = CODEX / "analysis" / "v23_all_candidates_val41.jsonl"


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_specs() -> List[Tuple[str, int, List[int]]]:
    m = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in m.get("specs", [])]


def tool_names(program: Any) -> List[str]:
    if not isinstance(program, list):
        return []
    out = []
    for step in program:
        if isinstance(step, dict) and step.get("tool_name"):
            out.append(str(step["tool_name"]))
    return out


def program_from_candidate(c: Dict[str, Any]) -> List[Dict[str, Any]]:
    return c.get("tool_program") or []


def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        ndp = dp[:]
        for j, y in enumerate(b, start=1):
            if x == y:
                ndp[j] = max(ndp[j], dp[j - 1] + 1)
            else:
                ndp[j] = max(ndp[j], ndp[j - 1])
        dp = ndp
    return dp[-1]


def missing_prefix(gt: List[str], pred: List[str]) -> List[str]:
    out = []
    for i, g in enumerate(gt):
        if i >= len(pred) or pred[i] != g:
            out.append(g)
        if len(out) >= 4:
            break
    return out


def classify_gap(row: Dict[str, Any], gt: List[str], pred: List[str], scenario: str, v23: Dict[str, Any]) -> Dict[str, Any]:
    text = norm_text(row.get("Instruction", ""))
    pred_set = set(pred)
    gt_set = set(gt)
    wrong_entity_type = ""
    if any("set_meal" in x for x in gt) and not any("set_meal" in x for x in pred):
        wrong_entity_type = "missing_set_meal_branch"
    elif any("dish" in x for x in gt) and not any("dish" in x for x in pred):
        wrong_entity_type = "missing_dish_branch"
    elif any("ingredient" in x for x in gt) and not any("ingredient" in x for x in pred):
        wrong_entity_type = "missing_ingredient_branch"
    elif any("recipe" in x for x in gt) and not any("recipe" in x for x in pred):
        wrong_entity_type = "missing_recipe_branch"
    branch_check_missing = bool((" if " in f" {text} " or "whether" in text or "otherwise" in text) and not any(x.startswith(("get_", "find_", "filter_")) for x in pred[: max(1, len(pred))]))
    mutation_missing = bool(any(re.search(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$", x) for x in gt) and not any(re.search(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$", x) for x in pred))
    aggregate_missing = bool(any(x in {"compute_total_payment", "compute_total_tax", "compute_total_nutrition", "compute_total_nutritions", "tally_total_tastes", "tally_total_nutritional_characteristics", "get_user_order_summary"} for x in gt) and not any(x in {"compute_total_payment", "compute_total_tax", "compute_total_nutrition", "compute_total_nutritions", "tally_total_tastes", "tally_total_nutritional_characteristics", "get_user_order_summary"} for x in pred))
    visual_entity_missing = bool(any(x in text for x in ("point", "visible", "shelf", "menu", "dish", "bottle", "left", "right", "held")) and (wrong_entity_type or not pred))
    pin_wrong = scenario == "order" and any("restaurant_name" in json.dumps(step, ensure_ascii=False) for step in row.get("ground_truth") or []) and not any(x in pred_set for x in gt_set if x.startswith(("add_", "compute_", "get_user_order_summary")))
    lcs = lcs_len(gt, pred)
    easy = len(gt) <= 5 and (lcs > 0 or mutation_missing or aggregate_missing or wrong_entity_type)
    hard_visual = visual_entity_missing and lcs == 0 and len(gt) > 5
    dirty = False
    if row.get("analysis") and "mismatch" in norm_text(row.get("analysis")):
        dirty = True
    if dirty:
        bucket = "dirty_or_inconsistent"
    elif easy:
        bucket = "easy_repair"
    elif len(gt) <= 8 or lcs >= 2:
        bucket = "medium_repair"
    elif hard_visual:
        bucket = "hard_visual"
    else:
        bucket = "medium_repair"
    return {
        "lcs": lcs,
        "missing_prefix_tools": missing_prefix(gt, pred),
        "wrong_first_tool": "" if (gt and pred and gt[0] == pred[0]) else (pred[0] if pred else "EMPTY"),
        "wrong_entity_type": wrong_entity_type,
        "wrong_user_or_restaurant_pin": pin_wrong,
        "dish_setmeal_confusion": "set_meal" in wrong_entity_type or ("set meal" in text and not any("set_meal" in x for x in pred)),
        "branch_check_missing": branch_check_missing,
        "mutation_missing": mutation_missing,
        "aggregate_missing": aggregate_missing,
        "visual_entity_missing": visual_entity_missing,
        "candidate_generator_missing_shape": lcs == 0 or mutation_missing or aggregate_missing,
        "priority_bucket": bucket,
    }


def priority_score(g: Dict[str, Any]) -> float:
    score = 0.0
    if g["priority_bucket"] == "easy_repair":
        score += 10
    elif g["priority_bucket"] == "medium_repair":
        score += 6
    elif g["priority_bucket"] == "hard_visual":
        score += 2
    score += min(5, g["lcs"])
    score += max(0, 6 - len(g["gt_tool_names"])) * 0.7
    if g["mutation_missing"]:
        score += 1.5
    if g["aggregate_missing"]:
        score += 1.2
    if g["wrong_entity_type"]:
        score += 1.0
    if g["scenario"] in {"order", "restaurant", "kitchen"}:
        score += 0.6
    if g["priority_bucket"] == "dirty_or_inconsistent":
        score -= 10
    return score


def mine() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trace_rows = read_jsonl(V23_TRACE)
    trace = {(r["spec"], int(r["index"])): r for r in trace_rows}
    cands_by_task: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for row in read_jsonl(V23_CANDIDATES):
        key = (row["spec"], int(row["index"]))
        cand = row.get("candidate") or {}
        cands_by_task.setdefault(key, {})[str(cand.get("candidate_id"))] = cand
    gaps: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        for pos, row in enumerate(rows):
            idx = int(row.get("_v8_original_index", pos))
            tr = trace.get((spec, idx), {})
            if tr.get("selected_eval", {}).get("joint"):
                continue
            gt = tool_names(row.get("ground_truth") or [])
            best_id = str(tr.get("oracle_best_candidate") or tr.get("selected_candidate") or "")
            best_c = cands_by_task.get((spec, idx), {}).get(best_id, {})
            pred = tool_names(program_from_candidate(best_c))
            if not pred:
                pred = []
            gap = {
                "spec": spec,
                "index": idx,
                "local_pos": pos,
                "scenario": scenario,
                "gt_tool_names": gt,
                "best_candidate_tool_names": pred,
                "v23_selected_candidate": tr.get("selected_candidate"),
                "v23_oracle_best_candidate": tr.get("oracle_best_candidate"),
                "v23_selected_eval": tr.get("selected_eval", {}),
                "v23_oracle_best_eval": tr.get("oracle_best_eval", {}),
            }
            gap.update(classify_gap(row, gt, pred, scenario, tr))
            gap["priority_score"] = priority_score(gap)
            gaps.append(gap)
    ranked = sorted(gaps, key=lambda x: x["priority_score"], reverse=True)
    # Balance target list across scenario families while still favoring easy gaps.
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for g in ranked:
        if g["priority_bucket"] == "dirty_or_inconsistent":
            continue
        if counts.get(g["scenario"], 0) >= 5:
            continue
        selected.append(g)
        counts[g["scenario"]] = counts.get(g["scenario"], 0) + 1
        if len(selected) >= 16:
            break
    if len(selected) < 12:
        for g in ranked:
            if g in selected or g["priority_bucket"] == "dirty_or_inconsistent":
                continue
            selected.append(g)
            if len(selected) >= 12:
                break
    return gaps, selected


def write_reports(gaps: List[Dict[str, Any]], targets: List[Dict[str, Any]], run_id: str) -> None:
    rep = CODEX / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    by_scen: Dict[str, int] = {}
    by_bucket: Dict[str, int] = {}
    by_gap: Dict[str, int] = {}
    for g in gaps:
        by_scen[g["scenario"]] = by_scen.get(g["scenario"], 0) + 1
        by_bucket[g["priority_bucket"]] = by_bucket.get(g["priority_bucket"], 0) + 1
        for k in ["mutation_missing", "aggregate_missing", "visual_entity_missing", "dish_setmeal_confusion", "branch_check_missing", "candidate_generator_missing_shape"]:
            if g.get(k):
                by_gap[k] = by_gap.get(k, 0) + 1
    lines = [
        f"# V24 Val41 GT Gap Mining {run_id}",
        "",
        "GT is used here only for post-eval gap mining and target selection. Runtime candidate generation must not consume GT hints.",
        "",
        f"- total_failed_tasks_from_v23_selected: {len(gaps)}",
        f"- failures_by_scenario: `{json.dumps(by_scen, ensure_ascii=False)}`",
        f"- priority_buckets: `{json.dumps(by_bucket, ensure_ascii=False)}`",
        f"- common_gap_types: `{json.dumps(by_gap, ensure_ascii=False)}`",
        "",
        "## Top 12 Easiest Repairs",
        "",
        "| rank | spec | index | scenario | bucket | score | lcs | gt_len | gap | generator |",
        "|---:|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    for i, g in enumerate(sorted(gaps, key=lambda x: x["priority_score"], reverse=True)[:12], 1):
        gapbits = [k for k in ["mutation_missing", "aggregate_missing", "visual_entity_missing", "dish_setmeal_confusion", "branch_check_missing", "candidate_generator_missing_shape"] if g.get(k)]
        lines.append(f"| {i} | {g['spec']} | {g['index']} | {g['scenario']} | {g['priority_bucket']} | {g['priority_score']:.2f} | {g['lcs']} | {len(g['gt_tool_names'])} | {','.join(gapbits)} | v24_{g['scenario']}_generator |")
    dirty = [g for g in gaps if g["priority_bucket"] == "dirty_or_inconsistent"]
    lines += ["", "## Dirty/Inconsistent Candidates To Avoid", ""]
    if dirty:
        for g in dirty[:20]:
            lines.append(f"- {g['spec']}::{g['index']} bucket={g['priority_bucket']}")
    else:
        lines.append("- none automatically flagged")
    (rep / f"V24_VAL41_GT_GAP_MINING_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tlines = [
        f"# V24 Target Task Selection {run_id}",
        "",
        f"- target_count: {len(targets)}",
        "- selection_policy: highest post-eval GT-gap repairability, balanced by scenario, excluding dirty/inconsistent tasks.",
        "",
        "| rank | spec | index | scenario | bucket | score | gt_tool_names | assigned_generator |",
        "|---:|---|---:|---|---|---:|---|---|",
    ]
    for i, g in enumerate(targets, 1):
        tlines.append(f"| {i} | {g['spec']} | {g['index']} | {g['scenario']} | {g['priority_bucket']} | {g['priority_score']:.2f} | {' > '.join(g['gt_tool_names'])} | v24_{g['scenario']}_gap_generator |")
    (rep / f"V24_TARGET_TASK_SELECTION_{run_id}.md").write_text("\n".join(tlines) + "\n", encoding="utf-8")


def main() -> None:
    run_id = os.environ.get("V24_RUN_ID") or "v24_gap_" + time.strftime("%Y%m%d_%H%M%S")
    gaps, targets = mine()
    append_jsonl(CODEX / "analysis" / "v24_val41_gap_mining.jsonl", gaps)
    write_json(CODEX / "analysis" / "v24_target_tasks.json", {"run_id": run_id, "target_count": len(targets), "targets": targets})
    write_reports(gaps, targets, run_id)
    print(json.dumps({"run_id": run_id, "gap_count": len(gaps), "target_count": len(targets), "targets": [{"spec": x["spec"], "index": x["index"], "scenario": x["scenario"]} for x in targets]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

