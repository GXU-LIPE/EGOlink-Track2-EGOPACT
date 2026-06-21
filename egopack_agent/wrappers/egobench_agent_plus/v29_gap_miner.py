#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 val41 GT-gap miner.

This module is deliberately dev/val41-only. It reads val41 ground_truth for
gap analysis and repair prioritisation; it never reads final data.
"""

from __future__ import annotations

from typing import Any, Dict, List


MUTATION_PREFIXES = ("add_", "remove_", "delete_", "update_", "modify_")
AGG_TOOLS = {
    "compute_total_payment",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
    "get_user_order_summary",
}


def tool_names(program: List[Dict[str, Any]] | None) -> List[str]:
    return [str(x.get("tool_name") or "") for x in (program or [])]


def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def classify_gap(
    task_key: str,
    scenario: str,
    row: Dict[str, Any],
    v22_score: Dict[str, Any],
    best_existing_program: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    gt = row.get("ground_truth") or []
    gt_names = tool_names(gt)
    pred_names = tool_names(best_existing_program or [])
    lcs = lcs_len(gt_names, pred_names)
    gt_mut = [n for n in gt_names if n.startswith(MUTATION_PREFIXES)]
    pred_mut = [n for n in pred_names if n.startswith(MUTATION_PREFIXES)]
    missing_prefix = []
    for i, name in enumerate(gt_names):
        if i >= len(pred_names) or pred_names[i] != name:
            missing_prefix = gt_names[i : min(len(gt_names), i + 4)]
            break
    if not missing_prefix and len(pred_names) < len(gt_names):
        missing_prefix = gt_names[len(pred_names) : min(len(gt_names), len(pred_names) + 4)]

    wrong_user_or_restaurant = False
    for call in gt:
        params = call.get("parameters") or {}
        if params.get("restaurant_name") and scenario in {"order"}:
            wrong_user_or_restaurant = True
            break

    repairability = "hard"
    if gt_names and len(gt_names) <= 4:
        repairability = "easy"
    elif gt_names and len(gt_names) <= 8:
        repairability = "medium"
    if not gt_names:
        repairability = "dirty"

    assigned = scenario if scenario in {"retail", "order", "restaurant", "kitchen"} else "skip"
    if v22_score.get("joint"):
        assigned = "skip"

    return {
        "task_key": task_key,
        "scenario": scenario,
        "gt_tools": gt_names,
        "best_existing_candidate_tools": pred_names,
        "first_wrong_tool": "" if not missing_prefix else (pred_names[len(gt_names) - len(missing_prefix)] if pred_names else "EMPTY"),
        "missing_prefix": missing_prefix,
        "wrong_entity": bool(gt and not pred_names),
        "wrong_restaurant_pin": bool(scenario == "order" and wrong_user_or_restaurant),
        "dish_setmeal_confusion": bool(any("set_meal" in n for n in gt_names) and not any("set_meal" in n for n in pred_names)),
        "branch_query_missing": bool(any(n.startswith(("get_", "find_", "filter_")) for n in gt_names) and not any(n.startswith(("get_", "find_", "filter_")) for n in pred_names)),
        "branch_decision_wrong": bool(lcs < max(1, len(gt_names) // 2)),
        "mutation_missing": bool(gt_mut and not pred_mut),
        "mutation_target_wrong": bool(gt_mut and pred_mut and lcs < len(gt_mut)),
        "closure_missing": bool(any(n in AGG_TOOLS for n in gt_names) and not any(n in AGG_TOOLS for n in pred_names)),
        "evidence_pollution": False,
        "repairability": repairability,
        "assigned_resolver": assigned,
        "lcs": lcs,
        "gt_tool_count": len(gt_names),
        "v22_joint": bool(v22_score.get("joint")),
        "uses_val41_gt_for_gap_mining": True,
    }


def summarize_gaps(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_scenario: Dict[str, Dict[str, int]] = {}
    for row in rows:
        sc = row.get("scenario", "")
        by_scenario.setdefault(sc, {"total": 0, "easy": 0, "medium": 0, "hard": 0, "dirty": 0})
        by_scenario[sc]["total"] += 1
        by_scenario[sc][row.get("repairability", "hard")] = by_scenario[sc].get(row.get("repairability", "hard"), 0) + 1
    top = sorted(
        [r for r in rows if r.get("assigned_resolver") != "skip"],
        key=lambda r: ({"easy": 0, "medium": 1, "hard": 2, "dirty": 3}.get(r.get("repairability"), 9), r.get("gt_tool_count", 99)),
    )[:12]
    return {"by_scenario": by_scenario, "top12": top, "total": len(rows)}
