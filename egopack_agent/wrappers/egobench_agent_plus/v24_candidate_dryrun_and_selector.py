#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-oracle dry-run selector for V24 target candidates."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


AGG_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
}
MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def required_closure(instruction: str, scenario: str) -> str:
    text = norm_text(instruction)
    if "total tax" in text:
        return "compute_total_tax"
    if any(x in text for x in ("total payment", "amount payable", "payable", "total cost")):
        return "compute_total_payment"
    if any(x in text for x in ("total nutrition", "total nutritional", "total calcium")):
        return "compute_total_nutritions" if scenario == "kitchen" else "compute_total_nutrition"
    if "total taste" in text:
        return "tally_total_tastes"
    if "nutritional characteristic" in text and "total" in text:
        return "tally_total_nutritional_characteristics"
    if "summary" in text and scenario in {"order", "restaurant"}:
        return "get_user_order_summary"
    return ""


def mutation_intent(instruction: str) -> bool:
    text = norm_text(instruction)
    return any(x in text for x in ("add", "remove", "update", "delete", "cart", "order", "shopping list", "menu"))


def dryrun_program(scenario: str, db: Any, program: List[Dict[str, Any]], instruction: str) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    mutation = 0
    aggregate = 0
    retrieval_nonempty = 0
    branch_observation = 0
    for i, step in enumerate(program):
        tool = str(step.get("tool_name") or "")
        params = step.get("parameters") or {}
        if not tool:
            errors.append({"idx": i, "reason": "missing_tool"})
            continue
        if not hasattr(db, tool):
            warnings.append({"idx": i, "tool": tool, "reason": "local_db_missing"})
            continue
        try:
            if tool.startswith(("get_", "find_", "list_", "filter_")):
                res = getattr(db, tool)(**params)
                if isinstance(res, dict):
                    if res.get("status") == "error":
                        warnings.append({"idx": i, "tool": tool, "reason": "retrieval_error", "message": res.get("message")})
                    elif any(res.get(k) for k in ("products", "product_names", "dishes", "dish_names", "recipes", "recipe_names", "ingredients", "ingredient_names", "set_meals", "items")):
                        retrieval_nonempty += 1
                    else:
                        # Single attribute getters returning value still count as
                        # branch-observation evidence.
                        if len(res) > 0:
                            branch_observation += 1
                else:
                    branch_observation += 1
            elif MUTATION_RE.search(tool):
                mutation += 1
            elif tool in AGG_TOOLS:
                aggregate += 1
        except Exception as exc:
            warnings.append({"idx": i, "tool": tool, "reason": "dryrun_exception", "message": str(exc)})
    names = [str(x.get("tool_name") or "") for x in program]
    closure = required_closure(instruction, scenario)
    broad_scan = any(n.startswith(("find_products_by_price_range", "get_all_", "list_all_")) for n in names[:2])
    missing_closure = bool(closure and closure not in names and not any(n in AGG_TOOLS for n in names))
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "mutation_count": mutation,
        "aggregate_count": aggregate,
        "retrieval_nonempty_count": retrieval_nonempty,
        "branch_observation_count": branch_observation,
        "closure_required": closure,
        "closure_complete": not missing_closure,
        "broad_scan": broad_scan,
        "tool_count": len(program),
    }


def select_v24_candidate(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    instruction = context.get("instruction", "")
    scenario = context.get("scenario", "")
    scored = []
    for cand in candidates:
        prog = cand.get("tool_program") or []
        names = [str(x.get("tool_name") or "") for x in prog]
        dry = cand.get("dryrun") or {}
        hard = []
        if not names:
            hard.append("empty_program")
        if dry.get("errors"):
            hard.append("dryrun_errors")
        if mutation_intent(instruction) and scenario in {"retail", "restaurant", "order"} and not any(MUTATION_RE.search(n) for n in names):
            hard.append("mutation_intent_no_mutation")
        if dry.get("closure_required") and not dry.get("closure_complete"):
            hard.append("missing_closure")
        if dry.get("broad_scan"):
            hard.append("leading_broad_scan")

        score = float(cand.get("shape_confidence") or 0.0)
        reasons = []
        if cand.get("source") == "V22_PROTECTED_BASE":
            score += 10
            reasons.append("protected_success_floor")
        if dry.get("ok"):
            score += 3
            reasons.append("dryrun_ok")
        if dry.get("retrieval_nonempty_count") or dry.get("branch_observation_count"):
            score += min(3, dry.get("retrieval_nonempty_count", 0) + dry.get("branch_observation_count", 0) * 0.5)
            reasons.append("observation_before_action")
        if dry.get("mutation_count"):
            score += 2
            reasons.append("mutation_present")
        if dry.get("closure_complete"):
            score += 2
            reasons.append("closure_complete")
        if names and names[0].startswith(("get_", "find_", "filter_")):
            score += 0.8
            reasons.append("retrieval_first")
        if not dry.get("broad_scan"):
            score += 0.8
            reasons.append("no_broad_scan")
        if "V24_" in str(cand.get("candidate_id")):
            score += 0.4
            reasons.append("scenario_generator")
        retrieve_count = sum(1 for n in names if n.startswith(("get_", "find_", "filter_", "list_")))
        if dry.get("mutation_count") and dry.get("closure_complete") and dry.get("tool_count", 0) <= 4:
            score += 2.2
            reasons.append("minimal_mutation_closure")
        if dry.get("mutation_count") and dry.get("closure_complete") and retrieve_count <= 1:
            score += 1.4
            reasons.append("low_retrieval_closed_program")
        if retrieve_count >= 4 and dry.get("mutation_count") and dry.get("closure_complete"):
            score -= 1.8
            reasons.append("penalize_retrieval_overexpanded_closed_program")
        score -= max(0, len(names) - 12) * 0.12
        if hard:
            score -= 25
        out = dict(cand)
        out["selector_score"] = score
        out["selector_reasons"] = reasons
        out["hard_filters"] = hard
        scored.append(out)
    scored.sort(key=lambda x: (x.get("selector_score", -999), -len(x.get("tool_program") or [])), reverse=True)
    return scored[0] if scored else {"candidate_id": "empty", "source": "empty", "tool_program": [], "selector_score": -999, "hard_filters": ["no_candidates"]}
