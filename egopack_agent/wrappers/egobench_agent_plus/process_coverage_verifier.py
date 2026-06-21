# -*- coding: utf-8 -*-
"""Process coverage verifier for Track2 V7.

This verifier checks shape-level process coverage only. It does not read GT
answers and does not encode task-specific final parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .tool_affordance_memory import tool_family
from .human_process_graph import infer_process_state


AGG_TOOLS = ("compute_total_tax", "compute_total_payment", "compute_total_price", "compute_total_nutrition", "compute_total_nutritions")


def _calls_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(state.get("executed_tool_calls") or [])


def _names(calls: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in calls if isinstance(c, dict)]


def _families(calls: List[Dict[str, Any]]) -> List[str]:
    return [tool_family(str(c.get("tool_name", ""))) for c in calls if isinstance(c, dict)]


def verify_process_coverage(
    scenario: str,
    candidate_calls: Any = None,
    state: Dict[str, Any] | None = None,
    natural_reply: bool = False,
) -> Dict[str, Any]:
    state = state or {}
    prior = _calls_from_state(state)
    cand = candidate_calls if isinstance(candidate_calls, list) else ([candidate_calls] if isinstance(candidate_calls, dict) else [])
    all_calls = prior + cand
    names = _names(all_calls)
    fams = _families(all_calls)
    ps = infer_process_state(scenario, state)
    score = 0.0
    missing: List[str] = []
    mismatch: List[str] = []
    should_continue = False
    should_stop = False
    suggested = ps.get("expected_next_tool_family", "read_only_retrieval")

    if "read_only_retrieval" in fams:
        score += 0.25
    if any(f.startswith("state_changing") for f in fams):
        score += 0.25
    if "aggregate_compute" in fams:
        score += 0.25
    if ps.get("current_stage") == "final_response":
        score += 0.25

    if scenario == "order":
        pins = state.get("pins") or {}
        if not pins.get("restaurant_name"):
            missing.append("pin_restaurant")
            suggested = "read_only_retrieval"
        has_mutation = any(f in {"state_changing_add", "state_changing_remove"} for f in fams)
        has_aggregate = any(str(n).startswith(AGG_TOOLS) for n in names)
        requested_agg = bool(state.get("order_requested_final_aggregate"))
        if (has_mutation or requested_agg) and not has_aggregate:
            missing.append("final_aggregate_after_order_process")
            suggested = "aggregate_compute"
            should_continue = True
        compute_payment_count = sum(1 for n in names if n == "compute_total_payment")
        if compute_payment_count >= 2 and "state_changing_remove" not in fams and any(w in str(state.get("user_instruction", "")).lower() for w in ("remove", "replace")):
            mismatch.append("repeated_payment_loop_before_remove")
            suggested = "state_changing_remove"
            should_continue = True
        for c in cand:
            if not isinstance(c, dict):
                continue
            name = str(c.get("tool_name", ""))
            params = c.get("parameters", {}) if isinstance(c.get("parameters", {}), dict) else {}
            if name == "remove_dish_from_order" and params.get("set_meal_name"):
                mismatch.append("set_meal_sent_to_dish_remove_tool")
            if name == "remove_set_meal_from_order" and params.get("dish_name") and not params.get("set_meal_name"):
                mismatch.append("dish_name_sent_to_set_meal_remove_tool")
    elif scenario == "kitchen":
        has_recipe = any("get_recipe_ingredients" in n for n in names)
        has_mutation = any(f in {"state_changing_add", "state_changing_remove"} for f in fams)
        has_compute = any(n.startswith("compute_total_nutritions") for n in names)
        if not has_recipe and any(w in str(state.get("user_instruction", "")).lower() for w in ("recipe", "menu", "shopping", "nutrition")):
            missing.append("recipe_ingredient_retrieval")
            suggested = "read_only_retrieval"
        if has_mutation and not has_compute and "nutrition" in str(state.get("user_instruction", "")).lower():
            missing.append("compute_total_nutritions")
            suggested = "aggregate_compute"
            should_continue = True
        if state.get("tool_call_count", 0) > 35:
            mismatch.append("kitchen_conservative_mode_active")
    elif scenario == "retail":
        if state.get("blocked_calls"):
            mismatch.append("duplicate_or_risky_mutation_seen")
    elif scenario == "restaurant":
        should_stop = natural_reply and not missing

    if not missing and not mismatch and ps.get("current_stage") == "final_response":
        should_stop = True
    return {
        "process_coverage_score": min(1.0, score),
        "missing_process_stage": missing,
        "suggested_next_tool_family": suggested,
        "should_continue": should_continue,
        "should_stop": should_stop,
        "should_retry": bool(mismatch),
        "tool_family_mismatch": mismatch,
        "process_state": ps,
    }
