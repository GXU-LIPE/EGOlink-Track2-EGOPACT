# -*- coding: utf-8 -*-
"""Human process graphs for Track2 scenarios.

The graph is a compact process-shape prior. It guides coverage of stages, not
task-specific answers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .tool_affordance_memory import describe_allowed_families, tool_family


GRAPH_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "retail": [
        {"stage": "identify_product_or_visible_candidate", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "retrieve_product_attributes", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "compare_constraints", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": False},
        {"stage": "apply_cart_or_list_mutation", "allowed_families": ["state_changing_add", "state_changing_remove"], "risky_if_skipped": True},
        {"stage": "compute_total_or_nutrition", "allowed_families": ["aggregate_compute"], "risky_if_skipped": False},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
    "kitchen": [
        {"stage": "identify_current_recipe_or_visible_ingredients", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "get_recipe_ingredients_once", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "determine_branch_from_menu_fridge_recipe", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "apply_menu_or_shopping_list_mutation", "allowed_families": ["state_changing_add", "state_changing_remove"], "risky_if_skipped": True},
        {"stage": "compute_total_nutritions", "allowed_families": ["aggregate_compute"], "risky_if_skipped": False},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
    "restaurant": [
        {"stage": "identify_dish_or_set_meal", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "retrieve_menu_attributes", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "apply_order_mutation", "allowed_families": ["state_changing_add", "state_changing_remove"], "risky_if_skipped": False},
        {"stage": "compute_total_nutrition_or_payment", "allowed_families": ["aggregate_compute"], "risky_if_skipped": False},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
    "order": [
        {"stage": "pin_restaurant", "allowed_families": ["read_only_retrieval"], "prerequisite_slots": ["restaurant_name"], "risky_if_skipped": True},
        {"stage": "inspect_current_order_or_menu_context", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "identify_target_dish_or_set_meal", "allowed_families": ["read_only_retrieval"], "risky_if_skipped": True},
        {"stage": "add_new_dish_if_needed", "allowed_families": ["state_changing_add"], "risky_if_skipped": False},
        {"stage": "remove_old_dish_or_set_meal_if_needed", "allowed_families": ["state_changing_remove"], "risky_if_skipped": False},
        {"stage": "compute_tax_or_payment", "allowed_families": ["aggregate_compute"], "risky_if_skipped": True},
        {"stage": "final_response", "allowed_families": [], "risky_if_skipped": False},
    ],
}


def _calls(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(state.get("executed_tool_calls") or [])


def _families(calls: List[Dict[str, Any]]) -> List[str]:
    return [tool_family(str(c.get("tool_name", ""))) for c in calls if isinstance(c, dict)]


def _tool_names(calls: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in calls if isinstance(c, dict)]


def _text_needs_mutation(text: str) -> bool:
    t = str(text or "").lower()
    return bool(re.search(r"\b(add|remove|replace|delete|order|cart|shopping list|menu|include|exclude)\b", t))


def _text_needs_aggregate(text: str) -> bool:
    t = str(text or "").lower()
    if re.search(r"\b(total|sum|aggregate|overall)\b", t):
        return True
    if re.search(r"\b(tax|payment|payable|amount due|checkout|bill)\b", t):
        return True
    if re.search(r"\b(total|sum|aggregate|overall)\s+(nutrition|nutritions|calorie|calories|protein|fat|carb|sodium)", t):
        return True
    return False


def infer_process_state(scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
    scenario = str(scenario or state.get("scenario") or "")
    calls = _calls(state)
    fams = _families(calls)
    names = _tool_names(calls)
    pins = state.get("pins") or {}
    instruction = str(state.get("user_instruction") or "")
    stage = "final_response"
    missing: List[str] = []
    coverage: Dict[str, bool] = {
        "has_retrieval": "read_only_retrieval" in fams,
        "has_add": "state_changing_add" in fams,
        "has_remove": "state_changing_remove" in fams,
        "has_update": "state_changing_update" in fams,
        "has_aggregate": "aggregate_compute" in fams,
    }
    needs_aggregate = _text_needs_aggregate(instruction)
    if scenario == "order":
        if state.get("order_requested_final_aggregate"):
            needs_aggregate = True
        if not pins.get("restaurant_name"):
            stage = "pin_restaurant"
            missing.append("restaurant_name")
        elif not coverage["has_retrieval"]:
            stage = "inspect_current_order_or_menu_context"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "identify_target_dish_or_set_meal"
        elif coverage["has_add"] and not coverage["has_remove"] and any(w in instruction.lower() for w in ("remove", "replace")):
            stage = "remove_old_dish_or_set_meal_if_needed"
        elif (coverage["has_add"] or coverage["has_remove"] or state.get("order_requested_final_aggregate")) and needs_aggregate and not coverage["has_aggregate"]:
            stage = "compute_tax_or_payment"
            missing.append("final_aggregate")
        else:
            stage = "final_response"
    elif scenario == "kitchen":
        if not coverage["has_retrieval"]:
            stage = "identify_current_recipe_or_visible_ingredients"
        elif not any("get_recipe_ingredients" in n for n in names):
            stage = "get_recipe_ingredients_once"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "determine_branch_from_menu_fridge_recipe"
        elif (coverage["has_add"] or coverage["has_remove"]) and not coverage["has_aggregate"]:
            stage = "compute_total_nutritions"
            missing.append("final_nutrition_compute")
        else:
            stage = "final_response"
        if state.get("tool_call_count", 0) > 25:
            missing.append("STOP_EXPLORING")
    elif scenario == "retail":
        if not coverage["has_retrieval"]:
            stage = "identify_product_or_visible_candidate"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "apply_cart_or_list_mutation"
        elif (coverage["has_add"] or coverage["has_remove"]) and needs_aggregate and not coverage["has_aggregate"]:
            stage = "compute_total_or_nutrition"
        else:
            stage = "final_response"
    elif scenario == "restaurant":
        if not coverage["has_retrieval"]:
            stage = "identify_dish_or_set_meal"
        elif _text_needs_mutation(instruction) and not (coverage["has_add"] or coverage["has_remove"]):
            stage = "apply_order_mutation"
        else:
            stage = "final_response"
    allowed = describe_allowed_families(scenario, stage)
    return {
        "scenario": scenario,
        "current_stage": stage,
        "missing_prerequisites": missing,
        "allowed_tool_families": allowed.get("allowed_families", []),
        "allowed_tool_set": allowed.get("candidate_tools", []),
        "expected_next_tool_family": (allowed.get("allowed_families") or ["message"])[0],
        "process_coverage_state": coverage,
        "template_nodes": [n["stage"] for n in GRAPH_TEMPLATES.get(scenario, [])],
    }


def prompt_snippet(scenario: str, state: Dict[str, Any]) -> str:
    ps = infer_process_state(scenario, state)
    pins = state.get("pins") or {}
    lines = [
        "Human-prior process state:",
        f"- scenario: {scenario}",
        f"- current_stage: {ps['current_stage']}",
        f"- pinned_user_id: {pins.get('user_id') or ''}",
        f"- pinned_restaurant_name: {pins.get('restaurant_name') or ''}",
        f"- allowed_next_families: {', '.join(ps['allowed_tool_families']) or 'short_message'}",
        f"- candidate_tools_cap: {', '.join(ps['allowed_tool_set'][:5])}",
    ]
    if ps["missing_prerequisites"]:
        lines.append(f"- missing_prerequisites: {', '.join(ps['missing_prerequisites'])}")
    return "\n".join(lines)
