# -*- coding: utf-8 -*-
"""Deterministic V8 kitchen branch helper."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

from .v8_event_logger import enabled, write_v8_event

READ_ONLY = {"get_recipe_ingredients", "get_ingredient_location", "get_ingredient_quantity", "get_ingredient_nutrition", "find_ingredient_category", "find_ingredients_by_location", "get_current_menu", "get_current_shopping_list"}
BRANCH_CRITICAL = {"get_ingredient_quantity", "get_current_menu", "get_current_shopping_list", "get_recipe_ingredients"}
MUTATION = {"add_to_shopping_list", "add_recipe_to_menu", "remove_from_shopping_list", "remove_recipe_from_menu"}
FINAL = {"compute_total_nutritions", "compute_total_nutrition"}

def _key(tool_name: str, params: Dict[str, Any]) -> str:
    return json.dumps({"tool_name": tool_name, "parameters": params}, ensure_ascii=False, sort_keys=True)

def infer_kitchen_stage(state: Dict[str, Any]) -> str:
    names = [str(c.get("tool_name", "")) for c in state.get("executed_tool_calls") or []]
    if not any(n == "get_recipe_ingredients" for n in names):
        return "K0_identify_visible_or_current_recipe"
    if not any(n in BRANCH_CRITICAL for n in names):
        return "K2_intersect_current_menu_fridge_stock"
    if not any(n in MUTATION for n in names):
        return "K3_determine_missing_or_replacement_branch"
    if "nutrition" in str(state.get("user_instruction", "")).lower() and not any(n in FINAL for n in names):
        return "K5_compute_total_nutritions"
    return "K6_final_response"

def _is_broad_scan(tool_name: str, params: Dict[str, Any], state: Dict[str, Any]) -> bool:
    lname = tool_name.lower()
    if lname in {"find_ingredient_category", "get_ingredient_location"} and state.get("tool_call_count", 0) > 20:
        return True
    if lname == "get_recipe_ingredients" and len(state.setdefault("v8_kitchen_recipe_seen", [])) >= 2:
        recipe = str(params.get("recipe_name", "")).strip().lower()
        return recipe not in state.get("v8_kitchen_recipe_seen", [])
    return False

def _quantity_allowed(tool_name: str, params: Dict[str, Any], state: Dict[str, Any]) -> bool:
    if tool_name not in BRANCH_CRITICAL:
        return False
    count = state.setdefault("v8_kitchen_branch_quantity_count", 0)
    if count >= 3:
        return False
    text = " ".join([str(state.get("user_instruction", "")), json.dumps(params, ensure_ascii=False)]).lower()
    return any(w in text for w in ("recipe", "ingredient", "menu", "shopping", "fridge", "stock", "quantity", "nutrition"))

def apply_kitchen_helper(calls_obj: Any, state: Dict[str, Any], turn: int) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not enabled("TRACK2_ENABLE_KITCHEN_HELPER") or state.get("scenario") != "kitchen":
        return calls_obj, [], []
    calls = calls_obj if isinstance(calls_obj, list) else [calls_obj]
    out: List[Dict[str, Any]] = []
    synthetic: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    stage = infer_kitchen_stage(state)
    state["kitchen_v8_stage"] = stage
    write_v8_event(state, "kitchen_helper", "stage", "kitchen_stage_transition", turn=turn, kitchen_stage=stage)
    for call in calls:
        if not isinstance(call, dict):
            continue
        call2 = copy.deepcopy(call)
        name = str(call2.get("tool_name", ""))
        lname = name.lower()
        params = call2.get("parameters", {})
        if not isinstance(params, dict):
            params = {}; call2["parameters"] = params
        before = copy.deepcopy(call2)
        if name == "get_recipe_ingredients":
            recipe = str(params.get("recipe_name", "")).strip().lower()
            seen = state.setdefault("v8_kitchen_recipe_seen", [])
            if recipe and recipe in seen:
                synthetic.append({"role": "tool", "content": "Kitchen helper skipped duplicate get_recipe_ingredients; use cached recipe ingredients.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "duplicate_recipe_ingredients"})
                write_v8_event(state, "kitchen_helper", "block", "duplicate_recipe_ingredients", turn=turn, before_action=before, whether_blocked=True, risk_score=0.3)
                continue
            if recipe:
                seen.append(recipe)
        qsig = _key(name, params)
        if name in READ_ONLY:
            cache = state.setdefault("v8_kitchen_readonly_cache", [])
            if qsig in cache:
                synthetic.append({"role": "tool", "content": "Kitchen helper skipped duplicate read-only query; use prior observation.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "duplicate_readonly_query"})
                write_v8_event(state, "kitchen_helper", "block", "duplicate_readonly_query", turn=turn, before_action=before, whether_blocked=True, risk_score=0.25)
                continue
            cache.append(qsig)
        tool_count = int(state.get("tool_call_count", 0))
        if tool_count > 35 and name not in MUTATION and name not in FINAL:
            if _quantity_allowed(name, params, state):
                state["v8_kitchen_branch_quantity_count"] += 1
                decisions.append({"tool_name": name, "decision": "allow", "reason": "branch_critical_quantity"})
                write_v8_event(state, "kitchen_helper", "allow", "kitchen_branch_critical_query_allowed", turn=turn, after_action=call2, risk_score=0.2)
            else:
                synthetic.append({"role": "tool", "content": "Kitchen helper blocked broad exploration in branch-aware conservative mode. Only branch-critical quantity, pending mutations, or final nutrition compute are allowed.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "branch_aware_conservative_mode"})
                write_v8_event(state, "kitchen_helper", "block", "kitchen_broad_scan_blocked", turn=turn, before_action=before, whether_blocked=True, risk_score=0.65)
                continue
        elif _is_broad_scan(name, params, state):
            synthetic.append({"role": "tool", "content": "Kitchen helper blocked broad scan. Stay on current recipe/menu/fridge branch and query only necessary quantities.", "blocked": True, "tool_name": name})
            decisions.append({"tool_name": name, "decision": "block", "reason": "broad_scan"})
            write_v8_event(state, "kitchen_helper", "block", "kitchen_broad_scan_blocked", turn=turn, before_action=before, whether_blocked=True, risk_score=0.5)
            continue
        if name in FINAL:
            ingredients = params.get("ingredients")
            if not isinstance(ingredients, list) or not ingredients:
                synthetic.append({"role": "tool", "content": "Kitchen helper blocked nutrition compute without ingredient provenance. Retrieve recipe/current menu/list quantities first.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "missing_nutrition_provenance"})
                write_v8_event(state, "kitchen_helper", "block", "kitchen_quantity_provenance", turn=turn, before_action=before, whether_blocked=True, risk_score=0.7)
                continue
            write_v8_event(state, "kitchen_helper", "allow", "kitchen_compute_ready", turn=turn, after_action=call2)
        out.append(call2)
    return (out if isinstance(calls_obj, list) else (out[0] if out else [])), synthetic, decisions
