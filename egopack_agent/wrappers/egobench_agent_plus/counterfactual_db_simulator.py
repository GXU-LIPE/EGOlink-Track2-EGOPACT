# -*- coding: utf-8 -*-
"""Counterfactual DB simulator lite for pre-execution sanity checks."""

from __future__ import annotations

import json
import re
import string
from typing import Any, Dict, List

from .tool_affordance_memory import tool_family


def _canon(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    table = str.maketrans("", "", "".join(ch for ch in string.punctuation if ch not in "&'"))
    return text.translate(table).strip()


def _entity(params: Dict[str, Any]) -> str:
    for key in ("product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name", "category"):
        if params.get(key):
            return f"{key}:{_canon(params.get(key))}"
    return ""


def _mutation_key(tool_name: str, params: Dict[str, Any], scenario: str, state: Dict[str, Any]) -> str:
    pins = state.get("pins") or {}
    user = params.get("user_id") or params.get("customer_id") or pins.get("user_id") or ""
    rest = params.get("restaurant_name") or pins.get("restaurant_name") or ""
    return "|".join([tool_family(tool_name), scenario, _canon(user), _canon(rest), _entity(params)])


def assess_call(tool_name: str, params: Dict[str, Any], scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
    params = params if isinstance(params, dict) else {}
    family = tool_family(tool_name)
    pins = state.get("pins") or {}
    risk = 0.0
    reasons: List[str] = []
    action = "allow"
    repaired_params = None

    if scenario == "order" and family.startswith("state_changing"):
        if not (params.get("restaurant_name") or pins.get("restaurant_name")):
            risk += 0.6
            reasons.append("order_mutation_without_restaurant_pin")
            action = "require_retrieval"
    if family.startswith("state_changing"):
        key = _mutation_key(tool_name, params, scenario, state)
        for ledger_key in (state.get("successful_mutation_ledger") or {}):
            if key and key in str(ledger_key):
                risk += 0.8
                reasons.append("duplicate_state_change_counterfactual")
                action = "block"
                break
    if scenario == "order" and str(tool_name) == "remove_dish_from_order" and params.get("dish_name"):
        # Existing db_guard has the canonical set-meal rewrite. Here we only
        # expose the process risk for telemetry/verifier weights.
        name_l = str(params.get("dish_name", "")).lower()
        if any(piece in name_l for piece in ("set meal", "combo", "platter", "meal for")):
            risk += 0.35
            reasons.append("possible_set_meal_in_dish_remove")
    if scenario == "kitchen" and str(tool_name).startswith("compute_total_nutritions"):
        ingredients = params.get("ingredients")
        if not isinstance(ingredients, list) or not ingredients:
            risk += 0.7
            reasons.append("nutrition_compute_without_confirmed_ingredients")
            action = "require_retrieval"

    return {
        "action": action,
        "allow": action == "allow",
        "risk_score": round(min(1.0, risk), 3),
        "risk_reason": reasons,
        "repaired_params": repaired_params,
        "before_after_prediction": {
            "family": family,
            "entity": _entity(params),
            "ledger_size": len(state.get("successful_mutation_ledger") or {}),
        },
    }


def assess_batch(calls: Any, scenario: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    call_list = calls if isinstance(calls, list) else ([calls] if isinstance(calls, dict) else [])
    out = []
    for call in call_list:
        if not isinstance(call, dict):
            continue
        out.append({
            "tool_name": call.get("tool_name"),
            "decision": assess_call(str(call.get("tool_name", "")), call.get("parameters", {}), scenario, state),
        })
    return out
