# -*- coding: utf-8 -*-
"""Deterministic V8 order process helper.

This module enforces process shape and type safety. It does not encode dev/final
answers and does not read final scenario JSON.
"""
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Tuple

from .db_guard import canonical_text, is_final_compute_tool, is_state_changing_tool
from .v8_event_logger import enabled, write_v8_event

SET_MEAL_HINTS = ("set meal", "combo", "platter", "meal for", "bundle")
AGG_TOOLS = {"compute_total_tax", "compute_total_payment", "compute_total_price", "compute_total_nutrition"}
VISUAL_QUESTION_RE = re.compile(r"\b(what|which).{0,40}(see|shown|visible|menu|dish|category|picture|video)\b", re.I)

class OrderReplaceStateMachine:
    stages = [
        "O0_pin_restaurant",
        "O1_inspect_current_order_or_menu",
        "O2_identify_target_dish_or_set_meal",
        "O3_add_target_dish_if_needed",
        "O4_remove_old_dish_or_set_meal_if_needed",
        "O5_compute_tax_or_payment",
        "O6_final_response",
    ]

def _calls(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(state.get("executed_tool_calls") or [])

def _names(state: Dict[str, Any]) -> List[str]:
    return [str(c.get("tool_name", "")) for c in _calls(state)]

def _intent(text: str) -> bool:
    t = str(text or "").lower()
    return any(w in t for w in ("replace", "remove", "change", "swap", "cancel", "set meal", "dish", "tax", "payment", "order"))

def infer_order_stage(state: Dict[str, Any]) -> str:
    pins = state.get("pins") or {}
    names = _names(state)
    if not pins.get("restaurant_name"):
        return "O0_pin_restaurant"
    if not any("order" in n.lower() or "dish" in n.lower() or "set_meal" in n.lower() or "set meal" in n.lower() for n in names):
        return "O1_inspect_current_order_or_menu"
    has_add = any(n.lower().startswith("add") and "order" in n.lower() for n in names)
    has_remove = any((n.lower().startswith("remove") or "from_order" in n.lower()) for n in names)
    if not (has_add or has_remove):
        return "O2_identify_target_dish_or_set_meal"
    if has_add and not has_remove and any(w in str(state.get("user_instruction", "")).lower() for w in ("replace", "remove", "swap", "change")):
        return "O4_remove_old_dish_or_set_meal_if_needed"
    if not any(n in AGG_TOOLS or n.startswith("compute_total") for n in names):
        return "O5_compute_tax_or_payment"
    return "O6_final_response"

def _normalize_aggregate_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    out = copy.deepcopy(params)
    changed = False
    dishes = out.get("dishes")
    if isinstance(dishes, list):
        for item in dishes:
            if not isinstance(item, dict):
                continue
            # Order aggregate tools use dishes[].product_name even though the
            # items are menu dishes. Converting to dish_name makes the order DB
            # aggregate return 0.0, which then triggers expensive retry loops.
            if "dish_name" in item and "product_name" not in item:
                item["product_name"] = item.pop("dish_name")
                changed = True
            elif "dish_name" in item and "product_name" in item:
                item.pop("dish_name", None)
                changed = True
    return out, changed


def _canonical_aggregate_fp(tool_name: str, params: Dict[str, Any]) -> str:
    fp_params = copy.deepcopy(params)
    dishes = fp_params.get("dishes")
    if isinstance(dishes, list):
        norm_items = []
        for item in dishes:
            if isinstance(item, dict):
                name = item.get("product_name", item.get("dish_name", ""))
                qty = item.get("quantity", item.get("qty", 1))
                norm_items.append({
                    "product_name": canonical_text(name),
                    "quantity": qty,
                })
        fp_params["dishes"] = sorted(norm_items, key=lambda x: (x.get("product_name", ""), str(x.get("quantity", ""))))
    return json.dumps({"tool_name": tool_name, "parameters": fp_params}, ensure_ascii=False, sort_keys=True)

def _looks_set_meal(value: Any) -> bool:
    t = canonical_text(value)
    return any(h.replace(" ", "") in t.replace(" ", "") for h in SET_MEAL_HINTS)

def apply_order_helper(calls_obj: Any, state: Dict[str, Any], turn: int) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not enabled("TRACK2_ENABLE_ORDER_HELPER") or state.get("scenario") != "order":
        return calls_obj, [], []
    calls = calls_obj if isinstance(calls_obj, list) else [calls_obj]
    out: List[Dict[str, Any]] = []
    synthetic: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    stage = infer_order_stage(state)
    state["order_v8_stage"] = stage
    write_v8_event(state, "order_helper", "stage", "order_stage_transition", turn=turn, risk_score=0.0, order_stage=stage)
    pins = state.get("pins") or {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        call2 = copy.deepcopy(call)
        name = str(call2.get("tool_name", ""))
        params = call2.get("parameters", {})
        if not isinstance(params, dict):
            params = {}; call2["parameters"] = params
        lname = name.lower()
        before = copy.deepcopy(call2)
        if (is_state_changing_tool(name, "order") or is_final_compute_tool(name)) and pins.get("restaurant_name") and not params.get("restaurant_name"):
            params["restaurant_name"] = pins["restaurant_name"]
            decisions.append({"tool_name": name, "decision": "autofill_restaurant"})
            write_v8_event(state, "order_helper", "repair", "restaurant_pin_autofill", turn=turn, before_action=before, after_action=call2, whether_repaired=True)
        if (is_state_changing_tool(name, "order") or is_final_compute_tool(name)) and not (params.get("restaurant_name") or pins.get("restaurant_name")):
            synthetic.append({"role": "tool", "content": "Order helper blocked mutation/aggregate: pin restaurant_name via retrieval or dialogue before modifying/computing order.", "blocked": True, "tool_name": name})
            decisions.append({"tool_name": name, "decision": "block", "reason": "missing_restaurant_pin"})
            write_v8_event(state, "order_helper", "block", "missing_restaurant_pin", turn=turn, before_action=before, whether_blocked=True, risk_score=0.7)
            continue
        if name == "remove_dish_from_order" and (_looks_set_meal(params.get("dish_name")) or params.get("set_meal_name")):
            meal = params.get("set_meal_name") or params.get("dish_name")
            params.pop("dish_name", None); params["set_meal_name"] = meal; call2["tool_name"] = "remove_set_meal_from_order"; name = call2["tool_name"]
            decisions.append({"tool_name": "remove_dish_from_order", "decision": "rewrite", "to": name})
            write_v8_event(state, "order_helper", "repair", "order_setmeal_dish_type_rewrite", turn=turn, before_action=before, after_action=call2, whether_repaired=True, risk_score=0.35)
        if name in AGG_TOOLS or is_final_compute_tool(name):
            params2, changed = _normalize_aggregate_params(params)
            if changed:
                call2["parameters"] = params2; params = params2
                decisions.append({"tool_name": name, "decision": "rewrite", "reason": "order_aggregate_uses_product_name"})
                write_v8_event(state, "order_helper", "repair", "aggregate_dishes_use_product_name", turn=turn, before_action=before, after_action=call2, whether_repaired=True)
            fp = _canonical_aggregate_fp(name, params)
            if fp in state.setdefault("v8_order_compute_ledger", {}):
                synthetic.append({"role": "tool", "content": "Order aggregate loop blocked: the same aggregate parameters were already computed. Do not retry this aggregate. Use item-level retrieval to resolve a zero/ambiguous result, perform the required mutation, or finish if the process is complete.", "blocked": True, "tool_name": name})
                decisions.append({"tool_name": name, "decision": "block", "reason": "aggregate_loop"})
                write_v8_event(state, "order_helper", "block", "order_aggregate_loop_blocked", turn=turn, before_action=before, whether_blocked=True, risk_score=0.6)
                continue
            state["v8_order_compute_ledger"][fp] = {"turn": turn, "tool_name": name}
            write_v8_event(state, "order_helper", "allow", "aggregate_tool_selected", turn=turn, after_action=call2)
        out.append(call2)
    return (out if isinstance(calls_obj, list) else (out[0] if out else [])), synthetic, decisions

def inspect_natural_reply(reply: str, state: Dict[str, Any], turn: int) -> Dict[str, Any]:
    if state.get("scenario") != "order" or not enabled("TRACK2_ENABLE_ORDER_HELPER"):
        return {"allow": True}
    no_visual = not state.get("contact_sheet_path") and os.environ.get("TRACK2_FINAL_EVAL", "0") == "1"
    if no_visual and VISUAL_QUESTION_RE.search(str(reply or "")):
        write_v8_event(state, "order_helper", "block", "order_no_visual_followup_blocked", turn=turn, before_action=reply, whether_blocked=True, risk_score=0.5)
        return {"allow": False, "replacement": "I will verify the order using the available restaurant/order tools instead of asking for visual details."}
    if _intent(state.get("user_instruction", "")) and infer_order_stage(state) != "O6_final_response":
        write_v8_event(state, "order_helper", "continue", "order_missing_stage_redirect", turn=turn, before_action=reply, order_stage=infer_order_stage(state), risk_score=0.4)
        return {"allow": False, "replacement": "I need to complete the required order tool process before finalizing."}
    return {"allow": True}
