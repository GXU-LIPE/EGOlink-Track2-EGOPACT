# -*- coding: utf-8 -*-
"""Episode-level safety checks for EgoBench tool calls.

The guard is intentionally deterministic. It does not try to solve the task; it
only enforces constraints that should hold for every service-agent backend.
"""

from __future__ import annotations

import json
import os
import re
import string
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema_loader import get_scenario_schema
from .canonical_resolver import build_canonical_cache
try:
    from .order_process_state_helper import apply_order_helper
except Exception:
    def apply_order_helper(calls_obj, state, turn):
        return calls_obj, [], []
try:
    from .kitchen_branch_helper import apply_kitchen_helper
except Exception:
    def apply_kitchen_helper(calls_obj, state, turn):
        return calls_obj, [], []
try:
    from .retail_candidate_narrower import apply_retail_narrower
except Exception:
    def apply_retail_narrower(calls_obj, state, turn):
        return calls_obj, [], []
try:
    from .v8_event_logger import write_v8_event
except Exception:
    def write_v8_event(*args, **kwargs):
        return None


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
USER_ID_VALUE_RE = re.compile(r"\b(?:user|customer|cook)[_-]?\d+\b|\bu\d+\b", re.I)
EXPLICIT_USER_ID_RE = re.compile(
    r"\b(?:user_id|customer_id|user\s*id|customer\s*id)\s*(?:is|=|:)?\s*['\"]?"
    r"((?:user|customer|cook)[_-]?\d+|u\d+)\b",
    re.I,
)
IGNORED_USER_ID_TOKENS = {"user_id", "userid", "user-id", "customer_id", "customerid", "customer-id"}
RESTAURANT_RE = re.compile(r"\b([A-Z][A-Za-z&' -]+ Restaurant)\b")
KNOWN_RESTAURANTS = (
    "Annie Italian Restaurant",
    "Mediterranean Greek Restaurant",
    "Afrikana Restaurant",
    "Butcher Restaurant",
    "Meraki Restaurant",
    "Pauhana Restaurant",
    "Sunny Restaurant",
)

MUTATION_MARKERS = (
    "add",
    "remove",
    "delete",
    "update",
    "modify",
)
MUTATION_SUBSTRINGS = (
    "_to_cart",
    "_from_cart",
    "_to_order",
    "_from_order",
    "_to_shopping_list",
    "_to_menu",
)
FINAL_COMPUTE_PREFIXES = (
    "compute_total_nutrition",
    "compute_total_nutritions",
    "compute_total_tax",
    "compute_total_price",
    "compute_total_payment",
)
ORDER_RESTAURANT_TOOLS = (
    "dish",
    "set_meal",
    "set meal",
    "order",
    "tax",
    "payment",
    "nutrition",
    "price",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def canonical_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    table = str.maketrans("", "", "".join(ch for ch in string.punctuation if ch not in "&"))
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 4 and text.endswith("s") and not text.endswith("ss"):
        text = text[:-1]
    return text


def _first_param(params: Dict[str, Any], keys: Iterable[str]) -> Tuple[str, Any]:
    for key in keys:
        if key in params and params[key] not in (None, ""):
            return key, params[key]
    return "", None


def action_family(tool_name: str) -> str:
    name = tool_name.lower()
    if name.startswith("remove") or "_from_" in name or name.startswith("delete"):
        return "remove" if not name.startswith("delete") else "delete"
    if name.startswith("update") or name.startswith("modify"):
        return "update"
    if name.startswith("add") or "_to_" in name:
        return "add"
    if name.startswith("delete"):
        return "delete"
    return "other"


def is_final_compute_tool(tool_name: str) -> bool:
    return tool_name.lower().startswith(FINAL_COMPUTE_PREFIXES)


def is_state_changing_tool(tool_name: str, scenario: str = "") -> bool:
    name = tool_name.lower()
    if is_final_compute_tool(name):
        return False
    if name.startswith(MUTATION_MARKERS):
        return True
    return any(marker in name for marker in MUTATION_SUBSTRINGS)


def entity_from_params(tool_name: str, params: Dict[str, Any]) -> Tuple[str, str, str]:
    candidates = [
        ("product", ("product_name", "product", "item_name", "item")),
        ("dish", ("dish_name", "dish")),
        ("set_meal", ("set_meal_name", "set_name", "set_meal")),
        ("recipe", ("recipe_name", "recipe")),
        ("ingredient", ("ingredient_name", "ingredient")),
        ("category", ("category", "category_name")),
    ]
    for entity_type, keys in candidates:
        key, value = _first_param(params, keys)
        if value is not None:
            return entity_type, canonical_text(value), str(value)
    return "unknown", canonical_text(tool_name), ""


def normalized_quantity(params: Dict[str, Any]) -> str:
    _, value = _first_param(params, ("qty", "quantity", "count", "amount"))
    if value is None:
        return ""
    try:
        fval = float(value)
        if fval.is_integer():
            return str(int(fval))
        return str(fval)
    except Exception:
        return canonical_text(value)


def extract_user_ids_from_text(text: str) -> List[str]:
    out: List[str] = []
    text_s = str(text or "")
    for match in EXPLICIT_USER_ID_RE.findall(text_s):
        cleaned = str(match).strip()
        if cleaned and cleaned.lower() not in IGNORED_USER_ID_TOKENS and cleaned not in out:
            out.append(cleaned)
    for match in USER_ID_VALUE_RE.findall(text_s):
        cleaned = str(match).strip()
        if cleaned and cleaned.lower() not in IGNORED_USER_ID_TOKENS and cleaned not in out:
            out.append(cleaned)
    return out


def extract_restaurants_from_text(text: str) -> List[str]:
    out: List[str] = []
    text_s = str(text or "")
    text_c = canonical_text(text_s)
    for restaurant in KNOWN_RESTAURANTS:
        if canonical_text(restaurant) in text_c and restaurant not in out:
            out.append(restaurant)
    if out:
        return out
    for match in RESTAURANT_RE.findall(text_s):
        cleaned = re.sub(r"\s+", " ", match).strip()
        # Avoid greedy captures such as "All tool calls ... Annie Italian Restaurant".
        for restaurant in KNOWN_RESTAURANTS:
            if canonical_text(restaurant) in canonical_text(cleaned):
                cleaned = restaurant
                break
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _history_text(history: List[Dict[str, Any]]) -> str:
    return "\n".join(str(msg.get("content", "")) for msg in history or [])


def make_episode_state(
    scenario: str,
    task_id: int,
    run_id: str,
    version: str,
    user_instruction: str = "",
    image_description: str = "",
    output_model_name: str = "",
) -> Dict[str, Any]:
    state = {
        "scenario": scenario,
        "task_id": task_id,
        "run_id": run_id,
        "version": version,
        "output_model_name": output_model_name,
        "pins": {
            "user_id": None,
            "restaurant_name": None,
        },
        "pin_sources": {},
        "successful_mutation_ledger": {},
        "compute_call_ledger": {},
        "blocked_calls": [],
        "tool_call_count": 0,
        "kitchen_stage": "K0_identify_current_recipe_or_visible_ingredients",
        "kitchen_seen_recipe_ingredients": [],
        "kitchen_seen_queries": [],
        "user_instruction": user_instruction,
        "order_requested_final_aggregate": _detect_order_requested_final_aggregate(user_instruction),
        "order_current_items": [],
        "external_api_used": False,
        "executed_tool_calls": [],
        "created_at": _now(),
    }
    update_pins_from_text(state, user_instruction, "task_instruction")
    update_pins_from_text(state, image_description, "image_description")
    return state


def _detect_order_requested_final_aggregate(text: str) -> str:
    text_c = canonical_text(text)
    if any(piece in text_c for piece in ("total tax", "sum up the total tax", "tax amount", "tax generated")):
        return "tax"
    if any(piece in text_c for piece in ("total payment", "amount payable", "total amount payable", "after discount")):
        return "payment"
    if any(piece in text_c for piece in ("total price", "total cost", "total payable")):
        return "payment"
    if any(piece in text_c for piece in ("total calorie", "total nutrition", "carbohydrate", "protein", "sodium", "fat", "fiber", "sugar")):
        return "nutrition"
    return ""


def wrapper_event_path(state: Dict[str, Any]) -> Path:
    version = state.get("version") or "V2_5"
    run_id = state.get("run_id") or time.strftime("manual_%Y%m%d_%H%M%S")
    task_id = state.get("task_id", "unknown")
    path = CODEX_ROOT / "runs" / str(version) / str(run_id) / "wrapper_events" / f"{task_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_wrapper_event(state: Optional[Dict[str, Any]], event: Dict[str, Any]) -> None:
    if state is None:
        return
    record = {
        "time": _now(),
        "task_id": state.get("task_id"),
        "scenario": state.get("scenario"),
        **event,
    }
    try:
        with open(wrapper_event_path(state), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _set_pin(state: Dict[str, Any], pin_type: str, value: Any, source: str) -> None:
    if not value:
        return
    value_s = str(value).strip()
    if not value_s:
        return
    current = state.setdefault("pins", {}).get(pin_type)
    if current:
        if pin_type == "restaurant_name" and state.get("scenario") == "order":
            # In order tasks, the early restaurant-comparison phase may query
            # both menus. A later user utterance such as "use Annie from now on"
            # is the true pin and must be allowed to replace exploratory params.
            source_s = str(source or "")
            if source_s in {"task_instruction", "dialogue", "image_description"}:
                current_c = canonical_text(current)
                value_c = canonical_text(value_s)
                text_c = canonical_text(value_s)
                if current_c != value_c:
                    state["pins"][pin_type] = value_s
                    state.setdefault("pin_sources", {})[pin_type] = source
                    append_wrapper_event(state, {
                        "event": "pin_override",
                        "pin_type": pin_type,
                        "old_value": current,
                        "value": value_s,
                        "source": source,
                    })
        return
    state["pins"][pin_type] = value_s
    state.setdefault("pin_sources", {})[pin_type] = source
    append_wrapper_event(state, {"event": "pin_set", "pin_type": pin_type, "value": value_s, "source": source})


def update_pins_from_text(state: Dict[str, Any], text: str, source: str) -> None:
    for uid in extract_user_ids_from_text(text):
        _set_pin(state, "user_id", uid, source)
        break
    restaurants = extract_restaurants_from_text(text)
    if state.get("scenario") == "order" and len(restaurants) > 1:
        append_wrapper_event(state, {
            "event": "pin_deferred",
            "pin_type": "restaurant_name",
            "source": source,
            "candidates": restaurants,
            "reason": "multiple_restaurants_in_order_comparison",
        })
        restaurants = []
    for restaurant in restaurants:
        _set_pin(state, "restaurant_name", restaurant, source)
        break


def update_pins_from_call_or_result(
    state: Dict[str, Any],
    call: Dict[str, Any],
    result: Any = None,
    source: str = "tool",
) -> None:
    params = call.get("parameters", {}) if isinstance(call, dict) else {}
    if isinstance(params, dict):
        _set_pin(state, "user_id", params.get("user_id") or params.get("customer_id"), f"{source}_params")
        # Order tasks often compare two restaurants with retrieval calls before
        # the user chooses one. Do not pin restaurant_name from exploratory
        # order retrieval params; let user instruction/dialogue or mutations pin.
        if not (
            state.get("scenario") == "order"
            and not is_state_changing_tool(str(call.get("tool_name", "")), "order")
            and not is_final_compute_tool(str(call.get("tool_name", "")))
        ):
            _set_pin(state, "restaurant_name", params.get("restaurant_name"), f"{source}_params")
    text = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result or "")
    update_pins_from_text(state, text, f"{source}_observation")


def mutation_signature(tool_name: str, params: Dict[str, Any], scenario: str, state: Dict[str, Any]) -> Dict[str, Any]:
    entity_type, entity_name, original_name = entity_from_params(tool_name, params)
    pins = state.get("pins", {})
    user_id = params.get("user_id") or params.get("customer_id") or pins.get("user_id") or ""
    restaurant = params.get("restaurant_name") or pins.get("restaurant_name") or ""
    return {
        "action_family": action_family(tool_name),
        "scenario": scenario,
        "user_id": canonical_text(user_id),
        "restaurant_name": canonical_text(restaurant),
        "entity_type": entity_type,
        "entity_name": entity_name,
        "quantity": normalized_quantity(params),
        "tool_name": tool_name,
        "original_entity_name": original_name,
    }


def signature_key(sig: Dict[str, Any], include_quantity: bool = True) -> str:
    fields = ["action_family", "scenario", "user_id", "restaurant_name", "entity_type", "entity_name"]
    if include_quantity:
        fields.append("quantity")
    return "|".join(str(sig.get(k, "")) for k in fields)


def _tool_needs_order_restaurant(tool_name: str, scenario: str) -> bool:
    if scenario != "order":
        return False
    name = tool_name.lower()
    return any(piece in name for piece in ORDER_RESTAURANT_TOOLS)


def _order_allows_cross_restaurant_retrieval(tool_name: str, params: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Allow menu-comparison retrieval before the order mutation/compute phase.

    Order tasks can start by asking the agent to compare two menus. A pinned
    restaurant should constrain later add/remove/aggregate calls, but blocking
    the exploratory retrieval from the other menu prevents the agent from
    gathering the process evidence the evaluator expects.
    """
    if state.get("scenario") != "order":
        return False
    if is_state_changing_tool(tool_name, "order") or is_final_compute_tool(tool_name):
        return False
    name = tool_name.lower()
    if not any(piece in name for piece in ("find", "get", "search", "list", "dish", "set_meal", "set meal", "category")):
        return False
    if not params.get("restaurant_name"):
        return False
    return True


@lru_cache(maxsize=1)
def _known_set_meal_names() -> Dict[str, str]:
    try:
        cache = build_canonical_cache()
        values = list((cache.get("set_meal_name") or {}).values())
    except Exception:
        values = []
    return {canonical_text(value): str(value).strip() for value in values if str(value or "").strip()}


def _canonical_set_meal_name(value: Any) -> Optional[str]:
    key = canonical_text(value)
    if not key:
        return None
    mapping = _known_set_meal_names()
    if key in mapping:
        return mapping[key]
    return None


def _state_has_successful_remove(state: Dict[str, Any]) -> bool:
    ledger = state.get("successful_mutation_ledger") or {}
    for item in ledger.values():
        if isinstance(item, dict):
            sig = item.get("signature") or {}
            if sig.get("action_family") == "remove":
                return True
    return False


def _synthetic_result(content: str, call: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "tool",
        "content": content,
        "blocked": True,
        "tool_name": call.get("tool_name"),
    }


def prepare_tool_calls_for_execution(
    tool_call_obj: Any,
    scenario: str,
    state: Optional[Dict[str, Any]] = None,
    turn: int = 0,
) -> Any:
    """Return an execution-only copy for official DB quirks.

    Order aggregate tool schemas and OrderDB both use dishes[].product_name for
    menu items. Older helper output sometimes used dish_name, so the execution
    copy fills product_name only when needed.
    """
    calls = deepcopy(tool_call_obj)
    call_list = calls if isinstance(calls, list) else [calls]
    changed = False
    for call in call_list:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name", ""))
        params = call.get("parameters", {})
        if not isinstance(params, dict):
            continue
        if scenario == "order" and is_final_compute_tool(name):
            dishes = params.get("dishes")
            if isinstance(dishes, list):
                for item in dishes:
                    if isinstance(item, dict) and "dish_name" in item and "product_name" not in item:
                        item["product_name"] = item["dish_name"]
                        changed = True
                if changed:
                    append_wrapper_event(state, {
                        "event": "aggregate_tool_selected",
                        "turn": turn,
                        "tool_name": name,
                        "decision": "execution_copy_add_product_name_for_orderdb",
                    })
    return calls


def _param_value(params: Dict[str, Any], key: str) -> str:
    value = params.get(key)
    return canonical_text(value)


def _kitchen_query_signature(tool_name: str, params: Dict[str, Any]) -> str:
    name = tool_name.lower()
    key = ""
    if name == "get_recipe_ingredients":
        key = _param_value(params, "recipe_name")
    elif "ingredient" in name:
        key = _param_value(params, "ingredient_name")
    else:
        key = json.dumps(params, ensure_ascii=False, sort_keys=True)
    return f"{name}|{key}"


def _looks_success(result: Any) -> bool:
    if isinstance(result, dict):
        if str(result.get("status", "")).lower() == "success":
            return True
        content = result.get("content", result)
    else:
        content = result
    text = str(content).lower()
    failure_words = ("error", "failed", "not found", "invalid", "cannot")
    if any(w in text for w in failure_words):
        return False
    return "success" in text or "added" in text or "removed" in text or "updated" in text


def apply_pre_execution_guard(
    tool_call_obj: Any,
    scenario: str,
    history: List[Dict[str, Any]],
    state: Dict[str, Any],
    turn: int,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    """Filter or autofill tool calls before execute_tool sees them."""
    update_pins_from_text(state, _history_text(history), "dialogue")
    calls = tool_call_obj if isinstance(tool_call_obj, list) else [tool_call_obj]
    filtered: List[Dict[str, Any]] = []
    synthetic_results: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    batch_has_order_remove = any(
        isinstance(c, dict)
        and action_family(str(c.get("tool_name", ""))) == "remove"
        and "order" in str(c.get("tool_name", "")).lower()
        for c in calls
    )

    if scenario == "order":
        tool_call_obj, v8_synth, v8_decisions = apply_order_helper(tool_call_obj, state, turn)
        synthetic_results.extend(v8_synth)
        decisions.extend(v8_decisions)
        calls = tool_call_obj if isinstance(tool_call_obj, list) else ([tool_call_obj] if tool_call_obj else [])
    elif scenario == "kitchen":
        tool_call_obj, v8_synth, v8_decisions = apply_kitchen_helper(tool_call_obj, state, turn)
        synthetic_results.extend(v8_synth)
        decisions.extend(v8_decisions)
        calls = tool_call_obj if isinstance(tool_call_obj, list) else ([tool_call_obj] if tool_call_obj else [])
    elif scenario == "retail":
        tool_call_obj, v9_synth, v9_decisions = apply_retail_narrower(tool_call_obj, state, turn)
        synthetic_results.extend(v9_synth)
        decisions.extend(v9_decisions)
        calls = tool_call_obj if isinstance(tool_call_obj, list) else ([tool_call_obj] if tool_call_obj else [])

    for call in calls:
        if not isinstance(call, dict):
            continue
        call2 = deepcopy(call)
        name = str(call2.get("tool_name", ""))
        params = call2.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
            call2["parameters"] = params

        if scenario == "order" and params.get("restaurant_name"):
            raw_restaurant = str(params.get("restaurant_name"))
            raw_c = canonical_text(raw_restaurant)
            for restaurant in KNOWN_RESTAURANTS:
                if canonical_text(restaurant) in raw_c and raw_restaurant != restaurant:
                    params["restaurant_name"] = restaurant
                    decisions.append({
                        "tool_name": name,
                        "decision": "canonicalize",
                        "field": "restaurant_name",
                        "from": raw_restaurant,
                        "to": restaurant,
                    })
                    append_wrapper_event(state, {
                        "event": "restaurant_canonicalized",
                        "turn": turn,
                        "tool_name": name,
                        "from": raw_restaurant,
                        "to": restaurant,
                    })
                    break

        if scenario == "order" and name == "remove_dish_from_order":
            maybe_set_meal = _canonical_set_meal_name(params.get("dish_name"))
            if maybe_set_meal:
                original_params = deepcopy(params)
                params.pop("dish_name", None)
                params["set_meal_name"] = maybe_set_meal
                call2["tool_name"] = "remove_set_meal_from_order"
                name = "remove_set_meal_from_order"
                decisions.append({
                    "tool_name": "remove_dish_from_order",
                    "decision": "rewrite",
                    "to": "remove_set_meal_from_order",
                    "reason": "known_set_meal_item",
                    "set_meal_name": maybe_set_meal,
                })
                append_wrapper_event(state, {
                    "event": "order_process_alignment",
                    "turn": turn,
                    "tool_name": "remove_dish_from_order",
                    "decision": "rewrite_to_remove_set_meal_from_order",
                    "reason": "known_set_meal_item",
                    "raw_params": original_params,
                    "rewritten_params": params,
                })

        if (
            scenario == "order"
            and state.get("order_requested_final_aggregate") == "tax"
            and name == "compute_total_payment"
            and (batch_has_order_remove or _state_has_successful_remove(state))
        ):
            call2["tool_name"] = "compute_total_tax"
            name = "compute_total_tax"
            decisions.append({
                "tool_name": "compute_total_payment",
                "decision": "rewrite",
                "to": "compute_total_tax",
                "reason": "user_requested_final_tax_after_order_removal",
            })
            append_wrapper_event(state, {
                "event": "aggregate_tool_selected",
                "turn": turn,
                "tool_name": "compute_total_tax",
                "decision": "rewrite_payment_to_tax",
                "reason": "user_requested_final_tax_after_order_removal",
            })

        if os.environ.get("TRACK2_ENABLE_HUMAN_PRIOR", "0") == "1":
            try:
                from .counterfactual_db_simulator import assess_call
                from .human_prior_controller import append_human_prior_event, level as human_prior_level
                if human_prior_level() in {"counterfactual", "helpers", "full"}:
                    cf = assess_call(name, params, scenario, state)
                    append_human_prior_event(state, {"event": "human_prior_counterfactual", "turn": turn, "tool_name": name, "parameters": params, "decision": cf})
                    if cf.get("action") == "block":
                        content = "Counterfactual DB simulator blocked risky state-changing call: " + ", ".join(cf.get("risk_reason") or [])
                        synthetic_results.append(_synthetic_result(content, call2))
                        decisions.append({"tool_name": name, "decision": "block", "reason": "human_prior_counterfactual", "counterfactual": cf})
                        continue
            except Exception:
                pass

        if scenario == "kitchen":
            lname = name.lower()
            seen_queries = state.setdefault("kitchen_seen_queries", [])
            if lname in {
                "get_recipe_ingredients",
                "get_ingredient_location",
                "get_ingredient_quantity",
                "get_ingredient_nutrition",
                "find_ingredient_category",
            }:
                qsig = _kitchen_query_signature(name, params)
                if qsig in seen_queries:
                    content = f"Duplicate kitchen query skipped: {name} with same parameters was already checked."
                    synthetic_results.append(_synthetic_result(content, call2))
                    decisions.append({"tool_name": name, "decision": "block", "reason": "duplicate_kitchen_query"})
                    append_wrapper_event(state, {
                        "event": "kitchen_prune",
                        "turn": turn,
                        "current_stage": state.get("kitchen_stage"),
                        "tool_count": state.get("tool_call_count"),
                        "blocked_tool": name,
                        "reason": "duplicate_kitchen_query",
                        "raw_params": params,
                    })
                    continue
                seen_queries.append(qsig)

            if lname == "get_ingredient_location":
                batch_location_calls = sum(
                    1
                    for c in calls
                    if isinstance(c, dict) and str(c.get("tool_name", "")).lower() == "get_ingredient_location"
                )
                location_count = state.setdefault("kitchen_location_query_count", 0)
                if batch_location_calls > 8 or location_count >= 12:
                    content = (
                        "Kitchen broad ingredient-location scan blocked. Use find_ingredients_by_location for the target "
                        "location, intersect with current recipe ingredients, then query quantities only for those candidates."
                    )
                    synthetic_results.append(_synthetic_result(content, call2))
                    decisions.append({"tool_name": name, "decision": "block", "reason": "kitchen_broad_location_scan"})
                    append_wrapper_event(state, {
                        "event": "kitchen_prune",
                        "turn": turn,
                        "current_stage": state.get("kitchen_stage"),
                        "tool_count": state.get("tool_call_count"),
                        "blocked_tool": name,
                        "reason": "kitchen_broad_location_scan",
                        "raw_params": params,
                    })
                    continue
                state["kitchen_location_query_count"] = location_count + 1

            if state.get("tool_call_count", 0) > 35 and lname not in {
                "add_to_shopping_list",
                "add_recipe_to_menu",
                "remove_from_shopping_list",
                "remove_recipe_from_menu",
                "compute_total_nutritions",
                "get_current_shopping_list",
                "get_current_menu",
            }:
                content = (
                    "Kitchen conservative mode blocked more exploration. Finish pending state changes, "
                    "get current list/menu if needed, or compute_total_nutritions."
                )
                synthetic_results.append(_synthetic_result(content, call2))
                decisions.append({"tool_name": name, "decision": "block", "reason": "kitchen_conservative_mode"})
                append_wrapper_event(state, {
                    "event": "kitchen_conservative_mode",
                    "turn": turn,
                    "current_stage": state.get("kitchen_stage"),
                    "tool_count": state.get("tool_call_count"),
                    "blocked_tool": name,
                    "reason": "tool_count_gt_35",
                })
                continue

        for pin_type, param_keys in (
            ("user_id", ("user_id", "customer_id")),
            ("restaurant_name", ("restaurant_name",)),
        ):
            pinned = state.get("pins", {}).get(pin_type)
            if not pinned:
                continue
            needs_restaurant = pin_type == "restaurant_name" and _tool_needs_order_restaurant(name, scenario)
            needs_user = pin_type == "user_id" and (is_state_changing_tool(name, scenario) or any(k in params for k in param_keys))
            if not needs_restaurant and not needs_user:
                continue
            present_key, present_value = _first_param(params, param_keys)
            if not present_key:
                params[param_keys[0]] = pinned
                decisions.append({"tool_name": name, "decision": "autofill", "pin_type": pin_type, "value": pinned})
                append_wrapper_event(state, {
                    "event": "pin_autofill",
                    "turn": turn,
                    "tool_name": name,
                    "pin_type": pin_type,
                    "value": pinned,
                    "source": state.get("pin_sources", {}).get(pin_type),
                })
            elif canonical_text(present_value) != canonical_text(pinned):
                if pin_type == "restaurant_name" and _order_allows_cross_restaurant_retrieval(name, params, state):
                    decisions.append({
                        "tool_name": name,
                        "decision": "allow_cross_restaurant_retrieval",
                        "pin_type": pin_type,
                        "pinned": pinned,
                        "value": present_value,
                    })
                    append_wrapper_event(state, {
                        "event": "order_process_alignment",
                        "turn": turn,
                        "tool_name": name,
                        "decision": "allow_cross_restaurant_retrieval",
                        "pinned_restaurant": pinned,
                        "requested_restaurant": present_value,
                    })
                    continue
                content = f"Pin conflict blocked: {pin_type} must be {pinned}, got {present_value}."
                synthetic_results.append(_synthetic_result(content, call2))
                decisions.append({"tool_name": name, "decision": "block", "reason": "pin_conflict", "pin_type": pin_type})
                append_wrapper_event(state, {
                    "event": "pin_conflict_blocked",
                    "turn": turn,
                    "tool_name": name,
                    "pin_type": pin_type,
                    "value": pinned,
                    "raw_params": params,
                    "decision": "block",
                })
                break
        else:
            if is_final_compute_tool(name):
                fp = json.dumps({"name": name, "params": params}, ensure_ascii=False, sort_keys=True)
                if fp in state.setdefault("compute_call_ledger", {}):
                    content = "Duplicate final compute call skipped: same parameters already computed in this episode."
                    synthetic_results.append(_synthetic_result(content, call2))
                    decisions.append({"tool_name": name, "decision": "block", "reason": "duplicate_compute"})
                    append_wrapper_event(state, {"event": "duplicate_compute_blocked", "turn": turn, "tool_name": name, "raw_params": params})
                    continue
                filtered.append(call2)
                continue

            if is_state_changing_tool(name, scenario):
                sig = mutation_signature(name, params, scenario, state)
                key = signature_key(sig, include_quantity=True)
                loose_key = signature_key(sig, include_quantity=False)
                ledger = state.setdefault("successful_mutation_ledger", {})
                if key in ledger or loose_key in ledger:
                    previous = ledger.get(key) or ledger.get(loose_key) or {}
                    content = "Duplicate state-changing call blocked: already completed in this episode."
                    synthetic_results.append(_synthetic_result(content, call2))
                    decisions.append({"tool_name": name, "decision": "block", "reason": "duplicate_mutation", "signature": sig})
                    state.setdefault("blocked_calls", []).append({"turn": turn, "call": call2, "signature": sig, "reason": "duplicate_mutation"})
                    append_wrapper_event(state, {
                        "event": "duplicate_mutation_blocked",
                        "turn": turn,
                        "tool_name": name,
                        "signature": sig,
                        "raw_params": params,
                        "previous_success_turn": previous.get("turn"),
                        "decision": "block",
                    })
                    continue
            filtered.append(call2)

    report = {
        "decision": "allow" if len(filtered) == len(calls) and not synthetic_results else "repair",
        "filtered_count": len(filtered),
        "blocked_count": len(synthetic_results),
        "decisions": decisions,
        "pins": deepcopy(state.get("pins", {})),
    }
    if not filtered:
        return [], synthetic_results, report
    return filtered if isinstance(tool_call_obj, list) else filtered[0], synthetic_results, report


def record_post_execution(
    tool_call_obj: Any,
    tool_results: List[Dict[str, Any]],
    scenario: str,
    state: Dict[str, Any],
    turn: int,
) -> None:
    calls = tool_call_obj if isinstance(tool_call_obj, list) else [tool_call_obj]
    results = tool_results if isinstance(tool_results, list) else [tool_results]
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        result = results[idx] if idx < len(results) else {}
        update_pins_from_call_or_result(state, call, result, source="successful_tool" if _looks_success(result) else "tool")
        name = str(call.get("tool_name", ""))
        params = call.get("parameters", {}) if isinstance(call.get("parameters", {}), dict) else {}
        state.setdefault("executed_tool_calls", []).append({"turn": turn, "tool_name": name, "parameters": params, "success": _looks_success(result), "result_preview": str(result)[:500]})
        if is_final_compute_tool(name):
            fp = json.dumps({"name": name, "params": params}, ensure_ascii=False, sort_keys=True)
            state.setdefault("compute_call_ledger", {})[fp] = {"turn": turn, "tool_name": name, "success": _looks_success(result), "result_preview": str(result)[:300]}
        if is_state_changing_tool(name, scenario) and _looks_success(result):
            sig = mutation_signature(name, params, scenario, state)
            state.setdefault("successful_mutation_ledger", {})[signature_key(sig, include_quantity=True)] = {
                "turn": turn,
                "tool_name": name,
                "signature": sig,
                "result": result,
            }
            state.setdefault("successful_mutation_ledger", {})[signature_key(sig, include_quantity=False)] = {
                "turn": turn,
                "tool_name": name,
                "signature": sig,
                "result": result,
            }
            append_wrapper_event(state, {
                "event": "mutation_recorded",
                "turn": turn,
                "tool_name": name,
                "signature": sig,
            })


def guard_tool_calls(tool_json: str, scenario: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Backward-compatible lightweight report used by older wrapper code."""
    schema = get_scenario_schema(scenario)
    report = {"decision": "allow", "risk_flags": [], "blocked": False}
    try:
        calls = json.loads(tool_json)
    except Exception:
        report["decision"] = "block"
        report["blocked"] = True
        report["risk_flags"].append("invalid_json")
        return report
    ids = extract_user_ids_from_text(_history_text(history))
    seen_mods = set()
    for call in calls if isinstance(calls, list) else []:
        name = call.get("tool_name", "")
        params = call.get("parameters", {}) if isinstance(call.get("parameters", {}), dict) else {}
        entry = schema.get(name, {})
        if not entry.get("state_changing") and not is_state_changing_tool(name, scenario):
            continue
        uid = params.get("user_id") or params.get("customer_id")
        if any(k in params for k in ("user_id", "customer_id")) and not uid:
            report["risk_flags"].append(f"missing_user_id:{name}")
        if uid and ids and str(uid) not in ids:
            report["risk_flags"].append(f"user_id_not_in_dialogue:{name}:{uid}")
        fp = json.dumps({"name": name, "params": params}, ensure_ascii=False, sort_keys=True)
        if fp in seen_mods:
            report["risk_flags"].append(f"duplicate_state_change:{name}")
        seen_mods.add(fp)
    if report["risk_flags"]:
        report["decision"] = "repair"
    return report
