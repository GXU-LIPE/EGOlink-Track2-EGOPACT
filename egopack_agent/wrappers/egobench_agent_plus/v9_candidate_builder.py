# -*- coding: utf-8 -*-
"""Build conservative V9.5 candidate variants before deterministic reranking.

This module does not solve tasks and does not read GT. It only creates safer
variants of a model-proposed tool batch when the batch has obvious process
risks, such as large retail attribute scans or missing pinned identifiers.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

from .v8_event_logger import enabled, write_v8_event


ATTRIBUTE_TERMS = ("price", "tax", "discount", "nutrition", "nutritional")
MUTATION_PREFIXES = ("add", "remove", "delete", "update", "modify")
AGGREGATE_PREFIXES = ("compute_total_tax", "compute_total_payment", "compute_total_price", "compute_total_nutrition", "compute_total_nutritions")


def _loads(candidate: Any) -> List[Dict[str, Any]]:
    if isinstance(candidate, str):
        try:
            data = json.loads(candidate)
        except Exception:
            return []
    else:
        data = candidate
    if isinstance(data, dict):
        data = [data]
    return [x for x in data if isinstance(x, dict)]


def _dumps(calls: List[Dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False, separators=(",", ":"))


def _name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def _params(call: Dict[str, Any]) -> Dict[str, Any]:
    params = call.get("parameters") or call.get("arguments") or {}
    return params if isinstance(params, dict) else {}


def _text(call: Dict[str, Any]) -> str:
    params = _params(call)
    return " ".join([_name(call)] + [str(k) for k in params] + [str(v) for v in params.values()]).lower()


def _is_attr(call: Dict[str, Any]) -> bool:
    return any(term in _text(call) for term in ATTRIBUTE_TERMS)


def _is_mutation_or_aggregate(call: Dict[str, Any]) -> bool:
    name = _name(call).lower()
    return (
        name.startswith(MUTATION_PREFIXES)
        or "_to_" in name
        or "_from_" in name
        or name.startswith("compute_total")
    )


def _is_mutation(call: Dict[str, Any]) -> bool:
    name = _name(call).lower()
    return name.startswith(MUTATION_PREFIXES) or "_to_" in name or "_from_" in name


def _is_aggregate(call: Dict[str, Any]) -> bool:
    return _name(call).lower().startswith(AGGREGATE_PREFIXES)


def _is_retrieval(call: Dict[str, Any]) -> bool:
    name = _name(call).lower()
    return name.startswith(("get_", "find_", "list_", "search_", "retrieve_"))


def _product_key(call: Dict[str, Any]) -> str:
    params = _params(call)
    for key in ("product_name", "name", "item_name", "product"):
        if params.get(key):
            return str(params.get(key)).strip().lower()
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _entity_key(call: Dict[str, Any]) -> str:
    params = _params(call)
    for key in ("product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name", "category"):
        if params.get(key):
            return f"{key}:{str(params.get(key)).strip().lower()}"
    return _product_key(call)


def _state_call_names(state: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for c in (state or {}).get("executed_tool_calls") or []:
        if isinstance(c, dict):
            out.append(str(c.get("tool_name") or ""))
    return out


def _requested_order_aggregate(state: Dict[str, Any]) -> str:
    requested = str((state or {}).get("order_requested_final_aggregate") or "").lower()
    text = str((state or {}).get("user_instruction") or "").lower()
    if requested == "tax" or "total tax" in text or "tax amount" in text:
        return "compute_total_tax"
    if requested in {"payment", "price"} or any(x in text for x in ("total payment", "amount payable", "total price", "total cost")):
        return "compute_total_payment"
    if requested == "nutrition" or "nutrition" in text or "calorie" in text:
        return "compute_total_nutrition"
    return ""


def _normalize_order_aggregate_items(params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    changed = False
    out = copy.deepcopy(params)
    dishes = out.get("dishes")
    if not isinstance(dishes, list):
        return out, changed
    fixed = []
    for item in dishes:
        if isinstance(item, dict):
            item2 = copy.deepcopy(item)
            if "dish_name" in item2 and "product_name" not in item2:
                item2["product_name"] = item2.pop("dish_name")
                changed = True
            fixed.append(item2)
        else:
            fixed.append({"product_name": item, "quantity": 1})
            changed = True
    out["dishes"] = fixed
    return out, changed


def _unique(candidates: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _retail_trimmed_candidate(calls: List[Dict[str, Any]], state: Dict[str, Any], turn: int) -> str | None:
    attr_calls = [c for c in calls if _is_attr(c)]
    product_keys = {_product_key(c) for c in attr_calls}
    broad = len(attr_calls) > 8 or len(product_keys) > 5
    if not broad:
        return None

    kept: List[Dict[str, Any]] = []
    kept_attr = 0
    attr_products_seen = set()
    retrieval_kept_by_entity = set()
    for call in calls:
        call2 = copy.deepcopy(call)
        if _is_attr(call2):
            key = _product_key(call2)
            # Keep enough attributes for the first few candidates, then let the
            # next turn use observations to narrow. Dropping the tail prevents
            # 40-80 call sweeps from being recorded as the service trajectory.
            if kept_attr >= 8 or len(attr_products_seen | {key}) > 4:
                continue
            kept_attr += 1
            attr_products_seen.add(key)
            kept.append(call2)
            continue
        if _is_retrieval(call2):
            ekey = _entity_key(call2)
            # Preserve process-shape retrieval, but cap duplicate retrievals for
            # the same entity. This is less destructive than dropping all late
            # process tools and keeps evaluator-visible narrowing steps.
            if ekey in retrieval_kept_by_entity and len(kept) >= 10:
                continue
            retrieval_kept_by_entity.add(ekey)
        if _is_mutation(call2) and attr_calls and len(attr_products_seen) > 4:
            # A mutation after an unresolved catalog sweep is unsafe. Keep
            # aggregates, because some retail tasks expect total/tax process
            # after already confirmed cart changes.
            continue
        kept.append(call2)

    if not kept or kept == calls:
        return None
    write_v8_event(
        state,
        "v9_candidate_builder",
        "repair",
        "retail_broad_scan_candidate_trimmed",
        turn=turn,
        whether_repaired=True,
        risk_score=0.65,
        original_calls=len(calls),
        original_attr_calls=len(attr_calls),
        product_candidate_count=len(product_keys),
        repaired_calls=len(kept),
    )
    return _dumps(kept)


def _order_pin_candidate(calls: List[Dict[str, Any]], state: Dict[str, Any], turn: int) -> str | None:
    pins = (state or {}).get("pins") or {}
    restaurant = pins.get("restaurant_name")
    if not restaurant:
        return None
    changed = False
    out: List[Dict[str, Any]] = []
    for call in calls:
        call2 = copy.deepcopy(call)
        name = _name(call2).lower()
        params = _params(call2)
        if (
            ("order" in name or "dish" in name or "set_meal" in name or name.startswith("compute_total"))
            and "restaurant_name" not in params
        ):
            params["restaurant_name"] = restaurant
            call2["parameters"] = params
            changed = True
        out.append(call2)
    if not changed:
        return None
    write_v8_event(
        state,
        "v9_candidate_builder",
        "repair",
        "order_pinned_restaurant_candidate",
        turn=turn,
        whether_repaired=True,
        risk_score=0.25,
        repaired_calls=len(out),
    )
    return _dumps(out)


def _order_process_candidates(calls: List[Dict[str, Any]], state: Dict[str, Any], turn: int) -> List[str]:
    out: List[str] = []
    names = [_name(c) for c in calls]
    prior_names = _state_call_names(state)
    requested_agg = _requested_order_aggregate(state)
    pins = (state or {}).get("pins") or {}
    user_id = pins.get("user_id")
    has_mutation = any(_is_mutation(c) for c in calls) or any(n.startswith(("add_", "remove_")) for n in prior_names)
    has_agg = any(_is_aggregate(c) for c in calls) or any(n.startswith("compute_total") for n in prior_names)

    # Variant 1: normalize aggregate shape and choose the requested compute
    # family when the model picked a nearby aggregate.
    changed = False
    fixed: List[Dict[str, Any]] = []
    for call in calls:
        call2 = copy.deepcopy(call)
        name = _name(call2)
        params = _params(call2)
        if name.startswith("compute_total"):
            if requested_agg and name != requested_agg:
                call2["tool_name"] = requested_agg
                changed = True
            params2, c2 = _normalize_order_aggregate_items(params)
            if c2:
                call2["parameters"] = params2
                changed = True
        fixed.append(call2)
    if changed:
        write_v8_event(
            state,
            "v9_candidate_builder",
            "repair",
            "order_process_aggregate_shape_candidate",
            turn=turn,
            whether_repaired=True,
            requested_aggregate=requested_agg,
            original_tools=names,
            repaired_tools=[_name(c) for c in fixed],
        )
        out.append(_dumps(fixed))

    # Variant 2: append the requested aggregate after a mutation if the model
    # already provided enough items in a nearby aggregate or state carries the
    # process requirement. This is intentionally conservative: no invented item
    # list is created.
    if requested_agg and has_mutation and not has_agg:
        source_dishes = None
        for call in calls:
            params = _params(call)
            if isinstance(params.get("dishes"), list):
                source_dishes = copy.deepcopy(params["dishes"])
                break
        if source_dishes is not None and user_id:
            params = {"user_id": user_id, "dishes": source_dishes}
            params, _ = _normalize_order_aggregate_items(params)
            appended = copy.deepcopy(calls) + [{"tool_name": requested_agg, "parameters": params}]
            write_v8_event(
                state,
                "v9_candidate_builder",
                "repair",
                "order_missing_aggregate_candidate",
                turn=turn,
                whether_repaired=True,
                requested_aggregate=requested_agg,
                repaired_calls=len(appended),
            )
            out.append(_dumps(appended))

    # Variant 3: when a set meal name is placed in dish_name for a remove call,
    # offer the set-meal remove shape. Canonical validation/db_guard will still
    # do the final authoritative rewrite.
    changed = False
    fixed = []
    for call in calls:
        call2 = copy.deepcopy(call)
        name = _name(call2)
        params = _params(call2)
        dish_value = str(params.get("dish_name") or "")
        if name == "remove_dish_from_order" and ("set" in dish_value.lower() or "meal" in dish_value.lower()):
            params2 = copy.deepcopy(params)
            params2["set_meal_name"] = params2.pop("dish_name")
            call2 = {"tool_name": "remove_set_meal_from_order", "parameters": params2}
            changed = True
        fixed.append(call2)
    if changed:
        write_v8_event(
            state,
            "v9_candidate_builder",
            "repair",
            "order_set_meal_remove_shape_candidate",
            turn=turn,
            whether_repaired=True,
            repaired_calls=len(fixed),
        )
        out.append(_dumps(fixed))
    return out


def build_candidates(normalized: str, scenario: str, state: Dict[str, Any], turn: int) -> List[str]:
    """Return original plus safe repair variants for V9.5 reranking."""
    if not enabled("TRACK2_ENABLE_MULTICANDIDATE"):
        return [normalized]
    calls = _loads(normalized)
    if not calls:
        return [normalized]
    candidates = [normalized]
    if scenario == "retail":
        repaired = _retail_trimmed_candidate(calls, state, turn)
        if repaired:
            candidates.append(repaired)
    elif scenario == "order":
        repaired = _order_pin_candidate(calls, state, turn)
        if repaired:
            candidates.append(repaired)
        candidates.extend(_order_process_candidates(calls, state, turn))
    return _unique(candidates)
