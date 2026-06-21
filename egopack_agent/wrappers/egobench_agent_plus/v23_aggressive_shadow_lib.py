#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V23 aggressive val41 shadow helpers.

This module is dev-only.  It never reads val41 GT for candidate generation or
selection.  GT is handled by the runner after all candidates are generated.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple


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
USER_ID_RE = re.compile(
    r"(?:User ID|user_id|customer_id|customer id|Your user ID is)\s*[:=]?\s*([A-Za-z_]+[A-Za-z0-9_]*\d[A-Za-z0-9_]*)",
    re.I,
)
RESTAURANT_RE = re.compile(r"\b([A-Z][A-Za-z'& ]{2,60}(?:Restaurant|Kitchen|Bistro|Cafe|Steakhouse|Pizzeria))\b")


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "__dataclass_fields__"):
        out = asdict(obj)
    elif isinstance(obj, dict):
        out = dict(obj)
    else:
        out = dict(getattr(obj, "__dict__", {}))
    for k, v in list(out.items()):
        if hasattr(v, "__dataclass_fields__"):
            out[k] = asdict(v)
    return out


def extract_user_id(instruction: str) -> str:
    m = USER_ID_RE.search(instruction or "")
    return m.group(1) if m else ""


def extract_restaurant_name(instruction: str, known: Iterable[str] = ()) -> str:
    text = instruction or ""
    lower = norm_text(text)
    best = ""
    for name in known:
        if norm_text(name) and norm_text(name) in lower:
            return str(name)
    m = RESTAURANT_RE.search(text)
    if m:
        best = m.group(1)
    return best


def required_closure(instruction: str, scenario: str) -> str:
    text = norm_text(instruction)
    if "total tax" in text:
        return "compute_total_tax"
    if any(x in text for x in ("total payment", "amount payable", "total cost", "payable")):
        return "compute_total_payment"
    if any(x in text for x in ("total nutrition", "total nutritional", "total calcium")):
        if scenario == "kitchen":
            return "compute_total_nutritions"
        return "compute_total_nutrition"
    if "total taste" in text:
        return "tally_total_tastes"
    if "nutritional characteristic" in text and "total" in text:
        return "tally_total_nutritional_characteristics"
    if "summary" in text and scenario in {"order", "restaurant"}:
        return "get_user_order_summary"
    return ""


def has_mutation_intent(instruction: str) -> bool:
    text = norm_text(instruction)
    return any(x in text for x in ("add", "remove", "update", "delete", "cart", "order", "shopping list", "menu"))


def has_visual_ref(instruction: str) -> bool:
    text = norm_text(instruction)
    return any(x in text for x in ("point", "bottle", "dish", "menu", "section", "shelf", "hand", "left", "right", "box", "tray", "pot", "wok", "cutting board"))


def has_branch(instruction: str) -> bool:
    text = f" {norm_text(instruction)} "
    return any(x in text for x in (" if ", " whether ", " otherwise ", " else ", " tied ", " tie "))


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        if not isinstance(block, dict):
            continue
        for call in block.get("calls") or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": copy.deepcopy(call.get("parameters") or {})})
    return out


def make_item(row: Dict[str, Any], program: List[Dict[str, Any]], label: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} V23 shadow candidate."}],
        "tool_calls": [
            {
                "turn": 0,
                "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program],
                "blocked_calls": [],
                "results": [],
                "v23_meta": meta or {},
            }
        ],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
    }


def products_from_cart_like(db: Any, user_id: str, names: List[str], scenario: str) -> List[Dict[str, Any]]:
    if scenario == "retail":
        current = getattr(db, "user_carts", {}).get(user_id, {})
        out = [{"product_name": as_dict(x).get("product_name"), "quantity": as_dict(x).get("quantity", 1)} for x in current.values()]
        key = "product_name"
    elif scenario in {"restaurant", "order"}:
        if scenario == "order":
            # order DB stores restaurant-specific orders; this helper is called
            # with a wrapper-added current_order where available.
            current = {}
        else:
            current = getattr(db, "user_orders", {}).get(user_id, {})
        out = [{"dish_name": as_dict(x).get("dish_name") or as_dict(x).get("name") or as_dict(x).get("set_meal_name"), "quantity": as_dict(x).get("quantity", 1)} for x in current.values()]
        key = "dish_name"
    else:
        out = []
        key = "ingredient_name"
    seen = {norm_text(x.get(key) or x.get("product_name") or x.get("dish_name")) for x in out}
    for name in names:
        if norm_text(name) not in seen:
            out.append({key: name, "quantity": 1})
            seen.add(norm_text(name))
    return out


def find_catalog_rows(db: Any, scenario: str) -> List[Dict[str, Any]]:
    if scenario == "retail":
        return [as_dict(x) for x in getattr(db, "catalog", {}).values()]
    if scenario in {"restaurant", "order"}:
        if scenario == "order":
            restaurants = getattr(db, "restaurants", {})
            rows = []
            for rname, rest in restaurants.items():
                for dish in getattr(rest, "dishes", {}).values():
                    row = as_dict(dish)
                    row["_restaurant_name"] = rname
                    rows.append(row)
                for meal in getattr(rest, "set_meals", {}).values():
                    row = as_dict(meal)
                    row["_restaurant_name"] = rname
                    row["_is_set_meal"] = True
                    rows.append(row)
            return rows
        rows = []
        for dish in getattr(db, "dishes", {}).values():
            row = as_dict(dish)
            rows.append(row)
        for meal in getattr(db, "set_meals", {}).values():
            row = as_dict(meal)
            row["_is_set_meal"] = True
            rows.append(row)
        return rows
    return []


def lexical_candidates(row: Dict[str, Any], db: Any, scenario: str, limit: int = 8) -> List[Dict[str, Any]]:
    text = norm_text(" ".join([
        str(row.get("Instruction", "")),
        str(row.get("image_description", "")),
        str(row.get("value", "")),
    ]))
    out: List[Dict[str, Any]] = []
    if scenario == "kitchen":
        recipes = getattr(db, "recipes", {})
        ingredients = getattr(db, "ingredients", {})
        for name, obj in list(recipes.items()) + list(ingredients.items()):
            score = 0.0
            n = norm_text(name)
            if n and n in text:
                score += 4.0
            for tok in n.split():
                if len(tok) > 3 and tok in text:
                    score += 0.5
            if score:
                out.append({"name": name, "score": score, "type": "recipe" if name in recipes else "ingredient"})
    else:
        for r in find_catalog_rows(db, scenario):
            name = r.get("name") or r.get("dish_name") or r.get("set_meal_name")
            if not name:
                continue
            score = 0.0
            n = norm_text(name)
            if n in text:
                score += 4.0
            for tok in n.split():
                if len(tok) > 3 and tok in text:
                    score += 0.4
            if r.get("category") and norm_text(r.get("category")) in text:
                score += 0.6
            if score:
                out.append({"name": name, "score": score, "row": r, "type": "set_meal" if r.get("_is_set_meal") else "entity"})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:limit]


def closure_call(scenario: str, db: Any, instruction: str, user_id: str, added: List[str], restaurant_name: str = "") -> Dict[str, Any] | None:
    tool = required_closure(instruction, scenario)
    if not tool:
        return None
    if scenario == "retail":
        return {"tool_name": tool, "parameters": {"user_id": user_id, "products": products_from_cart_like(db, user_id, added, "retail")}}
    if scenario == "restaurant":
        if tool == "get_user_order_summary":
            return {"tool_name": tool, "parameters": {"user_id": user_id}}
        return {"tool_name": tool, "parameters": {"user_id": user_id, "dishes": products_from_cart_like(db, user_id, added, "restaurant")}}
    if scenario == "order":
        if tool == "get_user_order_summary":
            return {"tool_name": tool, "parameters": {"restaurant_name": restaurant_name, "user_id": user_id}}
        return {"tool_name": tool, "parameters": {"restaurant_name": restaurant_name, "user_id": user_id, "dishes": [{"dish_name": x, "quantity": 1} for x in added]}}
    if scenario == "kitchen":
        if tool in {"tally_total_tastes", "tally_total_nutritional_characteristics"}:
            return {"tool_name": tool, "parameters": {"user_id": user_id, "recipes": added}}
        return {"tool_name": tool, "parameters": {"user_id": user_id, "ingredients": [{"ingredient_name": x, "quantity": 1} for x in added]}}
    return None


def dryrun_program(scenario: str, db: Any, program: List[Dict[str, Any]], instruction: str) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    mutation = 0
    aggregate = 0
    retrieval_nonempty = 0
    for i, step in enumerate(program):
        tool = str(step.get("tool_name") or "")
        params = step.get("parameters") or {}
        if not tool:
            errors.append({"idx": i, "reason": "missing_tool"})
            continue
        if not hasattr(db, tool):
            # Some evaluator-accepted process tools are absent in local DB.
            warnings.append({"idx": i, "tool": tool, "reason": "local_db_missing"})
            continue
        try:
            if tool.startswith(("get_", "find_", "list_")):
                res = getattr(db, tool)(**params)
                if isinstance(res, dict) and any(res.get(k) for k in ("products", "product_names", "dishes", "dish_names", "recipes", "ingredients", "items", "set_meals")):
                    retrieval_nonempty += 1
                elif isinstance(res, dict) and res.get("status") == "error":
                    warnings.append({"idx": i, "tool": tool, "reason": "retrieval_error", "message": res.get("message")})
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
    if has_mutation_intent(instruction) and not mutation and scenario != "kitchen":
        warnings.append({"reason": "mutation_intent_no_mutation"})
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "mutation_count": mutation,
        "aggregate_count": aggregate,
        "retrieval_nonempty_count": retrieval_nonempty,
        "closure_required": closure,
        "closure_complete": not missing_closure,
        "broad_scan": broad_scan,
        "tool_count": len(program),
    }


def select_candidate(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    scored = []
    instruction = context["instruction"]
    scenario = context["scenario"]
    v14_success_prior = context.get("v14_joint_prior", False)
    v22_success_prior = context.get("v22_joint_prior", False)
    for cand in candidates:
        dry = cand.get("dryrun") or {}
        names = [str(x.get("tool_name") or "") for x in cand.get("tool_program") or []]
        hard = []
        if not names:
            hard.append("empty_program")
        if dry.get("errors"):
            hard.append("schema_or_tool_error")
        if has_mutation_intent(instruction) and scenario in {"retail", "order", "restaurant"} and not any(MUTATION_RE.search(n) for n in names):
            hard.append("mutation_intent_no_mutation")
        if dry.get("closure_required") and not dry.get("closure_complete"):
            hard.append("missing_closure")
        if has_visual_ref(instruction) and any(n.startswith(("find_products_by_price_range", "get_all_", "list_all_")) for n in names[:2]):
            hard.append("visual_task_leading_broad_scan")
        score = 0.0
        reasons = []
        if cand.get("source") == "V14" and v14_success_prior:
            score += 5
            reasons.append("v14_success_prior")
        if cand.get("source") == "V22" and v22_success_prior:
            score += 5
            reasons.append("v22_success_prior")
        if dry.get("ok"):
            score += 3
            reasons.append("dryrun_ok")
        if dry.get("mutation_count"):
            score += 2
            reasons.append("mutation_happened")
        if dry.get("closure_complete"):
            score += 2
            reasons.append("closure_complete")
        if dry.get("retrieval_nonempty_count"):
            score += min(2, dry.get("retrieval_nonempty_count"))
            reasons.append("retrieval_nonempty")
        if any(n.startswith("get_") for n in names[: max(1, next((i for i, n in enumerate(names) if MUTATION_RE.search(n)), len(names)))]):
            score += 1.5
            reasons.append("specific_getter_before_mutation")
        if not dry.get("broad_scan"):
            score += 1
            reasons.append("no_broad_scan")
        score += float(cand.get("shape_confidence") or 0)
        score -= max(0, len(names) - 25) * 0.08
        if "foreign_slot_copy" in cand.get("risk_flags", []):
            score -= 5
        if hard:
            score -= 20
        out = dict(cand)
        out["selector_score"] = score
        out["selector_reasons"] = reasons
        out["hard_filters"] = hard
        scored.append(out)
    scored.sort(key=lambda x: (x["selector_score"], -len(x.get("tool_program") or [])), reverse=True)
    return scored[0] if scored else {"candidate_id": "empty", "source": "empty", "tool_program": [], "selector_score": -999, "hard_filters": ["no_candidates"]}
