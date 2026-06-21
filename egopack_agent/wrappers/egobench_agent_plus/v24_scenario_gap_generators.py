#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scenario-specific candidate generators for V24 val41 shadow runs.

Dev-only.  These generators do not consume val41 GT.  They build executable
program candidates from current instruction fields, Qwen/visual cards, DB
catalogs, and non-final V19 case-library candidates.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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


def scenario_from_spec(spec: str) -> str:
    return re.sub(r"\d+$", "", str(spec or ""))


def candidate(source: str, program: List[Dict[str, Any]], meta: Dict[str, Any] | None = None, confidence: float = 0.0) -> Dict[str, Any]:
    return {
        "candidate_id": source,
        "source": source,
        "tool_program": copy.deepcopy(program),
        "risk_flags": [],
        "shape_confidence": confidence,
        "meta": meta or {},
    }


def dedupe_programs(cands: Sequence[Dict[str, Any]], limit: int = 40) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for c in cands:
        prog = c.get("tool_program") or []
        sig = json.dumps(prog, ensure_ascii=False, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def qwen_entities(qwen: Dict[str, Any], types: Iterable[str] = ()) -> List[str]:
    allowed = {norm_text(x) for x in types if x}
    names: List[str] = []
    for key in ("top_k_candidates", "primary_entities"):
        vals = qwen.get(key) or []
        if not isinstance(vals, list):
            continue
        for obj in vals:
            if not isinstance(obj, dict):
                continue
            typ = norm_text(obj.get("type") or obj.get("entity_type"))
            if allowed and typ and typ not in allowed:
                continue
            name = obj.get("name") or obj.get("entity") or obj.get("product_name") or obj.get("dish_name") or obj.get("set_meal_name") or obj.get("ingredient_name") or obj.get("recipe_name")
            if name:
                names.append(str(name))
    return list(dict.fromkeys(names))


def row_values(row: Dict[str, Any]) -> List[str]:
    value = row.get("value")
    vals: List[str]
    if isinstance(value, list):
        vals = [str(x) for x in value if str(x).strip()]
    elif value:
        vals = [str(value)]
    else:
        vals = []
    return vals


def token_score(name: str, text: str) -> float:
    n = norm_text(name)
    t = norm_text(text)
    if not n:
        return 0.0
    score = 0.0
    if n in t:
        score += 6.0
    for tok in re.findall(r"[a-z0-9_']+", n):
        if len(tok) > 3 and tok in t:
            score += 0.6
    return score


def canonical_lookup(candidates: Iterable[str], names: Iterable[str]) -> List[str]:
    rows = list(names)
    by_norm = {norm_text(x): x for x in rows}
    out: List[str] = []
    for raw in candidates:
        nr = norm_text(raw)
        if not nr:
            continue
        if nr in by_norm:
            out.append(by_norm[nr])
            continue
        scored = sorted(((token_score(n, raw), n) for n in rows), reverse=True)
        if scored and scored[0][0] > 0:
            out.append(scored[0][1])
    return list(dict.fromkeys(out))


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


def has_mutation_intent(instruction: str) -> bool:
    text = norm_text(instruction)
    return any(x in text for x in ("add", "remove", "update", "delete", "cart", "order", "shopping list", "menu"))


def has_branch(instruction: str) -> bool:
    text = f" {norm_text(instruction)} "
    return any(x in text for x in (" if ", " whether ", " otherwise ", " else ", " tie ", " tied "))


def _retail_products(db: Any) -> List[Dict[str, Any]]:
    return [as_dict(x) for x in getattr(db, "catalog", {}).values()]


def _retail_product_names(db: Any) -> List[str]:
    return [x.get("name") for x in _retail_products(db) if x.get("name")]


def _product_by_name(db: Any, name: str) -> Dict[str, Any]:
    for row in _retail_products(db):
        if norm_text(row.get("name")) == norm_text(name):
            return row
    return {}


def _nutrition_val(row: Dict[str, Any], key: str) -> float:
    try:
        return float((row.get("nutrition") or {}).get(key, 0) or 0)
    except Exception:
        return 0.0


def _retail_cart_products(db: Any, user_id: str, added: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cart = getattr(db, "user_carts", {}).get(user_id, {})
    for item in getattr(cart, "values", lambda: [])():
        row = as_dict(item)
        out.append({"product_name": row.get("product_name"), "quantity": row.get("quantity", 1)})
    seen = {norm_text(x.get("product_name")) for x in out}
    for name in added:
        if norm_text(name) not in seen:
            out.append({"product_name": name, "quantity": 1})
            seen.add(norm_text(name))
    return out


def retail_candidates(row: Dict[str, Any], db: Any, qwen: Dict[str, Any] | None = None, max_candidates: int = 40) -> List[Dict[str, Any]]:
    qwen = qwen or {}
    instr = row.get("Instruction", "")
    text = norm_text(instr + "\n" + str(row.get("image_description", "")) + "\n" + json.dumps(qwen, ensure_ascii=False))
    user_id = extract_user_id(instr)
    names = canonical_lookup(row_values(row) + qwen_entities(qwen, {"product", "product_name"}), _retail_product_names(db))
    if not names:
        scored = []
        for p in _retail_products(db):
            s = token_score(p.get("name", ""), text) + token_score(p.get("category", ""), text) * 0.3
            if p.get("country_of_origin") and norm_text(p.get("country_of_origin")) in text:
                s += 1.0
            for t in p.get("taste") or []:
                if norm_text(t) in text:
                    s += 0.8
            for c in p.get("nutritional_characteristics") or []:
                if norm_text(c).replace("_", " ") in text or norm_text(c) in text:
                    s += 0.8
            if s:
                scored.append((s, p.get("name")))
        names = [n for _, n in sorted(scored, reverse=True)[:8]]
    if not names:
        names = _retail_product_names(db)[:5]

    # Branch/ranking pools from explicit instruction constraints.  This is not
    # a broad leading scan candidate; it is a DB-constrained program candidate.
    pools: List[Tuple[str, List[str], List[Dict[str, Any]]]] = []
    rows = _retail_products(db)
    if "france" in text:
        rows = [r for r in rows if norm_text(r.get("country_of_origin")) == "france"]
    if "italy" in text:
        rows = [r for r in rows if norm_text(r.get("country_of_origin")) == "italy"]
    if "low fat" in text or "low_fat" in text:
        rows = [r for r in rows if "low_fat" in [norm_text(x) for x in r.get("nutritional_characteristics", [])]]
    if "low sugar" in text or "low_sugar" in text:
        rows = [r for r in rows if "low_sugar" in [norm_text(x) for x in r.get("nutritional_characteristics", [])]]
    if "lowest sugar" in text and rows:
        m = min(_nutrition_val(r, "sugar_g") for r in rows)
        pools.append(("lowest_sugar_pool", [r["name"] for r in rows if _nutrition_val(r, "sugar_g") == m], rows))
    elif "highest sugar" in text and rows:
        m = max(_nutrition_val(r, "sugar_g") for r in rows)
        pools.append(("highest_sugar_pool", [r["name"] for r in rows if _nutrition_val(r, "sugar_g") == m], rows))
    elif "lowest" in text and "fat" in text and rows:
        m = min(_nutrition_val(r, "fat_g") for r in rows)
        pools.append(("lowest_fat_pool", [r["name"] for r in rows if _nutrition_val(r, "fat_g") == m], rows))
    elif "cheapest" in text or "lowest price" in text:
        pool_rows = [_product_by_name(db, n) for n in names]
        pool_rows = [r for r in pool_rows if r]
        if pool_rows:
            m = min(float(r.get("price", 0) or 0) for r in pool_rows)
            pools.append(("lowest_price_visual_pool", [r["name"] for r in pool_rows if float(r.get("price", 0) or 0) == m], pool_rows))

    target_sets = [("primary", names[:3])]
    for label, ns, _ in pools:
        if ns:
            target_sets.append((label, ns[:5]))

    out: List[Dict[str, Any]] = []
    closure = required_closure(instr, "retail")
    attr_tools = []
    if any(x in text for x in ("taste", "sweet", "bitter", "sour")):
        attr_tools.append("get_taste")
    if any(x in text for x in ("nutrition", "sugar", "fat", "calorie", "calcium")):
        attr_tools.append("get_nutrition")
    if "price" in text or "cheapest" in text:
        attr_tools.append("get_price")
    if "discount" in text:
        attr_tools.append("get_discount")
    if "tax" in text:
        attr_tools.append("get_tax_rate")

    for label, targets in target_sets:
        if not targets:
            continue
        # Query-only branch.
        prog: List[Dict[str, Any]] = []
        for name in targets:
            for tool in attr_tools[:3]:
                prog.append({"tool_name": tool, "parameters": {"product_name": name}})
        if prog:
            out.append(candidate(f"V24_RETAIL_QUERY_{label}", prog, {"targets": targets}, 0.5))

        prog = []
        for name in targets:
            for tool in attr_tools[:2]:
                prog.append({"tool_name": tool, "parameters": {"product_name": name}})
            if has_mutation_intent(instr):
                r = _product_by_name(db, name)
                if r:
                    prog.append({
                        "tool_name": "add_to_cart",
                        "parameters": {
                            "user_id": user_id,
                            "product_name": r.get("name"),
                            "qty": 1,
                            "category": r.get("category", ""),
                            "price": r.get("price", 0),
                            "tax_rate": r.get("tax_rate", 0),
                            "discount": r.get("discount", 1),
                        },
                    })
        if closure:
            prog.append({"tool_name": closure, "parameters": {"user_id": user_id, "products": _retail_cart_products(db, user_id, targets)}})
        if prog:
            out.append(candidate(f"V24_RETAIL_MUTATE_{label}", prog, {"targets": targets, "closure": closure}, 0.8 if label != "primary" else 0.6))

    return dedupe_programs(out, max_candidates)


def _restaurant_dishes(db: Any) -> List[Dict[str, Any]]:
    rows = []
    for obj in getattr(db, "catalog", {}).values():
        row = as_dict(obj)
        row["_type"] = "dish"
        rows.append(row)
    for obj in getattr(db, "set_meals", {}).values():
        row = as_dict(obj)
        row["_type"] = "set_meal"
        rows.append(row)
    return rows


def _restaurant_names_by_type(db: Any, typ: str | None = None) -> List[str]:
    out = []
    for r in _restaurant_dishes(db):
        if typ and r.get("_type") != typ:
            continue
        out.append(r.get("name") or r.get("set_meal_name"))
    return [x for x in out if x]


def _dish_order_list(db: Any, user_id: str, names: List[str]) -> List[Dict[str, Any]]:
    out = []
    current = getattr(db, "user_orders", {}).get(user_id, {})
    for item in getattr(current, "values", lambda: [])():
        row = as_dict(item)
        nm = row.get("dish_name") or row.get("set_meal_name") or row.get("name")
        if nm:
            out.append({"dish_name": nm, "quantity": row.get("quantity", 1)})
    seen = {norm_text(x.get("dish_name")) for x in out}
    for name in names:
        if norm_text(name) not in seen:
            out.append({"dish_name": name, "quantity": 1})
    return out


def _find_restaurant_row(db: Any, name: str) -> Dict[str, Any]:
    for r in _restaurant_dishes(db):
        nm = r.get("name") or r.get("set_meal_name")
        if norm_text(nm) == norm_text(name):
            return r
    return {}


def restaurant_candidates(row: Dict[str, Any], db: Any, qwen: Dict[str, Any] | None = None, max_candidates: int = 40) -> List[Dict[str, Any]]:
    qwen = qwen or {}
    instr = row.get("Instruction", "")
    text = norm_text(instr + "\n" + str(row.get("image_description", "")) + "\n" + json.dumps(qwen, ensure_ascii=False))
    user_id = extract_user_id(instr)
    all_names = _restaurant_names_by_type(db)
    raw_names = row_values(row) + qwen_entities(qwen, {"dish", "set_meal", "food", "menu_item"})
    names = canonical_lookup(raw_names, all_names)
    if not names:
        scored = []
        for r in _restaurant_dishes(db):
            nm = r.get("name") or r.get("set_meal_name")
            s = token_score(nm, text)
            if r.get("category") and norm_text(r.get("category")) in text:
                s += 1.0
            for t in r.get("taste") or []:
                if norm_text(t) in text:
                    s += 0.6
            if s:
                scored.append((s, nm))
        names = [n for _, n in sorted(scored, reverse=True)[:8]]
    if not names:
        names = all_names[:5]

    out: List[Dict[str, Any]] = []
    closure = required_closure(instr, "restaurant")
    attr_tools = []
    if "price" in text or "cost" in text:
        attr_tools.append("get_dish_price")
    if any(x in text for x in ("nutrition", "protein", "sugar", "calorie")):
        attr_tools.append("get_dish_nutrition")
    if any(x in text for x in ("taste", "spicy", "sweet", "sour")):
        attr_tools.append("get_dish_taste_profile")
    if "allergen" in text:
        attr_tools.append("get_dish_allergens")
    if "discount" in text:
        attr_tools.append("get_dish_discount")

    for name in names[:8]:
        prog: List[Dict[str, Any]] = []
        is_meal = norm_text(name) in {norm_text(x) for x in _restaurant_names_by_type(db, "set_meal")}
        if is_meal:
            prog.append({"tool_name": "get_set_meal_details", "parameters": {"set_meal_name": name}})
            if has_mutation_intent(instr):
                prog.append({"tool_name": "add_set_meal_to_order", "parameters": {"user_id": user_id, "set_meal_name": name, "quantity": 1}})
        else:
            rowinfo = _find_restaurant_row(db, name)
            for tool in attr_tools[:3]:
                prog.append({"tool_name": tool, "parameters": {"dish_name": name}})
            if "set meal" in text:
                prog.append({"tool_name": "find_set_meals_containing_dish", "parameters": {"dish_name": name}})
            if has_mutation_intent(instr):
                prog.append({
                    "tool_name": "add_dish_to_order",
                    "parameters": {
                        "user_id": user_id,
                        "dish_name": name,
                        "quantity": 1,
                        "category": rowinfo.get("category", ""),
                        "price": rowinfo.get("price", 0),
                        "tax_rate": rowinfo.get("tax_rate", 0),
                        "discount": rowinfo.get("discount", 1),
                    },
                })
        if closure == "get_user_order_summary":
            prog.append({"tool_name": closure, "parameters": {"user_id": user_id}})
        elif closure:
            prog.append({"tool_name": closure, "parameters": {"user_id": user_id, "dishes": _dish_order_list(db, user_id, [name])}})
        if prog:
            out.append(candidate(f"V24_RESTAURANT_{'MEAL' if is_meal else 'DISH'}_{norm_text(name)[:16]}", prog, {"target": name, "is_set_meal": is_meal}, 0.7))
    return dedupe_programs(out, max_candidates)


def _order_catalog(db: Any, restaurant: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rest = getattr(db, "restaurants", {}).get(restaurant)
    if not rest:
        return [], []
    dishes = []
    for obj in rest.get("catalog", {}).values():
        row = as_dict(obj)
        row["_restaurant_name"] = restaurant
        dishes.append(row)
    meals = []
    for obj in rest.get("set_meals", {}).values():
        row = as_dict(obj)
        row["_restaurant_name"] = restaurant
        meals.append(row)
    return dishes, meals


def _order_restaurants(db: Any) -> List[str]:
    return list(getattr(db, "restaurants", {}).keys())


def _extract_restaurant(instruction: str, db: Any) -> str:
    text = norm_text(instruction)
    for r in _order_restaurants(db):
        if norm_text(r) in text:
            return r
    return _order_restaurants(db)[0] if _order_restaurants(db) else ""


def _order_row(dishes: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for r in dishes:
        if norm_text(r.get("name")) == norm_text(name):
            return r
    return {}


def _order_dishes_param(names: List[str]) -> List[Dict[str, Any]]:
    return [{"dish_name": n, "quantity": 1} for n in names]


def order_candidates(row: Dict[str, Any], db: Any, qwen: Dict[str, Any] | None = None, max_candidates: int = 40) -> List[Dict[str, Any]]:
    qwen = qwen or {}
    instr = row.get("Instruction", "")
    text = norm_text(instr + "\n" + str(row.get("image_description", "")) + "\n" + json.dumps(qwen, ensure_ascii=False))
    user_id = extract_user_id(instr)
    restaurant = _extract_restaurant(instr, db)
    dishes, meals = _order_catalog(db, restaurant)
    dish_names = [x.get("name") for x in dishes if x.get("name")]
    meal_names = [x.get("set_meal_name") or x.get("name") for x in meals if x.get("set_meal_name") or x.get("name")]
    raw = row_values(row) + qwen_entities(qwen, {"dish", "set_meal", "food", "menu_item"})
    names = canonical_lookup(raw, dish_names + meal_names)
    if not names:
        scored = []
        for r in dishes:
            s = token_score(r.get("name"), text)
            if r.get("category") and norm_text(r.get("category")) in text:
                s += 1.0
            for t in r.get("taste") or []:
                if norm_text(t) in text:
                    s += 0.7
            if s:
                scored.append((s, r.get("name")))
        for r in meals:
            nm = r.get("set_meal_name") or r.get("name")
            s = token_score(nm, text)
            if s:
                scored.append((s, nm))
        names = [n for _, n in sorted(scored, reverse=True)[:8]]
    if not names:
        names = (dish_names + meal_names)[:6]

    closure = required_closure(instr, "order") or ("compute_total_payment" if has_mutation_intent(instr) else "")
    out: List[Dict[str, Any]] = []
    for name in names[:8]:
        is_meal = norm_text(name) in {norm_text(x) for x in meal_names}
        prog: List[Dict[str, Any]] = [{"tool_name": "get_user_order_summary", "parameters": {"restaurant_name": restaurant, "user_id": user_id}}]
        if is_meal:
            prog.append({"tool_name": "get_set_meal_details", "parameters": {"restaurant_name": restaurant, "set_meal_name": name}})
            if has_mutation_intent(instr):
                prog.append({"tool_name": "add_set_meal_to_order", "parameters": {"restaurant_name": restaurant, "user_id": user_id, "set_meal_name": name, "quantity": 1}})
        else:
            rowinfo = _order_row(dishes, name)
            if any(x in text for x in ("price", "highest", "lowest", "cheapest")):
                prog.append({"tool_name": "get_dish_price", "parameters": {"restaurant_name": restaurant, "dish_name": name}})
            if any(x in text for x in ("nutrition", "sugar", "calorie", "protein")):
                prog.append({"tool_name": "get_dish_nutrition", "parameters": {"restaurant_name": restaurant, "dish_name": name}})
            if any(x in text for x in ("taste", "buttery", "aroma", "spicy", "sweet")):
                prog.append({"tool_name": "get_dish_taste_profile", "parameters": {"restaurant_name": restaurant, "dish_name": name}})
            if "set meal" in text:
                prog.append({"tool_name": "find_set_meals_containing_dish", "parameters": {"restaurant_name": restaurant, "dish_name": name}})
            if has_mutation_intent(instr):
                if "remove" in text:
                    prog.append({"tool_name": "remove_dish_from_order", "parameters": {"restaurant_name": restaurant, "user_id": user_id, "dish_name": name, "quantity": 1}})
                else:
                    prog.append({
                        "tool_name": "add_dish_to_order",
                        "parameters": {
                            "restaurant_name": restaurant,
                            "user_id": user_id,
                            "dish_name": name,
                            "quantity": 1,
                            "category": rowinfo.get("category", ""),
                            "price": rowinfo.get("price", 0),
                            "tax_rate": rowinfo.get("tax_rate", 0),
                            "discount": rowinfo.get("discount", 1),
                        },
                    })
        if closure == "get_user_order_summary":
            prog.append({"tool_name": closure, "parameters": {"restaurant_name": restaurant, "user_id": user_id}})
        elif closure:
            prog.append({"tool_name": closure, "parameters": {"restaurant_name": restaurant, "user_id": user_id, "dishes": _order_dishes_param([name])}})
        out.append(candidate(f"V24_ORDER_{'MEAL' if is_meal else 'DISH'}_{norm_text(name)[:16]}", prog, {"restaurant": restaurant, "target": name, "is_set_meal": is_meal}, 0.75))

    # Branch-like category candidates.
    categories = list(dict.fromkeys([r.get("category") for r in dishes if r.get("category")]))
    for cat in categories[:6]:
        if norm_text(cat) in text or any(tok in text for tok in norm_text(cat).split()):
            prog = [
                {"tool_name": "find_dishes_by_category", "parameters": {"restaurant_name": restaurant, "category": cat}},
            ]
            out.append(candidate(f"V24_ORDER_CATEGORY_{norm_text(cat)[:16]}", prog, {"restaurant": restaurant, "category": cat}, 0.55))
    return dedupe_programs(out, max_candidates)


def _kitchen_recipe_names(db: Any) -> List[str]:
    return list(getattr(db, "recipes", {}).keys())


def _kitchen_ingredient_names(db: Any) -> List[str]:
    return list(getattr(db, "ingredients", {}).keys())


def kitchen_candidates(row: Dict[str, Any], db: Any, qwen: Dict[str, Any] | None = None, max_candidates: int = 40) -> List[Dict[str, Any]]:
    qwen = qwen or {}
    instr = row.get("Instruction", "")
    text = norm_text(instr + "\n" + str(row.get("image_description", "")) + "\n" + json.dumps(qwen, ensure_ascii=False))
    user_id = extract_user_id(instr)
    raw = row_values(row) + qwen_entities(qwen, {"recipe", "dish", "ingredient", "food"})
    recipes = canonical_lookup(raw, _kitchen_recipe_names(db))
    ingredients = canonical_lookup(raw, _kitchen_ingredient_names(db))
    if not recipes:
        scored = [(token_score(name, text), name) for name in _kitchen_recipe_names(db)]
        recipes = [n for s, n in sorted(scored, reverse=True)[:6] if s > 0]
    if not ingredients:
        scored = [(token_score(name, text), name) for name in _kitchen_ingredient_names(db)]
        ingredients = [n for s, n in sorted(scored, reverse=True)[:8] if s > 0]
    if not recipes:
        recipes = _kitchen_recipe_names(db)[:5]
    if not ingredients:
        ingredients = _kitchen_ingredient_names(db)[:5]

    out: List[Dict[str, Any]] = []
    closure = required_closure(instr, "kitchen")
    for recipe in recipes[:6]:
        prog: List[Dict[str, Any]] = [
            {"tool_name": "get_recipe_ingredients", "parameters": {"recipe_name": recipe}},
        ]
        if any(x in text for x in ("nutrition feature", "nutrition characteristic", "low sodium", "low_sodium")):
            prog.append({"tool_name": "get_recipe_nutritional_characteristics", "parameters": {"recipe_name": recipe}})
        if "allergen" in text:
            prog.append({"tool_name": "get_recipe_allergens", "parameters": {"recipe_name": recipe}})
        if "step" in text or "fewest preparation" in text:
            prog.append({"tool_name": "get_cooking_steps", "parameters": {"recipe_name": recipe}})
        if "add" in text and "menu" in text:
            prog.append({"tool_name": "add_recipe_to_menu", "parameters": {"user_id": user_id, "recipe_name": recipe}})
        if closure in {"tally_total_tastes", "tally_total_nutritional_characteristics"}:
            prog.append({"tool_name": closure, "parameters": {"user_id": user_id, "recipes": [recipe]}})
        out.append(candidate(f"V24_KITCHEN_RECIPE_{norm_text(recipe)[:16]}", prog, {"recipe": recipe}, 0.6))

    for ing in ingredients[:8]:
        prog = [
            {"tool_name": "find_ingredient_category", "parameters": {"ingredient_name": ing}},
            {"tool_name": "get_ingredient_nutrition", "parameters": {"ingredient_name": ing}},
        ]
        if "stock" in text or "quantity" in text or "zero" in text:
            prog.insert(1, {"tool_name": "get_ingredient_quantity", "parameters": {"ingredient_name": ing}})
        if "same location" in text or "stored" in text:
            prog.append({"tool_name": "get_ingredient_location", "parameters": {"ingredient_name": ing}})
        if "shopping list" in text and ("add" in text or "requirement" in text):
            qty = 500 if "500" in text else 200 if "200" in text else 1
            prog.append({"tool_name": "add_to_shopping_list", "parameters": {"user_id": user_id, "ingredient_name": ing, "quantity": qty}})
        if closure == "compute_total_nutritions":
            prog.append({"tool_name": closure, "parameters": {"user_id": user_id, "ingredients": [{"ingredient_name": ing, "quantity": 1}]}})
        out.append(candidate(f"V24_KITCHEN_ING_{norm_text(ing)[:16]}", prog, {"ingredient": ing}, 0.62))

    # Explicit constrained branch candidates from instruction words.
    if "egg/dairy/soy" in text or "egg dairy soy" in text:
        for cat in ["egg/dairy/soy product", "egg/dairy/soy products"]:
            prog = [{"tool_name": "get_ingredients_by_category", "parameters": {"category": cat}}]
            out.append(candidate(f"V24_KITCHEN_CAT_{norm_text(cat)[:16]}", prog, {"category": cat}, 0.55))
    if "soy allergen" in text:
        out.append(candidate("V24_KITCHEN_SOY_ALLERGEN", [{"tool_name": "find_recipes_by_allergen", "parameters": {"allergen": "soy"}}], {"allergen": "soy"}, 0.55))

    return dedupe_programs(out, max_candidates)


def generate_for_scenario(scenario: str, row: Dict[str, Any], db: Any, qwen: Dict[str, Any] | None = None, max_candidates: int = 40) -> List[Dict[str, Any]]:
    if scenario == "retail":
        return retail_candidates(row, db, qwen, max_candidates)
    if scenario == "restaurant":
        return restaurant_candidates(row, db, qwen, max_candidates)
    if scenario == "order":
        return order_candidates(row, db, qwen, max_candidates)
    if scenario == "kitchen":
        return kitchen_candidates(row, db, qwen, max_candidates)
    return []
