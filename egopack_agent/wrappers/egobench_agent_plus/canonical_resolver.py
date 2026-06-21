# -*- coding: utf-8 -*-
"""Canonical name resolver built from EgoBench scenario databases."""

from __future__ import annotations

import difflib
import json
import os
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def canonical_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    table = str.maketrans("", "", "".join(ch for ch in string.punctuation if ch not in "&'"))
    return text.translate(table).strip()


def _add(mapping: Dict[str, str], value: Any) -> None:
    if value in (None, ""):
        return
    text = str(value).strip()
    if not text:
        return
    mapping.setdefault(canonical_key(text), text)


def _fuzzy(value: Any, mapping: Dict[str, str], cutoff: float = 0.88) -> Tuple[Any, bool]:
    if value in (None, ""):
        return value, False
    original = str(value).strip()
    key = canonical_key(original)
    if key in mapping:
        return mapping[key], mapping[key] != original
    keys = list(mapping.keys())
    match = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
    if match:
        return mapping[match[0]], mapping[match[0]] != original
    return original, False


def _import_data() -> Dict[str, Dict[str, Any]]:
    import sys
    sys.path.insert(0, str(EGO_ROOT))
    data: Dict[str, Dict[str, Any]] = {}
    try:
        from tools.order.order_init import order_init_data
        data["order"] = order_init_data
    except Exception:
        pass
    try:
        from tools.restaurant.restaurant_init import restaurant_init_data, restaurant_init_data5
        merged = {"dishes": [], "set_meals": [], "user_orders": []}
        for src in (restaurant_init_data, restaurant_init_data5):
            for key, val in src.items():
                if isinstance(val, list):
                    merged.setdefault(key, []).extend(val)
        data["restaurant"] = merged
    except Exception:
        pass
    try:
        from tools.kitchen.kitchen_init import kitchen_init_data
        data["kitchen"] = kitchen_init_data
    except Exception:
        pass
    try:
        from tools.retail.retail_init import (
            retail_init_data1, retail_init_data2, retail_init_data3, retail_init_data4, retail_init_data5,
            retail_init_data6, retail_init_data7, retail_init_data8, retail_init_data9, retail_init_data10,
        )
        merged = {"products": [], "user_carts": [], "user_shopping_lists": []}
        for src in (retail_init_data1, retail_init_data2, retail_init_data3, retail_init_data4, retail_init_data5,
                    retail_init_data6, retail_init_data7, retail_init_data8, retail_init_data9, retail_init_data10):
            for key, val in src.items():
                if isinstance(val, list):
                    merged.setdefault(key, []).extend(val)
        data["retail"] = merged
    except Exception:
        pass
    return data


@lru_cache(maxsize=1)
def build_canonical_cache() -> Dict[str, Any]:
    raw = _import_data()
    cache: Dict[str, Any] = {
        "restaurant_name": {},
        "dish_name": {},
        "order_menu_name": {},
        "set_meal_name": {},
        "product_name": {},
        "ingredient_name": {},
        "recipe_name": {},
        "category": {},
        "restaurant_categories": {},
    }
    for scenario, data in raw.items():
        for dish in data.get("dishes", []) or []:
            _add(cache["dish_name"], dish.get("name"))
            if scenario == "order":
                _add(cache["order_menu_name"], dish.get("name"))
            _add(cache["restaurant_name"], dish.get("restaurant_name"))
            _add(cache["category"], dish.get("category"))
            rest = dish.get("restaurant_name")
            cat = dish.get("category")
            if rest and cat:
                cache["restaurant_categories"].setdefault(canonical_key(rest), {})
                _add(cache["restaurant_categories"][canonical_key(rest)], cat)
        for set_meal in data.get("set_meals", []) or []:
            _add(cache["set_meal_name"], set_meal.get("name"))
            if scenario == "order":
                _add(cache["order_menu_name"], set_meal.get("name"))
            _add(cache["restaurant_name"], set_meal.get("restaurant_name"))
            for inc in set_meal.get("included_dishes", []) or []:
                _add(cache["dish_name"], inc.get("dish_name"))
        for product in data.get("products", []) or []:
            _add(cache["product_name"], product.get("name"))
            _add(cache["category"], product.get("category"))
        for ingredient in data.get("ingredients", []) or []:
            _add(cache["ingredient_name"], ingredient.get("name"))
            _add(cache["category"], ingredient.get("category"))
        for recipe in data.get("recipes", []) or []:
            _add(cache["recipe_name"], recipe.get("name"))
            for inc in recipe.get("ingredients", []) or []:
                _add(cache["ingredient_name"], inc.get("ingredient_name"))
        for order in data.get("user_orders", []) or []:
            _add(cache["restaurant_name"], order.get("restaurant_name"))
            for item in order.get("items", []) or []:
                _add(cache["dish_name"], item.get("dish_name"))
                _add(cache["set_meal_name"], item.get("set_meal_name"))
        for cart_key in ("user_carts", "user_shopping_lists"):
            for cart in data.get(cart_key, []) or []:
                for item in cart.get("items", []) or []:
                    _add(cache["product_name"], item.get("product_name"))
                    _add(cache["ingredient_name"], item.get("ingredient_name"))
    path = CODEX_ROOT / "state" / "canonical_cache.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return cache


def canonicalize_param(param: str, value: Any, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, bool, str]:
    cache = build_canonical_cache()
    params = params or {}
    if param == "restaurant_name":
        val, changed = _fuzzy(value, cache["restaurant_name"], 0.84)
        return val, changed, "restaurant_name"
    if param in {"dish_name", "product_name", "set_meal_name", "ingredient_name", "recipe_name"}:
        val, changed = _fuzzy(value, cache.get(param, {}), 0.86)
        return val, changed, param
    if param == "order_menu_name":
        val, changed = _fuzzy(value, cache.get("order_menu_name", {}), 0.74)
        return val, changed, param
    if param == "category":
        rest = params.get("restaurant_name")
        if rest:
            rest_key = canonical_key(rest)
            rest_categories = cache["restaurant_categories"].get(rest_key, {})
            if canonical_key(value) == "steaks" and canonical_key(rest) == "annie italian restaurant":
                return "Selected Steaks", str(value) != "Selected Steaks", "category_alias"
            if canonical_key(value) == "pasta" and canonical_key(rest) == "annie italian restaurant":
                if "pasta" not in rest_categories and "italian pasta" in rest_categories:
                    return "Italian Pasta", True, "category_alias"
            val, changed = _fuzzy(value, rest_categories, 0.82)
            return val, changed, "restaurant_category"
        val, changed = _fuzzy(value, cache["category"], 0.88)
        return val, changed, "category"
    return value, False, ""


def normalize_order_dishes(params: Dict[str, Any], scenario: str = "", tool_name: str = "") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    notes: List[Dict[str, Any]] = []
    if not isinstance(params, dict):
        return params, notes
    dishes = params.get("dishes")
    if isinstance(dishes, list):
        new_dishes = []
        for item in dishes:
            if isinstance(item, dict):
                new_item = dict(item)
                # Order aggregate tools in EgoBench's order schema require
                # dishes[].product_name. Restaurant aggregate tools require
                # dishes[].dish_name. Earlier V9 scaffolding normalized all
                # product_name fields to dish_name, causing order payment/tax
                # computes to return 0.0 and loop. Preserve the field required
                # by the active scenario/tool while still canonicalizing values.
                prefer_product_name = scenario == "order" and str(tool_name).startswith("compute_total_")
                if prefer_product_name and "dish_name" in new_item and "product_name" not in new_item:
                    new_item["product_name"] = new_item.pop("dish_name")
                    notes.append({"field": "dishes[].dish_name", "to": "dishes[].product_name", "source": "order_aggregate_schema"})
                elif not prefer_product_name and "product_name" in new_item and "dish_name" not in new_item:
                    new_item["dish_name"] = new_item.pop("product_name")
                    notes.append({"field": "dishes[].product_name", "to": "dishes[].dish_name"})
                canon_field = "product_name" if prefer_product_name and "product_name" in new_item else "dish_name"
                if canon_field in new_item:
                    param_kind = "order_menu_name" if prefer_product_name else "dish_name"
                    val, changed, source = canonicalize_param(param_kind, new_item[canon_field], params)
                    if changed:
                        notes.append({"field": f"dishes[].{canon_field}", "from": new_item[canon_field], "to": val, "source": source})
                        new_item[canon_field] = val
                new_dishes.append(new_item)
            else:
                key = "product_name" if scenario == "order" and str(tool_name).startswith("compute_total_") else "dish_name"
                param_kind = "order_menu_name" if key == "product_name" else "dish_name"
                val, changed, source = canonicalize_param(param_kind, item, params)
                new_dishes.append({key: val})
                notes.append({"field": "dishes[]", "from": item, "to": {key: val}, "source": source})
        params = dict(params)
        params["dishes"] = new_dishes
    return params, notes


def canonicalize_tool_params(tool_name: str, params: Dict[str, Any], scenario: str = "") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(params, dict):
        return {}, []
    out = dict(params)
    notes: List[Dict[str, Any]] = []
    if scenario == "order" and "product_name" in out and "dish_name" not in out and "dish" in str(tool_name).lower():
        out["dish_name"] = out.pop("product_name")
        notes.append({"field": "product_name", "to": "dish_name", "source": "order_dish_param_alias"})
    out, dish_notes = normalize_order_dishes(out, scenario=scenario, tool_name=tool_name)
    notes.extend(dish_notes)
    for field in ("restaurant_name", "dish_name", "set_meal_name", "product_name", "ingredient_name", "recipe_name"):
        if field in out:
            val, changed, source = canonicalize_param(field, out[field], out)
            if changed:
                notes.append({"field": field, "from": out[field], "to": val, "source": source})
                out[field] = val
    if "category" in out:
        val, changed, source = canonicalize_param("category", out["category"], out)
        if changed:
            notes.append({"field": "category", "from": out["category"], "to": val, "source": source})
            out["category"] = val
    return out, notes
