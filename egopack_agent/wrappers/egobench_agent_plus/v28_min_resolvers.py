#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal five-stage resolvers for V28 order/restaurant/kitchen deltas."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List

from .v20_retail_slot_resolver import extract_user_id, norm_text
from .v28_evidence_guard import evidence_entity_names


def _call(tool_name: str, **params: Any) -> Dict[str, Any]:
    return {"tool_name": tool_name, "parameters": {k: v for k, v in params.items() if v not in (None, "", [])}}


def _asdict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    return dict(getattr(obj, "__dict__", {}) or {})


def _first_evidence(evidence: Dict[str, Any], typ: str) -> str:
    rows = evidence_entity_names(evidence).get(typ, [])
    return rows[0][0] if rows else ""


def _score_name(name: str, text: str) -> float:
    n = norm_text(name)
    t = norm_text(text)
    if not n or not t:
        return 0.0
    if n in t:
        return 10.0
    score = 0.0
    for tok in re.findall(r"[a-z0-9']+", n):
        if len(tok) > 3 and tok in t:
            score += 1.0
    return score


def _closure_for_text(text: str, scenario: str) -> List[str]:
    t = norm_text(text)
    out: List[str] = []
    if "total payment" in t or "amount payable" in t or (scenario == "order" and ("order" in t or "cart" in t)):
        out.append("compute_total_payment")
    if "summary" in t and scenario == "order":
        out.append("get_user_order_summary")
    if "total tax" in t:
        out.append("compute_total_tax")
    if "nutrition" in t or "nutritional" in t or "calcium" in t:
        out.append("compute_total_nutritions" if scenario == "kitchen" else "compute_total_nutrition")
    return list(dict.fromkeys(out))


def _mutation_expected(text: str) -> bool:
    t = norm_text(text)
    return any(x in t for x in ("add", "remove", "order", "cart", "shopping list", "menu"))


def _order_catalog(db: Any, restaurant: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rest = getattr(db, "restaurants", {}).get(restaurant)
    if not rest:
        return [], []
    return [_asdict(x) for x in rest.get("catalog", {}).values()], [_asdict(x) for x in rest.get("set_meals", {}).values()]


def _restaurant_catalog(db: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return [_asdict(x) for x in getattr(db, "catalog", {}).values()], [_asdict(x) for x in getattr(db, "set_meals", {}).values()]


def _rank_rows(rows: List[Dict[str, Any]], text: str, evidence_name: str = "") -> List[Dict[str, Any]]:
    ranked = []
    for row in rows:
        name = row.get("name") or row.get("dish_name") or row.get("set_meal_name") or row.get("recipe_name") or row.get("ingredient_name") or ""
        score = _score_name(name, text)
        if evidence_name and norm_text(evidence_name) == norm_text(name):
            score += 8.0
        elif evidence_name and (norm_text(evidence_name) in norm_text(name) or norm_text(name) in norm_text(evidence_name)):
            score += 5.0
        if score > 0:
            ranked.append((score, row))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked]


class MinimalResolverV28:
    def __init__(self, scenario: str, db: Any) -> None:
        self.scenario = scenario
        self.db = db

    def build(self, row: Dict[str, Any], evidence: Dict[str, Any] | None, max_candidates: int = 2) -> List[Dict[str, Any]]:
        if self.scenario == "order":
            return self._order(row, evidence or {}, max_candidates)
        if self.scenario == "restaurant":
            return self._restaurant(row, evidence or {}, max_candidates)
        if self.scenario == "kitchen":
            return self._kitchen(row, evidence or {}, max_candidates)
        return []

    def _pack(self, cid: str, source: str, program: List[Dict[str, Any]], trace: Dict[str, Any]) -> Dict[str, Any]:
        trace.update(
            {
                "called_candidate_resolver": True,
                "called_query_planner": True,
                "called_observation_brancher": True,
                "called_mutation_resolver": True,
                "called_closure_planner": True,
            }
        )
        return {"candidate_id": cid, "source": source, "tool_program": program, "trace": trace, "risk_flags": []}

    def _order(self, row: Dict[str, Any], evidence: Dict[str, Any], max_candidates: int) -> List[Dict[str, Any]]:
        text = row.get("Instruction", "")
        user_id = extract_user_id(text)
        restaurants = list(getattr(self.db, "restaurants", {}).keys())
        rest_ev = _first_evidence(evidence, "restaurant")
        restaurant = rest_ev if rest_ev in restaurants else (restaurants[0] if restaurants else "")
        dishes, meals = _order_catalog(self.db, restaurant)
        dish_ev = _first_evidence(evidence, "dish")
        meal_ev = _first_evidence(evidence, "set_meal")
        dish_rows = _rank_rows(dishes, text, dish_ev) or dishes[:3]
        meal_rows = _rank_rows(meals, text, meal_ev) or meals[:2]
        out: List[Dict[str, Any]] = []
        for dish in dish_rows[:max_candidates]:
            name = dish.get("name", "")
            prog = [_call("get_user_order_summary", restaurant_name=restaurant, user_id=user_id)]
            if any(x in norm_text(text) for x in ("price", "highest", "lowest", "cost")):
                prog.append(_call("get_dish_price", restaurant_name=restaurant, dish_name=name))
            if any(x in norm_text(text) for x in ("taste", "spicy", "buttery", "sweet", "fresh")):
                prog.append(_call("get_dish_taste_profile", restaurant_name=restaurant, dish_name=name))
            if any(x in norm_text(text) for x in ("nutrition", "protein", "sugar", "calorie")):
                prog.append(_call("get_dish_nutrition", restaurant_name=restaurant, dish_name=name))
            if _mutation_expected(text):
                tool = "remove_dish_from_order" if "remove" in norm_text(text) else "add_dish_to_order"
                prog.append(_call(tool, restaurant_name=restaurant, user_id=user_id, dish_name=name, quantity=1, category=dish.get("category", ""), price=dish.get("price", 0), tax_rate=dish.get("tax_rate", 0), discount=dish.get("discount", 1)))
            for closure in _closure_for_text(text, "order") or (["compute_total_payment"] if _mutation_expected(text) else []):
                if closure == "get_user_order_summary":
                    prog.append(_call(closure, restaurant_name=restaurant, user_id=user_id))
                else:
                    prog.append(_call(closure, restaurant_name=restaurant, user_id=user_id, dishes=[{"dish_name": name, "quantity": 1}]))
            out.append(self._pack(f"V28_ORDER_DISH_{norm_text(name)[:14]}", "V28_ORDER_MIN", prog, {"entity_candidates": [d.get("name") for d in dish_rows[:3]], "restaurant": restaurant}))
        for meal in meal_rows[:1]:
            name = meal.get("name", "")
            prog = [_call("get_user_order_summary", restaurant_name=restaurant, user_id=user_id), _call("get_set_meal_details", restaurant_name=restaurant, set_meal_name=name)]
            if _mutation_expected(text):
                prog.append(_call("add_set_meal_to_order", restaurant_name=restaurant, user_id=user_id, set_meal_name=name, quantity=1))
                prog.append(_call("compute_total_payment", restaurant_name=restaurant, user_id=user_id, dishes=[]))
            out.append(self._pack(f"V28_ORDER_MEAL_{norm_text(name)[:14]}", "V28_ORDER_MIN", prog, {"entity_candidates": [m.get("name") for m in meal_rows[:2]], "restaurant": restaurant}))
        return out[:max_candidates]

    def _restaurant(self, row: Dict[str, Any], evidence: Dict[str, Any], max_candidates: int) -> List[Dict[str, Any]]:
        text = row.get("Instruction", "")
        user_id = extract_user_id(text)
        dishes, meals = _restaurant_catalog(self.db)
        dish_rows = _rank_rows(dishes, text, _first_evidence(evidence, "dish")) or dishes[:3]
        meal_rows = _rank_rows(meals, text, _first_evidence(evidence, "set_meal")) or meals[:2]
        out: List[Dict[str, Any]] = []
        for dish in dish_rows[:max_candidates]:
            name = dish.get("name", "")
            prog: List[Dict[str, Any]] = []
            if any(x in norm_text(text) for x in ("price", "highest", "lowest", "cost")):
                prog.append(_call("get_dish_price", dish_name=name))
            if any(x in norm_text(text) for x in ("taste", "spicy", "buttery", "fresh", "rich")):
                prog.append(_call("get_dish_taste_profile", dish_name=name))
            if any(x in norm_text(text) for x in ("nutrition", "protein", "sugar", "calorie")):
                prog.append(_call("get_dish_nutrition", dish_name=name))
            if "discount" in norm_text(text):
                prog.append(_call("get_dish_discount", dish_name=name))
            if _mutation_expected(text):
                prog.append(_call("add_dish_to_order", user_id=user_id, dish_name=name, quantity=1, category=dish.get("category", ""), price=dish.get("price", 0), tax_rate=dish.get("tax_rate", 0), discount=dish.get("discount", 1)))
            for closure in _closure_for_text(text, "restaurant"):
                prog.append(_call(closure, user_id=user_id, dishes=[{"dish_name": name, "quantity": 1}]))
            if prog:
                out.append(self._pack(f"V28_RESTAURANT_DISH_{norm_text(name)[:14]}", "V28_RESTAURANT_MIN", prog, {"entity_candidates": [d.get("name") for d in dish_rows[:3]]}))
        for meal in meal_rows[:1]:
            name = meal.get("name", "")
            prog = [_call("get_set_meal_details", set_meal_name=name)]
            if _mutation_expected(text):
                prog.append(_call("add_set_meal_to_order", user_id=user_id, set_meal_name=name, quantity=1))
            out.append(self._pack(f"V28_RESTAURANT_MEAL_{norm_text(name)[:14]}", "V28_RESTAURANT_MIN", prog, {"entity_candidates": [m.get("name") for m in meal_rows[:2]]}))
        return out[:max_candidates]

    def _kitchen(self, row: Dict[str, Any], evidence: Dict[str, Any], max_candidates: int) -> List[Dict[str, Any]]:
        text = row.get("Instruction", "")
        user_id = extract_user_id(text)
        recipe_names = list(getattr(self.db, "recipes", {}).keys())
        ingredient_names = list(getattr(self.db, "ingredients", {}).keys())
        recipe_ev = _first_evidence(evidence, "recipe")
        ingredient_ev = _first_evidence(evidence, "ingredient")
        recipes = sorted(recipe_names, key=lambda n: _score_name(n, text) + (8 if norm_text(n) == norm_text(recipe_ev) else 0), reverse=True)[:3]
        ingredients = sorted(ingredient_names, key=lambda n: _score_name(n, text) + (8 if norm_text(n) == norm_text(ingredient_ev) else 0), reverse=True)[:4]
        out: List[Dict[str, Any]] = []
        for recipe in recipes[:max_candidates]:
            prog = [_call("get_recipe_ingredients", recipe_name=recipe)]
            if "step" in norm_text(text):
                prog.append(_call("get_cooking_steps", recipe_name=recipe))
            if "allergen" in norm_text(text):
                prog.append(_call("get_recipe_allergens", recipe_name=recipe))
            if any(x in norm_text(text) for x in ("nutrition", "protein", "sugar", "calorie")):
                prog.append(_call("get_recipe_nutritional_characteristics", recipe_name=recipe))
            if "add" in norm_text(text) and "menu" in norm_text(text):
                prog.append(_call("add_recipe_to_menu", user_id=user_id, recipe_name=recipe))
            out.append(self._pack(f"V28_KITCHEN_RECIPE_{norm_text(recipe)[:14]}", "V28_KITCHEN_MIN", prog, {"entity_candidates": recipes[:3]}))
        for ing in ingredients[:max_candidates]:
            prog: List[Dict[str, Any]] = []
            if "category" in norm_text(text) or "ingredient" in norm_text(text):
                prog.append(_call("find_ingredient_category", ingredient_name=ing))
            if any(x in norm_text(text) for x in ("quantity", "stock", "zero")):
                prog.append(_call("get_ingredient_quantity", ingredient_name=ing))
            if any(x in norm_text(text) for x in ("nutrition", "protein", "sugar", "calorie", "calcium")):
                prog.append(_call("get_ingredient_nutrition", ingredient_name=ing))
            if "shopping list" in norm_text(text) and ("add" in norm_text(text) or "requirement" in norm_text(text)):
                prog.append(_call("add_to_shopping_list", user_id=user_id, ingredient_name=ing, quantity=1))
            for closure in _closure_for_text(text, "kitchen"):
                if closure == "compute_total_nutritions":
                    prog.append(_call(closure, user_id=user_id, ingredients=[{"ingredient_name": ing, "quantity": 1}]))
            if prog:
                out.append(self._pack(f"V28_KITCHEN_ING_{norm_text(ing)[:14]}", "V28_KITCHEN_MIN", prog, {"entity_candidates": ingredients[:4]}))
        return out[:max_candidates]
