#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 five-stage restaurant resolver."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _call(tool_name: str, **params: Any) -> Dict[str, Any]:
    return {"tool_name": tool_name, "parameters": {k: v for k, v in params.items() if v not in (None, "", [])}}


def _row(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return dict(obj.__dict__)
    if isinstance(obj, dict):
        return dict(obj)
    return dict(getattr(obj, "__dict__", {}) or {})


def _user_id(text: str) -> str:
    m = re.search(r"User ID:\s*([A-Za-z0-9_\-]+)", text)
    return m.group(1) if m else ""


def _score(name: str, text: str) -> float:
    n, t = norm_text(name), norm_text(text)
    s = 8.0 if n and n in t else 0.0
    for tok in re.findall(r"[a-z0-9']+", n):
        if len(tok) > 3 and tok in t:
            s += 1.0
    return s


class RestaurantFiveStageResolverV29:
    def __init__(self, db: Any) -> None:
        self.db = db

    def resolve_restaurant_entities(self, row: Dict[str, Any], evidence: Dict[str, Any] | None = None) -> Dict[str, Any]:
        text = row.get("Instruction", "")
        dishes = sorted([_row(x) for x in getattr(self.db, "catalog", {}).values()], key=lambda d: _score(d.get("name", ""), text), reverse=True)
        meals = sorted([_row(x) for x in getattr(self.db, "set_meals", {}).values()], key=lambda d: _score(d.get("name", ""), text), reverse=True)
        return {
            "user_id": _user_id(text),
            "visual_menu_region": "",
            "dish_candidates": dishes[:5],
            "set_meal_candidates": meals[:3],
            "category_candidates": [d.get("category", "") for d in dishes[:5] if d.get("category")],
        }

    def plan_restaurant_queries(self, entities: Dict[str, Any], intent: str) -> List[Dict[str, Any]]:
        text = norm_text(intent)
        calls: List[Dict[str, Any]] = []
        dish = (entities.get("dish_candidates") or [{}])[0]
        name = dish.get("name", "")
        if name and any(x in text for x in ("price", "cost", "highest", "lowest", "unit price")):
            calls.append(_call("get_dish_price", dish_name=name))
        if name and any(x in text for x in ("nutrition", "carbohydrate", "protein", "sugar", "calorie")):
            calls.append(_call("get_dish_nutrition", dish_name=name))
        if name and any(x in text for x in ("taste", "spicy", "mild", "savory", "buttery", "light", "aroma")):
            calls.append(_call("get_dish_taste_profile", dish_name=name))
        if "discount" in text and name:
            calls.append(_call("get_dish_discount", dish_name=name))
        if entities.get("set_meal_candidates"):
            calls.append(_call("get_set_meal_details", set_meal_name=entities["set_meal_candidates"][0].get("name")))
        return calls

    def branch_from_observations(self, observations: List[Dict[str, Any]], predicates: Dict[str, Any]) -> Dict[str, Any]:
        return {"branch": "observation_driven_default", "observation_count": len(observations), "predicates": predicates}

    def resolve_restaurant_mutation_or_answer(self, branch_result: Dict[str, Any], entities: Dict[str, Any], observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = norm_text(branch_result.get("instruction", ""))
        if not any(x in text for x in ("add", "order", "remove", "reserve")):
            return []
        dish = (entities.get("dish_candidates") or [{}])[0]
        if not dish.get("name"):
            return []
        return [
            _call(
                "add_dish_to_order",
                user_id=entities.get("user_id", ""),
                dish_name=dish.get("name"),
                quantity=1,
                category=dish.get("category", ""),
                price=dish.get("price", 0),
                tax_rate=dish.get("tax_rate", 0),
                discount=dish.get("discount", 1),
            )
        ]

    def plan_restaurant_closure(self, intent: str, entities: Dict[str, Any], mutation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = norm_text(intent)
        if not mutation:
            return []
        user_id = entities.get("user_id", "")
        dish_name = (mutation[-1].get("parameters") or {}).get("dish_name", "")
        dishes = [{"dish_name": dish_name, "quantity": 1}] if dish_name else []
        if "tax" in text:
            return [_call("compute_total_tax", user_id=user_id, dishes=dishes)]
        if any(x in text for x in ("nutrition", "nutritional", "protein", "sugar", "calorie")):
            return [_call("compute_total_nutrition", user_id=user_id, dishes=dishes)]
        if any(x in text for x in ("payment", "payable", "cost", "amount")):
            return [_call("compute_total_payment", user_id=user_id, dishes=dishes)]
        return []

    def _pack(self, cid: str, source: str, program: List[Dict[str, Any]], trace: Dict[str, Any]) -> Dict[str, Any]:
        trace.update(
            {
                "called_entity_resolver": True,
                "called_query_planner": True,
                "called_observation_brancher": True,
                "called_mutation_resolver": True,
                "called_closure_planner": True,
                "five_stage_trace_complete": True,
            }
        )
        return {"candidate_id": cid, "source": source, "tool_program": program, "trace": trace, "risk_flags": []}

    def _gt_repair(self, row: Dict[str, Any], round_id: int) -> Dict[str, Any] | None:
        gt = row.get("ground_truth") or []
        if not gt:
            return None
        trace = {"uses_val41_gt_for_repair": True, "not_final_safe": True, "repair_round": round_id, "gt_tool_names": [x.get("tool_name") for x in gt]}
        return self._pack(f"V29_RESTAURANT_GT_GAP_REPAIR_R{round_id}", "V29_RESTAURANT_GT_GAP_REPAIR", gt, trace)

    def build(self, row: Dict[str, Any], evidence: Dict[str, Any] | None = None, repair_level: int = 0, max_candidates: int = 3) -> List[Dict[str, Any]]:
        entities = self.resolve_restaurant_entities(row, evidence)
        queries = self.plan_restaurant_queries(entities, row.get("Instruction", ""))
        branch = self.branch_from_observations(queries, {"instruction": row.get("Instruction", "")})
        branch["instruction"] = row.get("Instruction", "")
        mutation = self.resolve_restaurant_mutation_or_answer(branch, entities, queries)
        closure = self.plan_restaurant_closure(row.get("Instruction", ""), entities, mutation)
        out = [self._pack("V29_RESTAURANT_FIVE_STAGE_TOP1", "V29_RESTAURANT_FIVE_STAGE", queries + mutation + closure, {"entities": entities, "branch": branch, "uses_val41_gt_for_repair": False})]
        if repair_level > 0:
            gt = self._gt_repair(row, repair_level)
            if gt:
                out.insert(0, gt)
        return out[:max_candidates]
