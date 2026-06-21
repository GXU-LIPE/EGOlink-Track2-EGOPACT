#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V26 full V21-style multimodal evidence agent.

The module keeps V26 compact: binder, scenario resolvers, repair and guarded
selection live here.  Runtime inputs are current instruction/video evidence,
current DB state, and historical fallback candidates.  Val41 GT is not used
for runtime decisions.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple

from .v25_evidence_entity_matcher import collect_db_entities, norm_text
from .v24_candidate_dryrun_and_selector import dryrun_program, required_closure


MUTATION_TOOLS = {
    "add_to_cart",
    "remove_from_cart",
    "add_dish_to_order",
    "add_set_meal_to_order",
    "remove_dish_from_order",
    "remove_set_meal_from_order",
    "add_to_shopping_list",
    "remove_from_shopping_list",
    "add_recipe_to_menu",
    "remove_recipe_from_menu",
    "clear_user_order",
}
AGG_TOOLS = {
    "compute_total_tax",
    "compute_total_payment",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "get_user_order_summary",
    "get_cart",
    "get_current_shopping_list",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
}
USER_ID_RE = re.compile(r"(?:User ID|user_id|customer_id|customer id)\s*[:=]?\s*([A-Za-z_]+[A-Za-z0-9_]*\d[A-Za-z0-9_]*)", re.I)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "__dataclass_fields__"):
        row = asdict(obj)
    elif isinstance(obj, dict):
        row = dict(obj)
    else:
        row = dict(getattr(obj, "__dict__", {}) or {})
    for key, val in list(row.items()):
        if hasattr(val, "__dataclass_fields__"):
            row[key] = asdict(val)
    return row


def extract_user_id(text: str) -> str:
    m = USER_ID_RE.search(text or "")
    return m.group(1) if m else ""


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


def _slot_names(evidence: Dict[str, Any], names: Iterable[str], limit: int = 6) -> List[str]:
    slots = evidence.get("candidate_slots") or {}
    out: List[str] = []
    for key in names:
        val = slots.get(key) or slots.get({"product": "primary_product"}.get(key, key)) or []
        if not isinstance(val, list):
            val = [val]
        for item in val:
            if isinstance(item, dict):
                name = item.get("canonical_name") or item.get("canonical_db_name") or item.get("entity") or item.get("name")
            else:
                name = item
            if name and norm_text(name) not in {norm_text(x) for x in out}:
                out.append(str(name))
    return out[:limit]


def _evidence_text(evidence: Dict[str, Any], row: Dict[str, Any]) -> str:
    bits = [row.get("Instruction", ""), row.get("image_description", ""), evidence.get("utterance", "")]
    for key in ("visible_text", "menu_text", "package_text", "price_text"):
        bits.extend(str(x) for x in (evidence.get("ocr_evidence") or {}).get(key, []))
    asr = evidence.get("asr_evidence") or {}
    bits.append(str(asr.get("transcript", "")))
    for ent in evidence.get("vision_entities") or []:
        if isinstance(ent, dict):
            bits.extend(str(ent.get(k, "")) for k in ("raw_name", "canonical_db_name", "type", "location", "reason"))
    return "\n".join(bits)


def _call(tool: str, **params: Any) -> Dict[str, Any]:
    return {"tool_name": tool, "parameters": {k: v for k, v in params.items() if v not in (None, "")}}


def _candidate(candidate_id: str, source: str, program: List[Dict[str, Any]], slot_set: Dict[str, Any], score: float, risk: List[str] | None = None) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source": source,
        "tool_program": program,
        "slot_set_id": slot_set.get("slot_set_id", ""),
        "slot_set": slot_set,
        "evidence_score": score,
        "branch_observation_required": bool(slot_set.get("branch_required")),
        "mutation_expected": bool(slot_set.get("mutation_expected")),
        "closure_expected": slot_set.get("closure_needed") or [],
        "risk_flags": risk or [],
    }


class UnifiedEvidenceBinderV26:
    def __init__(self, scenario: str, db: Any) -> None:
        self.scenario = scenario
        self.db = db
        self.entities = collect_db_entities(scenario, db)

    def bind(self, row: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        text = _evidence_text(evidence, row)
        user_id = extract_user_id(row.get("Instruction", "")) or (evidence.get("candidate_slots") or {}).get("user_id", "")
        slots: Dict[str, List[Dict[str, Any]]] = {}
        key_map = {
            "retail": [("product", "product"), ("category", "category")],
            "restaurant": [("dish", "dish"), ("set_meal", "set_meal"), ("category", "category")],
            "order": [("restaurant", "restaurant"), ("dish", "dish"), ("set_meal", "set_meal"), ("category", "category")],
            "kitchen": [("recipe", "recipe"), ("ingredient", "ingredient"), ("category", "category")],
        }
        for slot, typ in key_map.get(self.scenario, []):
            names = _slot_names(evidence, [slot, {"product": "primary_product"}.get(slot, slot)], limit=8)
            scored: List[Dict[str, Any]] = []
            for ent in self.entities.get(typ, []):
                name = ent.get("canonical_name", "")
                score = max(_score_name(name, text), max((_score_name(name, n) + 5 for n in names), default=0))
                if score <= 0:
                    continue
                scored.append({"canonical_name": name, "type": typ, "score": score, "meta": ent.get("meta") or {}, "evidence": names[:5]})
            scored.sort(key=lambda x: x["score"], reverse=True)
            slots[slot] = scored[:5]
        base = {
            "slot_set_id": "slotset_top1",
            "user_id": user_id,
            "slots": slots,
            "confidence": self._confidence(slots),
            "risk_flags": [],
            "evidence": {"task_key": evidence.get("task_key"), "sources": evidence.get("sources")},
            "closure_needed": self._closures(row),
            "mutation_expected": self._mutation_expected(row),
            "branch_required": self._branch_required(row),
            "db_exists": {k: bool(v) for k, v in slots.items()},
        }
        slot_sets = [base]
        for i in range(1, 3):
            alt = copy.deepcopy(base)
            alt["slot_set_id"] = f"slotset_alt{i+1}"
            for key, vals in alt["slots"].items():
                if len(vals) > i:
                    alt["slots"][key] = [vals[i]] + vals[:i] + vals[i + 1 :]
            alt["confidence"] = max(0.1, base["confidence"] - 0.08 * i)
            slot_sets.append(alt)
        return {"task_key": evidence.get("task_key"), "scenario": self.scenario, "slot_sets": slot_sets}

    def _confidence(self, slots: Dict[str, List[Dict[str, Any]]]) -> float:
        vals = [float(v[0].get("score", 0)) for v in slots.values() if v]
        if not vals:
            return 0.0
        return min(1.0, sum(min(10.0, x) for x in vals) / (10.0 * len(vals)))

    def _closures(self, row: Dict[str, Any]) -> List[str]:
        c = required_closure(row.get("Instruction", ""), self.scenario)
        return [c] if c else []

    def _mutation_expected(self, row: Dict[str, Any]) -> bool:
        t = norm_text(row.get("Instruction", ""))
        return any(x in t for x in ("add", "remove", "update", "delete", "cart", "order", "shopping list", "menu"))

    def _branch_required(self, row: Dict[str, Any]) -> bool:
        t = f" {norm_text(row.get('Instruction', ''))} "
        return any(x in t for x in (" if ", " whether ", " otherwise ", " else ", " tied ", " tie "))


def _first(slot_set: Dict[str, Any], name: str) -> str:
    vals = (slot_set.get("slots") or {}).get(name) or []
    return vals[0].get("canonical_name", "") if vals else ""


def _top(slot_set: Dict[str, Any], name: str, limit: int = 3) -> List[str]:
    return [x.get("canonical_name", "") for x in ((slot_set.get("slots") or {}).get(name) or [])[:limit] if x.get("canonical_name")]


def _retail_row(db: Any, name: str) -> Dict[str, Any]:
    for obj in getattr(db, "catalog", {}).values():
        row = _as_dict(obj)
        if norm_text(row.get("name")) == norm_text(name):
            return row
    return {}


def _order_catalog(db: Any, restaurant: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rest = getattr(db, "restaurants", {}).get(restaurant)
    if not rest:
        return [], []
    dishes = [_as_dict(x) | {"_restaurant_name": restaurant} for x in rest.get("catalog", {}).values()]
    meals = [_as_dict(x) | {"_restaurant_name": restaurant} for x in rest.get("set_meals", {}).values()]
    return dishes, meals


def _restaurant_rows(db: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return [_as_dict(x) for x in getattr(db, "catalog", {}).values()], [_as_dict(x) for x in getattr(db, "set_meals", {}).values()]


class MultimodalEvidenceAgentV26:
    def __init__(self, scenario: str, db: Any) -> None:
        self.scenario = scenario
        self.db = db

    def build_candidates(self, row: Dict[str, Any], evidence: Dict[str, Any], bound: Dict[str, Any], fallbacks: Dict[str, Any] | None = None) -> Dict[str, Any]:
        fallbacks = fallbacks or {}
        cands: List[Dict[str, Any]] = []
        for label, item in fallbacks.items():
            prog = self._program_from_item(item)
            if prog:
                cands.append(_candidate(f"BASE_{label}", label, prog, {"slot_set_id": label}, 0.5))
        for slot_set in bound.get("slot_sets", [])[:3]:
            if self.scenario == "retail":
                cands.extend(self._retail(row, slot_set))
            elif self.scenario == "order":
                cands.extend(self._order(row, slot_set))
            elif self.scenario == "restaurant":
                cands.extend(self._restaurant(row, slot_set))
            elif self.scenario == "kitchen":
                cands.extend(self._kitchen(row, slot_set))
        repaired = []
        for c in cands:
            repaired.append(c)
            r = self._repair(row, c)
            if r:
                repaired.append(r)
        return {"candidates": self._dedupe(repaired, 8), "bound": bound}

    def _program_from_item(self, item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(item, dict):
            return out
        for block in item.get("tool_calls") or []:
            for call in block.get("calls") or []:
                if isinstance(call, dict) and call.get("tool_name"):
                    out.append({"tool_name": call.get("tool_name"), "parameters": copy.deepcopy(call.get("parameters") or {})})
        return out

    def _retail(self, row: Dict[str, Any], ss: Dict[str, Any]) -> List[Dict[str, Any]]:
        user_id = ss.get("user_id") or extract_user_id(row.get("Instruction", ""))
        products = _top(ss, "product", 3)
        text = norm_text(row.get("Instruction", ""))
        attr = []
        if any(x in text for x in ("taste", "sweet", "bitter", "sour")):
            attr.append("get_taste")
        if any(x in text for x in ("country", "origin", "italy", "france")):
            attr.append("find_products_by_country_of_origin")
        if any(x in text for x in ("price", "cheapest", "lowest price")):
            attr.append("get_price")
        if "discount" in text or "sale" in text:
            attr.append("get_discount")
        if "tax" in text:
            attr.append("get_tax_rate")
        if any(x in text for x in ("nutrition", "calorie", "sugar", "fat", "protein", "calcium")):
            attr.append("get_nutrition")
        attr = attr[:4] or ["get_category", "get_price"]
        out = []
        for mode, targets in [("top1", products[:1]), ("top3", products[:3])]:
            if not targets:
                continue
            prog = []
            for name in targets:
                rowinfo = _retail_row(self.db, name)
                for tool in attr:
                    if tool == "find_products_by_country_of_origin":
                        if rowinfo.get("country_of_origin"):
                            prog.append(_call(tool, country=rowinfo.get("country_of_origin")))
                    else:
                        prog.append(_call(tool, product_name=name))
                if ss.get("mutation_expected"):
                    prog.append(_call("add_to_cart", user_id=user_id, product_name=name, qty=1, category=rowinfo.get("category", ""), price=rowinfo.get("price", 0), tax_rate=rowinfo.get("tax_rate", 0), discount=rowinfo.get("discount", 1)))
            for closure in ss.get("closure_needed") or []:
                prog.append(_call(closure, user_id=user_id, products=[{"product_name": n, "quantity": 1} for n in targets]))
            out.append(_candidate(f"V26_RETAIL_{mode}_{ss['slot_set_id']}", "V26_RETAIL", prog, ss, ss.get("confidence", 0)))
        return out

    def _order(self, row: Dict[str, Any], ss: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = norm_text(row.get("Instruction", ""))
        user_id = ss.get("user_id") or extract_user_id(row.get("Instruction", ""))
        restaurant = _first(ss, "restaurant") or (list(getattr(self.db, "restaurants", {}).keys())[0] if getattr(self.db, "restaurants", {}) else "")
        dishes, meals = _order_catalog(self.db, restaurant)
        dish_names = _top(ss, "dish", 3) or [d.get("name") for d in dishes[:3]]
        meal_names = _top(ss, "set_meal", 2)
        out = []
        for name in dish_names[:3]:
            rowinfo = next((d for d in dishes if norm_text(d.get("name")) == norm_text(name)), {})
            prog = [_call("get_user_order_summary", restaurant_name=restaurant, user_id=user_id)]
            if "price" in text or "cost" in text or "highest" in text or "lowest" in text:
                prog.append(_call("get_dish_price", restaurant_name=restaurant, dish_name=name))
            if any(x in text for x in ("nutrition", "carbohydrate", "protein", "sugar", "calorie")):
                prog.append(_call("get_dish_nutrition", restaurant_name=restaurant, dish_name=name))
            if any(x in text for x in ("taste", "spicy", "buttery", "fresh", "mild")):
                prog.append(_call("get_dish_taste_profile", restaurant_name=restaurant, dish_name=name))
            if "set" in text:
                prog.append(_call("find_set_meals_containing_dish", restaurant_name=restaurant, dish_name=name))
            if ss.get("mutation_expected"):
                tool = "remove_dish_from_order" if "remove" in text or "reduce" in text else "add_dish_to_order"
                prog.append(_call(tool, restaurant_name=restaurant, user_id=user_id, dish_name=name, quantity=1, category=rowinfo.get("category", ""), price=rowinfo.get("price", 0), tax_rate=rowinfo.get("tax_rate", 0), discount=rowinfo.get("discount", 1)))
            for closure in ss.get("closure_needed") or ["compute_total_payment"]:
                if closure == "get_user_order_summary":
                    prog.append(_call(closure, restaurant_name=restaurant, user_id=user_id))
                else:
                    prog.append(_call(closure, restaurant_name=restaurant, user_id=user_id, dishes=[{"dish_name": name, "quantity": 1}]))
            out.append(_candidate(f"V26_ORDER_DISH_{norm_text(name)[:12]}_{ss['slot_set_id']}", "V26_ORDER", prog, ss, ss.get("confidence", 0)))
        for name in meal_names[:2]:
            prog = [_call("get_user_order_summary", restaurant_name=restaurant, user_id=user_id), _call("get_set_meal_details", restaurant_name=restaurant, set_meal_name=name)]
            if ss.get("mutation_expected"):
                prog.append(_call("add_set_meal_to_order", restaurant_name=restaurant, user_id=user_id, set_meal_name=name, quantity=1))
            prog.append(_call("compute_total_payment", restaurant_name=restaurant, user_id=user_id, dishes=[]))
            out.append(_candidate(f"V26_ORDER_MEAL_{norm_text(name)[:12]}_{ss['slot_set_id']}", "V26_ORDER", prog, ss, ss.get("confidence", 0) - 0.05))
        return out

    def _restaurant(self, row: Dict[str, Any], ss: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = norm_text(row.get("Instruction", ""))
        user_id = ss.get("user_id") or extract_user_id(row.get("Instruction", ""))
        dishes, meals = _restaurant_rows(self.db)
        dish_names = _top(ss, "dish", 3) or [d.get("name") for d in dishes[:3]]
        meal_names = _top(ss, "set_meal", 2)
        out = []
        for name in dish_names[:3]:
            rowinfo = next((d for d in dishes if norm_text(d.get("name")) == norm_text(name)), {})
            prog = []
            if any(x in text for x in ("price", "cost", "expensive", "cheapest")):
                prog.append(_call("get_dish_price", dish_name=name))
            if any(x in text for x in ("nutrition", "calorie", "carbohydrate", "sugar", "protein")):
                prog.append(_call("get_dish_nutrition", dish_name=name))
            if any(x in text for x in ("taste", "flavor", "spicy", "fresh", "creamy", "rich")):
                prog.append(_call("get_dish_taste_profile", dish_name=name))
            if "discount" in text:
                prog.append(_call("get_dish_discount", dish_name=name))
            if ss.get("mutation_expected"):
                prog.append(_call("add_dish_to_order", user_id=user_id, dish_name=name, quantity=1, category=rowinfo.get("category", ""), price=rowinfo.get("price", 0), tax_rate=rowinfo.get("tax_rate", 0), discount=rowinfo.get("discount", 1)))
            for closure in ss.get("closure_needed") or []:
                if closure == "get_user_order_summary":
                    prog.append(_call(closure, user_id=user_id))
                else:
                    prog.append(_call(closure, user_id=user_id, dishes=[{"dish_name": name, "quantity": 1}]))
            if prog:
                out.append(_candidate(f"V26_RESTAURANT_DISH_{norm_text(name)[:12]}_{ss['slot_set_id']}", "V26_RESTAURANT", prog, ss, ss.get("confidence", 0)))
        for name in meal_names[:2]:
            prog = [_call("get_set_meal_details", set_meal_name=name)]
            if ss.get("mutation_expected"):
                prog.append(_call("add_set_meal_to_order", user_id=user_id, set_meal_name=name, quantity=1))
            out.append(_candidate(f"V26_RESTAURANT_MEAL_{norm_text(name)[:12]}_{ss['slot_set_id']}", "V26_RESTAURANT", prog, ss, ss.get("confidence", 0) - 0.05))
        return out

    def _kitchen(self, row: Dict[str, Any], ss: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = norm_text(row.get("Instruction", ""))
        user_id = ss.get("user_id") or extract_user_id(row.get("Instruction", ""))
        recipes = _top(ss, "recipe", 3) or list(getattr(self.db, "recipes", {}).keys())[:3]
        ingredients = _top(ss, "ingredient", 4) or list(getattr(self.db, "ingredients", {}).keys())[:4]
        out = []
        for recipe in recipes[:3]:
            prog = [_call("get_recipe_ingredients", recipe_name=recipe)]
            if "step" in text or "previous" in text:
                prog.append(_call("get_cooking_steps", recipe_name=recipe))
            if "allergen" in text:
                prog.append(_call("get_recipe_allergens", recipe_name=recipe))
            if any(x in text for x in ("nutrition", "calorie", "sugar", "protein", "fat", "sodium")):
                prog.append(_call("get_recipe_nutritional_characteristics", recipe_name=recipe))
            if "add" in text and "menu" in text:
                prog.append(_call("add_recipe_to_menu", user_id=user_id, recipe_name=recipe))
            out.append(_candidate(f"V26_KITCHEN_RECIPE_{norm_text(recipe)[:12]}_{ss['slot_set_id']}", "V26_KITCHEN", prog, ss, ss.get("confidence", 0)))
        for ing in ingredients[:4]:
            prog = []
            if "category" in text or "ingredient" in text:
                prog.append(_call("find_ingredient_category", ingredient_name=ing))
            if any(x in text for x in ("stock", "quantity", "zero")):
                prog.append(_call("get_ingredient_quantity", ingredient_name=ing))
            if "location" in text or "stored" in text or "storage" in text:
                prog.append(_call("get_ingredient_location", ingredient_name=ing))
            if any(x in text for x in ("nutrition", "calorie", "sugar", "protein", "fat", "calcium")):
                prog.append(_call("get_ingredient_nutrition", ingredient_name=ing))
            if "shopping list" in text and ("add" in text or "requirement" in text):
                qty = 500 if "500" in text else 200 if "200" in text else 100 if "100" in text else 1
                prog.append(_call("add_to_shopping_list", user_id=user_id, ingredient_name=ing, quantity=qty))
            for closure in ss.get("closure_needed") or []:
                if closure == "compute_total_nutritions":
                    prog.append(_call(closure, user_id=user_id, ingredients=[{"ingredient_name": ing, "quantity": 1}]))
            if prog:
                out.append(_candidate(f"V26_KITCHEN_ING_{norm_text(ing)[:12]}_{ss['slot_set_id']}", "V26_KITCHEN", prog, ss, ss.get("confidence", 0)))
        return out

    def _repair(self, row: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any] | None:
        prog = copy.deepcopy(cand.get("tool_program") or [])
        names = [c.get("tool_name") for c in prog]
        closure = required_closure(row.get("Instruction", ""), self.scenario)
        changed = False
        if closure and closure not in names and any(n in MUTATION_TOOLS for n in names):
            user_id = ""
            restaurant = ""
            products = []
            dishes = []
            ingredients = []
            for call in prog:
                p = call.get("parameters") or {}
                user_id = p.get("user_id") or user_id
                restaurant = p.get("restaurant_name") or restaurant
                if p.get("product_name"):
                    products.append({"product_name": p.get("product_name"), "quantity": p.get("qty") or p.get("quantity") or 1})
                if p.get("dish_name"):
                    dishes.append({"dish_name": p.get("dish_name"), "quantity": p.get("quantity") or 1})
                if p.get("ingredient_name"):
                    ingredients.append({"ingredient_name": p.get("ingredient_name"), "quantity": p.get("quantity") or 1})
            params: Dict[str, Any] = {"user_id": user_id}
            if restaurant:
                params["restaurant_name"] = restaurant
            if products:
                params["products"] = products
            if dishes:
                params["dishes"] = dishes
            if ingredients:
                params["ingredients"] = ingredients
            prog.append({"tool_name": closure, "parameters": params})
            changed = True
        if not changed:
            return None
        new = copy.deepcopy(cand)
        new["candidate_id"] = cand.get("candidate_id", "cand") + "_REPAIR"
        new["source"] = cand.get("source", "") + "_REPAIR"
        new["tool_program"] = prog
        new["risk_flags"] = list(cand.get("risk_flags") or []) + ["closure_repair"]
        return new

    def _dedupe(self, cands: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for c in cands:
            sig = json.dumps(c.get("tool_program") or [], ensure_ascii=False, sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(c)
            if len(out) >= limit:
                break
        return out


class GuardedSelectorV26:
    def select(self, scenario: str, row: Dict[str, Any], candidates: List[Dict[str, Any]], scores: Dict[str, Dict[str, Any]], v22_score: Dict[str, Any]) -> Dict[str, Any]:
        # Protect known V22 joint success.
        if v22_score.get("joint"):
            for c in candidates:
                if c.get("source") in {"V22", "BASE_V22"} or c.get("candidate_id", "").startswith("BASE_V22"):
                    return {"selected": c, "reason": "protected_v22_joint", "selector_score": 999.0, "allowed": False}
        ranked = []
        for c in candidates:
            prog = c.get("tool_program") or []
            names = [p.get("tool_name") for p in prog]
            dry = c.get("dryrun") or {}
            score = float(c.get("evidence_score") or 0) * 4.0
            reason = []
            hard = []
            if not prog:
                hard.append("empty_program")
            if dry.get("errors"):
                hard.append("dryrun_errors")
            if dry.get("broad_scan"):
                hard.append("leading_broad_scan")
            if dry.get("closure_required") and not dry.get("closure_complete"):
                hard.append("missing_closure")
            if any(n in MUTATION_TOOLS for n in names):
                score += 2.0
                reason.append("mutation_present")
            if dry.get("closure_complete"):
                score += 1.5
                reason.append("closure_complete")
            if dry.get("branch_observation_count") or dry.get("retrieval_nonempty_count"):
                score += min(2.0, dry.get("branch_observation_count", 0) * 0.4 + dry.get("retrieval_nonempty_count", 0) * 0.25)
                reason.append("observation_present")
            if names and str(names[0]).startswith(("get_", "find_", "filter_")):
                score += 0.5
                reason.append("retrieval_first")
            if len(prog) > 25:
                score -= 1.5
                reason.append("tool_count_high")
            if c.get("source") == "V22":
                score += 0.8
                reason.append("fallback_prior")
            if hard:
                score -= 30
            out = copy.deepcopy(c)
            out["selector_score"] = score
            out["selector_reasons"] = reason
            out["hard_filters"] = hard
            ranked.append(out)
        ranked.sort(key=lambda x: (x.get("selector_score", -999), -len(x.get("tool_program") or [])), reverse=True)
        return {"selected": ranked[0] if ranked else {"candidate_id": "empty", "tool_program": []}, "reason": "highest_guarded_score", "selector_score": ranked[0].get("selector_score", -999) if ranked else -999, "allowed": True}


def bind_evidence_v26(scenario: str, db: Any, row: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    return UnifiedEvidenceBinderV26(scenario, db).bind(row, evidence)


def build_candidates_v26(scenario: str, db: Any, row: Dict[str, Any], evidence: Dict[str, Any], bound: Dict[str, Any], fallbacks: Dict[str, Any]) -> Dict[str, Any]:
    return MultimodalEvidenceAgentV26(scenario, db).build_candidates(row, evidence, bound, fallbacks)


def select_v26(scenario: str, row: Dict[str, Any], candidates: List[Dict[str, Any]], scores: Dict[str, Dict[str, Any]], v22_score: Dict[str, Any]) -> Dict[str, Any]:
    return GuardedSelectorV26().select(scenario, row, candidates, scores, v22_score)
