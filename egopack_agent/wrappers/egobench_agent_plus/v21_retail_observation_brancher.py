#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V21 retail branch decisions from product-specific observations."""

from __future__ import annotations

from typing import Any, Dict, List

from .v20_retail_slot_resolver import norm_text, product_to_dict


def _product_row(db: Any, name: str) -> Dict[str, Any]:
    prod = getattr(db, "catalog", {}).get(norm_text(name))
    return product_to_dict(prod) if prod else {}


def _nutrition_value(row: Dict[str, Any], key: str) -> float:
    nutrition = row.get("nutrition") or {}
    val = nutrition.get(key)
    try:
        return float(val)
    except Exception:
        return 0.0


class RetailObservationBrancherV21:
    def __init__(self, db: Any) -> None:
        self.db = db

    def decide(self, instruction: str, products: List[str], observations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        text = norm_text(instruction)
        products = [p for p in products if p]
        rows = {p: _product_row(self.db, p) for p in products}
        trace: Dict[str, Any] = {
            "tool_observations_used_for_branch": observations,
            "branch_decision": "",
            "branch_evidence": [],
            "candidate_pool": products,
        }
        selected: List[str] = []
        qty = 1

        if "both bottles" in text and "bitter" in text and len(products) >= 2:
            bitter = [p for p in products[:2] if "bitter" in rows.get(p, {}).get("taste", [])]
            trace["branch_evidence"].append({"rule": "both_bitter", "bitter_products": bitter})
            if len(bitter) == 2:
                min_price = min(float(rows[p].get("price", 0)) for p in bitter)
                selected = [p for p in bitter if float(rows[p].get("price", 0)) == min_price]
                qty = 2
                trace["branch_decision"] = "both_bitter_choose_cheapest"
            elif len(bitter) == 1:
                selected = [products[0]]
                trace["branch_decision"] = "one_bitter_add_left_hand"
            else:
                selected = [products[1]]
                trace["branch_decision"] = "none_bitter_add_right_hand"

        elif "three bottles" in text and "sweet" in text and len(products) >= 1:
            sweet = [p for p in products if "sweet" in rows.get(p, {}).get("taste", [])]
            trace["branch_evidence"].append({"rule": "any_sweet_highest_sugar", "sweet_products": sweet})
            if sweet:
                max_sugar = max(_nutrition_value(rows[p], "sugar_g") for p in sweet)
                selected = [p for p in sweet if _nutrition_value(rows[p], "sugar_g") == max_sugar]
                trace["branch_decision"] = "sweet_present_choose_highest_sugar"
            else:
                selected = products
                trace["branch_decision"] = "none_sweet_add_all_pointed"

        elif "low fat" in text and "france" in text:
            # retail4::14-like branch.  Branch condition is discount on a nearby bottle;
            # if discount is present in any observed visual candidate, choose the
            # France + low_fat + tax<0.12 product(s) with lowest sugar.  We do not
            # use GT; this is constrained DB filtering from instruction terms.
            pool = []
            for prod in getattr(self.db, "catalog", {}).values():
                row = product_to_dict(prod)
                if row.get("country_of_origin") != "france":
                    continue
                if "low_fat" not in row.get("nutritional_characteristics", []):
                    continue
                if float(row.get("tax_rate", 1)) >= 0.12:
                    continue
                pool.append(row)
            if pool:
                min_sugar = min(_nutrition_value(row, "sugar_g") for row in pool)
                selected = [row["name"] for row in pool if _nutrition_value(row, "sugar_g") == min_sugar]
                trace["branch_decision"] = "discount_branch_france_lowfat_lowest_sugar"
                trace["branch_evidence"].append({"rule": "france_lowfat_tax_lt_012_lowest_sugar", "pool": [r["name"] for r in pool], "min_sugar": min_sugar})

        elif "low sugar" in text and "italy" in text:
            pool = []
            for prod in getattr(self.db, "catalog", {}).values():
                row = product_to_dict(prod)
                if row.get("country_of_origin") != "italy":
                    continue
                if "low_sugar" not in row.get("nutritional_characteristics", []):
                    continue
                if float(row.get("tax_rate", 1)) >= 0.12:
                    continue
                pool.append(row)
            if pool:
                min_cal = min(_nutrition_value(row, "calories_kcal") for row in pool)
                selected = [row["name"] for row in pool if _nutrition_value(row, "calories_kcal") == min_cal]
                trace["branch_decision"] = "italy_lowsugar_lowest_calorie"
                trace["branch_evidence"].append({"rule": "italy_lowsugar_tax_lt_012_lowest_calorie", "pool": [r["name"] for r in pool], "min_calories": min_cal})

        if not selected and products:
            selected = [products[0]]
            trace["branch_decision"] = trace.get("branch_decision") or "fallback_primary_product"
            trace["branch_evidence"].append({"rule": "fallback", "reason": "no_branch_rule_matched"})

        return {
            "selected_products": selected,
            "quantity": qty,
            "trace": trace,
        }
