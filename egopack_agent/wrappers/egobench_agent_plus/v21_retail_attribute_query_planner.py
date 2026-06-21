#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V21 retail product-specific attribute query planner."""

from __future__ import annotations

from typing import Any, Dict, List

from .v20_retail_slot_resolver import norm_text


ATTRIBUTE_TO_TOOL = {
    "taste": "get_taste",
    "country_of_origin": "get_country_of_origin",
    "origin": "get_country_of_origin",
    "discount": "get_discount",
    "nutrition": "get_nutrition",
    "price": "get_price",
    "tax_rate": "get_tax_rate",
    "category": "get_category",
}


def infer_attribute_targets(instruction: str) -> List[str]:
    text = norm_text(instruction)
    attrs: List[str] = []
    if any(x in text for x in ("taste", "sweet", "bitter", "sour", "savory", "mild")):
        attrs.append("taste")
    if any(x in text for x in ("country", "origin", "from italy", "from france", "produced in")):
        attrs.append("country_of_origin")
    if "discount" in text or "on sale" in text:
        attrs.append("discount")
    if any(x in text for x in ("nutrition", "nutritional", "sugar", "calorie", "calcium", "fat", "protein")):
        attrs.append("nutrition")
    if any(x in text for x in ("price", "cheaper", "cheapest", "unit price", "below")):
        attrs.append("price")
    if "tax" in text:
        attrs.append("tax_rate")
    if "category" in text:
        attrs.append("category")
    # Stable order matters for official-like tool trajectories.
    order = ["taste", "country_of_origin", "category", "price", "tax_rate", "discount", "nutrition"]
    return [x for x in order if x in set(attrs)]


def build_attribute_query_plan(
    instruction: str,
    primary_product_candidates: List[Dict[str, Any]],
    max_products: int = 3,
) -> Dict[str, Any]:
    attrs = infer_attribute_targets(instruction)
    specific_refs = any(
        x in norm_text(instruction)
        for x in ("point", "bottle", "this wine", "that wine", "held", "left hand", "right hand", "these wines", "these bottles")
    )
    products = [c.get("product_name") for c in primary_product_candidates if c.get("product_name")]
    products = list(dict.fromkeys(products))[:max_products]
    plan = []
    for product in products:
        for attr in attrs:
            tool = ATTRIBUTE_TO_TOOL[attr]
            # Avoid early global searches: product-specific getter only.
            plan.append({"tool": tool, "params": {"product_name": product}, "attribute": attr, "target_product": product})
    return {
        "target_products": products,
        "branch_attribute_targets": attrs,
        "attribute_queries": plan,
        "global_search_allowed": not specific_refs and not products,
        "reason": (
            "User refers to specific visual/held/pointed product(s); product-specific getters before branch."
            if specific_refs or products
            else "No specific visual target resolved."
        ),
    }
