#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V21 retail add-target resolver."""

from __future__ import annotations

from typing import Any, Dict, List

from .v20_retail_slot_resolver import norm_text, product_to_dict


class RetailAddTargetResolverV21:
    def __init__(self, db: Any) -> None:
        self.db = db

    def exists(self, product_name: str) -> bool:
        return norm_text(product_name) in getattr(self.db, "catalog", {})

    def product_info(self, product_name: str) -> Dict[str, Any]:
        row = getattr(self.db, "catalog", {}).get(norm_text(product_name))
        return product_to_dict(row) if row else {}

    def build_add_calls(self, user_id: str, products: List[str], qty: int | float = 1) -> List[Dict[str, Any]]:
        calls = []
        seen = set()
        for product in products:
            key = norm_text(product)
            if not key or key in seen or not self.exists(product):
                continue
            seen.add(key)
            info = self.product_info(product)
            calls.append(
                {
                    "tool_name": "add_to_cart",
                    "parameters": {
                        "user_id": user_id,
                        "product_name": info.get("name", key),
                        "qty": qty,
                        "category": info.get("category", ""),
                        "price": info.get("price", 0),
                        "tax_rate": info.get("tax_rate", 0),
                        "discount": info.get("discount", 1),
                    },
                    "stage": "mutation",
                    "evidence": {
                        "product_exists_in_db": True,
                        "source": "branch_selected_target",
                    },
                }
            )
        return calls

    def candidate_rows(self, products: List[str]) -> List[Dict[str, Any]]:
        out = []
        for product in products:
            if self.exists(product):
                info = self.product_info(product)
                out.append(
                    {
                        "product_name": info.get("name"),
                        "category": info.get("category"),
                        "price": info.get("price"),
                        "tax_rate": info.get("tax_rate"),
                        "discount": info.get("discount"),
                        "taste": info.get("taste"),
                        "country_of_origin": info.get("country_of_origin"),
                        "nutritional_characteristics": info.get("nutritional_characteristics"),
                        "nutrition": info.get("nutrition"),
                    }
                )
        return out[:5]
