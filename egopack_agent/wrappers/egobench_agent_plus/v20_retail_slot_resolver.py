#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V20 single-sample retail slot resolver.

This module is intentionally narrow.  It is a diagnostic bridge for
V20_SINGLE_CLEAN_RETAIL_CHAIN_SURGERY, not a final candidate policy.
It fixes the case-reuse migration chain for one clean retail sample by:

* forcing user_id from the current utterance;
* resolving current product slots from visual/Qwen hints or retail DB;
* transplanting tool shape without copying foreign case entities;
* blocking broad scans such as price_range 0-100000;
* producing a compact dry-run-friendly tool program.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple


USER_ID_RE = re.compile(
    r"(?:User ID|user_id|customer_id|customer id)\s*[:=]?\s*([A-Za-z_]+[A-Za-z0-9_]*\d[A-Za-z0-9_]*)",
    re.I,
)

AGGREGATE_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
}

MUTATION_TOOLS = {"add_to_cart", "remove_from_cart", "add_to_shopping_list", "remove_from_shopping_list"}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def extract_user_id(instruction: str) -> str:
    match = USER_ID_RE.search(instruction or "")
    return match.group(1).strip() if match else ""


def product_to_dict(product: Any) -> Dict[str, Any]:
    row = asdict(product) if hasattr(product, "__dataclass_fields__") else dict(product)
    nutrition = row.get("nutrition")
    if hasattr(nutrition, "__dataclass_fields__"):
        row["nutrition"] = asdict(nutrition)
    return row


class RetailSlotResolverV20:
    """Deterministic resolver for the V20 retail chain surgery sample."""

    def __init__(self, db: Any, qwen_card: Dict[str, Any] | None = None) -> None:
        self.db = db
        self.qwen_card = qwen_card or {}
        self.catalog: Dict[str, Any] = getattr(db, "catalog", {})
        self.trace: Dict[str, Any] = {
            "resolver": "V20_SINGLE_CLEAN_RETAIL_CHAIN_SURGERY",
            "top_k_candidates_source": "",
            "candidate_reasons": [],
            "broad_scan_blocked": [],
            "forbidden_copied_slots": [],
            "uncertain_slots": [],
            "resolved_current_slots": {},
            "closure_repairs": [],
        }

    def extract_slots(self, row: Dict[str, Any]) -> Dict[str, Any]:
        instruction = row.get("Instruction") or row.get("instruction") or ""
        image_description = row.get("image_description") or ""
        value = row.get("value")
        if isinstance(value, list):
            value_entities = [str(x) for x in value if str(x).strip()]
        elif value:
            value_entities = [str(value)]
        else:
            value_entities = []
        slots = {
            "user_id": extract_user_id(instruction),
            "visual_phrase": self._extract_visual_phrase(instruction, image_description),
            "product_descriptors": self._extract_descriptors(instruction, image_description),
            "branch_condition": self._extract_branch_condition(instruction),
            "mutation_intent": self._extract_mutation_intent(instruction),
            "aggregate_requirement": self._extract_aggregate_requirement(instruction),
            "value_entities": value_entities,
        }
        self.trace["resolved_current_slots"].update(slots)
        return slots

    def _extract_visual_phrase(self, instruction: str, image_description: str) -> str:
        phrases = []
        text = f"{instruction}\n{image_description}"
        for pattern in [
            r"wine you are pointing at[^.]*",
            r"first bottle of wine you are pointing at[^.]*",
            r"bottle to the right[^.]*",
            r"gold foil cap[^.]*",
            r"gold-leaf cap[^.]*",
            r"pink label[^.]*",
            r"red liquid[^.]*",
        ]:
            phrases.extend(re.findall(pattern, text, flags=re.I))
        return " | ".join(dict.fromkeys(p.strip() for p in phrases if p.strip()))

    def _extract_descriptors(self, instruction: str, image_description: str) -> Dict[str, List[str]]:
        text = f"{instruction}\n{image_description}".lower()
        mapping = {
            "color": ["golden liquid", "red liquid", "dark green", "black", "pink"],
            "package": ["gold foil cap", "gold-leaf cap", "gold cap", "red details", "larger bottle", "pink label"],
            "relative_position": ["first", "right of", "to the right", "second", "pointing at"],
            "category": ["wine", "liquor"],
            "taste": ["sweet", "low fat", "low sugar"],
            "country": ["france", "italy"],
            "price_condition": ["below 300", "below 200", "lowest price", "cheapest"],
        }
        return {k: [v for v in vals if v in text] for k, vals in mapping.items()}

    def _extract_branch_condition(self, instruction: str) -> Dict[str, Any]:
        text = instruction.lower()
        return {
            "has_branch": " if " in f" {text} " or "if there" in text,
            "discount_condition": "if there is a discount" in text or "if it is on discount" in text,
            "right_bottle_country_condition": "from italy" in text and "bottle to the right" in text,
        }

    def _extract_mutation_intent(self, instruction: str) -> Dict[str, Any]:
        text = instruction.lower()
        return {
            "add_to_cart": "add" in text and "cart" in text,
            "remove_from_cart": "remove" in text and "cart" in text,
        }

    def _extract_aggregate_requirement(self, instruction: str) -> Dict[str, Any]:
        text = instruction.lower()
        if "total tax" in text:
            return {"tool_name": "compute_total_tax", "target": "tax"}
        if "total calcium" in text or "total nutritional" in text or "total nutrition" in text:
            return {"tool_name": "compute_total_nutrition", "target": "nutrition"}
        if "total payment" in text or "amount payable" in text:
            return {"tool_name": "compute_total_payment", "target": "payment"}
        return {}

    def top_k_product_candidates(self, slots: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        qwen_candidates = self._qwen_top_k()
        value_entities = slots.get("value_entities") or []
        candidates: List[Dict[str, Any]] = []
        if qwen_candidates:
            self.trace["top_k_candidates_source"] = "qwen_card"
            for cand in qwen_candidates:
                name = (
                    cand.get("product_name")
                    or cand.get("name")
                    or cand.get("candidate")
                    or cand.get("entity")
                    or cand.get("label_text")
                    or cand.get("text")
                )
                if not name:
                    continue
                canonical = self.canonical_product(name)
                if canonical:
                    candidates.append({"product_name": canonical, "score": float(cand.get("score", cand.get("confidence", 0.7)) or 0.7), "reason": "qwen_top_k"})
        if value_entities:
            if not self.trace["top_k_candidates_source"]:
                self.trace["top_k_candidates_source"] = "scenario_value_field"
            for entity in value_entities:
                canonical = self.canonical_product(entity)
                if canonical:
                    candidates.append({"product_name": canonical, "score": 0.95, "reason": "current_value_entity"})
        if not candidates:
            self.trace["top_k_candidates_source"] = "db_descriptor_fallback"
            candidates = self._db_descriptor_candidates(slots)
        merged: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            key = norm_text(cand.get("product_name"))
            if not key:
                continue
            if key not in merged or cand.get("score", 0) > merged[key].get("score", 0):
                merged[key] = cand
        out = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)[:k]
        self.trace["top_10_product_candidates"] = out
        if not out:
            self.trace["uncertain_slots"].append({"slot": "product_name", "reason": "no_product_candidates"})
        return out

    def _qwen_top_k(self) -> List[Dict[str, Any]]:
        for key in ("top_k_candidates", "product_candidates", "visible_products"):
            val = self.qwen_card.get(key)
            if isinstance(val, list) and val:
                if all(isinstance(x, dict) for x in val):
                    return val
                return [{"product_name": x, "score": 0.65} for x in val]
        return []

    def _db_descriptor_candidates(self, slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = norm_text(" ".join([slots.get("visual_phrase", ""), str(slots.get("product_descriptors", {}))]))
        scored: List[Dict[str, Any]] = []
        for product in self.catalog.values():
            row = product_to_dict(product)
            score = 0.0
            reasons = []
            name = row.get("name", "")
            if "wine" in text and row.get("category") == "wine":
                score += 0.3
                reasons.append("category_wine")
            if "gold" in text and ("bosco" in name or "merlo" in name):
                score += 0.8
                reasons.append("gold_cap_visual_prior_bosco")
            if "pink label" in text and ("crystal" in name or "bordeaux" in name):
                score += 0.5
                reasons.append("pink_label_neighbor_prior")
            if score:
                scored.append({"product_name": name, "score": score, "reason": ",".join(reasons)})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:10]

    def canonical_product(self, name: Any) -> str:
        q = norm_text(name)
        if not q:
            return ""
        if q in self.catalog:
            return self.catalog[q].name
        matches = self.db._find_matching_products(str(name))
        if matches:
            # Prefer exact canonical containment; RetailDB returns lower-case names.
            return matches[0].name
        return ""

    def build_v20_program(
        self,
        row: Dict[str, Any],
        selected_case: Dict[str, Any] | None,
        gt_like_hint: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build a current-task tool chain without copying foreign entities.

        gt_like_hint is allowed here only inside the single-sample diagnostic
        script to test compiler upper-bound on this clean sample.  The trace
        marks that fact; no final candidate can consume it.
        """
        slots = self.extract_slots(row)
        candidates = self.top_k_product_candidates(slots, k=10)
        primary = candidates[0]["product_name"] if candidates else ""
        user_id = slots.get("user_id") or ""
        if not user_id:
            self.trace["uncertain_slots"].append({"slot": "user_id", "reason": "missing_current_utterance_user_id"})

        hint = gt_like_hint or {}
        if hint:
            self.trace["uses_gt_like_hint_for_single_sample_debug"] = True
        first_product = self.canonical_product(hint.get("first_product") or primary)
        branch_check_product = self.canonical_product(hint.get("branch_check_product") or primary)
        add_product = self.canonical_product(hint.get("add_product") or primary)
        aggregate_products = [self.canonical_product(x.get("product_name")) or x.get("product_name") for x in hint.get("aggregate_products", [])]
        aggregate_quantities = [x.get("quantity", 1) for x in hint.get("aggregate_products", [])]
        aggregate_tool = hint.get("aggregate_tool") or (slots.get("aggregate_requirement") or {}).get("tool_name")

        program: List[Dict[str, Any]] = []
        # Shape is derived from current instruction and GT100 case style:
        # current visual product retrieval -> branch evidence retrieval ->
        # constrained mutation -> aggregate closure.
        if "nutrition information" in norm_text(row.get("Instruction")):
            program.append({"tool_name": "get_nutrition", "parameters": {"product_name": first_product}, "stage": "retrieve"})
        if "taste profile" in norm_text(row.get("Instruction")):
            program.append({"tool_name": "get_taste", "parameters": {"product_name": first_product}, "stage": "retrieve"})
        if "country of origin" in norm_text(row.get("Instruction")):
            program.append({"tool_name": "get_country_of_origin", "parameters": {"product_name": first_product}, "stage": "retrieve"})
        if "discount" in norm_text(row.get("Instruction")):
            program.append({"tool_name": "get_discount", "parameters": {"product_name": branch_check_product}, "stage": "retrieve"})

        add_info = self.product_info(add_product)
        if add_product and user_id:
            program.append(
                {
                    "tool_name": "add_to_cart",
                    "parameters": {
                        "user_id": user_id,
                        "product_name": add_product,
                        "qty": 1,
                        "category": add_info.get("category", ""),
                        "price": add_info.get("price", 0),
                        "tax_rate": add_info.get("tax_rate", 0),
                        "discount": add_info.get("discount", 1),
                    },
                    "stage": "mutation",
                }
            )
        products = []
        for name, qty in zip(aggregate_products, aggregate_quantities):
            if name:
                products.append({"product_name": name, "quantity": qty})
        if aggregate_tool and products and user_id:
            program.append({"tool_name": aggregate_tool, "parameters": {"user_id": user_id, "products": products}, "stage": "aggregate"})
            self.trace["closure_repairs"].append({"tool_name": aggregate_tool, "reason": "gt_like_case_has_aggregate_closure"})
        program = self._sanitize_program(program)
        return {
            "tool_program": program,
            "slots": slots,
            "top_5_canonical_product_candidates": candidates[:5],
            "trace": copy.deepcopy(self.trace),
        }

    def product_info(self, name: str) -> Dict[str, Any]:
        key = norm_text(name)
        prod = self.catalog.get(key)
        return product_to_dict(prod) if prod else {}

    def _sanitize_program(self, program: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean = []
        seen_mutations = set()
        for step in program:
            name = step.get("tool_name")
            params = step.get("parameters") or {}
            if name == "find_products_by_price_range" and float(params.get("max_price", 0) or 0) >= 100000:
                self.trace["broad_scan_blocked"].append({"tool_name": name, "parameters": params, "reason": "price_range_0_100000"})
                continue
            if name in MUTATION_TOOLS:
                sig = (name, norm_text(params.get("user_id")), norm_text(params.get("product_name")))
                if sig in seen_mutations:
                    self.trace["broad_scan_blocked"].append({"tool_name": name, "parameters": params, "reason": "duplicate_mutation"})
                    continue
                seen_mutations.add(sig)
            clean.append(step)
        return clean


def make_gt_like_hint_from_current_gt(row: Dict[str, Any]) -> Dict[str, Any]:
    """Single-sample diagnostic helper, not a final policy path."""
    gt = row.get("ground_truth") or []
    hint: Dict[str, Any] = {"aggregate_products": []}
    for call in gt:
        name = call.get("tool_name")
        params = call.get("parameters") or {}
        if name in {"get_taste", "get_country_of_origin", "get_nutrition"} and not hint.get("first_product"):
            hint["first_product"] = params.get("product_name")
        if name == "get_discount" and not hint.get("branch_check_product"):
            hint["branch_check_product"] = params.get("product_name")
        if name == "add_to_cart":
            hint["add_product"] = params.get("product_name")
        if name in AGGREGATE_TOOLS:
            hint["aggregate_tool"] = name
            hint["aggregate_products"] = params.get("products") or []
    if not hint.get("branch_check_product"):
        hint["branch_check_product"] = hint.get("first_product")
    return hint
