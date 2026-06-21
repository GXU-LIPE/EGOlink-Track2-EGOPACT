#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V27 bridge from multimodal evidence into the real V21 retail resolver.

This file intentionally does not compile tool programs itself.  It adapts
V25/V26 evidence into the qwen_card/row shape consumed by RetailResolverV21 and
then calls the real V21 executable resolver.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Tuple

from .v20_retail_slot_resolver import extract_user_id, norm_text
from .v21_retail_resolver import RetailResolverV21
from .v21_retail_attribute_query_planner import build_attribute_query_plan, infer_attribute_targets
from .v21_retail_observation_brancher import RetailObservationBrancherV21
from .v21_retail_add_target_resolver import RetailAddTargetResolverV21


BRANCH_ATTRIBUTE_WORDS = {
    "taste": ("taste", "sweet", "bitter", "sour", "savory", "mild"),
    "country_of_origin": ("country", "origin", "italy", "france", "produced"),
    "nutrition": ("nutrition", "nutritional", "sugar", "calorie", "calcium", "fat", "protein"),
    "discount": ("discount", "sale", "on sale"),
    "price": ("price", "cheapest", "lowest", "expensive", "below"),
    "tax_rate": ("tax",),
    "category": ("category", "wine", "snack", "drink"),
}


def _product_row(db: Any, name: str) -> Dict[str, Any]:
    product = getattr(db, "catalog", {}).get(norm_text(name))
    if not product:
        return {}
    if hasattr(product, "__dataclass_fields__"):
        from dataclasses import asdict

        row = asdict(product)
    else:
        row = dict(product)
    nutrition = row.get("nutrition")
    if hasattr(nutrition, "__dataclass_fields__"):
        from dataclasses import asdict

        row["nutrition"] = asdict(nutrition)
    return row


def _canonical_product(db: Any, name: Any) -> str:
    q = norm_text(name)
    if not q:
        return ""
    row = _product_row(db, q)
    if row:
        return row.get("name", q)
    finder = getattr(db, "_find_matching_products", None)
    if finder:
        try:
            matches = finder(str(name))
        except Exception:
            matches = []
        if matches:
            match = matches[0]
            return getattr(match, "name", "") or (match.get("name") if isinstance(match, dict) else "")
    return ""


def _add_candidate(out: List[Dict[str, Any]], db: Any, name: Any, score: float, reason: str, evidence: Any = None) -> None:
    canonical = _canonical_product(db, name)
    if not canonical:
        return
    key = norm_text(canonical)
    for item in out:
        if norm_text(item.get("product_name")) == key:
            if score > float(item.get("score", 0) or 0):
                item["score"] = score
                item["reason"] = reason
            return
    row = _product_row(db, canonical)
    out.append(
        {
            "product_name": canonical,
            "name": canonical,
            "score": float(score),
            "confidence": float(score),
            "reason": reason,
            "category": row.get("category", ""),
            "evidence": evidence,
        }
    )


def _iter_slot_entities(evidence: Dict[str, Any], slot_names: Iterable[str]) -> Iterable[Tuple[Any, float, str, Any]]:
    slots = evidence.get("candidate_slots") or {}
    for slot_name in slot_names:
        val = slots.get(slot_name) or []
        if not isinstance(val, list):
            val = [val]
        for item in val:
            if isinstance(item, dict):
                name = (
                    item.get("canonical_name")
                    or item.get("canonical_db_name")
                    or item.get("product_name")
                    or item.get("entity")
                    or item.get("name")
                    or item.get("raw_name")
                )
                score = item.get("score", item.get("confidence", 0.7))
            else:
                name = item
                score = 0.65
            if name:
                yield name, float(score or 0.65), f"evidence_slot:{slot_name}", item


def _row_value_products(row: Dict[str, Any]) -> List[str]:
    value = row.get("value")
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value:
        return [str(value)]
    return []


def product_candidates_from_evidence(db: Any, row: Dict[str, Any], evidence: Dict[str, Any] | None, limit: int = 10) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    evidence = evidence or {}
    for name, score, reason, ev in _iter_slot_entities(evidence, ("primary_product", "product", "visible_products")):
        _add_candidate(candidates, db, name, min(1.0, score), reason, ev)
    for ent in evidence.get("vision_entities") or []:
        if not isinstance(ent, dict):
            continue
        if ent.get("type") not in (None, "", "product", "wine", "item"):
            continue
        name = ent.get("canonical_db_name") or ent.get("canonical_name") or ent.get("raw_name") or ent.get("name")
        _add_candidate(candidates, db, name, float(ent.get("confidence", 0.62) or 0.62), "vision_entity", ent)
    for name in _row_value_products(row):
        _add_candidate(candidates, db, name, 0.96, "current_value_entity", {"value": name})
    candidates.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    return candidates[:limit]


def branch_attribute_targets(row: Dict[str, Any], evidence: Dict[str, Any] | None) -> List[str]:
    text_bits = [row.get("Instruction", ""), row.get("image_description", "")]
    evidence = evidence or {}
    for item in (evidence.get("candidate_slots") or {}).get("branch_attributes", []) or []:
        if isinstance(item, dict):
            text_bits.extend([str(item.get("attribute", "")), str(item.get("value", "")), " ".join(map(str, item.get("evidence", []) or []))])
        else:
            text_bits.append(str(item))
    attrs = set(infer_attribute_targets(" ".join(text_bits)))
    text = norm_text(" ".join(text_bits))
    for attr, words in BRANCH_ATTRIBUTE_WORDS.items():
        if any(w in text for w in words):
            attrs.add(attr)
    order = ["taste", "country_of_origin", "category", "price", "tax_rate", "discount", "nutrition"]
    return [x for x in order if x in attrs]


def closure_needed(row: Dict[str, Any], evidence: Dict[str, Any] | None) -> List[str]:
    text = norm_text(row.get("Instruction", ""))
    evidence = evidence or {}
    for item in (evidence.get("candidate_slots") or {}).get("closure_needed", []) or []:
        text += " " + norm_text(item)
    out: List[str] = []
    if "total tax" in text:
        out.append("compute_total_tax")
    if any(x in text for x in ("total nutritional", "total nutrition", "total calcium", "nutritional value")):
        out.append("compute_total_nutrition")
    if any(x in text for x in ("total payment", "amount payable", "total price")):
        out.append("compute_total_payment")
    return list(dict.fromkeys(out))


def mutation_intent(row: Dict[str, Any], evidence: Dict[str, Any] | None) -> str:
    text = norm_text(row.get("Instruction", ""))
    evidence = evidence or {}
    text += " " + norm_text((evidence.get("candidate_slots") or {}).get("mutation_intent", ""))
    if "remove" in text and "cart" in text:
        return "remove_from_cart"
    if "add" in text and "cart" in text:
        return "add_to_cart"
    if "cart" in text:
        return "update_cart"
    return "query_only"


def build_v21_qwen_card_from_evidence(db: Any, row: Dict[str, Any], evidence: Dict[str, Any] | None) -> Dict[str, Any]:
    products = product_candidates_from_evidence(db, row, evidence, limit=10)
    attrs = branch_attribute_targets(row, evidence)
    return {
        "status": "v27_v25_evidence_adapter",
        "top_k_candidates": products,
        "product_candidates": products,
        "branch_attribute_targets": attrs,
        "mutation_intent": mutation_intent(row, evidence),
        "closure_needed": closure_needed(row, evidence),
        "evidence_summary": {
            "task_key": (evidence or {}).get("task_key"),
            "sources": (evidence or {}).get("sources"),
            "candidate_count": len(products),
        },
    }


def enrich_row_for_v21(row: Dict[str, Any], db: Any, evidence: Dict[str, Any] | None, use_evidence: bool) -> Dict[str, Any]:
    erow = copy.deepcopy(row)
    if not use_evidence:
        return erow
    products = product_candidates_from_evidence(db, row, evidence, limit=10)
    if products:
        erow["value"] = [p["product_name"] for p in products[:5]]
    lines = [str(row.get("image_description", ""))]
    slots = (evidence or {}).get("candidate_slots") or {}
    if products:
        lines.append("V27 multimodal evidence product candidates: " + ", ".join(p["product_name"] for p in products[:5]))
    if slots.get("branch_attributes"):
        lines.append("V27 multimodal evidence branch attributes: " + re.sub(r"\s+", " ", str(slots.get("branch_attributes"))[:1000]))
    erow["image_description"] = "\n".join(x for x in lines if x)
    return erow


def build_v27_v21_candidate(
    db: Any,
    row: Dict[str, Any],
    evidence: Dict[str, Any] | None = None,
    qwen_card: Dict[str, Any] | None = None,
    use_evidence: bool = False,
) -> Dict[str, Any]:
    """Call the real V21 resolver and return candidate plus wiring trace."""
    if use_evidence:
        v21_qwen = build_v21_qwen_card_from_evidence(db, row, evidence)
        erow = enrich_row_for_v21(row, db, evidence, use_evidence=True)
        source = "V27_DIRECT_V21_PLUS_V25_EVIDENCE"
    else:
        v21_qwen = copy.deepcopy(qwen_card or {})
        erow = enrich_row_for_v21(row, db, evidence, use_evidence=False)
        source = "V27_DIRECT_V21_ORIGINAL_INPUT"

    trace: Dict[str, Any] = {
        "adapter": "V27_DIRECT_V21_WIRING_AND_MM_EVIDENCE_BRIDGE",
        "source": source,
        "called_RetailResolverV21": False,
        "called_attribute_query_planner": False,
        "called_observation_brancher": False,
        "called_add_target_resolver": False,
        "imported_RetailResolverV21": RetailResolverV21 is not None,
        "imported_build_attribute_query_plan": build_attribute_query_plan is not None,
        "imported_RetailObservationBrancherV21": RetailObservationBrancherV21 is not None,
        "imported_RetailAddTargetResolverV21": RetailAddTargetResolverV21 is not None,
        "v21_input_context": v21_qwen,
        "current_user_id": extract_user_id(erow.get("Instruction", "")),
        "uses_v25_evidence": bool(use_evidence),
    }
    try:
        resolver = RetailResolverV21(db, v21_qwen)
        obj = resolver.build(erow, None)
        trace["called_RetailResolverV21"] = True
        v21_trace = obj.get("trace") or {}
        # RetailResolverV21.build always calls these three components.  Keep the
        # trace explicit so the driver can fail runtime wiring if a future
        # refactor silently bypasses V21 internals.
        trace["called_attribute_query_planner"] = "attribute_query_plan" in v21_trace
        trace["called_observation_brancher"] = "branch_decision" in v21_trace
        trace["called_add_target_resolver"] = "mutation_target" in v21_trace
        trace["v21_trace"] = v21_trace
        candidate = copy.deepcopy(obj.get("candidate") or {})
        candidate["candidate_id"] = "V27_EVIDENCE_V21" if use_evidence else "V27_DIRECT_V21"
        candidate["source"] = source
        candidate["v27_trace"] = trace
        return {"candidate": candidate, "trace": trace, "error": ""}
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
        return {"candidate": {"candidate_id": "V27_V21_ERROR", "source": source, "tool_program": [], "risk_flags": ["v21_error"]}, "trace": trace, "error": trace["error"]}
