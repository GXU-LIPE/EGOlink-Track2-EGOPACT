#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V25-new evidence-driven resolver.

This reuses V21's observation-driven retail path and V24's compact
scenario generators.  The only new behavior is feeding current-task
multimodal canonical evidence into those generators without copying case
entities or using GT.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from .v25_evidence_entity_matcher import norm_text


def _slot_names(evidence: Dict[str, Any], slot: str, limit: int = 6) -> List[str]:
    vals = (evidence.get("candidate_slots") or {}).get(slot) or []
    out: List[str] = []
    if isinstance(vals, list):
        for item in vals:
            if isinstance(item, dict):
                name = item.get("canonical_name") or item.get("name") or item.get("entity")
            else:
                name = item
            if name and norm_text(name) not in {norm_text(x) for x in out}:
                out.append(str(name))
    return out[:limit]


def _qwen_from_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    slots = evidence.get("candidate_slots") or {}
    top: List[Dict[str, Any]] = []
    for slot, typ in [
        ("primary_product", "product"),
        ("dish", "dish"),
        ("set_meal", "set_meal"),
        ("restaurant", "restaurant"),
        ("ingredient", "ingredient"),
        ("recipe", "recipe"),
    ]:
        for item in slots.get(slot) or []:
            if isinstance(item, dict):
                name = item.get("canonical_name") or item.get("entity") or item.get("name")
                conf = item.get("score", 0.65)
                reason = ",".join(item.get("raw_mentions") or []) if isinstance(item.get("raw_mentions"), list) else ""
            else:
                name, conf, reason = item, 0.6, ""
            if name:
                top.append({"entity": name, "type": typ, "confidence": float(conf or 0.6), "evidence": reason or "v25_evidence_table"})
    visible_text = (evidence.get("ocr_evidence") or {}).get("visible_text") or []
    return {
        "status": "v25_evidence_table",
        "scenario": evidence.get("scenario"),
        "scene_summary": " | ".join([str(x.get("raw_name", "")) for x in evidence.get("vision_entities", []) if isinstance(x, dict)][:10]),
        "visible_text": visible_text,
        "top_k_candidates": top[:12],
        "_path": (evidence.get("sources") or {}).get("qwen_card_path", ""),
    }


def _enriched_row(row: Dict[str, Any], evidence: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    r = copy.deepcopy(row)
    # Runtime evidence may use canonical names; this is not GT.  Existing V21/V24
    # resolvers read value/key as current visual slots, so feed only evidence
    # table candidates here.
    if scenario == "retail":
        vals = _slot_names(evidence, "primary_product", 5)
        r["key"] = "product_name"
        if vals:
            r["value"] = vals
    elif scenario in {"restaurant", "order"}:
        vals = _slot_names(evidence, "dish", 4) + _slot_names(evidence, "set_meal", 4)
        if vals:
            r["key"] = "dish_name"
            r["value"] = vals
    elif scenario == "kitchen":
        vals = _slot_names(evidence, "recipe", 3) + _slot_names(evidence, "ingredient", 5)
        if vals:
            r["key"] = "recipe_or_ingredient"
            r["value"] = vals
    text_bits = []
    for key in ("visible_text", "menu_text", "package_text", "price_text"):
        text_bits += [str(x) for x in (evidence.get("ocr_evidence") or {}).get(key, [])]
    for ent in evidence.get("vision_entities") or []:
        if isinstance(ent, dict):
            text_bits.append(str(ent.get("raw_name", "")))
            text_bits.append(str(ent.get("canonical_db_name", "")))
            text_bits.append(str(ent.get("reason", "")))
    r["image_description"] = (r.get("image_description") or "") + "\n[V25 evidence] " + " | ".join([x for x in text_bits if norm_text(x)][:80])
    return r


def _program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        if not isinstance(block, dict):
            continue
        for call in block.get("calls") or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": call.get("parameters") or {}})
    return out


def _candidate(candidate_id: str, source: str, program: List[Dict[str, Any]], evidence: Dict[str, Any], confidence: float) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source": source,
        "tool_program": program,
        "confidence": confidence,
        "meta": {
            "evidence_task_key": evidence.get("task_key"),
            "evidence_sources": evidence.get("sources"),
            "canonical_slots": evidence.get("candidate_slots"),
        },
    }


def build_v25_candidates(
    *,
    scenario: str,
    row: Dict[str, Any],
    db: Any,
    evidence: Dict[str, Any],
    v14_item: Dict[str, Any] | None = None,
    v22_item: Dict[str, Any] | None = None,
    max_candidates: int = 6,
) -> Dict[str, Any]:
    """Return few high-quality evidence resolver candidates."""
    qwen = _qwen_from_evidence(evidence)
    erow = _enriched_row(row, evidence, scenario)
    candidates: List[Dict[str, Any]] = []
    trace: Dict[str, Any] = {
        "resolver": "V25_NEW_MULTIMODAL_EVIDENCE_V21_AGENT",
        "scenario": scenario,
        "evidence_task_key": evidence.get("task_key"),
        "evidence_slots": evidence.get("candidate_slots"),
        "qwen_status": (evidence.get("sources") or {}).get("qwen_status"),
        "gpt55_vision_status": (evidence.get("sources") or {}).get("gpt55_vision_status"),
    }
    if v14_item:
        candidates.append(_candidate("A_V14_BASE", "V14", _program_from_item(v14_item), evidence, 0.5))
    if v22_item:
        candidates.append(_candidate("B_V22_BASE", "V22", _program_from_item(v22_item), evidence, 0.55))

    if scenario == "retail":
        try:
            from .v21_retail_resolver import RetailResolverV21

            obj = RetailResolverV21(db, qwen).build(erow, None)
            prog = obj["candidate"]["tool_program"]
            candidates.append(_candidate("C_V25_RETAIL_EVIDENCE_V21", "V25_RETAIL_V21_EVIDENCE", prog, evidence, 0.78))
            trace["v21_trace"] = obj.get("trace") or {}
        except Exception as exc:
            trace["v21_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from .v24_scenario_gap_generators import generate_for_scenario

        gen = generate_for_scenario(scenario, erow, db, qwen, max_candidates=max(4, max_candidates))
        for i, cand in enumerate(gen[: max(0, max_candidates - len(candidates))]):
            candidates.append(
                _candidate(
                    f"D_V25_SCENARIO_EVIDENCE_{i+1}",
                    "V25_SCENARIO_EVIDENCE",
                    cand.get("tool_program") or [],
                    evidence,
                    float(cand.get("confidence", 0.65) or 0.65),
                )
            )
    except Exception as exc:
        trace["scenario_generator_error"] = f"{type(exc).__name__}: {exc}"

    # Closure repair for mutation candidates that lack the obvious final
    # aggregate mentioned in the instruction.
    repaired: List[Dict[str, Any]] = []
    text = norm_text(row.get("Instruction", ""))
    closure = ""
    if "total tax" in text:
        closure = "compute_total_tax"
    elif "total nutrition" in text or "total nutritional" in text or "total calcium" in text:
        closure = "compute_total_nutrition" if scenario != "kitchen" else "compute_total_nutritions"
    elif "total payment" in text or "amount payable" in text or "total cost" in text:
        closure = "compute_total_payment"
    for cand in candidates:
        prog = copy.deepcopy(cand.get("tool_program") or [])
        names = [x.get("tool_name") for x in prog]
        if closure and any(n in names for n in ("add_to_cart", "add_dish_to_order", "add_set_meal_to_order", "add_to_shopping_list")) and closure not in names:
            user_id = ""
            for call in prog:
                user_id = (call.get("parameters") or {}).get("user_id") or user_id
            if user_id:
                params: Dict[str, Any] = {"user_id": user_id}
                if scenario == "order":
                    for call in prog:
                        rn = (call.get("parameters") or {}).get("restaurant_name")
                        if rn:
                            params["restaurant_name"] = rn
                            break
                prog.append({"tool_name": closure, "parameters": params})
                repaired.append(_candidate(cand["candidate_id"] + "_CLOSURE", "V25_CLOSURE_REPAIR", prog, evidence, cand.get("confidence", 0.5) + 0.02))
    candidates.extend(repaired[:1])

    dedup: List[Dict[str, Any]] = []
    seen = set()
    for cand in candidates:
        sig = json.dumps(
            [
                {"tool_name": c.get("tool_name"), "parameters": c.get("parameters") or {}}
                for c in cand.get("tool_program") or []
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(cand)
        if len(dedup) >= max_candidates:
            break
    trace["candidate_count"] = len(dedup)
    return {"candidates": dedup, "trace": trace, "enriched_row": erow, "evidence_qwen_card": qwen}
