#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V28 evidence guard.

Multimodal evidence is allowed to veto, tiebreak, and hint queries.  It is not
allowed to directly replace mutation targets or decide branches.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from .v20_retail_slot_resolver import norm_text


MUTATION_KEYS = {
    "product_name",
    "dish_name",
    "set_meal_name",
    "recipe_name",
    "ingredient_name",
}


def _names_from_slot_items(items: Any) -> List[Tuple[str, float, str]]:
    if not isinstance(items, list):
        items = [items] if items else []
    out: List[Tuple[str, float, str]] = []
    for item in items:
        if isinstance(item, dict):
            name = (
                item.get("canonical_name")
                or item.get("canonical_db_name")
                or item.get("product_name")
                or item.get("dish_name")
                or item.get("set_meal_name")
                or item.get("recipe_name")
                or item.get("ingredient_name")
                or item.get("raw_name")
                or item.get("name")
            )
            score = float(item.get("score", item.get("confidence", 0.6)) or 0.6)
            source = ",".join(item.get("matched_by") or []) if isinstance(item.get("matched_by"), list) else str(item.get("matched_by") or "evidence")
        else:
            name, score, source = item, 0.55, "evidence"
        if name:
            out.append((str(name), score, source))
    return out


def evidence_entity_names(evidence: Dict[str, Any] | None) -> Dict[str, List[Tuple[str, float, str]]]:
    evidence = evidence or {}
    slots = evidence.get("candidate_slots") or {}
    out: Dict[str, List[Tuple[str, float, str]]] = {}
    slot_map = {
        "product": ("primary_product", "product", "visible_products"),
        "dish": ("dish",),
        "set_meal": ("set_meal",),
        "recipe": ("recipe",),
        "ingredient": ("ingredient",),
        "restaurant": ("restaurant",),
    }
    for typ, keys in slot_map.items():
        rows: List[Tuple[str, float, str]] = []
        for key in keys:
            rows.extend(_names_from_slot_items(slots.get(key)))
        for ent in evidence.get("vision_entities") or []:
            if not isinstance(ent, dict):
                continue
            etype = norm_text(ent.get("type"))
            if typ in etype or (typ == "product" and etype in {"item", "wine"}):
                name = ent.get("canonical_db_name") or ent.get("canonical_name") or ent.get("raw_name") or ent.get("name")
                if name:
                    rows.append((str(name), float(ent.get("confidence", 0.55) or 0.55), "vision_entity"))
        dedup: Dict[str, Tuple[str, float, str]] = {}
        for name, score, source in rows:
            key = norm_text(name)
            if key and (key not in dedup or score > dedup[key][1]):
                dedup[key] = (name, score, source)
        out[typ] = sorted(dedup.values(), key=lambda x: x[1], reverse=True)
    return out


def mutation_targets(program: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    for call in program or []:
        params = call.get("parameters") or {}
        for key in MUTATION_KEYS:
            if params.get(key):
                targets.append((key, str(params.get(key))))
    seen = set()
    out = []
    for key, name in targets:
        sig = (key, norm_text(name))
        if sig not in seen:
            seen.add(sig)
            out.append((key, name))
    return out


def _target_type(param_key: str) -> str:
    return {
        "product_name": "product",
        "dish_name": "dish",
        "set_meal_name": "set_meal",
        "recipe_name": "recipe",
        "ingredient_name": "ingredient",
    }.get(param_key, "entity")


def _evidence_support(name: str, candidates: Iterable[Tuple[str, float, str]]) -> Tuple[bool, float, str]:
    n = norm_text(name)
    best = (False, 0.0, "")
    for cand, score, source in candidates:
        c = norm_text(cand)
        if not c:
            continue
        if n == c or n in c or c in n:
            if score > best[1]:
                best = (score >= 0.55, score, source)
    return best


def branch_attribute_hints(evidence: Dict[str, Any] | None, instruction: str) -> List[str]:
    text = norm_text(instruction)
    evidence = evidence or {}
    for item in (evidence.get("candidate_slots") or {}).get("branch_attributes", []) or []:
        text += " " + norm_text(item)
    hints = []
    for attr, words in {
        "taste": ("taste", "sweet", "bitter", "sour"),
        "country": ("country", "origin", "italy", "france"),
        "nutrition": ("nutrition", "sugar", "fat", "protein", "calorie", "calcium"),
        "discount": ("discount", "sale"),
        "price": ("price", "cheapest", "lowest"),
        "tax": ("tax",),
    }.items():
        if any(w in text for w in words):
            hints.append(attr)
    return hints


def guard_candidate(
    task_key: str,
    scenario: str,
    instruction: str,
    evidence: Dict[str, Any] | None,
    candidate: Dict[str, Any],
    *,
    allow_override: bool = False,
) -> Dict[str, Any]:
    program = candidate.get("tool_program") or []
    entities = evidence_entity_names(evidence)
    trace = {
        "task_key": task_key,
        "scenario": scenario,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_before": mutation_targets(program),
        "evidence_candidate": {k: v[:5] for k, v in entities.items()},
        "action": "keep_original",
        "reason": "",
        "risk_flags": [],
        "query_hints": branch_attribute_hints(evidence, instruction),
        "allow": True,
    }
    if not evidence:
        trace["action"] = "ignore"
        trace["reason"] = "no_evidence"
        return trace
    for param_key, name in trace["candidate_before"]:
        typ = _target_type(param_key)
        supported, score, source = _evidence_support(name, entities.get(typ, []))
        if supported:
            trace["action"] = "tiebreak"
            trace["reason"] = f"{typ}:{name} supported by {source}:{score:.2f}"
            continue
        strong_alt = [x for x in entities.get(typ, []) if x[1] >= 0.88]
        if strong_alt:
            trace["risk_flags"].append("evidence_disagrees_with_mutation_target")
            trace["reason"] = f"{typ}:{name} not supported; strong evidence alt {strong_alt[0][0]}"
            if allow_override:
                trace["action"] = "veto"
                trace["allow"] = False
            else:
                trace["action"] = "query_hint"
        else:
            trace["risk_flags"].append("low_confidence_evidence")
            trace["action"] = "query_hint" if trace["query_hints"] else "ignore"
            trace["reason"] = f"{typ}:{name} no strong evidence match"
    return trace


def evidence_error_bucket(trace: Dict[str, Any], candidate_score: Dict[str, Any], base_score: Dict[str, Any]) -> str:
    if trace.get("action") == "veto" and base_score.get("joint"):
        return "evidence_vetoed_correct_candidate"
    if "evidence_disagrees_with_mutation_target" in (trace.get("risk_flags") or []) and not candidate_score.get("joint"):
        return "evidence_changed_or_conflicted_mutation_target_wrong"
    if trace.get("action") in {"tiebreak", "query_hint"} and candidate_score.get("joint") and not base_score.get("joint"):
        return "evidence_helped_choose_correct_entity"
    if trace.get("action") in {"ignore", "query_hint"}:
        return "evidence_ignored_due_to_low_confidence"
    return "no_evidence_error"
