#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V19 slot rewriting from retrieved GT100 cases to current task context."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple


ENTITY_KEYS = {
    "user_id",
    "restaurant_name",
    "product_name",
    "dish_name",
    "set_meal_name",
    "ingredient_name",
    "recipe_name",
    "category",
}
USER_RE = re.compile(r"\b(?:user|customer|cook|manager|merchant|expert|critic|enthusiast|nutritionist)_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b", re.I)
ID_LABEL_RE = re.compile(r"(?:User ID|user_id|customer_id|customer id)\s*[:=]?\s*([A-Za-z_]+[A-Za-z0-9_]*\d[A-Za-z0-9_]*)", re.I)
RESTAURANT_RE = re.compile(r"\b([A-Z][A-Za-z'& ]{2,40}(?:Restaurant|Bistro|Cafe|Steakhouse|Pizzeria))\b")


def canonical_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def extract_current_slots(context: Dict[str, Any]) -> Dict[str, List[Any]]:
    text = "\n".join([
        str(context.get("instruction", "")),
        str(context.get("dialogue_history", "")),
        str(context.get("visual_text", "")),
    ])
    slots: Dict[str, List[Any]] = {}
    ids = ID_LABEL_RE.findall(text) or USER_RE.findall(text)
    if ids:
        slots["user_id"] = list(dict.fromkeys([x.strip() for x in ids]))
    restaurants = RESTAURANT_RE.findall(text)
    if restaurants:
        slots["restaurant_name"] = list(dict.fromkeys([x.strip() for x in restaurants]))
    visual_candidates = context.get("visual_candidates") or {}
    if isinstance(visual_candidates, dict):
        for key, vals in visual_candidates.items():
            if key in ENTITY_KEYS:
                if isinstance(vals, list):
                    slots.setdefault(key, [])
                    for v in vals:
                        if v not in slots[key]:
                            slots[key].append(v)
                elif vals:
                    slots.setdefault(key, []).append(vals)
    return slots


def collect_case_slots(tool_program: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    slots: Dict[str, List[Any]] = {}
    def visit(obj: Any, key_hint: str | None = None) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ENTITY_KEYS:
                    slots.setdefault(k, [])
                    if v not in slots[k]:
                        slots[k].append(v)
                visit(v, k)
        elif isinstance(obj, list):
            for x in obj:
                visit(x, key_hint)
    for step in tool_program:
        visit(step.get("parameters", {}))
    return slots


def choose_slot(key: str, case_value: Any, current_slots: Dict[str, List[Any]]) -> Tuple[Any, float, str]:
    vals = current_slots.get(key) or []
    if key == "user_id" and vals:
        return vals[0], 1.0, "current_user_id"
    if key == "restaurant_name" and vals:
        return vals[0], 0.9, "current_restaurant"
    if vals:
        # Prefer exact lexical overlap, otherwise first visual/current candidate.
        cv = canonical_text(case_value)
        for v in vals:
            vv = canonical_text(v)
            if cv and vv and (cv in vv or vv in cv):
                return v, 0.85, "current_entity_overlap"
        return vals[0], 0.45, "current_entity_top1"
    return case_value, 0.05, "copied_case_slot_forbidden"


def rewrite_value(key: str, value: Any, current_slots: Dict[str, List[Any]], trace: Dict[str, Any]) -> Any:
    if key in ENTITY_KEYS:
        new_value, conf, reason = choose_slot(key, value, current_slots)
        trace.setdefault("rewritten_slots", []).append({
            "slot": key,
            "original": value,
            "rewritten": new_value,
            "confidence": conf,
            "reason": reason,
        })
        if reason == "copied_case_slot_forbidden":
            trace.setdefault("forbidden_copied_slots", []).append({"slot": key, "value": value})
        if conf < 0.5:
            trace.setdefault("uncertain_slots", []).append({"slot": key, "value": new_value, "confidence": conf, "reason": reason})
        return new_value
    if isinstance(value, dict):
        return {k: rewrite_value(k, v, current_slots, trace) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite_value(key, v, current_slots, trace) for v in value]
    return copy.deepcopy(value)


def rewrite_case_program(case: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    tool_program = case.get("tool_program") or []
    current_slots = extract_current_slots(context)
    original_slots = collect_case_slots(tool_program)
    trace: Dict[str, Any] = {
        "case_id": case.get("case_id"),
        "original_case_slots": original_slots,
        "current_extracted_slots": current_slots,
        "rewritten_slots": [],
        "uncertain_slots": [],
        "forbidden_copied_slots": [],
    }
    rewritten = []
    for step in tool_program:
        params = rewrite_value("", step.get("parameters", {}), current_slots, trace)
        rewritten.append({
            "tool_name": step.get("tool_name"),
            "parameters": params,
            "stage": step.get("stage", ""),
        })
    trace["final_candidate_program"] = rewritten
    return {
        "tool_program": rewritten,
        "trace": trace,
        "slot_confidence": {
            item["slot"]: max(item.get("confidence", 0.0), 0.0)
            for item in trace.get("rewritten_slots", [])
        },
        "risk_flags": (
            ["forbidden_copied_slots"] if trace.get("forbidden_copied_slots") else []
        ) + (
            ["uncertain_slots"] if trace.get("uncertain_slots") else []
        ),
    }
