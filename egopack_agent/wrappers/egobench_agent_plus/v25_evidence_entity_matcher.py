#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical DB entity matcher for V25-new multimodal evidence.

This module is deliberately small.  It maps current-task evidence strings
from vision/OCR/ASR/instruction to canonical names in the active scenario DB.
It does not copy entities from retrieved cases and does not inspect GT.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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


def _ratio(a: str, b: str) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.88
    toks_a = set(re.findall(r"[a-z0-9']+", a))
    toks_b = set(re.findall(r"[a-z0-9']+", b))
    overlap = len(toks_a & toks_b) / max(1, len(toks_a | toks_b))
    return max(SequenceMatcher(None, a, b).ratio(), overlap)


def _score_name(name: str, mentions: Iterable[str], context: str = "") -> Dict[str, Any]:
    best = 0.0
    raw_mentions: List[str] = []
    for raw in mentions:
        score = _ratio(name, raw)
        if score > best:
            best = score
        if score >= 0.35:
            raw_mentions.append(str(raw))
    ctx = norm_text(context)
    n = norm_text(name)
    if n and n in ctx:
        best = max(best, 0.82)
        raw_mentions.append("instruction_or_context_exact")
    else:
        for tok in re.findall(r"[a-z0-9']+", n):
            if len(tok) > 3 and tok in ctx:
                best = max(best, 0.45)
                raw_mentions.append(f"context_token:{tok}")
    risk = "exact" if best >= 0.98 else "fuzzy" if best >= 0.62 else "ambiguous" if best >= 0.42 else "low_confidence"
    return {"score": round(best, 4), "raw_mentions": list(dict.fromkeys(raw_mentions)), "risk": risk}


def _add_entity(rows: List[Dict[str, Any]], name: Any, typ: str, meta: Dict[str, Any] | None = None) -> None:
    if not name:
        return
    rows.append({"canonical_name": str(name), "type": typ, "meta": meta or {}})


def collect_db_entities(scenario: str, db: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Return canonical entity rows grouped by entity type."""
    out: Dict[str, List[Dict[str, Any]]] = {
        "product": [],
        "dish": [],
        "set_meal": [],
        "restaurant": [],
        "ingredient": [],
        "recipe": [],
        "category": [],
    }
    if scenario == "retail":
        for obj in getattr(db, "catalog", {}).values():
            row = _as_dict(obj)
            _add_entity(out["product"], row.get("name"), "product", row)
            _add_entity(out["category"], row.get("category"), "category", {"product": row.get("name")})
    elif scenario == "restaurant":
        for obj in getattr(db, "catalog", {}).values():
            row = _as_dict(obj)
            _add_entity(out["dish"], row.get("name"), "dish", row)
            _add_entity(out["category"], row.get("category"), "category", {"dish": row.get("name")})
        for obj in getattr(db, "set_meals", {}).values():
            row = _as_dict(obj)
            _add_entity(out["set_meal"], row.get("set_meal_name") or row.get("name"), "set_meal", row)
    elif scenario == "order":
        for restaurant, rest in getattr(db, "restaurants", {}).items():
            _add_entity(out["restaurant"], restaurant, "restaurant", {})
            for obj in getattr(rest, "get", lambda *_: {})("catalog", {}).values():
                row = _as_dict(obj)
                row["_restaurant_name"] = restaurant
                _add_entity(out["dish"], row.get("name"), "dish", row)
                _add_entity(out["category"], row.get("category"), "category", {"restaurant": restaurant, "dish": row.get("name")})
            for obj in getattr(rest, "get", lambda *_: {})("set_meals", {}).values():
                row = _as_dict(obj)
                row["_restaurant_name"] = restaurant
                _add_entity(out["set_meal"], row.get("set_meal_name") or row.get("name"), "set_meal", row)
    elif scenario == "kitchen":
        for name, obj in getattr(db, "recipes", {}).items():
            row = _as_dict(obj)
            _add_entity(out["recipe"], name or row.get("name") or row.get("recipe_name"), "recipe", row)
        for name, obj in getattr(db, "ingredients", {}).items():
            row = _as_dict(obj)
            _add_entity(out["ingredient"], name or row.get("name") or row.get("ingredient_name"), "ingredient", row)
            _add_entity(out["category"], row.get("category"), "category", {"ingredient": name})
    for key in list(out):
        dedup: Dict[str, Dict[str, Any]] = {}
        for row in out[key]:
            n = norm_text(row.get("canonical_name"))
            if n and n not in dedup:
                dedup[n] = row
        out[key] = list(dedup.values())
    return out


def evidence_mentions(evidence: Dict[str, Any]) -> List[str]:
    mentions: List[str] = []
    for path in [
        ("utterance",),
        ("frame_evidence", "key_regions"),
        ("ocr_evidence", "visible_text"),
        ("ocr_evidence", "menu_text"),
        ("ocr_evidence", "package_text"),
        ("asr_evidence", "spoken_entities"),
        ("asr_evidence", "transcript"),
    ]:
        obj: Any = evidence
        for key in path:
            obj = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(obj, str):
            mentions.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    mentions.extend(str(v) for v in item.values() if isinstance(v, (str, int, float)))
                else:
                    mentions.append(str(item))
    for ent in evidence.get("vision_entities") or []:
        if isinstance(ent, dict):
            mentions.extend(str(ent.get(k, "")) for k in ("raw_name", "canonical_db_name", "reason", "location"))
        else:
            mentions.append(str(ent))
    return [m for m in mentions if norm_text(m)]


def match_entities(
    scenario: str,
    db: Any,
    evidence: Dict[str, Any],
    *,
    top_k: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Match evidence strings to canonical current DB entities."""
    entities = collect_db_entities(scenario, db)
    mentions = evidence_mentions(evidence)
    context = "\n".join(mentions)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for typ, rows in entities.items():
        scored: List[Dict[str, Any]] = []
        for row in rows:
            name = row.get("canonical_name", "")
            score = _score_name(name, mentions, context)
            if score["score"] <= 0:
                continue
            scored.append(
                {
                    "canonical_name": name,
                    "type": typ,
                    "score": score["score"],
                    "matched_by": ["vision_or_text_fuzzy"] if score["risk"] != "exact" else ["exact_text"],
                    "raw_mentions": score["raw_mentions"][:6],
                    "risk": score["risk"],
                    "meta": row.get("meta") or {},
                }
            )
        scored.sort(key=lambda x: (x["score"], len(x.get("raw_mentions") or [])), reverse=True)
        result[typ] = scored[:top_k]
    return result


def compact_db_entity_list(scenario: str, db: Any, limit_per_type: int = 80) -> Dict[str, List[str]]:
    entities = collect_db_entities(scenario, db)
    return {
        typ: [str(r.get("canonical_name")) for r in rows[:limit_per_type]]
        for typ, rows in entities.items()
        if rows
    }
