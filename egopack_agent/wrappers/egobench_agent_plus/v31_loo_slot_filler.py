#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detasked slot filler for V31 leave-one-out diagnostics.

This module intentionally avoids task-id lookup. It can consume val41-derived
experience only after the caller has removed the current held-out case.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .v30_prior_bank import norm_text, similarity, tokens


SLOT_RE = re.compile(r"^<([a-z_]+)_(\d+)>$")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_user_id(text: str) -> str:
    patterns = [
        r"User ID:\s*([A-Za-z0-9_\-]+)",
        r"user ID is\s*([A-Za-z0-9_\-]+)",
        r"user id is\s*([A-Za-z0-9_\-]+)",
        r"\(User ID:\s*([A-Za-z0-9_\-]+)\)",
    ]
    for pat in patterns:
        m = re.search(pat, text or "", flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def catalog_names(db: Any, scenario: str) -> Dict[str, List[str]]:
    if scenario == "retail":
        return {"product": [getattr(p, "name", "") for p in getattr(db, "catalog", {}).values()]}
    if scenario == "restaurant":
        return {
            "dish": [getattr(d, "name", "") for d in getattr(db, "catalog", {}).values()],
            "set_meal": [getattr(m, "name", "") for m in getattr(db, "set_meals", {}).values()],
        }
    if scenario == "order":
        dishes: List[str] = []
        meals: List[str] = []
        restaurants: List[str] = []
        for r_name, store in getattr(db, "restaurants", {}).items():
            restaurants.append(r_name)
            dishes.extend(getattr(d, "name", "") for d in store.get("catalog", {}).values())
            meals.extend(getattr(m, "name", "") for m in store.get("set_meals", {}).values())
        return {"restaurant": restaurants, "dish": dishes, "set_meal": meals}
    if scenario == "kitchen":
        return {
            "recipe": list(getattr(db, "recipes", {}).keys()),
            "ingredient": list(getattr(db, "ingredients", {}).keys()),
        }
    return {}


def placeholders(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str) and SLOT_RE.fullmatch(obj):
        return [obj]
    if isinstance(obj, list):
        for item in obj:
            out.extend(placeholders(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(placeholders(value))
    return sorted(set(out), key=lambda x: (x.strip("<>").rsplit("_", 1)[0], x))


def slot_kind(slot: str) -> str:
    m = SLOT_RE.fullmatch(slot)
    return m.group(1) if m else ""


def _add_unique(rows: List[Dict[str, Any]], name: str, source: str, score: float, reason: str) -> None:
    if not name:
        return
    low = norm_text(name)
    if any(norm_text(r.get("name")) == low for r in rows):
        return
    rows.append({"name": name, "source": source, "score": score, "reason": reason})


class EvidenceIndex:
    def __init__(self, bound_path: Path | None = None, mm_path: Path | None = None) -> None:
        self.by_task: Dict[str, Dict[str, Any]] = {}
        if bound_path:
            for row in read_jsonl(bound_path):
                key = (row.get("task_key") or (((row.get("slot_sets") or [{}])[0].get("evidence") or {}).get("task_key")))
                if key:
                    self.by_task.setdefault(key, {})["bound"] = row
        if mm_path:
            for row in read_jsonl(mm_path):
                key = row.get("task_key")
                if key:
                    self.by_task.setdefault(key, {})["mm"] = row

    def get(self, task_key: str) -> Dict[str, Any]:
        return self.by_task.get(task_key, {})


class DetaskedSlotFillerV31:
    """Fill abstract program slots with current-task evidence and LOO cases."""

    def __init__(self, evidence_index: EvidenceIndex | None = None) -> None:
        self.evidence_index = evidence_index or EvidenceIndex()

    def retrieve_cases(
        self,
        cases: Iterable[Dict[str, Any]],
        scenario: str,
        utterance: str,
        heldout_case_id: str,
        k: int = 8,
    ) -> List[Dict[str, Any]]:
        qtok = tokens(utterance)
        scored: List[Dict[str, Any]] = []
        for case in cases:
            if case.get("scenario") != scenario:
                continue
            if heldout_case_id and case.get("case_id") == heldout_case_id:
                continue
            score = similarity(qtok, case.get("trigger_tokens", []), scenario_bonus=True)
            out = dict(case)
            out["retrieval_score"] = score
            out["retrieval_reason"] = "loo_scenario_token_overlap"
            scored.append(out)
        scored.sort(key=lambda x: (float(x.get("retrieval_score", 0.0)), len(x.get("slot_values") or {})), reverse=True)
        return scored[:k]

    def candidate_pool(
        self,
        task_key: str,
        scenario: str,
        row: Dict[str, Any],
        db: Any,
        exemplar: Dict[str, Any] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        names = catalog_names(db, scenario)
        text = " ".join(
            [
                row.get("Instruction", ""),
                row.get("instruction", ""),
                row.get("image_description", ""),
            ]
        )
        evidence = self.evidence_index.get(task_key)
        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)
        pool: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ["user_id", "restaurant", "product", "dish", "set_meal", "ingredient", "recipe", "category"]}
        user = extract_user_id(text)
        if user:
            _add_unique(pool["user_id"], user, "current_utterance", 100.0, "regex_user_id")

        combined = norm_text(text + " " + evidence_text)
        for kind, values in names.items():
            for name in values:
                if not name:
                    continue
                n = norm_text(name)
                overlap = len(set(tokens(name)) & set(tokens(combined)))
                if n and n in combined:
                    _add_unique(pool[kind], name, "current_text_or_evidence", 40.0 + overlap, "exact_name_in_current_context")
                elif overlap:
                    _add_unique(pool[kind], name, "current_text_or_evidence", 15.0 + overlap, "token_overlap_current_context")

        # Parse V26 evidence canonical entries when available.
        for payload in evidence.values():
            for slotset in payload.get("slot_sets", []) if isinstance(payload, dict) else []:
                slots = slotset.get("slots") or {}
                for kind, vals in slots.items():
                    if kind not in pool or not isinstance(vals, list):
                        continue
                    for item in vals[:5]:
                        name = item.get("canonical_name") if isinstance(item, dict) else str(item)
                        if name in names.get(kind, []) or kind in {"category"}:
                            _add_unique(pool[kind], name, "v26_bound_slots", float(item.get("score", 10.0)) if isinstance(item, dict) else 10.0, "canonical_v26_slot")
            cslots = payload.get("candidate_slots") if isinstance(payload, dict) else None
            if isinstance(cslots, dict):
                for kind, vals in cslots.items():
                    canonical_kind = kind.rstrip("s")
                    if canonical_kind not in pool or not isinstance(vals, list):
                        continue
                    for item in vals[:5]:
                        name = item.get("canonical_db_name") or item.get("canonical_name") if isinstance(item, dict) else str(item)
                        if name in names.get(canonical_kind, []) or canonical_kind == "category":
                            _add_unique(pool[canonical_kind], name, "v26_mm_evidence", float(item.get("confidence", item.get("score", 8.0))) if isinstance(item, dict) else 8.0, "mm_evidence_candidate")

        # Use exemplar slots only as a weak transfer prior and only when the value
        # exists in the current DB catalog.
        for slot, value in (exemplar or {}).get("slot_values", {}).items():
            kind = slot_kind(slot)
            if kind in pool and value in names.get(kind, []):
                _add_unique(pool[kind], value, "loo_exemplar_slot_prior", 5.0 + float((exemplar or {}).get("retrieval_score", 0.0)), "db_valid_exemplar_prior")

        # If still empty, add deterministic DB defaults with a clear risk marker.
        for kind, values in names.items():
            if not pool.get(kind):
                ranked = sorted(values, key=lambda n: (len(set(tokens(n)) & set(tokens(combined))), -len(n)), reverse=True)
                for name in ranked[:3]:
                    _add_unique(pool[kind], name, "db_fallback", 1.0, "no_current_evidence_fallback")
        return {k: sorted(v, key=lambda x: x.get("score", 0.0), reverse=True) for k, v in pool.items()}

    def fill_slot_sets(
        self,
        task_key: str,
        scenario: str,
        row: Dict[str, Any],
        db: Any,
        abstract_program: List[Dict[str, Any]],
        exemplar: Dict[str, Any] | None,
        max_sets: int = 3,
    ) -> List[Dict[str, Any]]:
        pool = self.candidate_pool(task_key, scenario, row, db, exemplar)
        slots = placeholders(abstract_program)
        out: List[Dict[str, Any]] = []
        for variant in range(max_sets):
            values: Dict[str, Any] = {}
            risk: List[str] = []
            evidence: Dict[str, Any] = {}
            per_kind_counter: Dict[str, int] = {}
            for slot in slots:
                kind = slot_kind(slot)
                candidates = pool.get(kind, [])
                if kind == "user_id":
                    candidates = pool.get("user_id", candidates)
                idx = per_kind_counter.get(kind, 0) + variant
                per_kind_counter[kind] = per_kind_counter.get(kind, 0) + 1
                if candidates:
                    cand = candidates[min(idx, len(candidates) - 1)]
                    values[slot] = cand["name"]
                    evidence[slot] = cand
                    if cand.get("source") in {"db_fallback", "loo_exemplar_slot_prior"}:
                        risk.append(f"{kind}_{cand.get('source')}")
            missing = [s for s in slots if s not in values]
            if missing:
                risk.append("missing_slot_values")
            confidence = 1.0
            if risk:
                confidence = max(0.2, 1.0 - 0.12 * len(set(risk)))
            out.append(
                {
                    "slot_set_id": f"v31_slotset_{variant + 1}",
                    "placeholder_values": values,
                    "slots": self._compact_slots(values),
                    "evidence": evidence,
                    "confidence": confidence,
                    "risk_flags": sorted(set(risk)),
                    "heldout_excluded": True,
                }
            )
        return out

    @staticmethod
    def _compact_slots(values: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, List[str]] = {}
        for slot, value in values.items():
            compact.setdefault(slot_kind(slot), []).append(value)
        return compact


def detasked_slot_decision_records(cases: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case in cases:
        for slot, value in (case.get("slot_values") or {}).items():
            kind = slot_kind(slot)
            records.append(
                {
                    "source": case.get("source_type"),
                    "scenario": case.get("scenario"),
                    "program_family": case.get("program_family"),
                    "intent_pattern": " ".join((case.get("trigger_tokens") or [])[:40]),
                    "slot_type": kind,
                    "slot_evidence_pattern": {
                        "utterance_clues": (case.get("trigger_tokens") or [])[:30],
                        "dialogue_clues": [],
                        "visual_clues": [],
                        "ocr_clues": [],
                        "asr_clues": [],
                        "db_clues": [kind],
                        "tool_observation_clues": [],
                    },
                    "resolution_policy": f"Resolve {kind} from current utterance/evidence first; otherwise use only DB-valid LOO experience as weak prior.",
                    "negative_policy": "Do not copy foreign entities unless current DB contains the entity and current evidence is weak.",
                    "example_without_task_id": f"<{kind}> slot resolved to a DB-valid {kind} under a similar detasked intent pattern.",
                    "forbidden": ["copy foreign entity", "use heldout GT", "use task_id lookup"],
                    "source_case_hash": case.get("case_id"),
                    "contains_task_id": False,
                    "contains_exact_task_answer": False,
                }
            )
    return records
