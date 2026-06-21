#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prior-retrieval service agent for V30."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List

from .v30_prior_bank import ExperiencePriorBank, norm_text


def _materialize(value: Any, slot_values: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return slot_values.get(value, value)
    if isinstance(value, list):
        return [_materialize(v, slot_values) for v in value]
    if isinstance(value, dict):
        return {k: _materialize(v, slot_values) for k, v in value.items()}
    return value


def materialize_program(abstract_tool_chain: List[Dict[str, Any]], slot_values: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for call in abstract_tool_chain:
        out.append(
            {
                "tool_name": call.get("tool_name"),
                "parameters": _materialize(copy.deepcopy(call.get("parameters") or {}), slot_values),
            }
        )
    return out


def extract_user_id(utterance: str) -> str:
    m = re.search(r"User ID:\s*([A-Za-z0-9_\-]+)", utterance or "")
    return m.group(1) if m else ""


def _catalog_names(db: Any, scenario: str) -> Dict[str, List[str]]:
    if scenario == "retail":
        return {"product": [getattr(p, "name", "") for p in getattr(db, "catalog", {}).values()]}
    if scenario == "restaurant":
        return {
            "dish": [getattr(d, "name", "") for d in getattr(db, "catalog", {}).values()],
            "set_meal": [getattr(m, "name", "") for m in getattr(db, "set_meals", {}).values()],
        }
    if scenario == "order":
        dishes, meals, restaurants = [], [], []
        for r, store in getattr(db, "restaurants", {}).items():
            restaurants.append(r)
            dishes.extend(getattr(d, "name", "") for d in store.get("catalog", {}).values())
            meals.extend(getattr(m, "name", "") for m in store.get("set_meals", {}).values())
        return {"restaurant": restaurants, "dish": dishes, "set_meal": meals}
    if scenario == "kitchen":
        return {
            "recipe": list(getattr(db, "recipes", {}).keys()),
            "ingredient": list(getattr(db, "ingredients", {}).keys()),
        }
    return {}


def heuristic_slot_values(abstract_program: List[Dict[str, Any]], utterance: str, db: Any, scenario: str) -> Dict[str, Any]:
    """Slot-only fallback. This does not use concrete dev case slots."""
    names = _catalog_names(db, scenario)
    text = norm_text(utterance)
    slot_values: Dict[str, Any] = {}
    user = extract_user_id(utterance)
    counters: Dict[str, int] = {}

    def set_slot(slot: str) -> None:
        kind = slot.strip("<>").rsplit("_", 1)[0]
        if kind == "user_id" and user:
            slot_values[slot] = user
            return
        pool = names.get(kind, [])
        ranked = sorted(pool, key=lambda n: (1 if norm_text(n) in text else 0, len(set(norm_text(n).split()) & set(text.split()))), reverse=True)
        if ranked:
            counters[kind] = counters.get(kind, 0) + 1
            slot_values[slot] = ranked[min(counters[kind] - 1, len(ranked) - 1)]

    def scan(v: Any) -> None:
        if isinstance(v, str) and re.fullmatch(r"<[a-z_]+_\d+>", v):
            set_slot(v)
        elif isinstance(v, list):
            for x in v:
                scan(x)
        elif isinstance(v, dict):
            for x in v.values():
                scan(x)

    scan(abstract_program)
    return slot_values


class PriorRetrievalServiceAgentV30:
    def __init__(self, bank: ExperiencePriorBank, mode: str = "slot_only") -> None:
        self.bank = bank
        self.mode = mode

    def build_candidates(self, task_key: str, scenario: str, row: Dict[str, Any], db: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        utterance = row.get("Instruction", "")
        include_dev = self.mode in {"dev_experience", "dev_calibrated"}
        retrieved = self.bank.retrieve(scenario, utterance, k=top_k, include_dev_cases=include_dev)
        candidates: List[Dict[str, Any]] = []
        for rank, prior_case in enumerate(retrieved, 1):
            abstract_program = prior_case.get("abstract_tool_chain") or []
            if include_dev and prior_case.get("slot_values"):
                slots = dict(prior_case.get("slot_values") or {})
                slot_source = "retrieved_dev_experience_case"
            else:
                slots = heuristic_slot_values(abstract_program, utterance, db, scenario)
                slot_source = "current_db_heuristic"
            program = materialize_program(abstract_program, slots)
            candidate_id = f"V30_PRIOR_{self.mode}_{rank}_{prior_case.get('program_family','family')}"[:180]
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source": f"V30_PRIOR_AGENT_{self.mode}",
                    "tool_program": program,
                    "prior_id": prior_case.get("prior_id"),
                    "prior_case_id": prior_case.get("case_id"),
                    "program_family": prior_case.get("program_family"),
                    "retrieval_score": prior_case.get("retrieval_score", 0.0),
                    "retrieval_reason": prior_case.get("retrieval_reason"),
                    "slot_source": slot_source,
                    "slot_values_used": slots,
                    "risk_flags": ["dev_only_prior"] if prior_case.get("not_final_safe") else [],
                    "trace": {
                        "called_prior_retriever": True,
                        "called_service_agent_slot_filler": True,
                        "called_program_materializer": True,
                        "called_dryrun_selector": True,
                        "mode": self.mode,
                        "uses_val41_gt_experience": bool(prior_case.get("uses_val41_gt")),
                        "not_final_safe": bool(prior_case.get("not_final_safe")),
                    },
                }
            )
        return candidates
