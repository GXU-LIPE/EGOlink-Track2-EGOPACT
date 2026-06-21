# -*- coding: utf-8 -*-
"""Deterministic candidate verifier for Track2 V6.

This module does not use ground truth. It scores candidate service-agent
outputs using only schema validity and episode safety signals.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .db_guard import is_state_changing_tool, mutation_signature, signature_key
from .json_repair import repair_tool_json
from .tool_validator import validate_tool_json


def score_candidate(text: str, scenario: str, episode_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    score = 0
    reasons: List[str] = []
    repaired_ok, repaired, repair_report = repair_tool_json(text)
    if repaired_ok:
        score += 10
        reasons.append("json_repair_ok")
    else:
        return {"score": -100, "reasons": ["invalid_json"], "repair_report": repair_report, "normalized": text}
    valid, normalized, validation = validate_tool_json(repaired, scenario)
    if valid:
        score += 30
        reasons.append("schema_valid")
    else:
        score -= 15 * int(validation.get("invalid_tool_name_count", 0))
        score -= 10 * int(validation.get("missing_required_param_count", 0))
        score -= 10 * int(validation.get("wrong_param_type_count", 0))
        reasons.append("schema_issues")
    try:
        calls = json.loads(normalized)
    except Exception:
        calls = []
    risky = 0
    duplicates = 0
    if isinstance(calls, list):
        ledger = (episode_state or {}).get("successful_mutation_ledger", {})
        seen = set()
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("tool_name", ""))
            params = call.get("parameters", {})
            if is_state_changing_tool(name, scenario):
                risky += 1
                sig = mutation_signature(name, params if isinstance(params, dict) else {}, scenario, episode_state or {})
                key = signature_key(sig, include_quantity=False)
                if key in ledger or key in seen:
                    duplicates += 1
                seen.add(key)
    score -= risky * 2
    score -= duplicates * 25
    if duplicates:
        reasons.append("duplicate_mutation_risk")
    if risky:
        reasons.append("state_changing")
    return {"score": score, "reasons": reasons, "validation": validation, "normalized": normalized}


def choose_candidate(candidates: List[str], scenario: str, episode_state: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any]]:
    scored = [(score_candidate(c, scenario, episode_state), c) for c in candidates]
    scored.sort(key=lambda item: item[0]["score"], reverse=True)
    best_score, best_text = scored[0]
    best_score["all_scores"] = [s for s, _ in scored]
    return best_text, best_score

