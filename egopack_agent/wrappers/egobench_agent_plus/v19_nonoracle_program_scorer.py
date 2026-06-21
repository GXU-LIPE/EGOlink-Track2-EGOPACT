#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-oracle V19 candidate scorer.

This scorer deliberately does not read val41 GT. It scores structural and
evidence-risk features only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")
AGGREGATE_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
}


def tool_name(step: Dict[str, Any]) -> str:
    return str(step.get("tool_name") or "")


def flatten_tools(program: List[Dict[str, Any]]) -> List[str]:
    return [tool_name(x) for x in program if tool_name(x)]


def has_foreign_copy(candidate: Dict[str, Any]) -> bool:
    return "forbidden_copied_slots" in (candidate.get("risk_flags") or [])


def program_has_retrieval_before_mutation(names: List[str]) -> bool:
    first_mut = next((i for i, n in enumerate(names) if MUTATION_RE.search(n)), None)
    if first_mut is None:
        return True
    return any(n.startswith(("get_", "find_", "filter_", "search_")) for n in names[:first_mut]) or first_mut == 0


def score_candidate(candidate: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    program = candidate.get("tool_program") or []
    names = flatten_tools(program)
    risk = list(candidate.get("risk_flags") or [])
    score = 0.0
    reasons = []
    if names:
        score += 1.0
        reasons.append("nonempty_program")
    if candidate.get("source") == "v14":
        score += 2.0
        reasons.append("v14_historical_signal")
    if candidate.get("source") == "v10":
        score += 0.8
        reasons.append("v10_historical_signal")
    if candidate.get("source") in {"case_top1", "case_top3", "diverse", "repair"}:
        score += 1.2
        reasons.append("gt100_case_program")
    confs = candidate.get("slot_confidence") or {}
    if confs:
        avg_conf = sum(float(v) for v in confs.values()) / len(confs)
        score += 2.0 * avg_conf
        reasons.append(f"slot_confidence={avg_conf:.3f}")
    if not has_foreign_copy(candidate):
        score += 1.2
        reasons.append("no_foreign_slot_copy")
    else:
        score -= 4.0
        reasons.append("foreign_slot_copy_risk")
    if program_has_retrieval_before_mutation(names):
        score += 0.5
        reasons.append("retrieval_before_mutation_ok")
    else:
        score -= 1.0
        risk.append("mutation_before_retrieval")
    if any(MUTATION_RE.search(n) for n in names):
        score += 0.4
        reasons.append("mutation_closure_present")
    if any(n in AGGREGATE_TOOLS for n in names):
        score += 0.4
        reasons.append("aggregate_closure_present")
    if len(names) > 60:
        score -= 2.0
        risk.append("tool_count_extreme")
    elif len(names) > 30:
        score -= 0.8
        risk.append("tool_count_high")
    shape_conf = float(candidate.get("program_shape_confidence") or 0.0)
    score += shape_conf
    if shape_conf:
        reasons.append(f"shape_confidence={shape_conf:.3f}")
    if "uncertain_slots" in risk:
        score -= 1.0
    return {
        "candidate_id": candidate.get("candidate_id"),
        "score": score,
        "reasons": reasons,
        "risk_flags": sorted(set(risk)),
    }


def rank_candidates(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    scored = []
    for cand in candidates:
        row = score_candidate(cand, context)
        out = dict(cand)
        out["score"] = row["score"]
        out["score_reasons"] = row["reasons"]
        out["risk_flags"] = row["risk_flags"]
        scored.append(out)
    scored.sort(key=lambda x: (x.get("score", 0), -len(x.get("tool_program") or [])), reverse=True)
    return scored
