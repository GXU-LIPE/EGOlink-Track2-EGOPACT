# -*- coding: utf-8 -*-
"""V9 multi-candidate deterministic reranker.

This is intentionally conservative: the current runner usually supplies one
candidate, but the scorer and telemetry are ready for high-risk multi-candidate
generation without using GT.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .counterfactual_db_simulator import assess_batch
from .deepseek_cross_validator import crosscheck, should_crosscheck
from .guard_policy import classify_policy
from .process_coverage_verifier import verify_process_coverage
from .v8_event_logger import enabled, write_v8_event


def _json_candidate(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    try:
        return json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(candidate)


def _calls_candidate(candidate: Any) -> List[Dict[str, Any]]:
    if isinstance(candidate, list):
        return [x for x in candidate if isinstance(x, dict)]
    if isinstance(candidate, dict):
        return [candidate]
    if isinstance(candidate, str):
        try:
            parsed = json.loads(candidate)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _scan_risk(candidate: Any) -> float:
    text = _json_candidate(candidate).lower()
    calls = _calls_candidate(candidate)
    attr_calls = 0
    products = set()
    for call in calls:
        if not isinstance(call, dict):
            continue
        params = call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
        name = str(call.get("tool_name") or "")
        blob = " ".join([name] + [str(k) for k in params] + [str(v) for v in params.values()]).lower()
        if any(term in blob for term in ("price", "tax", "discount", "nutrition", "nutritional")):
            attr_calls += 1
            for key in ("product_name", "dish_name", "set_meal_name", "ingredient_name"):
                if params.get(key):
                    products.add(str(params.get(key)).strip().lower())
    risk = 0.0
    if text.count("price") + text.count("tax") + text.count("discount") + text.count("nutrition") > 8:
        risk += 0.3
    if attr_calls > 8:
        risk += 0.25
    if len(products) > 5:
        risk += 0.25
    if len(calls) > 20:
        risk += 0.2
    if "could you" in text or "what is shown" in text or "share the" in text:
        risk += 0.4
    if text.count("get_") + text.count("find_") > 18:
        risk += 0.2
    return min(risk, 0.9)


def score_candidate(candidate: Any, scenario: str, state: Dict[str, Any], turn: int = 0) -> Dict[str, Any]:
    text = _json_candidate(candidate)
    calls = _calls_candidate(candidate)
    coverage = verify_process_coverage(scenario, calls, state)
    cf = assess_batch(calls, scenario, state)
    policy = classify_policy(text, scenario, {}, state)
    cf_risk = max([float(x.get("decision", {}).get("risk_score", 0.0)) for x in cf] or [0.0])
    scan_risk = _scan_risk(candidate)
    score = float(coverage.get("process_coverage_score", 0.0))
    score -= cf_risk
    score -= scan_risk
    if coverage.get("missing_process_stage"):
        score -= 0.2
    if coverage.get("tool_family_mismatch"):
        score -= 0.2
    if scenario in {"order", "retail", "restaurant"} and calls:
        has_retrieval = any(str(c.get("tool_name", "")).startswith(("get_", "find_", "list_", "search_", "retrieve_")) for c in calls)
        has_mutation = any(str(c.get("tool_name", "")).startswith(("add_", "remove_", "update_", "delete_")) or "_to_" in str(c.get("tool_name", "")) or "_from_" in str(c.get("tool_name", "")) for c in calls)
        has_aggregate = any(str(c.get("tool_name", "")).startswith("compute_total") for c in calls)
        if has_retrieval:
            score += 0.08
        if has_mutation and coverage.get("missing_process_stage"):
            score -= 0.15
        if scenario == "order" and has_mutation and not has_aggregate and (state or {}).get("order_requested_final_aggregate"):
            score -= 0.25
        if scenario == "retail" and not has_retrieval and (has_mutation or has_aggregate):
            score -= 0.25
    if policy.get("hard_blocks"):
        score -= 1.0
    if "possible_broad_scan_or_overlarge_batch" in policy.get("soft_warnings", []):
        score -= 0.3

    ds = None
    ds_risk = 0.0
    payload = {"scenario": scenario, "candidate": calls or candidate, "coverage": coverage, "policy": policy, "counterfactual": cf}
    if should_crosscheck(scenario, state, payload, risk_score=max(cf_risk, scan_risk)):
        ds = crosscheck(payload, state, turn)
        risk_label = str(ds.get("risk", "low")).lower()
        ds_risk = {"low": 0.0, "medium": 0.2, "high": 0.5}.get(risk_label, 0.1)
        score -= ds_risk
        if ds.get("recommended_action") == "reject":
            score -= 0.5
    return {
        "score": round(score, 4),
        "coverage": coverage,
        "counterfactual": cf,
        "policy": policy,
        "risk": round(max(cf_risk, scan_risk, ds_risk), 4),
        "deepseek": ds,
    }


def select_candidate(candidates: List[Any], scenario: str, state: Dict[str, Any], turn: int) -> Dict[str, Any]:
    if not candidates:
        return {"selected_index": -1, "selected": None, "scores": []}
    if not enabled("TRACK2_ENABLE_MULTICANDIDATE") and not enabled("TRACK2_ENABLE_DEEPSEEK_CROSSCHECK"):
        return {"selected_index": 0, "selected": candidates[0], "scores": []}
    scores = [score_candidate(c, scenario, state, turn=turn) for c in candidates]
    idx = max(range(len(scores)), key=lambda i: scores[i]["score"])
    write_v8_event(
        state,
        "v9_multicandidate_reranker",
        "select",
        "multicandidate_score",
        turn=turn,
        selected_candidate=idx,
        scores=scores,
        rejected_candidates=[i for i in range(len(candidates)) if i != idx],
    )
    return {"selected_index": idx, "selected": candidates[idx], "scores": scores, "rejected_candidates": [i for i in range(len(candidates)) if i != idx]}
