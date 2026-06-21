# -*- coding: utf-8 -*-
"""Three-level V9 guard policy scaffold.

Existing db_guard/schema/canonicalization logic remains authoritative. This
module classifies events into hard_block, soft_warning, and rerank_signal for
prompt feedback, logging, and later reranking without adding brittle FSM blocks.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


STATE_CHANGING_HINTS = ("add_", "remove_", "update_", "delete_", "clear_", "set_")
AGGREGATE_HINTS = ("compute_", "total", "tax", "payment", "nutrition")
RETRIEVAL_HINTS = ("get_", "find_", "search_", "retrieve_", "list_")
BROAD_SCAN_CALL_THRESHOLD = 12
HIGH_RISK_BROAD_SCAN_THRESHOLD = 24


def _stable_params(params: Dict[str, Any]) -> str:
    try:
        return json.dumps(params, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(params)


def _load_calls(tool_json: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(tool_json)
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    return [x for x in data if isinstance(x, dict)]


def _name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def _params(call: Dict[str, Any]) -> Dict[str, Any]:
    params = call.get("parameters") or call.get("arguments") or {}
    return params if isinstance(params, dict) else {}


def is_mutation(name: str) -> bool:
    low = name.lower()
    return low.startswith(STATE_CHANGING_HINTS) or any(x in low for x in ("add_to", "remove_from"))


def is_aggregate(name: str) -> bool:
    low = name.lower()
    return low.startswith("compute_") or any(x in low for x in AGGREGATE_HINTS)


def is_retrieval(name: str) -> bool:
    low = name.lower()
    return low.startswith(RETRIEVAL_HINTS)


def classify_policy(tool_json: str, scenario: str, validation: Dict[str, Any] | None = None, episode_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    validation = validation or {}
    episode_state = episode_state or {}
    calls = _load_calls(tool_json)
    hard_blocks: List[str] = []
    soft_warnings: List[str] = []
    rerank_signals: List[str] = []

    if validation.get("invalid_tool_name_count", 0):
        hard_blocks.append("nonexistent_tool")
    if validation.get("missing_required_params"):
        hard_blocks.append("required_parameter_missing")
    if not calls and tool_json.strip():
        hard_blocks.append("illegal_or_unparseable_json")

    names = [_name(c) for c in calls]
    has_mutation = any(is_mutation(n) for n in names)
    has_aggregate = any(is_aggregate(n) for n in names)
    has_retrieval = any(is_retrieval(n) for n in names)
    repeated_call_count = len(calls) - len({(_name(c), _stable_params(_params(c))) for c in calls})

    if has_mutation and not has_retrieval:
        soft_warnings.append("retrieval_recommended_before_mutation")
    if has_aggregate and len(calls) > 1 and not names[-1].lower().startswith("compute_"):
        soft_warnings.append("aggregate_may_not_be_final_step")
    if has_aggregate and not has_mutation and len(episode_state.get("executed_tool_calls") or []) < 2:
        soft_warnings.append("aggregate_may_be_too_early")
    if has_aggregate and repeated_call_count:
        soft_warnings.append("duplicate_aggregate_or_retrieval_in_batch")
        rerank_signals.append("loop_risk")
    if len(calls) >= BROAD_SCAN_CALL_THRESHOLD:
        soft_warnings.append("possible_broad_scan_or_overlarge_batch")
        rerank_signals.append("broad_scan_risk")
    if len(calls) >= HIGH_RISK_BROAD_SCAN_THRESHOLD:
        rerank_signals.append("high_tool_count_candidate")

    if scenario == "order":
        for call in calls:
            name = _name(call).lower()
            params = _params(call)
            if (is_mutation(name) or is_aggregate(name)) and "restaurant_name" not in params:
                soft_warnings.append("order_restaurant_pin_or_parameter_missing")
            if "set_meal" in str(params).lower() and "dish_name" in params and "set_meal" not in name:
                soft_warnings.append("possible_dish_set_meal_confusion")
    if scenario == "kitchen":
        if sum(1 for n in names if "recipe" in n.lower() or "ingredient" in n.lower()) >= 3:
            soft_warnings.append("possible_kitchen_broad_scan")
    if scenario == "retail":
        retrieval_count = sum(1 for n in names if is_retrieval(n))
        mutation_count = sum(1 for n in names if is_mutation(n))
        if retrieval_count >= 10 or mutation_count >= 6:
            soft_warnings.append("possible_retail_catalog_sweep")
            rerank_signals.append("retail_catalog_sweep_risk")

    if has_mutation:
        rerank_signals.append("db_state_risk")
    if has_aggregate:
        rerank_signals.append("process_coverage_aggregate_present")
    if soft_warnings:
        rerank_signals.append("soft_warning_present")

    level = "allow"
    if hard_blocks:
        level = "hard_block"
    elif soft_warnings:
        level = "soft_warning"
    return {
        "level": level,
        "hard_blocks": sorted(set(hard_blocks)),
        "soft_warnings": sorted(set(soft_warnings)),
        "rerank_signals": sorted(set(rerank_signals)),
        "num_calls": len(calls),
        "tool_names": names,
    }


def build_soft_guard_prompt() -> str:
    return "\n".join([
        "[V9 Soft Guard Policy]",
        "- Hard blocks are only for invalid JSON, nonexistent tools, unrecoverable missing required parameters, hidden-final access, duplicate successful mutation, or unsafe type mismatch.",
        "- Soft warnings do not block: missing process stage, early aggregate, retrieval recommended before mutation, uncertain canonical entity, dish/set-meal confusion, broad kitchen scan, visual uncertainty, missing final aggregate.",
        "- Broad catalog sweeps and same-parameter aggregate retries are high-risk. Narrow the candidate set or use item-level retrieval instead of scanning every possible item.",
        "- If a soft warning applies, prefer a lower-risk tool step, but do not follow a rigid FSM when the current evidence supports another valid process.",
        "- Rerank signals are advisory: process coverage, result risk, DB risk, loop risk, broad-scan risk, and trajectory risk.",
    ])
