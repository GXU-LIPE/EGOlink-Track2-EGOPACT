# -*- coding: utf-8 -*-
"""V9 deterministic retail candidate narrowing guard.

The goal is to stop low-value catalog sweeps. It does not choose final answers;
it nudges the agent toward candidate narrowing before expensive attributes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .v8_event_logger import enabled, write_v8_event


ATTRIBUTE_TERMS = ("price", "tax", "discount", "nutrition", "nutritional")
NARROW_TERMS = ("category", "country", "origin", "taste", "profile", "brand", "name", "product")


def is_enabled() -> bool:
    return enabled("TRACK2_ENABLE_RETAIL_NARROWER")


def build_retail_narrowing_prompt() -> str:
    if not is_enabled():
        return ""
    return "\n".join([
        "[V9 Retail Candidate Narrower]",
        "- For retail visual/filter tasks, first narrow candidates by category, country/origin, taste/profile, brand/name, or product text.",
        "- If more than five products may match, do not scan price/tax/discount/nutrition for the whole catalog.",
        "- Query price/tax/discount/nutrition only for narrowed candidates.",
        "- 'lowest', 'cheapest', 'highest', and 'healthiest' are ranking objectives. They do not by themselves require cart mutation or final aggregate.",
        "- Prefer one candidate-narrowing retrieval step over many attribute calls.",
    ])


def _name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def _params(call: Dict[str, Any]) -> Dict[str, Any]:
    params = call.get("parameters") or call.get("arguments") or {}
    return params if isinstance(params, dict) else {}


def _is_attr_call(name: str, params: Dict[str, Any]) -> bool:
    text = " ".join([name] + [str(k) for k in params] + [str(v) for v in params.values()]).lower()
    return any(term in text for term in ATTRIBUTE_TERMS)


def _is_narrow_call(name: str, params: Dict[str, Any]) -> bool:
    text = " ".join([name] + [str(k) for k in params] + [str(v) for v in params.values()]).lower()
    return any(term in text for term in NARROW_TERMS)


def _product_key(params: Dict[str, Any]) -> str:
    for key in ("product_name", "name", "item_name", "product"):
        if params.get(key):
            return str(params.get(key)).strip().lower()
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def apply_retail_narrower(calls_obj: Any, state: Dict[str, Any], turn: int) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not is_enabled() or state.get("scenario") != "retail":
        return calls_obj, [], []
    calls = calls_obj if isinstance(calls_obj, list) else ([calls_obj] if calls_obj else [])
    attr_calls = [c for c in calls if isinstance(c, dict) and _is_attr_call(_name(c), _params(c))]
    narrow_calls = [c for c in calls if isinstance(c, dict) and _is_narrow_call(_name(c), _params(c))]
    product_keys = {_product_key(_params(c)) for c in attr_calls}
    synth: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    history = state.setdefault("retail_narrowing_history", {"narrow_steps": 0, "attribute_products": []})
    if narrow_calls:
        history["narrow_steps"] = int(history.get("narrow_steps", 0)) + len(narrow_calls)
        write_v8_event(state, "retail_candidate_narrower", "allow", "retail_candidate_narrowed", turn=turn, candidate_count=len(product_keys), tool_names=[_name(c) for c in narrow_calls])

    broad = len(attr_calls) > 5 or (len(product_keys) > 5 and not narrow_calls and not history.get("narrow_steps"))
    if not broad:
        history.setdefault("attribute_products", []).extend(sorted(product_keys)[:20])
        return calls_obj, synth, decisions

    filtered = []
    kept_attr = 0
    for call in calls:
        if not isinstance(call, dict):
            continue
        if _is_attr_call(_name(call), _params(call)):
            kept_attr += 1
            if kept_attr > 5:
                decisions.append({"tool_name": _name(call), "decision": "block", "reason": "retail_broad_scan_blocked"})
                synth.append({
                    "role": "tool",
                    "content": "Retail broad attribute scan blocked: narrow candidates by category/country/taste/profile/brand/name before querying price/tax/discount/nutrition.",
                    "blocked": True,
                    "tool_name": _name(call),
                })
                continue
        filtered.append(call)
    write_v8_event(
        state,
        "retail_candidate_narrower",
        "block",
        "retail_broad_scan_blocked",
        turn=turn,
        whether_blocked=True,
        risk_score=0.7,
        candidate_count=len(product_keys),
        blocked_count=len(calls) - len(filtered),
    )
    return filtered if isinstance(calls_obj, list) else (filtered[0] if filtered else []), synth, decisions
