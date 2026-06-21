# -*- coding: utf-8 -*-
"""V16 non-oracle candidate scoring helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def flatten_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for entry in result.get("tool_calls", []) or []:
        if isinstance(entry, dict) and isinstance(entry.get("calls"), list):
            calls.extend([c for c in entry["calls"] if isinstance(c, dict)])
    return calls


def score_result_non_oracle(result: Dict[str, Any], instruction: str = "", scenario: str = "") -> Dict[str, Any]:
    calls = flatten_tool_calls(result)
    names = [str(c.get("tool_name") or "") for c in calls]
    text = (instruction or "").lower()
    score = 0.0
    if calls:
        score += 5
    if any(re.search(r"^(add|remove|delete|update|modify)_", n) for n in names):
        score += 5
    if any(n.startswith("compute_total_") or n.startswith("tally_total_") or "summary" in n for n in names):
        if any(x in text for x in ["total", "tax", "payment", "nutrition", "summary", "taste"]):
            score += 6
    if scenario == "order" and any("restaurant_name" in (c.get("parameters") or {}) for c in calls):
        score += 4
    if scenario == "retail" and len(calls) > 60:
        score -= 12
    if len(calls) > 40:
        score -= (len(calls) - 40) * 0.2
    natural = json.dumps(result.get("dialogue", []), ensure_ascii=False).lower()
    if "which" in natural and "?" in natural and any(x in natural for x in ["visible", "point", "menu", "product", "dish"]):
        score -= 8
    if len(set(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls)) < len(calls):
        score -= 4
    return {"score": score, "tool_count": len(calls), "tool_names": names}
