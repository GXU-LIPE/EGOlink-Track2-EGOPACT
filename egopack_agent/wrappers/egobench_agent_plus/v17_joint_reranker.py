# -*- coding: utf-8 -*-
"""V17 joint reranker scoring helpers for smoke analysis."""

from __future__ import annotations

import json
from typing import Any, Dict


def score_candidate(candidate_json: str, scenario: str) -> Dict[str, Any]:
    try:
        obj = json.loads(candidate_json)
        calls = obj if isinstance(obj, list) else [obj]
    except Exception:
        return {"score": -100, "reason": "invalid_json"}
    score = 0
    names = [str(c.get("tool_name", "")).lower() for c in calls if isinstance(c, dict)]
    if names:
        score += 5
    if scenario == "order" and any("order" in n or n.startswith("compute_total") for n in names):
        score += 3
    if scenario == "retail" and any("cart" in n or "product" in n for n in names):
        score += 2
    if len(names) > 12:
        score -= 2
    return {"score": score, "tool_names": names}


__all__ = ["score_candidate"]
