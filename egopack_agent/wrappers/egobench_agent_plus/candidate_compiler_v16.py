# -*- coding: utf-8 -*-
"""V16 candidate compiler hooks.

Current integration is intentionally light-touch: it exposes prompt text and a
normalization hook for future candidates without blocking the existing runner.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def compiler_prompt_v16() -> str:
    return (
        "[V16 Candidate Compiler]\n"
        "- Generate the minimal next action that advances the GT100-like skeleton.\n"
        "- Do not output exploratory broad scans when a constrained resolver can fill the slot.\n"
        "- If a mutation/aggregate closure is missing, prefer repairing the closure over asking the user."
    )


def repair_v16_tool_output(normalized: str, scenario: str, state: Dict[str, Any], turn: int) -> Tuple[str, Dict[str, Any]]:
    """No-risk structural hook. It currently annotates only; core repairs stay in db_guard/canonical_resolver."""
    try:
        arr = json.loads(normalized)
    except Exception:
        return normalized, {"applied": False, "reason": "not_json"}
    if not isinstance(arr, list):
        return normalized, {"applied": False, "reason": "not_list"}
    return normalized, {"applied": False, "reason": "pass_through", "candidate_count": len(arr), "scenario": scenario, "turn": turn}
