# -*- coding: utf-8 -*-
"""Working-memory cap for Track2 V7 prompts."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def compact_history(history: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, str]]:
    out = []
    for msg in (history or [])[-limit:]:
        content = str(msg.get("content", ""))
        out.append({"role": str(msg.get("role", "")), "content": content[:700]})
    return out


def summarize_ledgers(state: Dict[str, Any]) -> Dict[str, Any]:
    ledger = state.get("successful_mutation_ledger") or {}
    blocked = state.get("blocked_calls") or []
    return {
        "successful_mutations": min(len(ledger), 20),
        "blocked_calls": len(blocked),
        "tool_call_count": state.get("tool_call_count", 0),
        "kitchen_stage": state.get("kitchen_stage"),
    }


def cap_list(values: Any, limit: int = 5) -> List[str]:
    if not isinstance(values, list):
        return [str(values)] if values else []
    out = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def build_working_memory_prompt(scenario: str, state: Dict[str, Any], history: List[Dict[str, Any]], process_state: Dict[str, Any], visual_slots: Dict[str, Any]) -> str:
    pins = state.get("pins") or {}
    top_slots = {}
    for key in ("restaurant_name", "dish_candidates", "set_meal_candidates", "ingredient_candidates", "product_candidates", "current_recipe_candidates", "pointed_entity"):
        val = visual_slots.get(key)
        if val:
            top_slots[key] = cap_list(val, 3) if isinstance(val, list) else str(val)[:200]
    prompt = {
        "current_scenario": scenario,
        "current_process_stage": process_state.get("current_stage"),
        "pinned": pins,
        "allowed_next_tool_families": process_state.get("allowed_tool_families", [])[:5],
        "active_candidate_tools": process_state.get("allowed_tool_set", [])[:5],
        "mutation_ledger_summary": summarize_ledgers(state),
        "top_visual_slots": top_slots,
        "recent_history": compact_history(history, 3),
    }
    if scenario == "kitchen" and state.get("tool_call_count", 0) > 25:
        prompt["mode"] = "STOP_EXPLORING"
    if scenario == "kitchen" and state.get("tool_call_count", 0) > 35:
        prompt["mode"] = "CONSERVATIVE_ONLY"
    return "Compact working memory:\n" + json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))[:2600]
