# -*- coding: utf-8 -*-
"""V9 visual grounding resolver for text-only EgoBench visual references.

This module does not hardcode dev answers. It turns vague visual follow-up
questions into retrieval-first behavior and adds compact candidate-narrowing
rules to the prompt.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict

from .v8_event_logger import enabled, write_v8_event


VISUAL_FOLLOWUP_RE = re.compile(
    r"\b("
    r"could you|can you|please|what|which|share|provide|tell me"
    r").{0,120}\b("
    r"see|shown|visible|image|picture|video|frame|label|brand|menu|category|dish|product|bottle|box|package|front label"
    r")",
    re.I,
)


def is_enabled() -> bool:
    return enabled("TRACK2_ENABLE_VISUAL_GROUNDING_RESOLVER")


def _scenario_hint(scenario: str) -> str:
    if scenario == "retail":
        return (
            "Use retail tools to narrow candidates from category/country/taste/profile/brand/name cues. "
            "Ask category or product retrieval first, then price/tax/discount/nutrition only for narrowed candidates."
        )
    if scenario == "order":
        return (
            "Pin or infer restaurant from context, inspect current order/menu, retrieve dish/set-meal/category candidates "
            "inside that restaurant, then mutate/aggregate if requested."
        )
    if scenario == "restaurant":
        return (
            "Use dish/set-meal/menu retrieval from available restaurant context. Resolve dish vs set_meal before mutation or aggregate."
        )
    if scenario == "kitchen":
        return (
            "Use recipe/menu/shopping-list retrieval and recipe ingredients once. Do not ask for visual details when tools can verify."
        )
    return "Use tools to retrieve candidates before asking for missing visual labels."


def build_visual_grounding_prompt(scenario: str) -> str:
    if not is_enabled():
        return ""
    return "\n".join([
        "[V9 Visual Grounding Resolver]",
        "- Do not ask the simulated user to provide product/dish/category/label/menu text when the benchmark visual reference is ambiguous.",
        "- If contact sheet or visual_state is missing, infer a small candidate set from task text, memory cards, prior observations, and retrieval tools.",
        "- For pointing/label/shape/color/category tasks, generate top-k entity candidates and verify them with scenario tools.",
        "- Narrow first by scenario/category/country/taste/profile/filter/restaurant, then query price/tax/discount/nutrition only for the narrowed candidates.",
        "- When confidence is low, take the safest retrieval step; do not end the task with a visual follow-up question.",
        f"- Scenario-specific grounding: {_scenario_hint(scenario)}",
    ])


def rewrite_visual_followup(reply: str, scenario: str, state: Dict[str, Any] | None, turn: int | None) -> Dict[str, Any]:
    if not is_enabled():
        return {"allow": True}
    text = str(reply or "")
    if scenario not in {"retail", "restaurant", "order", "kitchen"}:
        return {"allow": True}
    if not VISUAL_FOLLOWUP_RE.search(text):
        return {"allow": True}
    replacement = (
        "I will use the available context and retrieval tools to narrow the visual candidate instead of asking for more visual details."
    )
    write_v8_event(
        state,
        "visual_grounding_resolver",
        "rewrite",
        "visual_followup_blocked",
        turn=turn,
        before_action=text,
        after_action=replacement,
        whether_repaired=True,
        whether_blocked=True,
        risk_score=0.6,
        scenario=scenario,
    )
    return {"allow": False, "replacement": replacement, "event": "visual_followup_blocked"}


def visual_state_summary(state: Dict[str, Any] | None) -> Dict[str, Any]:
    state = state or {}
    return {
        "pins": state.get("pins"),
        "task_id": state.get("task_id"),
        "tool_call_count": state.get("tool_call_count"),
        "last_grounding": state.get("last_visual_grounding"),
    }
