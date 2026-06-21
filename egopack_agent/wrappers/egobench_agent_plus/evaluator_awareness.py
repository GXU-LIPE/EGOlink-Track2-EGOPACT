# -*- coding: utf-8 -*-
"""Evaluator-aware scoring reminders for EgoBench Track2.

The card is intentionally short and checklist-shaped. It teaches the service
agent that Track2 is evaluated on both final DB state and process/tool coverage.
"""

from __future__ import annotations

from typing import Optional


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def infer_task_type(user_text: str = "", stage: str = "", scenario: str = "") -> str:
    text = _norm(" ".join([user_text or "", stage or "", scenario or ""]))
    checks = [
        ("replace", ["replace", "swap", "instead", "change"]),
        ("remove", ["remove", "delete", "take out", "cancel"]),
        ("add", ["add", "put", "buy", "order", "include"]),
        ("payment/tax", ["total", "payment", "pay", "tax", "price", "amount"]),
        ("nutrition", ["nutrition", "calorie", "protein", "fat", "carb"]),
        ("recipe/menu/fridge", ["recipe", "ingredient", "fridge", "stock", "menu", "shopping"]),
        ("compare", ["lowest", "cheapest", "highest", "healthiest", "compare", "least", "most"]),
        ("query", ["what", "which", "how many", "where"]),
    ]
    for name, words in checks:
        if any(word in text for word in words):
            return name
    return "general"


def build_evaluator_awareness_card(scenario, stage="", task_type="", final_eval=False):
    scenario = _norm(scenario) or "unknown"
    stage = stage or "unknown"
    task_type = task_type or infer_task_type(stage, stage, scenario)
    lines = [
        "[Track2 Scoring Reminder]",
        "- This is not ordinary QA.",
        "- A task succeeds only when both final database result and required tool process are successful.",
        "- Do not only make the user sound satisfied; execute the necessary tools.",
        "- Use retrieval before state-changing actions when entity, quantity, price, location, menu item, order item, or recipe evidence is uncertain.",
        "- State-changing calls must use canonical entities and pinned user/restaurant identifiers when applicable.",
        "- Do not repeat identical successful state-changing calls.",
        "- If the task asks for total, tax, payment, nutrition, or final amount, execute the corresponding aggregate tool near the end.",
        "- Aggregate tools should normally be final computation steps, not exploratory tools.",
        "- If process stages are missing, continue with tools rather than final natural-language response.",
        "- Each response must be exactly one format: either JSON array of tool calls, or short natural language. Never mix JSON and explanation.",
    ]
    if final_eval:
        lines.append("- In final_eval mode, do not use hidden final scenario JSON metadata.")
    if scenario == "retail":
        lines += [
            "[Retail checklist]",
            "- Identify/retrieve product before cart/list mutation.",
            "- Do not duplicate add/remove unless user explicitly asks for quantity >1 or repeated item.",
            "- Lowest/cheapest/highest/healthiest is filtering, not necessarily final aggregate.",
        ]
    elif scenario == "order":
        lines += [
            "[Order checklist]",
            "- Pin restaurant and user before mutation/aggregate.",
            "- Separate dish_name and set_meal_name.",
            "- Replacement usually needs current order/menu inspection, add target if needed, remove old item if needed, then compute tax/payment if asked.",
            "- No visual follow-up if contact sheet is missing; use retrieval.",
        ]
    elif scenario == "kitchen":
        lines += [
            "[Kitchen checklist]",
            "- Identify recipe/current state, retrieve recipe ingredients once, compare with menu/fridge/stock/list, then mutate or compute only from confirmed evidence.",
            "- Do not broad scan after branch is identified.",
            "- Quantity queries are allowed if branch-critical.",
        ]
    elif scenario == "restaurant":
        lines += [
            "[Restaurant checklist]",
            "- Distinguish dish and set meal.",
            "- Compute nutrition/payment only if requested or required by process.",
        ]
    lines.append(f"[Context] scenario={scenario}; stage={stage}; task_type={task_type}.")
    return "\n".join(lines)
