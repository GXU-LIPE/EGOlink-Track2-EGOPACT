# -*- coding: utf-8 -*-
"""V16 executable GT100-distilled process-policy prompt cards."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def _load_jsonl(path: Path, limit: int = 20) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
        if len(rows) >= limit:
            break
    return rows


def infer_v16_task_type(goal: str, scenario: str) -> str:
    text = str(goal or "").lower()
    if any(x in text for x in ["if ", "otherwise", "else", "whether", "exceeds", "less than", "greater than"]):
        if any(x in text for x in ["add", "remove", "cart", "order", "menu", "shopping list"]):
            return "branch-then-mutation+aggregate"
    if any(x in text for x in ["add", "remove", "delete", "update", "cart", "order", "menu", "shopping list"]):
        if any(x in text for x in ["total", "tax", "payment", "nutrition", "price"]):
            return "mutation+aggregate"
        return "cart/order/menu mutation"
    if any(x in text for x in ["total", "tax", "payment", "nutrition", "calorie", "carbohydrate", "protein", "fat"]):
        return "aggregate-required"
    if any(x in text for x in ["highest", "lowest", "cheapest", "most", "least", "maximum", "minimum"]):
        return "ranking/filtering"
    if any(x in text for x in ["pointed", "visible", "image", "video", "looked at", "shelf", "menu"]):
        return "visual-entity query"
    return "query-only"


def _select_templates(automata: Dict[str, Any], scenario: str, task_type: str, limit: int = 4) -> List[Dict[str, Any]]:
    templates = automata.get("templates") or []
    scored = []
    for item in templates:
        score = 0
        if item.get("scenario") == scenario:
            score += 4
        if item.get("task_type") == task_type:
            score += 4
        if scenario == "order" and item.get("scenario") == "order":
            score += 1
        scored.append((score, item.get("count", 0), item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in scored if x[0] > 0][:limit]


def build_v16_process_policy_prompt(scenario: str) -> str:
    if os.environ.get("TRACK2_ENABLE_V16_PROCESS_POLICY") != "1":
        return ""
    bank = Path(os.environ.get("TRACK2_V16_DISTILL_DIR") or (CODEX_ROOT / "gt_distill_v16"))
    goal = os.environ.get("TRACK2_CURRENT_USER_GOAL", "")
    task_type = infer_v16_task_type(goal, scenario)
    automata = _load_json(bank / "tool_sequence_automata.json", {})
    slot_rules = _load_json(bank / "slot_resolver_rules.json", {})
    lexicon = _load_json(bank / "scenario_entity_lexicon.json", {})
    anti = _load_json(bank / "anti_failure_rules.json", {})
    repairs = _load_json(bank / "process_repair_templates.json", {})
    gpt_rules = _load_jsonl(bank / "gpt55_rule_distillation.jsonl", limit=60)
    templates = _select_templates(automata, scenario, task_type)
    scenario_lex = lexicon.get(scenario, {})

    lines = [
        "[V16 GT100 Distilled Executable Process Policy]",
        f"- inferred_task_type: {task_type}",
        "- Built only from non-final, non-val41 GT100/success trajectories. Do not use val41 GT or final hidden metadata.",
        "- This policy is executable guidance: first fill slots, then follow minimal skeleton, then close mutation/aggregate. Do not convert it into broad exploration.",
        "- Joint target: match required tool process and final DB state. Micro-only partial progress is not enough.",
        "- Mutation preconditions: pinned user_id when required; pinned restaurant_name for order/restaurant; canonical entity name; quantity/category/price/tax/discount when the tool schema requires them.",
        "- Closure check before natural-language final answer: if user requested add/remove/update, the mutation must be completed exactly once; if user requested total/payment/tax/nutrition/summary, call the matching aggregate/summary near the end.",
        "- Candidate selection: prefer schema-valid, GT-like skeleton coverage, complete slots, no broad scan, no visual follow-up, lower tool count.",
    ]
    if templates:
        lines.append("- GT100-like minimal skeleton candidates:")
        for item in templates:
            names = item.get("tool_names") or []
            fam = item.get("tool_family_sequence") or []
            lines.append(f"  * {item.get('scenario')}/{item.get('task_type')} count={item.get('count')}: " + " -> ".join(names[:10]))
            if fam:
                lines.append("    family: " + " -> ".join(fam[:10]))
    if slot_rules.get("tool_param_rules"):
        lines.append("- Slot resolver discipline:")
        shown = 0
        for item in slot_rules["tool_param_rules"]:
            ents = item.get("entity_params") or []
            params = item.get("common_params") or []
            if ents and shown < 8:
                lines.append(f"  * {item.get('tool_name')}: resolve {', '.join(ents)}; common params {', '.join(params[:8])}")
                shown += 1
    if scenario_lex:
        lines.append("- Scenario lexicon examples from non-final GT100:")
        for slot, values in list(scenario_lex.items())[:6]:
            if values:
                lines.append(f"  * {slot}: " + ", ".join(str(x) for x in values[:10]))
    if anti.get("rules"):
        lines.append("- Anti-failure rules:")
        for item in anti["rules"]:
            if item.get("scenario") in ("global", scenario):
                lines.append(f"  * {item.get('name')}: {item.get('rule')}")
    if repairs.get("templates"):
        lines.append("- Process repair templates:")
        for item in repairs["templates"]:
            lines.append(f"  * if {item.get('if')} then {item.get('then')}")
    matching_gpt = [r for r in gpt_rules if str(r.get("task_type")) == task_type or not r.get("task_type")]
    if matching_gpt[:3]:
        lines.append("- GPT-5.5 distilled generalized rules:")
        for item in matching_gpt[:3]:
            rule = str(item.get("generalizable_rule") or "")[:260]
            if rule:
                lines.append(f"  * {rule}")
    if scenario == "order":
        lines += [
            "- Order V16 hard preferences:",
            "  * Pin restaurant first. The phrase 'from now on' or 'use that restaurant' switches active restaurant for all later order tools.",
            "  * Separate dish_name and set_meal_name. For set-meal membership, use get_set_meal_details before branch mutation.",
            "  * After add/remove order mutation, use get_user_order_summary and compute_total_payment/tax/nutrition only if requested.",
            "  * Aggregate dishes[] in order tools use product_name + quantity according to OrderDB schema.",
        ]
    if scenario == "retail":
        lines += [
            "- Retail V16 hard preferences:",
            "  * Narrow candidates by visual/category/country/brand/taste before price/tax/discount/nutrition.",
            "  * Candidate count >5 means continue narrowing; do not broad-scan numeric attributes.",
            "  * For cart mutation, add/remove exactly once after canonical product_name is known.",
        ]
    if scenario == "restaurant":
        lines += [
            "- Restaurant V16 hard preferences:",
            "  * Resolve pointed menu region to dish/set meal/category with constrained retrieval.",
            "  * Query-only tasks should stay short; mutation tasks must close add/remove and requested aggregate.",
        ]
    if scenario == "kitchen":
        lines += [
            "- Kitchen V16 hard preferences:",
            "  * Identify current/target recipe, get recipe ingredients once, determine branch, apply menu/shopping-list change, then compute nutrition if requested.",
            "  * Ingredient/nutrition lists must come from confirmed current menu/shopping-list/tool observations.",
        ]
    return "\n".join(lines)
