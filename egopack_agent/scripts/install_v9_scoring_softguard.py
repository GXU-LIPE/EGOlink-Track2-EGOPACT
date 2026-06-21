#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import time


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
WRAP = CODEX / "wrappers" / "egobench_agent_plus"
REPORTS = CODEX / "reports"
BACKUPS = CODEX / "backups" / f"v9_scoring_softguard_{time.strftime('%Y%m%d_%H%M%S')}"


EVALUATOR_AWARENESS = r'''# -*- coding: utf-8 -*-
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
'''


GUARD_POLICY = r'''# -*- coding: utf-8 -*-
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

    if has_mutation and not has_retrieval:
        soft_warnings.append("retrieval_recommended_before_mutation")
    if has_aggregate and len(calls) > 1 and not names[-1].lower().startswith("compute_"):
        soft_warnings.append("aggregate_may_not_be_final_step")
    if has_aggregate and not has_mutation and len(episode_state.get("executed_tool_calls") or []) < 2:
        soft_warnings.append("aggregate_may_be_too_early")

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
        "- If a soft warning applies, prefer a lower-risk tool step, but do not follow a rigid FSM when the current evidence supports another valid process.",
        "- Rerank signals are advisory: process coverage, result risk, DB risk, loop risk, and trajectory risk.",
    ])
'''


def backup(path: Path) -> None:
    if path.exists():
        rel = path.relative_to(CODEX)
        dst = BACKUPS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())


def write(path: Path, text: str) -> None:
    backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def patch_prompt_builder() -> bool:
    path = WRAP / "prompt_builder.py"
    backup(path)
    text = path.read_text(encoding="utf-8")
    changed = False
    if "from .evaluator_awareness import build_evaluator_awareness_card" not in text:
        marker = "from .planner import planner_prompt\n"
        text = text.replace(marker, marker + "from .evaluator_awareness import build_evaluator_awareness_card, infer_task_type\nfrom .guard_policy import build_soft_guard_prompt\n")
        changed = True
    needle = "def enhance_prompt(base_prompt: str, scenario: str) -> str:\n"
    if "TRACK2_ENABLE_EVALUATOR_AWARENESS" not in text:
        repl = needle + "    final_eval = os.environ.get(\"TRACK2_FINAL_EVAL\") == \"1\" or os.environ.get(\"TRACK2_FINAL_COMPLIANT\") == \"1\"\n"
        repl += "    run_version = os.environ.get(\"TRACK2_RUN_VERSION\", \"\")\n"
        repl += "    task_stage = os.environ.get(\"TRACK2_TASK_STAGE\", \"turn\")\n"
        repl += "    task_type = infer_task_type(os.environ.get(\"TRACK2_CURRENT_USER_GOAL\", \"\"), task_stage, scenario)\n"
        repl += "    v9_enabled = os.environ.get(\"TRACK2_ENABLE_EVALUATOR_AWARENESS\") == \"1\" or run_version.startswith(\"V9_\")\n"
        text = text.replace(needle, repl)
        changed = True
    marker = "    if os.environ.get(\"TRACK2_ENABLE_PLANNER\") == \"1\":\n"
    if "build_evaluator_awareness_card(scenario" not in text:
        insert = "    if v9_enabled:\n"
        insert += "        text += \"\\n\\n\" + build_evaluator_awareness_card(scenario, task_stage, task_type, final_eval=final_eval)\n"
        insert += "    if os.environ.get(\"TRACK2_ENABLE_V9_SOFT_GUARD\") == \"1\" or run_version.startswith(\"V9_2\") or run_version.startswith(\"V9_3\") or run_version.startswith(\"V9_4\") or run_version.startswith(\"V9_5\"):\n"
        insert += "        text += \"\\n\\n\" + build_soft_guard_prompt()\n"
        text = text.replace(marker, insert + marker)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_service_agent_wrapper() -> bool:
    path = WRAP / "service_agent_wrapper.py"
    backup(path)
    text = path.read_text(encoding="utf-8")
    changed = False
    if "from .guard_policy import classify_policy" not in text:
        text = text.replace("from .tool_validator import validate_tool_json\n", "from .tool_validator import validate_tool_json\nfrom .guard_policy import classify_policy\n")
        changed = True
    if '"v9_policy": None' not in text:
        text = text.replace('"guard_decision": None,\n', '"guard_decision": None,\n        "v9_policy": None,\n')
        changed = True
    if "v9_policy = classify_policy(normalized, scenario, validation, episode_state or {})" not in text:
        old = "        guard = guard_tool_calls(normalized, scenario, history)\n        record[\"guard\"] = guard\n"
        new = old + "        v9_policy = classify_policy(normalized, scenario, validation, episode_state or {})\n        record[\"v9_policy\"] = v9_policy\n"
        text = text.replace(old, new)
        changed = True
    if '"v9_policy": v9_policy,' not in text:
        text = text.replace('"guard_decision": guard,\n', '"guard_decision": guard,\n            "v9_policy": v9_policy,\n', 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    BACKUPS.mkdir(parents=True, exist_ok=True)
    write(WRAP / "evaluator_awareness.py", EVALUATOR_AWARENESS)
    write(WRAP / "guard_policy.py", GUARD_POLICY)
    changes = {
        "prompt_builder.py": patch_prompt_builder(),
        "service_agent_wrapper.py": patch_service_agent_wrapper(),
        "evaluator_awareness.py": True,
        "guard_policy.py": True,
        "backup_dir": str(BACKUPS),
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    (REPORTS / f"V9_EVALUATOR_AWARENESS_IMPLEMENTATION_{ts}.md").write_text(
        "\n".join([
            "# V9 Evaluator Awareness Implementation",
            "",
            f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            f"- backup_dir: `{BACKUPS}`",
            "- added: `wrappers/egobench_agent_plus/evaluator_awareness.py`",
            "- wired: `prompt_builder.enhance_prompt` behind `TRACK2_ENABLE_EVALUATOR_AWARENESS=1` or `V9_*` version.",
            "- card_shape: concise scoring checklist plus scenario-specific checklist.",
            "- final_submission: not submitted",
        ]) + "\n",
        encoding="utf-8",
    )
    (REPORTS / f"V9_GUARD_REFACTOR_{ts}.md").write_text(
        "\n".join([
            "# V9 Guard Refactor",
            "",
            f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            f"- backup_dir: `{BACKUPS}`",
            "- added: `wrappers/egobench_agent_plus/guard_policy.py`",
            "- hard_block scope: invalid JSON, nonexistent tools, unrecoverable missing required parameters, hidden-final access, duplicate successful mutation, unsafe type mismatch.",
            "- soft_warning scope: retrieval before mutation, early aggregate, canonical uncertainty, dish/set-meal confusion, kitchen broad scan, missing process/aggregate risk.",
            "- rerank_signal scope: process coverage, result risk, DB risk, loop/trajectory risk.",
            "- existing db_guard/schema/canonical/duplicate/pinning logic was not removed.",
            "- overfit hard FSM behavior is not added; V9 policy currently logs/advises and injects soft guard prompt for V9_2+.",
            "- final_submission: not submitted",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(changes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
