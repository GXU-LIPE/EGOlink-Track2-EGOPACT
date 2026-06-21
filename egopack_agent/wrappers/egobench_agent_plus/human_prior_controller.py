# -*- coding: utf-8 -*-
"""Controller glue for V7 human-prior modules."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .counterfactual_db_simulator import assess_batch
from .human_process_graph import infer_process_state, prompt_snippet
from .process_coverage_verifier import verify_process_coverage
from .visual_slot_prior import load_visual_slots, compact_slot_text
from .working_memory_manager import build_working_memory_prompt


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def human_prior_enabled() -> bool:
    return os.environ.get("TRACK2_ENABLE_HUMAN_PRIOR", "0") == "1"


def level() -> str:
    forced = os.environ.get("TRACK2_HUMAN_PRIOR_LEVEL")
    if forced:
        return forced
    version = os.environ.get("TRACK2_RUN_VERSION", "")
    if "V7_4" in version:
        return "full"
    if "V7_3" in version:
        return "helpers"
    if "V7_2" in version:
        return "counterfactual"
    if "V7_1" in version:
        return "verifier"
    if "V7_0" in version:
        return "graph"
    return "full"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def event_path(state: Dict[str, Any]) -> Path:
    version = state.get("version") or os.environ.get("TRACK2_RUN_VERSION") or "V7"
    run_id = state.get("run_id") or os.environ.get("TRACK2_RUN_ID") or time.strftime("manual_%Y%m%d_%H%M%S")
    task_id = state.get("task_id", "unknown")
    path = CODEX_ROOT / "runs" / str(version) / str(run_id) / "human_prior_events" / f"{task_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_human_prior_event(state: Dict[str, Any] | None, event: Dict[str, Any]) -> None:
    if not state:
        return
    record = {
        "time": _now(),
        "task_id": state.get("task_id"),
        "scenario": state.get("scenario"),
        "version": state.get("version"),
        "run_id": state.get("run_id"),
        "human_prior_level": level(),
        **event,
    }
    try:
        with event_path(state).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append_policy_trace(state: Dict[str, Any] | None, record: Dict[str, Any]) -> None:
    if not state:
        return
    try:
        path = CODEX_ROOT / "train_data" / "human_prior_policy_traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "time": _now(),
            "task_id": state.get("task_id"),
            "scenario": state.get("scenario"),
            "user_utterance": state.get("user_instruction"),
            "source_version": state.get("version"),
            **record,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    except Exception:
        pass


STATIC_PROMPT = """
Human-prior service policy:
- Act like a careful service worker: confirm the goal, retrieve necessary evidence, mutate state once, then verify with the final aggregate tool.
- Cover process stages, not just final DB state. If an order mutation occurs, finish the requested tax/payment/nutrition aggregate before a final message.
- Never repeat a successful non-idempotent state-changing call.
- Do not treat set meals as ordinary dishes; use set_meal tools for set_meal_name and dish tools for dish_name.
- Do not modify an order before restaurant_name is pinned.
- Order with missing contact sheet: do not ask the simulated user for visual dish/category names. Use benchmark image_description/task analysis/layout hint plus restaurant-pinned retrieval to choose a grounded candidate, then continue the process.
- If an order aggregate compute returns 0.0 for a nonempty order, do not repeat the same aggregate call. Switch to the missing remove/add/final-tax stage or finish with the best supported process.
- For kitchen, get recipe ingredients once, choose the branch from evidence, avoid broad scans, and compute nutrition only from confirmed ingredients.
- Keep active tool/entity candidates small. If tools are needed, output only a JSON array; otherwise output only a short user-facing sentence.
""".strip()


def static_prompt() -> str:
    return STATIC_PROMPT if human_prior_enabled() else ""


def build_turn_system_message(scenario: str, state: Dict[str, Any], history: List[Dict[str, Any]], visual_context: Dict[str, Any] | None = None, turn: int | None = None) -> str:
    if not human_prior_enabled():
        return ""
    ps = infer_process_state(scenario, state)
    slots = load_visual_slots(scenario, state, visual_context)
    wm = build_working_memory_prompt(scenario, state, history, ps, slots)
    text = "\n\n".join([
        prompt_snippet(scenario, state),
        compact_slot_text(slots, scenario),
        wm,
    ]).strip()
    append_human_prior_event(state, {
        "event": "human_prior_turn_prompt",
        "turn": turn,
        "process_state": ps,
        "visual_slots": slots,
    })
    return text[:5000]


def observe_model_reply(reply: str, scenario: str, history: List[Dict[str, Any]], state: Dict[str, Any] | None, turn: int | None = None) -> None:
    if not human_prior_enabled() or not state:
        return
    verdict = verify_process_coverage(scenario, None, state, natural_reply=True)
    append_human_prior_event(state, {
        "event": "human_prior_model_reply",
        "turn": turn,
        "raw_model_output": str(reply)[:4000],
        "verifier_decision": verdict,
    })


def observe_validated_output(raw_reply: str, repaired_output: str, normalized_output: str, scenario: str, state: Dict[str, Any] | None, turn: int | None = None, validation: Dict[str, Any] | None = None) -> None:
    if not human_prior_enabled() or not state:
        return
    try:
        calls = json.loads(normalized_output)
    except Exception:
        calls = []
    verifier = verify_process_coverage(scenario, calls, state, natural_reply=False)
    counterfactual = assess_batch(calls, scenario, state) if level() in {"counterfactual", "helpers", "full"} else []
    process_state = infer_process_state(scenario, state)
    slots = load_visual_slots(scenario, state, None)
    append_human_prior_event(state, {
        "event": "human_prior_validated_action",
        "turn": turn,
        "process_stage": process_state.get("current_stage"),
        "raw_model_output": str(raw_reply)[:4000],
        "repaired_output": repaired_output,
        "validated_action": normalized_output,
        "verifier_decision": verifier,
        "counterfactual_decision": counterfactual,
        "validation": validation or {},
    })
    append_policy_trace(state, {
        "visual_slots": slots,
        "process_stage": process_state.get("current_stage"),
        "allowed_tools": process_state.get("allowed_tool_set", []),
        "forbidden_tools": [],
        "model_raw_output": str(raw_reply)[:4000],
        "repaired_output": repaired_output,
        "verifier_decision": verifier,
        "counterfactual_decision": counterfactual,
        "final_executed_action": normalized_output,
    })
