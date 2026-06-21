# -*- coding: utf-8 -*-
"""Service-agent output wrapper for copied EgoBench runner."""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from .db_guard import append_wrapper_event, guard_tool_calls
from .json_repair import repair_tool_json
from .prompt_builder import enhance_prompt
from .schema_loader import load_schema
from .tool_validator import validate_tool_json
from .guard_policy import classify_policy
from .multicandidate_reranker import select_candidate
try:
    from .v9_candidate_builder import build_candidates
except Exception:
    def build_candidates(normalized, scenario, state, turn):
        return [normalized]
try:
    from .v17_process_compiler import repair_tool_output as v17_repair_tool_output
except Exception:
    def v17_repair_tool_output(normalized, scenario, state, turn):
        return normalized, {"enabled": False}
try:
    from .human_prior_controller import observe_model_reply, observe_validated_output
except Exception:
    def observe_model_reply(*args, **kwargs):
        return None
    def observe_validated_output(*args, **kwargs):
        return None
try:
    from .order_process_state_helper import inspect_natural_reply
except Exception:
    def inspect_natural_reply(reply, state, turn):
        return {"allow": True}
try:
    from .visual_grounding_resolver import rewrite_visual_followup
except Exception:
    def rewrite_visual_followup(reply, scenario, state, turn):
        return {"allow": True}


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def enhance_service_prompt(base_prompt: str, scenario: str) -> str:
    load_schema()
    return enhance_prompt(base_prompt, scenario)


def _looks_like_tool_attempt(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.startswith("[")
        or stripped.startswith("{")
        or "tool_name" in stripped
        or "function_call" in stripped
        or "```json" in stripped.lower()
    )


def _append_log(record: Dict[str, Any]) -> None:
    log_dir = CODEX_ROOT / "logs" / "guard"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"guard_{time.strftime('%Y%m%d')}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_repair_agent_reply(reply: str, scenario: str, history: List[Dict[str, Any]], episode_state: Dict[str, Any] = None, turn: int = None) -> str:
    record: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenario": scenario,
        "task_id": (episode_state or {}).get("task_id"),
        "turn": turn,
        "raw_output": reply,
        "final_action": reply,
        "repair_applied": False,
        "model_backend": os.environ.get("SERVICE_MODEL_BACKEND") or "direct_api",
        "whether_external_api_used": os.environ.get("TRACK2_FINAL_COMPLIANT") != "1",
    }
    observe_model_reply(reply, scenario, history, episode_state, turn)
    if not _looks_like_tool_attempt(reply):
        natural_decision = inspect_natural_reply(reply, episode_state or {}, turn)
        if not natural_decision.get("allow", True):
            reply = natural_decision.get("replacement") or reply
            record["final_action"] = reply
            record["natural_reply_rewritten"] = True
        visual_decision = rewrite_visual_followup(reply, scenario, episode_state or {}, turn)
        if not visual_decision.get("allow", True):
            reply = visual_decision.get("replacement") or reply
            record["final_action"] = reply
            record["natural_reply_rewritten"] = True
            record["visual_followup_rewritten"] = True
        _append_log(record)
        append_wrapper_event(episode_state, {
            "event": "agent_reply",
            "turn": turn,
            "raw_model_output": reply,
            "repaired_output": reply,
            "validated_action": None,
            "guard_decision": None,
        "v9_policy": None,
            "planner_state": {"scenario": scenario},
            "pins": (episode_state or {}).get("pins"),
            "mutation_ledger": (episode_state or {}).get("successful_mutation_ledger"),
            "blocked_calls": (episode_state or {}).get("blocked_calls"),
            "model_backend": record["model_backend"],
            "whether_external_api_used": record["whether_external_api_used"],
        })
        return reply

    repaired_ok, repaired, repair_report = repair_tool_json(reply)
    record["repair_report"] = repair_report
    if not repaired_ok:
        record["structural_error"] = True
        _append_log(record)
        append_wrapper_event(episode_state, {
            "event": "structural_error",
            "turn": turn,
            "raw_model_output": reply,
            "repair_report": repair_report,
            "model_backend": record["model_backend"],
            "whether_external_api_used": record["whether_external_api_used"],
        })
        return reply

    valid, normalized, validation = validate_tool_json(repaired, scenario)
    record["validation_result"] = validation
    if valid:
        guard = guard_tool_calls(normalized, scenario, history)
        record["guard"] = guard
        v9_policy = classify_policy(normalized, scenario, validation, episode_state or {})
        record["v9_policy"] = v9_policy
        rerank_result = None
        multicandidate_enabled = (
            os.environ.get("TRACK2_ENABLE_MULTICANDIDATE") == "1"
            or os.environ.get("TRACK2_ENABLE_MULTICANDIDATE_RERANK") == "1"
        )
        if multicandidate_enabled or os.environ.get("TRACK2_ENABLE_DEEPSEEK_CROSSCHECK") == "1":
            try:
                candidates = build_candidates(normalized, scenario, episode_state or {}, turn or 0)
                rerank_result = select_candidate(candidates, scenario, episode_state or {}, turn or 0)
                if rerank_result.get("selected") is not None:
                    normalized = rerank_result["selected"]
                    record["v9_reranker"] = rerank_result
                    record["v9_candidate_count"] = len(candidates)
            except Exception as exc:
                record["v9_reranker_error"] = type(exc).__name__
        if os.environ.get("TRACK2_ENABLE_V17_COMPILER") == "1" or str(os.environ.get("TRACK2_RUN_VERSION", "")).startswith("V17_"):
            normalized, v17_report = v17_repair_tool_output(normalized, scenario, episode_state or {}, turn or 0)
            record["v17_compiler"] = v17_report
        observe_validated_output(reply, repaired, normalized, scenario, episode_state, turn, validation)
        # V1 is format/schema-only. In V3+ we record risk but avoid blocking unless JSON invalid.
        record["final_action"] = normalized
        record["repair_applied"] = normalized != reply
        _append_log(record)
        append_wrapper_event(episode_state, {
            "event": "agent_tool_output",
            "turn": turn,
            "raw_model_output": reply,
            "repaired_output": repaired,
            "validated_action": normalized,
            "validation_result": validation,
            "guard_decision": guard,
            "v9_policy": v9_policy,
            "v9_reranker": rerank_result,
            "planner_state": {
                "scenario": scenario,
                "kitchen_stage": (episode_state or {}).get("kitchen_stage"),
                "tool_call_count": (episode_state or {}).get("tool_call_count"),
            },
            "pins": (episode_state or {}).get("pins"),
            "mutation_ledger": (episode_state or {}).get("successful_mutation_ledger"),
            "blocked_calls": (episode_state or {}).get("blocked_calls"),
            "model_backend": record["model_backend"],
            "whether_external_api_used": record["whether_external_api_used"],
        })
        return normalized

    # Return normalized partial call only if validator at least corrected parse/name and did not drop all calls.
    try:
        arr = json.loads(normalized)
        if isinstance(arr, list) and arr and validation.get("invalid_tool_name_count", 0) == 0:
            observe_validated_output(reply, repaired, normalized, scenario, episode_state, turn, validation)
            record["final_action"] = normalized
            record["repair_applied"] = True
            _append_log(record)
            append_wrapper_event(episode_state, {
                "event": "agent_tool_output_partial",
                "turn": turn,
                "raw_model_output": reply,
                "repaired_output": repaired,
                "validated_action": normalized,
                "validation_result": validation,
                "pins": (episode_state or {}).get("pins"),
                "mutation_ledger": (episode_state or {}).get("successful_mutation_ledger"),
                "blocked_calls": (episode_state or {}).get("blocked_calls"),
                "model_backend": record["model_backend"],
                "whether_external_api_used": record["whether_external_api_used"],
            })
            return normalized
    except Exception:
        pass
    record["structural_error"] = True
    _append_log(record)
    append_wrapper_event(episode_state, {
        "event": "validation_failed",
        "turn": turn,
        "raw_model_output": reply,
        "repaired_output": repaired,
        "validation_result": validation,
        "model_backend": record["model_backend"],
        "whether_external_api_used": record["whether_external_api_used"],
    })
    return reply
