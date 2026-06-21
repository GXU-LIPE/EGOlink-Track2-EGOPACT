#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native GPT-5.5 vision service-agent loop for EgoBench V32."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .v32_prompt_variants import variant_prompt
from .v32_tool_loop_guard import execute_calls, normalize_tool_calls, summarize_trace, trace_risk_flags


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[len("export ") :]
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _image_data_url(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _openai_client() -> tuple[Any, str, str]:
    _load_env(CODEX / "state" / ".openai_env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"openai import failed: {type(exc).__name__}: {exc}") from exc
    base_url = os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("SERVICE_MODEL_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("TRACK2_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.5"
    timeout_s = float(os.environ.get("TRACK2_OPENAI_TIMEOUT", "120"))
    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout_s, "max_retries": 1}
    if base_url:
        kwargs["base_url"] = base_url
    if os.environ.get("TRACK2_OPENAI_NO_PROXY", "1") != "0":
        try:
            import httpx

            kwargs["http_client"] = httpx.Client(timeout=timeout_s, trust_env=False)
        except Exception:
            pass
    return OpenAI(**kwargs), model, "responses"


def call_gpt55(messages: List[Dict[str, Any]], image_url: str = "", max_tokens: int = 1600) -> Dict[str, Any]:
    client, model, _ = _openai_client()
    started = time.time()
    # Prefer Responses API; fall back to Chat Completions for OpenAI-compatible endpoints.
    try:
        inp: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                converted = content
            else:
                converted = [{"type": "input_text", "text": str(content)}]
            inp.append({"role": msg.get("role", "user"), "content": converted})
        if image_url and inp:
            inp[-1]["content"].append({"type": "input_image", "image_url": image_url})
        resp = client.responses.create(model=model, input=inp, max_output_tokens=max_tokens)
        text = getattr(resp, "output_text", "") or ""
        return {"ok": True, "api": "responses", "model": model, "text": text, "latency": round(time.time() - started, 3)}
    except Exception as exc:
        responses_error = f"{type(exc).__name__}: {exc}"
    try:
        chat_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            if image_url and msg is messages[-1]:
                content = [{"type": "text", "text": str(content)}]
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            chat_messages.append({"role": msg.get("role", "user"), "content": content})
        resp = client.chat.completions.create(model=model, messages=chat_messages, max_tokens=max_tokens, temperature=0)
        text = resp.choices[0].message.content or ""
        return {
            "ok": True,
            "api": "chat_completions",
            "model": model,
            "text": text,
            "latency": round(time.time() - started, 3),
            "responses_error": responses_error,
        }
    except Exception as exc:
        return {"ok": False, "api": "none", "model": model, "text": "", "latency": round(time.time() - started, 3), "error": f"responses={responses_error}; chat={type(exc).__name__}: {exc}"}


def _compact_tool_schema(tool_schema: Any) -> Any:
    if not isinstance(tool_schema, list):
        return tool_schema
    out: List[Dict[str, Any]] = []
    for tool in tool_schema[:90]:
        if not isinstance(tool, dict):
            continue
        row = {
            "name": tool.get("name") or tool.get("tool_name"),
            "description": str(tool.get("description", ""))[:500],
        }
        params = tool.get("parameters")
        if isinstance(params, dict):
            row["parameters"] = params
        elif "input_schema" in tool:
            row["parameters"] = tool.get("input_schema")
        out.append(row)
    return out


def db_context_payload(db_summary: Dict[str, List[str]], tool_schema: Any) -> Dict[str, Any]:
    return {
        "db_canonical_entities": {k: v[:90] for k, v in db_summary.items()},
        "tool_schema": _compact_tool_schema(tool_schema),
    }


def initial_context(
    *,
    row: Dict[str, Any],
    scenario: str,
    spec: str,
    tool_schema: Any,
    db_summary: Dict[str, List[str]],
    evidence: Dict[str, Any] | None,
    repair_hint: str = "",
) -> str:
    safe_row = {
        "scenario": scenario,
        "spec": spec,
        "task_id": row.get("task_id"),
        "instruction": row.get("Instruction", ""),
        "image_path": row.get("image_path", ""),
    }
    ev = evidence or {}
    ev_compact = {
        "ocr_visible_text": (ev.get("ocr_evidence") or {}).get("visible_text", [])[:40],
        "asr_text": (ev.get("asr_evidence") or {}).get("transcript", "")[:1000],
        "vision_entities": (ev.get("vision_entities") or [])[:20],
        "canonical_matches": {k: v[:5] for k, v in (ev.get("canonical_matches") or {}).items()},
        "qwen_status": (ev.get("sources") or {}).get("qwen_status"),
        "gpt55_evidence_status": (ev.get("sources") or {}).get("gpt55_vision_status"),
    }
    payload = {
        "current_task": safe_row,
        "runtime_boundaries": {
            "final_run": False,
            "no_final_hidden_metadata": True,
            "no_val41_gt_runtime_hint": True,
            "excluded_fields": ["analysis", "ground_truth", "image_description"],
        },
        "multimodal_evidence_summary": ev_compact,
        "db_and_tools": db_context_payload(db_summary, tool_schema),
        "repair_hint": repair_hint[:1800] if repair_hint else "",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_native_service_agent(
    *,
    row: Dict[str, Any],
    scenario: str,
    spec: str,
    db: Any,
    db_summary: Dict[str, List[str]],
    tool_schema: Any,
    contact_sheet_path: str,
    evidence: Dict[str, Any] | None,
    variant: str,
    repair_hint: str = "",
    max_rounds: int = 8,
    max_tool_calls: int = 80,
) -> Dict[str, Any]:
    image_url = _image_data_url(contact_sheet_path)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": variant_prompt(variant, repair_hint=repair_hint)},
        {"role": "user", "content": initial_context(row=row, scenario=scenario, spec=spec, tool_schema=tool_schema, db_summary=db_summary, evidence=evidence, repair_hint=repair_hint)},
    ]
    history_log = {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": "",
        "dialogue": [],
        "tool_calls": [],
        "rounds_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls_count": 0,
        "agent_response_time_seconds": 0.0,
        "user_response_time_seconds": 0.0,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
        "v32_meta": {"variant": variant, "contact_sheet_path": contact_sheet_path, "vision_image_attached": bool(image_url)},
    }
    all_calls: List[Dict[str, Any]] = []
    raw_replies: List[Dict[str, Any]] = []
    api_errors: List[str] = []
    final_text = ""
    for round_idx in range(max_rounds):
        resp = call_gpt55(messages, image_url=image_url if round_idx == 0 else "", max_tokens=1800)
        history_log["agent_response_time_seconds"] += float(resp.get("latency", 0) or 0)
        raw_text = resp.get("text", "")
        raw_replies.append(resp)
        if not resp.get("ok"):
            api_errors.append(resp.get("error", "api_error"))
            final_text = "[V32 API error]"
            break
        is_tool, calls, reason = normalize_tool_calls(raw_text)
        if not is_tool:
            final_text = raw_text.strip()[:4000]
            history_log["dialogue"].append({"role": "agent", "turn": round_idx, "content": final_text})
            history_log["rounds_count"] += 1
            break
        if len(all_calls) + len(calls) > max_tool_calls:
            calls = calls[: max(0, max_tool_calls - len(all_calls))]
        results = execute_calls(db, calls)
        all_calls.extend(calls)
        history_log["tool_calls"].append({"turn": round_idx, "calls": calls, "results": results})
        history_log["tool_calls_count"] = len(all_calls)
        obs = {
            "round": round_idx,
            "tool_calls": calls,
            "tool_results": results,
            "instruction_reminder": row.get("Instruction", ""),
            "process_reminder": "Continue from observations. If task is complete, final answer; otherwise output next JSON tool array only.",
        }
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({"role": "user", "content": "Tool execution observations:\n" + json.dumps(obs, ensure_ascii=False, default=str)[:12000]})
        if len(all_calls) >= max_tool_calls:
            final_text = "[V32 stopped: max tool calls]"
            break
    if not final_text:
        final_text = "[V32 stopped: max rounds]"
    history_log["v32_meta"].update({
        "api_errors": api_errors,
        "raw_reply_count": len(raw_replies),
        "api_modes": list(dict.fromkeys([r.get("api", "") for r in raw_replies])),
        "model": next((r.get("model") for r in raw_replies if r.get("model")), ""),
        "risk_flags": trace_risk_flags(all_calls, final_text, row.get("Instruction", "")),
        "compact_trace": summarize_trace(messages),
    })
    return {
        "item": history_log,
        "tool_program": all_calls,
        "final_text": final_text,
        "api_errors": api_errors,
        "raw_replies": raw_replies,
        "risk_flags": history_log["v32_meta"]["risk_flags"],
        "vision_success": bool(image_url) and any(r.get("ok") for r in raw_replies),
    }


def make_repair_hint(trace: Dict[str, Any]) -> str:
    flags = trace.get("risk_flags") or []
    names = [x.get("tool_name") for x in trace.get("tool_program") or []]
    return json.dumps(
        {
            "non_gt_failed_trace_risks": flags,
            "previous_tool_names": names[-30:],
            "previous_final_text": trace.get("final_text", "")[:1000],
            "repair_goal": "Rerun without ground truth. Fix only process defects: retrieval before mutation, branch observation, canonical entity support, and closure.",
        },
        ensure_ascii=False,
    )
