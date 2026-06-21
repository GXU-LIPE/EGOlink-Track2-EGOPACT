# -*- coding: utf-8 -*-
"""Low-frequency DeepSeek cross-validator for V9.

DeepSeek is only a risk reviewer. It never becomes the service agent and its
output is cached. Missing key or disabled env degrades to a neutral stub.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict
import time

import requests

from .v8_event_logger import enabled, write_v8_event

CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))

def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def should_crosscheck(scenario: str, state: Dict[str, Any], call: Any, risk_score: float = 0.0) -> bool:
    if not enabled("TRACK2_ENABLE_DEEPSEEK_CROSSCHECK"):
        return False
    if scenario not in {"order", "kitchen", "retail", "restaurant"}:
        return False
    if os.environ.get("TRACK2_USE_DEEPSEEK_CROSSCHECK", "0") != "1":
        return False
    if state.setdefault("v8_deepseek_crosscheck_count", 0) >= 2:
        return False
    text = json.dumps(call, ensure_ascii=False, default=str).lower()
    triggers = (
        "soft_warning" in text
        or "broad_scan" in text
        or "visual" in text
        or "aggregate" in text
        or "compute_" in text
        or "add_" in text
        or "remove_" in text
        or "state" in text
    )
    return risk_score >= 0.35 or bool(state.get("blocked_calls")) or triggers


def _load_env_file() -> None:
    for path in (CODEX_ROOT / "state" / ".deepseek_env", CODEX_ROOT / "state" / ".openai_env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("export ") or "=" not in line:
                continue
            key, value = line[len("export "):].split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _online_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    _load_env_file()
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("TRACK2_DEEPSEEK_API_KEY")
    if not api_key:
        return {"risk": "unknown", "recommended_action": "accept", "repair_hint": "", "confidence": 0.0, "online_call_performed": False, "missing_key": True}
    base_url = os.environ.get("DEEPSEEK_API_BASE") or os.environ.get("TRACK2_DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1"
    model = os.environ.get("TRACK2_DEEPSEEK_CROSSCHECK_MODEL", "deepseek-chat")
    prompt = (
        "You are a Track2 risk reviewer, not the service agent. Return only JSON with keys: "
        "risk(low|medium|high), process_missing(list), tool_type_confusion(list), db_state_risk(list), "
        "visual_grounding_risk(list), recommended_action(accept|repair|reject), repair_hint(string), confidence(number). "
        "Do not solve with final GT. Review this candidate:\n"
        + json.dumps(payload, ensure_ascii=False, default=str)[:12000]
    )
    url = base_url.rstrip("/") + "/chat/completions"
    timeout = (float(os.environ.get("TRACK2_DEEPSEEK_CONNECT_TIMEOUT", "8")), float(os.environ.get("TRACK2_DEEPSEEK_READ_TIMEOUT", "60")))
    proxies = {k: v for k, v in {
        "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
        "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
    }.items() if v}
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 600, "stream": False},
        timeout=timeout,
        proxies=proxies or None,
    )
    resp.raise_for_status()
    content = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "{}"
    try:
        data = json.loads(content)
    except Exception:
        data = {"risk": "medium", "recommended_action": "repair", "repair_hint": content[:500], "confidence": 0.1}
    data["online_call_performed"] = True
    data["model"] = model
    return data

def crosscheck(payload: Dict[str, Any], state: Dict[str, Any], turn: int) -> Dict[str, Any]:
    cache_dir = CODEX_ROOT / "teacher_cache" / "deepseek_crosscheck"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(payload)
    path = cache_dir / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        write_v8_event(state, "deepseek_crosscheck", "cache_hit", "deepseek_crosscheck_cache_hit", turn=turn, whether_crosschecked=True)
        return data
    started = time.time()
    try:
        if os.environ.get("TRACK2_USE_DEEPSEEK_CROSSCHECK", "0") == "1":
            data = _online_review(payload)
        else:
            data = {"risk": "low", "process_missing": [], "tool_type_confusion": [], "db_state_risk": [], "visual_grounding_risk": [], "recommended_action": "accept", "repair_hint": "", "confidence": 0.0, "online_call_performed": False, "disabled": True}
    except Exception as exc:
        data = {"risk": "unknown", "process_missing": [], "tool_type_confusion": [], "db_state_risk": [], "visual_grounding_risk": [], "recommended_action": "accept", "repair_hint": "", "confidence": 0.0, "online_call_performed": False, "api_error": type(exc).__name__}
    data["latency_sec"] = round(time.time() - started, 3)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    state["v8_deepseek_crosscheck_count"] = state.get("v8_deepseek_crosscheck_count", 0) + 1
    write_v8_event(state, "deepseek_crosscheck", data.get("recommended_action", "accept"), "deepseek_crosscheck_review", turn=turn, whether_crosschecked=True, risk=data.get("risk"), online_call_performed=data.get("online_call_performed", False), cache_key=key)
    return data
