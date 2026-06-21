# -*- coding: utf-8 -*-
"""V8 event logger for Track2 helper modules."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))

def enabled(name: str) -> bool:
    return os.environ.get(name, "0") == "1"

def _safe(value: Any) -> Any:
    if isinstance(value, str):
        # Redact obvious API-key-looking strings without logging secrets.
        if value.startswith("sk-") and len(value) > 12:
            return "sk-[REDACTED]"
        return value[:4000]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if "key" not in str(k).lower() and "token" not in str(k).lower()}
    if isinstance(value, list):
        return [_safe(v) for v in value[:50]]
    return value

def write_v8_event(state: Dict[str, Any] | None, module: str, decision: str, reason: str = "", **kwargs: Any) -> None:
    state = state or {}
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": state.get("task_id"),
        "scenario": state.get("scenario"),
        "turn": kwargs.pop("turn", None),
        "module": module,
        "decision": decision,
        "reason": reason,
        "before_action": _safe(kwargs.pop("before_action", None)),
        "after_action": _safe(kwargs.pop("after_action", None)),
        "risk_score": kwargs.pop("risk_score", 0.0),
        "whether_repaired": kwargs.pop("whether_repaired", False),
        "whether_blocked": kwargs.pop("whether_blocked", False),
        "whether_crosschecked": kwargs.pop("whether_crosschecked", False),
        "whether_final_eval": os.environ.get("TRACK2_FINAL_EVAL", "0") == "1",
        "no_key_logged": True,
    }
    record.update({k: _safe(v) for k, v in kwargs.items()})
    version = state.get("version") or os.environ.get("TRACK2_RUN_VERSION", "V8")
    run_id = state.get("run_id") or os.environ.get("TRACK2_RUN_ID", "manual")
    task_id = state.get("task_id", "unknown")
    out = CODEX_ROOT / "runs" / str(version) / str(run_id) / "v8_events" / f"{task_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
