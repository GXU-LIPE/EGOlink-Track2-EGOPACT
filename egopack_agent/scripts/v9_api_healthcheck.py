#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def check_chat(kind: str, base: str, key: str, model: str, timeout: float = 20.0):
    started = time.time()
    if not key:
        return {"available": False, "key_present": False, "failure_class": "missing_api_key", "elapsed_sec": 0}
    try:
        resp = requests.post(
            base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "Return exactly: ok"}], "max_tokens": 8, "temperature": 0, "stream": False},
            timeout=(8, timeout),
        )
        if resp.status_code >= 400:
            return {"available": False, "key_present": True, "failure_class": f"http_{resp.status_code}", "elapsed_sec": round(time.time() - started, 3)}
        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return {"available": bool(content), "key_present": True, "failure_class": "" if content else "empty_content", "elapsed_sec": round(time.time() - started, 3)}
    except requests.exceptions.Timeout:
        return {"available": False, "key_present": True, "failure_class": "timeout", "elapsed_sec": round(time.time() - started, 3)}
    except Exception as exc:
        return {"available": False, "key_present": True, "failure_class": type(exc).__name__, "elapsed_sec": round(time.time() - started, 3)}


load_env(CODEX / "state" / ".openai_env")
load_env(CODEX / "state" / ".deepseek_env")
ts = time.strftime("%Y%m%d_%H%M%S")
gpt_base = os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("SERVICE_MODEL_API_BASE") or "https://ai-pixel.online/v1"
gpt_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SERVICE_MODEL_API_KEY") or ""
gpt_model = os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5")
deep_base = os.environ.get("DEEPSEEK_API_BASE") or os.environ.get("TRACK2_DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1"
deep_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("TRACK2_DEEPSEEK_API_KEY") or ""
deep_model = os.environ.get("TRACK2_DEEPSEEK_CROSSCHECK_MODEL", "deepseek-chat")

gpt = check_chat("gpt55", gpt_base, gpt_key, gpt_model)
deep = check_chat("deepseek", deep_base, deep_key, deep_model, timeout=15)
report = CODEX / "reports" / f"V9_API_HEALTHCHECK_{ts}.md"
report.write_text("\n".join([
    f"# V9 API Healthcheck {ts}",
    "",
    f"- gpt55_available: {gpt['available']}",
    f"- gpt55_key_present: {gpt['key_present']}",
    f"- gpt55_base_url: `{gpt_base}`",
    f"- gpt55_model: `{gpt_model}`",
    f"- gpt55_failure_class: `{gpt['failure_class']}`",
    f"- gpt55_elapsed_sec: `{gpt['elapsed_sec']}`",
    f"- deepseek_available: {deep['available']}",
    f"- deepseek_key_present: {deep['key_present']}",
    f"- deepseek_base_url: `{deep_base}`",
    f"- deepseek_model: `{deep_model}`",
    f"- deepseek_failure_class: `{deep['failure_class']}`",
    f"- deepseek_elapsed_sec: `{deep['elapsed_sec']}`",
    "",
    "No API key values were written.",
]), encoding="utf-8")
(CODEX / "state" / "v9_api_healthcheck_latest.json").write_text(json.dumps({"timestamp": ts, "gpt55": gpt, "deepseek": deep, "report": str(report)}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"report": str(report), "gpt55": gpt, "deepseek": deep}, ensure_ascii=False, indent=2))
