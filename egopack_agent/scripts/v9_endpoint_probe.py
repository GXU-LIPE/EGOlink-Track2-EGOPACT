#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe OpenAI-compatible Track2 endpoints without printing API keys."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def post_chat(base: str, key: str, model: str, read_timeout: float) -> dict:
    started = time.time()
    try:
        r = requests.post(
            base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK only."}],
                "max_tokens": 8,
                "temperature": 0,
                "stream": False,
            },
            timeout=(10, read_timeout),
            proxies={},
        )
        status = r.status_code
        if status >= 400:
            return {"available": False, "status": status, "failure_class": f"http_{status}", "elapsed_sec": round(time.time() - started, 3)}
        data = r.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        return {
            "available": bool(content),
            "status": status,
            "failure_class": "" if content else "empty_content",
            "elapsed_sec": round(time.time() - started, 3),
            "content_preview": content[:20],
        }
    except requests.exceptions.Timeout:
        return {"available": False, "failure_class": "timeout", "elapsed_sec": round(time.time() - started, 3)}
    except Exception as exc:
        return {"available": False, "failure_class": type(exc).__name__, "elapsed_sec": round(time.time() - started, 3)}


def get_models(base: str, key: str) -> dict:
    started = time.time()
    try:
        r = requests.get(
            base.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=(8, 30),
            proxies={},
        )
        return {"ok": r.status_code < 400, "status": r.status_code, "elapsed_sec": round(time.time() - started, 3)}
    except requests.exceptions.Timeout:
        return {"ok": False, "failure_class": "timeout", "elapsed_sec": round(time.time() - started, 3)}
    except Exception as exc:
        return {"ok": False, "failure_class": type(exc).__name__, "elapsed_sec": round(time.time() - started, 3)}


def main() -> None:
    load_env(CODEX / "state" / ".openai_env")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SERVICE_MODEL_API_KEY") or ""
    configured = os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("SERVICE_MODEL_API_BASE") or "https://ai-pixel.online/v1"
    bases = []
    for b in [configured, "https://ai-pixel.online/v1", "https://cf.ai-pixel.online/v1"]:
        if b and b not in bases:
            bases.append(b)
    model = os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5")
    out = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "key_present": bool(key),
        "model": model,
        "proxies_present": bool(os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")),
        "results": [],
    }
    for base in bases:
        item = {"base_url": base, "models": get_models(base, key) if key else {"ok": False, "failure_class": "missing_key"}}
        item["chat_45s"] = post_chat(base, key, model, 45) if key else {"available": False, "failure_class": "missing_key"}
        item["chat_120s"] = post_chat(base, key, model, 120) if key and not item["chat_45s"].get("available") else {"skipped": True}
        out["results"].append(item)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = CODEX / "reports" / f"V9_ENDPOINT_PROBE_{ts}.json"
    report.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report), "key_present": out["key_present"], "results": out["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
