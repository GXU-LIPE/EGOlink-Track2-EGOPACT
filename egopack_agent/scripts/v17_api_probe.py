#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


def load_env_file(path: Path):
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                k, v = line[len("export "):].split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    env = os.environ.copy()
    env.update(load_env_file(CODEX / "state" / ".openai_env"))
    key = env.get("OPENAI_API_KEY") or env.get("SERVICE_MODEL_API_KEY")
    base = env.get("TRACK2_OPENAI_BASE_URL") or env.get("SERVICE_MODEL_API_BASE") or "https://ai-pixel.online/v1"
    model = env.get("TRACK2_OPENAI_MODEL") or env.get("SERVICE_MODEL_NAME") or "gpt-5.5"
    url = base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return JSON only: {\"ok\":true}"}],
        "temperature": 0,
        "max_tokens": 30,
    }
    report = {
        "base_url": base,
        "model": model,
        "key_present": bool(key),
        "proxy_env_present": {k: bool(os.environ.get(k)) for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]},
        "no_proxy": os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "",
        "status": "unknown",
        "latency_sec": None,
        "error": "",
    }
    start = time.time()
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        with opener.open(req, timeout=25) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        report["status"] = "ok"
        report["latency_sec"] = round(time.time() - start, 2)
        report["response_preview"] = body[:300]
    except Exception as exc:
        report["status"] = "fail"
        report["latency_sec"] = round(time.time() - start, 2)
        report["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out = CODEX / "gt_distill_v17" / "v17_api_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
