# -*- coding: utf-8 -*-
"""Probe the configured OpenAI-compatible API without printing secrets."""

import argparse
import json
import os
import time
from pathlib import Path


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("SERVICE_MODEL_NAME", "deepseek-v4-pro"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from openai import OpenAI

    started = time.time()
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "base_url": os.environ.get("SERVICE_API_BASE_URL") or os.environ.get("LLM_API_BASE_URL", ""),
        "ok": False,
        "elapsed": None,
        "error_type": "",
        "error": "",
        "content_preview": "",
        "usage": {},
    }
    try:
        client = OpenAI(
            api_key=os.environ.get("SERVICE_API_KEY") or os.environ.get("API_KEY"),
            base_url=result["base_url"],
            timeout=args.timeout,
        )
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a concise API probe."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            max_tokens=8,
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        result["ok"] = "OK" in content
        result["content_preview"] = content[:80]
        if getattr(resp, "usage", None):
            result["usage"] = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            }
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    result["elapsed"] = round(time.time() - started, 3)

    out = Path(args.output) if args.output else CODEX_ROOT / "state" / "api_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
