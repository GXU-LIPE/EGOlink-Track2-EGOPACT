# -*- coding: utf-8 -*-
"""Bounded DeepSeek API/key check using requests, not OpenAI SDK."""

import argparse
import json
import os
from pathlib import Path
import time

import requests


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def post_chat(base_url, api_key, model, connect_timeout, read_timeout, max_tokens):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly OK"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    started = time.time()
    result = {"model": model, "ok": False, "elapsed": None}
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(connect_timeout, read_timeout),
        )
        result["elapsed"] = round(time.time() - started, 3)
        result["status_code"] = resp.status_code
        result["body_preview"] = resp.text[:1000]
        try:
            data = resp.json()
        except Exception:
            data = {}
        if isinstance(data, dict):
            result["usage"] = data.get("usage", {})
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content")
                result["content_preview"] = content[:200] if isinstance(content, str) else str(content)
                result["ok"] = resp.status_code == 200 and isinstance(content, str) and bool(content.strip())
    except Exception as exc:
        result["elapsed"] = round(time.time() - started, 3)
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    return result


def get_models(base_url, api_key, connect_timeout, read_timeout):
    url = base_url.rstrip("/") + "/models"
    started = time.time()
    result = {"ok": False, "elapsed": None}
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=(connect_timeout, read_timeout),
        )
        result["elapsed"] = round(time.time() - started, 3)
        result["status_code"] = resp.status_code
        result["body_preview"] = resp.text[:2000]
        result["ok"] = resp.status_code == 200
    except Exception as exc:
        result["elapsed"] = round(time.time() - started, 3)
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("SERVICE_API_BASE_URL") or os.environ.get("LLM_API_BASE_URL") or "https://api.deepseek.com/v1")
    parser.add_argument("--models", nargs="*", default=["deepseek-v4-pro", "deepseek-chat"])
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("SERVICE_API_KEY") or os.environ.get("API_KEY") or ""
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": args.base_url,
        "has_api_key": bool(api_key),
        "models_endpoint": {},
        "chat": [],
        "usable_models": [],
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not api_key:
        result["error"] = "missing API key"
    else:
        result["models_endpoint"] = get_models(args.base_url, api_key, args.connect_timeout, args.read_timeout)
        for model in args.models:
            r = post_chat(args.base_url, api_key, model, args.connect_timeout, args.read_timeout, args.max_tokens)
            result["chat"].append(r)
            if r.get("ok"):
                result["usable_models"].append(model)

    out = Path(args.output) if args.output else CODEX_ROOT / "state" / "deepseek_api_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["usable_models"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
