# -*- coding: utf-8 -*-
"""Check DeepSeek/OpenAI-compatible model names and response shape."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


def curl_models(base_url: str, api_key: str) -> dict:
    url = base_url.rstrip("/") + "/models"
    cmd = [
        "curl",
        "-sS",
        "--connect-timeout",
        "5",
        "--max-time",
        "20",
        "-H",
        f"Authorization: Bearer {api_key}",
        url,
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=25,
        )
        return {
            "url": url,
            "exit_code": proc.returncode,
            "elapsed": round(time.time() - started, 3),
            "stdout_preview": proc.stdout[:2000],
            "stderr_preview": proc.stderr[:1000],
        }
    except Exception as exc:
        return {"url": url, "exit_code": -1, "elapsed": round(time.time() - started, 3), "error": repr(exc)}


def chat_probe(model: str, base_url: str, api_key: str, timeout: float) -> dict:
    from openai import OpenAI

    started = time.time()
    result = {"model": model, "ok": False}
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply exactly OK"}],
            max_tokens=16,
            temperature=0,
        )
        msg = resp.choices[0].message
        content = getattr(msg, "content", None)
        result.update(
            {
                "ok": content is not None and "OK" in content,
                "elapsed": round(time.time() - started, 3),
                "content": content,
                "message_dump": getattr(msg, "model_dump", lambda: {})(),
                "usage": getattr(getattr(resp, "usage", None), "model_dump", lambda: {})(),
            }
        )
    except Exception as exc:
        result.update(
            {
                "elapsed": round(time.time() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"])
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("SERVICE_API_KEY") or os.environ.get("API_KEY") or ""
    base_url = os.environ.get("SERVICE_API_BASE_URL") or os.environ.get("LLM_API_BASE_URL") or "https://api.deepseek.com/v1"
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": base_url,
        "models": [],
        "models_endpoint": {},
        "usable_model": "",
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result["models_endpoint"] = curl_models(base_url, api_key)
    for model in args.models:
        r = chat_probe(model, base_url, api_key, args.timeout)
        result["models"].append(r)
        if r.get("ok") and not result["usable_model"]:
            result["usable_model"] = model
    out = Path(args.output) if args.output else CODEX_ROOT / "state" / "api_compat.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["usable_model"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
