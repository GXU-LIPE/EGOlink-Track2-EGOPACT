#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path
from urllib import request, error

CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
ENV = CODEX / "state" / ".openai_env"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


def post_json(url: str, payload: dict, key: str, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", "replace")
            return {"ok": True, "status": resp.status, "body_head": body[:500]}
    except error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        return {"ok": False, "status": exc.code, "reason": exc.reason, "body_head": body[:1000]}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def main() -> int:
    load_env(ENV)
    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("TRACK2_OPENAI_BASE_URL", "https://ai-pixel.online/v1").rstrip("/")
    model = os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Return exactly OK."}],
        "temperature": 0,
        "max_tokens": 16,
    }
    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": base,
        "model": model,
        "key_present": bool(key),
        "key_logged": False,
        "chat_completions": None,
    }
    if key:
        result["chat_completions"] = post_json(f"{base}/chat/completions", payload, key)
    out = CODEX / "reports" / f"GPT55_ENDPOINT_PROBE_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    sys.stdout.write(json.dumps({**result, "report": str(out)}, ensure_ascii=True, indent=2) + "\n")
    return 0 if result["chat_completions"] and result["chat_completions"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
