#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect prior Track2 API usage without exposing secrets."""

import json
import os
import re
from pathlib import Path


ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{12,}", re.I),
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.I),
]


def redact(text):
    if text is None:
        return ""
    text = str(text)
    for pat in SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
    for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "SERVICE_API_KEY", "API_KEY"):
        val = os.environ.get(key)
        if val:
            text = text.replace(val, "[REDACTED]")
    return text


def load_env_names(path):
    found = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)", line.strip())
        if m:
            key, value = m.group(1), m.group(2).strip().strip("'\"")
            if "KEY" in key or "TOKEN" in key:
                found[key] = {"present": bool(value), "value": "[REDACTED]" if value else ""}
            else:
                found[key] = {"present": bool(value), "value": redact(value)}
    return found


def safe_json_summary(path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"path": str(path), "parse_error": type(exc).__name__}

    def walk(obj, depth=0):
        if depth > 4:
            return "..."
        if isinstance(obj, dict):
            keep = {}
            for k, v in obj.items():
                lk = k.lower()
                if "key" in lk or "token" in lk:
                    keep[k] = bool(v) if isinstance(v, str) else "[REDACTED]"
                elif k in {
                    "ok",
                    "available",
                    "model",
                    "base_url",
                    "error_type",
                    "error_class",
                    "status_code",
                    "elapsed",
                    "usable_models",
                    "service_model",
                    "model_name",
                    "provider",
                    "timestamp",
                    "content_preview",
                    "chat",
                    "models_endpoint",
                    "failure_class",
                    "crosscheck_enabled",
                    "main_experiment_allowed",
                    "deepseek_crosscheck_allowed",
                }:
                    keep[k] = walk(v, depth + 1)
            return keep
        if isinstance(obj, list):
            return [walk(x, depth + 1) for x in obj[:5]]
        if isinstance(obj, str):
            return redact(obj)
        return obj

    return {"path": str(path), "summary": walk(data)}


def grep_sanitized(paths, pattern):
    hits = []
    rx = re.compile(pattern, re.I)
    for path in paths:
        if not path.exists() or not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            if rx.search(line):
                hits.append({"path": str(path), "line": idx, "text": redact(line[:500])})
                if len(hits) >= 120:
                    return hits
    return hits


def main():
    print("ROOT", ROOT)
    print("ENV_FILES")
    for rel in ["state/.openai_env", "state/secrets.env", "state/env_choice.sh"]:
        path = ROOT / rel
        print(json.dumps({"path": str(path), "exists": path.exists(), "vars": load_env_names(path)}, ensure_ascii=False))

    print("STATE_JSON")
    for rel in [
        "state/env_choice.json",
        "state/api_probe.json",
        "state/deepseek_api_check.json",
        "state/best_track2_api_version.json",
    ]:
        print(json.dumps(safe_json_summary(ROOT / rel), ensure_ascii=False))

    print("RECENT_API_REPORTS")
    report_paths = sorted(
        list((ROOT / "reports").glob("*API*.*"))
        + list((ROOT / "reports").glob("*api*.*"))
        + list((ROOT / "reports").glob("*ATTENTION*.*")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:25]
    for p in report_paths:
        print(json.dumps({"path": str(p), "mtime": int(p.stat().st_mtime), "size": p.stat().st_size}, ensure_ascii=False))

    print("SANITIZED_HITS")
    scan_paths = []
    for folder in ["scripts", "reports", "logs", "state"]:
        if (ROOT / folder).exists():
            scan_paths.extend([p for p in (ROOT / folder).glob("**/*") if p.is_file()])
    hits = grep_sanitized(
        scan_paths,
        r"gpt-5\.5|gpt55|ai-pixel|cf\.ai-pixel|SERVICE_API|SERVICE_MODEL|api_probe|OK|usable_models|deepseek",
    )
    for hit in hits[:120]:
        print(json.dumps(hit, ensure_ascii=False))


if __name__ == "__main__":
    main()
