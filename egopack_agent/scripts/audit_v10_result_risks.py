#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench/results/V10_full_memory_final_candidate_draft")


def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


for path in sorted(ROOT.glob("*_easy.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    for idx, item in enumerate(data, 1):
        nonempty_api_errors = []
        rate_strings = []
        for obj in walk(item):
            if "api_error" in obj and obj.get("api_error"):
                nonempty_api_errors.append(obj.get("api_error"))
        text = json.dumps(item, ensure_ascii=False)
        for token in ("RateLimitError", "ReadTimeout", "ConnectionError", "APIError", "429"):
            if token.lower() in text.lower():
                rate_strings.append(token)
        tool_calls = item.get("tool_calls") or []
        if nonempty_api_errors or rate_strings or not tool_calls:
            print(
                json.dumps(
                    {
                        "file": path.name,
                        "task": idx,
                        "empty_tool_calls": not bool(tool_calls),
                        "tool_calls_count": item.get("tool_calls_count"),
                        "rounds_count": item.get("rounds_count"),
                        "nonempty_api_errors": nonempty_api_errors[:3],
                        "rate_strings": rate_strings,
                    },
                    ensure_ascii=False,
                )
            )
