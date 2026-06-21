#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch json_repair to extract balanced JSON prefix from mixed output."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
PATH = CODEX / "wrappers" / "egobench_agent_plus" / "json_repair.py"


OLD = '''def extract_json_candidate(text: str) -> str:
    text = strip_fence(normalize_quotes(text))
    if text.startswith("[") or text.startswith("{"):
        return text
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr > start_arr:
        return text[start_arr : end_arr + 1]
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1]
    return text
'''


NEW = '''def _balanced_json_span(text: str, start: int) -> str:
    if start < 0 or start >= len(text) or text[start] not in "[{":
        return ""
    opens = {"[": "]", "{": "}"}
    stack = [opens[text[start]]]
    in_str = False
    escape = False
    for idx in range(start + 1, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in opens:
            stack.append(opens[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[start : idx + 1]
    return ""


def extract_json_candidate(text: str) -> str:
    text = strip_fence(normalize_quotes(text))
    starts = [idx for idx in (text.find("["), text.find("{")) if idx != -1]
    for start in sorted(starts):
        candidate = _balanced_json_span(text, start)
        if candidate:
            return candidate
    return text
'''


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if OLD not in text:
        print("pattern_not_found_or_already_patched")
        return 0
    backup = CODEX / "backups" / f"json_repair_prefix_{time.strftime('%Y%m%d_%H%M%S')}" / PATH.relative_to(CODEX)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup)
    PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched {PATH}")
    print(f"backup {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
