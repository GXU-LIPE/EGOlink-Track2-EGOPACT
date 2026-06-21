# -*- coding: utf-8 -*-
"""Conservative repair for model-produced EgoBench tool-call JSON."""

import ast
import json
import re
from typing import Any, Dict, List, Tuple


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("，", ",")
        .replace("：", ":")
    )


def strip_fence(text: str) -> str:
    match = FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _balanced_json_span(text: str, start: int) -> str:
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
            elif ch == "\\":
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


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def coerce_tool_call(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(coerce_tool_call(item))
        return out
    if not isinstance(obj, dict):
        return []
    if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
        return coerce_tool_call(obj["tool_calls"])
    if "function_call" in obj and isinstance(obj["function_call"], dict):
        fc = obj["function_call"]
        args = fc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                try:
                    args = ast.literal_eval(args)
                except Exception:
                    args = {}
        return [{"tool_name": fc.get("name", ""), "parameters": args if isinstance(args, dict) else {}}]
    name = obj.get("tool_name") or obj.get("name") or obj.get("tool") or obj.get("function")
    params = obj.get("parameters", obj.get("arguments", {}))
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            try:
                params = ast.literal_eval(params)
            except Exception:
                params = {}
    if name:
        return [{"tool_name": str(name), "parameters": params if isinstance(params, dict) else {}}]
    return []


def repair_tool_json(text: str) -> Tuple[bool, str, Dict[str, Any]]:
    raw = text
    candidate = remove_trailing_commas(extract_json_candidate(text))
    errors = []
    parsed = None
    for parser_name, parser in (("json", json.loads), ("literal_eval", ast.literal_eval)):
        try:
            parsed = parser(candidate)
            break
        except Exception as exc:
            errors.append(f"{parser_name}: {exc}")
    if parsed is None:
        return False, raw, {"errors": errors, "candidate": candidate}
    calls = coerce_tool_call(parsed)
    if not calls:
        return False, raw, {"errors": errors + ["no_tool_calls"], "candidate": candidate}
    repaired = json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
    return True, repaired, {"errors": errors, "candidate": candidate, "calls": calls}
