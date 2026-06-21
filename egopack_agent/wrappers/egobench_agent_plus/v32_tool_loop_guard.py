#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small guards for V32 native tool loop."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, List, Tuple


MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")
AGG_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
    "get_user_order_summary",
}


def safe_json_from_text(text: str) -> Any:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
        s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\[.*\]", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return [json.loads(m.group(0))]
        except Exception:
            pass
    return None


def _scan_json_values(text: str) -> List[Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
        s = re.sub(r"```$", "", s).strip()
    decoder = json.JSONDecoder()
    values: List[Any] = []
    i = 0
    while i < len(s):
        while i < len(s) and s[i] not in "[{":
            i += 1
        if i >= len(s):
            break
        try:
            value, end = decoder.raw_decode(s, i)
            values.append(value)
            i = max(end, i + 1)
        except Exception:
            i += 1
    return values


def normalize_tool_calls(obj: Any) -> Tuple[bool, List[Dict[str, Any]], str]:
    raw = obj if isinstance(obj, str) else json.dumps(obj)
    data = safe_json_from_text(raw)
    values = [data] if data is not None else []
    values.extend(_scan_json_values(raw))
    calls: List[Dict[str, Any]] = []
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("tool_name") or item.get("name")
            params = item.get("parameters", item.get("arguments", {}))
            if not name:
                continue
            row = {"tool_name": str(name), "parameters": params if isinstance(params, dict) else {}}
            if row not in calls:
                calls.append(row)
    if not calls:
        return False, [], "json_without_tool_calls"
    return True, calls, "ok"


def filter_params(db: Any, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not hasattr(db, tool_name):
        return params
    try:
        sig = inspect.signature(getattr(db, tool_name))
        return {k: v for k, v in (params or {}).items() if k in sig.parameters}
    except Exception:
        return params or {}


def execute_calls(db: Any, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen_success: set[str] = set()
    for call in calls:
        name = call.get("tool_name", "")
        params = filter_params(db, name, call.get("parameters") or {})
        key = json.dumps({"tool": name, "params": params}, sort_keys=True, ensure_ascii=False)
        if MUTATION_RE.search(name) and key in seen_success:
            results.append({
                "role": "tool",
                "tool_name": name,
                "parameters": params,
                "content": json.dumps({"status": "blocked", "message": "duplicate mutation blocked"}, ensure_ascii=False),
                "status": "blocked",
            })
            continue
        if not hasattr(db, name):
            results.append({
                "role": "tool",
                "tool_name": name,
                "parameters": params,
                "content": json.dumps({"status": "error", "message": f"Tool '{name}' not found"}, ensure_ascii=False),
                "status": "error",
            })
            continue
        try:
            res = getattr(db, name)(**params)
            status = "success"
            if isinstance(res, dict) and res.get("status") == "error":
                status = "error"
            if MUTATION_RE.search(name) and status == "success":
                seen_success.add(key)
            results.append({
                "role": "tool",
                "tool_name": name,
                "parameters": params,
                "content": json.dumps(res, ensure_ascii=False, default=str),
                "status": status,
            })
        except Exception as exc:
            results.append({
                "role": "tool",
                "tool_name": name,
                "parameters": params,
                "content": json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False),
                "status": "error",
            })
    return results


def summarize_trace(history: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    parts: List[str] = []
    for item in history[-16:]:
        role = item.get("role")
        content = item.get("content", "")
        if isinstance(content, list):
            content = "[multimodal content]"
        parts.append(f"{role}: {str(content)[:1200]}")
    text = "\n".join(parts)
    return text[-max_chars:]


def trace_risk_flags(tool_calls: List[Dict[str, Any]], final_text: str, instruction: str) -> List[str]:
    flags: List[str] = []
    names = [c.get("tool_name", "") for c in tool_calls]
    if not names:
        flags.append("no_tool_call")
    if any(n.startswith(("find_products_by_price_range", "get_all_", "list_all_")) for n in names[:2]):
        flags.append("leading_broad_scan")
    if any(MUTATION_RE.search(n) for n in names) and not any(n.startswith(("get_", "find_", "filter_", "list_")) for n in names[: max(1, min(3, len(names)))]):
        flags.append("mutation_without_prefix_retrieval")
    text = (instruction or "").lower()
    if any(x in text for x in ("total payment", "payable", "total cost")) and "compute_total_payment" not in names:
        flags.append("possibly_missing_payment_closure")
    if "total tax" in text and "compute_total_tax" not in names:
        flags.append("possibly_missing_tax_closure")
    if any(x in text for x in ("total nutrition", "total nutritional", "nutrition content")) and not any(n in {"compute_total_nutrition", "compute_total_nutritions"} for n in names):
        flags.append("possibly_missing_nutrition_closure")
    if final_text and ("sorry" in final_text.lower() or "cannot" in final_text.lower()):
        flags.append("refusal_or_ask_user")
    return flags
