# -*- coding: utf-8 -*-
"""Validate and lightly normalize EgoBench tool calls."""

import difflib
import json
from typing import Any, Dict, List, Tuple

from .schema_loader import get_scenario_schema
from .canonical_resolver import canonicalize_tool_params


def _coerce_value(value: Any, typ: str) -> Tuple[bool, Any]:
    if typ in ("any", ""):
        return True, value
    if "|" in typ:
        for part in typ.split("|"):
            ok, coerced = _coerce_value(value, part)
            if ok:
                return True, coerced
        return False, value
    try:
        if typ == "string":
            if value is None:
                return False, value
            return True, str(value)
        if typ == "integer":
            if isinstance(value, bool):
                return False, value
            return True, int(value)
        if typ == "number":
            if isinstance(value, bool):
                return False, value
            return True, float(value)
        if typ == "boolean":
            if isinstance(value, bool):
                return True, value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return True, value.lower() == "true"
            return False, value
        if typ == "array":
            if isinstance(value, list):
                return True, value
            return True, [value]
        if typ == "object":
            return isinstance(value, dict), value
    except Exception:
        return False, value
    return True, value


def _normalize_name(name: str, names: List[str]) -> Tuple[str, bool]:
    if name in names:
        return name, False
    lowered = {n.lower(): n for n in names}
    key = name.lower()
    if key in lowered:
        return lowered[key], True
    compact = name.lower().replace("-", "_").replace(" ", "_")
    lowered_compact = {n.lower().replace("-", "_").replace(" ", "_"): n for n in names}
    if compact in lowered_compact:
        return lowered_compact[compact], True
    match = difflib.get_close_matches(compact, list(lowered_compact.keys()), n=1, cutoff=0.86)
    if match:
        return lowered_compact[match[0]], True
    return name, False


def _required_satisfied(tool_name: str, required: str, params: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    alias_map = {
        "compute_total_nutrition": {"dishes": "products", "products": "dishes"},
    }
    if required in params:
        return True, {}
    alias = alias_map.get(tool_name, {}).get(required)
    if alias and alias in params:
        return True, {"tool": tool_name, "required": required, "satisfied_by": alias}
    return False, {}


def validate_tool_json(text: str, scenario: str) -> Tuple[bool, str, Dict[str, Any]]:
    schema = get_scenario_schema(scenario)
    names = list(schema.keys())
    report: Dict[str, Any] = {
        "valid": False,
        "invalid_json_count": 0,
        "invalid_tool_name_count": 0,
        "missing_required_param_count": 0,
        "wrong_param_type_count": 0,
        "name_corrections": [],
        "param_coercions": [],
        "param_aliases": [],
        "errors": [],
    }
    try:
        data = json.loads(text)
    except Exception as exc:
        report["invalid_json_count"] = 1
        report["errors"].append(f"json_error: {exc}")
        return False, text, report
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        report["errors"].append("not_nonempty_array")
        return False, text, report

    normalized = []
    for call in data:
        if not isinstance(call, dict):
            report["errors"].append("call_not_object")
            continue
        name = str(call.get("tool_name") or call.get("name") or call.get("tool") or "")
        name2, corrected = _normalize_name(name, names)
        if not name2 or name2 not in schema:
            report["invalid_tool_name_count"] += 1
            report["errors"].append(f"invalid_tool_name:{name}")
            continue
        if corrected:
            report["name_corrections"].append({"from": name, "to": name2})
        params = call.get("parameters", call.get("arguments", {}))
        if not isinstance(params, dict):
            params = {}
            report["wrong_param_type_count"] += 1
        params, canonical_notes = canonicalize_tool_params(name2, params, scenario)
        if canonical_notes:
            report.setdefault("canonical_corrections", []).extend(
                {"tool": name2, **note} for note in canonical_notes
            )
        tool_schema = schema[name2]
        for req in tool_schema.get("required", []):
            ok_required, alias = _required_satisfied(name2, req, params)
            if alias:
                report["param_aliases"].append(alias)
            if not ok_required:
                report["missing_required_param_count"] += 1
                report["errors"].append(f"missing:{name2}.{req}")
        new_params = dict(params)
        for key, typ in tool_schema.get("param_types", {}).items():
            if key in new_params:
                ok, coerced = _coerce_value(new_params[key], typ)
                if not ok:
                    report["wrong_param_type_count"] += 1
                    report["errors"].append(f"type:{name2}.{key}:{typ}")
                elif coerced != new_params[key]:
                    report["param_coercions"].append({"tool": name2, "param": key, "from": new_params[key], "to": coerced})
                    new_params[key] = coerced
        normalized.append({"tool_name": name2, "parameters": new_params})

    hard_errors = (
        report["invalid_tool_name_count"]
        + report["missing_required_param_count"]
        + report["wrong_param_type_count"]
    )
    report["valid"] = bool(normalized) and hard_errors == 0
    output = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) if normalized else text
    return report["valid"], output, report
