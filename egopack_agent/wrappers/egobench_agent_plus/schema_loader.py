# -*- coding: utf-8 -*-
"""Load EgoBench tool schemas into a compact cache."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))


STATE_CHANGE_HINTS = (
    "add", "remove", "delete", "update", "modify", "clear", "set", "create",
    "place", "cancel", "cart", "order", "list", "menu",
)


def _tool_name(tool: Dict[str, Any]) -> str:
    fn = tool.get("function")
    fn_name = ""
    if isinstance(fn, dict):
        fn_name = fn.get("tool_name") or fn.get("name") or ""
    return str(tool.get("name") or tool.get("tool_name") or fn_name or "")


def _parameters(tool: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(tool.get("parameters"), dict):
        return tool["parameters"]
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
        return fn["parameters"]
    return {}


def _properties(params: Dict[str, Any]) -> Dict[str, Any]:
    props = params.get("properties")
    return props if isinstance(props, dict) else {}


def _required(params: Dict[str, Any]) -> List[str]:
    req = params.get("required")
    return [str(x) for x in req] if isinstance(req, list) else []


def _param_type(spec: Any) -> str:
    if isinstance(spec, dict):
        typ = spec.get("type", "any")
        if isinstance(typ, list):
            return "|".join(str(x) for x in typ)
        return str(typ)
    return "any"


def load_schema(force: bool = False) -> Dict[str, Any]:
    cache = CODEX_ROOT / "state" / "schema_cache.json"
    if cache.exists() and not force:
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)

    schema: Dict[str, Any] = {"tools": {}, "scenarios": {}}
    tools_root = EGO_ROOT / "tools"
    for path in tools_root.glob("*/*_tools.json"):
        scenario = path.parent.name
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        schema["scenarios"].setdefault(scenario, [])
        for tool in data:
            if not isinstance(tool, dict):
                continue
            name = _tool_name(tool)
            if not name:
                continue
            params = _parameters(tool)
            props = _properties(params)
            required = _required(params)
            entry = {
                "name": name,
                "scenario": scenario,
                "required": required,
                "optional": [p for p in props if p not in required],
                "param_types": {p: _param_type(spec) for p, spec in props.items()},
                "state_changing": any(h in name.lower() for h in STATE_CHANGE_HINTS),
                "raw": tool,
            }
            schema["tools"][name] = entry
            schema["scenarios"][scenario].append(name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    return schema


def get_scenario_schema(scenario: str) -> Dict[str, Any]:
    schema = load_schema()
    names = set(schema.get("scenarios", {}).get(scenario, []))
    return {name: schema["tools"][name] for name in names if name in schema.get("tools", {})}
