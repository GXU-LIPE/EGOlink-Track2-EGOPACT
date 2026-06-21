#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-fast wiring check for V27 direct V21 bridge."""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
ADAPTER = CODEX / "wrappers" / "egobench_agent_plus" / "v27_v25_to_v21_adapter.py"
REPORTS = CODEX / "reports"

REQUIRED_IMPORTS = {
    "RetailResolverV21": "v21_retail_resolver",
    "build_attribute_query_plan": "v21_retail_attribute_query_planner",
    "infer_attribute_targets": "v21_retail_attribute_query_planner",
    "RetailObservationBrancherV21": "v21_retail_observation_brancher",
    "RetailAddTargetResolverV21": "v21_retail_add_target_resolver",
}
REQUIRED_TRACE_FLAGS = [
    "called_RetailResolverV21",
    "called_attribute_query_planner",
    "called_observation_brancher",
    "called_add_target_resolver",
]


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def static_imports() -> Dict[str, Any]:
    result = {"adapter_exists": ADAPTER.exists(), "found": {}, "missing": []}
    if not ADAPTER.exists():
        result["missing"] = list(REQUIRED_IMPORTS)
        return result
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                found[alias.name] = mod
    for name, module_part in REQUIRED_IMPORTS.items():
        ok = name in found and module_part in found[name]
        result["found"][name] = {"module": found.get(name, ""), "ok": ok}
        if not ok:
            result["missing"].append(name)
    return result


def runtime_trace_ok(trace_path: Path | None) -> Dict[str, Any]:
    out = {"checked": False, "retail_records": 0, "bad_records": [], "required_flags": REQUIRED_TRACE_FLAGS}
    if not trace_path:
        return out
    out["checked"] = True
    if not trace_path.exists():
        out["bad_records"].append({"reason": "trace_file_missing", "path": str(trace_path)})
        return out
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("scenario") != "retail":
            continue
        out["retail_records"] += 1
        for key in ("direct_v21_trace", "evidence_v21_trace"):
            trace = row.get(key) or {}
            if not trace:
                out["bad_records"].append({"task_key": row.get("task_key"), "candidate": key, "reason": "missing_trace"})
                continue
            missing = [flag for flag in REQUIRED_TRACE_FLAGS if trace.get(flag) is not True]
            if missing:
                out["bad_records"].append({"task_key": row.get("task_key"), "candidate": key, "missing": missing})
    return out


def write_report(run_id: str, static: Dict[str, Any], runtime: Dict[str, Any]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ok = not static.get("missing") and (not runtime.get("checked") or not runtime.get("bad_records"))
    lines: List[str] = [
        f"# V27 Direct V21 Wiring Check {run_id}",
        "",
        f"- status: {'PASS' if ok else 'FAIL'}",
        f"- adapter: `{ADAPTER}`",
        f"- adapter_exists: {static.get('adapter_exists')}",
        "",
        "## Static Imports",
        "",
        "| symbol | required module | found module | ok |",
        "|---|---|---|---:|",
    ]
    for name, module_part in REQUIRED_IMPORTS.items():
        info = (static.get("found") or {}).get(name) or {}
        lines.append(f"| {name} | {module_part} | {info.get('module','')} | {info.get('ok', False)} |")
    lines += [
        "",
        "## Runtime Trace",
        "",
        f"- checked: {runtime.get('checked')}",
        f"- retail_records: {runtime.get('retail_records')}",
        f"- bad_records: {len(runtime.get('bad_records') or [])}",
    ]
    for bad in (runtime.get("bad_records") or [])[:30]:
        lines.append(f"- bad: `{json.dumps(bad, ensure_ascii=False)}`")
    path = REPORTS / f"V27_DIRECT_V21_WIRING_CHECK_{run_id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v27_wiring_" + stamp())
    ap.add_argument("--runtime-trace", default="")
    args = ap.parse_args()
    static = static_imports()
    runtime = runtime_trace_ok(Path(args.runtime_trace) if args.runtime_trace else None)
    report = write_report(args.run_id, static, runtime)
    payload = {"run_id": args.run_id, "ok": not static.get("missing") and (not runtime.get("checked") or not runtime.get("bad_records")), "static": static, "runtime": runtime, "report": str(report)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["ok"] else 2)


if __name__ == "__main__":
    main()
