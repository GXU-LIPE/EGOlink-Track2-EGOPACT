#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V37 read-only audit after official EgoBench sync."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

from egobench_agent_plus import v37_official_adapter as adapter  # noqa: E402


V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
V10_EXPECTED_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def safe_keys(obj: Any) -> List[str]:
    if isinstance(obj, dict):
        return sorted(str(k) for k in obj.keys())[:40]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return sorted(str(k) for k in obj[0].keys())[:40]
    return []


def infer_scenario(path: Path) -> Dict[str, Any]:
    m = re.match(r"([a-zA-Z_]+)(\d+)\.json$", path.name)
    return {"scenario": m.group(1), "number": int(m.group(2))} if m else {"scenario": "", "number": 0}


def build_file_inventory() -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    by_top = Counter()
    by_suffix = Counter()
    for path in sorted(EGO.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(EGO).as_posix()
        st = path.stat()
        top = rel.split("/", 1)[0]
        by_top[top] += 1
        by_suffix[path.suffix.lower() or "<none>"] += 1
        files.append(
            {
                "path": rel,
                "size": st.st_size,
                "sha256": sha256_file(path),
                "mtime": int(st.st_mtime),
                "top": top,
                "suffix": path.suffix.lower(),
            }
        )
    scenario_files = [f for f in files if f["path"].startswith("scenarios/")]
    video_files = [f for f in files if f["path"].startswith("videos/")]
    tool_files = [f for f in files if f["path"].startswith("tools/")]
    return {
        "ego_root": str(EGO),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "by_top_dir": dict(by_top),
        "by_suffix": dict(by_suffix),
        "scenario_file_count": len(scenario_files),
        "video_file_count": len(video_files),
        "tool_file_count": len(tool_files),
        "files": files,
    }


def audit_scenarios() -> Dict[str, Any]:
    scenario_rows: List[Dict[str, Any]] = []
    split_counts = Counter()
    key_counts = defaultdict(Counter)
    task_total = 0
    final_inventory_only = True
    for path in sorted((EGO / "scenarios").rglob("*.json")):
        rel = path.relative_to(EGO).as_posix()
        split = path.parent.name
        info = infer_scenario(path)
        data = read_json(path, None)
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data)
        else:
            count = 0
        task_total += count
        split_counts[split] += count
        keys = safe_keys(data)
        for k in keys:
            key_counts[split][k] += 1
        image_paths: List[str] = []
        if isinstance(data, list):
            # For final, keep only schema-level and path-level inventory.  Do
            # not preserve instructions, analysis, image_description, or GT.
            for row in ([] if split == "final" else data[:5]):
                if isinstance(row, dict) and row.get("image_path"):
                    image_paths.append(str(row.get("image_path")))
        scenario_rows.append(
            {
                "path": rel,
                "split": split,
                "scenario": info["scenario"],
                "number": info["number"],
                "task_count": count,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "schema_keys": keys,
                "sample_image_path_basenames": [Path(x).name for x in image_paths[:5]],
                "final_inventory_only": split == "final",
            }
        )
    val_manifest = adapter.load_val41_manifest()
    val41_tasks = adapter.load_val41_tasks()
    return {
        "scenario_files": scenario_rows,
        "scenario_file_count": len(scenario_rows),
        "task_total_across_json_files": task_total,
        "split_task_counts": dict(split_counts),
        "split_key_counts": {k: dict(v) for k, v in key_counts.items()},
        "val41_manifest": val_manifest,
        "val41_task_count": len(val41_tasks),
        "final_inventory_only": final_inventory_only,
        "final_content_used_for_experiment": False,
    }


def audit_tools_and_db() -> Dict[str, Any]:
    scenarios = ["retail", "restaurant", "order", "kitchen"]
    rows: Dict[str, Any] = {}
    schema_mismatches: List[Dict[str, Any]] = []
    for scenario in scenarios:
        schema = adapter.tool_schema_summary(scenario)
        specs = []
        # Instantiate all official scenario-number variants visible from
        # scenario JSONs, including final, without reading task content.
        numbers = sorted(
            {
                int(m.group(1))
                for p in (EGO / "scenarios").rglob(f"{scenario}*.json")
                for m in [re.match(rf"{scenario}(\d+)\.json$", p.name)]
                if m
            }
        )
        for number in numbers:
            try:
                db = adapter.init_db(scenario, number)
                methods = adapter.public_db_methods(db)
                db_info = adapter.db_entity_counts_and_samples(scenario, db)
                missing_db_methods = [name for name in schema["names"] if name not in methods]
                extra_db_methods = [name for name in methods if name not in schema["names"]]
                if missing_db_methods:
                    schema_mismatches.append({"scenario": scenario, "number": number, "missing_db_methods_for_schema": missing_db_methods})
                specs.append(
                    {
                        "spec": f"{scenario}{number}",
                        "db_class": type(db).__name__,
                        "db_methods": methods,
                        "schema_tool_names": schema["names"],
                        "schema_tool_count": schema["count"],
                        "missing_db_methods_for_schema": missing_db_methods,
                        "extra_db_methods_not_in_schema": extra_db_methods,
                        "db_entity_summary": db_info,
                    }
                )
            except Exception as exc:
                specs.append({"spec": f"{scenario}{number}", "error": f"{type(exc).__name__}: {exc}"})
                schema_mismatches.append({"scenario": scenario, "number": number, "error": f"{type(exc).__name__}: {exc}"})
        rows[scenario] = {"tool_schema": schema, "specs": specs}
    return {
        "tools": rows,
        "schema_mismatches": schema_mismatches,
        "schema_compatible": not schema_mismatches,
    }


def audit_runner_compatibility(tool_db: Dict[str, Any]) -> Dict[str, Any]:
    paths = {
        "v32_runner": CODEX / "scripts" / "run_v32_native_vision_val41.py",
        "v34_runner": CODEX / "scripts" / "run_v34_v32_expansion_protected_merge.py",
        "v32_agent": CODEX / "wrappers" / "egobench_agent_plus" / "v32_native_vision_service_agent.py",
        "v32_guard": CODEX / "wrappers" / "egobench_agent_plus" / "v32_tool_loop_guard.py",
        "v22_result_dir": EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915",
        "materialized_val41": CODEX / "state" / "materialized_splits" / "validation_A_limit30" / "manifest.json",
    }
    imports_ok: Dict[str, str] = {}
    for name, module in [
        ("run_v32_native_vision_val41", "run_v32_native_vision_val41"),
        ("v32_native_vision_service_agent", "egobench_agent_plus.v32_native_vision_service_agent"),
        ("v32_tool_loop_guard", "egobench_agent_plus.v32_tool_loop_guard"),
    ]:
        try:
            if str(CODEX / "scripts") not in sys.path:
                sys.path.insert(0, str(CODEX / "scripts"))
            importlib.import_module(module)
            imports_ok[name] = "ok"
        except Exception as exc:
            imports_ok[name] = f"{type(exc).__name__}: {exc}"
    val_tasks = adapter.load_val41_tasks()
    context_ok: List[Dict[str, Any]] = []
    for task in adapter.select_one_per_scenario(val_tasks):
        try:
            db = adapter.init_db(task["scenario"], task["number"])
            tools = adapter.load_tool_schema(task["scenario"])
            row = task["row"]
            video_base = Path(str(row.get("image_path") or "")).name
            video_exists = bool(video_base and (EGO / "videos" / video_base).exists())
            context_ok.append(
                {
                    "task_key": f"{task['spec']}::{task['index']}",
                    "scenario": task["scenario"],
                    "tool_count": len(tools),
                    "db_class": type(db).__name__,
                    "instruction_present": bool(row.get("Instruction")),
                    "video_basename": video_base,
                    "video_exists": video_exists,
                    "status": "ok",
                }
            )
        except Exception as exc:
            context_ok.append({"task_key": f"{task['spec']}::{task['index']}", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    runner_assumptions = {
        "materialized_val41_count": len(val_tasks),
        "v32_uses_split_dir": str(CODEX / "state" / "materialized_splits" / "validation_A_limit30"),
        "v32_load_tool_schema_paths_exist": all((EGO / "tools" / s / f"{s}_tools.json").exists() for s in ["retail", "restaurant", "order", "kitchen"]),
        "v32_db_init_imports_supported": all(v.get("schema_compatible", True) for v in [tool_db]),
        "final_hidden_runtime_policy": "not used by V37/V32/V34 runtime; final directory inventory only",
    }
    return {
        "checked_paths": {k: {"path": str(v), "exists": v.exists(), "sha256": sha256_file(v) if v.is_file() else ""} for k, v in paths.items()},
        "imports": imports_ok,
        "one_context_per_scenario": context_ok,
        "runner_assumptions": runner_assumptions,
        "compatibility_risks": [
            "Official final scenario JSON and DB/tool schemas were updated; historical V34 scores are not directly comparable until rerun.",
            "V32/V34 materialized val41 split still points to old frozen rows; smoke verifies it can construct runtime context under new DB/tools.",
        ],
    }


def write_report(run_id: str, inventory: Dict[str, Any], dataset: Dict[str, Any], tool_db: Dict[str, Any], compat: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V37_OFFICIAL_SYNC_AUDIT_{run_id}.md"
    v10_sha = sha256_file(V10_ZIP) if V10_ZIP.exists() else ""
    scenario_count = dataset.get("scenario_file_count", 0)
    video_count = inventory.get("by_top_dir", {}).get("videos", 0)
    tool_file_count = inventory.get("by_top_dir", {}).get("tools", 0)
    lines = [
        "# V37 Official Sync Audit",
        "",
        f"- run_id: `{run_id}`",
        f"- EgoBench root: `{EGO}`",
        f"- total files: {inventory.get('total_files')} ({inventory.get('total_bytes')} bytes)",
        f"- scenario JSON files: {scenario_count}",
        f"- videos: {video_count}",
        f"- tools/db/init files under tools: {tool_file_count}",
        f"- val41 materialized tasks: {dataset.get('val41_task_count')}",
        f"- final inventory only: {dataset.get('final_inventory_only')}",
        f"- final content used for experiment: {dataset.get('final_content_used_for_experiment')}",
        f"- V10 zip sha256: `{v10_sha}`",
        f"- V10 sha matches expected: {v10_sha == V10_EXPECTED_SHA}",
        "",
        "## Split / Scenario Inventory",
        "",
        "| split | task count |",
        "|---|---:|",
    ]
    for split, count in sorted((dataset.get("split_task_counts") or {}).items()):
        lines.append(f"| {split} | {count} |")
    lines.extend(["", "## Tool/DB Compatibility", "", f"- schema compatible: {tool_db.get('schema_compatible')}", ""])
    for scenario, row in (tool_db.get("tools") or {}).items():
        schema = row.get("tool_schema") or {}
        spec_count = len(row.get("specs") or [])
        lines.append(f"- {scenario}: schema tools={schema.get('count')}, specs={spec_count}")
    if tool_db.get("schema_mismatches"):
        lines.extend(["", "### Schema mismatches", ""])
        for mm in tool_db["schema_mismatches"][:20]:
            lines.append(f"- `{json.dumps(mm, ensure_ascii=False)}`")
    lines.extend(
        [
            "",
            "## Runner Compatibility",
            "",
            "| item | status |",
            "|---|---|",
        ]
    )
    for name, status in (compat.get("imports") or {}).items():
        lines.append(f"| {name} | {status} |")
    for row in compat.get("one_context_per_scenario") or []:
        lines.append(f"| context {row.get('task_key')} | {row.get('status')} video={row.get('video_exists')} tools={row.get('tool_count')} |")
    lines.extend(["", "## Main Differences / Risks", ""])
    for risk in compat.get("compatibility_risks") or []:
        lines.append(f"- {risk}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `analysis/v37_official_file_inventory.json`",
            "- `analysis/v37_official_dataset_audit.json`",
            "- `analysis/v37_tool_db_schema_audit.json`",
            "- `analysis/v37_runner_compatibility_audit.json`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    run_id = os.environ.get("V37_RUN_ID") or f"v37_official_audit_{stamp()}"
    analysis = CODEX / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    inventory = build_file_inventory()
    dataset = audit_scenarios()
    tool_db = audit_tools_and_db()
    compat = audit_runner_compatibility(tool_db)
    write_json(analysis / "v37_official_file_inventory.json", inventory)
    write_json(analysis / "v37_official_dataset_audit.json", dataset)
    write_json(analysis / "v37_tool_db_schema_audit.json", tool_db)
    write_json(analysis / "v37_runner_compatibility_audit.json", compat)
    report = write_report(run_id, inventory, dataset, tool_db, compat)
    state = {
        "run_id": run_id,
        "report": str(report),
        "inventory_total_files": inventory.get("total_files"),
        "scenario_json_files": dataset.get("scenario_file_count"),
        "val41_task_count": dataset.get("val41_task_count"),
        "schema_compatible": tool_db.get("schema_compatible"),
        "final_inventory_only": dataset.get("final_inventory_only"),
        "final_content_used_for_experiment": dataset.get("final_content_used_for_experiment"),
    }
    write_json(CODEX / "state" / "latest_v37_official_sync_audit.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
