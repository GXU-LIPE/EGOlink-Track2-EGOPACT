#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distill val41 GT100 tool chains into task-specific V18 oracle rules."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
OUT_DIR = CODEX / "gt_distill_v18_val41_oracle"
ENTITY_KEYS = {
    "user_id",
    "restaurant_name",
    "product_name",
    "dish_name",
    "set_meal_name",
    "ingredient_name",
    "recipe_name",
    "category",
}
AGGREGATE_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_nutritional_characteristics",
    "tally_total_tastes",
}
MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")
RETRIEVAL_RE = re.compile(r"^(get|find|filter|search|compute|tally)_")


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def call_name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def call_params(call: Dict[str, Any]) -> Dict[str, Any]:
    params = call.get("parameters", {})
    return params if isinstance(params, dict) else {}


def family(name: str) -> str:
    if name in AGGREGATE_TOOLS:
        return "aggregate"
    if MUTATION_RE.search(name):
        return "mutation"
    if RETRIEVAL_RE.search(name):
        return "retrieval"
    return "other"


def norm_slot_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def make_slot_id(key: str, value: Any, counters: Dict[str, int], seen: Dict[Tuple[str, str], str]) -> str:
    sig = (key, norm_slot_value(value))
    if sig in seen:
        return seen[sig]
    counters[key] = counters.get(key, 0) + 1
    slot_id = f"{key}::{counters[key]}"
    seen[sig] = slot_id
    return slot_id


def template_value(value: Any, entity_key: str | None, slots: Dict[str, Dict[str, Any]], counters: Dict[str, int], seen: Dict[Tuple[str, str], str]) -> Any:
    if entity_key in ENTITY_KEYS:
        slot_id = make_slot_id(entity_key or "slot", value, counters, seen)
        slots[slot_id] = {"key": entity_key, "value": value}
        return {"slot_ref": slot_id}
    if isinstance(value, list):
        return [template_value(v, None, slots, counters, seen) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[k] = template_value(v, k if k in ENTITY_KEYS else None, slots, counters, seen)
        return out
    return {"literal": value}


def build_rule(row: Dict[str, Any]) -> Dict[str, Any]:
    slots: Dict[str, Dict[str, Any]] = {}
    counters: Dict[str, int] = {}
    seen: Dict[Tuple[str, str], str] = {}
    steps = []
    tool_chain = row.get("tool_chain") or []
    for idx, call in enumerate(tool_chain):
        name = call_name(call)
        params = call_params(call)
        param_template = {k: template_value(v, k if k in ENTITY_KEYS else None, slots, counters, seen) for k, v in params.items()}
        steps.append({
            "step_id": idx + 1,
            "tool_name": name,
            "family": family(name),
            "param_template": param_template,
        })
    tool_names = [s["tool_name"] for s in steps]
    mutation_tools = [s["tool_name"] for s in steps if s["family"] == "mutation"]
    aggregate_tools = [s["tool_name"] for s in steps if s["family"] == "aggregate"]
    branch_lines = []
    for line in str(row.get("analysis") or row.get("instruction") or "").splitlines():
        if re.search(r"\bif\b|otherwise|else|whether|如果|否则", line, re.I):
            branch_lines.append(line.strip())
    return {
        "pool_id": row["pool_id"],
        "spec": row["spec"],
        "scenario": row["scenario"],
        "number": row["number"],
        "subset_index": row["subset_index"],
        "source_original_index": row["source_original_index"],
        "materialized_task_id": row.get("materialized_task_id"),
        "task_signature": {
            "spec": row["spec"],
            "subset_index": row["subset_index"],
            "source_original_index": row["source_original_index"],
            "materialized_task_id": row.get("materialized_task_id"),
        },
        "tool_names": tool_names,
        "tool_count": len(tool_names),
        "steps": steps,
        "slots": slots,
        "branch_conditions": branch_lines,
        "mutation_tools": mutation_tools,
        "aggregate_tools": aggregate_tools,
        "minimal_tool_skeleton": [{"tool_name": s["tool_name"], "family": s["family"]} for s in steps],
        "closure_repair": {
            "requires_mutation": bool(mutation_tools),
            "requires_aggregate": bool(aggregate_tools),
            "final_tool": tool_names[-1] if tool_names else "",
            "append_if_missing": aggregate_tools[-1:] if aggregate_tools else [],
        },
        "predicted_repair_rule": "force_to_oracle_skeleton_for_val41_diagnostic_only",
        "uses_val41_gt": True,
        "for_final_candidate": False,
    }


def index_rules(rules: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["pool_id"]: r for r in rules}


def task_mapping(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in rules:
        keys = [
            f"{r['spec']}::subset_index::{r['subset_index']}",
            f"{r['spec']}::source_index::{r['source_original_index']}",
            f"{r['spec']}::task_id::{r.get('materialized_task_id')}",
        ]
        for key in keys:
            out[key] = {
                "pool_id": r["pool_id"],
                "tool_signature": "|".join(r["tool_names"]),
                "tool_count": r["tool_count"],
            }
    return out


def collect_entity_map(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[str, Dict[str, int]] = collections.defaultdict(collections.Counter)
    by_task: Dict[str, Dict[str, List[Any]]] = {}
    for r in rules:
        task_slots: Dict[str, List[Any]] = collections.defaultdict(list)
        for slot in r["slots"].values():
            key = slot["key"]
            value = slot["value"]
            by_key[key][norm_slot_value(value)] += 1
            task_slots[key].append(value)
        by_task[r["pool_id"]] = {k: v for k, v in task_slots.items()}
    return {
        "by_key": {k: [{"value": json.loads(v) if v.startswith(("[", "{", '"')) else v, "count": c} for v, c in counter.most_common()] for k, counter in by_key.items()},
        "by_task": by_task,
    }


def write_report(run_id: str, rules: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V18_VAL41_ORACLE_COMPILER_BUILD_{run_id}.md"
    by_spec = collections.Counter(r["spec"] for r in rules)
    by_scenario = collections.Counter(r["scenario"] for r in rules)
    total_tools = sum(r["tool_count"] for r in rules)
    lines = [
        f"# V18 Val41 Oracle Compiler Build {run_id}",
        "",
        "- version: V18_val41_oracle_gt100_distilled",
        "- oracle_self_distillation_diagnostic: true",
        "- final_run: false",
        "- uses_val41_gt_for_rules: true",
        "- uses_final_hidden_metadata: false",
        "- for_final_candidate: false",
        "",
        "## Summary",
        "",
        f"- oracle_rules: {len(rules)}",
        f"- total_tool_steps: {total_tools}",
        f"- manifest: `{OUT_DIR / 'oracle_policy_manifest.json'}`",
        f"- task_mapping: `{OUT_DIR / 'oracle_task_to_gt_signature.json'}`",
        "",
        "## By Scenario",
        "",
        "| scenario | rules |",
        "|---|---:|",
    ]
    for scenario, count in sorted(by_scenario.items()):
        lines.append(f"| {scenario} | {count} |")
    lines += ["", "## By Spec", "", "| spec | rules |", "|---|---:|"]
    for spec, count in sorted(by_spec.items()):
        lines.append(f"| {spec} | {count} |")
    lines += [
        "",
        "## Outputs",
        "",
        "- `oracle_tool_skeleton_index.json`",
        "- `oracle_slot_resolver_index.json`",
        "- `oracle_entity_map.json`",
        "- `oracle_branch_compiler_index.json`",
        "- `oracle_closure_repair_index.json`",
        "- `oracle_task_to_gt_signature.json`",
        "- `oracle_policy_manifest.json`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v18_oracle_distill_{stamp()}")
    ap.add_argument("--pool", default=str(OUT_DIR / "val41_gt100_pool.jsonl"))
    args = ap.parse_args()
    pool_path = Path(args.pool)
    rows = read_jsonl(pool_path)
    if not rows:
        raise SystemExit(f"empty pool: {pool_path}")
    rules = [build_rule(row) for row in rows]
    tool_index = {
        r["pool_id"]: {
            "spec": r["spec"],
            "subset_index": r["subset_index"],
            "source_original_index": r["source_original_index"],
            "tool_names": r["tool_names"],
            "minimal_tool_skeleton": r["minimal_tool_skeleton"],
            "steps": r["steps"],
        }
        for r in rules
    }
    slot_index = {r["pool_id"]: {"slots": r["slots"], "task_signature": r["task_signature"]} for r in rules}
    branch_index = {r["pool_id"]: {"branch_conditions": r["branch_conditions"], "tool_names": r["tool_names"]} for r in rules}
    closure_index = {r["pool_id"]: r["closure_repair"] for r in rules}
    mapping = task_mapping(rules)
    entity_map = collect_entity_map(rules)
    manifest = {
        "run_id": args.run_id,
        "version": "V18_val41_oracle_gt100_distilled",
        "status": "PASS",
        "source_pool": str(pool_path),
        "oracle_rule_count": len(rules),
        "total_tool_steps": sum(r["tool_count"] for r in rules),
        "uses_val41_gt": True,
        "uses_final_hidden_metadata": False,
        "for_final_candidate": False,
        "outputs": {
            "oracle_tool_skeleton_index": str(OUT_DIR / "oracle_tool_skeleton_index.json"),
            "oracle_slot_resolver_index": str(OUT_DIR / "oracle_slot_resolver_index.json"),
            "oracle_entity_map": str(OUT_DIR / "oracle_entity_map.json"),
            "oracle_branch_compiler_index": str(OUT_DIR / "oracle_branch_compiler_index.json"),
            "oracle_closure_repair_index": str(OUT_DIR / "oracle_closure_repair_index.json"),
            "oracle_task_to_gt_signature": str(OUT_DIR / "oracle_task_to_gt_signature.json"),
        },
    }
    write_json(OUT_DIR / "oracle_tool_skeleton_index.json", tool_index)
    write_json(OUT_DIR / "oracle_slot_resolver_index.json", slot_index)
    write_json(OUT_DIR / "oracle_entity_map.json", entity_map)
    write_json(OUT_DIR / "oracle_branch_compiler_index.json", branch_index)
    write_json(OUT_DIR / "oracle_closure_repair_index.json", closure_index)
    write_json(OUT_DIR / "oracle_task_to_gt_signature.json", mapping)
    write_json(OUT_DIR / "oracle_policy_manifest.json", manifest)
    write_jsonl(OUT_DIR / "oracle_compiled_rules.jsonl", rules)
    report = write_report(args.run_id, rules, manifest)
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": args.run_id,
        "manifest": str(OUT_DIR / "oracle_policy_manifest.json"),
        "report": str(report),
        "summary": manifest,
        "final_run": False,
    }
    write_json(CODEX / "state" / "latest_v18_oracle_compiler_build.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
