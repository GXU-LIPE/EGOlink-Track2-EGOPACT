#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V19 GT100 executable case library from non-final/non-val41 pool."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
POOL = CODEX / "gt_distill_v16" / "gt100_pool.jsonl"
MANIFEST = CODEX / "gt_distill_v16" / "gt100_pool_manifest.json"
OUT = CODEX / "gt_case_library_v19"
FINAL309_SPECS = {"retail6", "retail10", "kitchen4", "restaurant5", "order2"}
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


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def call_name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def call_params(call: Dict[str, Any]) -> Dict[str, Any]:
    params = call.get("parameters", {})
    return params if isinstance(params, dict) else {}


def stage_for_tool(name: str) -> str:
    if name in AGGREGATE_TOOLS:
        return "aggregate"
    if MUTATION_RE.search(name):
        return "mutation"
    if name.startswith(("get_", "find_", "filter_", "search_")):
        return "retrieve"
    return "answer"


def program_shape(names: List[str]) -> str:
    return " > ".join(f"{stage_for_tool(n)}:{n}" for n in names)


def extract_slots(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    slots: Dict[str, List[Any]] = collections.defaultdict(list)
    def add(key: str, value: Any) -> None:
        if value is None or value == "":
            return
        if value not in slots[key]:
            slots[key].append(value)
    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ENTITY_KEYS:
                    add(k, v)
                visit(v)
        elif isinstance(obj, list):
            for x in obj:
                visit(x)
    for call in calls:
        visit(call_params(call))
    return dict(slots)


def visual_phrase(row: Dict[str, Any]) -> str:
    text = " ".join([str(row.get("instruction", "")), str(row.get("image_description", "")), str(row.get("analysis", ""))])
    hits = []
    for pat in [r"point[^.]{0,120}", r"visible[^.]{0,120}", r"holding[^.]{0,120}", r"left[^.]{0,120}", r"right[^.]{0,120}", r"menu[^.]{0,120}", r"shelf[^.]{0,120}"]:
        hits.extend(re.findall(pat, text, re.I))
    return " | ".join(hits[:5])


def branch_logic(row: Dict[str, Any]) -> Dict[str, Any]:
    lines = [x.strip() for x in str(row.get("analysis") or row.get("instruction") or "").splitlines() if re.search(r"\bif\b|otherwise|else|whether", x, re.I)]
    return {"branch_condition_text": lines, "has_branch": bool(lines)}


def make_case(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    calls = row.get("tool_calls") or row.get("tool_chain") or []
    names = [call_name(c) for c in calls]
    tool_program = []
    for call in calls:
        name = call_name(call)
        tool_program.append({
            "tool_name": name,
            "parameters": call_params(call),
            "stage": stage_for_tool(name),
        })
    slots = row.get("entity_slots") or extract_slots(calls)
    entity_types = sorted(k for k, v in slots.items() if v)
    case_id = row.get("pool_id") or f"v19_case_{idx:04d}"
    return {
        "case_id": case_id,
        "scenario": row.get("scenario"),
        "spec": row.get("spec"),
        "source_task_id": row.get("task_id") or row.get("replay_spec_task_id"),
        "instruction_text": row.get("instruction", ""),
        "user_utterance_pattern": re.sub(r"\s+", " ", str(row.get("instruction", "")))[:600],
        "task_type": row.get("task_type") or "unknown",
        "visual_phrase": visual_phrase(row),
        "tool_program": tool_program,
        "tool_name_sequence": names,
        "program_shape": program_shape(names),
        "required_slots": {
            "user_id": slots.get("user_id", []),
            "restaurant_name": slots.get("restaurant_name", []),
            "product_name": slots.get("product_name", []),
            "dish_name": slots.get("dish_name", []),
            "set_meal_name": slots.get("set_meal_name", []),
            "ingredient_name": slots.get("ingredient_name", []),
            "recipe_name": slots.get("recipe_name", []),
            "category": slots.get("category", []),
            "branch_condition": branch_logic(row).get("branch_condition_text", []),
            "aggregate_target": [n for n in names if n in AGGREGATE_TOOLS],
        },
        "entity_types": entity_types,
        "branch_logic": branch_logic(row),
        "mutation_closure": [n for n in names if MUTATION_RE.search(n)],
        "aggregate_closure": [n for n in names if n in AGGREGATE_TOOLS],
        "replay_joint100": row.get("replay_status") == "joint100",
        "excluded_final309": bool(row.get("excluded_final309")) and row.get("spec") not in FINAL309_SPECS,
        "excluded_val41": bool(row.get("excluded_val41")),
        "source_path": row.get("source_path"),
        "no_final_metadata": bool(row.get("no_final_metadata", True)),
    }


def index_by(cases: List[Dict[str, Any]], key: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = collections.defaultdict(list)
    for case in cases:
        value = case.get(key)
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for v in values:
            out[str(v)].append(case["case_id"])
    return dict(out)


def write_reflection(run_id: str) -> Path:
    path = CODEX / "reports" / f"V19_ROUTE_CORRECTION_REFLECTION_{run_id}.md"
    lines = [
        f"# V19 Route Correction Reflection {run_id}",
        "",
        "- final_run: false",
        "- protected_best_updated: false",
        "- v10_zip_overwritten: false",
        "",
        "1. V10 memory mainly stored natural-language scoring, process, failure, and canonicalization cards. It did not preserve complete executable GT programs.",
        "2. V12 Qwen3-VL improved part of visual grounding, but visual priors alone cannot guarantee official tool trajectory shape.",
        "3. V14 oracle showed exact GT trajectories can reach 100%, but the non-oracle slot-filling and trajectory transfer remained weak.",
        "4. V16/V17 compressed GT100 into abstract rules/compiler behavior, losing case-level program structure; the val41 smoke/clean results did not beat V14.",
        "5. V18 oracle compiler reached 100%, confirming runner/evaluator/execution plumbing is sound. The bottleneck is transferring a similar GT case to the current task.",
        "6. V19 keeps full GT100 cases and tests kNN trajectory reuse plus slot rewrite plus non-oracle program scoring.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_report(run_id: str, cases: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Path:
    path = CODEX / "reports" / f"V19_CASE_LIBRARY_AUDIT_{run_id}.md"
    by_scenario = collections.Counter(c["scenario"] for c in cases)
    by_type = collections.Counter(c["task_type"] for c in cases)
    lines = [
        f"# V19 Case Library Audit {run_id}",
        "",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_label_for_training: false",
        "- excludes_final309: true",
        "- excludes_val41: true",
        "- preserves_full_gt_programs: true",
        "",
        "## Summary",
        "",
        f"- cases: {len(cases)}",
        f"- replay_joint100_rate: {manifest.get('replay_joint100_rate'):.4f}",
        f"- all_excluded_final309: {manifest.get('all_excluded_final309')}",
        f"- all_excluded_val41: {manifest.get('all_excluded_val41')}",
        f"- scenario_coverage: {', '.join(sorted(by_scenario))}",
        "",
        "## By Scenario",
        "",
        "| scenario | cases |",
        "|---|---:|",
    ]
    for k, v in sorted(by_scenario.items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "## By Task Type", "", "| task_type | cases |", "|---|---:|"]
    for k, v in sorted(by_type.items()):
        lines.append(f"| {k} | {v} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v19_case_library_{stamp()}")
    ap.add_argument("--min-cases", type=int, default=600)
    args = ap.parse_args()
    raw = read_jsonl(POOL)
    cases = []
    for idx, row in enumerate(raw, start=1):
        case = make_case(row, idx)
        if case["replay_joint100"] and case["excluded_final309"] and case["excluded_val41"]:
            cases.append(case)
    if len(cases) < args.min_cases:
        raise SystemExit(f"effective case count below threshold: {len(cases)} < {args.min_cases}")
    replay_rate = sum(1 for c in cases if c["replay_joint100"]) / len(cases)
    manifest_in = read_json(MANIFEST, {})
    manifest = {
        "run_id": args.run_id,
        "source_pool": str(POOL),
        "source_manifest": str(MANIFEST),
        "input_pool_rows": len(raw),
        "cases": len(cases),
        "replay_joint100_rate": replay_rate,
        "all_excluded_final309": all(c["excluded_final309"] for c in cases),
        "all_excluded_val41": all(c["excluded_val41"] for c in cases),
        "scenario_distribution": dict(collections.Counter(c["scenario"] for c in cases)),
        "task_type_distribution": dict(collections.Counter(c["task_type"] for c in cases)),
        "v16_manifest_total_pool_rows": manifest_in.get("total_pool_rows"),
        "uses_final_hidden_metadata": False,
        "uses_val41_label_for_training": False,
        "for_final_candidate": False,
    }
    write_jsonl(OUT / "gt100_cases.jsonl", cases)
    write_json(OUT / "gt100_case_manifest.json", manifest)
    write_jsonl(OUT / "case_embedding_texts.jsonl", [{
        "case_id": c["case_id"],
        "text": "\n".join([c["scenario"], c["task_type"], c["instruction_text"], c["visual_phrase"], c["program_shape"]]),
    } for c in cases])
    write_jsonl(OUT / "case_tool_programs.jsonl", [{"case_id": c["case_id"], "tool_program": c["tool_program"], "tool_name_sequence": c["tool_name_sequence"], "program_shape": c["program_shape"]} for c in cases])
    write_jsonl(OUT / "case_slot_signatures.jsonl", [{"case_id": c["case_id"], "required_slots": c["required_slots"], "entity_types": c["entity_types"]} for c in cases])
    write_json(OUT / "case_task_type_index.json", index_by(cases, "task_type"))
    write_json(OUT / "case_scenario_index.json", index_by(cases, "scenario"))
    lex: Dict[str, List[str]] = collections.defaultdict(list)
    for c in cases:
        for key, vals in c["required_slots"].items():
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and v:
                        lex[v.lower()].append(c["case_id"])
    write_json(OUT / "case_lexical_entity_index.json", dict(lex))
    write_json(OUT / "case_program_shape_index.json", index_by(cases, "program_shape"))
    reflection = write_reflection(args.run_id)
    report = write_report(args.run_id, cases, manifest)
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": args.run_id,
        "case_dir": str(OUT),
        "manifest": str(OUT / "gt100_case_manifest.json"),
        "reflection_report": str(reflection),
        "case_audit_report": str(report),
        "summary": manifest,
        "final_run": False,
    }
    write_json(CODEX / "state" / "latest_v19_case_library.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
