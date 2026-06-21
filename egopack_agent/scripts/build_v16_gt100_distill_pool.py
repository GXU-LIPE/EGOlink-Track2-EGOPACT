#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V16 GT100 distillation pool from non-final, non-val41 Track2 data.

The script deliberately excludes official final309 specs and frozen val41
indices. It prefers full non-final scenario snapshots recovered from official
backup files, then validates exact-GT oracle outputs with EgoBench's official
evaluate_interaction_success. It does not call external APIs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
OUT_DIR = CODEX / "gt_distill_v16"

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
    "location",
    "taste",
    "tag",
}
AGGREGATE_TOOLS = {
    "compute_total_payment",
    "compute_total_price",
    "compute_total_tax",
    "compute_total_nutrition",
    "compute_total_nutritions",
    "tally_total_tastes",
}
MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")
RETRIEVAL_RE = re.compile(r"^(get|find|filter|search|compute|tally)_")


def now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def normalize_spec(value: str) -> str:
    return str(value or "").replace("_easy", "").strip()


def scenario_from_spec(spec: str) -> str:
    return re.sub(r"\d+$", "", normalize_spec(spec))


def call_name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def call_params(call: Dict[str, Any]) -> Dict[str, Any]:
    params = call.get("parameters", {})
    return params if isinstance(params, dict) else {}


def is_mutation(name: str) -> bool:
    return bool(MUTATION_RE.search(str(name)))


def is_retrieval(name: str) -> bool:
    return bool(RETRIEVAL_RE.search(str(name))) and str(name) not in AGGREGATE_TOOLS


def task_type_from_text_and_calls(text: str, calls: List[Dict[str, Any]]) -> str:
    names = [call_name(c) for c in calls]
    text_l = str(text or "").lower()
    if any(is_mutation(n) for n in names) and any(n in AGGREGATE_TOOLS for n in names):
        return "branch-then-mutation+aggregate" if any(x in text_l for x in ["if ", "otherwise", "else", "whether"]) else "mutation+aggregate"
    if any(is_mutation(n) for n in names):
        return "cart/order/menu mutation"
    if any(n in AGGREGATE_TOOLS for n in names):
        return "aggregate-required"
    if any(x in text_l for x in ["highest", "lowest", "cheapest", "most", "least"]):
        return "ranking/filtering"
    if any(x in text_l for x in ["pointed", "visible", "image", "video", "menu", "shelf"]):
        return "visual-entity query"
    return "query-only"


def family_sequence(calls: List[Dict[str, Any]]) -> List[str]:
    out = []
    for call in calls:
        name = call_name(call)
        if name in AGGREGATE_TOOLS:
            out.append("aggregate:" + name)
        elif is_mutation(name):
            out.append("mutation:" + name)
        elif is_retrieval(name):
            out.append("retrieval:" + name)
        else:
            out.append("other:" + name)
    return out


def extract_slots(calls: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    slots: Dict[str, List[Any]] = collections.defaultdict(list)
    for call in calls:
        params = call_params(call)
        for key, value in params.items():
            if key in ENTITY_KEYS:
                slots[key].append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for kk, vv in item.items():
                            if kk in ENTITY_KEYS:
                                slots[kk].append(vv)
    clean: Dict[str, List[Any]] = {}
    for key, vals in slots.items():
        seen = set()
        clean_vals = []
        for val in vals:
            sig = json.dumps(val, ensure_ascii=False, sort_keys=True) if isinstance(val, (dict, list)) else str(val)
            if sig not in seen:
                seen.add(sig)
                clean_vals.append(val)
        clean[key] = clean_vals
    return clean


def load_val41_exclusions() -> Dict[str, set]:
    excluded: Dict[str, set] = collections.defaultdict(set)
    if not (SPLIT_DIR / "manifest.json").exists():
        return excluded
    manifest = read_json(SPLIT_DIR / "manifest.json")
    for scenario, num, idxs in manifest.get("specs", []):
        spec = f"{scenario}{num}"
        for idx in idxs:
            try:
                excluded[spec].add(int(idx))
            except Exception:
                pass
        split_file = SPLIT_DIR / f"{spec}.json"
        if split_file.exists():
            data = read_json(split_file)
            if isinstance(data, dict):
                data = list(data.values())
            for item in data:
                for key in ("_v8_original_index", "task_id"):
                    if item.get(key) is not None:
                        try:
                            excluded[spec].add(int(item[key]))
                        except Exception:
                            pass
    return excluded


def list_candidate_scenario_files() -> Dict[str, List[Path]]:
    roots = [
        EGO / "scenarios" / "final",
        CODEX / "runs" / "V8_tmp_scenarios",
    ]
    by_spec: Dict[str, List[Path]] = collections.defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        if root == EGO / "scenarios" / "final":
            files = root.glob("*.json")
        else:
            files = root.glob("*/*.official_backup.json")
        for path in files:
            name = path.name.replace(".official_backup", "")
            if not name.endswith(".json"):
                continue
            spec = normalize_spec(name[:-5])
            if spec in FINAL309_SPECS:
                continue
            by_spec[spec].append(path)
    return by_spec


def file_gt_count(path: Path) -> Tuple[int, int]:
    try:
        data = read_json(path)
        if isinstance(data, dict):
            data = list(data.values())
        gt = sum(1 for item in data if isinstance(item, dict) and item.get("ground_truth"))
        return len(data), gt
    except Exception:
        return 0, 0


def choose_best_sources() -> Dict[str, Path]:
    chosen: Dict[str, Path] = {}
    for spec, paths in list_candidate_scenario_files().items():
        scored = []
        for path in paths:
            total, gt = file_gt_count(path)
            if gt:
                scored.append((gt, total, len(str(path)), path))
        if scored:
            scored.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
            chosen[spec] = scored[0][3]
    return chosen


def build_oracle_result(item: Dict[str, Any], out_task_id: int) -> Dict[str, Any]:
    calls = item.get("ground_truth") or []
    return {
        "task_id": out_task_id,
        "mode": "text",
        "instruction": item.get("Instruction", ""),
        "image_description": item.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": "V16 exact-GT replay audit."}],
        "tool_calls": [{"turn": 0, "calls": calls, "blocked_calls": [], "results": []}],
        "tool_calls_count": len(calls),
        "rounds_count": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_consumed": 0,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt": False,
    }


def evaluate_replay(gt_path: Path, result_path: Path, scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    metrics = evaluate_interaction_success(
        str(gt_path),
        str(result_path),
        scenario=scenario,
        args=_argparse.Namespace(scenario_number=int(number)),
        silent=True,
        num_samples=0,
    )
    micro = metrics.get("micro_tool_stats", {}) or {}
    return {
        "valid": metrics.get("valid_scenarios", 0),
        "joint": metrics.get("joint_success", {}).get("success_rate", 0.0),
        "result": metrics.get("result_based", {}).get("success_rate", 0.0),
        "tool": metrics.get("tool_based", {}).get("success_rate", 0.0),
        "micro": micro.get("micro_accuracy", 0.0),
        "correct_calls": micro.get("total_correct_calls", 0),
        "gt_calls": micro.get("total_ground_truth_calls", 0),
        "interaction_calls": micro.get("total_interaction_calls", 0),
    }


def collect_from_scenarios(run_id: str, replay: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    val41 = load_val41_exclusions()
    sources = choose_best_sources()
    candidates: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    source_audit = {}

    for spec, path in sorted(sources.items()):
        scenario = scenario_from_spec(spec)
        number_s = spec[len(scenario):] or "0"
        number = int(number_s)
        data = read_json(path)
        if isinstance(data, dict):
            data = list(data.values())
        source_audit[spec] = {"source_path": str(path), "raw_count": len(data), "with_gt": 0, "kept_pre_replay": 0}
        gt_items = []
        results = []
        for ordinal, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            calls = item.get("ground_truth") or []
            if not calls:
                excluded.append({"spec": spec, "ordinal": ordinal, "reason": "missing_ground_truth", "source_path": str(path)})
                continue
            source_audit[spec]["with_gt"] += 1
            original_index = item.get("_v8_original_index") or item.get("task_id") or ordinal
            try:
                original_index_i = int(original_index)
            except Exception:
                original_index_i = ordinal
            if original_index_i in val41.get(spec, set()):
                excluded.append({"spec": spec, "ordinal": ordinal, "task_id": item.get("task_id"), "original_index": original_index_i, "reason": "excluded_frozen_val41", "source_path": str(path)})
                continue
            source_audit[spec]["kept_pre_replay"] += 1
            out_task_id = len(gt_items) + 1
            copied = dict(item)
            copied["task_id"] = out_task_id
            gt_items.append(copied)
            results.append(build_oracle_result(copied, out_task_id))
            calls_hash = hashlib.sha256(json.dumps(calls, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            candidates.append({
                "pool_id": f"scenario::{spec}::{original_index_i}::{calls_hash[:10]}",
                "source_kind": "scenario_gt",
                "source_path": str(path),
                "scenario": scenario,
                "spec": spec,
                "task_id": item.get("task_id"),
                "original_index": original_index_i,
                "instruction": item.get("Instruction", ""),
                "analysis": item.get("analysis", ""),
                "image_description": item.get("image_description", ""),
                "tool_calls": calls,
                "tool_names": [call_name(c) for c in calls],
                "tool_family_sequence": family_sequence(calls),
                "entity_slots": extract_slots(calls),
                "task_type": task_type_from_text_and_calls((item.get("Instruction", "") + "\n" + item.get("analysis", "")), calls),
                "replay_spec_task_id": out_task_id,
                "replay_status": "pending",
                "no_final_metadata": True,
                "excluded_final309": True,
                "excluded_val41": True,
            })

        if replay and gt_items:
            replay_dir = CODEX / "runs" / "V16_GT100_REPLAY" / run_id
            gt_path = replay_dir / "gt" / f"{spec}.json"
            result_path = replay_dir / "results" / f"{spec}_easy.json"
            write_json(gt_path, gt_items)
            write_json(result_path, results)
            try:
                metrics = evaluate_replay(gt_path, result_path, scenario, number)
                source_audit[spec]["replay_metrics"] = metrics
                # Exact GT should replay joint=1. If not, keep per-spec out but mark rows excluded.
                ok = metrics.get("valid", 0) == len(gt_items) and metrics.get("joint", 0.0) >= 0.999 and metrics.get("tool", 0.0) >= 0.999
            except Exception as exc:
                source_audit[spec]["replay_error"] = f"{type(exc).__name__}: {exc}"
                ok = False
            for row in candidates:
                if row["spec"] == spec and row["source_path"] == str(path):
                    row["replay_status"] = "joint100" if ok else "replay_failed"
        elif not replay:
            for row in candidates:
                if row["spec"] == spec and row["source_path"] == str(path):
                    row["replay_status"] = "not_replayed"
    return candidates, excluded, source_audit


def collect_train_data_fallback(existing_keys: set) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    val41 = load_val41_exclusions()
    rows: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for path in [CODEX / "train_data" / "sft_track2_tooluse_train.jsonl", CODEX / "train_data" / "sft_track2_tooluse_val.jsonl"]:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                excluded.append({"source_path": str(path), "line": line_no, "reason": "json_error"})
                continue
            source = item.get("source")
            if source not in {"gt", "success_trace", "teacher_corrected"}:
                continue
            spec = normalize_spec(item.get("scenario") or "")
            if spec in FINAL309_SPECS:
                excluded.append({"source_path": str(path), "line": line_no, "spec": spec, "reason": "excluded_final309"})
                continue
            scenario = scenario_from_spec(spec)
            try:
                task_id_i = int(item.get("task_id"))
            except Exception:
                task_id_i = -1
            if task_id_i in val41.get(spec, set()):
                excluded.append({"source_path": str(path), "line": line_no, "spec": spec, "task_id": task_id_i, "reason": "excluded_frozen_val41"})
                continue
            target = item.get("target") or ""
            try:
                calls = json.loads(target)
            except Exception:
                excluded.append({"source_path": str(path), "line": line_no, "spec": spec, "reason": "target_json_error"})
                continue
            if not isinstance(calls, list) or not calls:
                continue
            calls_hash = hashlib.sha256(json.dumps(calls, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            key = (spec, task_id_i, calls_hash)
            if key in existing_keys:
                continue
            rows.append({
                "pool_id": f"train::{spec}::{task_id_i}::{calls_hash[:10]}",
                "source_kind": "train_data_gt_or_success",
                "source_path": f"{path}:{line_no}",
                "scenario": scenario,
                "spec": spec,
                "task_id": item.get("task_id"),
                "original_index": task_id_i,
                "instruction": "",
                "analysis": item.get("planner_state", ""),
                "image_description": item.get("visual_context", ""),
                "tool_calls": calls,
                "tool_names": [call_name(c) for c in calls],
                "tool_family_sequence": family_sequence(calls),
                "entity_slots": extract_slots(calls),
                "task_type": task_type_from_text_and_calls(item.get("planner_state", ""), calls),
                "replay_status": "not_replayed_train_data_fallback",
                "no_final_metadata": True,
                "excluded_final309": True,
                "excluded_val41": True,
            })
    return rows, excluded


def build_modules(pool: List[Dict[str, Any]]) -> Dict[str, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_key: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
    slot_counter: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    lexicon: Dict[str, Dict[str, set]] = collections.defaultdict(lambda: collections.defaultdict(set))
    branch_templates = []

    for row in pool:
        names = tuple(row.get("tool_names") or [])
        key = (row["scenario"], row["task_type"], names)
        rec = by_key.setdefault(key, {
            "scenario": row["scenario"],
            "task_type": row["task_type"],
            "tool_names": list(names),
            "tool_family_sequence": row.get("tool_family_sequence", []),
            "count": 0,
            "examples": [],
        })
        rec["count"] += 1
        if len(rec["examples"]) < 8:
            rec["examples"].append({"spec": row["spec"], "task_id": row.get("task_id"), "source_kind": row["source_kind"]})
        for call in row["tool_calls"]:
            tool = call_name(call)
            for param, value in call_params(call).items():
                slot_counter[tool][param] += 1
        for slot, values in (row.get("entity_slots") or {}).items():
            for value in values:
                if isinstance(value, str) and value.strip():
                    lexicon[row["scenario"]][slot].add(value.strip())
        text = (row.get("instruction") or "") + "\n" + (row.get("analysis") or "")
        if any(x in text.lower() for x in ["if ", "otherwise", "else", "whether"]):
            branch_templates.append({
                "scenario": row["scenario"],
                "task_type": row["task_type"],
                "condition_hint": text[:1000],
                "tool_names": row["tool_names"],
                "entity_slots": row.get("entity_slots", {}),
                "source_pool_id": row["pool_id"],
                "no_final_metadata": True,
            })

    automata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": "V16 GT100 distilled executable skeletons from non-final non-val41 joint100 trajectories.",
        "templates": sorted(by_key.values(), key=lambda x: (-x["count"], x["scenario"], x["task_type"]))[:500],
        "default_stages": {
            "retrieval_stage": ["resolve pins", "retrieve constrained candidates", "canonicalize slots"],
            "branch_check_stage": ["query branch-critical attribute", "choose exactly one branch"],
            "mutation_stage": ["perform required add/remove/update once"],
            "final_aggregate_stage": ["compute requested payment/tax/nutrition/summary once near end"],
        },
        "no_final_metadata": True,
    }
    slot_rules = {
        "generated_at": automata["generated_at"],
        "tool_param_rules": [
            {
                "tool_name": tool,
                "common_params": [k for k, _ in ctr.most_common()],
                "entity_params": [k for k, _ in ctr.most_common() if k in ENTITY_KEYS],
                "observed_param_count": int(sum(ctr.values())),
            }
            for tool, ctr in sorted(slot_counter.items())
        ],
        "branch_condition_templates": branch_templates[:300],
        "no_final_metadata": True,
    }
    lexicon_json = {
        scenario: {slot: sorted(values)[:300] for slot, values in slots.items()}
        for scenario, slots in sorted(lexicon.items())
    }
    anti = {
        "rules": [
            {"scenario": "global", "name": "missing_aggregate", "rule": "If GT-like skeleton contains an aggregate/final compute and the candidate lacks one, append the minimal matching aggregate before final answer."},
            {"scenario": "order", "name": "wrong_restaurant_pin", "rule": "After restaurant pin, every order dish/set_meal/aggregate call must use that restaurant_name."},
            {"scenario": "order", "name": "dish_set_meal_confusion", "rule": "Use get_set_meal_details for set meals and do not send set_meal_name to dish mutation tools."},
            {"scenario": "retail", "name": "broad_scan", "rule": "Do not use unbounded price/tax/discount/nutrition scans before candidate narrowing."},
            {"scenario": "restaurant", "name": "query_only_overplanning", "rule": "For single item attribute questions, retrieve/answer with the matching minimal tool; do not mutate."},
            {"scenario": "kitchen", "name": "branch_quantity_failure", "rule": "Nutrition/shopping list quantities must come from confirmed recipe/current state, not memory."},
            {"scenario": "global", "name": "visual_followup", "rule": "Use visual prior plus constrained retrieval; do not ask user for visible names when tools can narrow."},
        ],
        "no_final_metadata": True,
    }
    repairs = {
        "templates": [
            {"if": "missing_final_aggregate", "then": "append compute_total_payment/tax/nutrition/summary matching user request and current scenario schema"},
            {"if": "active_restaurant_missing", "then": "pin restaurant from user instruction/ledger/visual prior before order mutation"},
            {"if": "visual_entity_uncertain", "then": "retrieve constrained candidates only, then canonicalize"},
            {"if": "branch_unresolved", "then": "run branch-check retrieval tools before mutation"},
        ],
        "no_final_metadata": True,
    }
    weights = {
        "schema_valid": 8,
        "skeleton_coverage": 12,
        "slot_completeness": 10,
        "mutation_closure": 14,
        "aggregate_closure": 12,
        "restaurant_user_pin_correct": 14,
        "unconstrained_broad_scan_penalty": -14,
        "duplicate_mutation_penalty": -12,
        "visual_followup_penalty": -10,
        "tool_count_penalty_per_extra": -0.2,
        "gt_like_sequence_bonus": 8,
        "no_final_metadata": True,
    }
    counts = {
        "tool_sequence_automata": len(automata["templates"]),
        "slot_resolver_tool_rules": len(slot_rules["tool_param_rules"]),
        "branch_condition_templates": len(branch_templates[:300]),
        "scenario_entity_lexicon_scenarios": len(lexicon_json),
        "anti_failure_rules": len(anti["rules"]),
        "process_repair_templates": len(repairs["templates"]),
    }
    write_json(OUT_DIR / "tool_sequence_automata.json", automata)
    write_json(OUT_DIR / "slot_resolver_rules.json", slot_rules)
    write_json(OUT_DIR / "scenario_entity_lexicon.json", lexicon_json)
    write_json(OUT_DIR / "anti_failure_rules.json", anti)
    write_json(OUT_DIR / "process_repair_templates.json", repairs)
    write_json(OUT_DIR / "candidate_rerank_weights.json", weights)
    return counts


def write_pool_report(run_id: str, manifest: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V16_GT100_POOL_AUDIT_{run_id}.md"
    lines = [
        f"# V16 GT100 Pool Audit {run_id}",
        "",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- excludes_official_final309: true",
        "- excludes_frozen_val41: true",
        "- uses_val41_label_for_training: false",
        "",
        "## Counts",
        "",
    ]
    for key in [
        "raw_scenario_gt_candidates",
        "scenario_joint100_kept",
        "train_data_fallback_kept",
        "total_pool_rows",
        "excluded_rows",
    ]:
        lines.append(f"- {key}: {manifest.get(key, 0)}")
    lines += ["", "## Scenario Distribution", ""]
    for key, val in sorted((manifest.get("scenario_distribution") or {}).items()):
        lines.append(f"- {key}: {val}")
    lines += ["", "## Task Type Distribution", ""]
    for key, val in sorted((manifest.get("task_type_distribution") or {}).items()):
        lines.append(f"- {key}: {val}")
    lines += ["", "## Module Counts", ""]
    for key, val in sorted((manifest.get("module_counts") or {}).items()):
        lines.append(f"- {key}: {val}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v16_gt100_pool_{now()}")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()

    scenario_rows, excluded, source_audit = collect_from_scenarios(args.run_id, replay=not args.no_replay)
    raw_count = len(scenario_rows)
    joint100_rows = [r for r in scenario_rows if r["replay_status"] in {"joint100", "not_replayed"}]
    existing = {(r["spec"], int(r.get("original_index") or -1), hashlib.sha256(json.dumps(r["tool_calls"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()) for r in joint100_rows}
    fallback_rows, fallback_excluded = collect_train_data_fallback(existing)
    pool = joint100_rows + fallback_rows
    pool = sorted(pool, key=lambda r: (r["scenario"], r["spec"], int(r.get("original_index") or 0), r["pool_id"]))
    module_counts = build_modules(pool)

    scenario_dist = collections.Counter(r["scenario"] for r in pool)
    type_dist = collections.Counter(r["task_type"] for r in pool)
    manifest = {
        "run_id": args.run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "excludes_official_final309": True,
        "excluded_final309_specs": sorted(FINAL309_SPECS),
        "excludes_frozen_val41": True,
        "frozen_val41_manifest": str(SPLIT_DIR / "manifest.json"),
        "uses_val41_label_for_training": False,
        "raw_scenario_gt_candidates": raw_count,
        "scenario_joint100_kept": len(joint100_rows),
        "train_data_fallback_kept": len(fallback_rows),
        "total_pool_rows": len(pool),
        "excluded_rows": len(excluded) + len(fallback_excluded) + (raw_count - len(joint100_rows)),
        "scenario_distribution": dict(scenario_dist),
        "task_type_distribution": dict(type_dist),
        "module_counts": module_counts,
        "source_audit": source_audit,
        "outputs": {
            "pool": str(OUT_DIR / "gt100_pool.jsonl"),
            "manifest": str(OUT_DIR / "gt100_pool_manifest.json"),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "gt100_pool.jsonl", pool)
    write_jsonl(OUT_DIR / "gt100_pool_excluded.jsonl", excluded + fallback_excluded)
    write_json(OUT_DIR / "gt100_pool_manifest.json", manifest)
    report = write_pool_report(args.run_id, manifest)
    manifest["report"] = str(report)
    write_json(OUT_DIR / "gt100_pool_manifest.json", manifest)
    write_json(CODEX / "state" / "latest_v16_gt100_pool.json", manifest)
    print(json.dumps({"run_id": args.run_id, "pool_rows": len(pool), "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
