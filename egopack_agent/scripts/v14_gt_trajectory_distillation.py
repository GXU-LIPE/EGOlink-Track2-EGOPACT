#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 GT trajectory audit, oracle replay, and distilled process-bank builder.

This script is validation-only. It reads the frozen validation_A_medium split
and non-final/dev artifacts already available under codex. It does not read
final hidden metadata and does not call any external API.
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
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V12_MODEL_DIR = EGO / "results" / "gpt-5.5-V12_official_style_qwen3vl_memory-V12_qwen3vl_prior_all_modules_val41_parallel_20260619_170302"


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
}
MUTATION_RE = re.compile(r"^(add|remove|delete|update|modify)_|_(to|from)_(cart|order|shopping_list|menu)$")
RETRIEVAL_RE = re.compile(r"^(get|find|filter|search|compute)_")


def now_stamp() -> str:
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def load_val41() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    manifest = read_json(SPLIT_DIR / "manifest.json")
    for scenario, num, idxs in manifest["specs"]:
        spec = f"{scenario}{num}"
        data = read_json(SPLIT_DIR / f"{spec}.json")
        if isinstance(data, dict):
            data = list(data.values())
        pred_path = V12_MODEL_DIR / f"{spec}_easy.json"
        pred_data = read_json(pred_path) if pred_path.exists() else []
        for i, item in enumerate(data):
            rows.append(
                {
                    "scenario": scenario,
                    "number": int(num),
                    "spec": spec,
                    "subset_index": i + 1,
                    "original_index": item.get("_v8_original_index"),
                    "task_id": item.get("task_id", i + 1),
                    "gt": item,
                    "pred": pred_data[i] if i < len(pred_data) else None,
                }
            )
    return rows


def flatten_calls_from_result(result: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(result, dict):
        return out
    for entry in result.get("tool_calls", []) or []:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("calls"), list):
            out.extend([c for c in entry["calls"] if isinstance(c, dict)])
        elif isinstance(entry.get("call"), dict):
            out.append(entry["call"])
    return out


def call_name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def call_params(call: Dict[str, Any]) -> Dict[str, Any]:
    params = call.get("parameters", {})
    return params if isinstance(params, dict) else {}


def is_mutation(name: str) -> bool:
    return bool(MUTATION_RE.search(name))


def is_retrieval(name: str) -> bool:
    return bool(RETRIEVAL_RE.search(name)) and name not in AGGREGATE_TOOLS


def extract_slots(calls: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    slots: Dict[str, List[Any]] = collections.defaultdict(list)
    for call in calls:
        for k, v in call_params(call).items():
            if k in ENTITY_KEYS:
                slots[k].append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for kk, vv in item.items():
                            if kk in ENTITY_KEYS:
                                slots[kk].append(vv)
    return {k: list(dict.fromkeys([json.dumps(x, ensure_ascii=False, sort_keys=True) if isinstance(x, (dict, list)) else x for x in vals])) for k, vals in slots.items()}


def sequence_family(calls: List[Dict[str, Any]]) -> List[str]:
    fam = []
    for call in calls:
        name = call_name(call)
        if name in AGGREGATE_TOOLS:
            fam.append("aggregate:" + name)
        elif is_mutation(name):
            fam.append("mutation:" + name)
        elif is_retrieval(name):
            fam.append("retrieval:" + name)
        else:
            fam.append("other:" + name)
    return fam


def first_entity_mismatch(gt_calls: List[Dict[str, Any]], pred_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_slots = extract_slots(gt_calls)
    pred_slots = extract_slots(pred_calls)
    diffs = {}
    for key in sorted(set(gt_slots) | set(pred_slots)):
        g = {normalize_text(x) for x in gt_slots.get(key, [])}
        p = {normalize_text(x) for x in pred_slots.get(key, [])}
        if g and p and not (g & p):
            diffs[key] = {"gt": gt_slots.get(key, []), "pred": pred_slots.get(key, [])}
    return diffs


def classify_mismatch(row: Dict[str, Any], gt_calls: List[Dict[str, Any]], pred_calls: List[Dict[str, Any]]) -> List[str]:
    types: List[str] = []
    gt_names = [call_name(c) for c in gt_calls]
    pred_names = [call_name(c) for c in pred_calls]
    gt_name_set = set(gt_names)
    pred_name_set = set(pred_names)
    slot_diff = first_entity_mismatch(gt_calls, pred_calls)
    if not pred_calls and gt_calls:
        types.append("query-only over-planning")
    if gt_name_set - pred_name_set:
        types.append("tool type wrong")
    if any(name in AGGREGATE_TOOLS for name in gt_names) and not any(name in AGGREGATE_TOOLS for name in pred_names):
        types.append("missing final aggregate")
    if slot_diff:
        if "restaurant_name" in slot_diff or "user_id" in slot_diff:
            types.append("restaurant/user pin wrong")
        entity_keys = {"product_name", "dish_name", "set_meal_name", "ingredient_name", "recipe_name", "category"}
        if entity_keys & set(slot_diff):
            types.append("visual entity wrong")
            types.append("canonical name mismatch")
    if pred_calls:
        pred_fam = sequence_family(pred_calls)
        if pred_fam and pred_fam[0].startswith("mutation:") and any(x.startswith("retrieval:") for x in sequence_family(gt_calls)):
            types.append("missing retrieval before mutation")
        if sum(1 for x in pred_fam if x.startswith("retrieval:")) > max(8, 3 * max(1, len(gt_calls))):
            types.append("broad scan instead of constrained candidate")
        if len(pred_calls) > max(20, 5 * max(1, len(gt_calls))):
            types.append("overlong trajectory / loop")
    analysis = row["gt"].get("analysis") or row["gt"].get("Instruction") or ""
    if re.search(r"\bif\b|otherwise|else|是否|如果", analysis, re.I) and set(gt_names) != set(pred_names):
        types.append("branch condition wrong")
    if not types and gt_names != pred_names:
        types.append("tool type wrong")
    return list(dict.fromkeys(types))


def build_oracle_result(row: Dict[str, Any]) -> Dict[str, Any]:
    gt = row["gt"]
    calls = gt.get("ground_truth") or []
    return {
        "task_id": row["subset_index"],
        "mode": "text",
        "instruction": gt.get("Instruction", ""),
        "image_description": gt.get("image_description", ""),
        "visual_cache_id": f"{row['spec']}_{row['subset_index']}",
        "analysis": gt.get("analysis", ""),
        "v14_oracle_teacher": True,
        "uses_val41_gt": True,
        "final_run": False,
        "dialogue": [
            {
                "role": "agent",
                "turn": 0,
                "content": "V14 oracle teacher replay for val41 debugging only.",
            }
        ],
        "tool_calls": [
            {
                "turn": 0,
                "calls": calls,
                "blocked_calls": [],
                "results": [],
                "v14_source": "val41_ground_truth",
            }
        ],
        "tokens_consumed": 0,
        "rounds_count": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls_count": len(calls),
    }


def evaluate_model_dir(model_dir: Path, run_id: str, version: str) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    rows = []
    manifest = read_json(SPLIT_DIR / "manifest.json")
    for scenario, num, _idxs in manifest["specs"]:
        spec = f"{scenario}{num}"
        gt_path = SPLIT_DIR / f"{spec}.json"
        result_path = model_dir / f"{spec}_easy.json"
        if not result_path.exists():
            rows.append({"spec": spec, "valid": 0, "error": "missing_result"})
            continue
        try:
            metrics = evaluate_interaction_success(
                str(gt_path),
                str(result_path),
                scenario=scenario,
                args=_argparse.Namespace(scenario_number=int(num)),
                silent=True,
                num_samples=0,
            )
            micro = metrics.get("micro_tool_stats", {}) or {}
            rows.append(
                {
                    "spec": spec,
                    "scenario": scenario,
                    "number": int(num),
                    "valid": metrics.get("valid_scenarios", 0),
                    "joint": metrics.get("joint_success", {}).get("success_rate", 0.0),
                    "result": metrics.get("result_based", {}).get("success_rate", 0.0),
                    "tool": metrics.get("tool_based", {}).get("success_rate", 0.0),
                    "micro": micro.get("micro_accuracy", 0.0),
                    "avg_task_accuracy": micro.get("avg_task_accuracy", 0.0),
                    "correct_calls": micro.get("total_correct_calls", 0),
                    "gt_calls": micro.get("total_ground_truth_calls", 0),
                    "interaction_calls": micro.get("total_interaction_calls", 0),
                    "detailed_results": metrics.get("detailed_results", []),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append({"spec": spec, "valid": 0, "error": f"{type(exc).__name__}: {exc}"})
    total_valid = sum(x.get("valid", 0) for x in rows)
    def wavg(key: str) -> float:
        return sum(x.get(key, 0.0) * x.get("valid", 0) for x in rows) / total_valid if total_valid else 0.0
    correct = sum(x.get("correct_calls", 0) for x in rows)
    gt_calls = sum(x.get("gt_calls", 0) for x in rows)
    interaction = sum(x.get("interaction_calls", 0) for x in rows)
    summary = {
        "valid": total_valid,
        "joint": wavg("joint"),
        "result": wavg("result"),
        "tool": wavg("tool"),
        "micro": correct / gt_calls if gt_calls else wavg("micro"),
        "avg_task_accuracy": wavg("avg_task_accuracy"),
        "correct_calls": correct,
        "gt_calls": gt_calls,
        "interaction_calls": interaction,
    }
    out = {"run_id": run_id, "version": version, "model_dir": str(model_dir), "summary": summary, "rows": rows}
    write_json(CODEX / "runs" / version / run_id / "eval_summary.json", out)
    return out


def collect_dev_gt_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # SFT files are non-final/dev-oriented artifacts already built from GT/success traces.
    for path in [CODEX / "train_data" / "sft_track2_tooluse_train.jsonl", CODEX / "train_data" / "sft_track2_tooluse_val.jsonl"]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("source") not in {"gt", "success_trace", "teacher_corrected"}:
                continue
            target = item.get("target") or ""
            try:
                calls = json.loads(target)
            except Exception:
                calls = []
            if isinstance(calls, list) and calls:
                rows.append(
                    {
                        "source_path": str(path),
                        "scenario": item.get("scenario", "").rstrip("0123456789") or item.get("scenario"),
                        "scenario_spec": item.get("scenario"),
                        "task_id": item.get("task_id"),
                        "analysis": item.get("planner_state", ""),
                        "ground_truth": calls,
                        "source": item.get("source"),
                    }
                )
    for row in load_val41():
        rows.append(
            {
                "source_path": str(SPLIT_DIR / f"{row['spec']}.json"),
                "scenario": row["scenario"],
                "scenario_spec": row["spec"],
                "task_id": row["task_id"],
                "analysis": row["gt"].get("analysis", ""),
                "ground_truth": row["gt"].get("ground_truth") or [],
                "source": "val41_oracle_debug",
                "val41_only_not_final": True,
            }
        )
    return rows


def infer_task_type_from_calls(calls: List[Dict[str, Any]], analysis: str = "") -> str:
    names = [call_name(c) for c in calls]
    text = normalize_text(analysis)
    if any(is_mutation(n) for n in names) and any(n in AGGREGATE_TOOLS for n in names):
        return "branch-then-mutation+aggregate"
    if any(is_mutation(n) for n in names):
        return "cart/order/menu mutation"
    if any(n in AGGREGATE_TOOLS for n in names):
        return "aggregate-required"
    if "highest" in text or "lowest" in text or "最" in text:
        return "ranking/filtering"
    if "point" in text or "visible" in text or "visual" in text:
        return "visual-entity query"
    return "query-only"


def build_process_bank(dev_rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_counter: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
    tool_param_keys: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    branch_cards = []
    val41_cards = []
    for row in dev_rows:
        calls = row.get("ground_truth") or []
        if not calls:
            continue
        scenario = (row.get("scenario") or "").rstrip("0123456789")
        task_type = infer_task_type_from_calls(calls, row.get("analysis", ""))
        names = tuple(call_name(c) for c in calls)
        key = (scenario, task_type, names)
        rec = seq_counter.setdefault(
            key,
            {
                "card_id": "v14::tool_sequence::" + hashlib.sha1(("|".join(key[0:2]) + "|" + "->".join(names)).encode()).hexdigest()[:12],
                "scenario": scenario,
                "task_type": task_type,
                "tool_names": list(names),
                "tool_family_sequence": sequence_family(calls),
                "count": 0,
                "examples": [],
                "no_final_metadata": True,
            },
        )
        rec["count"] += 1
        if len(rec["examples"]) < 5:
            rec["examples"].append({"scenario_spec": row.get("scenario_spec"), "task_id": row.get("task_id"), "source": row.get("source")})
        for call in calls:
            for k in call_params(call):
                tool_param_keys[call_name(call)][k] += 1
        analysis = row.get("analysis", "")
        if re.search(r"\bif\b|otherwise|else|是否|如果", analysis, re.I):
            branch_cards.append(
                {
                    "card_id": "v14::branch::" + hashlib.sha1((str(row.get("scenario_spec")) + str(row.get("task_id")) + analysis[:300]).encode()).hexdigest()[:12],
                    "scenario": scenario,
                    "task_type": task_type,
                    "condition_text": analysis[:1200],
                    "tool_names": list(names),
                    "source": row.get("source"),
                    "no_final_metadata": True,
                    "val41_only_not_final": bool(row.get("val41_only_not_final")),
                }
            )
        if row.get("val41_only_not_final"):
            val41_cards.append(
                {
                    "card_id": f"v14::val41_oracle::{row.get('scenario_spec')}::{row.get('task_id')}",
                    "scenario": scenario,
                    "scenario_spec": row.get("scenario_spec"),
                    "task_id": row.get("task_id"),
                    "tool_names": list(names),
                    "entity_slots": extract_slots(calls),
                    "analysis": analysis[:1500],
                    "val41_only_not_final": True,
                    "for_final_candidate": False,
                }
            )

    seq_cards = sorted(seq_counter.values(), key=lambda x: (-x["count"], x["scenario"], x["task_type"]))
    slot_rules = []
    for tool, counter in sorted(tool_param_keys.items()):
        slot_rules.append(
            {
                "card_id": "v14::slot_rule::" + tool,
                "tool_name": tool,
                "required_or_common_params": [k for k, _ in counter.most_common()],
                "entity_params": [k for k, _ in counter.most_common() if k in ENTITY_KEYS],
                "counted_from_gt": sum(counter.values()),
                "no_final_metadata": True,
            }
        )
    skeletons = [
        {"card_id": "v14::skeleton::query_only", "task_type": "query-only", "steps": ["resolve entity", "call minimal attribute tool", "answer concise"], "hard_policy": True},
        {"card_id": "v14::skeleton::visual_entity", "task_type": "visual-entity query", "steps": ["use visual/Qwen prior", "pin scenario context", "verify with one constrained retrieval", "call attribute tool"], "hard_policy": True},
        {"card_id": "v14::skeleton::branch_mutation", "task_type": "branch-then-mutation", "steps": ["evaluate branch condition with tool", "choose one branch", "canonicalize entity", "mutate once", "final aggregate if requested"], "hard_policy": True},
        {"card_id": "v14::skeleton::ranking", "task_type": "ranking/filtering", "steps": ["narrow candidates by category/location/restaurant", "query numeric attribute only for candidates", "choose min/max/ties", "mutate/answer"], "hard_policy": True},
        {"card_id": "v14::skeleton::aggregate", "task_type": "aggregate-required", "steps": ["inspect current state", "build aggregate item list from state", "call one aggregate tool", "answer"], "hard_policy": True},
    ]
    anti_broad = [
        {"card_id": "v14::anti_broad::retail_price_range", "scenario": "retail", "rule": "Do not use price_range 0-100000 for visual localization; narrow by category/origin/brand/name/taste first."},
        {"card_id": "v14::anti_broad::restaurant_all_category", "scenario": "restaurant", "rule": "Do not scan every category unless the skeleton explicitly requires category comparison."},
        {"card_id": "v14::anti_broad::kitchen_recipe_sweep", "scenario": "kitchen", "rule": "Do not scan all recipes/allergens; identify recipe or candidate ingredient then branch."},
        {"card_id": "v14::anti_broad::order_cross_restaurant", "scenario": "order", "rule": "Do not search across restaurants after restaurant is pinned."},
    ]
    order_pin = [
        {"card_id": "v14::order_pin::from_now_on", "scenario": "order", "rule": "When the user says from now on/use that restaurant, set pinned restaurant and use it for every later order tool."},
        {"card_id": "v14::order_pin::set_meal", "scenario": "order", "rule": "For set meal membership, call get_set_meal_details first and keep dish_name vs set_meal_name separate."},
        {"card_id": "v14::order_pin::aggregate", "scenario": "order", "rule": "After add/remove order mutation, use get_user_order_summary then compute_total_payment/tax/nutrition if requested."},
    ]
    shortcuts = [
        {"card_id": "v14::shortcut::single_attribute", "task_type": "query-only", "rule": "For a single known entity attribute query, use exactly the matching get_* tool and answer."},
        {"card_id": "v14::shortcut::no_mutation", "task_type": "query-only", "rule": "Do not mutate cart/order/menu for pure query tasks."},
    ]

    counts = {
        "tool_sequence_templates": write_jsonl(out_dir / "tool_sequence_templates.jsonl", seq_cards),
        "branch_condition_templates": write_jsonl(out_dir / "branch_condition_templates.jsonl", branch_cards),
        "entity_slot_mapping_rules": write_jsonl(out_dir / "entity_slot_mapping_rules.jsonl", slot_rules),
        "minimal_process_skeletons": write_jsonl(out_dir / "minimal_process_skeletons.jsonl", skeletons),
        "anti_broad_scan_rules": write_jsonl(out_dir / "anti_broad_scan_rules.jsonl", anti_broad),
        "order_restaurant_pin_rules": write_jsonl(out_dir / "order_restaurant_pin_rules.jsonl", order_pin),
        "query_only_shortcuts": write_jsonl(out_dir / "query_only_shortcuts.jsonl", shortcuts),
        "val41_oracle_debug_cards": write_jsonl(out_dir / "val41_oracle_debug_cards.jsonl", val41_cards),
    }
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_rows": len(dev_rows),
        "counts": counts,
        "uses_final_hidden_metadata": False,
        "val41_oracle_cards_for_debug_only": True,
    }
    write_json(out_dir / "memory_bank_manifest.json", manifest)
    return manifest


def write_audit_report(run_id: str, rows: List[Dict[str, Any]], mismatch_rows: List[Dict[str, Any]], bank_manifest: Dict[str, Any]) -> Path:
    counts = collections.Counter()
    scenario_counts = collections.Counter()
    for r in mismatch_rows:
        scenario_counts[r["spec"]] += 1
        for t in r.get("mismatch_types", []):
            counts[t] += 1
    report = CODEX / "reports" / f"V14_VAL41_GT_TRAJECTORY_AUDIT_{run_id}.md"
    lines = [
        f"# V14 Val41 GT Trajectory Audit {run_id}",
        "",
        "- final_run: false",
        "- uses_val41_gt: true, for audit/oracle debugging only",
        "- uses_final_hidden_metadata: false",
        f"- frozen_val41_tasks: {len(rows)}",
        f"- mismatch_records: {len(mismatch_rows)}",
        "",
        "## Mismatch Type Counts",
        "",
    ]
    for k, v in counts.most_common():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Per Spec Mismatch Records", ""]
    for k, v in sorted(scenario_counts.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Process Bank", ""]
    for k, v in bank_manifest.get("counts", {}).items():
        lines.append(f"- {k}: {v}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_oracle_report(run_id: str, eval_result: Dict[str, Any], model_dir: Path) -> Path:
    report = CODEX / "reports" / f"V14_ORACLE_TEACHER_VAL41_{run_id}.md"
    s = eval_result["summary"]
    lines = [
        f"# V14 Oracle Teacher Val41 {run_id}",
        "",
        "- tier: A",
        "- uses_val41_gt: true",
        "- final_run: false",
        "- for_final_candidate: false",
        f"- result_dir: `{model_dir}`",
        "",
        "## Summary",
        "",
        f"- valid: {s.get('valid', 0)}",
        f"- joint: {s.get('joint', 0):.4f}",
        f"- result: {s.get('result', 0):.4f}",
        f"- tool: {s.get('tool', 0):.4f}",
        f"- micro: {s.get('micro', 0):.4f}",
        f"- avg_task_accuracy: {s.get('avg_task_accuracy', 0):.4f}",
        f"- tool_call_match_counts: {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} gt, interaction_calls={s.get('interaction_calls', 0)}",
        "",
        "## Per File",
        "",
        "| spec | valid | joint | result | tool | micro | calls | error |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in eval_result.get("rows", []):
        lines.append(f"| {row.get('spec')} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('error', '')} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v14_gt_distill_{now_stamp()}")
    args = ap.parse_args()
    run_id = args.run_id
    version = "V14_GT_TRAJECTORY_DISTILLATION_VAL41"
    out_dir = CODEX / "runs" / version / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    val_rows = load_val41()
    mismatch_records = []
    for row in val_rows:
        gt_calls = row["gt"].get("ground_truth") or []
        pred_calls = flatten_calls_from_result(row.get("pred"))
        mismatch_types = classify_mismatch(row, gt_calls, pred_calls)
        record = {
            "run_id": run_id,
            "spec": row["spec"],
            "scenario": row["scenario"],
            "number": row["number"],
            "subset_index": row["subset_index"],
            "original_index": row["original_index"],
            "task_id": row["task_id"],
            "gt_tool_names": [call_name(c) for c in gt_calls],
            "pred_tool_names": [call_name(c) for c in pred_calls],
            "gt_required_entity_slots": extract_slots(gt_calls),
            "pred_entity_slots": extract_slots(pred_calls),
            "gt_branch_condition": row["gt"].get("analysis", "")[:1200],
            "gt_final_mutation_or_aggregate": [call_name(c) for c in gt_calls if is_mutation(call_name(c)) or call_name(c) in AGGREGATE_TOOLS],
            "mismatch_types": mismatch_types,
            "uses_val41_gt": True,
            "final_run": False,
        }
        mismatch_records.append(record)

    mismatch_path = CODEX / "analysis" / "v14_val41_gt_vs_pred_mismatch.jsonl"
    write_jsonl(mismatch_path, mismatch_records)

    oracle_model_dir = EGO / "results" / f"V14_val41_oracle_teacher-{run_id}"
    oracle_model_dir.mkdir(parents=True, exist_ok=True)
    by_spec: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in val_rows:
        by_spec[row["spec"]].append(build_oracle_result(row))
    for spec, items in by_spec.items():
        write_json(oracle_model_dir / f"{spec}_easy.json", items)

    oracle_eval = evaluate_model_dir(oracle_model_dir, run_id, "V14_val41_oracle_teacher")
    oracle_report = write_oracle_report(run_id, oracle_eval, oracle_model_dir)

    dev_rows = collect_dev_gt_rows()
    bank_dir = CODEX / "memory_bank_v14_gt_trajectory"
    bank_manifest = build_process_bank(dev_rows, bank_dir)
    audit_report = write_audit_report(run_id, val_rows, mismatch_records, bank_manifest)

    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "version": version,
        "val41_tasks": len(val_rows),
        "oracle_result_dir": str(oracle_model_dir),
        "oracle_summary": oracle_eval["summary"],
        "mismatch_jsonl": str(mismatch_path),
        "memory_bank": str(bank_dir),
        "memory_manifest": bank_manifest,
        "reports": {
            "audit": str(audit_report),
            "oracle": str(oracle_report),
        },
        "final_run": False,
        "v10_zip_overwritten": False,
        "uses_val41_gt": True,
        "uses_final_hidden_metadata": False,
    }
    write_json(CODEX / "state" / "latest_v14_gt_distillation.json", state)
    write_json(out_dir / "state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
