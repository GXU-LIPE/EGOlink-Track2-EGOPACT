#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit EgoBench Track2 scenario files and create stable V8 validation splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

FINAL_FILES = {"retail6.json", "retail10.json", "kitchen4.json", "restaurant5.json", "order2.json"}
SCENARIOS = ["retail", "kitchen", "restaurant", "order"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_name(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"([a-z]+)(\d+)$", stem)
    return m.group(1) if m else stem


def scenario_number(name: str) -> int:
    stem = Path(name).stem
    m = re.match(r"([a-z]+)(\d+)$", stem)
    return int(m.group(2)) if m else 0


def task_family(task: Dict[str, Any]) -> str:
    text = " ".join(str(task.get(k, "")) for k in ["Instruction", "analysis", "image_description"]).lower()
    if any(w in text for w in ["replace", "swap", "change", "remove", "cancel"]):
        return "replace_remove_change"
    if any(w in text for w in ["add", "buy", "order", "put"]):
        return "add_order"
    if any(w in text for w in ["nutrition", "calorie", "protein", "fat", "carb"]):
        return "nutrition"
    if any(w in text for w in ["recipe", "ingredient", "shopping list", "menu"]):
        return "recipe_inventory"
    if any(w in text for w in ["tax", "payment", "total", "price"]):
        return "aggregate_payment"
    return "other"


def has_key_any(tasks: List[Dict[str, Any]], keys: Iterable[str]) -> bool:
    keys = set(keys)
    return any(any(k in t and t.get(k) not in (None, "", []) for k in keys) for t in tasks if isinstance(t, dict))


def item_has_gt(t: Dict[str, Any]) -> bool:
    return any(k in t and t.get(k) not in (None, "", []) for k in ["ground_truth", "Ground_truth", "gt_tool_calls", "answer"])


def make_splits(records: List[Dict[str, Any]], seed: int = 20260617) -> Dict[str, Any]:
    rng = random.Random(seed)
    measurable = [r for r in records if r["is_allowed_for_eval"]]
    by_scenario = defaultdict(list)
    for r in measurable:
        by_scenario[r["scenario_name"]].append(r)
    split = {"seed": seed, "validation_A": [], "validation_B_holdout": [], "train_rule_mining": []}
    for scen in sorted(by_scenario):
        group = by_scenario[scen]
        by_family = defaultdict(list)
        for r in group:
            by_family[r["task_family"]].append(r)
        val_a, val_b = [], []
        for fam, fam_items in sorted(by_family.items()):
            items = fam_items[:]
            rng.shuffle(items)
            n = len(items)
            # Minimum one per family when possible; around 5% each overall.
            take = 1 if n >= 3 else (1 if n >= 2 and scen in {"order", "kitchen"} else 0)
            take = max(take, round(n * 0.05))
            take = min(take, max(0, n // 3)) if n > 3 else take
            if take:
                val_a.extend(items[:take])
                val_b.extend(items[take:take * 2])
        # Scenario-level guardrails: enough order/kitchen samples for validation.
        min_target = 8 if scen in {"order", "kitchen"} else 5
        remain = [x for x in group if x not in val_a and x not in val_b]
        rng.shuffle(remain)
        while len(val_a) < min(min_target, max(1, len(group)//4)) and remain:
            val_a.append(remain.pop())
        while len(val_b) < min(min_target, max(1, len(group)//4)) and remain:
            val_b.append(remain.pop())
        train = [x for x in group if x not in val_a and x not in val_b]
        split["validation_A"].extend([x["task_uid"] for x in sorted(val_a, key=lambda r: r["task_uid"])])
        split["validation_B_holdout"].extend([x["task_uid"] for x in sorted(val_b, key=lambda r: r["task_uid"])])
        split["train_rule_mining"].extend([x["task_uid"] for x in sorted(train, key=lambda r: r["task_uid"])])
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego-root", default="/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
    ap.add_argument("--codex-root", default="/home/data-gxu/acm/egolink2026-main/code/track2/codex")
    ap.add_argument("--seed", type=int, default=20260617)
    args = ap.parse_args()
    ego = Path(args.ego_root)
    codex = Path(args.codex_root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for d in ["reports", "analysis", "state"]:
        (codex / d).mkdir(parents=True, exist_ok=True)

    scenario_dir = ego / "scenarios"
    files = sorted(scenario_dir.rglob("*.json"))
    file_rows: List[Dict[str, Any]] = []
    task_records: List[Dict[str, Any]] = []
    for p in files:
        rel = p.relative_to(scenario_dir)
        try:
            data = load_json(p)
        except Exception as e:
            file_rows.append({"file_path": str(p), "scenario_name": p.stem, "task_count": 0, "error": str(e)})
            continue
        tasks = data if isinstance(data, list) else []
        scen = scenario_name(p.name)
        num = scenario_number(p.name)
        is_official_final = p.name in FINAL_FILES
        any_gt = any(isinstance(t, dict) and item_has_gt(t) for t in tasks)
        has_img_desc = has_key_any(tasks, ["image_description"])
        has_analysis = has_key_any(tasks, ["analysis", "Analysis", "task_analysis"])
        has_video = has_key_any(tasks, ["image_path", "video_path", "image_name", "video"])
        split_type = "final" if is_official_final else "dev/offline"
        row = {
            "file_path": str(p),
            "relative_path": str(rel),
            "scenario_file": p.name,
            "scenario_name": scen,
            "scenario_number": num,
            "task_count": len(tasks),
            "split_type": split_type,
            "has_ground_truth": any_gt,
            "has_image_description": has_img_desc,
            "has_analysis": has_analysis,
            "has_video_path": has_video,
            "has_final_hidden_metadata": bool(is_official_final and (has_img_desc or has_analysis)),
            "is_allowed_for_training_or_rule_mining": bool((not is_official_final) and any_gt),
            "is_allowed_for_eval": bool((not is_official_final) and any_gt),
            "is_final_no_gt": bool(is_official_final and not any_gt),
            "error": "",
        }
        file_rows.append(row)
        for idx, t in enumerate(tasks, start=1):
            if not isinstance(t, dict):
                continue
            has_gt = item_has_gt(t)
            uid = f"{p.stem}::{idx}"
            task_records.append({
                "task_uid": uid,
                "scenario_file": p.name,
                "scenario_name": scen,
                "scenario_number": num,
                "task_index": idx,
                "task_id": t.get("task_id", idx),
                "split_type": split_type,
                "has_ground_truth": has_gt,
                "is_allowed_for_eval": bool((not is_official_final) and has_gt),
                "is_allowed_for_training_or_rule_mining": bool((not is_official_final) and has_gt),
                "is_final_no_gt": bool(is_official_final and not has_gt),
                "has_image_description": bool(t.get("image_description")),
                "has_analysis": bool(t.get("analysis") or t.get("Analysis") or t.get("task_analysis")),
                "has_video_path": bool(t.get("image_path") or t.get("video_path") or t.get("image_name")),
                "task_family": task_family(t),
                "instruction_hash": hashlib.sha256(str(t.get("Instruction", "")).encode("utf-8")).hexdigest()[:12],
            })

    split = make_splits(task_records, seed=args.seed)
    uid_to_record = {r["task_uid"]: r for r in task_records}
    for name, uids in split.items():
        if isinstance(uids, list):
            for uid in uids:
                uid_to_record[uid]["v8_split"] = name
    for r in task_records:
        r.setdefault("v8_split", "final_sanity_only" if r["is_final_no_gt"] else "not_measurable")

    csv_path = codex / "analysis" / f"dataset_task_counts_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_path", "relative_path", "scenario_file", "scenario_name", "scenario_number", "task_count", "split_type", "has_ground_truth", "has_image_description", "has_analysis", "has_video_path", "has_final_hidden_metadata", "is_allowed_for_training_or_rule_mining", "is_allowed_for_eval", "is_final_no_gt", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(file_rows)
    task_csv = codex / "analysis" / f"dataset_task_splits_{ts}.csv"
    with task_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["task_uid", "scenario_file", "scenario_name", "scenario_number", "task_index", "task_id", "split_type", "v8_split", "task_family", "has_ground_truth", "is_allowed_for_eval", "is_allowed_for_training_or_rule_mining", "is_final_no_gt", "has_image_description", "has_analysis", "has_video_path", "instruction_hash"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(sorted(task_records, key=lambda r: (r["scenario_name"], r["scenario_number"], r["task_index"])))

    final_rows = [r for r in file_rows if r.get("split_type") == "final"]
    dev_rows = [r for r in file_rows if r.get("split_type") != "final"]
    final_total = sum(int(r["task_count"]) for r in final_rows)
    dev_eval_total = sum(1 for r in task_records if r["is_allowed_for_eval"])
    per_scenario = defaultdict(lambda: {"dev_eval": 0, "final": 0, "validation_A": 0, "validation_B_holdout": 0, "train_rule_mining": 0})
    for r in task_records:
        scen = r["scenario_name"]
        if r["is_allowed_for_eval"]:
            per_scenario[scen]["dev_eval"] += 1
        if r["is_final_no_gt"]:
            per_scenario[scen]["final"] += 1
        if r.get("v8_split") in per_scenario[scen]:
            per_scenario[scen][r["v8_split"]] += 1

    state = {
        "generated_at": ts,
        "ego_root": str(ego),
        "scenario_dir": str(scenario_dir),
        "final_files_official": sorted(FINAL_FILES),
        "final_total_tasks": final_total,
        "final_expected_309": final_total == 309,
        "final_has_ground_truth_any": any(r["has_ground_truth"] for r in final_rows),
        "dev_offline_files": len(dev_rows),
        "dev_offline_measurable_tasks": dev_eval_total,
        "split_counts": {k: len(v) for k, v in split.items() if isinstance(v, list)},
        "per_scenario": dict(per_scenario),
        "csv": str(csv_path),
        "task_split_csv": str(task_csv),
    }
    state_path = codex / "state" / f"track2_data_split_{ts}.json"
    state_path.write_text(json.dumps({**state, "splits": split}, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = codex / "state" / "track2_data_split_latest.json"
    latest_path.write_text(json.dumps({**state, "splits": split}, ensure_ascii=False, indent=2), encoding="utf-8")

    report = codex / "reports" / f"V8_DATASET_COUNT_AUDIT_{ts}.md"
    lines = [
        f"# V8 Dataset Count Audit {ts}", "",
        f"- EgoBench root: `{ego}`",
        f"- Scenario JSON files: {len(file_rows)}",
        f"- Official final task total: {final_total} (expected 309: {final_total == 309})",
        f"- Official final has any ground_truth: {state['final_has_ground_truth_any']}",
        f"- Dev/offline measurable tasks with GT: {dev_eval_total}",
        f"- File count CSV: `{csv_path}`",
        f"- Task split CSV: `{task_csv}`",
        f"- Split state: `{state_path}`",
        "", "## Per Scenario", "",
        "| scenario | dev/offline GT tasks | validation_A | validation_B | train_rule_mining | final no-GT tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scen in SCENARIOS:
        d = per_scenario[scen]
        lines.append(f"| {scen} | {d['dev_eval']} | {d['validation_A']} | {d['validation_B_holdout']} | {d['train_rule_mining']} | {d['final']} |")
    lines += ["", "## File Rows", "", "| file | split | tasks | GT | image_description | analysis | video_path | allowed_eval | final_hidden_metadata |", "|---|---|---:|---|---|---|---|---|---|"]
    for r in sorted(file_rows, key=lambda x: x.get("relative_path", "")):
        lines.append(f"| {r['relative_path']} | {r['split_type']} | {r['task_count']} | {r['has_ground_truth']} | {r['has_image_description']} | {r['has_analysis']} | {r['has_video_path']} | {r['is_allowed_for_eval']} | {r['has_final_hidden_metadata']} |")
    lines += ["", "## Policy", "", "- Final files are excluded from training, rule mining, validation_A, and validation_B.", "- Validation_B is a holdout and must not be used for prompt/rule tuning.", "- Future long validation must consume `state/track2_data_split_latest.json`.", "- No API key was read or logged."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (codex / "README_STATUS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## V8 Dataset Count Audit {ts}\n\n- Report: `{report}`\n- Final total: {final_total}/309, final GT present: {state['final_has_ground_truth_any']}\n- Dev/offline measurable tasks: {dev_eval_total}\n- Split state: `{state_path}`\n- No final submission was made.\n")
    print(json.dumps({"report": str(report), "state": str(state_path), "csv": str(csv_path), "task_csv": str(task_csv), "summary": state}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
