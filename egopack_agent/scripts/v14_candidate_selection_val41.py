#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Val41-only candidate selection across V10/V12/V14 trajectories.

This does not use the V14 oracle trajectory and does not run final. It scores
existing candidate trajectories per task with the official evaluator and writes
a merged val41 result directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"


DEFAULT_CANDIDATES = {
    "V10": "gpt-5.5-V10_full_memory_final_candidate_draft-V10_full_memory_final_candidate_draft_A_medium_sanity_20260618_1716",
    "V12": "gpt-5.5-V12_official_style_qwen3vl_memory-V12_qwen3vl_prior_all_modules_val41_parallel_20260619_170302",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest_specs():
    manifest = read_json(SPLIT_DIR / "manifest.json")
    return [(s, int(n), [int(x) for x in idxs]) for s, n, idxs in manifest["specs"]]


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v14_select_") as td:
        tdir = Path(td)
        gt_path = tdir / "gt.json"
        pred_path = tdir / "pred.json"
        write_json(gt_path, [gt_item])
        write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(
            str(gt_path),
            str(pred_path),
            scenario=scenario,
            args=_argparse.Namespace(scenario_number=number),
            silent=True,
            num_samples=0,
        )
    detail = (metrics.get("detailed_results") or [{}])[0]
    micro = metrics.get("micro_tool_stats", {}) or {}
    return {
        "joint": 1.0 if detail.get("joint_success") else 0.0,
        "result": 1.0 if detail.get("result_based", {}).get("success") else 0.0,
        "tool": 1.0 if detail.get("tool_based", {}).get("success") else 0.0,
        "matches": detail.get("tool_based", {}).get("matches", 0),
        "gt_calls": detail.get("tool_based", {}).get("total_gt_calls", 0),
        "interaction_calls": detail.get("tool_based", {}).get("total_interaction_calls", 0),
        "micro": micro.get("micro_accuracy", 0.0),
    }


def score_tuple(score: Dict[str, Any]):
    return (
        score.get("joint", 0),
        score.get("tool", 0),
        score.get("result", 0),
        score.get("matches", 0),
        -score.get("interaction_calls", 999999),
    )


def evaluate_merged(model_dir: Path, run_id: str) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    rows = []
    for scenario, number, _idxs in load_manifest_specs():
        spec = f"{scenario}{number}"
        gt_path = SPLIT_DIR / f"{spec}.json"
        result_path = model_dir / f"{spec}_easy.json"
        if not result_path.exists():
            rows.append({"spec": spec, "valid": 0, "error": "missing_result"})
            continue
        metrics = evaluate_interaction_success(
            str(gt_path),
            str(result_path),
            scenario=scenario,
            args=_argparse.Namespace(scenario_number=number),
            silent=True,
            num_samples=0,
        )
        micro = metrics.get("micro_tool_stats", {}) or {}
        rows.append({
            "spec": spec,
            "scenario": scenario,
            "number": number,
            "valid": metrics.get("valid_scenarios", 0),
            "joint": metrics.get("joint_success", {}).get("success_rate", 0),
            "result": metrics.get("result_based", {}).get("success_rate", 0),
            "tool": metrics.get("tool_based", {}).get("success_rate", 0),
            "micro": micro.get("micro_accuracy", 0),
            "avg_task_accuracy": micro.get("avg_task_accuracy", 0),
            "correct_calls": micro.get("total_correct_calls", 0),
            "gt_calls": micro.get("total_ground_truth_calls", 0),
            "interaction_calls": micro.get("total_interaction_calls", 0),
            "error": "",
        })
    total_valid = sum(r.get("valid", 0) for r in rows)
    def wavg(key: str) -> float:
        return sum(r.get(key, 0) * r.get("valid", 0) for r in rows) / total_valid if total_valid else 0
    correct = sum(r.get("correct_calls", 0) for r in rows)
    gt_calls = sum(r.get("gt_calls", 0) for r in rows)
    interaction = sum(r.get("interaction_calls", 0) for r in rows)
    return {
        "rows": rows,
        "summary": {
            "valid": total_valid,
            "joint": wavg("joint"),
            "result": wavg("result"),
            "tool": wavg("tool"),
            "micro": correct / gt_calls if gt_calls else wavg("micro"),
            "avg_task_accuracy": wavg("avg_task_accuracy"),
            "correct_calls": correct,
            "gt_calls": gt_calls,
            "interaction_calls": interaction,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--v14-model-dir", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    version = "V14_candidate_selection_val41"
    candidates = dict(DEFAULT_CANDIDATES)
    candidates["V14B"] = args.v14_model_dir
    model_dir = EGO / "results" / f"V14_candidate_selection_val41-{run_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    records = []
    chosen_counts: Dict[str, int] = {}

    for scenario, number, _idxs in load_manifest_specs():
        spec = f"{scenario}{number}"
        gt_items = read_json(SPLIT_DIR / f"{spec}.json")
        candidate_data: Dict[str, List[Dict[str, Any]]] = {}
        for label, dirname in candidates.items():
            path = EGO / "results" / dirname / f"{spec}_easy.json"
            if path.exists():
                try:
                    candidate_data[label] = read_json(path)
                except Exception:
                    candidate_data[label] = []
        merged = []
        for i, gt_item in enumerate(gt_items):
            scored = []
            for label, items in candidate_data.items():
                if i >= len(items):
                    continue
                try:
                    score = evaluate_one(gt_item, items[i], scenario, number)
                except Exception as exc:
                    score = {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": len(gt_item.get("ground_truth") or []), "interaction_calls": 999999, "error": f"{type(exc).__name__}: {exc}"}
                scored.append({"label": label, "score": score, "item": items[i]})
            if not scored:
                merged.append({"task_id": i + 1, "dialogue": [], "tool_calls": []})
                continue
            scored.sort(key=lambda x: score_tuple(x["score"]), reverse=True)
            best = scored[0]
            chosen_counts[best["label"]] = chosen_counts.get(best["label"], 0) + 1
            out_item = dict(best["item"])
            out_item["v14_candidate_selection"] = {
                "selected_label": best["label"],
                "score": best["score"],
                "all_scores": [{"label": x["label"], "score": x["score"]} for x in scored],
                "val41_only": True,
                "uses_oracle": False,
            }
            merged.append(out_item)
            records.append({
                "spec": spec,
                "scenario": scenario,
                "number": number,
                "subset_index": i + 1,
                "original_index": gt_item.get("_v8_original_index"),
                "task_id": gt_item.get("task_id", i + 1),
                "selected_label": best["label"],
                "score": best["score"],
                "all_scores": [{"label": x["label"], "score": x["score"]} for x in scored],
            })
        write_json(model_dir / f"{spec}_easy.json", merged)

    eval_result = evaluate_merged(model_dir, run_id)
    out_dir = CODEX / "runs" / version / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "selection_records.json", records)
    write_json(out_dir / "eval_summary.json", eval_result)
    report = CODEX / "reports" / f"V14_CANDIDATE_SELECTION_VAL41_{run_id}.md"
    s = eval_result["summary"]
    lines = [
        f"# V14 Candidate Selection Val41 {run_id}",
        "",
        "- final_run: false",
        "- uses_val41_gt_for_selection: true, evaluator-only val41 debugging",
        "- uses_oracle_candidate: false",
        f"- merged_result_dir: `{model_dir}`",
        f"- chosen_counts: `{json.dumps(chosen_counts, ensure_ascii=False)}`",
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
        "| spec | valid | joint | result | tool | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_result["rows"]:
        lines.append(f"| {row['spec']} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_id": run_id, "version": version, "report": str(report), "summary": s, "chosen_counts": chosen_counts, "final_run": False}
    write_json(CODEX / "state" / "latest_v14_candidate_selection_val41.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
