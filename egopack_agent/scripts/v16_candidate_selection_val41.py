#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-oracle candidate selection for V16 val41 runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for entry in result.get("tool_calls", []) or []:
        if isinstance(entry, dict) and isinstance(entry.get("calls"), list):
            out.extend([c for c in entry["calls"] if isinstance(c, dict)])
    return out


def score_non_oracle(result: Dict[str, Any], instruction: str, scenario: str, source_name: str) -> Dict[str, Any]:
    calls = flatten_tool_calls(result)
    names = [str(c.get("tool_name") or "") for c in calls]
    text = (instruction or "").lower()
    score = 0.0
    if calls:
        score += 5
    if source_name.lower().startswith("v16"):
        score += 1.0
    if source_name.lower().startswith("v14"):
        score += 0.4
    mutation = any(re.search(r"^(add|remove|delete|update|modify)_|_(to|from)_", n) for n in names)
    aggregate = any(n.startswith("compute_total_") or n.startswith("tally_total_") or "summary" in n for n in names)
    if any(x in text for x in ["add", "remove", "cart", "order", "menu", "shopping list"]) and mutation:
        score += 7
    if any(x in text for x in ["total", "tax", "payment", "nutrition", "summary", "taste", "calorie"]) and aggregate:
        score += 7
    if scenario == "order":
        if any("restaurant_name" in (c.get("parameters") or {}) for c in calls):
            score += 5
        if any("set_meal" in n for n in names):
            score += 1
    if scenario == "retail":
        if len(calls) > 60:
            score -= 12
        elif len(calls) <= 25:
            score += 2
    if len(calls) > 40:
        score -= (len(calls) - 40) * 0.25
    if not calls and any(x in text for x in ["add", "remove", "total", "tax", "payment", "nutrition"]):
        score -= 12
    dialogue = json.dumps(result.get("dialogue", []), ensure_ascii=False).lower()
    if "?" in dialogue and any(x in dialogue for x in ["which", "what", "visible", "point", "menu", "product", "dish"]):
        score -= 9
    seen = set()
    duplicates = 0
    for call in calls:
        sig = json.dumps(call, ensure_ascii=False, sort_keys=True)
        if sig in seen:
            duplicates += 1
        seen.add(sig)
    score -= duplicates * 4
    return {"score": round(score, 3), "tool_count": len(calls), "tool_names": names[:20], "duplicates": duplicates}


def evaluate_dir(model_dir: Path, run_id: str) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    manifest = read_json(SPLIT_DIR / "manifest.json")
    rows = []
    for scenario, num, _idxs in manifest.get("specs", []):
        spec = f"{scenario}{num}"
        gt = SPLIT_DIR / f"{spec}.json"
        result = model_dir / f"{spec}_easy.json"
        if not result.exists():
            rows.append({"spec": spec, "valid": 0, "error": "missing_result"})
            continue
        try:
            metrics = evaluate_interaction_success(str(gt), str(result), scenario=scenario, args=_argparse.Namespace(scenario_number=int(num)), silent=True, num_samples=0)
            micro = metrics.get("micro_tool_stats", {}) or {}
            rows.append({
                "spec": spec,
                "scenario": scenario,
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
        except Exception as exc:
            rows.append({"spec": spec, "valid": 0, "error": f"{type(exc).__name__}: {exc}"})
    total_valid = sum(r.get("valid", 0) for r in rows)
    def wavg(key: str) -> float:
        return sum(r.get(key, 0) * r.get("valid", 0) for r in rows) / total_valid if total_valid else 0
    correct = sum(r.get("correct_calls", 0) for r in rows)
    gt_calls = sum(r.get("gt_calls", 0) for r in rows)
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
            "interaction_calls": sum(r.get("interaction_calls", 0) for r in rows),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v16_candidate_selection_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--candidate", action="append", default=[], help="name=path")
    args = ap.parse_args()

    candidates = {}
    for raw in args.candidate:
        if "=" not in raw:
            continue
        name, path = raw.split("=", 1)
        p = Path(path)
        if p.exists():
            candidates[name] = p
    if not candidates:
        raise SystemExit("no candidate result dirs provided")

    out_model = f"V16_candidate_selection_val41-{args.run_id}"
    out_dir = EGO / "results" / out_model
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(SPLIT_DIR / "manifest.json")
    decisions = []
    for scenario, num, _idxs in manifest.get("specs", []):
        spec = f"{scenario}{num}"
        gt_items = read_json(SPLIT_DIR / f"{spec}.json")
        if isinstance(gt_items, dict):
            gt_items = list(gt_items.values())
        per_source = {}
        for name, path in candidates.items():
            f = path / f"{spec}_easy.json"
            if f.exists():
                per_source[name] = read_json(f)
        selected_items = []
        for i, gt_item in enumerate(gt_items):
            scored = []
            for name, data in per_source.items():
                if i < len(data) and isinstance(data[i], dict):
                    score = score_non_oracle(data[i], gt_item.get("Instruction", ""), scenario, name)
                    scored.append((score["score"], name, data[i], score))
            if not scored:
                selected_items.append({"task_id": i + 1, "dialogue": [], "tool_calls": [], "v16_selection_error": "no_candidate"})
                continue
            scored.sort(key=lambda x: (x[0], 1 if x[1].lower().startswith("v16") else 0), reverse=True)
            best_score, best_name, best_item, score_detail = scored[0]
            chosen = dict(best_item)
            chosen["v16_candidate_selection"] = {"selected_from": best_name, "score": score_detail, "uses_val41_gt_for_selection": False}
            selected_items.append(chosen)
            decisions.append({"spec": spec, "task_index": i + 1, "selected_from": best_name, "score": score_detail, "all_scores": [{"name": n, "score": d} for _s, n, _it, d in scored]})
        write_json(out_dir / f"{spec}_easy.json", selected_items)

    eval_result = evaluate_dir(out_dir, args.run_id)
    run_dir = CODEX / "runs" / "V16_candidate_selection_val41" / args.run_id
    write_json(run_dir / "decisions.json", decisions)
    write_json(run_dir / "eval_summary.json", eval_result)
    report = CODEX / "reports" / f"V16_FAILURE_DIFF_V14_TO_V16_{args.run_id}.md"
    s = eval_result["summary"]
    by_source = {}
    for d in decisions:
        by_source[d["selected_from"]] = by_source.get(d["selected_from"], 0) + 1
    lines = [
        f"# V16 Candidate Selection / Failure Diff {args.run_id}",
        "",
        "- final_run: false",
        "- uses_val41_gt_for_selection: false",
        "- selection_signal: heuristic process coverage, slot closure, tool count, anti-ask/broad-scan penalties",
        "",
        "## Selected Summary",
        "",
        f"- joint: {s.get('joint', 0):.4f}",
        f"- result: {s.get('result', 0):.4f}",
        f"- tool: {s.get('tool', 0):.4f}",
        f"- micro: {s.get('micro', 0):.4f}",
        f"- calls: {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} gt, interaction_calls={s.get('interaction_calls', 0)}",
        "",
        "## Candidate Source Counts",
    ]
    for name, count in sorted(by_source.items()):
        lines.append(f"- {name}: {count}")
    lines += ["", "## Per Spec", "", "| spec | valid | joint | result | tool | micro | calls | error |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in eval_result["rows"]:
        lines.append(f"| {row.get('spec')} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('error', '')} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = {"run_id": args.run_id, "result_dir": str(out_dir), "summary": s, "report": str(report), "uses_val41_gt_for_selection": False, "final_run": False}
    write_json(CODEX / "state" / "latest_v16_candidate_selection.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
