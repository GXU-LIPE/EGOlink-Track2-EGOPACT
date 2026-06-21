#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect V7 human-prior ablation runs and update best only under strict rules."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
TASKS = ["retail9", "restaurant4", "order1", "kitchen2"]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fval(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def metric(data: Dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def eval_metrics(output_model: str, task: str) -> Dict[str, Any]:
    path = EGO_ROOT / "eval_result" / output_model / f"{task}_easy_eval.json"
    data = load_json(path)
    if not isinstance(data, dict):
        return {"missing": True, "eval_path": str(path), "joint_success": 0.0, "result_success": 0.0, "tool_success": 0.0, "micro_tool_accuracy": 0.0, "avg_rounds": 0.0, "avg_tool_calls": 0.0, "tool_matches": 0.0, "gt_tool_calls": 0.0, "interaction_tool_calls": 0.0}
    detailed = data.get("detailed_results") or []
    tb = detailed[0].get("tool_based", {}) if detailed and isinstance(detailed[0], dict) else {}
    return {
        "missing": False,
        "eval_path": str(path),
        "joint_success": fval(metric(data, "joint_success.success_rate")),
        "result_success": fval(metric(data, "result_based.success_rate")),
        "tool_success": fval(metric(data, "tool_based.success_rate")),
        "micro_tool_accuracy": fval(metric(data, "micro_tool_stats.micro_accuracy")),
        "avg_rounds": fval(metric(data, "performance_metrics.avg_rounds_count")),
        "avg_tool_calls": fval(metric(data, "performance_metrics.avg_tool_calls_count")),
        "tool_matches": fval(tb.get("matches")),
        "gt_tool_calls": fval(tb.get("total_gt_calls")),
        "interaction_tool_calls": fval(tb.get("total_interaction_calls")),
    }


def mean(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(fval(r.get(key)) for r in rows) / max(1, len(rows))


def count_events(version: str, run_id: str) -> Dict[str, int]:
    root = CODEX_ROOT / "runs" / version / run_id
    counts = {"human_prior_events": 0, "policy_traces_total": 0, "counterfactual_decisions": 0, "process_verifier_events": 0, "visual_slot_events": 0, "duplicate_mutation_blocks": 0}
    for path in (root / "human_prior_events").glob("*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                counts["human_prior_events"] += 1
                rec = json.loads(line)
                if rec.get("counterfactual_decision"):
                    counts["counterfactual_decisions"] += 1
                if rec.get("verifier_decision"):
                    counts["process_verifier_events"] += 1
                if rec.get("visual_slots"):
                    counts["visual_slot_events"] += 1
        except Exception:
            pass
    for path in (root / "wrapper_events").glob("*.jsonl"):
        try:
            counts["duplicate_mutation_blocks"] += path.read_text(encoding="utf-8").count("duplicate_mutation_blocked")
        except Exception:
            pass
    trace_path = CODEX_ROOT / "train_data" / "human_prior_policy_traces.jsonl"
    if trace_path.exists():
        try:
            counts["policy_traces_total"] = sum(1 for _ in trace_path.open("r", encoding="utf-8"))
        except Exception:
            pass
    return counts


def summarize_run(version: str, run_id: str, model: str) -> Dict[str, Any]:
    output_model = f"{model}-{version}-{run_id}"
    rows = []
    for task in TASKS:
        row = {"version": version, "run_id": run_id, "output_model": output_model, "task": task}
        row.update(eval_metrics(output_model, task))
        rows.append(row)
    summary = {
        "version": version,
        "run_id": run_id,
        "model": model,
        "output_model": output_model,
        "joint_success": mean(rows, "joint_success"),
        "result_success": mean(rows, "result_success"),
        "tool_success": mean(rows, "tool_success"),
        "micro_tool_accuracy": mean(rows, "micro_tool_accuracy"),
        "avg_rounds": mean(rows, "avg_rounds"),
        "avg_tool_calls": mean(rows, "avg_tool_calls"),
    }
    summary.update(count_events(version, run_id))
    return {"summary": summary, "rows": rows}


def should_update_best(candidate: Dict[str, Any], rows: List[Dict[str, Any]], best: Dict[str, Any]) -> bool:
    best_joint = fval(best.get("joint_success"))
    cand_joint = fval(candidate.get("joint_success"))
    cand_micro = fval(candidate.get("micro_tool_accuracy"))
    cand_result = fval(candidate.get("result_success"))
    if cand_joint > best_joint:
        return True
    retail_ok = next((r for r in rows if r["task"] == "retail9"), {}).get("joint_success", 0) >= 1.0
    restaurant_ok = next((r for r in rows if r["task"] == "restaurant4"), {}).get("joint_success", 0) >= 1.0
    order_micro = next((r for r in rows if r["task"] == "order1"), {}).get("micro_tool_accuracy", 0)
    if abs(cand_joint - best_joint) < 1e-9 and cand_micro > 0.7083 and cand_result >= 0.75:
        return True
    if abs(cand_joint - best_joint) < 1e-9 and order_micro > 0.3334 and retail_ok and restaurant_ok:
        return True
    return False


def write_reports(all_rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]], updated: bool, best_candidate: Dict[str, Any] | None, stamp: str) -> None:
    analysis = CODEX_ROOT / "analysis" / f"human_prior_ablation_{stamp}.csv"
    analysis.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in all_rows))) if all_rows else ["version"]
    with analysis.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    impl = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_IMPLEMENTATION_{stamp}.md"
    impl.write_text("\n".join([
        "# Human Prior Implementation",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- base_version: V6_1_3_gpt55_guarded_endpoint",
        "- final_auto_submitted: no",
        "- api_key_logged: no",
        "",
        "## Modules",
        "",
        "- Human Process Graph: stage prior for retail/kitchen/restaurant/order.",
        "- Tool Affordance Memory: schema-derived read/mutate/aggregate tags.",
        "- Process-Coverage Verifier: shape checks for order aggregate and kitchen branch flow.",
        "- Counterfactual DB Simulator Lite: pre-execution risk checks from pins and mutation ledger.",
        "- Visual-to-Slot Prior: cached visual_state candidates only, verified by tools before mutation.",
        "- Working Memory Manager: caps prompt state to pins, current stage, ledgers, recent turns, and top slots.",
        "- Human Prior Controller: telemetry and policy trace glue behind TRACK2_ENABLE_HUMAN_PRIOR=1.",
        "",
        f"- ablation_csv: {analysis}",
    ]) + "\n", encoding="utf-8")

    gate = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_GATE_SUMMARY_{stamp}.md"
    lines = [
        "# Human Prior Gate Summary",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- final_auto_submitted: no",
        "- api_key_logged: no",
        f"- best_state_updated: {'yes' if updated else 'no'}",
        f"- selected_candidate: {(best_candidate or {}).get('version', 'none')}",
        "",
        "| version | run_id | joint | result | tool | micro | avg_tool_calls | hp_events | cf_decisions | duplicate_blocks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(f"| {s['version']} | {s['run_id']} | {s['joint_success']:.3f} | {s['result_success']:.3f} | {s['tool_success']:.3f} | {s['micro_tool_accuracy']:.3f} | {s['avg_tool_calls']:.2f} | {s.get('human_prior_events',0)} | {s.get('counterfactual_decisions',0)} | {s.get('duplicate_mutation_blocks',0)} |")
    lines += ["", "## Per-Task Rows", "", "| version | task | joint | result | tool | micro | tool_calls | matches/gt |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in all_rows:
        lines.append(f"| {r['version']} | {r['task']} | {r['joint_success']:.3f} | {r['result_success']:.3f} | {r['tool_success']:.3f} | {r['micro_tool_accuracy']:.3f} | {r['avg_tool_calls']:.1f} | {r['tool_matches']:.0f}/{r['gt_tool_calls']:.0f} |")
    gate.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ablation = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_ABLATION_{stamp}.md"
    ablation.write_text("\n".join([
        "# Human Prior Ablation",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- baseline: V6_1_3_gpt55_guarded_endpoint joint 0.50, result 0.75, tool 0.50, micro 0.7083",
        "- full_run_expanded: no",
        "- reason: only expand if 4-task joint >= 0.75 or order/kitchen clearly improves.",
        "",
        "## Summary",
        "",
        "\n".join(f"- {s['version']}: joint={s['joint_success']:.3f}, result={s['result_success']:.3f}, tool={s['tool_success']:.3f}, micro={s['micro_tool_accuracy']:.3f}, avg_tool_calls={s['avg_tool_calls']:.2f}" for s in summaries),
    ]) + "\n", encoding="utf-8")

    paper = CODEX_ROOT / "reports" / f"HUMAN_PRIOR_PAPER_NOTES_{stamp}.md"
    paper.write_text("\n".join([
        "# Human-Prior Tool Agent for Egocentric Service Tasks",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "- Track2 policy: commercial API allowed; GPT-5.5 used as service agent through OpenAI-compatible endpoint.",
        "- final_auto_submitted: no",
        "- claim boundary: motivated by human cognitive priors; effectiveness must be judged by dev gate/ablation metrics.",
        "",
        "## Components",
        "",
        "- Human Process Graph: encodes scenario process stages without GT answers.",
        "- Tool Affordance Memory: derives tool risk/timing tags from schema.",
        "- Visual-to-Slot Prior: converts cached visual evidence into candidate slots, never direct final answers.",
        "- Counterfactual DB Simulator: checks mutation consequences against pins and episode ledger.",
        "- Process-Coverage Verifier: targets result/tool mismatch by requiring aggregate stages after mutations.",
        "- Working Memory Manager: caps active tool and entity candidates to reduce order/kitchen drift.",
        "- Socially Robust User Guidance: short replies and no misleading follow-up after subgoals complete.",
        "",
        "## Ablation Metrics",
        "",
        "\n".join(f"- {s['version']}: joint={s['joint_success']:.3f}, micro={s['micro_tool_accuracy']:.3f}, events={s.get('human_prior_events',0)}" for s in summaries),
    ]) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--runs", nargs="+", required=True, help="version:run_id pairs")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    expanded = []
    for item in args.runs:
        version, run_id = item.split(":", 1)
        got = summarize_run(version, run_id, args.model)
        summaries.append(got["summary"])
        all_rows.extend(got["rows"])
        expanded.append(got)
    best_path = CODEX_ROOT / "state" / "best_track2_api_version.json"
    current_best = load_json(best_path) or {}
    updated = False
    best_candidate = None
    for got in expanded:
        if should_update_best(got["summary"], got["rows"], current_best):
            if best_candidate is None or (got["summary"]["joint_success"], got["summary"]["micro_tool_accuracy"]) > (best_candidate["joint_success"], best_candidate["micro_tool_accuracy"]):
                best_candidate = got["summary"]
    if best_candidate:
        out = {
            "version": best_candidate["version"],
            "joint_success": best_candidate["joint_success"],
            "tool_success": best_candidate["tool_success"],
            "result_success": best_candidate["result_success"],
            "micro_tool_accuracy": best_candidate["micro_tool_accuracy"],
            "avg_rounds": best_candidate["avg_rounds"],
            "avg_tool_calls": best_candidate["avg_tool_calls"],
            "run_id": best_candidate["run_id"],
            "model": best_candidate["model"],
            "endpoint": os.environ.get("TRACK2_OPENAI_BASE_URL", "https://ai-pixel.online/v1"),
            "external_api_used": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "Updated by V7 human-prior strict gate criterion; final not auto-submitted.",
        }
        best_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated = True
    write_reports(all_rows, summaries, updated, best_candidate, stamp)
    state = CODEX_ROOT / "state" / "latest_human_prior_ablation.json"
    state.write_text(json.dumps({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": args.runs, "best_updated": updated, "selected": best_candidate}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reports_stamp": stamp, "best_updated": updated, "selected": best_candidate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
