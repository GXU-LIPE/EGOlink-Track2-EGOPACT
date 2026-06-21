#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V18 val41 oracle-compiler backtest."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
VERSION = "V18_val41_oracle_gt100_distilled"


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_specs() -> List[Tuple[str, int, str]]:
    manifest = read_json(SPLIT_DIR / "manifest.json", {})
    specs = []
    for scenario, number, _idxs in manifest.get("specs", []):
        specs.append((str(scenario), int(number), f"{scenario}{int(number)}"))
    return specs


def compile_spec(spec_info: Tuple[str, int, str], model_dir: Path, run_id: str, oracle_dir: str) -> Dict[str, Any]:
    sys.path.insert(0, str(CODEX / "wrappers"))
    sys.path.insert(0, str(CODEX))
    from egobench_agent_plus.v18_val41_oracle_compiler import V18Val41OracleCompiler

    scenario, number, spec = spec_info
    compiler = V18Val41OracleCompiler(oracle_dir=oracle_dir)
    rows = read_json(SPLIT_DIR / f"{spec}.json", [])
    out_rows = []
    traces = []
    for pos, row in enumerate(rows):
        subset_index = pos + 1
        source_index = row.get("_v8_original_index")
        task_id = row.get("task_id")
        compiled = compiler.compile(spec, subset_index=subset_index, source_original_index=source_index, task_id=task_id)
        calls = compiled.get("tool_calls", [])
        trace = compiled.get("trace", {})
        traces.append(trace)
        out_rows.append({
            "task_id": subset_index,
            "mode": "text",
            "instruction": row.get("Instruction", ""),
            "image_description": row.get("image_description", ""),
            "dialogue": [{"role": "agent", "turn": 0, "content": "V18 oracle-compiled trajectory for val41 diagnostic only."}],
            "tool_calls": [{"turn": 0, "calls": calls, "blocked_calls": [], "results": [], "v18_trace": trace}],
            "tool_calls_count": len(calls),
            "rounds_count": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens_consumed": 0,
            "final_run": False,
            "uses_val41_gt_oracle_rules": True,
            "uses_final_hidden_metadata": False,
            "for_final_candidate": False,
            "v18_oracle_trace": trace,
        })
    write_json(model_dir / f"{spec}_easy.json", out_rows)
    write_json(CODEX / "runs" / VERSION / run_id / "shards" / f"{spec}.trace.json", traces)
    return {
        "spec": spec,
        "scenario": scenario,
        "number": number,
        "task_count": len(rows),
        "result_file": str(model_dir / f"{spec}_easy.json"),
        "rule_hits": sum(1 for t in traces if t.get("oracle_rule_hit")),
        "missing_rules": [t for t in traces if not t.get("oracle_rule_hit")],
    }


def evaluate_model_dir(model_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    rows = []
    for scenario, number, spec in load_specs():
        gt_path = SPLIT_DIR / f"{spec}.json"
        pred_path = model_dir / f"{spec}_easy.json"
        if not pred_path.exists():
            rows.append({"spec": spec, "scenario": scenario, "number": number, "valid": 0, "error": "missing_prediction"})
            continue
        try:
            metrics = evaluate_interaction_success(
                str(gt_path),
                str(pred_path),
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
            })
        except Exception as exc:
            rows.append({"spec": spec, "scenario": scenario, "number": number, "valid": 0, "error": f"{type(exc).__name__}: {exc}"})
    valid = sum(r.get("valid", 0) for r in rows)
    correct = sum(r.get("correct_calls", 0) for r in rows)
    gt_calls = sum(r.get("gt_calls", 0) for r in rows)

    def wavg(key: str) -> float:
        return sum(r.get(key, 0) * r.get("valid", 0) for r in rows) / valid if valid else 0.0

    by_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        sr = [r for r in rows if r["scenario"] == scenario]
        sv = sum(r.get("valid", 0) for r in sr)
        sc = sum(r.get("correct_calls", 0) for r in sr)
        sg = sum(r.get("gt_calls", 0) for r in sr)
        by_scenario[scenario] = {
            "valid": sv,
            "joint": sum(r.get("joint", 0) * r.get("valid", 0) for r in sr) / sv if sv else 0.0,
            "result": sum(r.get("result", 0) * r.get("valid", 0) for r in sr) / sv if sv else 0.0,
            "tool": sum(r.get("tool", 0) * r.get("valid", 0) for r in sr) / sv if sv else 0.0,
            "micro": sc / sg if sg else 0.0,
            "correct_calls": sc,
            "gt_calls": sg,
            "interaction_calls": sum(r.get("interaction_calls", 0) for r in sr),
        }
    return {
        "rows": rows,
        "summary": {
            "valid": valid,
            "joint": wavg("joint"),
            "result": wavg("result"),
            "tool": wavg("tool"),
            "micro": correct / gt_calls if gt_calls else wavg("micro"),
            "avg_task_accuracy": wavg("avg_task_accuracy"),
            "correct_calls": correct,
            "gt_calls": gt_calls,
            "interaction_calls": sum(r.get("interaction_calls", 0) for r in rows),
        },
        "by_scenario": by_scenario,
    }


def baseline_summaries() -> Dict[str, Any]:
    paths = {
        "V10_val41": CODEX / "runs" / "V10_full_memory_final_candidate_draft" / "V10_full_memory_final_candidate_draft_A_medium_sanity_20260618_1716" / "eval_summary.json",
        "V14_oracle_teacher": CODEX / "runs" / "V14_val41_oracle_teacher" / "v14_gt_distill_20260619_211502" / "eval_summary.json",
        "V14_candidate_selection": CODEX / "runs" / "V14_candidate_selection_val41" / "v14_candidate_selection_20260619_2134" / "eval_summary.json",
        "V17_clean": CODEX / "runs" / "V17_clean_repaired_eval" / "v17_clean_20260620_1515" / "eval_summary.json",
        "V17_smoke5": CODEX / "runs" / "V17_GT100_EXECUTABLE_COMPILER_SMOKE5" / "v17_smoke5_20260620_1140" / "eval_summary.json",
    }
    out = {}
    for name, path in paths.items():
        data = read_json(path, {})
        out[name] = data.get("summary") or data.get("by_candidate") or {}
    return out


def failure_records(eval_result: Dict[str, Any], compile_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    missing_by_spec = {item["spec"]: item.get("missing_rules", []) for item in compile_items}
    out = []
    for row in eval_result["rows"]:
        for i, detail in enumerate(row.get("detailed_results", []) or [], start=1):
            joint = bool(detail.get("joint_success"))
            if joint:
                continue
            tb = detail.get("tool_based", {}) or {}
            rb = detail.get("result_based", {}) or {}
            reason = []
            if missing_by_spec.get(row["spec"]):
                reason.append("oracle_rule_lookup_failed")
            if not tb.get("success"):
                reason.append("tool_mismatch_or_compiler_output_shape")
            if not rb.get("success"):
                reason.append("result_db_hash_mismatch")
            out.append({
                "spec": row["spec"],
                "scenario": row["scenario"],
                "subset_index": i,
                "task_id": detail.get("task_id"),
                "reason": reason or ["unknown"],
                "tool_based": tb,
                "result_based": rb,
            })
    return out


def write_eval_report(run_id: str, model_dir: Path, eval_result: Dict[str, Any], compile_items: List[Dict[str, Any]], baselines: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V18_VAL41_ORACLE_EVAL_{run_id}.md"
    s = eval_result["summary"]
    lines = [
        f"# V18 Val41 Oracle Compiler Eval {run_id}",
        "",
        "- version: V18_val41_oracle_gt100_distilled",
        "- oracle_self_distillation_diagnostic: true",
        "- final_run: false",
        "- auto_submit: false",
        "- protected_best_updated: false",
        "- v10_zip_overwritten: false",
        "- uses_val41_gt_oracle_rules: true",
        "- uses_final_hidden_metadata: false",
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
        "## Per Scenario",
        "",
        "| scenario | valid | joint | result | tool | micro | calls | interaction_calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, row in sorted(eval_result["by_scenario"].items()):
        lines.append(f"| {scenario} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('interaction_calls', 0)} |")
    lines += [
        "",
        "## Per Spec",
        "",
        "| spec | valid | joint | result | tool | micro | calls | interaction_calls | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in eval_result["rows"]:
        lines.append(f"| {row['spec']} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} | {row.get('interaction_calls', 0)} | {row.get('error', '')} |")
    lines += [
        "",
        "## Controls",
        "",
        "| version | valid | joint | result | tool | micro | calls | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, data in baselines.items():
        if isinstance(data, dict) and "valid" in data:
            lines.append(f"| {name} | {data.get('valid', 0)} | {data.get('joint', 0):.4f} | {data.get('result', 0):.4f} | {data.get('tool', 0):.4f} | {data.get('micro', 0):.4f} | {data.get('correct_calls', 0)}/{data.get('gt_calls', 0)} | existing artifact |")
        else:
            lines.append(f"| {name} | 0 | 0 | 0 | 0 | 0 | 0/0 | unavailable or nested summary |")
    lines.append(f"| V18_oracle_compiler | {s.get('valid', 0)} | {s.get('joint', 0):.4f} | {s.get('result', 0):.4f} | {s.get('tool', 0):.4f} | {s.get('micro', 0):.4f} | {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} | current run |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_failure_report(run_id: str, failures: List[Dict[str, Any]], eval_result: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V18_ORACLE_FAILURE_ANALYSIS_{run_id}.md"
    reason_counts = collections.Counter(reason for f in failures for reason in f.get("reason", []))
    lines = [
        f"# V18 Oracle Failure Analysis {run_id}",
        "",
        "- final_run: false",
        "- oracle_self_distillation_diagnostic: true",
        "- uses_val41_gt_oracle_rules: true",
        "- uses_final_hidden_metadata: false",
        "",
        f"- failed_tasks: {len(failures)}",
        "",
        "## Reason Counts",
        "",
    ]
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines += ["", "## Failures", "", "| spec | subset_index | task_id | reason | tool_matches | result_success |", "|---|---:|---:|---|---:|---|"]
    for f in failures:
        tb = f.get("tool_based", {}) or {}
        rb = f.get("result_based", {}) or {}
        lines.append(f"| {f.get('spec')} | {f.get('subset_index')} | {f.get('task_id')} | {', '.join(f.get('reason', []))} | {tb.get('matches', 0)}/{tb.get('total_gt_calls', 0)} | {rb.get('success')} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_decision_report(run_id: str, eval_result: Dict[str, Any], failures: List[Dict[str, Any]]) -> Path:
    report = CODEX / "reports" / f"V18_NEXT_DECISION_{run_id}.md"
    s = eval_result["summary"]
    joint = float(s.get("joint") or 0)
    micro = float(s.get("micro") or 0)
    passed = joint >= 0.80 and micro >= 0.90
    near_perfect = joint >= 0.999 and micro >= 0.999
    if near_perfect:
        diagnosis = "compiler_upper_bound_confirmed; future bottleneck is non-oracle slot filling and visual/entity resolution"
    elif passed:
        diagnosis = "compiler_mostly_works_but_remaining failures need index/slot/runner/evaluator audit"
    else:
        diagnosis = "compiler_or_rule_reconstruction_bug; below oracle upper-bound threshold"
    lines = [
        f"# V18 Next Decision {run_id}",
        "",
        f"- success_gate_passed_joint80_micro90: {str(passed).lower()}",
        f"- near_perfect_joint100_micro100: {str(near_perfect).lower()}",
        "- final_run: false",
        "- promote_to_final: false",
        "- protected_best_updated: false",
        "- v10_zip_overwritten: false",
        "- use_for_final_candidate: false",
        "",
        "## Required Answers",
        "",
        f"1. val41 GT100 pool complete: {'yes' if int(s.get('valid', 0)) == 41 else 'no'} for evaluated predictions; see pool manifest for retained GT100 count.",
        f"2. V18 reached oracle upper bound: {'yes' if near_perfect else 'no'}, joint={joint:.4f}, micro={micro:.4f}.",
        f"3. Failure diagnosis: {diagnosis}.",
        "4. If 100%, compiler upper bound is valid but cannot be used for final because it is val41 oracle/self-distillation.",
        "5. Next step: replace oracle slot lookup with non-final GT100-derived skeletons plus online visual/entity resolver, then retest on clean/non-oracle validation.",
    ]
    if failures:
        lines += ["", "## Failure Task Count", "", f"- failures: {len(failures)}"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v18_oracle_eval_{stamp()}")
    ap.add_argument("--oracle-dir", default=str(CODEX / "gt_distill_v18_val41_oracle"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    model_dir = EGO / "results" / f"V18_val41_oracle_gt100_distilled-{args.run_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir = CODEX / "runs" / VERSION / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = load_specs()
    compile_items: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(compile_spec, spec, model_dir, args.run_id, args.oracle_dir) for spec in specs]
        for fut in cf.as_completed(futures):
            item = fut.result()
            compile_items.append(item)
            print(json.dumps({"compiled": item["spec"], "task_count": item["task_count"], "rule_hits": item["rule_hits"]}, ensure_ascii=False), flush=True)
    compile_items = sorted(compile_items, key=lambda x: x["spec"])
    write_json(out_dir / "compile_items.json", compile_items)
    eval_result = evaluate_model_dir(model_dir)
    write_json(out_dir / "eval_summary.json", eval_result)
    failures = failure_records(eval_result, compile_items)
    write_json(out_dir / "failure_records.json", failures)
    baselines = baseline_summaries()
    eval_report = write_eval_report(args.run_id, model_dir, eval_result, compile_items, baselines)
    failure_report = write_failure_report(args.run_id, failures, eval_result)
    decision_report = write_decision_report(args.run_id, eval_result, failures)
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": args.run_id,
        "version": VERSION,
        "result_dir": str(model_dir),
        "eval_report": str(eval_report),
        "failure_report": str(failure_report),
        "decision_report": str(decision_report),
        "summary": eval_result["summary"],
        "final_run": False,
        "protected_best_updated": False,
        "v10_zip_overwritten": False,
    }
    write_json(CODEX / "state" / "latest_v18_oracle_eval.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
