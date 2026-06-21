#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V19 smoke10 and, optionally, full val41 if smoke passes."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
VERSION = "V19_gt100_case_reuse_val41"
V10_DIR = EGO / "results" / "gpt-5.5-V10_full_memory_final_candidate_draft-V10_full_memory_final_candidate_draft_A_medium_sanity_20260618_1716"
V14_DIR = EGO / "results" / "V14_candidate_selection_val41-v14_candidate_selection_20260619_2134"


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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_specs() -> List[Tuple[str, int, str]]:
    m = read_json(SPLIT / "manifest.json", {})
    return [(str(s), int(n), f"{s}{int(n)}") for s, n, _ in m.get("specs", [])]


def flatten_result_tools(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(item, dict):
        return out
    for entry in item.get("tool_calls", []) or []:
        calls = entry.get("calls") if isinstance(entry, dict) else None
        if isinstance(calls, list):
            out.extend([c for c in calls if isinstance(c, dict)])
    return out


def make_result_item(row: Dict[str, Any], subset_index: int, candidate: Dict[str, Any], selected_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls = [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters", {})} for x in candidate.get("tool_program", []) if x.get("tool_name")]
    return {
        "task_id": subset_index,
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": "V19 non-oracle case trajectory reuse candidate."}],
        "tool_calls": [{"turn": 0, "calls": calls, "blocked_calls": [], "results": [], "v19_candidate": {
            "candidate_id": candidate.get("candidate_id"),
            "source": candidate.get("source"),
            "source_case_ids": candidate.get("source_case_ids"),
            "score": candidate.get("score"),
            "risk_flags": candidate.get("risk_flags"),
        }}],
        "tool_calls_count": len(calls),
        "rounds_count": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_consumed": 0,
        "final_run": False,
        "uses_val41_gt_for_policy": False,
        "uses_final_hidden_metadata": False,
        "v19_scores": selected_scores,
    }


def evaluate_dir(model_dir: Path, specs: List[Tuple[str, int, str]]) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    rows = []
    for scenario, number, spec in specs:
        gt = SPLIT / f"{spec}.json"
        pred = model_dir / f"{spec}_easy.json"
        if not pred.exists():
            rows.append({"spec": spec, "scenario": scenario, "number": number, "valid": 0, "error": "missing_pred"})
            continue
        metrics = evaluate_interaction_success(str(gt), str(pred), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
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
            "detailed_results": metrics.get("detailed_results", []),
            "error": "",
        })
    valid = sum(r.get("valid", 0) for r in rows)
    correct = sum(r.get("correct_calls", 0) for r in rows)
    gt_calls = sum(r.get("gt_calls", 0) for r in rows)
    def wavg(k: str) -> float:
        return sum(r.get(k, 0) * r.get("valid", 0) for r in rows) / valid if valid else 0.0
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
    }


def per_task_scores(model_dir: Path, specs: List[Tuple[str, int, str]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    out = {}
    for scenario, number, spec in specs:
        gt_items = read_json(SPLIT / f"{spec}.json", [])
        pred_items = read_json(model_dir / f"{spec}_easy.json", [])
        for i, gt_item in enumerate(gt_items):
            if i >= len(pred_items):
                continue
            tmp = CODEX / "runs" / VERSION / "_tmp_eval"
            write_json(tmp / "gt.json", [gt_item])
            write_json(tmp / "pred.json", [pred_items[i]])
            metrics = evaluate_interaction_success(str(tmp / "gt.json"), str(tmp / "pred.json"), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
            d = (metrics.get("detailed_results") or [{}])[0]
            tb = d.get("tool_based", {}) or {}
            out[(spec, i)] = {
                "joint": 1.0 if d.get("joint_success") else 0.0,
                "result": 1.0 if (d.get("result_based", {}) or {}).get("success") else 0.0,
                "tool": 1.0 if tb.get("success") else 0.0,
                "matches": tb.get("matches", 0),
                "gt_calls": tb.get("total_gt_calls", 0),
                "interaction_calls": tb.get("total_interaction_calls", 0),
            }
    return out


def choose_smoke_specs() -> Dict[str, List[int]]:
    clean_state = read_json(CODEX / "state" / "latest_val41_clean_audit.json", {})
    clean_dir = Path(clean_state.get("clean_split", ""))
    selected: Dict[str, List[int]] = collections.defaultdict(list)
    if clean_dir.exists():
        m = read_json(clean_dir / "manifest.json", {})
        for info in m.get("files", []):
            spec = Path(info["file"]).stem
            for pos in info.get("source_local_positions", []):
                selected[spec].append(int(pos))
    # Add V14 near-miss: use GT only here for choosing debug smoke, not for policy.
    specs = load_specs()
    v14_scores = per_task_scores(V14_DIR, specs)
    candidates = []
    for (spec, pos), score in v14_scores.items():
        if pos in selected.get(spec, []):
            continue
        if score["joint"] < 1.0 and (score["matches"] > 0 or score["result"] > 0):
            candidates.append((score["matches"], score["result"], spec, pos))
    candidates.sort(reverse=True)
    for _matches, _result, spec, pos in candidates:
        if sum(len(v) for v in selected.values()) >= 10:
            break
        selected[spec].append(pos)
    return {k: sorted(set(v)) for k, v in selected.items()}


def build_context(spec: str, scenario: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "spec": spec,
        "scenario": scenario,
        "instruction": row.get("Instruction", ""),
        "visual_text": "\n".join([str(row.get("image_description", "")), str(row.get("key", "")), str(row.get("value", ""))]),
        "task_type": "",
        "entity_types": [],
    }


def build_predictions(run_id: str, selected: Dict[str, List[int]] | None, out_model: str) -> Tuple[Path, List[Dict[str, Any]]]:
    sys.path.insert(0, str(CODEX / "wrappers"))
    sys.path.insert(0, str(CODEX))
    from egobench_agent_plus.v19_program_transplanter import generate_candidates
    from egobench_agent_plus.v19_case_retriever import classify_task_type

    model_dir = EGO / "results" / out_model
    model_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario, number, spec in load_specs():
        rows = read_json(SPLIT / f"{spec}.json", [])
        v10 = read_json(V10_DIR / f"{spec}_easy.json", [])
        v14 = read_json(V14_DIR / f"{spec}_easy.json", [])
        out_rows = []
        for pos, row in enumerate(rows):
            if selected is not None and pos not in selected.get(spec, []):
                continue
            context = build_context(spec, scenario, row)
            context["task_type"] = classify_task_type(context["instruction"], [])
            generated = generate_candidates(context, v10_item=v10[pos] if pos < len(v10) else None, v14_item=v14[pos] if pos < len(v14) else None)
            ranked = generated["ranked"]
            best = ranked[0] if ranked else {"candidate_id": "empty", "source": "none", "tool_program": [], "score": 0}
            out_rows.append(make_result_item(row, len(out_rows) + 1, best, [{
                "candidate_id": c.get("candidate_id"),
                "source": c.get("source"),
                "score": c.get("score"),
                "risk_flags": c.get("risk_flags"),
                "source_case_ids": c.get("source_case_ids"),
            } for c in ranked]))
            rec = {
                "run_id": run_id,
                "spec": spec,
                "source_local_pos": pos,
                "source_task_id": row.get("task_id"),
                "selected_candidate_id": best.get("candidate_id"),
                "selected_source": best.get("source"),
                "selected_score": best.get("score"),
                "case_hit_trace": generated["traces"].get("case_hit_trace"),
                "slot_rewrite_trace": generated["traces"].get("slot_rewrite_trace"),
                "program_scores": [{
                    "candidate_id": c.get("candidate_id"),
                    "source": c.get("source"),
                    "score": c.get("score"),
                    "risk_flags": c.get("risk_flags"),
                } for c in ranked],
            }
            records.append(rec)
            append_jsonl(CODEX / "analysis" / "v19_case_retrieval_trace.jsonl", {"run_id": run_id, "spec": spec, "pos": pos, "hits": rec["case_hit_trace"]})
            append_jsonl(CODEX / "analysis" / "v19_slot_rewrite_trace.jsonl", {"run_id": run_id, "spec": spec, "pos": pos, "traces": rec["slot_rewrite_trace"]})
            append_jsonl(CODEX / "analysis" / "v19_program_scores.jsonl", {"run_id": run_id, "spec": spec, "pos": pos, "scores": rec["program_scores"]})
        if out_rows:
            write_json(model_dir / f"{spec}_easy.json", out_rows)
    return model_dir, records


def materialize_smoke_gt(run_id: str, selected: Dict[str, List[int]]) -> Path:
    out = CODEX / "runs" / VERSION / run_id / "smoke_gt"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for spec, positions in sorted(selected.items()):
        rows = read_json(SPLIT / f"{spec}.json", [])
        subset = [rows[i] for i in positions if i < len(rows)]
        if not subset:
            continue
        write_json(out / f"{spec}.json", subset)
        scenario = "".join(ch for ch in spec if not ch.isdigit())
        number = int(spec[len(scenario):])
        files.append((scenario, number, spec))
    write_json(out / "manifest.json", {"run_id": run_id, "selected": selected, "files": files, "uses_val41_gt_for_policy": False, "uses_val41_gt_for_smoke_selection": True})
    return out


def evaluate_smoke(model_dir: Path, smoke_gt: Path) -> Dict[str, Any]:
    old_split = globals()["SPLIT"]
    try:
        globals()["SPLIT"] = smoke_gt
        specs = [(s, int(n), spec) for s, n, spec in read_json(smoke_gt / "manifest.json", {}).get("files", [])]
        return evaluate_dir(model_dir, specs)
    finally:
        globals()["SPLIT"] = old_split


def write_retrieval_impl_report(run_id: str) -> Path:
    path = CODEX / "reports" / f"V19_CASE_RETRIEVAL_IMPLEMENTATION_{run_id}.md"
    lines = [
        f"# V19 Case Retrieval Implementation {run_id}",
        "",
        "- final_run: false",
        "- uses_val41_gt_for_retrieval: false",
        "- uses_final_hidden_metadata: false",
        "",
        "Implementation keeps full GT100 cases in `gt_case_library_v19/gt100_cases.jsonl` and scores by scenario, task type, lexical/visual overlap, entity compatibility, and program-shape compatibility.",
        "Slot rewriting blocks direct transfer of case user_id and restaurant/entity slots when current evidence supplies a replacement; low-confidence copies are flagged.",
        "Candidate scoring is non-oracle: it checks slot completeness, schema/shape risks, foreign slot copy, closure, broad tool-count risk, and historical V10/V14 candidate presence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_smoke_report(run_id: str, selected: Dict[str, List[int]], eval_result: Dict[str, Any], v14_smoke: Dict[str, Any], passed: bool) -> Path:
    path = CODEX / "reports" / f"V19_SMOKE10_CASE_REUSE_{run_id}.md"
    s = eval_result["summary"]; b = v14_smoke["summary"]
    lines = [
        f"# V19 Smoke10 Case Reuse {run_id}",
        "",
        "- final_run: false",
        "- uses_val41_gt_for_policy: false",
        "- uses_val41_gt_for_smoke_selection_and_eval: true",
        f"- smoke_passed: {str(passed).lower()}",
        "",
        f"- selected: `{json.dumps(selected, ensure_ascii=False)}`",
        "",
        "## Summary",
        "",
        "| version | valid | joint | result | tool | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V14_candidate_smoke | {b.get('valid', 0)} | {b.get('joint', 0):.4f} | {b.get('result', 0):.4f} | {b.get('tool', 0):.4f} | {b.get('micro', 0):.4f} | {b.get('correct_calls', 0)}/{b.get('gt_calls', 0)} |",
        f"| V19_smoke | {s.get('valid', 0)} | {s.get('joint', 0):.4f} | {s.get('result', 0):.4f} | {s.get('tool', 0):.4f} | {s.get('micro', 0):.4f} | {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_full_reports(run_id: str, eval_result: Dict[str, Any], v14_eval: Dict[str, Any], records: List[Dict[str, Any]]) -> Tuple[Path, Path, Path]:
    result_path = CODEX / "reports" / f"V19_VAL41_CASE_REUSE_RESULT_{run_id}.md"
    diff_path = CODEX / "reports" / f"V19_V14_DIFF_ANALYSIS_{run_id}.md"
    decision_path = CODEX / "reports" / f"V19_NEXT_DECISION_{run_id}.md"
    s = eval_result["summary"]; b = v14_eval.get("summary", {})
    lines = [
        f"# V19 Val41 Case Reuse Result {run_id}",
        "",
        "- final_run: false",
        "- protected_best_updated: false",
        "- v10_zip_overwritten: false",
        "- uses_val41_gt_for_policy: false",
        "- uses_final_hidden_metadata: false",
        "",
        "## Summary",
        "",
        "| version | valid | joint | result | tool | micro | calls | interaction_calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| V14_candidate_selection | {b.get('valid', 0)} | {b.get('joint', 0):.4f} | {b.get('result', 0):.4f} | {b.get('tool', 0):.4f} | {b.get('micro', 0):.4f} | {b.get('correct_calls', 0)}/{b.get('gt_calls', 0)} | {b.get('interaction_calls', 0)} |",
        f"| V19_case_reuse | {s.get('valid', 0)} | {s.get('joint', 0):.4f} | {s.get('result', 0):.4f} | {s.get('tool', 0):.4f} | {s.get('micro', 0):.4f} | {s.get('correct_calls', 0)}/{s.get('gt_calls', 0)} | {s.get('interaction_calls', 0)} |",
        "",
        "## Per Spec",
        "",
        "| spec | valid | joint | result | tool | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_result["rows"]:
        lines.append(f"| {row['spec']} | {row.get('valid', 0)} | {row.get('joint', 0):.3f} | {row.get('result', 0):.3f} | {row.get('tool', 0):.3f} | {row.get('micro', 0):.3f} | {row.get('correct_calls', 0)}/{row.get('gt_calls', 0)} |")
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Diff needs per-task GT only after eval.
    v14_scores = per_task_scores(V14_DIR, load_specs())
    v19_scores = per_task_scores(EGO / "results" / f"V19_gt100_case_reuse_val41-{run_id}", load_specs())
    new_joint = []
    losses = []
    for key, score in v19_scores.items():
        base = v14_scores.get(key, {})
        if score.get("joint", 0) > base.get("joint", 0):
            new_joint.append({"spec": key[0], "pos": key[1], "v19": score, "v14": base})
        elif score.get("joint", 0) < base.get("joint", 0):
            losses.append({"spec": key[0], "pos": key[1], "v19": score, "v14": base})
    diff_lines = [
        f"# V19 vs V14 Diff Analysis {run_id}",
        "",
        "- final_run: false",
        "- val41_gt_used_after_eval_only: true",
        "",
        f"- new_joint_success_tasks: {len(new_joint)}",
        f"- joint_losses: {len(losses)}",
        "",
        "## New Joint Success",
        "",
    ]
    for row in new_joint:
        diff_lines.append(f"- {row['spec']} pos={row['pos']} matches={row['v19'].get('matches')}/{row['v19'].get('gt_calls')}")
    diff_lines += ["", "## Losses", ""]
    for row in losses:
        diff_lines.append(f"- {row['spec']} pos={row['pos']} v14={row['v14'].get('joint')} v19={row['v19'].get('joint')}")
    diff_path.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
    exceeded = s.get("joint", 0) > b.get("joint", 0) or s.get("micro", 0) > b.get("micro", 0)
    strong = s.get("joint", 0) >= 0.1951
    dec = [
        f"# V19 Next Decision {run_id}",
        "",
        f"- smoke_passed: true",
        f"- val41_exceeds_v14: {str(exceeded).lower()}",
        f"- strong_goal_joint_ge_19_51: {str(strong).lower()}",
        "- run_final: false",
        "- protected_best_updated: false",
        "- v10_zip_overwritten: false",
        "- auto_submit: false",
        "",
        "## Required Answers",
        "",
        "1. V19 preserves full GT100 cases in `gt_case_library_v19/gt100_cases.jsonl`; it does not compress them into only rules.",
        "2. The case library count and exclusions are in `gt100_case_manifest.json` and the audit report.",
        "3. Smoke10 status is recorded in the smoke report.",
        f"4. Val41 exceeds V14: {exceeded}.",
        f"5. New joint tasks: {len(new_joint)}.",
        "6. Failure causes are traceable through retrieval, slot rewrite, and scorer JSONL traces.",
        f"7. Worth final-style online continuation: {exceeded and strong}.",
        "8. V10 protected zip was not overwritten.",
        "9. No final submission was made.",
    ]
    decision_path.write_text("\n".join(dec) + "\n", encoding="utf-8")
    return result_path, diff_path, decision_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v19_case_reuse_{stamp()}")
    ap.add_argument("--full-if-smoke-pass", action="store_true")
    args = ap.parse_args()
    write_retrieval_impl_report(args.run_id)
    selected = choose_smoke_specs()
    smoke_gt = materialize_smoke_gt(args.run_id, selected)
    smoke_model, smoke_records = build_predictions(args.run_id + "_smoke", selected, f"V19_gt100_case_reuse_smoke10-{args.run_id}")
    smoke_eval = evaluate_smoke(smoke_model, smoke_gt)
    # V14 smoke baseline on same subset.
    v14_smoke_dir = CODEX / "runs" / VERSION / args.run_id / "v14_smoke_pred"
    v14_smoke_dir.mkdir(parents=True, exist_ok=True)
    for spec, positions in selected.items():
        rows = read_json(V14_DIR / f"{spec}_easy.json", [])
        write_json(v14_smoke_dir / f"{spec}_easy.json", [rows[i] for i in positions if i < len(rows)])
    v14_smoke = evaluate_smoke(v14_smoke_dir, smoke_gt)
    ss = smoke_eval["summary"]; bs = v14_smoke["summary"]
    passed = (
        ss.get("joint", 0) > bs.get("joint", 0)
        or (ss.get("micro", 0) - bs.get("micro", 0)) >= 0.10
        or (ss.get("correct_calls", 0) > 0 and bs.get("correct_calls", 0) == 0)
    )
    smoke_report = write_smoke_report(args.run_id, selected, smoke_eval, v14_smoke, passed)
    out_dir = CODEX / "runs" / VERSION / args.run_id
    write_json(out_dir / "smoke_eval_summary.json", {"v19": smoke_eval, "v14": v14_smoke, "passed": passed, "selected": selected, "records": smoke_records})
    state = {
        "run_id": args.run_id,
        "smoke_report": str(smoke_report),
        "smoke_passed": passed,
        "full_run": False,
        "final_run": False,
    }
    if not passed:
        decision = CODEX / "reports" / f"V19_NEXT_DECISION_{args.run_id}.md"
        decision.write_text(f"# V19 Next Decision {args.run_id}\n\n- smoke_passed: false\n- run_full_val41: false\n- run_final: false\n- protected_best_updated: false\n- v10_zip_overwritten: false\n\nSmoke10 did not improve over V14 candidate selection, so full val41 was not run.\n", encoding="utf-8")
        state["decision_report"] = str(decision)
        write_json(CODEX / "state" / "latest_v19_case_reuse.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.full_if_smoke_pass:
        full_model, full_records = build_predictions(args.run_id, None, f"V19_gt100_case_reuse_val41-{args.run_id}")
        full_eval = evaluate_dir(full_model, load_specs())
        v14_eval = read_json(CODEX / "runs" / "V14_candidate_selection_val41" / "v14_candidate_selection_20260619_2134" / "eval_summary.json", {})
        write_json(out_dir / "eval_summary.json", full_eval)
        write_json(out_dir / "selection_records.json", full_records)
        result_report, diff_report, decision_report = write_full_reports(args.run_id, full_eval, v14_eval, full_records)
        state.update({
            "full_run": True,
            "result_report": str(result_report),
            "diff_report": str(diff_report),
            "decision_report": str(decision_report),
            "summary": full_eval["summary"],
        })
    write_json(CODEX / "state" / "latest_v19_case_reuse.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
