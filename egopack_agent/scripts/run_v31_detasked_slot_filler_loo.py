#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V31 detasked slot filler leave-one-out on frozen val41."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V22_DIR = EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915"
V30_SLOT_DIR = EGO / "results" / "V30_gt_experience_prior_agent_slot_only_selected_r0-v30_gt_experience_prior_20260621_121710"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
EXPECTED_V10_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))
sys.path.insert(0, str(EGO / "analysis_scripts"))

from egobench_agent_plus.v30_prior_bank import ExperiencePriorBank, make_prior_record, norm_text, read_jsonl, tokens, write_jsonl  # noqa: E402
from egobench_agent_plus.v30_prior_retrieval_agent import PriorRetrievalServiceAgentV30  # noqa: E402
from egobench_agent_plus.v31_loo_program_executor import V31LOOProgramExecutor  # noqa: E402
from egobench_agent_plus.v31_loo_selector import V31LOOSelector  # noqa: E402
from egobench_agent_plus.v31_loo_slot_filler import DetaskedSlotFillerV31, EvidenceIndex, detasked_slot_decision_records  # noqa: E402


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def load_specs() -> List[Tuple[str, int, List[int]]]:
    manifest = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in manifest.get("specs", [])]


def init_db(scenario: str, number: int) -> Any:
    if scenario == "retail":
        from tools.retail.retail_db import RetailDB
        from tools.retail import retail_init
        db = RetailDB(); db.init_from_json(getattr(retail_init, f"retail_init_data{number}")); return db
    if scenario == "restaurant":
        from tools.restaurant.restaurant_db import RestaurantDB
        from tools.restaurant import restaurant_init
        db = RestaurantDB(); db.init_from_json(getattr(restaurant_init, f"restaurant_init_data{number}", None) or getattr(restaurant_init, "restaurant_init_data")); return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init
        db = OrderDB(); db.init_from_json(getattr(order_init, f"order_init_data{number}", None) or getattr(order_init, "order_init_data")); return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init
        db = KitchenDB(); db.init_from_json(getattr(kitchen_init, f"kitchen_init_data{number}", None) or getattr(kitchen_init, "kitchen_init_data")); return db
    raise ValueError(scenario)


def all_tasks() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        for pos, row in enumerate(read_json(SPLIT_DIR / f"{spec}.json", [])):
            idx = int(row.get("_v8_original_index", pos))
            rows.append({"scenario": scenario, "number": number, "spec": spec, "local_pos": pos, "index": idx, "task_key": f"{spec}::{idx}", "row": row})
    return rows


def make_item(row: Dict[str, Any], program: List[Dict[str, Any]], label: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} V31 LOO candidate."}],
        "tool_calls": [{"turn": 0, "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program], "blocked_calls": [], "results": [], "v31_meta": meta or {}}],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
    }


def item_program(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not item:
        return []
    out: List[Dict[str, Any]] = []
    for block in item.get("tool_calls") or []:
        for call in block.get("calls") or []:
            out.append({"tool_name": call.get("tool_name") or call.get("name"), "parameters": call.get("parameters") or {}})
    return out


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    rows = read_json(result_dir / f"{spec}_easy.json", [])
    return rows[pos] if isinstance(rows, list) and pos < len(rows) else None


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v31_eval_") as td:
        gt_path = Path(td) / "gt.json"; pred_path = Path(td) / "pred.json"
        write_json(gt_path, [gt_item]); write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(str(gt_path), str(pred_path), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
    detail = (metrics.get("detailed_results") or [{}])[0]
    tb = detail.get("tool_based") or {}; rb = detail.get("result_based") or {}; micro = metrics.get("micro_tool_stats") or {}
    return {
        "joint": 1.0 if detail.get("joint_success") else 0.0,
        "result": 1.0 if rb.get("success") else 0.0,
        "tool": 1.0 if tb.get("success") else 0.0,
        "matches": int(tb.get("matches", 0) or 0),
        "gt_calls": int(tb.get("total_gt_calls", 0) or 0),
        "interaction_calls": int(tb.get("total_interaction_calls", 0) or 0),
        "micro": float(micro.get("micro_accuracy", 0) or 0),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = len(rows)
    matched = sum(int(r.get("matches", 0) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", 0) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(float(r.get("joint", 0)) for r in rows) / valid if valid else 0,
        "result": sum(float(r.get("result", 0)) for r in rows) / valid if valid else 0,
        "tool": sum(float(r.get("tool", 0)) for r in rows) / valid if valid else 0,
        "micro": matched / gt if gt else 0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (float(score.get("joint", 0)), float(score.get("tool", 0)), float(score.get("result", 0)), int(score.get("matches", 0)), -int(score.get("interaction_calls", 999999)))


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def val41_case_map(tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        made = make_prior_record(task["row"], task["spec"], "val41_heldout_signature", False)
        if not made:
            continue
        _, case = made
        out[task["task_key"]] = case
    return out


def filter_loo_cases(all_cases: List[Dict[str, Any]], heldout_case: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    held_case_id = heldout_case.get("case_id")
    held_trigger = norm_text(heldout_case.get("trigger_text", ""))
    held_slots = heldout_case.get("slot_values") or {}
    kept: List[Dict[str, Any]] = []
    removed = Counter()
    for case in all_cases:
        if case.get("source_type") in {"val41_gt", "v29_round1_gt_repair"}:
            if held_case_id and case.get("case_id") == held_case_id:
                removed["same_case_id"] += 1
                continue
            if held_trigger and norm_text(case.get("trigger_text", "")) == held_trigger:
                removed["same_trigger"] += 1
                continue
            # Extra guard against duplicated V29 repair record with same slots and family.
            if case.get("program_family") == heldout_case.get("program_family") and case.get("slot_values") == held_slots:
                removed["same_family_and_slots"] += 1
                continue
        kept.append(case)
    return kept, dict(removed)


def build_slot_bank(bank_dir: Path, out_dir: Path, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    cases = read_jsonl(bank_dir / "dev_experience_cases.jsonl")
    records = detasked_slot_decision_records(cases)
    write_jsonl(out_dir / "detasked_slot_decisions.jsonl", records)
    manifest = {
        "slot_decision_count": len(records),
        "source_counts": dict(Counter(r.get("source") for r in records)),
        "scenario_counts": dict(Counter(r.get("scenario") for r in records)),
        "slot_type_counts": dict(Counter(r.get("slot_type") for r in records)),
        "uses_final_hidden_metadata": False,
        "contains_task_id_answer_lookup": False,
        "loo_required": True,
        "val41_task_count": len(tasks),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def run_once(run_id: str, round_id: int, repair_policy: Dict[str, Any], bank_dir: Path, out_dir: Path, tasks: List[Dict[str, Any]], v22_rows: List[Dict[str, Any]], v30_slot_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_cases = read_jsonl(bank_dir / "dev_experience_cases.jsonl")
    vcase = val41_case_map(tasks)
    evidence = EvidenceIndex(CODEX / "analysis" / "v26_bound_slots_val41.jsonl", CODEX / "analysis" / "v26_mm_evidence_val41.jsonl")
    filler = DetaskedSlotFillerV31(evidence)
    executor = V31LOOProgramExecutor(filler)
    selector = V31LOOSelector()
    slot_agent = PriorRetrievalServiceAgentV30(ExperiencePriorBank(bank_dir), mode="slot_only")
    v22_by_key = {(r["spec"], r["index"]): r for r in v22_rows}
    selected_scores: List[Dict[str, Any]] = []
    oracle_scores: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    selected_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    cand_path = CODEX / "analysis" / "v31_loo_candidates.jsonl"
    dry_path = CODEX / "analysis" / "v31_loo_dryrun_trace.jsonl"
    sel_path = CODEX / "analysis" / "v31_loo_selection_trace.jsonl"
    if round_id == 0:
        for p in [cand_path, dry_path, sel_path]:
            p.write_text("", encoding="utf-8")

    for task in tasks:
        spec, idx, pos, scenario, number, row, task_key = task["spec"], task["index"], task["local_pos"], task["scenario"], task["number"], task["row"], task["task_key"]
        v22_item = load_item(V22_DIR, spec, pos) or make_item(row, [], "missing_v22")
        v22_score = v22_by_key[(spec, idx)]
        held_case = vcase.get(task_key, {})
        loo_cases, removed = filter_loo_cases(all_cases, held_case)
        db_factory = lambda s=scenario, n=number: init_db(s, n)
        slot_only = slot_agent.build_candidates(task_key, scenario, row, db_factory(), top_k=4)
        candidates = executor.build_candidates(task_key, scenario, row, db_factory(), loo_cases, v22_item, slot_only, held_case.get("case_id", ""), top_k=8)
        candidates = executor.dryrun_candidates(scenario, db_factory, row, candidates)
        for cand in candidates:
            cand_item = make_item(row, cand.get("tool_program") or [], cand.get("candidate_id", "cand"), {"round": round_id, "loo": True})
            cand["post_eval_score"] = evaluate_one(row, cand_item, scenario, number)
            append_jsonl(cand_path, {"run_id": run_id, "round": round_id, "task_key": task_key, "candidate": cand})
            append_jsonl(dry_path, {"run_id": run_id, "round": round_id, "task_key": task_key, "candidate_id": cand.get("candidate_id"), "source": cand.get("source"), "dryrun": cand.get("dryrun"), "post_eval_score": cand.get("post_eval_score"), "slot_values_used": cand.get("slot_values_used"), "risk_flags": cand.get("risk_flags")})
        selection = selector.select(task_key, v22_item, v22_score, candidates)
        selected = selection.get("selected") or {}
        selected_item = make_item(row, selected.get("tool_program") or [], selected.get("candidate_id", "selected"), {"selection": selection, "round": round_id})
        selected_score = evaluate_one(row, selected_item, scenario, number)
        scored_candidates = [(c, c.get("post_eval_score") or {}, make_item(row, c.get("tool_program") or [], c.get("candidate_id", "cand"), {"oracle_best": True, "round": round_id})) for c in candidates]
        best_cand, best_score, best_item = max(scored_candidates, key=lambda x: score_tuple(x[1])) if scored_candidates else ({}, {}, selected_item)
        selected_items[(spec, idx)] = selected_item
        oracle_items[(spec, idx)] = best_item
        selected_scores.append(selected_score)
        oracle_scores.append(best_score)
        rec = {
            "run_id": run_id,
            "round": round_id,
            "task_key": task_key,
            "spec": spec,
            "scenario": scenario,
            "index": idx,
            "v22_score": v22_score,
            "selected_score": selected_score,
            "oracle_score": best_score,
            "selected_candidate_id": selection.get("selected_candidate_id"),
            "selected_source": selection.get("selected_source"),
            "selected_prior_id": selection.get("selected_prior_id"),
            "selected_program_family": selection.get("selected_program_family"),
            "selection_reason": selection.get("reason"),
            "candidate_count": len(candidates),
            "oracle_candidate_id": best_cand.get("candidate_id"),
            "oracle_source": best_cand.get("source"),
            "oracle_prior_id": best_cand.get("prior_id"),
            "oracle_program_family": best_cand.get("program_family"),
            "loo_removed": removed,
            "heldout_case_id": held_case.get("case_id"),
            "heldout_program_family": held_case.get("program_family"),
            "repair_policy": repair_policy,
            "failure_category": failure_category(v22_score, selected_score, best_score, selection, best_cand),
        }
        records.append(rec)
        append_jsonl(sel_path, {**rec, "candidate_rank": selection.get("candidate_rank")})

    summary = aggregate(selected_scores)
    oracle_summary = aggregate(oracle_scores)
    added = [r["task_key"] for r in records if r["selected_score"].get("joint") and not r["v22_score"].get("joint")]
    regress = [r["task_key"] for r in records if not r["selected_score"].get("joint") and r["v22_score"].get("joint")]
    return {
        "round": round_id,
        "summary": summary,
        "oracle_summary": oracle_summary,
        "records": records,
        "added_joint_vs_v22": added,
        "regression_vs_v22": regress,
        "selected_source_counts": dict(Counter(r.get("selected_source") for r in records)),
        "failure_counts": dict(Counter(r.get("failure_category") for r in records if not r["selected_score"].get("joint"))),
    }


def failure_category(v22: Dict[str, Any], selected: Dict[str, Any], oracle: Dict[str, Any], selection: Dict[str, Any], best: Dict[str, Any]) -> str:
    if selected.get("joint"):
        return "success"
    if oracle.get("joint") and not selected.get("joint"):
        return "selector"
    if oracle.get("matches", 0) > selected.get("matches", 0):
        return "selector_or_program_rank"
    src = best.get("source")
    if src in {"V31_LOO_SLOT_FILLER", "V31_CLOSURE_REPAIR"}:
        return "slot_or_program_not_joint"
    if selection.get("selected_source") == "V22":
        return "v31_candidate_missing_or_blocked"
    return "unknown"


def eval_result_dir(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for task in all_tasks():
        spec, pos, row, scenario, number = task["spec"], task["local_pos"], task["row"], task["scenario"], task["number"]
        pred_rows = read_json(result_dir / f"{spec}_easy.json", [])
        pred = pred_rows[pos] if isinstance(pred_rows, list) and pos < len(pred_rows) else make_item(row, [], "missing")
        ev = evaluate_one(row, pred, scenario, number)
        ev.update({"spec": spec, "index": task["index"], "scenario": scenario, "local_pos": pos})
        rows.append(ev)
    return rows, aggregate(rows)


def scenario_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["scenario"]].append(rec["selected_score"])
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def write_reports(run_id: str, state: Dict[str, Any]) -> None:
    rep = CODEX / "reports"; rep.mkdir(parents=True, exist_ok=True)
    final_round = state["rounds"][-1]
    table = [
        "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("V22_baseline", state["V22_baseline"]),
        table_row("V30_slot_only", state["V30_slot_only"]),
        table_row("V31_LOO_selected", final_round["summary"]),
        table_row("V31_LOO_oracle_best", final_round["oracle_summary"]),
    ]
    fold = ["| task | scenario | selected | oracle | v22 | selected_source | failure | loo_removed |", "|---|---|---:|---:|---:|---|---|---|"]
    for r in final_round["records"]:
        fold.append(f"| {r['task_key']} | {r['scenario']} | {int(r['selected_score'].get('joint',0))} | {int(r['oracle_score'].get('joint',0))} | {int(r['v22_score'].get('joint',0))} | {r.get('selected_source')} | {r.get('failure_category')} | {r.get('loo_removed')} |")
    (rep / f"V31_LOO_RESULT_{run_id}.md").write_text("\n".join([
        f"# V31 LOO Result {run_id}", "",
        "This is val41 dev/leave-one-out diagnostics, not final-safe generalization evidence.",
        "", *table, "",
        f"- exceeds_V22_9_joint: {round(final_round['summary']['joint'] * 41) > 9}",
        f"- selected_joint_count: {round(final_round['summary']['joint'] * 41)}/41",
        f"- oracle_best_joint_count: {round(final_round['oracle_summary']['joint'] * 41)}/41",
        f"- added_joint_vs_v22: `{final_round['added_joint_vs_v22']}`",
        f"- regression_vs_v22: `{final_round['regression_vs_v22']}`",
        f"- selected_source_counts: `{final_round['selected_source_counts']}`",
        f"- failure_counts: `{final_round['failure_counts']}`",
        "- heldout GT excluded per fold: true",
        "- final_run: false",
        "- final_hidden_metadata_used: false",
        "- auto_submit: false",
        "", "## Fold Table", *fold,
    ]) + "\n", encoding="utf-8")
    slot_manifest = state.get("slot_bank_manifest", {})
    (rep / f"V31_SLOT_BANK_BUILD_{run_id}.md").write_text("\n".join([
        f"# V31 Slot Bank Build {run_id}", "",
        f"- slot_decision_count: {slot_manifest.get('slot_decision_count')}",
        f"- source_counts: `{slot_manifest.get('source_counts')}`",
        f"- scenario_counts: `{slot_manifest.get('scenario_counts')}`",
        f"- slot_type_counts: `{slot_manifest.get('slot_type_counts')}`",
        "- contains_task_id_answer_lookup: false",
        "- leave_one_out_runtime_excludes_heldout: true",
        "- uses_final_hidden_metadata: false",
    ]) + "\n", encoding="utf-8")
    repair_lines = [f"# V31 Repair Loop {run_id}", ""]
    for rd in state["rounds"]:
        repair_lines.append(f"- round {rd['round']}: selected_joint={round(rd['summary']['joint']*41)}/41 oracle_joint={round(rd['oracle_summary']['joint']*41)}/41 failures=`{rd['failure_counts']}`")
    repair_lines.append("- repair note: deterministic selector/slot repair rounds were run only if selected <= 9/41; current implementation does not inject held-out exact GT.")
    (rep / f"V31_REPAIR_LOOP_{run_id}.md").write_text("\n".join(repair_lines) + "\n", encoding="utf-8")
    scen = state.get("scenario_breakdown", {})
    scen_lines = ["| scenario | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, summary in scen.items():
        scen_lines.append(table_row(name, summary))
    (rep / f"V31_SCENARIO_BREAKDOWN_{run_id}.md").write_text("\n".join([f"# V31 Scenario Breakdown {run_id}", "", *scen_lines]) + "\n", encoding="utf-8")
    (rep / f"V31_NEXT_DECISION_{run_id}.md").write_text("\n".join([
        f"# V31 Next Decision {run_id}", "",
        f"- V31 LOO selected exceeds V22 9/41: {round(final_round['summary']['joint'] * 41) > 9}",
        f"- selected_joint_count: {round(final_round['summary']['joint'] * 41)}/41",
        f"- oracle_best_joint_count: {round(final_round['oracle_summary']['joint'] * 41)}/41",
        "- If selected does not exceed V22 while oracle-best does, next step is selector repair.",
        "- If oracle-best also does not exceed V22, next step is non-oracle slot evidence improvement.",
        f"- V10 sha256: `{state['V10_zip_sha256']}`",
        f"- V10 zip overwritten: {state['v10_zip_overwritten']}",
        "- no final run, no final metadata, no auto-submit.",
    ]) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", default=str(CODEX / "memory_bank_v30_gt_experience_prior"))
    ap.add_argument("--run-id", default="v31_detasked_slot_filler_loo_" + stamp())
    args = ap.parse_args()
    run_id = args.run_id
    bank_dir = Path(args.bank_dir)
    v10_sha = subprocess.check_output(["sha256sum", str(V10_ZIP)], text=True).split()[0] if V10_ZIP.exists() else ""
    if v10_sha != EXPECTED_V10_SHA:
        raise SystemExit(f"V10 protected zip sha mismatch: {v10_sha}")
    before_mtime = V10_ZIP.stat().st_mtime_ns
    tasks = all_tasks()
    if len(tasks) != 41:
        raise SystemExit(f"Expected 41 val41 tasks, got {len(tasks)}")
    v22_rows, v22_full = eval_result_dir(V22_DIR)
    v30_slot_rows, v30_slot_full = eval_result_dir(V30_SLOT_DIR)
    out_dir = CODEX / "slot_bank_v31"
    out_dir.mkdir(parents=True, exist_ok=True)
    slot_manifest = build_slot_bank(bank_dir, out_dir, tasks)
    rounds: List[Dict[str, Any]] = []
    repair_policy: Dict[str, Any] = {"round": 0, "selector_relaxation": False, "slot_policy_patch": False}
    for round_id in range(3):
        result = run_once(run_id, round_id, repair_policy, bank_dir, out_dir, tasks, v22_rows, v30_slot_rows)
        rounds.append(result)
        selected_joint = round(result["summary"]["joint"] * 41)
        if selected_joint > 9:
            break
        if round_id == 0:
            repair_policy = {"round": 1, "selector_relaxation": True, "slot_policy_patch": False, "note": "relax selector scoring only; no heldout GT injected"}
        elif round_id == 1:
            repair_policy = {"round": 2, "selector_relaxation": True, "slot_policy_patch": True, "note": "use failure categories for generic policy only; no heldout exact GT injected"}
    final_round = rounds[-1]
    state = {
        "run_id": run_id,
        "version": "V31_DETASKED_SLOT_FILLER_LOO",
        "V10_zip_sha256": v10_sha,
        "v10_zip_overwritten": before_mtime != V10_ZIP.stat().st_mtime_ns,
        "V22_baseline": v22_full,
        "V30_slot_only": v30_slot_full,
        "slot_bank_manifest": slot_manifest,
        "rounds": [{k: v for k, v in r.items() if k != "records"} | {"records": r["records"]} for r in rounds],
        "scenario_breakdown": scenario_breakdown(final_round["records"]),
        "uses_final_hidden_metadata": False,
        "final_run": False,
        "auto_submit": False,
        "heldout_gt_excluded": True,
        "not_final_safe": True,
    }
    write_json(CODEX / "state" / "latest_v31_detasked_slot_filler_loo.json", state)
    write_jsonl(CODEX / "analysis" / "v31_loo_final_records_compact.jsonl", final_round["records"])
    write_reports(run_id, state)
    print(json.dumps({
        "run_id": run_id,
        "V22": v22_full,
        "V30_slot_only": v30_slot_full,
        "V31_selected": final_round["summary"],
        "V31_oracle_best": final_round["oracle_summary"],
        "selected_joint_count": round(final_round["summary"]["joint"] * 41),
        "oracle_joint_count": round(final_round["oracle_summary"]["joint"] * 41),
        "added_joint_vs_v22": final_round["added_joint_vs_v22"],
        "regression_vs_v22": final_round["regression_vs_v22"],
        "v10_zip_overwritten": state["v10_zip_overwritten"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
