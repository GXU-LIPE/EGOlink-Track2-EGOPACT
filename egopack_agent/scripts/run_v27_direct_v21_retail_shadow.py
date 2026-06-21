#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V27 direct RetailResolverV21 retail shadow on frozen val41.

Only retail tasks get V27 candidates. Non-retail tasks remain V22 fallback.
Val41 GT is used only by the post-eval scorer, never for runtime selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V22_DIR = EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
EVIDENCE_PATHS = [
    CODEX / "analysis" / "v26_mm_evidence_val41.jsonl",
    CODEX / "analysis" / "v25_new_mm_evidence.jsonl",
]

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))

from egobench_agent_plus.v24_candidate_dryrun_and_selector import dryrun_program  # noqa: E402
from egobench_agent_plus.v27_v25_to_v21_adapter import build_v27_v21_candidate  # noqa: E402


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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_specs() -> List[Tuple[str, int, List[int]]]:
    manifest = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in manifest.get("specs", [])]


def init_db(scenario: str, number: int) -> Any:
    sys.path.insert(0, str(EGO))
    if scenario == "retail":
        from tools.retail.retail_db import RetailDB
        from tools.retail import retail_init

        db = RetailDB()
        db.init_from_json(getattr(retail_init, f"retail_init_data{number}"))
        return db
    if scenario == "restaurant":
        from tools.restaurant.restaurant_db import RestaurantDB
        from tools.restaurant import restaurant_init

        db = RestaurantDB()
        db.init_from_json(getattr(restaurant_init, f"restaurant_init_data{number}", None) or getattr(restaurant_init, "restaurant_init_data"))
        return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init

        db = OrderDB()
        db.init_from_json(getattr(order_init, f"order_init_data{number}", None) or getattr(order_init, "order_init_data"))
        return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init

        db = KitchenDB()
        db.init_from_json(getattr(kitchen_init, f"kitchen_init_data{number}", None) or getattr(kitchen_init, "kitchen_init_data"))
        return db
    raise ValueError(scenario)


def qwen_card(spec: str, pos: int) -> Dict[str, Any]:
    for path in [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_143308" / f"{spec}_{pos + 1}.json",
    ]:
        data = read_json(path)
        if isinstance(data, dict):
            data["_path"] = str(path)
            return data
    return {"status": "missing", "_path": ""}


def load_evidence_index() -> Dict[Tuple[str, int], Dict[str, Any]]:
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in EVIDENCE_PATHS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            spec = str(row.get("spec") or "")
            idx = row.get("index")
            if spec and idx is not None:
                key = (spec, int(idx))
                # Prefer full V26 val41 evidence over V25 smoke rows.
                index.setdefault(key, row)
    return index


def all_tasks() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        for pos, row in enumerate(rows):
            out.append({"scenario": scenario, "number": number, "spec": spec, "local_pos": pos, "index": int(row.get("_v8_original_index", pos)), "row": row})
    return out


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        for call in block.get("calls") or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": copy.deepcopy(call.get("parameters") or {})})
    return out


def make_item(row: Dict[str, Any], program: List[Dict[str, Any]], label: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} val41 shadow candidate."}],
        "tool_calls": [
            {
                "turn": 0,
                "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program],
                "blocked_calls": [],
                "results": [],
                "v27_meta": meta or {},
            }
        ],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
    }


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    rows = read_json(result_dir / f"{spec}_easy.json", [])
    return rows[pos] if isinstance(rows, list) and pos < len(rows) else None


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v27_eval_") as td:
        gt_path = Path(td) / "gt.json"
        pred_path = Path(td) / "pred.json"
        write_json(gt_path, [gt_item])
        write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(str(gt_path), str(pred_path), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
    detail = (metrics.get("detailed_results") or [{}])[0]
    tb = detail.get("tool_based") or {}
    rb = detail.get("result_based") or {}
    micro = metrics.get("micro_tool_stats") or {}
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
    if not valid:
        return {"valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "matched_tools": 0, "gt_tools": 0, "interaction_calls": 0}
    matched = sum(int(r.get("matches", 0) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", 0) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(float(r.get("joint", 0)) for r in rows) / valid,
        "result": sum(float(r.get("result", 0)) for r in rows) / valid,
        "tool": sum(float(r.get("tool", 0)) for r in rows) / valid,
        "micro": matched / gt if gt else 0.0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (float(score.get("joint", 0)), float(score.get("tool", 0)), float(score.get("result", 0)), int(score.get("matches", 0)), -int(score.get("interaction_calls", 999999)))


def trace_flags_ok(trace: Dict[str, Any]) -> bool:
    return all(trace.get(k) is True for k in ("called_RetailResolverV21", "called_attribute_query_planner", "called_observation_brancher", "called_add_target_resolver"))


def v21_candidate_allowed(candidate: Dict[str, Any], trace: Dict[str, Any], dry: Dict[str, Any], row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not trace_flags_ok(trace):
        reasons.append("v21_trace_flags_missing")
    v21_trace = trace.get("v21_trace") or {}
    if not v21_trace.get("primary_product_candidates"):
        reasons.append("primary_product_candidates_empty")
    if not v21_trace.get("mutation_target"):
        reasons.append("mutation_target_empty")
    names = [x.get("tool_name") for x in candidate.get("tool_program") or []]
    if any(n == "add_to_cart" for n in names) and not any(str(n).startswith(("get_", "find_")) for n in names[:3]):
        # Some compact V21 paths intentionally trim branch queries for known
        # exact-trajectory retail cases. Allow them when branch trace exists.
        if not v21_trace.get("branch_decision"):
            reasons.append("missing_product_query_before_mutation")
    if dry.get("errors"):
        reasons.append("dryrun_errors")
    if dry.get("broad_scan"):
        reasons.append("leading_broad_scan")
    if dry.get("closure_required") and not dry.get("closure_complete"):
        reasons.append("closure_incomplete")
    return (not reasons), reasons


def write_result_dir(result_dir: Path, item_by_key: Dict[Tuple[str, int], Dict[str, Any]], fallback_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        base = read_json(fallback_dir / f"{spec}_easy.json", [])
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        out = []
        for pos, row in enumerate(rows):
            idx = int(row.get("_v8_original_index", pos))
            out.append(item_by_key.get((spec, idx)) or (base[pos] if isinstance(base, list) and pos < len(base) else make_item(row, [], "missing_base")))
        write_json(result_dir / f"{spec}_easy.json", out)


def eval_result_dir(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        gt_rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_rows = read_json(result_dir / f"{spec}_easy.json", [])
        for pos, row in enumerate(gt_rows):
            pred = pred_rows[pos] if isinstance(pred_rows, list) and pos < len(pred_rows) else make_item(row, [], "missing")
            ev = evaluate_one(row, pred, scenario, number)
            ev.update({"spec": spec, "index": int(row.get("_v8_original_index", pos)), "scenario": scenario, "local_pos": pos})
            rows.append(ev)
    return rows, aggregate(rows)


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table = [
        "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("V22_baseline_reference", state["V22_baseline_reference"]),
        table_row("V27_direct_v21_retail_shadow", state["V27_direct_v21_retail_shadow"]),
        table_row("V27_direct_v21_plus_v25evidence_retail_shadow", state["V27_direct_v21_plus_v25evidence_retail_shadow"]),
    ]
    retail = state["retail_only"]
    evidence_better = state["V27_direct_v21_plus_v25evidence_retail_shadow"]["joint"] > state["V27_direct_v21_retail_shadow"]["joint"]
    (reports / f"V27_RETAIL_FULL_SHADOW_RESULT_{run_id}.md").write_text(
        "\n".join(
            [
                f"# V27 Retail Full Shadow Result {run_id}",
                "",
                *table,
                "",
                "## Retail Only",
                "",
                *[
                    "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                    table_row("V22_retail", retail["V22"]),
                    table_row("Direct_V21_retail", retail["direct"]),
                    table_row("Evidence_V21_retail", retail["evidence"]),
                ],
                "",
                f"- V21 selected count: {state['selection_counts']['direct_selected']}",
                f"- V21 evidence selected count: {state['selection_counts']['evidence_selected']}",
                f"- added_joint_direct_vs_v22: {state['added_joint_direct_vs_v22']}",
                f"- added_joint_evidence_vs_v22: {state['added_joint_evidence_vs_v22']}",
                f"- regressions_direct_vs_v22: {state['regressions_direct_vs_v22']}",
                f"- regressions_evidence_vs_v22: {state['regressions_evidence_vs_v22']}",
                f"- final_run: false",
                f"- final_hidden_metadata_used: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (reports / f"V27_V25_EVIDENCE_ABLATION_{run_id}.md").write_text(
        "\n".join(
            [
                f"# V27 V25 Evidence Ablation {run_id}",
                "",
                f"- direct V21 full joint: {state['V27_direct_v21_retail_shadow']['joint']*100:.2f}%",
                f"- evidence V21 full joint: {state['V27_direct_v21_plus_v25evidence_retail_shadow']['joint']*100:.2f}%",
                f"- evidence improves direct V21: {evidence_better}",
                f"- evidence records found for retail tasks: {state['evidence_records_retail']}/{state['retail_task_count']}",
                f"- conclusion: {'V25 evidence enhanced V21' if evidence_better else 'V25 evidence did not enhance V21 on joint'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chain = [f"# V27 Single Chain Failures {run_id}", "", "| spec | index | direct_joint | evidence_joint | v22_joint | first_failure |", "|---|---:|---:|---:|---:|---|"]
    focus = {("retail2", 5), ("restaurant3", 24), ("restaurant3", 54), ("kitchen1", 31)}
    for rec in records:
        if (rec["spec"], rec["index"]) in focus or rec["scenario"] == "retail":
            chain.append(f"| {rec['spec']} | {rec['index']} | {int(rec['direct_score']['joint'])} | {int(rec['evidence_score']['joint'])} | {int(rec['v22_score']['joint'])} | {rec['first_failure']} |")
    (reports / f"V27_SINGLE_CHAIN_FAILURES_{run_id}.md").write_text("\n".join(chain) + "\n", encoding="utf-8")
    decision = "direct_v21_or_evidence_added_joint" if state["added_joint_direct_vs_v22"] or state["added_joint_evidence_vs_v22"] else "no_new_joint_retail_direction_exhausted_or_inputs_still_wrong"
    (reports / f"V27_NEXT_DECISION_{run_id}.md").write_text(
        "\n".join(
            [
                f"# V27 Next Decision {run_id}",
                "",
                *table,
                "",
                f"- decision: {decision}",
                f"- true import and call of RetailResolverV21: {state['all_runtime_wiring_ok']}",
                f"- retail runtime trace all true: {state['all_runtime_wiring_ok']}",
                f"- direct V21 exceeds V22: {state['V27_direct_v21_retail_shadow']['joint'] > state['V22_baseline_reference']['joint']}",
                f"- evidence V21 exceeds direct V21: {evidence_better}",
                f"- selected new joint tasks direct: {state['added_joint_direct_vs_v22']}",
                f"- selected new joint tasks evidence: {state['added_joint_evidence_vs_v22']}",
                f"- no final: true",
                f"- no final hidden metadata: true",
                f"- V10 zip overwritten: {state['v10_zip_overwritten']}",
                f"- auto_submit: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v27_direct_v21_" + stamp())
    args = ap.parse_args()
    run_id = args.run_id
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    v10_sha = subprocess.check_output(["sha256sum", str(V10_ZIP)], text=True).split()[0] if V10_ZIP.exists() else ""
    evidence_index = load_evidence_index()
    tasks = all_tasks()
    trace_path = CODEX / "analysis" / "v27_direct_v21_retail_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    item_direct: Dict[Tuple[str, int], Dict[str, Any]] = {}
    item_evidence: Dict[Tuple[str, int], Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    all_wiring_ok = True
    for i, task in enumerate(tasks, 1):
        spec, pos, scenario, number, idx, row = task["spec"], task["local_pos"], task["scenario"], task["number"], task["index"], task["row"]
        v22_item = load_item(V22_DIR, spec, pos) or make_item(row, [], "missing_v22")
        v22_score = evaluate_one(row, v22_item, scenario, number)
        if scenario != "retail":
            item_direct[(spec, idx)] = v22_item
            item_evidence[(spec, idx)] = v22_item
            continue
        db = init_db(scenario, number)
        evidence = evidence_index.get((spec, idx), {})
        qwen = qwen_card(spec, pos)
        direct_obj = build_v27_v21_candidate(db, row, evidence=evidence, qwen_card=qwen, use_evidence=False)
        evidence_obj = build_v27_v21_candidate(db, row, evidence=evidence, qwen_card=qwen, use_evidence=True)
        candidates = []
        for label, obj in [("direct", direct_obj), ("evidence", evidence_obj)]:
            cand = copy.deepcopy(obj.get("candidate") or {})
            prog = cand.get("tool_program") or []
            dry = dryrun_program("retail", init_db(scenario, number), prog, row.get("Instruction", ""))
            allowed, block_reasons = v21_candidate_allowed(cand, obj.get("trace") or {}, dry, row)
            score = evaluate_one(row, make_item(row, prog, cand.get("candidate_id", label), {"source": cand.get("source"), "trace": obj.get("trace")}), scenario, number)
            candidates.append({"label": label, "obj": obj, "candidate": cand, "dryrun": dry, "allowed": allowed, "block_reasons": block_reasons, "score": score})
            all_wiring_ok = all_wiring_ok and trace_flags_ok(obj.get("trace") or {})
        # Runtime-style selection: only replace V22 when the V21 candidate
        # passes non-oracle guards. Post-eval scores below are diagnostics and
        # must not decide the selected program.
        direct_choice = candidates[0]
        evidence_choice = candidates[1]
        direct_selected = direct_choice["allowed"]
        evidence_selected = evidence_choice["allowed"]
        item_direct[(spec, idx)] = make_item(row, direct_choice["candidate"].get("tool_program") or [], "V27_DIRECT_V21_SELECTED", {"selected_v21": direct_selected, "trace": direct_choice["obj"].get("trace"), "dryrun": direct_choice["dryrun"]}) if direct_selected else v22_item
        item_evidence[(spec, idx)] = make_item(row, evidence_choice["candidate"].get("tool_program") or [], "V27_EVIDENCE_V21_SELECTED", {"selected_v21": evidence_selected, "trace": evidence_choice["obj"].get("trace"), "dryrun": evidence_choice["dryrun"]}) if evidence_selected else v22_item
        rec = {
            "task_key": f"{spec}::{idx}",
            "spec": spec,
            "index": idx,
            "local_pos": pos,
            "scenario": scenario,
            "evidence_found": bool(evidence),
            "direct_v21_trace": direct_obj.get("trace"),
            "evidence_v21_trace": evidence_obj.get("trace"),
            "direct_score": direct_choice["score"],
            "evidence_score": evidence_choice["score"],
            "v22_score": v22_score,
            "direct_allowed": direct_choice["allowed"],
            "evidence_allowed": evidence_choice["allowed"],
            "direct_selected": direct_selected,
            "evidence_selected": evidence_selected,
            "direct_block_reasons": direct_choice["block_reasons"],
            "evidence_block_reasons": evidence_choice["block_reasons"],
        }
        if evidence_choice["score"]["joint"] or direct_choice["score"]["joint"] or v22_score["joint"]:
            rec["first_failure"] = "resolved"
        elif not evidence:
            rec["first_failure"] = "evidence_missing"
        elif not (evidence_obj.get("trace") or {}).get("v21_input_context", {}).get("top_k_candidates"):
            rec["first_failure"] = "canonical_slot_empty"
        elif (evidence_obj.get("trace") or {}).get("v21_trace", {}).get("mutation_target") == []:
            rec["first_failure"] = "mutation_target_wrong"
        else:
            rec["first_failure"] = "branch_or_tool_shape_wrong"
        records.append(rec)
        append_jsonl(trace_path, rec)
        if i % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] processed {i}/{len(tasks)}")
    direct_dir = EGO / "results" / f"V27_direct_v21_retail_shadow-{run_id}"
    evidence_dir = EGO / "results" / f"V27_direct_v21_plus_v25evidence_retail_shadow-{run_id}"
    write_result_dir(direct_dir, item_direct, V22_DIR)
    write_result_dir(evidence_dir, item_evidence, V22_DIR)
    v22_rows, v22_full = eval_result_dir(V22_DIR)
    direct_rows, direct_full = eval_result_dir(direct_dir)
    evidence_rows, evidence_full = eval_result_dir(evidence_dir)
    retail_v22 = aggregate([r for r in v22_rows if r["scenario"] == "retail"])
    retail_direct = aggregate([r for r in direct_rows if r["scenario"] == "retail"])
    retail_evidence = aggregate([r for r in evidence_rows if r["scenario"] == "retail"])
    v22_by_key = {(r["spec"], r["index"]): r for r in v22_rows}
    direct_by_key = {(r["spec"], r["index"]): r for r in direct_rows}
    evidence_by_key = {(r["spec"], r["index"]): r for r in evidence_rows}
    added_direct = [f"{spec}::{idx}" for (spec, idx), r in direct_by_key.items() if r["joint"] and not v22_by_key[(spec, idx)]["joint"]]
    added_evidence = [f"{spec}::{idx}" for (spec, idx), r in evidence_by_key.items() if r["joint"] and not v22_by_key[(spec, idx)]["joint"]]
    reg_direct = [f"{spec}::{idx}" for (spec, idx), r in direct_by_key.items() if not r["joint"] and v22_by_key[(spec, idx)]["joint"]]
    reg_evidence = [f"{spec}::{idx}" for (spec, idx), r in evidence_by_key.items() if not r["joint"] and v22_by_key[(spec, idx)]["joint"]]
    state = {
        "run_id": run_id,
        "version": "V27_DIRECT_V21_WIRING_AND_MM_EVIDENCE_BRIDGE",
        "V10_zip_sha256": v10_sha,
        "V10_zip_mtime_before": before_mtime,
        "V22_baseline_reference": v22_full,
        "V27_direct_v21_retail_shadow": direct_full,
        "V27_direct_v21_plus_v25evidence_retail_shadow": evidence_full,
        "retail_only": {"V22": retail_v22, "direct": retail_direct, "evidence": retail_evidence},
        "direct_result_dir": str(direct_dir),
        "evidence_result_dir": str(evidence_dir),
        "trace_path": str(trace_path),
        "selection_counts": {"direct_selected": sum(1 for r in records if r["direct_selected"]), "evidence_selected": sum(1 for r in records if r["evidence_selected"])},
        "added_joint_direct_vs_v22": added_direct,
        "added_joint_evidence_vs_v22": added_evidence,
        "regressions_direct_vs_v22": reg_direct,
        "regressions_evidence_vs_v22": reg_evidence,
        "retail_task_count": sum(1 for t in tasks if t["scenario"] == "retail"),
        "evidence_records_retail": sum(1 for r in records if r["evidence_found"]),
        "all_runtime_wiring_ok": all_wiring_ok,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_runtime": False,
        "v10_zip_overwritten": before_mtime != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }
    state_path = CODEX / "state" / "latest_v27_direct_v21_retail_shadow.json"
    write_json(state_path, state)
    write_jsonl(CODEX / "analysis" / "v27_direct_v21_retail_records_compact.jsonl", records)
    write_reports(run_id, state, records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
