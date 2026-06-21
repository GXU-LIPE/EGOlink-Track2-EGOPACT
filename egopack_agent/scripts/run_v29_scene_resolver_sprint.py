#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V29 scene resolver sprint on frozen val41.

V29 is a dev/val41 optimisation experiment. Round 0 uses non-GT five-stage
resolvers. If it fails to beat V22, repair rounds may add val41-GT-informed
repair candidates and select them with post-eval diagnostics. This is not a
final-safe procedure and is reported as such.
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
V28_DIR = EGO / "results" / "V28_protected_delta_evidence_veto_val41_selected-v28_protected_delta_guarded_20260621_1115"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
EVIDENCE_PATHS = [CODEX / "analysis" / "v26_mm_evidence_val41.jsonl", CODEX / "analysis" / "v25_new_mm_evidence.jsonl"]

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))

from egobench_agent_plus.v24_candidate_dryrun_and_selector import dryrun_program  # noqa: E402
from egobench_agent_plus.v29_gap_miner import classify_gap, summarize_gaps  # noqa: E402
from egobench_agent_plus.v29_kitchen_five_stage_resolver import KitchenFiveStageResolverV29  # noqa: E402
from egobench_agent_plus.v29_order_five_stage_resolver import OrderFiveStageResolverV29  # noqa: E402
from egobench_agent_plus.v29_protected_selector import ProtectedSelectorV29  # noqa: E402
from egobench_agent_plus.v29_repair_loop import RepairLoopV29  # noqa: E402
from egobench_agent_plus.v29_restaurant_five_stage_resolver import RestaurantFiveStageResolverV29  # noqa: E402
from egobench_agent_plus.v29_retail_guarded_overlay import RetailGuardedOverlayV29  # noqa: E402


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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


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
            spec, idx = str(row.get("spec") or ""), row.get("index")
            if not spec and row.get("task_key"):
                spec, idx_s = str(row["task_key"]).split("::", 1)
                idx = int(idx_s)
            if spec and idx is not None:
                index.setdefault((spec, int(idx)), row)
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
        "tool_calls": [{"turn": 0, "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program], "blocked_calls": [], "results": [], "v29_meta": meta or {}}],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
    }


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    rows = read_json(result_dir / f"{spec}_easy.json", [])
    return rows[pos] if isinstance(rows, list) and pos < len(rows) else None


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v29_eval_") as td:
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
    matched = sum(int(r.get("matches", 0) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", 0) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(float(r.get("joint", 0)) for r in rows) / valid if valid else 0,
        "result": sum(float(r.get("result", 0)) for r in rows) / valid if valid else 0,
        "tool": sum(float(r.get("tool", 0)) for r in rows) / valid if valid else 0,
        "micro": matched / gt if gt else 0.0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (float(score.get("joint", 0)), float(score.get("tool", 0)), float(score.get("result", 0)), int(score.get("matches", 0)), -int(score.get("interaction_calls", 999999)))


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


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


def build_candidates(task: Dict[str, Any], evidence: Dict[str, Any], repair_level: int) -> List[Dict[str, Any]]:
    scenario, number, row = task["scenario"], task["number"], task["row"]
    db = init_db(scenario, number)
    if scenario == "retail":
        return RetailGuardedOverlayV29(db, qwen_card(task["spec"], task["local_pos"])).build(row, evidence, repair_level=repair_level, max_candidates=4)
    if scenario == "order":
        return OrderFiveStageResolverV29(db).build(row, evidence, repair_level=repair_level, max_candidates=4)
    if scenario == "restaurant":
        return RestaurantFiveStageResolverV29(db).build(row, evidence, repair_level=repair_level, max_candidates=4)
    if scenario == "kitchen":
        return KitchenFiveStageResolverV29(db).build(row, evidence, repair_level=repair_level, max_candidates=4)
    return []


def run_round(run_id: str, round_id: int, repair_level: int, evidence_index: Dict[Tuple[str, int], Dict[str, Any]], v22_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    selector = ProtectedSelectorV29()
    selected_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    v22_by_key = {(r["spec"], r["index"]): r for r in v22_rows}
    trace_paths = {
        "order": CODEX / "analysis" / "v29_order_trace.jsonl",
        "restaurant": CODEX / "analysis" / "v29_restaurant_trace.jsonl",
        "kitchen": CODEX / "analysis" / "v29_kitchen_trace.jsonl",
        "retail": CODEX / "analysis" / "v29_retail_trace.jsonl",
        "candidates": CODEX / "analysis" / "v29_candidate_programs.jsonl",
        "selection": CODEX / "analysis" / "v29_selection_trace.jsonl",
    }
    if round_id == 0:
        for p in trace_paths.values():
            p.write_text("", encoding="utf-8")

    for i, task in enumerate(all_tasks(), 1):
        spec, idx, pos, scenario, number, row = task["spec"], task["index"], task["local_pos"], task["scenario"], task["number"], task["row"]
        task_key = f"{spec}::{idx}"
        v22_item = load_item(V22_DIR, spec, pos) or make_item(row, [], "missing_v22")
        v22_score = v22_by_key.get((spec, idx)) or evaluate_one(row, v22_item, scenario, number)
        candidates: List[Dict[str, Any]] = []
        if not v22_score.get("joint"):
            candidates = build_candidates(task, evidence_index.get((spec, idx), {}), repair_level)
            for cand in candidates:
                db = init_db(scenario, number)
                dry = dryrun_program(scenario, db, cand.get("tool_program") or [], row.get("Instruction", ""))
                cand["dryrun"] = dry
                item = make_item(row, cand.get("tool_program") or [], cand.get("candidate_id", "cand"), {"round": round_id, "repair_level": repair_level})
                cand["post_eval_score"] = evaluate_one(row, item, scenario, number)
                append_jsonl(trace_paths["candidates"], {"round": round_id, "task_key": task_key, "candidate": cand})
                if scenario in trace_paths:
                    append_jsonl(trace_paths[scenario], {"round": round_id, "task_key": task_key, "trace": cand.get("trace"), "tool_names": [x.get("tool_name") for x in cand.get("tool_program") or []]})
        allow_dev_gt = repair_level > 0
        selection = selector.select(task_key, scenario, v22_item, v22_score, candidates, allow_dev_gt_selection=allow_dev_gt)
        selected = selection["selected"]
        if isinstance(selected, dict) and selected.get("tool_program") is not None:
            selected_item = make_item(row, selected.get("tool_program") or [], selected.get("candidate_id", "V29_SELECTED"), {"selection": selection, "round": round_id, "repair_level": repair_level})
        else:
            selected_item = v22_item
        selected_score = evaluate_one(row, selected_item, scenario, number)
        scored_candidates = [(cand, cand.get("post_eval_score") or {}, make_item(row, cand.get("tool_program") or [], cand.get("candidate_id", "cand"), {"oracle_best_post_eval": True})) for cand in candidates]
        if scored_candidates:
            best_cand, best_score, best_item = max(scored_candidates + [({"candidate_id": "V22"}, v22_score, v22_item)], key=lambda x: score_tuple(x[1]))
        else:
            best_cand, best_score, best_item = {"candidate_id": "V22"}, v22_score, v22_item
        selected_items[(spec, idx)] = selected_item
        oracle_items[(spec, idx)] = best_item
        rec = {
            "round": round_id,
            "task_key": task_key,
            "spec": spec,
            "index": idx,
            "scenario": scenario,
            "v22_score": v22_score,
            "selected_score": selected_score,
            "oracle_score": best_score,
            "oracle_candidate": best_cand.get("candidate_id"),
            "selection": selection,
            "candidate_count": len(candidates),
            "uses_val41_gt_for_repair": any((c.get("trace") or {}).get("uses_val41_gt_for_repair") for c in candidates),
        }
        records.append(rec)
        append_jsonl(trace_paths["selection"], {"round": round_id, "task_key": task_key, "selected_candidate_id": selection.get("selected_candidate_id"), "selected_source": selection.get("selected_source"), "selection_reason": selection.get("reason"), "selected_score": selected_score, "v22_score": v22_score, "oracle_score": best_score, "uses_val41_gt_for_selection": selection.get("uses_val41_gt_for_selection", False)})
        if i % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] round {round_id} processed {i}/41")

    selected_dir = EGO / "results" / f"V29_scene_resolver_sprint_val41_selected_r{round_id}-{run_id}"
    oracle_dir = EGO / "results" / f"V29_scene_resolver_sprint_val41_oracle_best_r{round_id}-{run_id}"
    write_result_dir(selected_dir, selected_items, V22_DIR)
    write_result_dir(oracle_dir, oracle_items, V22_DIR)
    selected_rows, selected_full = eval_result_dir(selected_dir)
    oracle_rows, oracle_full = eval_result_dir(oracle_dir)
    selected_by_key = {(r["spec"], r["index"]): r for r in selected_rows}
    added = [f"{k[0]}::{k[1]}" for k, r in selected_by_key.items() if r["joint"] and not v22_by_key[k]["joint"]]
    regression = [f"{k[0]}::{k[1]}" for k, r in selected_by_key.items() if not r["joint"] and v22_by_key[k]["joint"]]
    per = {}
    for spec in sorted({r["spec"] for r in selected_rows}):
        per[spec] = {"selected": aggregate([r for r in selected_rows if r["spec"] == spec]), "oracle": aggregate([r for r in oracle_rows if r["spec"] == spec])}
    return {
        "round": round_id,
        "repair_level": repair_level,
        "selected_dir": str(selected_dir),
        "oracle_dir": str(oracle_dir),
        "selected_rows": selected_rows,
        "oracle_rows": oracle_rows,
        "records": records,
        "summary": selected_full,
        "oracle_summary": oracle_full,
        "added_joint_vs_v22": added,
        "regression_vs_v22": regression,
        "per_scenario": per,
    }


def write_reports(run_id: str, state: Dict[str, Any], gap_summary: Dict[str, Any]) -> None:
    rep = CODEX / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    table = [
        "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("V22_baseline", state["V22_baseline"]),
        table_row("V29_round0_selected", state["rounds"][0]["summary"]),
        table_row("V29_final_selected", state["V29_selected_final"]),
        table_row("V29_final_oracle_best", state["V29_oracle_best_final"]),
    ]
    (rep / f"V29_PREFLIGHT_{run_id}.md").write_text("\n".join([
        f"# V29 Preflight {run_id}",
        "",
        f"- V10 zip: `{V10_ZIP}`",
        f"- V10 zip sha256: `{state['V10_zip_sha256']}`",
        f"- V10 zip overwritten: {state['v10_zip_overwritten']}",
        f"- V22 dir exists: {V22_DIR.exists()}",
        f"- V28 dir exists: {V28_DIR.exists()}",
        f"- val41 task count: {state['val41_task_count']}",
        f"- evidence records: {state['evidence_records']}",
        "- final_run: false",
        "- final_hidden_metadata_used: false",
    ]) + "\n", encoding="utf-8")
    lines = [f"# V29 GT Gap Mining {run_id}", "", "- uses_val41_gt_for_gap_mining: true", "- final_hidden_metadata_used: false", "", "## Scenario Counts", "", "| scenario | total | easy | medium | hard | dirty |", "|---|---:|---:|---:|---:|---:|"]
    for sc, row in sorted(gap_summary.get("by_scenario", {}).items()):
        lines.append(f"| {sc} | {row.get('total',0)} | {row.get('easy',0)} | {row.get('medium',0)} | {row.get('hard',0)} | {row.get('dirty',0)} |")
    lines += ["", "## Top 12 Repair Targets", ""]
    for row in gap_summary.get("top12", []):
        lines.append(f"- {row['task_key']}: {row['repairability']}, gt={row['gt_tools']}, missing={row['missing_prefix']}")
    (rep / f"V29_GT_GAP_MINING_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (rep / f"V29_RESOLVER_IMPLEMENTATION_{run_id}.md").write_text("\n".join([
        f"# V29 Resolver Implementation {run_id}",
        "",
        "- Implemented order/restaurant/kitchen resolver files with explicit five-stage functions.",
        "- Implemented retail guarded overlay around the real V21 resolver.",
        "- Runtime traces record entity resolver, query planner, observation brancher, mutation resolver, and closure planner.",
        "- Round 0 is non-GT resolver output.",
        "- Repair rounds may use val41 GT-informed candidates and are marked not final-safe.",
        "- No V18 oracle compiler imported or called.",
    ]) + "\n", encoding="utf-8")
    repair_lines = [f"# V29 Repair Loop {run_id}", "", *table, ""]
    for rr in state["rounds"]:
        joint_count = round(rr["summary"]["joint"] * rr["summary"]["valid"])
        repair_lines.append(f"- round {rr['round']}: repair_level={rr['repair_level']}, joint={joint_count}/41, added={rr['added_joint_vs_v22']}, regression={rr['regression_vs_v22']}")
    repair_lines += ["", f"- final_status: {state['repair_status']}", f"- uses_val41_gt_for_repair: {state['uses_val41_gt_for_repair']}", "- not_final_safe_if_gt_repair_used: true"]
    (rep / f"V29_REPAIR_LOOP_{run_id}.md").write_text("\n".join(repair_lines) + "\n", encoding="utf-8")
    result_lines = [f"# V29 Val41 Final Result {run_id}", "", *table, "", f"- selected_result_dir: `{state['selected_result_dir']}`", f"- oracle_result_dir: `{state['oracle_result_dir']}`", f"- added_joint_vs_v22: {state['added_joint_vs_v22']}", f"- regression_vs_v22: {state['regression_vs_v22']}", f"- selected_sources: `{state['selected_sources']}`", f"- uses_val41_gt_for_repair: {state['uses_val41_gt_for_repair']}", f"- final_safe: {not state['uses_val41_gt_for_repair']}", "- final_run: false", "- final_hidden_metadata_used: false", "- auto_submit: false"]
    (rep / f"V29_VAL41_FINAL_RESULT_{run_id}.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    sc_lines = [f"# V29 Scenario Breakdown {run_id}", "", "| spec | selected joint | oracle joint | selected micro | oracle micro |", "|---|---:|---:|---:|---:|"]
    for spec, row in state["per_scenario"].items():
        sc_lines.append(f"| {spec} | {row['selected']['joint']*100:.2f}% | {row['oracle']['joint']*100:.2f}% | {row['selected']['micro']:.4f} | {row['oracle']['micro']:.4f} |")
    (rep / f"V29_SCENARIO_BREAKDOWN_{run_id}.md").write_text("\n".join(sc_lines) + "\n", encoding="utf-8")
    decision = "do_not_promote_final_candidate_gt_informed" if state["uses_val41_gt_for_repair"] else ("candidate_exceeds_v22_non_gt" if state["V29_selected_final"]["joint"] > state["V22_baseline"]["joint"] else "failed_no_gain")
    (rep / f"V29_NEXT_DECISION_{run_id}.md").write_text("\n".join([
        f"# V29 Next Decision {run_id}",
        "",
        *table,
        "",
        f"- decision: {decision}",
        f"- exceeds_v22_9_of_41: {state['V29_joint_count'] > 9}",
        f"- reaches_15_of_41: {state['V29_joint_count'] >= 15}",
        f"- reaches_21_of_41: {state['V29_joint_count'] >= 21}",
        f"- V10 protected zip sha256 still expected: `{state['V10_zip_sha256']}`",
        "- no final run, no final hidden metadata, no V10 overwrite, no auto-submit.",
    ]) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v29_scene_resolver_sprint_" + stamp())
    args = ap.parse_args()
    run_id = args.run_id
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    v10_sha = subprocess.check_output(["sha256sum", str(V10_ZIP)], text=True).split()[0] if V10_ZIP.exists() else ""
    if v10_sha != "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d":
        raise SystemExit(f"V10 protected zip sha mismatch: {v10_sha}")
    tasks = all_tasks()
    if len(tasks) != 41:
        raise SystemExit(f"Expected 41 val tasks, got {len(tasks)}")
    evidence_index = load_evidence_index()
    v22_rows, v22_full = eval_result_dir(V22_DIR)
    v22_by_key = {(r["spec"], r["index"]): r for r in v22_rows}
    gaps = []
    for task in tasks:
        spec, idx, row = task["spec"], task["index"], task["row"]
        v22_item = load_item(V22_DIR, spec, task["local_pos"])
        gaps.append(classify_gap(f"{spec}::{idx}", task["scenario"], row, v22_by_key[(spec, idx)], program_from_item(v22_item)))
    write_jsonl(CODEX / "analysis" / "v29_gap_mining.jsonl", gaps)
    gap_summary = summarize_gaps(gaps)
    repair = RepairLoopV29(baseline_joint_count=9, max_rounds=4)
    rounds: List[Dict[str, Any]] = []
    round_id = 0
    while True:
        repair_level = repair.next_repair_level()
        rr = run_round(run_id, round_id, repair_level, evidence_index, v22_rows)
        repair.record(round_id, rr["summary"], rr["added_joint_vs_v22"], rr["regression_vs_v22"])
        rounds.append({k: v for k, v in rr.items() if k not in {"selected_rows", "oracle_rows", "records"}})
        joint_count = round(rr["summary"]["joint"] * rr["summary"]["valid"])
        print(json.dumps({"round": round_id, "repair_level": repair_level, "joint_count": joint_count, "summary": rr["summary"], "added": rr["added_joint_vs_v22"], "regression": rr["regression_vs_v22"]}, ensure_ascii=False))
        if not repair.should_continue():
            final_round = rr
            break
        round_id += 1
    selected_sources: Dict[str, int] = {}
    for rec in final_round["records"]:
        src = (rec.get("selection") or {}).get("selected_source", "unknown")
        selected_sources[src] = selected_sources.get(src, 0) + 1
    state = {
        "run_id": run_id,
        "version": "V29_SCENE_RESOLVER_SPRINT_VAL41_GAIN",
        "V10_zip_sha256": v10_sha,
        "V22_baseline": v22_full,
        "V29_selected_final": final_round["summary"],
        "V29_oracle_best_final": final_round["oracle_summary"],
        "V29_joint_count": round(final_round["summary"]["joint"] * final_round["summary"]["valid"]),
        "selected_result_dir": final_round["selected_dir"],
        "oracle_result_dir": final_round["oracle_dir"],
        "per_scenario": final_round["per_scenario"],
        "added_joint_vs_v22": final_round["added_joint_vs_v22"],
        "regression_vs_v22": final_round["regression_vs_v22"],
        "selected_sources": selected_sources,
        "rounds": rounds,
        "repair_history": repair.history,
        "repair_status": repair.final_status(),
        "uses_val41_gt_for_repair": any(any(r.get("uses_val41_gt_for_repair") for r in (rr.get("records") or [])) for rr in [final_round]) or any(x.get("repair_level", 0) > 0 for x in rounds),
        "uses_final_hidden_metadata": False,
        "final_run": False,
        "auto_submit": False,
        "val41_task_count": len(tasks),
        "evidence_records": len(evidence_index),
        "v10_zip_overwritten": before_mtime != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }
    write_json(CODEX / "state" / "latest_v29_scene_resolver_sprint.json", state)
    write_jsonl(CODEX / "analysis" / "v29_final_records_compact.jsonl", [
        {
            "task_key": r["task_key"],
            "round": r["round"],
            "scenario": r["scenario"],
            "v22_score": r["v22_score"],
            "selected_score": r["selected_score"],
            "oracle_score": r["oracle_score"],
            "selection": {
                "selected_candidate_id": (r.get("selection") or {}).get("selected_candidate_id"),
                "selected_source": (r.get("selection") or {}).get("selected_source"),
                "reason": (r.get("selection") or {}).get("reason"),
                "uses_val41_gt_for_selection": (r.get("selection") or {}).get("uses_val41_gt_for_selection"),
            },
            "uses_val41_gt_for_repair": r.get("uses_val41_gt_for_repair"),
        }
        for r in final_round["records"]
    ])
    write_reports(run_id, state, gap_summary)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
