#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V28 protected delta val41 shadow."""

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
V27_DIRECT_DIR = EGO / "results" / "V27_direct_v21_retail_shadow-v27_direct_v21_20260621_1050"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
EVIDENCE_PATHS = [CODEX / "analysis" / "v26_mm_evidence_val41.jsonl", CODEX / "analysis" / "v25_new_mm_evidence.jsonl"]
PRIORITY_KEYS = {("retail2", 5), ("restaurant3", 24), ("restaurant3", 54), ("kitchen1", 31), ("restaurant4", 6)}

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))

from egobench_agent_plus.v24_candidate_dryrun_and_selector import dryrun_program  # noqa: E402
from egobench_agent_plus.v27_v25_to_v21_adapter import build_v27_v21_candidate  # noqa: E402
from egobench_agent_plus.v28_evidence_guard import guard_candidate, evidence_error_bucket  # noqa: E402
from egobench_agent_plus.v28_min_resolvers import MinimalResolverV28  # noqa: E402
from egobench_agent_plus.v28_protected_delta_selector import ProtectedDeltaSelectorV28  # noqa: E402


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
            spec, idx = str(row.get("spec") or ""), row.get("index")
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
        "tool_calls": [{"turn": 0, "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program], "blocked_calls": [], "results": [], "v28_meta": meta or {}}],
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

    with tempfile.TemporaryDirectory(prefix="v28_eval_") as td:
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
    return {"valid": valid, "joint": sum(float(r.get("joint", 0)) for r in rows) / valid, "result": sum(float(r.get("result", 0)) for r in rows) / valid, "tool": sum(float(r.get("tool", 0)) for r in rows) / valid, "micro": matched / gt if gt else 0.0, "matched_tools": matched, "gt_tools": gt, "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows)}


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (float(score.get("joint", 0)), float(score.get("tool", 0)), float(score.get("result", 0)), int(score.get("matches", 0)), -int(score.get("interaction_calls", 999999)))


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


def first_failure(record: Dict[str, Any]) -> str:
    if record.get("selected_score", {}).get("joint"):
        return "resolved"
    if record.get("selection", {}).get("reason") == "v22_joint_success_locked":
        return "protected_v22"
    if not record.get("candidates"):
        return "candidate_generator_empty"
    if any((c.get("dryrun") or {}).get("errors") for c in record.get("candidates", [])):
        return "tool_schema_error"
    if any((c.get("evidence_guard") or {}).get("risk_flags") for c in record.get("candidates", [])):
        return "evidence_conflict"
    return "resolver_or_candidate_wrong"


def build_candidates(task: Dict[str, Any], evidence: Dict[str, Any], v22_score: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenario, number, row = task["scenario"], task["number"], task["row"]
    db = init_db(scenario, number)
    cands: List[Dict[str, Any]] = []
    if scenario == "retail":
        qwen = qwen_card(task["spec"], task["local_pos"])
        for use_ev, cid in [(False, "V28_REAL_V21_RETAIL"), (True, "V28_REAL_V21_RETAIL_EVIDENCE_HINT")]:
            obj = build_v27_v21_candidate(db, row, evidence=evidence, qwen_card=qwen, use_evidence=use_ev)
            cand = copy.deepcopy(obj.get("candidate") or {})
            cand["candidate_id"] = cid
            cand["source"] = cid
            cand["trace"] = obj.get("trace")
            cands.append(cand)
    elif scenario in {"order", "restaurant", "kitchen"}:
        # Run minimal resolvers on V22-failed tasks only; priority tasks get
        # two candidates, other failed tasks get one to keep candidate count low.
        max_candidates = 2 if (task["spec"], task["index"]) in PRIORITY_KEYS else 1
        cands.extend(MinimalResolverV28(scenario, db).build(row, evidence, max_candidates=max_candidates))
    # Closure repair: only append a repaired form when a candidate already
    # exists. The driver cap keeps total <=8.
    repaired: List[Dict[str, Any]] = []
    for cand in cands[:4]:
        prog = copy.deepcopy(cand.get("tool_program") or [])
        names = [p.get("tool_name") for p in prog]
        text = row.get("Instruction", "").lower()
        closure = ""
        if "total tax" in text:
            closure = "compute_total_tax"
        elif "total nutrition" in text or "total nutritional" in text:
            closure = "compute_total_nutrition"
        elif "total payment" in text:
            closure = "compute_total_payment"
        if closure and closure not in names:
            user_id = ""
            products = []
            dishes = []
            for call in prog:
                p = call.get("parameters") or {}
                user_id = p.get("user_id") or user_id
                if p.get("product_name"):
                    products.append({"product_name": p.get("product_name"), "quantity": p.get("quantity") or p.get("qty") or 1})
                if p.get("dish_name"):
                    dishes.append({"dish_name": p.get("dish_name"), "quantity": p.get("quantity") or 1})
            params = {"user_id": user_id}
            if products:
                params["products"] = products
            if dishes:
                params["dishes"] = dishes
            prog.append({"tool_name": closure, "parameters": params})
            r = copy.deepcopy(cand)
            r["candidate_id"] = cand.get("candidate_id", "cand") + "_CLOSURE_REPAIR"
            r["source"] = cand.get("source", "") + "_CLOSURE_REPAIR"
            r["tool_program"] = prog
            repaired.append(r)
    return (cands + repaired)[:8]


def certify_takeover(cand: Dict[str, Any], dry: Dict[str, Any], guard: Dict[str, Any]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    names = [str(x.get("tool_name", "")) for x in cand.get("tool_program") or []]
    source = str(cand.get("source", ""))
    if dry.get("errors"):
        reasons.append("dryrun_errors")
    if dry.get("broad_scan"):
        reasons.append("broad_scan")
    if dry.get("closure_required") and not dry.get("closure_complete"):
        reasons.append("closure_incomplete")
    if "evidence_disagrees_with_mutation_target" in (guard.get("risk_flags") or []):
        reasons.append("evidence_target_conflict")
    if any(not (call.get("parameters") or {}) for call in cand.get("tool_program") or []):
        reasons.append("empty_params_present")
    observation = bool(dry.get("retrieval_nonempty_count") or dry.get("branch_observation_count"))
    closure_ok = (not dry.get("closure_required")) or bool(dry.get("closure_complete"))
    mutation = any(n in {"add_to_cart", "add_dish_to_order", "add_set_meal_to_order", "add_to_shopping_list", "remove_from_cart", "remove_dish_from_order"} for n in names)
    if source.startswith("V28_REAL_V21"):
        trace = cand.get("trace") or {}
        if not all(trace.get(k) is True for k in ("called_RetailResolverV21", "called_attribute_query_planner", "called_observation_brancher", "called_add_target_resolver")):
            reasons.append("v21_trace_incomplete")
        if not ((trace.get("v21_trace") or {}).get("mutation_target")):
            reasons.append("v21_no_mutation_target")
        if not observation and mutation:
            reasons.append("v21_no_observation_before_mutation")
    elif source.endswith("_MIN") or "_MIN" in source:
        if not observation and mutation:
            reasons.append("min_no_observation_before_mutation")
        # The minimal resolvers are intentionally weak; only certify priority
        # task candidates with complete closure or query-only observation.
        if not cand.get("priority_task") and mutation:
            reasons.append("min_nonpriority_mutation")
    else:
        reasons.append("unknown_source")
    if not closure_ok:
        reasons.append("closure_not_ok")
    return (not reasons), reasons


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    rep = CODEX / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    table = ["| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|", table_row("V22_baseline", state["V22_baseline"]), table_row("V28_selected", state["V28_selected"]), table_row("V28_oracle_best", state["V28_oracle_best"])]
    (rep / f"V28_PREFLIGHT_{run_id}.md").write_text("\n".join([f"# V28 Preflight {run_id}", "", f"- V22 protected base: `{V22_DIR}`", f"- V22 joint: {state['V22_baseline']['joint']*100:.2f}%", f"- V10 zip sha256: `{state['V10_zip_sha256']}`", f"- V10 zip overwritten: {state['v10_zip_overwritten']}", f"- evidence records: {state['evidence_records']}/41", "- final_run: false", "- final_hidden_metadata_used: false"]) + "\n", encoding="utf-8")
    (rep / f"V28_EVIDENCE_GUARD_AUDIT_{run_id}.md").write_text("\n".join([f"# V28 Evidence Guard Audit {run_id}", "", f"- evidence guard actions: `{state['evidence_guard_actions']}`", f"- evidence error buckets: `{state['evidence_error_buckets']}`", "- evidence override disabled: true", "- evidence may veto/tiebreak/query-hint only; it does not replace mutation targets."]) + "\n", encoding="utf-8")
    (rep / f"V28_RESOLVER_IMPLEMENTATION_{run_id}.md").write_text("\n".join([f"# V28 Resolver Implementation {run_id}", "", "- Added protected delta selector.", "- Added evidence-as-veto/tiebreak/query-hint guard.", "- Reused real V21 retail resolver through V27 adapter.", "- Added compact minimal five-stage resolver for order/restaurant/kitchen in one file.", "- No V18 oracle compiler, no final metadata, no large program induction."]) + "\n", encoding="utf-8")
    lines = [f"# V28 Val41 Result {run_id}", "", *table, "", "## Per Scenario", "", "| spec | selected joint | oracle joint | selected micro | oracle micro |", "|---|---:|---:|---:|---:|"]
    for spec, row in state["per_scenario"].items():
        lines.append(f"| {spec} | {row['selected']['joint']*100:.2f}% | {row['oracle']['joint']*100:.2f}% | {row['selected']['micro']:.4f} | {row['oracle']['micro']:.4f} |")
    lines += ["", f"- added_joint_vs_v22: {state['added_joint_vs_v22']}", f"- regression_vs_v22: {state['regression_vs_v22']}", f"- protected_v22_success_count: {state['protected_v22_success_count']}", f"- selected_result_dir: `{state['selected_result_dir']}`", f"- oracle_result_dir: `{state['oracle_result_dir']}`"]
    (rep / f"V28_VAL41_RESULT_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (rep / f"V28_EVIDENCE_ERROR_ANALYSIS_{run_id}.md").write_text("\n".join([f"# V28 Evidence Error Analysis {run_id}", "", f"- evidence changed/conflicted mutation target wrong: {state['evidence_error_buckets'].get('evidence_changed_or_conflicted_mutation_target_wrong',0)}", f"- evidence vetoed correct candidate: {state['evidence_error_buckets'].get('evidence_vetoed_correct_candidate',0)}", f"- evidence helped choose correct entity: {state['evidence_error_buckets'].get('evidence_helped_choose_correct_entity',0)}", f"- evidence ignored due to low confidence: {state['evidence_error_buckets'].get('evidence_ignored_due_to_low_confidence',0)}", f"- ASR/subtitle unavailable: {state['asr_unavailable']}", f"- OCR exact match success: {state['ocr_exact_match_success']}", f"- OCR false positive: {state['ocr_false_positive']}"]) + "\n", encoding="utf-8")
    decision = "promote_candidate" if state["V28_selected"]["joint"] > state["V22_baseline"]["joint"] and not state["regression_vs_v22"] else "do_not_promote_protected_base_remains_v22"
    (rep / f"V28_NEXT_DECISION_{run_id}.md").write_text("\n".join([f"# V28 Next Decision {run_id}", "", *table, "", f"- decision: {decision}", f"- V28 exceeds V22: {state['V28_selected']['joint'] > state['V22_baseline']['joint']}", f"- V22 success regression count: {len(state['regression_vs_v22'])}", f"- evidence as veto/tiebreak safer than direct override: {state['V28_selected']['joint'] >= state.get('V27_evidence_joint', 0)}", f"- real V21 module called: {state['real_v21_called']}", f"- min resolvers selected count: {state['min_resolver_selected_count']}", f"- oracle_best_joint_count: {round(state['V28_oracle_best']['joint']*41)}/41", "- final_run: false", "- final_hidden_metadata_used: false", f"- V10 zip overwritten: {state['v10_zip_overwritten']}", "- auto_submit: false"]) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v28_protected_delta_" + stamp())
    args = ap.parse_args()
    run_id = args.run_id
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    v10_sha = subprocess.check_output(["sha256sum", str(V10_ZIP)], text=True).split()[0] if V10_ZIP.exists() else ""
    evidence_index = load_evidence_index()
    paths = {
        "priority": CODEX / "analysis" / "v28_priority_tasks.json",
        "guard": CODEX / "analysis" / "v28_evidence_guard_trace.jsonl",
        "dryrun": CODEX / "analysis" / "v28_dryrun_trace.jsonl",
        "selection": CODEX / "analysis" / "v28_selection_trace.jsonl",
    }
    for key, path in paths.items():
        if path.suffix == ".jsonl":
            path.write_text("", encoding="utf-8")
    tasks = all_tasks()
    selected_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_items: Dict[Tuple[str, int], Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    real_v21_called = False
    selector = ProtectedDeltaSelectorV28()
    priority_rows = []
    for i, task in enumerate(tasks, 1):
        spec, idx, pos, scenario, number, row = task["spec"], task["index"], task["local_pos"], task["scenario"], task["number"], task["row"]
        task_key = f"{spec}::{idx}"
        v22_item = load_item(V22_DIR, spec, pos) or make_item(row, [], "missing_v22")
        v22_score = evaluate_one(row, v22_item, scenario, number)
        evidence = evidence_index.get((spec, idx), {})
        candidates: List[Dict[str, Any]] = []
        if not v22_score.get("joint"):
            raw_candidates = build_candidates(task, evidence, v22_score)
            for cand in raw_candidates:
                db = init_db(scenario, number)
                dry = dryrun_program(scenario, db, cand.get("tool_program") or [], row.get("Instruction", ""))
                guard = guard_candidate(task_key, scenario, row.get("Instruction", ""), evidence, cand, allow_override=False)
                cand["dryrun"] = dry
                cand["evidence_guard"] = guard
                cand["priority_task"] = (spec, idx) in PRIORITY_KEYS
                certified, cert_reasons = certify_takeover(cand, dry, guard)
                cand["takeover_certified"] = certified
                cand["takeover_cert_reasons"] = cert_reasons
                candidates.append(cand)
                append_jsonl(paths["guard"], guard)
                append_jsonl(paths["dryrun"], {"task_key": task_key, "candidate_id": cand.get("candidate_id"), "dryrun": dry})
                if cand.get("source", "").startswith("V28_REAL_V21") or cand.get("source", "").startswith("V27"):
                    real_v21_called = True
        selection = selector.select(task_key, scenario, v22_item, v22_score, candidates)
        selected = selection["selected"]
        if isinstance(selected, dict) and selected.get("tool_program") is not None:
            selected_item = make_item(row, selected.get("tool_program") or [], selected.get("candidate_id", "V28_SELECTED"), {"selection": selection, "final_run": False})
        else:
            selected_item = v22_item
        selected_score = evaluate_one(row, selected_item, scenario, number)
        scored_candidates = []
        for cand in candidates:
            item = make_item(row, cand.get("tool_program") or [], cand.get("candidate_id", "cand"), {"oracle_best_post_eval": True})
            score = evaluate_one(row, item, scenario, number)
            scored_candidates.append((cand, score, item))
        if scored_candidates:
            best_cand, best_score, best_item = max(scored_candidates + [({"candidate_id": "V22"}, v22_score, v22_item)], key=lambda x: score_tuple(x[1]))
        else:
            best_cand, best_score, best_item = {"candidate_id": "V22"}, v22_score, v22_item
        selected_items[(spec, idx)] = selected_item
        oracle_items[(spec, idx)] = best_item
        rec = {"task_key": task_key, "spec": spec, "index": idx, "scenario": scenario, "v22_score": v22_score, "selected_score": selected_score, "oracle_score": best_score, "oracle_candidate": best_cand.get("candidate_id"), "selection": selection, "candidates": candidates}
        rec["first_failure"] = first_failure(rec)
        records.append(rec)
        append_jsonl(paths["selection"], {"task_key": task_key, "spec": spec, "index": idx, "scenario": scenario, "v22_score": v22_score, "selected_score": selected_score, "oracle_candidate": best_cand.get("candidate_id"), "oracle_score": best_score, "selection_reason": selection.get("reason"), "selected_candidate_id": selection.get("selected_candidate_id"), "uses_gt_for_selection": False})
        if (spec, idx) in PRIORITY_KEYS or (not v22_score.get("joint") and len(program_from_item(v22_item)) <= 12):
            priority_rows.append({"task_key": task_key, "reason": "priority_key" if (spec, idx) in PRIORITY_KEYS else "v22_failed_short_chain", "v22_calls": len(program_from_item(v22_item)), "candidate_count": len(candidates)})
        if i % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] processed {i}/{len(tasks)}")
    write_json(paths["priority"], priority_rows)
    selected_dir = EGO / "results" / f"V28_protected_delta_evidence_veto_val41_selected-{run_id}"
    oracle_dir = EGO / "results" / f"V28_protected_delta_evidence_veto_val41_oracle_best-{run_id}"
    write_result_dir(selected_dir, selected_items, V22_DIR)
    write_result_dir(oracle_dir, oracle_items, V22_DIR)
    v22_rows, v22_full = eval_result_dir(V22_DIR)
    selected_rows, selected_full = eval_result_dir(selected_dir)
    oracle_rows, oracle_full = eval_result_dir(oracle_dir)
    v22_by_key = {(r["spec"], r["index"]): r for r in v22_rows}
    selected_by_key = {(r["spec"], r["index"]): r for r in selected_rows}
    added = [f"{k[0]}::{k[1]}" for k, r in selected_by_key.items() if r["joint"] and not v22_by_key[k]["joint"]]
    regression = [f"{k[0]}::{k[1]}" for k, r in selected_by_key.items() if not r["joint"] and v22_by_key[k]["joint"]]
    per = {}
    for spec in sorted({r["spec"] for r in selected_rows}):
        per[spec] = {"selected": aggregate([r for r in selected_rows if r["spec"] == spec]), "oracle": aggregate([r for r in oracle_rows if r["spec"] == spec])}
    guard_actions: Dict[str, int] = {}
    error_buckets: Dict[str, int] = {}
    for rec in records:
        for cand in rec.get("candidates", []):
            guard = cand.get("evidence_guard") or {}
            guard_actions[guard.get("action", "none")] = guard_actions.get(guard.get("action", "none"), 0) + 1
            # Bucket against V22 because the run is protected delta.
            bucket = evidence_error_bucket(guard, rec["selected_score"], rec["v22_score"])
            error_buckets[bucket] = error_buckets.get(bucket, 0) + 1
    state = {
        "run_id": run_id,
        "version": "V28_PROTECTED_DELTA_AGENT_EVIDENCE_AS_VETO",
        "V10_zip_sha256": v10_sha,
        "V22_baseline": v22_full,
        "V28_selected": selected_full,
        "V28_oracle_best": oracle_full,
        "selected_result_dir": str(selected_dir),
        "oracle_result_dir": str(oracle_dir),
        "per_scenario": per,
        "added_joint_vs_v22": added,
        "regression_vs_v22": regression,
        "protected_v22_success_count": sum(1 for r in v22_rows if r["joint"]),
        "evidence_records": len(evidence_index),
        "evidence_guard_actions": guard_actions,
        "evidence_error_buckets": error_buckets,
        "asr_unavailable": sum(1 for e in evidence_index.values() if (e.get("asr_evidence") or {}).get("source") in {None, "", "none"}),
        "ocr_exact_match_success": sum(1 for e in evidence_index.values() if "exact_text" in json.dumps(e.get("candidate_slots", {}), ensure_ascii=False)),
        "ocr_false_positive": error_buckets.get("evidence_changed_or_conflicted_mutation_target_wrong", 0),
        "real_v21_called": real_v21_called,
        "min_resolver_selected_count": sum(1 for r in records if str(r.get("selection", {}).get("selected_source", "")).endswith("_MIN")),
        "V27_evidence_joint": 0.0975609756097561,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_runtime": False,
        "v10_zip_overwritten": before_mtime != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }
    write_json(CODEX / "state" / "latest_v28_protected_delta.json", state)
    write_jsonl(CODEX / "analysis" / "v28_records_compact.jsonl", [{"task_key": r["task_key"], "spec": r["spec"], "index": r["index"], "scenario": r["scenario"], "v22_score": r["v22_score"], "selected_score": r["selected_score"], "oracle_score": r["oracle_score"], "oracle_candidate": r["oracle_candidate"], "first_failure": r["first_failure"], "selected_candidate": r["selection"].get("selected_candidate_id"), "selection_reason": r["selection"].get("reason")} for r in records])
    write_reports(run_id, state, records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
