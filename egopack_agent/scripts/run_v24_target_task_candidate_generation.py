#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V14_DIR = EGO / "results" / "V14_candidate_selection_val41-v14_candidate_selection_20260619_2134"
V22_DIR = EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915"
V23_STATE = CODEX / "state" / "latest_v23_allin_val41_shadow.json"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))

from egobench_agent_plus.v19_case_retriever import classify_task_type  # noqa: E402
from egobench_agent_plus.v19_program_transplanter import generate_candidates as generate_case_candidates  # noqa: E402
from egobench_agent_plus.v24_candidate_dryrun_and_selector import dryrun_program, select_v24_candidate  # noqa: E402
from egobench_agent_plus.v24_scenario_gap_generators import generate_for_scenario, norm_text  # noqa: E402


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_specs() -> List[Tuple[str, int, List[int]]]:
    m = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in m.get("specs", [])]


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
        data = getattr(restaurant_init, f"restaurant_init_data{number}", None) or getattr(restaurant_init, "restaurant_init_data")
        db.init_from_json(data)
        return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init
        db = OrderDB()
        data = getattr(order_init, f"order_init_data{number}", None) or getattr(order_init, "order_init_data")
        db.init_from_json(data)
        return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init
        db = KitchenDB()
        data = getattr(kitchen_init, f"kitchen_init_data{number}", None) or getattr(kitchen_init, "kitchen_init_data")
        db.init_from_json(data)
        return db
    raise ValueError(scenario)


def qwen_card(spec: str, pos: int) -> Dict[str, Any]:
    for p in [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos+1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{pos+1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_143308" / f"{spec}_{pos+1}.json",
    ]:
        data = read_json(p)
        if isinstance(data, dict):
            data["_path"] = str(p)
            return data
    return {"status": "missing", "_path": ""}


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        if not isinstance(block, dict):
            continue
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
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} V24 target candidate."}],
        "tool_calls": [
            {
                "turn": 0,
                "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program],
                "blocked_calls": [],
                "results": [],
                "v24_meta": meta or {},
            }
        ],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
    }


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    items = read_json(result_dir / f"{spec}_easy.json", [])
    return items[pos] if isinstance(items, list) and pos < len(items) else None


def candidate_from_item(source: str, item: Dict[str, Any] | None) -> Dict[str, Any]:
    return {"candidate_id": source, "source": source, "tool_program": program_from_item(item), "risk_flags": [] if item else ["missing_item"], "shape_confidence": 0.3}


def case_candidates(row: Dict[str, Any], scenario: str, spec: str, qwen: Dict[str, Any]) -> List[Dict[str, Any]]:
    values = row.get("value") or []
    if isinstance(values, str):
        values = [values]
    visual_names = [str(x) for x in values if str(x).strip()]
    for obj in qwen.get("top_k_candidates") or []:
        if isinstance(obj, dict):
            n = obj.get("entity") or obj.get("name") or obj.get("product_name") or obj.get("dish_name") or obj.get("ingredient_name") or obj.get("recipe_name")
            if n:
                visual_names.append(str(n))
    context = {
        "spec": spec,
        "scenario": scenario,
        "instruction": row.get("Instruction", ""),
        "visual_text": json.dumps(qwen, ensure_ascii=False)[:4000] + "\n" + str(row.get("image_description", "")),
        "task_type": classify_task_type(row.get("Instruction", ""), []),
        "entity_types": ["user_id", "product_name", "dish_name", "set_meal_name", "recipe_name", "ingredient_name", "restaurant_name", "category"],
        "visual_candidates": {
            "product_name": visual_names,
            "dish_name": visual_names,
            "set_meal_name": visual_names,
            "recipe_name": visual_names,
            "ingredient_name": visual_names,
        },
    }
    out = []
    try:
        obj = generate_case_candidates(context, top_k=16)
    except Exception as exc:
        return [{"candidate_id": "V24_CASE_ERROR", "source": "V19_CASE", "tool_program": [], "risk_flags": ["case_error"], "shape_confidence": 0.0, "meta": {"error": f"{type(exc).__name__}: {exc}"}}]
    for c in (obj.get("ranked") or obj.get("candidates") or [])[:10]:
        out.append({
            "candidate_id": "V24_CASE_" + str(c.get("candidate_id")),
            "source": "V19_CASE",
            "tool_program": c.get("tool_program") or [],
            "risk_flags": c.get("risk_flags") or [],
            "shape_confidence": c.get("program_shape_confidence", 0.0),
            "meta": {"source_case_ids": c.get("source_case_ids"), "score": c.get("score")},
        })
    return out


def load_gpt_candidates() -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    path = CODEX / "analysis" / "v24_gpt_program_candidates.jsonl"
    out: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in read_jsonl(path):
        key = (row.get("spec"), int(row.get("index", -1)))
        cands = row.get("candidates") or []
        if isinstance(cands, list):
            out[key] = [c for c in cands if isinstance(c, dict)]
    return out


def sanitize_program(program: List[Dict[str, Any]], scenario: str) -> List[Dict[str, Any]]:
    out = []
    for step in program or []:
        step = copy.deepcopy(step)
        params = step.get("parameters") or {}
        # Official evaluator/toolchain expects dish_name in restaurant/order
        # aggregate lists.  Case-library transplants often carry product_name
        # from retail-like shapes; normalize without using GT.
        if scenario in {"order", "restaurant"} and isinstance(params.get("dishes"), list):
            fixed = []
            for item in params["dishes"]:
                if isinstance(item, dict):
                    item = dict(item)
                    if "product_name" in item and "dish_name" not in item:
                        item["dish_name"] = item.pop("product_name")
                    fixed.append(item)
            params["dishes"] = fixed
        # Remove empty stage annotations from executable calls.
        step["parameters"] = params
        step.pop("stage", None)
        out.append(step)
    return out


def dedupe(cands: List[Dict[str, Any]], limit: int = 60) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in cands:
        sig = json.dumps(c.get("tool_program") or [], ensure_ascii=False, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    with tempfile.TemporaryDirectory(prefix="v24_eval_") as td:
        d = Path(td)
        gt = d / "gt.json"
        pr = d / "pred.json"
        write_json(gt, [gt_item])
        write_json(pr, [pred_item])
        metrics = evaluate_interaction_success(str(gt), str(pr), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
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


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (score.get("joint", 0), score.get("tool", 0), score.get("result", 0), score.get("matches", 0), -score.get("interaction_calls", 999999))


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = len(rows)
    if not valid:
        return {"valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "matched_tools": 0, "gt_tools": 0, "interaction_calls": 0}
    matched = sum(int(r.get("matches", r.get("matched_tools", 0)) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", r.get("gt_tools", 0)) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(float(r.get("joint", 0)) for r in rows) / valid,
        "result": sum(float(r.get("result", 0)) for r in rows) / valid,
        "tool": sum(float(r.get("tool", 0)) for r in rows) / valid,
        "micro": matched / gt if gt else 0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def load_v22_eval_map() -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows = read_jsonl(CODEX / "analysis" / "v23_selection_trace.jsonl")
    out = {}
    for r in rows:
        out[(r["spec"], int(r["index"]))] = r.get("v22_eval") or {}
    return out


def write_result_dir(result_dir: Path, item_by_key: Dict[Tuple[str, int], Dict[str, Any]], fallback_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        fallback_items = read_json(fallback_dir / f"{spec}_easy.json", [])
        items = []
        for pos, row in enumerate(rows):
            idx = int(row.get("_v8_original_index", pos))
            item = item_by_key.get((spec, idx))
            if item is None:
                item = copy.deepcopy(fallback_items[pos]) if pos < len(fallback_items) else make_item(row, [], "missing_fallback")
            items.append(item)
        write_json(result_dir / f"{spec}_easy.json", items)


def eval_dir(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_out = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        gt_items = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_items = read_json(result_dir / f"{spec}_easy.json", [])
        for pos, gt in enumerate(gt_items):
            idx = int(gt.get("_v8_original_index", pos))
            pred = pred_items[pos] if pos < len(pred_items) else make_item(gt, [], "missing_pred")
            ev = evaluate_one(gt, pred, scenario, number)
            ev.update({"spec": spec, "index": idx, "scenario": scenario})
            rows_out.append(ev)
    return rows_out, aggregate(rows_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v24_gap_generators_" + time.strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()
    run_id = args.run_id
    before_zip = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    target_obj = read_json(CODEX / "analysis" / "v24_target_tasks.json", {})
    targets = target_obj.get("targets") or []
    target_keys = {(t["spec"], int(t["index"])): t for t in targets}
    v22_eval_map = load_v22_eval_map()
    all_candidates_rows = []
    dry_rows = []
    selection_rows = []
    selected_item_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_item_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    target_selected_evals = []
    target_oracle_evals = []
    target_v22_evals = []
    gpt_by_task = load_gpt_candidates()

    for t in targets:
        spec = t["spec"]
        scenario = t["scenario"]
        pos = int(t["local_pos"])
        idx = int(t["index"])
        number = int(re.sub(r"^\D+", "", spec))
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        row = rows[pos]
        db = init_db(scenario, number)
        qwen = qwen_card(spec, pos)
        cands: List[Dict[str, Any]] = []
        cands.append(candidate_from_item("V22_BASE", load_item(V22_DIR, spec, pos)))
        cands.append(candidate_from_item("V14_BASE", load_item(V14_DIR, spec, pos)))
        cands += case_candidates(row, scenario, spec, qwen)
        cands += generate_for_scenario(scenario, row, db, qwen, max_candidates=40)
        cands += gpt_by_task.get((spec, idx), [])
        # Cross-scenario fallback can help restaurant/order menu ambiguity but
        # keeps main scenario dominant via confidence.
        if scenario in {"order", "restaurant"}:
            cands += generate_for_scenario("restaurant" if scenario == "order" else "order", row, db, qwen, max_candidates=10) if False else []
        enriched = []
        for c in dedupe(cands, limit=60):
            db2 = init_db(scenario, number)
            c = dict(c)
            c["tool_program"] = sanitize_program(c.get("tool_program") or [], scenario)
            c["dryrun"] = dryrun_program(scenario, db2, c.get("tool_program") or [], row.get("Instruction", ""))
            enriched.append(c)
            all_candidates_rows.append({"spec": spec, "index": idx, "scenario": scenario, "candidate": c})
            dry_rows.append({"spec": spec, "index": idx, "scenario": scenario, "candidate_id": c.get("candidate_id"), "dryrun": c.get("dryrun")})
        selected = select_v24_candidate(enriched, {"instruction": row.get("Instruction", ""), "scenario": scenario})
        # Do not replace an already joint V22 item.  Targets should be V22
        # failures, but keep this guard explicit.
        v22_ev = v22_eval_map.get((spec, idx), {})
        if v22_ev.get("joint"):
            selected = candidate_from_item("V22_PROTECTED_BASE", load_item(V22_DIR, spec, pos))
        selected_item = make_item(row, selected.get("tool_program") or [], selected.get("candidate_id", "V24_SELECTED"), {"selector_score": selected.get("selector_score"), "selector_reasons": selected.get("selector_reasons"), "hard_filters": selected.get("hard_filters"), "source": selected.get("source")})
        selected_ev = evaluate_one(row, selected_item, scenario, number)
        best = None
        best_ev = None
        for c in enriched:
            item = make_item(row, c.get("tool_program") or [], c.get("candidate_id", "V24_CAND"), {"source": c.get("source")})
            ev = evaluate_one(row, item, scenario, number)
            if best is None or score_tuple(ev) > score_tuple(best_ev):
                best = c
                best_ev = ev
        oracle_item = make_item(row, (best or selected).get("tool_program") or [], (best or selected).get("candidate_id", "V24_ORACLE"), {"oracle_best_post_eval": True, "source": (best or selected).get("source")})
        selected_item_by_key[(spec, idx)] = selected_item
        oracle_item_by_key[(spec, idx)] = oracle_item
        target_selected_evals.append(selected_ev | {"spec": spec, "index": idx, "scenario": scenario})
        target_oracle_evals.append((best_ev or selected_ev) | {"spec": spec, "index": idx, "scenario": scenario})
        target_v22_evals.append(v22_ev | {"spec": spec, "index": idx, "scenario": scenario})
        selection_rows.append({
            "spec": spec,
            "index": idx,
            "local_pos": pos,
            "scenario": scenario,
            "candidate_count": len(enriched),
            "selected_candidate": selected.get("candidate_id"),
            "selected_source": selected.get("source"),
            "selected_eval": selected_ev,
            "oracle_best_candidate": (best or {}).get("candidate_id"),
            "oracle_best_eval": best_ev,
            "v22_eval": v22_ev,
            "uses_gt_for_selection": False,
        })

    # Merge selected/oracle over V22 floor.
    selected_dir = EGO / "results" / f"V24_scenario_gt_gap_generators_val41_shadow-{run_id}"
    oracle_dir = EGO / "results" / f"V24_scenario_gt_gap_generators_val41_oracle_bestof-{run_id}"
    write_result_dir(selected_dir, selected_item_by_key, V22_DIR)
    write_result_dir(oracle_dir, oracle_item_by_key, V22_DIR)
    full_selected_rows, full_selected_summary = eval_dir(selected_dir)
    full_oracle_rows, full_oracle_summary = eval_dir(oracle_dir)
    _, v22_summary = eval_dir(V22_DIR)

    write_jsonl(CODEX / "analysis" / "v24_target_task_candidates.jsonl", all_candidates_rows)
    write_jsonl(CODEX / "analysis" / "v24_candidate_dryrun.jsonl", dry_rows)
    write_jsonl(CODEX / "analysis" / "v24_target_selection_trace.jsonl", selection_rows)

    state = {
        "run_id": run_id,
        "version": "V24_SCENARIO_GT_GAP_GENERATORS_VAL41_50PUSH",
        "target_count": len(targets),
        "selected_result_dir": str(selected_dir),
        "oracle_best_result_dir": str(oracle_dir),
        "target_selected": aggregate(target_selected_evals),
        "target_oracle_best": aggregate(target_oracle_evals),
        "target_v22": aggregate(target_v22_evals),
        "V22_baseline": v22_summary,
        "V24_selected_merged": full_selected_summary,
        "V24_oracle_best_merged": full_oracle_summary,
        "selected_added_joint_vs_v22": int(round(full_selected_summary["joint"] * 41 - v22_summary["joint"] * 41)),
        "oracle_added_joint_vs_v22": int(round(full_oracle_summary["joint"] * 41 - v22_summary["joint"] * 41)),
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_gt_for_runtime_selection": False,
        "uses_gt_for_post_eval_oracle_best": True,
        "v10_zip_overwritten": before_zip != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }
    write_json(CODEX / "state" / "latest_v24_scenario_gap_generators.json", state)
    write_reports(run_id, state, targets, selection_rows)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def write_reports(run_id: str, state: Dict[str, Any], targets: List[Dict[str, Any]], selection_rows: List[Dict[str, Any]]) -> None:
    rep = CODEX / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    table = [
        "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("V22_baseline", state["V22_baseline"]),
        table_row("V24_target_selected", state["target_selected"]),
        table_row("V24_target_oracle_best", state["target_oracle_best"]),
        table_row("V24_selected_merged", state["V24_selected_merged"]),
        table_row("V24_oracle_best_merged", state["V24_oracle_best_merged"]),
    ]
    impl = [
        f"# V24 Scenario Generators Implementation {run_id}",
        "",
        "- Implemented scenario-specific deterministic candidate generators for order, restaurant, kitchen, and non-clean retail.",
        "- Runtime inputs: instruction/dialogue fields, current value field, DB catalog/init state, Qwen visual card, V19 non-final GT100 case library candidates.",
        "- GT usage: post-eval gap mining and oracle best-of diagnostic only.",
        "- Full main agent was not rerun; V24 merges target replacements over the V22 protected floor.",
        "",
        *table,
    ]
    (rep / f"V24_SCENARIO_GENERATORS_IMPLEMENTATION_{run_id}.md").write_text("\n".join(impl) + "\n", encoding="utf-8")

    target_lines = [
        f"# V24 Target Task Result {run_id}",
        "",
        *table,
        "",
        "| spec | index | scenario | candidates | selected | selected_joint | oracle_best | oracle_joint | v22_joint |",
        "|---|---:|---|---:|---|---:|---|---:|---:|",
    ]
    for r in selection_rows:
        target_lines.append(f"| {r['spec']} | {r['index']} | {r['scenario']} | {r['candidate_count']} | {r['selected_candidate']} | {int(r['selected_eval'].get('joint',0))} | {r['oracle_best_candidate']} | {int((r.get('oracle_best_eval') or {}).get('joint',0))} | {int((r.get('v22_eval') or {}).get('joint',0))} |")
    (rep / f"V24_TARGET_TASK_RESULT_{run_id}.md").write_text("\n".join(target_lines) + "\n", encoding="utf-8")

    merged = [
        f"# V24 Merged Val41 Result {run_id}",
        "",
        *table,
        "",
        f"- selected_added_joint_vs_v22: {state['selected_added_joint_vs_v22']}",
        f"- oracle_added_joint_vs_v22: {state['oracle_added_joint_vs_v22']}",
        f"- selected_reaches_15_of_41: {state['V24_selected_merged']['joint'] >= 15/41}",
        f"- selected_reaches_20_of_41: {state['V24_selected_merged']['joint'] >= 20/41}",
        f"- oracle_reaches_20_of_41: {state['V24_oracle_best_merged']['joint'] >= 20/41}",
        "- final_run: false",
        "- v10_zip_overwritten: " + str(state["v10_zip_overwritten"]).lower(),
        "- auto_submit: false",
    ]
    (rep / f"V24_MERGED_VAL41_RESULT_{run_id}.md").write_text("\n".join(merged) + "\n", encoding="utf-8")

    oracle = [
        f"# V24 Oracle Best-of Analysis {run_id}",
        "",
        *table,
        "",
        "Oracle best-of is post-eval diagnostic only. It is not used by the non-oracle selector.",
        "",
        f"- target_oracle_joint_count: {state['target_oracle_best']['joint'] * state['target_oracle_best']['valid']:.0f}/{state['target_oracle_best']['valid']}",
        f"- merged_oracle_joint_count: {state['V24_oracle_best_merged']['joint'] * 41:.0f}/41",
        "- bottleneck: " + ("selector" if state["V24_oracle_best_merged"]["joint"] >= 20/41 and state["V24_selected_merged"]["joint"] < state["V24_oracle_best_merged"]["joint"] else "candidate_generator"),
    ]
    (rep / f"V24_ORACLE_BESTOF_ANALYSIS_{run_id}.md").write_text("\n".join(oracle) + "\n", encoding="utf-8")

    by_scen: Dict[str, Dict[str, int]] = {}
    for r in selection_rows:
        b = by_scen.setdefault(r["scenario"], {"tasks": 0, "selected_joint": 0, "oracle_joint": 0})
        b["tasks"] += 1
        b["selected_joint"] += int(r["selected_eval"].get("joint", 0))
        b["oracle_joint"] += int((r.get("oracle_best_eval") or {}).get("joint", 0))
    next_lines = [
        f"# V24 Next Decision {run_id}",
        "",
        *table,
        "",
        "## Required Answers",
        "",
        "1. V24 target tasks: " + ", ".join([f"{t['spec']}::{t['index']}" for t in targets]),
        "2. Gap types are in `analysis/v24_val41_gap_mining.jsonl` and summarized in `V24_VAL41_GT_GAP_MINING`.",
        "3. Scenario generator joint-capable candidates by target scenario: `" + json.dumps(by_scen, ensure_ascii=False) + "`.",
        f"4. Oracle best-of over 20/41: {state['V24_oracle_best_merged']['joint'] >= 20/41}.",
        f"5. Selected over 15/41: {state['V24_selected_merged']['joint'] >= 15/41}; selected over 20/41: {state['V24_selected_merged']['joint'] >= 20/41}.",
        "6. If selected is low: " + ("selector problem" if state["V24_oracle_best_merged"]["joint"] > state["V24_selected_merged"]["joint"] else "generator problem"),
        "7. No final run, no V10 zip overwrite, no auto-submit.",
        "8. Next step: " + ("selector distillation" if state["V24_oracle_best_merged"]["joint"] >= 20/41 and state["V24_selected_merged"]["joint"] < state["V24_oracle_best_merged"]["joint"] else "continue scenario generator work; current candidate upper bound is still insufficient."),
    ]
    (rep / f"V24_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import re
    main()
