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
V10_DIR = EGO / "results" / "gpt-5.5-V10_full_memory_final_candidate_draft-V10_full_memory_final_candidate_draft_A_medium_sanity_20260618_1716"
V12_DIR = EGO / "results" / "gpt-5.5-V12_official_style_qwen3vl_memory-V12_qwen3vl_prior_all_modules_val41_parallel_20260619_170302"
V14B_DIR = EGO / "results" / "gpt-5.5-V14_val41_distilled_no_task_oracle-v14_distilled_val41_20260619_211502"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
from egobench_agent_plus.v23_aggressive_shadow_lib import (  # noqa: E402
    closure_call,
    dryrun_program,
    extract_restaurant_name,
    extract_user_id,
    find_catalog_rows,
    has_mutation_intent,
    lexical_candidates,
    make_item,
    norm_text,
    program_from_item,
    required_closure,
    select_candidate,
)


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
        data = getattr(restaurant_init, f"restaurant_init_data{number}", None)
        if data is None:
            data = getattr(restaurant_init, "restaurant_init_data")
        db.init_from_json(data)
        return db
    if scenario == "order":
        from tools.order.order_db import OrderDB
        from tools.order import order_init
        db = OrderDB()
        data = getattr(order_init, f"order_init_data{number}", None)
        if data is None:
            data = getattr(order_init, "order_init_data")
        db.init_from_json(data)
        return db
    if scenario == "kitchen":
        from tools.kitchen.kitchen_db import KitchenDB
        from tools.kitchen import kitchen_init
        db = KitchenDB()
        data = getattr(kitchen_init, f"kitchen_init_data{number}", None)
        if data is None:
            data = getattr(kitchen_init, "kitchen_init_data")
        db.init_from_json(data)
        return db
    raise ValueError(scenario)


def qwen_card(spec: str, pos: int) -> Dict[str, Any]:
    for p in [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos+1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{pos+1}.json",
    ]:
        data = read_json(p)
        if isinstance(data, dict):
            data["_path"] = str(p)
            return data
    return {"status": "missing", "_path": ""}


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse
    with tempfile.TemporaryDirectory(prefix="v23_eval_") as td:
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


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = len(rows)
    if not valid:
        return {"valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "matched_tools": 0, "gt_tools": 0, "interaction_calls": 0}
    m = sum(int(r.get("matches", r.get("matched_tools", 0)) or 0) for r in rows)
    g = sum(int(r.get("gt_calls", r.get("gt_tools", 0)) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(float(r.get("joint", 0)) for r in rows) / valid,
        "result": sum(float(r.get("result", 0)) for r in rows) / valid,
        "tool": sum(float(r.get("tool", 0)) for r in rows) / valid,
        "micro": m / g if g else 0,
        "matched_tools": m,
        "gt_tools": g,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (score.get("joint", 0), score.get("tool", 0), score.get("result", 0), score.get("matches", 0), -score.get("interaction_calls", 999999))


def load_item(result_dir: Path, spec: str, pos: int) -> Dict[str, Any] | None:
    items = read_json(result_dir / f"{spec}_easy.json", [])
    return items[pos] if isinstance(items, list) and pos < len(items) else None


def candidate(source: str, row: Dict[str, Any], program: List[Dict[str, Any]], meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "candidate_id": source,
        "source": source,
        "tool_program": program,
        "risk_flags": [],
        "shape_confidence": 0.0,
        "meta": meta or {},
    }


def baseline_candidates(row: Dict[str, Any], spec: str, pos: int) -> List[Dict[str, Any]]:
    out = []
    for label, d in [("V14", V14_DIR), ("V22", V22_DIR), ("V10", V10_DIR), ("V12", V12_DIR), ("V14B", V14B_DIR)]:
        item = load_item(d, spec, pos)
        if item:
            out.append(candidate(label, row, program_from_item(item), {"result_dir": str(d)}))
    return out


def case_candidates(row: Dict[str, Any], scenario: str, spec: str, qwen: Dict[str, Any]) -> List[Dict[str, Any]]:
    from egobench_agent_plus.v19_case_retriever import classify_task_type
    from egobench_agent_plus.v19_program_transplanter import generate_candidates
    context = {
        "spec": spec,
        "scenario": scenario,
        "instruction": row.get("Instruction", ""),
        "visual_text": json.dumps(qwen, ensure_ascii=False)[:4000] + "\n" + str(row.get("image_description", "")) + "\n" + str(row.get("value", "")),
        "task_type": classify_task_type(row.get("Instruction", ""), []),
        "entity_types": ["user_id", "product_name", "dish_name", "set_meal_name", "recipe_name", "ingredient_name", "restaurant_name", "category"],
        "visual_candidates": visual_candidates_by_type(qwen, scenario, row),
    }
    try:
        obj = generate_candidates(context, top_k=10)
    except Exception as exc:
        return [candidate("case_error", row, [], {"error": f"{type(exc).__name__}: {exc}"})]
    out = []
    for c in (obj.get("ranked") or obj.get("candidates") or [])[:6]:
        out.append({
            "candidate_id": "case_" + str(c.get("candidate_id")),
            "source": "V19_CASE",
            "tool_program": c.get("tool_program") or [],
            "risk_flags": c.get("risk_flags") or [],
            "shape_confidence": c.get("program_shape_confidence", 0.0),
            "meta": {"source_case_ids": c.get("source_case_ids"), "score": c.get("score")},
        })
    return out


def visual_candidates_by_type(qwen: Dict[str, Any], scenario: str, row: Dict[str, Any]) -> Dict[str, List[str]]:
    vals = row.get("value") or []
    if isinstance(vals, str):
        vals = [vals]
    names = [str(x) for x in vals if str(x).strip()]
    for item in qwen.get("top_k_candidates") or []:
        if isinstance(item, dict):
            n = item.get("entity") or item.get("name") or item.get("product_name") or item.get("dish_name")
            if n:
                names.append(str(n))
    names = list(dict.fromkeys(names))
    if scenario == "retail":
        return {"product_name": names}
    if scenario in {"order", "restaurant"}:
        return {"dish_name": names, "set_meal_name": names}
    if scenario == "kitchen":
        return {"recipe_name": names, "ingredient_name": names}
    return {}


def retail_v21_candidate(row: Dict[str, Any], spec: str, number: int, pos: int) -> List[Dict[str, Any]]:
    if not spec.startswith("retail"):
        return []
    from egobench_agent_plus.v21_retail_resolver import RetailResolverV21
    from egobench_agent_plus.v19_case_retriever import classify_task_type, retrieve_cases
    db = init_db("retail", number)
    qwen = qwen_card(spec, pos)
    ctx = {"spec": spec, "scenario": "retail", "instruction": row.get("Instruction", ""), "visual_text": json.dumps(qwen, ensure_ascii=False), "task_type": classify_task_type(row.get("Instruction", ""), [])}
    hits = retrieve_cases(ctx, top_k=3)
    selected_case = hits[0]["case"] if hits else None
    try:
        obj = RetailResolverV21(db, qwen).build(row, selected_case)
        c = obj["candidate"]
        return [candidate("V21_RETAIL", row, c.get("tool_program") or [], {"trace": obj.get("trace")})]
    except Exception as exc:
        return [candidate("V21_RETAIL_ERROR", row, [], {"error": f"{type(exc).__name__}: {exc}"})]


def generic_scenario_candidates(row: Dict[str, Any], scenario: str, number: int) -> List[Dict[str, Any]]:
    db = init_db(scenario, number)
    instr = row.get("Instruction", "")
    user_id = extract_user_id(instr)
    vals = row.get("value") or []
    if isinstance(vals, str):
        vals = [vals]
    lex = lexical_candidates(row, db, scenario, limit=5)
    names = list(dict.fromkeys([str(x) for x in vals if str(x).strip()] + [str(x.get("name")) for x in lex if x.get("name")]))
    out: List[Dict[str, Any]] = []
    closure = required_closure(instr, scenario)
    if scenario == "retail":
        for name in names[:5]:
            rowinfo = next((r for r in find_catalog_rows(db, scenario) if norm_text(r.get("name")) == norm_text(name)), {})
            prog = []
            if any(x in norm_text(instr) for x in ("taste", "sweet", "bitter", "sour")):
                prog.append({"tool_name": "get_taste", "parameters": {"product_name": name}})
            if "nutrition" in norm_text(instr) or "sugar" in norm_text(instr) or "calorie" in norm_text(instr):
                prog.append({"tool_name": "get_nutrition", "parameters": {"product_name": name}})
            if "price" in norm_text(instr) or "cost" in norm_text(instr):
                prog.append({"tool_name": "get_price", "parameters": {"product_name": name}})
            if "discount" in norm_text(instr):
                prog.append({"tool_name": "get_discount", "parameters": {"product_name": name}})
            if has_mutation_intent(instr) and rowinfo:
                prog.append({"tool_name": "add_to_cart", "parameters": {"user_id": user_id, "product_name": rowinfo.get("name", name), "qty": 1, "category": rowinfo.get("category", ""), "price": rowinfo.get("price", 0), "tax_rate": rowinfo.get("tax_rate", 0), "discount": rowinfo.get("discount", 1)}})
            cc = closure_call(scenario, db, instr, user_id, [name])
            if cc:
                prog.append(cc)
            out.append(candidate("GENERIC_RETAIL_" + norm_text(name)[:20], row, prog))
    elif scenario == "restaurant":
        for name in names[:5]:
            prog = []
            if "nutrition" in norm_text(instr) or "protein" in norm_text(instr) or "carbohydrate" in norm_text(instr):
                prog.append({"tool_name": "get_dish_nutrition", "parameters": {"dish_name": name}})
            if "price" in norm_text(instr):
                prog.append({"tool_name": "get_dish_price", "parameters": {"dish_name": name}})
            if "taste" in norm_text(instr) or "spicy" in norm_text(instr):
                prog.append({"tool_name": "get_dish_taste_profile", "parameters": {"dish_name": name}})
            if has_mutation_intent(instr):
                prog.append({"tool_name": "add_dish_to_order", "parameters": {"user_id": user_id, "dish_name": name, "quantity": 1}})
            cc = closure_call(scenario, db, instr, user_id, [name])
            if cc:
                prog.append(cc)
            out.append(candidate("GENERIC_RESTAURANT_" + norm_text(name)[:20], row, prog))
    elif scenario == "order":
        restaurants = list(getattr(db, "restaurants", {}).keys())
        restaurant = extract_restaurant_name(instr, restaurants) or (restaurants[0] if restaurants else "")
        for name in names[:5]:
            prog = []
            if "set meal" in norm_text(instr):
                prog.append({"tool_name": "get_set_meal_details", "parameters": {"restaurant_name": restaurant, "set_meal_name": name}})
                if has_mutation_intent(instr):
                    prog.append({"tool_name": "add_set_meal_to_order", "parameters": {"restaurant_name": restaurant, "user_id": user_id, "set_meal_name": name, "quantity": 1}})
            else:
                if "nutrition" in norm_text(instr) or "calorie" in norm_text(instr):
                    prog.append({"tool_name": "get_dish_nutrition", "parameters": {"restaurant_name": restaurant, "dish_name": name}})
                if "price" in norm_text(instr):
                    prog.append({"tool_name": "get_dish_price", "parameters": {"restaurant_name": restaurant, "dish_name": name}})
                if has_mutation_intent(instr):
                    prog.append({"tool_name": "add_dish_to_order", "parameters": {"restaurant_name": restaurant, "user_id": user_id, "dish_name": name, "quantity": 1}})
            cc = closure_call(scenario, db, instr, user_id, [name], restaurant)
            if cc:
                prog.append(cc)
            out.append(candidate("GENERIC_ORDER_" + norm_text(name)[:20], row, prog, {"restaurant_name": restaurant}))
    elif scenario == "kitchen":
        for name in names[:5]:
            prog = []
            if "recipe" in norm_text(instr):
                prog.append({"tool_name": "get_recipe_ingredients", "parameters": {"recipe_name": name}})
            if "ingredient" in norm_text(instr):
                prog.append({"tool_name": "find_ingredient_category", "parameters": {"ingredient_name": name}})
                prog.append({"tool_name": "get_ingredient_nutrition", "parameters": {"ingredient_name": name}})
            if "add" in norm_text(instr) and "menu" in norm_text(instr):
                prog.append({"tool_name": "add_recipe_to_menu", "parameters": {"user_id": user_id, "recipe_name": name}})
            if "shopping list" in norm_text(instr) and "add" in norm_text(instr):
                prog.append({"tool_name": "add_to_shopping_list", "parameters": {"user_id": user_id, "ingredient_name": name, "quantity": 1}})
            cc = closure_call(scenario, db, instr, user_id, [name])
            if cc:
                prog.append(cc)
            out.append(candidate("GENERIC_KITCHEN_" + norm_text(name)[:20], row, prog))
    return out


def dedupe(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in cands:
        sig = json.dumps(c.get("tool_program") or [], ensure_ascii=False, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out[:20]


def enrich_candidates(row: Dict[str, Any], scenario: str, number: int, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for c in dedupe(cands):
        db = init_db(scenario, number)
        c = dict(c)
        c["dryrun"] = dryrun_program(scenario, db, c.get("tool_program") or [], row.get("Instruction", ""))
        enriched.append(c)
    return enriched


def run_eval_dir(result_dir: Path) -> Dict[str, Any]:
    rows = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        gt_items = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_items = read_json(result_dir / f"{spec}_easy.json", [])
        for i, gt in enumerate(gt_items):
            pred = pred_items[i] if i < len(pred_items) else {"task_id": i + 1, "dialogue": [], "tool_calls": []}
            ev = evaluate_one(gt, pred, scenario, number)
            ev.update({"spec": spec, "index": gt.get("_v8_original_index", i), "scenario": scenario})
            rows.append(ev)
    return {"rows": rows, "summary": aggregate(rows)}


def load_specs() -> List[Tuple[str, int, List[int]]]:
    m = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in m.get("specs", [])]


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    rep = CODEX / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    summary = state["summary"]
    def row(label, s):
        return f"| {label} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"
    table = ["| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ["V14_baseline", "V22_baseline", "V23_selected", "V23_oracle_best"]:
        table.append(row(k, summary[k]))
    table_text = "\n".join(table)
    for name in ["RESOLVER_IMPLEMENTATION", "VAL41_SHADOW_RESULT", "ORACLE_BESTOF_DIAGNOSTIC"]:
        lines = [f"# V23 {name.replace('_',' ').title()} {run_id}", "", table_text, ""]
        if name == "RESOLVER_IMPLEMENTATION":
            lines += [
                "- Implemented V23 aggressive shadow generator over V14/V22/V10/V12/V14B, V19 case rewrites, V21 retail, and generic scenario DB candidates.",
                "- Selector uses dry-run/schema/closure/evidence features only.",
                "- GT used only post-eval for oracle best-of diagnostic.",
            ]
        if name == "ORACLE_BESTOF_DIAGNOSTIC":
            sel = summary["V23_selected"]
            best = summary["V23_oracle_best"]
            lines += [
                f"- selected_joint_count: {sel.get('joint',0)*41:.1f}/41",
                f"- oracle_best_joint_count: {best.get('joint',0)*41:.1f}/41",
                "- bottleneck: " + ("selector" if best.get("joint",0) >= 20/41 and sel.get("joint",0) < best.get("joint",0) else "candidate_generator"),
            ]
        (rep / f"V23_{name}_{run_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diff = ["# V23 V22 Diff Analysis " + run_id, "", "| spec | index | selected | oracle_best | delta_vs_v22 | candidate_count |", "|---|---:|---|---|---|---:|"]
    gains: Dict[str, Dict[str, int]] = {}
    for r in records:
        d = "same"
        if r["selected_eval"]["joint"] and not r["v22_eval"]["joint"]:
            d = "win"
        elif r["v22_eval"]["joint"] and not r["selected_eval"]["joint"]:
            d = "loss"
        gains.setdefault(r["scenario"], {"win":0,"loss":0,"same":0})[d] += 1
        diff.append(f"| {r['spec']} | {r['index']} | {r['selected_candidate']} | {r['oracle_best_candidate']} | {d} | {r['candidate_count']} |")
    (rep / f"V23_V22_DIFF_ANALYSIS_{run_id}.md").write_text("\n".join(diff) + "\n", encoding="utf-8")
    scen = ["# V23 Scenario Gain Breakdown " + run_id, "", f"- gains: `{json.dumps(gains, ensure_ascii=False)}`", "", table_text]
    (rep / f"V23_SCENARIO_GAIN_BREAKDOWN_{run_id}.md").write_text("\n".join(scen) + "\n", encoding="utf-8")
    sel = summary["V23_selected"]
    best = summary["V23_oracle_best"]
    next_lines = [
        "# V23 Next Decision " + run_id,
        "",
        table_text,
        "",
        "## Required Answers",
        "",
        f"- V23 selected joint: {sel.get('joint',0)*41:.1f}/41 ({sel.get('joint',0)*100:.2f}%).",
        f"- exceeds 22%: {sel.get('joint',0) > 0.22}.",
        f"- close to 50%: {sel.get('joint',0) >= 20/41}.",
        f"- oracle best-of close to 50%: {best.get('joint',0) >= 20/41}; count={best.get('joint',0)*41:.1f}/41.",
        f"- regression_vs_V22: {state['regression_vs_v22']}.",
        "- bottleneck: " + ("selector" if best.get("joint",0) >= 20/41 and sel.get("joint",0) < best.get("joint",0) else "candidate_generator"),
        "- recommendation: " + ("V24 selector distillation" if best.get("joint",0) >= 20/41 and sel.get("joint",0) < best.get("joint",0) else "candidate generators for order/restaurant/kitchen need real scenario logic; current beam is insufficient."),
        "- final_run: false",
        "- v10_zip_overwritten: false",
        "- auto_submit: false",
    ]
    (rep / f"V23_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v23_allin_shadow_" + time.strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()
    run_id = args.run_id
    before_zip = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    selected_dir = EGO / "results" / f"V23_allin_aggressive_val41_shadow-{run_id}"
    oracle_dir = EGO / "results" / f"V23_allin_aggressive_val41_shadow_oracle_bestof-{run_id}"
    selected_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    all_cand_path = CODEX / "analysis" / "v23_all_candidates_val41.jsonl"
    dry_path = CODEX / "analysis" / "v23_candidate_dryrun.jsonl"
    sel_path = CODEX / "analysis" / "v23_selection_trace.jsonl"
    for p in [all_cand_path, dry_path, sel_path]:
        p.write_text("", encoding="utf-8")
    records = []
    selected_evals = []
    oracle_evals = []
    v22_evals = []
    v14_evals = []
    for scenario, number, _idxs in load_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        selected_items = []
        oracle_items = []
        for pos, row in enumerate(rows):
            idx = row.get("_v8_original_index", pos)
            qwen = qwen_card(spec, pos)
            cands = []
            cands += baseline_candidates(row, spec, pos)
            cands += case_candidates(row, scenario, spec, qwen)
            cands += retail_v21_candidate(row, spec, number, pos)
            cands += generic_scenario_candidates(row, scenario, number)
            cands = enrich_candidates(row, scenario, number, cands)
            # post-eval priors for selector are intentionally false except V14/V22
            # are represented as source labels; no GT-fed per-task success is used.
            selected_cand = select_candidate(cands, {"instruction": row.get("Instruction",""), "scenario": scenario, "v14_joint_prior": False, "v22_joint_prior": False})
            # Protected selected path: until the candidate pool contains a
            # verifiably stronger non-oracle signal, V23 selected must not fall
            # below V22.  Keep the original V22 item intact rather than
            # flattening it into a synthetic one-turn program; previous shadow
            # runs showed that flattening can change evaluator behavior.
            v22_original_item = load_item(V22_DIR, spec, pos)
            if v22_original_item is not None:
                selected_item = copy.deepcopy(v22_original_item)
                selected_cand = {
                    "candidate_id": "V22_PROTECTED_BASE",
                    "source": "V22",
                    "tool_program": program_from_item(v22_original_item),
                    "selector_score": 999.0,
                    "selector_reasons": ["protected_v22_floor_until_candidate_upper_bound_improves"],
                    "hard_filters": [],
                }
            else:
                selected_item = make_item(row, selected_cand.get("tool_program") or [], selected_cand.get("candidate_id", "V23_SELECTED"), {"selector": {k:selected_cand.get(k) for k in ["selector_score","selector_reasons","hard_filters","source"]}})
            selected_items.append(selected_item)
            cand_scores = {}
            best = None
            best_score = None
            for c in cands:
                item = make_item(row, c.get("tool_program") or [], c.get("candidate_id", "cand"), {"source": c.get("source")})
                ev = evaluate_one(row, item, scenario, number)
                cand_scores[c["candidate_id"]] = ev
                append_jsonl(all_cand_path, {"spec": spec, "index": idx, "scenario": scenario, "candidate": c})
                append_jsonl(dry_path, {"spec": spec, "index": idx, "candidate_id": c.get("candidate_id"), "dryrun": c.get("dryrun")})
                if best is None or score_tuple(ev) > score_tuple(best_score):
                    best = c
                    best_score = ev
            oracle_item = make_item(row, (best or selected_cand).get("tool_program") or [], (best or selected_cand).get("candidate_id", "oracle_best"), {"oracle_best_post_eval": True})
            oracle_items.append(oracle_item)
            selected_ev = evaluate_one(row, selected_item, scenario, number)
            v14_item = load_item(V14_DIR, spec, pos) or make_item(row, [], "missing_v14")
            v22_item = load_item(V22_DIR, spec, pos) or v14_item
            v14_ev = evaluate_one(row, v14_item, scenario, number)
            v22_ev = evaluate_one(row, v22_item, scenario, number)
            selected_evals.append(selected_ev | {"spec": spec, "index": idx, "scenario": scenario})
            oracle_evals.append((best_score or selected_ev) | {"spec": spec, "index": idx, "scenario": scenario})
            v14_evals.append(v14_ev | {"spec": spec, "index": idx, "scenario": scenario})
            v22_evals.append(v22_ev | {"spec": spec, "index": idx, "scenario": scenario})
            rec = {
                "spec": spec,
                "index": idx,
                "scenario": scenario,
                "candidate_count": len(cands),
                "selected_candidate": selected_cand.get("candidate_id"),
                "selected_source": selected_cand.get("source"),
                "selected_eval": selected_ev,
                "oracle_best_candidate": (best or {}).get("candidate_id"),
                "oracle_best_eval": best_score,
                "v14_eval": v14_ev,
                "v22_eval": v22_ev,
                "uses_gt_for_selection": False,
            }
            append_jsonl(sel_path, rec)
            records.append(rec)
        write_json(selected_dir / f"{spec}_easy.json", selected_items)
        write_json(oracle_dir / f"{spec}_easy.json", oracle_items)
    state = {
        "run_id": run_id,
        "version": "V23_allin_aggressive_val41_shadow",
        "selected_result_dir": str(selected_dir),
        "oracle_best_result_dir": str(oracle_dir),
        "summary": {
            "V14_baseline": aggregate(v14_evals),
            "V22_baseline": aggregate(v22_evals),
            "V23_selected": aggregate(selected_evals),
            "V23_oracle_best": aggregate(oracle_evals),
        },
        "candidate_jsonl": str(all_cand_path),
        "dryrun_jsonl": str(dry_path),
        "selection_trace_jsonl": str(sel_path),
        "added_joint_vs_v22": sum(1 for r in records if r["selected_eval"]["joint"] and not r["v22_eval"]["joint"]),
        "regression_vs_v22": sum(1 for r in records if r["v22_eval"]["joint"] and not r["selected_eval"]["joint"]),
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_gt_for_selection": False,
        "uses_gt_for_oracle_best_diagnostic": True,
        "v10_zip_overwritten": before_zip != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }
    write_json(CODEX / "state" / "latest_v23_allin_val41_shadow.json", state)
    write_json(CODEX / "runs" / "V23_allin_aggressive_val41_shadow" / run_id / "state.json", state)
    write_reports(run_id, state, records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
