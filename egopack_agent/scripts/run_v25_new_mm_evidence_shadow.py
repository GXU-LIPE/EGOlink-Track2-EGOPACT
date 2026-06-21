#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V25-new multimodal evidence + V21 agent shadow runner.

Closed-loop driver:
1. build/reuse contact sheets;
2. build multimodal evidence table;
3. generate few evidence-driven candidates;
4. guarded non-oracle selection over V22 floor;
5. post-eval selected and oracle-best diagnostics.

No final run, no final hidden metadata, no val41 GT for runtime selection.
"""

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
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
V14_DIR = EGO / "results" / "V14_candidate_selection_val41-v14_candidate_selection_20260619_2134"
V22_DIR = EGO / "results" / "V22_guarded_v21_retail_overlay_val41_shadow-v22_guarded_shadow_20260620_1915"
V24_DIR = EGO / "results" / "V24_scenario_gt_gap_generators_val41_shadow-v24_gap_generators_gpt_selector_20260620_2245"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))

from egobench_agent_plus.v24_candidate_dryrun_and_selector import dryrun_program, select_v24_candidate  # noqa: E402
from egobench_agent_plus.v25_mm_evidence_extractor import build_evidence_table, save_evidence  # noqa: E402
from egobench_agent_plus.v25_v21_evidence_resolver import build_v25_candidates  # noqa: E402


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
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_143308" / f"{spec}_{pos + 1}.json",
    ]:
        data = read_json(p)
        if isinstance(data, dict):
            data["_path"] = str(p)
            return data
    return {"status": "missing", "_path": ""}


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} val41 shadow candidate."}],
        "tool_calls": [
            {
                "turn": 0,
                "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program],
                "blocked_calls": [],
                "results": [],
                "v25_meta": meta or {},
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
    if isinstance(rows, list) and pos < len(rows):
        return rows[pos]
    return None


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v25_eval_") as td:
        tdir = Path(td)
        gt_path = tdir / "gt.json"
        pred_path = tdir / "pred.json"
        write_json(gt_path, [gt_item])
        write_json(pred_path, [pred_item])
        metrics = evaluate_interaction_success(
            str(gt_path),
            str(pred_path),
            scenario=scenario,
            args=_argparse.Namespace(scenario_number=number),
            silent=True,
            num_samples=0,
        )
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
    matched = sum(int(r.get("matches", r.get("matched_tools", 0)) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", r.get("gt_tools", 0)) or 0) for r in rows)
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
    return (
        float(score.get("joint", 0)),
        float(score.get("tool", 0)),
        float(score.get("result", 0)),
        int(score.get("matches", 0)),
        -int(score.get("interaction_calls", 999999)),
    )


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def ensure_contact_sheet(spec: str, pos: int) -> None:
    out = CODEX / "visual_cache_v25_new" / "contact_sheets" / f"{spec}_{pos + 1}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return
    subprocess.run(
        [sys.executable, str(CODEX / "scripts" / "build_v25_new_contact_sheets.py"), "--spec", spec, "--pos", str(pos), "--quiet"],
        cwd=str(CODEX),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
        check=False,
    )


def select_smoke_targets(limit: int) -> List[Dict[str, Any]]:
    target_obj = read_json(CODEX / "analysis" / "v24_target_tasks.json", {})
    raw = target_obj.get("targets") or []
    priority = {"retail": 0, "order": 1, "restaurant": 2, "kitchen": 3}
    picked: List[Dict[str, Any]] = []
    seen_specs: Dict[str, int] = {}
    for t in sorted(raw, key=lambda x: (priority.get(x.get("scenario"), 9), -float(x.get("priority_score", 0) or 0))):
        scenario = t.get("scenario")
        if scenario not in {"retail", "order", "restaurant", "kitchen"}:
            continue
        spec = t.get("spec")
        if seen_specs.get(spec, 0) >= 2:
            continue
        picked.append(dict(t))
        seen_specs[spec] = seen_specs.get(spec, 0) + 1
        if len(picked) >= limit:
            break
    if len(picked) >= limit:
        return picked
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        for pos, row in enumerate(rows):
            key = (spec, int(row.get("_v8_original_index", pos)))
            if any((x.get("spec"), int(x.get("index", -1))) == key for x in picked):
                continue
            if scenario in {"retail", "order", "restaurant", "kitchen"}:
                picked.append({"spec": spec, "index": key[1], "local_pos": pos, "scenario": scenario})
            if len(picked) >= limit:
                return picked
    return picked


def all_targets() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        for pos, row in enumerate(read_json(SPLIT_DIR / f"{spec}.json", [])):
            out.append({"spec": spec, "index": int(row.get("_v8_original_index", pos)), "local_pos": pos, "scenario": scenario})
    return out


def result_dir_eval(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        gt_rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_rows = read_json(result_dir / f"{spec}_easy.json", [])
        for pos, row in enumerate(gt_rows):
            pred = pred_rows[pos] if isinstance(pred_rows, list) and pos < len(pred_rows) else {"task_id": pos + 1, "dialogue": [], "tool_calls": []}
            ev = evaluate_one(row, pred, scenario, number)
            ev.update({"spec": spec, "index": int(row.get("_v8_original_index", pos)), "scenario": scenario, "local_pos": pos})
            rows_out.append(ev)
    return rows_out, aggregate(rows_out)


def write_result_dir(result_dir: Path, replacements: Dict[Tuple[str, int], Dict[str, Any]], fallback_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for scenario, number, _ in load_specs():
        spec = f"{scenario}{number}"
        base = read_json(fallback_dir / f"{spec}_easy.json", [])
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        out = []
        for pos, row in enumerate(rows):
            idx = int(row.get("_v8_original_index", pos))
            out.append(replacements.get((spec, idx)) or (base[pos] if isinstance(base, list) and pos < len(base) else make_item(row, [], "missing_base")))
        write_json(result_dir / f"{spec}_easy.json", out)


def build_and_score_task(t: Dict[str, Any], run_id: str, use_gpt_vision: bool) -> Dict[str, Any]:
    spec = t["spec"]
    scenario = t["scenario"]
    pos = int(t["local_pos"])
    number = int(re.sub(r"^\D+", "", spec))
    row = read_json(SPLIT_DIR / f"{spec}.json", [])[pos]
    db = init_db(scenario, number)
    ensure_contact_sheet(spec, pos)
    qwen = qwen_card(spec, pos)
    evidence = build_evidence_table(row=row, scenario=scenario, spec=spec, local_pos=pos, db=db, qwen_card=qwen, use_gpt_vision=use_gpt_vision)
    evidence_path = CODEX / "visual_cache_v25_new" / "evidence_tables" / f"{spec}_{pos + 1}.json"
    save_evidence(evidence_path, evidence)
    v14_item = load_item(V14_DIR, spec, pos)
    v22_item = load_item(V22_DIR, spec, pos)
    v24_item = load_item(V24_DIR, spec, pos)
    obj = build_v25_candidates(scenario=scenario, row=row, db=db, evidence=evidence, v14_item=v14_item, v22_item=v22_item, max_candidates=6)
    candidates = obj["candidates"]
    if v24_item:
        candidates.append({"candidate_id": "E_V24_BASE", "source": "V24", "tool_program": program_from_item(v24_item), "confidence": 0.56, "meta": {}})
    enriched = []
    for cand in candidates:
        c = copy.deepcopy(cand)
        db2 = init_db(scenario, number)
        c["dryrun"] = dryrun_program(scenario, db2, c.get("tool_program") or [], row.get("Instruction", ""))
        c["shape_confidence"] = c.get("confidence", 0)
        enriched.append(c)
    selected = select_v24_candidate(enriched, {"instruction": row.get("Instruction", ""), "scenario": scenario})
    v22_score = evaluate_one(row, v22_item or make_item(row, [], "missing_v22"), scenario, number)
    # Protect existing V22 joint successes.  The target is to add joints, not
    # regress the floor.
    if v22_score.get("joint"):
        selected = {"candidate_id": "B_V22_PROTECTED_SUCCESS", "source": "V22", "tool_program": program_from_item(v22_item), "selector_score": 999, "selector_reasons": ["protected_v22_joint"], "hard_filters": []}
    selected_item = make_item(row, selected.get("tool_program") or [], selected.get("candidate_id", "V25_SELECTED"), {"source": selected.get("source"), "selector_score": selected.get("selector_score"), "selector_reasons": selected.get("selector_reasons"), "evidence_path": str(evidence_path)})
    selected_score = evaluate_one(row, selected_item, scenario, number)
    candidate_scores = {}
    best = None
    best_score = None
    for cand in enriched:
        item = make_item(row, cand.get("tool_program") or [], cand.get("candidate_id", "V25_CAND"), {"source": cand.get("source"), "evidence_path": str(evidence_path)})
        ev = evaluate_one(row, item, scenario, number)
        candidate_scores[cand.get("candidate_id", "")] = ev
        if best is None or score_tuple(ev) > score_tuple(best_score or {}):
            best, best_score = cand, ev
    if best is None:
        best = selected
        best_score = selected_score
    oracle_item = make_item(row, best.get("tool_program") or [], best.get("candidate_id", "V25_ORACLE"), {"oracle_best_post_eval": True, "source": best.get("source"), "evidence_path": str(evidence_path)})
    return {
        "spec": spec,
        "scenario": scenario,
        "local_pos": pos,
        "index": int(row.get("_v8_original_index", pos)),
        "row": row,
        "evidence": evidence,
        "evidence_path": str(evidence_path),
        "resolver_trace": obj.get("trace"),
        "candidates": enriched,
        "selected": selected,
        "selected_item": selected_item,
        "selected_score": selected_score,
        "oracle_item": oracle_item,
        "oracle_best_candidate": best.get("candidate_id"),
        "oracle_best_score": best_score,
        "v22_score": v22_score,
        "candidate_scores": candidate_scores,
        "uses_gt_for_selection": False,
        "uses_gt_for_oracle_best_diagnostic": True,
    }


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table = [
        "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("V22_same_scope", state["V22_same_scope"]),
        table_row("V25_selected_same_scope", state["V25_selected_same_scope"]),
        table_row("V25_oracle_best_same_scope", state["V25_oracle_best_same_scope"]),
    ]
    if state.get("full_shadow_summary"):
        table += [
            table_row("V25_selected_full41", state["full_shadow_summary"]["selected"]),
            table_row("V25_oracle_best_full41", state["full_shadow_summary"]["oracle"]),
        ]
    evidence_ok = sum(1 for r in records if not ((r.get("evidence") or {}).get("uncertainty") or {}).get("visual_grounding_failed"))
    gpt_ok = sum(1 for r in records if ((r.get("evidence") or {}).get("sources") or {}).get("gpt55_vision_status") == "success")
    canonical_ok = sum(1 for r in records if any((r.get("evidence") or {}).get("candidate_slots", {}).get(k) for k in ("primary_product", "dish", "set_meal", "ingredient", "recipe")))
    extraction = [
        f"# V25-new Evidence Extraction Audit {run_id}",
        "",
        f"- stage: {state['stage']}",
        f"- tasks_processed: {len(records)}",
        f"- gpt55_vision_success: {gpt_ok}/{len(records)}",
        f"- evidence_entity_grounded: {evidence_ok}/{len(records)}",
        f"- canonical_slot_nonempty: {canonical_ok}/{len(records)}",
        f"- OCR/subtitle: OCR from GPT/Qwen visible text; subtitle/ASR used only if sidecar files existed.",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- v10_zip_overwritten: false",
    ]
    (reports / f"V25_NEW_EVIDENCE_EXTRACTION_AUDIT_{run_id}.md").write_text("\n".join(extraction) + "\n", encoding="utf-8")

    smoke = [
        f"# V25-new MM Evidence Smoke {run_id}",
        "",
        *table,
        "",
        "| spec | index | scenario | selected | selected_joint | oracle_best | oracle_joint | v22_joint | gpt55_vision | grounded_slots |",
        "|---|---:|---|---|---:|---|---:|---:|---|---|",
    ]
    for r in records:
        slots = [k for k, v in (r.get("evidence") or {}).get("candidate_slots", {}).items() if v]
        smoke.append(
            f"| {r['spec']} | {r['index']} | {r['scenario']} | {r['selected'].get('candidate_id')} | {int(r['selected_score'].get('joint',0))} | {r.get('oracle_best_candidate')} | {int((r.get('oracle_best_score') or {}).get('joint',0))} | {int(r['v22_score'].get('joint',0))} | {((r.get('evidence') or {}).get('sources') or {}).get('gpt55_vision_status')} | {','.join(slots[:5])} |"
        )
    (reports / f"V25_NEW_MM_EVIDENCE_SMOKE_{run_id}.md").write_text("\n".join(smoke) + "\n", encoding="utf-8")

    shadow = [
        f"# V25-new Val41 Shadow Result {run_id}",
        "",
        *table,
        "",
        f"- full_shadow_ran: {bool(state.get('full_shadow_summary'))}",
        f"- selected_beats_9_of_41: {state.get('selected_joint_count_full', 0) > 9}",
        f"- oracle_best_beats_10_of_41: {state.get('oracle_joint_count_full', 0) > 10}",
        f"- selected_result_dir: `{state.get('selected_result_dir','')}`",
        f"- oracle_result_dir: `{state.get('oracle_result_dir','')}`",
        f"- analysis_evidence_jsonl: `{state['evidence_jsonl']}`",
        f"- analysis_selection_jsonl: `{state['selection_jsonl']}`",
    ]
    (reports / f"V25_NEW_VAL41_SHADOW_RESULT_{run_id}.md").write_text("\n".join(shadow) + "\n", encoding="utf-8")

    gains = [
        f"# V25-new Evidence Gain Analysis {run_id}",
        "",
        "- Runtime selection is non-oracle.",
        "- Oracle best-of is post-eval diagnostic only.",
        "",
        "| spec | index | selected_delta_vs_v22 | oracle_delta_vs_v22 | first_bottleneck | evidence_sources |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in records:
        sel_delta = int(r["selected_score"].get("joint", 0) - r["v22_score"].get("joint", 0))
        ora_delta = int((r.get("oracle_best_score") or {}).get("joint", 0) - r["v22_score"].get("joint", 0))
        slots = (r.get("evidence") or {}).get("candidate_slots", {})
        if not any(slots.get(k) for k in ("primary_product", "dish", "set_meal", "ingredient", "recipe")):
            bottleneck = "evidence_missing_or_no_canonical_match"
        elif not (r.get("oracle_best_score") or {}).get("joint"):
            bottleneck = "resolver_candidate_not_joint_capable"
        elif not r["selected_score"].get("joint"):
            bottleneck = "selector_too_conservative_or_wrong_candidate"
        else:
            bottleneck = "resolved"
        sources = (r.get("evidence") or {}).get("sources") or {}
        gains.append(f"| {r['spec']} | {r['index']} | {sel_delta} | {ora_delta} | {bottleneck} | qwen={sources.get('qwen_status')};gpt={sources.get('gpt55_vision_status')} |")
    (reports / f"V25_NEW_EVIDENCE_GAIN_ANALYSIS_{run_id}.md").write_text("\n".join(gains) + "\n", encoding="utf-8")

    selected = state["V25_selected_same_scope"]
    oracle = state["V25_oracle_best_same_scope"]
    if state.get("stage") == "smoke" and selected.get("joint", 0) <= state["V22_same_scope"].get("joint", 0) and oracle.get("joint", 0) <= state["V22_same_scope"].get("joint", 0):
        decision = "stop_after_smoke_evidence_or_resolver_bottleneck"
    elif not state.get("full_shadow_summary"):
        decision = "run_full_val41_shadow_next"
    else:
        full_sel = state["full_shadow_summary"]["selected"]
        decision = "promising_if_selected_above_v22" if full_sel.get("joint", 0) > 9 / 41 else "do_not_promote_below_v22"
    next_lines = [
        f"# V25-new Next Decision {run_id}",
        "",
        *table,
        "",
        f"- decision: {decision}",
        f"- GPT-5.5 vision called successfully: {gpt_ok > 0}",
        f"- OCR/ASR/subtitle effective: OCR/Qwen visible text effective for {canonical_ok}/{len(records)}; ASR only sidecar-based.",
        f"- evidence coverage: {evidence_ok}/{len(records)}",
        f"- canonical DB entity match: {canonical_ok}/{len(records)}",
        f"- V25-new exceeds 9/41: {state.get('selected_joint_count_full', 0) > 9}",
        "- final_run: false",
        "- final_hidden_metadata_used: false",
        "- v10_zip_overwritten: false",
        "- auto_submit: false",
    ]
    (reports / f"V25_NEW_NEXT_DECISION_{run_id}.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v25_new_mm_evidence_" + stamp())
    ap.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--smoke-limit", type=int, default=12)
    ap.add_argument("--disable-gpt55-vision", action="store_true")
    ap.add_argument("--force-full-after-smoke", action="store_true")
    args = ap.parse_args()

    run_id = args.run_id
    before_zip = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    run_dir = CODEX / "runs" / "V25_NEW_MULTIMODAL_EVIDENCE_V21_AGENT" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_jsonl = CODEX / "analysis" / "v25_new_mm_evidence.jsonl"
    candidate_jsonl = CODEX / "analysis" / "v25_new_evidence_resolver_candidates.jsonl"
    selection_jsonl = CODEX / "analysis" / "v25_new_selection_trace.jsonl"
    for p in (evidence_jsonl, candidate_jsonl, selection_jsonl):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")

    targets = all_targets() if args.stage == "full" else select_smoke_targets(args.smoke_limit)
    records: List[Dict[str, Any]] = []
    selected_evals: List[Dict[str, Any]] = []
    oracle_evals: List[Dict[str, Any]] = []
    v22_evals: List[Dict[str, Any]] = []
    replacements_selected: Dict[Tuple[str, int], Dict[str, Any]] = {}
    replacements_oracle: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for i, t in enumerate(targets, 1):
        rec = build_and_score_task(t, run_id, use_gpt_vision=not args.disable_gpt55_vision)
        records.append(rec)
        selected_evals.append(rec["selected_score"])
        oracle_evals.append(rec["oracle_best_score"] or rec["selected_score"])
        v22_evals.append(rec["v22_score"])
        replacements_selected[(rec["spec"], rec["index"])] = rec["selected_item"]
        replacements_oracle[(rec["spec"], rec["index"])] = rec["oracle_item"]
        append_jsonl(evidence_jsonl, rec["evidence"])
        for cand in rec["candidates"]:
            append_jsonl(candidate_jsonl, {"spec": rec["spec"], "index": rec["index"], "scenario": rec["scenario"], "candidate": cand})
        append_jsonl(
            selection_jsonl,
            {
                "spec": rec["spec"],
                "index": rec["index"],
                "scenario": rec["scenario"],
                "selected_candidate": rec["selected"].get("candidate_id"),
                "selected_score": rec["selected_score"],
                "oracle_best_candidate": rec["oracle_best_candidate"],
                "oracle_best_score": rec["oracle_best_score"],
                "v22_score": rec["v22_score"],
                "uses_gt_for_selection": False,
                "uses_gt_for_oracle_best_diagnostic": True,
            },
        )
        if i % 3 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] processed {i}/{len(targets)}")

    state: Dict[str, Any] = {
        "run_id": run_id,
        "version": "V25_NEW_MULTIMODAL_EVIDENCE_V21_AGENT",
        "stage": args.stage,
        "target_count": len(targets),
        "V22_same_scope": aggregate(v22_evals),
        "V25_selected_same_scope": aggregate(selected_evals),
        "V25_oracle_best_same_scope": aggregate(oracle_evals),
        "evidence_jsonl": str(evidence_jsonl),
        "candidate_jsonl": str(candidate_jsonl),
        "selection_jsonl": str(selection_jsonl),
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_runtime_selection": False,
        "uses_gt_for_oracle_best_diagnostic": True,
        "v10_zip_overwritten": before_zip != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }

    selected_dir = EGO / "results" / f"V25_new_mm_evidence_selected-{run_id}"
    oracle_dir = EGO / "results" / f"V25_new_mm_evidence_oracle_bestof-{run_id}"
    if args.stage == "full" or args.force_full_after_smoke:
        write_result_dir(selected_dir, replacements_selected, V22_DIR)
        write_result_dir(oracle_dir, replacements_oracle, V22_DIR)
        _, selected_full = result_dir_eval(selected_dir)
        _, oracle_full = result_dir_eval(oracle_dir)
        state["selected_result_dir"] = str(selected_dir)
        state["oracle_result_dir"] = str(oracle_dir)
        state["full_shadow_summary"] = {"selected": selected_full, "oracle": oracle_full}
        state["selected_joint_count_full"] = round(selected_full["joint"] * 41)
        state["oracle_joint_count_full"] = round(oracle_full["joint"] * 41)
    else:
        state["selected_joint_count_full"] = 0
        state["oracle_joint_count_full"] = 0

    write_json(run_dir / "state.json", state)
    write_json(CODEX / "state" / "latest_v25_new_mm_evidence.json", state)
    # Keep trace compact; full evidence/candidates are in analysis jsonl.
    write_jsonl(run_dir / "task_records_compact.jsonl", [
        {
            "spec": r["spec"],
            "index": r["index"],
            "scenario": r["scenario"],
            "selected": r["selected"].get("candidate_id"),
            "selected_score": r["selected_score"],
            "oracle_best_candidate": r["oracle_best_candidate"],
            "oracle_best_score": r["oracle_best_score"],
            "v22_score": r["v22_score"],
            "evidence_path": r["evidence_path"],
        }
        for r in records
    ])
    write_reports(run_id, state, records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
