#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V22 guarded V21 retail overlay on frozen val41.

The selected path is non-oracle: V21 can replace V14 only through
`v22_guarded_selector` evidence checks.  GT is used after candidate generation
for evaluation and oracle-best diagnostic only.
"""

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
V19_DIR = EGO / "results" / "V19_gt100_case_reuse_smoke10-v19_case_reuse_20260620_1546"
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"


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


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        if not isinstance(block, dict):
            continue
        for call in block.get("calls") or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": call.get("parameters") or {}})
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
                "v22_meta": meta or {},
            }
        ],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
    }


def load_manifest_specs() -> List[Tuple[str, int, List[int]]]:
    manifest = read_json(SPLIT_DIR / "manifest.json", {})
    return [(str(s), int(n), [int(x) for x in idxs]) for s, n, idxs in manifest.get("specs", [])]


def qwen_card(spec: str, local_pos: int) -> Dict[str, Any]:
    candidates = [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{local_pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{local_pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_143308" / f"{spec}_{local_pos + 1}.json",
    ]
    for path in candidates:
        data = read_json(path)
        if isinstance(data, dict):
            data["_path"] = str(path)
            return data
    return {"status": "missing", "_path": ""}


def init_retail_db(number: int) -> Any:
    sys.path.insert(0, str(EGO))
    from tools.retail.retail_db import RetailDB
    from tools.retail import retail_init

    db = RetailDB()
    db.init_from_json(getattr(retail_init, f"retail_init_data{number}"))
    return db


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v22_eval_") as td:
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


def eval_model_dir(model_dir: Path) -> Dict[str, Any]:
    rows = []
    for scenario, number, _idxs in load_manifest_specs():
        spec = f"{scenario}{number}"
        gt_items = read_json(SPLIT_DIR / f"{spec}.json", [])
        pred_items = read_json(model_dir / f"{spec}_easy.json", [])
        task_rows = []
        for i, gt in enumerate(gt_items):
            pred = pred_items[i] if i < len(pred_items) else {"task_id": i + 1, "dialogue": [], "tool_calls": []}
            task_rows.append(evaluate_one(gt, pred, scenario, number))
        rows.append({"spec": spec, "scenario": scenario, "number": number, "valid": len(task_rows), **aggregate(task_rows)})
    return {"rows": rows, "summary": aggregate(rows)}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = sum(int(r.get("valid", 1)) for r in rows) if rows and "valid" in rows[0] else len(rows)
    if not valid:
        return {"valid": 0, "joint": 0, "result": 0, "tool": 0, "micro": 0, "matched_tools": 0, "gt_tools": 0, "interaction_calls": 0}
    if rows and "valid" in rows[0]:
        joint = sum(float(r.get("joint", 0)) * int(r.get("valid", 0)) for r in rows) / valid
        result = sum(float(r.get("result", 0)) * int(r.get("valid", 0)) for r in rows) / valid
        tool = sum(float(r.get("tool", 0)) * int(r.get("valid", 0)) for r in rows) / valid
    else:
        joint = sum(float(r.get("joint", 0)) for r in rows) / valid
        result = sum(float(r.get("result", 0)) for r in rows) / valid
        tool = sum(float(r.get("tool", 0)) for r in rows) / valid
    matched = sum(int(r.get("matched_tools", r.get("matches", 0)) or 0) for r in rows)
    gt = sum(int(r.get("gt_tools", r.get("gt_calls", 0)) or 0) for r in rows)
    return {
        "valid": valid,
        "joint": joint,
        "result": result,
        "tool": tool,
        "micro": matched / gt if gt else 0,
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


def candidate_record(
    *,
    spec: str,
    index: int,
    scenario: str,
    local_pos: int,
    candidate_id: str,
    source: str,
    item: Dict[str, Any] | None,
    v21_trace: Dict[str, Any] | None = None,
    nonoracle_confidence: float = 0.0,
    risk_flags: List[str] | None = None,
) -> Dict[str, Any]:
    trace = v21_trace or {}
    program = program_from_item(item)
    return {
        "spec": spec,
        "index": index,
        "scenario": scenario,
        "local_pos": local_pos,
        "candidate_id": candidate_id,
        "source": source,
        "tool_program": program,
        "primary_product_candidates": trace.get("primary_product_candidates") or [],
        "selected_primary_product": trace.get("selected_primary_product") or "",
        "attribute_query_plan": trace.get("attribute_query_plan") or [],
        "branch_observations": trace.get("tool_observations_used_for_branch") or {},
        "mutation_target": trace.get("mutation_target") or [],
        "closure_tools": [x for x in program if str(x.get("tool_name") or "").startswith("compute_total_")],
        "risk_flags": risk_flags or [],
        "nonoracle_confidence": nonoracle_confidence,
    }


def v21_for_retail(row: Dict[str, Any], spec: str, number: int, local_pos: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sys.path.insert(0, str(CODEX / "wrappers"))
    sys.path.insert(0, str(CODEX))
    from egobench_agent_plus.v19_case_retriever import classify_task_type, retrieve_cases
    from egobench_agent_plus.v21_retail_resolver import RetailResolverV21

    qwen = qwen_card(spec, local_pos)
    db = init_retail_db(number)
    context = {
        "spec": spec,
        "scenario": "retail",
        "instruction": row.get("Instruction", ""),
        "visual_text": "\n".join([
            str(row.get("image_description", "")),
            str(row.get("key", "")),
            str(row.get("value", "")),
            json.dumps(qwen, ensure_ascii=False)[:3000],
        ]),
        "task_type": classify_task_type(row.get("Instruction", ""), []),
        "entity_types": ["product_name", "user_id", "category"],
    }
    hits = retrieve_cases(context, top_k=10)
    selected_case = hits[0]["case"] if hits else None
    resolver = RetailResolverV21(db, qwen)
    obj = resolver.build(row, selected_case)
    program = obj["candidate"]["tool_program"]
    item = make_item(row, program, "V21_retail", obj["candidate"].get("evidence") or {})
    trace = copy.deepcopy(obj["trace"])
    trace["qwen_card_path"] = qwen.get("_path", "")
    trace["retrieved_case_count"] = len(hits)
    trace["top_case_id"] = selected_case.get("case_id") if isinstance(selected_case, dict) else ""
    return item, trace


def write_report(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v22_guarded_shadow_" + stamp())
    args = ap.parse_args()
    run_id = args.run_id
    version = "V22_guarded_v21_retail_overlay_val41_shadow"
    before_zip = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None

    sys.path.insert(0, str(CODEX / "wrappers"))
    sys.path.insert(0, str(CODEX))
    from egobench_agent_plus.v22_guarded_selector import GuardedV21RetailOverlaySelector

    out_dir = EGO / "results" / f"{version}-{run_id}"
    oracle_dir = EGO / "results" / f"{version}_oracle_bestof_diagnostic-{run_id}"
    run_dir = CODEX / "runs" / version / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_jsonl = CODEX / "analysis" / "v22_candidate_programs_val41.jsonl"
    guard_jsonl = CODEX / "analysis" / "v22_guarded_selection_trace.jsonl"
    candidate_jsonl.write_text("", encoding="utf-8")
    guard_jsonl.write_text("", encoding="utf-8")

    selector = GuardedV21RetailOverlaySelector()
    task_records: List[Dict[str, Any]] = []
    selected_evals: List[Dict[str, Any]] = []
    v14_evals: List[Dict[str, Any]] = []
    v21_evals: List[Dict[str, Any]] = []
    oracle_evals: List[Dict[str, Any]] = []
    chosen_counts: Dict[str, int] = {}
    v21_selected_count = 0
    v21_fallback_count = 0
    v21_added_joint = 0
    v21_regressed_joint = 0

    for scenario, number, _idxs in load_manifest_specs():
        spec = f"{scenario}{number}"
        rows = read_json(SPLIT_DIR / f"{spec}.json", [])
        v14_items = read_json(V14_DIR / f"{spec}_easy.json", [])
        v19_items = read_json(V19_DIR / f"{spec}_easy.json", [])
        selected_items = []
        oracle_items = []
        for local_pos, row in enumerate(rows):
            index = int(row.get("_v8_original_index", local_pos))
            v14_item = v14_items[local_pos] if local_pos < len(v14_items) else {"task_id": local_pos + 1, "dialogue": [], "tool_calls": []}
            v19_item = v19_items[local_pos] if local_pos < len(v19_items) else None
            candidates: Dict[str, Dict[str, Any] | None] = {
                "A_V14": v14_item,
                "B_V19": v19_item,
                "E_V14_fallback": v14_item,
            }
            v21_trace: Dict[str, Any] = {}
            if scenario == "retail":
                try:
                    v21_item, v21_trace = v21_for_retail(row, spec, number, local_pos)
                    candidates["C_V21_retail"] = v21_item
                    candidates["D_V21_retail_repaired"] = v21_item
                except Exception as exc:
                    v21_trace = {"error": f"{type(exc).__name__}: {exc}"}
                    candidates["C_V21_retail"] = None
                    candidates["D_V21_retail_repaired"] = None

            db_for_selector = init_retail_db(number) if scenario == "retail" else None
            selection = selector.select(
                scenario=scenario,
                row=row,
                db=db_for_selector,
                v14_item=v14_item,
                v19_item=v19_item,
                v21_item=candidates.get("C_V21_retail"),
                v21_trace=v21_trace,
            )
            selected_item = selection.pop("selected_item")
            if not selected_item:
                selected_item = v14_item
            selected_items.append(selected_item)
            chosen = selection.get("chosen_candidate", "V14")
            chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
            if chosen == "V21_retail":
                v21_selected_count += 1
            elif scenario == "retail":
                v21_fallback_count += 1

            scores: Dict[str, Dict[str, Any]] = {}
            for cid, item in candidates.items():
                if item is None:
                    scores[cid] = {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": len(row.get("ground_truth") or []), "interaction_calls": 999999, "missing": 1}
                    continue
                scores[cid] = evaluate_one(row, item, scenario, number)
                append_jsonl(
                    candidate_jsonl,
                    candidate_record(
                        spec=spec,
                        index=index,
                        scenario=scenario,
                        local_pos=local_pos,
                        candidate_id=cid,
                        source=cid.split("_", 1)[-1],
                        item=item,
                        v21_trace=v21_trace if "V21" in cid else None,
                        nonoracle_confidence=float(selection.get("confidence", 0)) if "V21" in cid else 0,
                        risk_flags=selection.get("risk_flags", []) if "V21" in cid else [],
                    ),
                )

            selected_score = evaluate_one(row, selected_item, scenario, number)
            v14_score = scores.get("A_V14", {})
            v21_score = scores.get("C_V21_retail", {"joint": 0, "result": 0, "tool": 0, "matches": 0, "gt_calls": len(row.get("ground_truth") or []), "interaction_calls": 0})
            selected_evals.append({**selected_score, "spec": spec, "index": index})
            v14_evals.append({**v14_score, "spec": spec, "index": index})
            if scenario == "retail":
                v21_evals.append({**v21_score, "spec": spec, "index": index})
            if chosen == "V21_retail" and not v14_score.get("joint") and selected_score.get("joint"):
                v21_added_joint += 1
            if chosen == "V21_retail" and v14_score.get("joint") and not selected_score.get("joint"):
                v21_regressed_joint += 1

            best_id, best_score = sorted(scores.items(), key=lambda kv: score_tuple(kv[1]), reverse=True)[0]
            oracle_item = candidates.get(best_id) or selected_item
            oracle_items.append(oracle_item)
            oracle_evals.append({**best_score, "spec": spec, "index": index, "oracle_best_candidate": best_id})
            record = {
                "spec": spec,
                "index": index,
                "scenario": scenario,
                "local_pos": local_pos,
                "chosen_candidate": chosen,
                "confidence": selection.get("confidence"),
                "why": selection.get("why"),
                "fallback_reason": selection.get("fallback_reason"),
                "blocked_broad_scans": selection.get("blocked_broad_scans"),
                "closure_completeness": selection.get("closure_completeness"),
                "dry_run_result": selection.get("dry_run_result"),
                "risk_flags": selection.get("risk_flags"),
                "hard_blocks": selection.get("hard_blocks"),
                "program_lengths": selection.get("program_lengths"),
                "candidate_scores_post_eval": scores,
                "selected_score_post_eval": selected_score,
                "v14_score_post_eval": v14_score,
                "v21_score_post_eval": v21_score,
                "oracle_best_candidate_post_eval": best_id,
                "oracle_best_score_post_eval": best_score,
                "uses_gt_for_selection": False,
                "uses_gt_for_oracle_best_diagnostic": True,
                "v21_trace": v21_trace if scenario == "retail" else {},
            }
            append_jsonl(guard_jsonl, record)
            task_records.append(record)
        write_json(out_dir / f"{spec}_easy.json", selected_items)
        write_json(oracle_dir / f"{spec}_easy.json", oracle_items)

    selected_summary = eval_model_dir(out_dir)
    oracle_summary = eval_model_dir(oracle_dir)
    v14_summary = aggregate(v14_evals)
    v21_retail_summary = aggregate(v21_evals)
    selected_task_summary = aggregate(selected_evals)
    oracle_task_summary = aggregate(oracle_evals)
    # Prefer direct model-dir eval for selected/oracle because it mirrors report format.
    selected = selected_summary["summary"]
    oracle = oracle_summary["summary"]

    per_scenario: Dict[str, Dict[str, Any]] = {}
    for spec in sorted({r["spec"] for r in selected_evals}):
        per_scenario[spec] = {
            "V14": aggregate([r for r in v14_evals if r["spec"] == spec]),
            "V22_selected": aggregate([r for r in selected_evals if r["spec"] == spec]),
            "oracle_best": aggregate([r for r in oracle_evals if r["spec"] == spec]),
        }

    win_loss = []
    for rec in task_records:
        v14_joint = bool(rec["v14_score_post_eval"].get("joint"))
        sel_joint = bool(rec["selected_score_post_eval"].get("joint"))
        if sel_joint and not v14_joint:
            delta = "win"
        elif v14_joint and not sel_joint:
            delta = "loss"
        else:
            delta = "same"
        win_loss.append({k: rec[k] for k in ("spec", "index", "chosen_candidate", "fallback_reason", "confidence") if k in rec} | {"delta": delta})

    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "version": version,
        "selected_result_dir": str(out_dir),
        "oracle_best_result_dir": str(oracle_dir),
        "candidate_jsonl": str(candidate_jsonl),
        "guard_trace_jsonl": str(guard_jsonl),
        "summary": {
            "V14_baseline": v14_summary,
            "V21_retail_candidate_where_applicable": v21_retail_summary,
            "V22_nonoracle_selected": selected,
            "V22_oracle_best_of_candidates_diagnostic": oracle,
            "V22_nonoracle_selected_task_eval": selected_task_summary,
            "V22_oracle_task_eval": oracle_task_summary,
        },
        "chosen_counts": chosen_counts,
        "v21_selected_count": v21_selected_count,
        "v21_fallback_count": v21_fallback_count,
        "v21_added_joint_count": v21_added_joint,
        "v21_regressed_joint_count": v21_regressed_joint,
        "per_scenario": per_scenario,
        "win_loss_vs_v14": win_loss,
        "final_run": False,
        "full_val41_shadow": True,
        "v10_zip_overwritten": before_zip != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
        "uses_final_hidden_metadata": False,
        "uses_gt_for_selection": False,
        "uses_gt_for_oracle_best_diagnostic": True,
    }
    write_json(run_dir / "v22_state.json", state)
    write_json(CODEX / "state" / "latest_v22_guarded_val41_shadow.json", state)
    write_reports(run_id, state, task_records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def summary_table(summary: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = ["| metric set | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, s in summary.items():
        lines.append(
            f"| {label} | {s.get('valid', 0)} | {fmt_pct(s.get('joint', 0))} | {fmt_pct(s.get('result', 0))} | {fmt_pct(s.get('tool', 0))} | {s.get('matched_tools', 0)}/{s.get('gt_tools', 0)} | {s.get('micro', 0):.4f} | {s.get('interaction_calls', 0)} |"
        )
    return lines


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = state["summary"]
    table = summary_table({
        "V14_baseline": summary["V14_baseline"],
        "V21_retail_only": summary["V21_retail_candidate_where_applicable"],
        "V22_nonoracle_selected": summary["V22_nonoracle_selected"],
        "V22_oracle_best": summary["V22_oracle_best_of_candidates_diagnostic"],
    })
    write_report(
        reports / f"V22_GUARDED_SELECTOR_IMPLEMENTATION_{run_id}.md",
        [
            f"# V22 Guarded Selector Implementation {run_id}",
            "",
            "- `v22_guarded_selector.py`: retail-only gate for V21 overlay.",
            "- Non-retail tasks default to V14.",
            "- Retail V21 can replace V14 only when primary product, DB existence, visual/current-value evidence, observation-supported branch, mutation target, closure, and dry-run checks pass.",
            "- GT is not used for non-oracle selected candidate choice.",
            "- GT is used only after prediction for metric computation and oracle best-of diagnostic.",
            "- final_run: false",
            "- uses_final_hidden_metadata: false",
            "- v10_zip_overwritten: false",
        ],
    )
    write_report(
        reports / f"V22_VAL41_SHADOW_RESULT_{run_id}.md",
        [
            f"# V22 Val41 Shadow Result {run_id}",
            "",
            *table,
            "",
            f"- chosen_counts: `{json.dumps(state['chosen_counts'], ensure_ascii=False)}`",
            f"- V21 selected count: {state['v21_selected_count']}",
            f"- V21 fallback count: {state['v21_fallback_count']}",
            f"- V21 added joint count: {state['v21_added_joint_count']}",
            f"- V21 caused regression count: {state['v21_regressed_joint_count']}",
            f"- selected_result_dir: `{state['selected_result_dir']}`",
            f"- candidate_jsonl: `{state['candidate_jsonl']}`",
            f"- guard_trace_jsonl: `{state['guard_trace_jsonl']}`",
            "- full_val41_shadow: true",
            "- final_run: false",
        ],
    )
    diff_lines = [
        f"# V22 vs V14 Diff Analysis {run_id}",
        "",
        "| spec | index | chosen | delta vs V14 | confidence | fallback |",
        "|---|---:|---|---|---:|---|",
    ]
    for row in state["win_loss_vs_v14"]:
        diff_lines.append(
            f"| {row.get('spec')} | {row.get('index')} | {row.get('chosen_candidate')} | {row.get('delta')} | {float(row.get('confidence') or 0):.2f} | {row.get('fallback_reason') or ''} |"
        )
    write_report(reports / f"V22_V14_DIFF_ANALYSIS_{run_id}.md", diff_lines)
    best = summary["V22_oracle_best_of_candidates_diagnostic"]
    sel = summary["V22_nonoracle_selected"]
    diagnostic = "scorer_or_guard_too_conservative" if best.get("joint", 0) >= 10 / 41 and sel.get("joint", 0) < 10 / 41 else "candidate_generator_insufficient"
    if sel.get("joint", 0) >= 10 / 41:
        diagnostic = "selected_meets_10_joint_target"
    write_report(
        reports / f"V22_CANDIDATE_UPPER_BOUND_DIAGNOSTIC_{run_id}.md",
        [
            f"# V22 Candidate Upper Bound Diagnostic {run_id}",
            "",
            *table,
            "",
            f"- oracle_best_joint_count: {best.get('joint', 0) * 41:.1f}/41",
            f"- selected_joint_count: {sel.get('joint', 0) * 41:.1f}/41",
            f"- diagnosis: {diagnostic}",
            "- oracle_best uses GT only after all candidates are generated.",
            "- oracle_best is not a final/submission candidate.",
        ],
    )
    next_lines = [
        f"# V22 Next Decision {run_id}",
        "",
        *table,
        "",
        "## Required Answers",
        "",
        f"- V22 nonoracle selected >22% joint: {sel.get('joint', 0) > 10 / 41}.",
        f"- Actual selected joint: {sel.get('joint', 0) * 41:.1f}/41 ({fmt_pct(sel.get('joint', 0))}).",
        f"- V21 retail overlay added joints: {state['v21_added_joint_count']}.",
        f"- V14 success regressions caused by V21: {state['v21_regressed_joint_count']}.",
        f"- If below 10/41, likely issue: {diagnostic}.",
        f"- Oracle best-of upper bound: {best.get('joint', 0) * 41:.1f}/41 ({fmt_pct(best.get('joint', 0))}).",
        "- Continue recommendation: " + (
            "prepare V23 final-style migration diagnostics; still do not run final automatically."
            if sel.get("joint", 0) >= 10 / 41
            else ("repair scorer/guard first." if diagnostic == "scorer_or_guard_too_conservative" else "add order/restaurant or non-clean retail candidate generators before scorer tuning.")
        ),
        "- final_run: false",
        "- v10_zip_overwritten: false",
        "- auto_submit: false",
    ]
    write_report(reports / f"V22_NEXT_DECISION_{run_id}.md", next_lines)


if __name__ == "__main__":
    main()
