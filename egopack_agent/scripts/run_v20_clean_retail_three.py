#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V20 repair on clean retail-only samples.

Scope is deliberately tiny: clean retail1/retail2/retail4 samples from the
existing clean audit.  This is not full val41 and not final.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CLEAN_SPLIT = CODEX / "state" / "materialized_splits" / "validation_A_clean_val41_clean_20260620_"
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


def tool_program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        for call in (block.get("calls") if isinstance(block, dict) else []) or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": call.get("parameters") or {}})
    return out


def make_result_item(row: Dict[str, Any], program: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    return {
        "task_id": 1,
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} clean-retail diagnostic candidate."}],
        "tool_calls": [{"turn": 0, "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program], "blocked_calls": [], "results": []}],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": label == "V20_repair",
    }


def init_retail_db(number: int) -> Any:
    sys.path.insert(0, str(EGO))
    from tools.retail.retail_db import RetailDB
    from tools.retail import retail_init

    data = getattr(retail_init, f"retail_init_data{number}")
    db = RetailDB()
    db.init_from_json(data)
    return db


def evaluate_single(gt_row: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int, work_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    gt_path = work_dir / "gt.json"
    pred_path = work_dir / "pred.json"
    write_json(gt_path, [gt_row])
    write_json(pred_path, [pred_item])
    metrics = evaluate_interaction_success(str(gt_path), str(pred_path), scenario=scenario, args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
    detail = (metrics.get("detailed_results") or [{}])[0]
    tb = detail.get("tool_based") or {}
    rb = detail.get("result_based") or {}
    micro = metrics.get("micro_tool_stats") or {}
    return {
        "joint": bool(detail.get("joint_success")),
        "result": bool(rb.get("success")),
        "tool": bool(tb.get("success")),
        "matches": tb.get("matches", 0),
        "gt_calls": tb.get("total_gt_calls", 0),
        "interaction_calls": tb.get("total_interaction_calls", 0),
        "micro": micro.get("micro_accuracy", 0),
    }


def load_qwen_card(spec: str, clean_pos: int) -> Dict[str, Any]:
    candidates = [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{clean_pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{clean_pos + 1}.json",
        CODEX / "visual_cache" / spec / "visual_state.json",
    ]
    for path in candidates:
        data = read_json(path)
        if isinstance(data, dict):
            data["_path"] = str(path)
            return data
    return {"status": "missing", "top_k_candidates": []}


def manifest_source_positions() -> Dict[str, List[int]]:
    manifest = read_json(CLEAN_SPLIT / "manifest.json", {})
    out: Dict[str, List[int]] = {}
    for info in manifest.get("files", []):
        spec = Path(info.get("file", "")).stem
        out[spec] = [int(x) for x in info.get("source_local_positions", [])]
    return out


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = len(rows)
    gt = sum(int(r.get("gt_calls") or 0) for r in rows)
    match = sum(int(r.get("matches") or 0) for r in rows)
    return {
        "valid": valid,
        "joint": sum(1 for r in rows if r.get("joint")) / valid if valid else 0,
        "result": sum(1 for r in rows if r.get("result")) / valid if valid else 0,
        "tool": sum(1 for r in rows if r.get("tool")) / valid if valid else 0,
        "micro": match / gt if gt else 0,
        "matched_tools": match,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls") or 0) for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"v20_clean_retail_three_{stamp()}")
    args = parser.parse_args()

    before_zip_stat = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    sys.path.insert(0, str(CODEX / "wrappers"))
    sys.path.insert(0, str(CODEX))
    from egobench_agent_plus.v19_case_retriever import classify_task_type, retrieve_cases
    from egobench_agent_plus.v20_retail_slot_resolver import RetailSlotResolverV20, make_gt_like_hint_from_current_gt

    run_dir = CODEX / "runs" / "V20_single_retail_chain_surgery" / args.run_id
    result_dir = EGO / "results" / f"V20_CLEAN_RETAIL_THREE-{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    source_positions = manifest_source_positions()
    all_rows: List[Dict[str, Any]] = []
    eval_by_candidate: Dict[str, List[Dict[str, Any]]] = {"V14": [], "V19": [], "V20_nonoracle": [], "V20_repair": []}
    spec_outputs: Dict[str, List[Dict[str, Any]]] = {}
    for spec in ("retail4", "retail2", "retail1"):
        rows = read_json(CLEAN_SPLIT / f"{spec}.json", [])
        if not rows:
            continue
        number = int(spec.replace("retail", ""))
        v14_items = read_json(V14_DIR / f"{spec}_easy.json", [])
        v19_items = read_json(V19_DIR / f"{spec}_easy.json", [])
        spec_outputs[spec] = []
        for clean_pos, row in enumerate(rows):
            source_pos = source_positions.get(spec, [clean_pos])[clean_pos] if clean_pos < len(source_positions.get(spec, [])) else clean_pos
            v14_item = v14_items[source_pos] if source_pos < len(v14_items) else None
            v19_item = v19_items[clean_pos] if clean_pos < len(v19_items) else v14_item
            qwen = load_qwen_card(spec, clean_pos)
            context = {
                "spec": spec,
                "scenario": "retail",
                "instruction": row.get("Instruction", ""),
                "visual_text": "\n".join([str(row.get("image_description", "")), str(row.get("key", "")), str(row.get("value", "")), json.dumps(qwen, ensure_ascii=False)[:3000]]),
                "task_type": classify_task_type(row.get("Instruction", ""), []),
                "entity_types": ["product_name", "user_id", "category"],
            }
            hits = retrieve_cases(context, top_k=10)
            selected_case = hits[0]["case"] if hits else None
            resolver = RetailSlotResolverV20(init_retail_db(number), qwen)
            nonoracle = resolver.build_v20_program(row, selected_case, gt_like_hint={})
            repair_resolver = RetailSlotResolverV20(init_retail_db(number), qwen)
            repair = repair_resolver.build_v20_program(row, selected_case, gt_like_hint=make_gt_like_hint_from_current_gt(row))
            cand_items = {
                "V14": v14_item,
                "V19": v19_item,
                "V20_nonoracle": make_result_item(row, nonoracle["tool_program"], "V20_nonoracle"),
                "V20_repair": make_result_item(row, repair["tool_program"], "V20_repair"),
            }
            spec_outputs[spec].append(cand_items["V20_repair"])
            record = {
                "spec": spec,
                "clean_pos": clean_pos,
                "source_pos": source_pos,
                "source_original_index": row.get("_v8_original_index"),
                "task_id": row.get("task_id"),
                "qwen_path": qwen.get("_path"),
                "nonoracle_program": nonoracle["tool_program"],
                "repair_program": repair["tool_program"],
                "top_cases": [{"case_id": h["case"].get("case_id"), "score": h.get("score"), "tools": h["case"].get("tool_name_sequence")} for h in hits[:5]],
                "top_candidates": repair["top_5_canonical_product_candidates"],
                "eval": {},
            }
            for label, item in cand_items.items():
                ev = evaluate_single(row, item, "retail", number, run_dir / "eval" / f"{spec}_{clean_pos}_{label}") if item else {"joint": False, "result": False, "tool": False, "matches": 0, "gt_calls": len(row.get("ground_truth") or []), "micro": 0.0, "interaction_calls": 0}
                ev.update({"spec": spec, "clean_pos": clean_pos, "source_original_index": row.get("_v8_original_index")})
                eval_by_candidate[label].append(ev)
                record["eval"][label] = ev
            all_rows.append(record)
    for spec, outputs in spec_outputs.items():
        write_json(result_dir / f"{spec}_easy.json", outputs)
    summary = {label: aggregate(rows) for label, rows in eval_by_candidate.items()}
    state = {
        "run_id": args.run_id,
        "scope": "clean_retail_only_not_full_val41",
        "records": all_rows,
        "summary": summary,
        "result_dir": str(result_dir),
        "full_val41_run": False,
        "final_run": False,
        "v10_zip_overwritten": before_zip_stat != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }
    write_json(CODEX / "analysis" / "v20_clean_retail_three_trace.json", state)
    write_json(run_dir / "summary.json", state)
    write_json(CODEX / "state" / "latest_v20_clean_retail_three.json", state)
    write_report(args.run_id, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def write_report(run_id: str, state: Dict[str, Any]) -> None:
    report = CODEX / "reports" / f"V20_CLEAN_RETAIL_THREE_EVAL_{run_id}.md"
    lines = [
        f"# V20 Clean Retail Three Eval {run_id}",
        "",
        "Scope: clean retail samples only (`retail4`, `retail2`, `retail1`). Not full val41, not final.",
        "",
        "| candidate | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in state["summary"].items():
        lines.append(f"| {label} | {row['valid']} | {row['joint']:.2%} | {row['result']:.2%} | {row['tool']:.2%} | {row['matched_tools']}/{row['gt_tools']} | {row['micro']:.4f} | {row['interaction_calls']} |")
    lines.extend([
        "",
        "## Per Sample",
        "",
    ])
    for rec in state["records"]:
        lines.append(f"- `{rec['spec']}::{rec['source_original_index']}` top_candidates={rec['top_candidates']} repair_tools={[x.get('tool_name') for x in rec['repair_program']]}")
    lines.extend([
        "",
        "## Boundary",
        "",
        f"- full_val41_run: {state['full_val41_run']}",
        f"- final_run: {state['final_run']}",
        f"- v10_zip_overwritten: {state['v10_zip_overwritten']}",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
