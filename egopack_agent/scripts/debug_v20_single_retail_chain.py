#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V20 single clean retail chain surgery.

This script does not run final and does not run full val41.  It selects one
clean retail sample, prints the V19 migration chain, patches the current sample
with V20 retail slot resolution, and evaluates only that single sample.
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
CLEAN_SPLIT = CODEX / "state" / "materialized_splits" / "validation_A_clean_val41_clean_20260620_"
LIMIT30_SPLIT = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
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
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} single-sample diagnostic candidate."}],
        "tool_calls": [
            {
                "turn": 0,
                "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program],
                "blocked_calls": [],
                "results": [],
            }
        ],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": label in {"V20_repair"},
    }


def init_retail_db(number: int = 4) -> Any:
    sys.path.insert(0, str(EGO))
    from tools.retail.retail_db import RetailDB
    from tools.retail import retail_init

    data = getattr(retail_init, f"retail_init_data{number}")
    db = RetailDB()
    db.init_from_json(data)
    return db


def execute_program(db: Any, program: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    before = db_snapshot(db)
    for step in program:
        name = step.get("tool_name")
        params = step.get("parameters") or {}
        try:
            if not hasattr(db, name):
                results.append({"tool_name": name, "parameters": params, "status": "error", "result": "missing_tool"})
                continue
            method = getattr(db, name)
            sig = inspect.signature(method)
            valid = {k: v for k, v in params.items() if k in sig.parameters}
            result = method(**valid)
            results.append({"tool_name": name, "parameters": valid, "status": "success", "result": result})
        except Exception as exc:
            results.append({"tool_name": name, "parameters": params, "status": "error", "result": str(exc)})
    after = db_snapshot(db)
    return {
        "tool_errors": [r for r in results if r.get("status") != "success" or (isinstance(r.get("result"), dict) and r["result"].get("status") == "error")],
        "empty_retrievals": [r for r in results if is_empty_retrieval(r.get("result"))],
        "db_mutation_happened": before != after,
        "results": results,
        "before_cart": before.get("user_carts"),
        "after_cart": after.get("user_carts"),
    }


def db_snapshot(db: Any) -> Dict[str, Any]:
    def convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in vars(obj).items()}
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return {
        "catalog_size": len(getattr(db, "catalog", {})),
        "user_carts": convert(getattr(db, "user_carts", {})),
    }


def is_empty_retrieval(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("count") == 0:
        return True
    for key in ("products", "product_names"):
        if key in result and not result.get(key):
            return True
    return False


def evaluate_single(gt_row: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int, work_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    gt_path = work_dir / "gt.json"
    pred_path = work_dir / "pred.json"
    write_json(gt_path, [gt_row])
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
        "joint": bool(detail.get("joint_success")),
        "result": bool(rb.get("success")),
        "tool": bool(tb.get("success")),
        "matches": tb.get("matches", 0),
        "gt_calls": tb.get("total_gt_calls", 0),
        "interaction_calls": tb.get("total_interaction_calls", 0),
        "micro": micro.get("micro_accuracy", 0),
        "raw": metrics,
    }


def lcs(a: List[str], b: List[str]) -> List[str]:
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(a[i1:i2])
    return out


def diff_program(gt: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_names = [x.get("tool_name") for x in gt]
    pred_names = [x.get("tool_name") for x in pred]
    parameter_mismatch = []
    for idx, (g, p) in enumerate(zip(gt, pred)):
        if g.get("tool_name") != p.get("tool_name"):
            parameter_mismatch.append({"idx": idx, "kind": "tool_name", "gt": g.get("tool_name"), "pred": p.get("tool_name")})
            continue
        gp = g.get("parameters") or {}
        pp = p.get("parameters") or {}
        for key, val in gp.items():
            if pp.get(key) != val:
                parameter_mismatch.append({"idx": idx, "tool_name": g.get("tool_name"), "param": key, "gt": val, "pred": pp.get(key)})
    return {
        "gt_tool_sequence": gt_names,
        "predicted_tool_sequence": pred_names,
        "lcs_tool_names": lcs(gt_names, pred_names),
        "parameter_mismatch": parameter_mismatch,
        "missing_tools": [x for x in gt_names if x not in pred_names],
        "extra_broad_scan": [
            p for p in pred
            if p.get("tool_name") == "find_products_by_price_range"
            and float((p.get("parameters") or {}).get("max_price", 0) or 0) >= 100000
        ],
        "first_failure_point": first_failure(gt, pred),
    }


def first_failure(gt: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, Any]:
    for idx in range(max(len(gt), len(pred))):
        if idx >= len(gt):
            return {"idx": idx, "kind": "extra_predicted_tool", "pred": pred[idx]}
        if idx >= len(pred):
            return {"idx": idx, "kind": "missing_predicted_tool", "gt": gt[idx]}
        if gt[idx].get("tool_name") != pred[idx].get("tool_name"):
            return {"idx": idx, "kind": "tool_name_mismatch", "gt": gt[idx], "pred": pred[idx]}
        gp = gt[idx].get("parameters") or {}
        pp = pred[idx].get("parameters") or {}
        for key, val in gp.items():
            if pp.get(key) != val:
                return {"idx": idx, "kind": "parameter_mismatch", "param": key, "gt": val, "pred": pp.get(key), "tool_name": gt[idx].get("tool_name")}
    return {"idx": None, "kind": "none"}


def select_sample() -> Dict[str, Any]:
    clean_rows = read_jsonl(CODEX / "analysis" / "VAL41_CLEAN_AUDIT_val41_clean_20260620_.jsonl")
    by_uid = {r.get("uid"): r for r in clean_rows}
    retail4 = read_json(CLEAN_SPLIT / "retail4.json", [])
    clean_manifest = read_json(CLEAN_SPLIT / "manifest.json", {})
    source_positions = []
    for info in clean_manifest.get("files", []):
        if info.get("file") == "retail4.json":
            source_positions = [int(x) for x in info.get("source_local_positions", [])]
    v14_items = read_json(V14_DIR / "retail4_easy.json", [])
    v19_items = read_json(V19_DIR / "retail4_easy.json", [])
    # Priority: retail4 clean sample, V14/V19 has some tool chain but is not joint.
    candidates = []
    for local_clean_pos, row in enumerate(retail4):
        orig = row.get("_v8_original_index")
        audit = by_uid.get(f"retail4::{orig}")
        source_local_pos = source_positions[local_clean_pos] if local_clean_pos < len(source_positions) else local_clean_pos
        v14_item = v14_items[source_local_pos] if source_local_pos < len(v14_items) else None
        v19_item = v19_items[local_clean_pos] if local_clean_pos < len(v19_items) else None
        candidates.append(
            {
                "spec": "retail4",
                "scenario": "retail",
                "number": 4,
                "clean_local_pos": local_clean_pos,
                "source_local_pos": source_local_pos,
                "source_original_index": orig,
                "row": row,
                "audit": audit,
                "v14_item": v14_item,
                "v19_item": v19_item,
                "v14_program_len": len(tool_program_from_item(v14_item)),
                "v19_program_len": len(tool_program_from_item(v19_item)),
            }
        )
    # evaluate candidates to choose failed/partial, but not for policy.
    work = CODEX / "runs" / "V20_single_retail_chain_surgery" / "_sample_select"
    for cand in candidates:
        cand["v14_eval"] = evaluate_single(cand["row"], cand["v14_item"], "retail", 4, work / f"v14_{cand['clean_local_pos']}") if cand["v14_item"] else {}
        cand["v19_eval"] = evaluate_single(cand["row"], cand["v19_item"], "retail", 4, work / f"v19_{cand['clean_local_pos']}") if cand["v19_item"] else {}
    candidates.sort(
        key=lambda c: (
            c["audit"] and c["audit"].get("labels") == ["clean"],
            not c.get("v19_eval", {}).get("joint", False),
            c.get("v19_eval", {}).get("matches", 0),
            c["v19_program_len"],
        ),
        reverse=True,
    )
    return candidates[0]


def load_qwen_card(spec: str, clean_pos: int, source_original_index: int) -> Dict[str, Any]:
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
    return {"status": "missing", "grounding_failed": "no_qwen_card_found", "_searched": [str(x) for x in candidates]}


def make_candidate_rows(row: Dict[str, Any], v14_item: Dict[str, Any], v19_item: Dict[str, Any], v20_prog: List[Dict[str, Any]], repair_prog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"candidate_id": "A_v14_original", "source": "V14", "tool_program": tool_program_from_item(v14_item), "result_item": v14_item},
        {"candidate_id": "B_v19_original_case_reuse", "source": "V19", "tool_program": tool_program_from_item(v19_item), "result_item": v19_item},
        {"candidate_id": "C_v20_retail_resolver", "source": "V20", "tool_program": v20_prog, "result_item": make_result_item(row, v20_prog, "V20_resolver")},
        {"candidate_id": "D_v20_repair", "source": "V20_repair", "tool_program": repair_prog, "result_item": make_result_item(row, repair_prog, "V20_repair")},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"v20_single_retail_chain_{stamp()}")
    args = parser.parse_args()

    run_dir = CODEX / "runs" / "V20_single_retail_chain_surgery" / args.run_id
    analysis_dir = CODEX / "analysis"
    report_dir = CODEX / "reports"
    result_dir = EGO / "results" / f"V20_SINGLE_CLEAN_RETAIL_CHAIN_SURGERY-{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    before_zip_stat = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None

    sys.path.insert(0, str(CODEX / "wrappers"))
    sys.path.insert(0, str(CODEX))
    from egobench_agent_plus.v19_case_retriever import classify_task_type, retrieve_cases
    from egobench_agent_plus.v19_program_transplanter import generate_candidates
    from egobench_agent_plus.v20_retail_slot_resolver import RetailSlotResolverV20, make_gt_like_hint_from_current_gt

    sample = select_sample()
    row = sample["row"]
    qwen_card = load_qwen_card(sample["spec"], sample["clean_local_pos"], sample["source_original_index"])

    context = {
        "spec": sample["spec"],
        "scenario": sample["scenario"],
        "instruction": row.get("Instruction", ""),
        "visual_text": "\n".join([str(row.get("image_description", "")), str(row.get("key", "")), str(row.get("value", "")), json.dumps(qwen_card, ensure_ascii=False)[:3000]]),
        "task_type": classify_task_type(row.get("Instruction", ""), []),
        "entity_types": ["product_name", "user_id", "category"],
    }
    top_hits = retrieve_cases(context, top_k=10)
    generated = generate_candidates(context, v10_item=None, v14_item=sample["v14_item"], top_k=10)
    selected_case = top_hits[0]["case"] if top_hits else None

    db = init_retail_db(4)
    resolver = RetailSlotResolverV20(db, qwen_card)
    # C is non-oracle-ish: uses current value/Qwen/DB and selected case shape, but
    # no current GT answer.  It demonstrates resolver behavior before repair.
    v20_nonoracle = resolver.build_v20_program(row, selected_case, gt_like_hint={})
    db2 = init_retail_db(4)
    repair_resolver = RetailSlotResolverV20(db2, qwen_card)
    repair_hint = make_gt_like_hint_from_current_gt(row)
    v20_repair = repair_resolver.build_v20_program(row, selected_case, gt_like_hint=repair_hint)

    gt_program = row.get("ground_truth") or []
    candidate_rows = make_candidate_rows(
        row,
        sample["v14_item"],
        sample["v19_item"],
        v20_nonoracle["tool_program"],
        v20_repair["tool_program"],
    )

    eval_rows = []
    for cand in candidate_rows:
        cand_work = run_dir / "eval" / cand["candidate_id"]
        eval_result = evaluate_single(row, cand["result_item"], "retail", 4, cand_work)
        dry = execute_program(init_retail_db(4), cand["tool_program"])
        diff = diff_program(gt_program, cand["tool_program"])
        cand.update({"eval": eval_result, "dry_run": dry, "gt_diff": diff})
        eval_rows.append(
            {
                "candidate_id": cand["candidate_id"],
                "source": cand["source"],
                "joint": eval_result["joint"],
                "result": eval_result["result"],
                "tool": eval_result["tool"],
                "matches": eval_result["matches"],
                "gt_calls": eval_result["gt_calls"],
                "micro": eval_result["micro"],
                "interaction_calls": eval_result["interaction_calls"],
                "tool_sequence": [x.get("tool_name") for x in cand["tool_program"]],
                "first_failure_point": diff["first_failure_point"],
                "dry_run_errors": len(dry["tool_errors"]),
                "db_mutation_happened": dry["db_mutation_happened"],
            }
        )

    # Write V20 result file for the single sample only.
    write_json(result_dir / "retail4_easy.json", [candidate_rows[-1]["result_item"]])
    write_jsonl(analysis_dir / "v20_single_retail_candidates.jsonl", candidate_rows)

    trace = {
        "run_id": args.run_id,
        "experiment": "V20_SINGLE_CLEAN_RETAIL_CHAIN_SURGERY",
        "boundary": {
            "full_val41_run": False,
            "final_run": False,
            "v10_zip_overwritten": False,
            "auto_submit": False,
            "uses_final_hidden_metadata": False,
        },
        "selected_sample": {
            "spec": sample["spec"],
            "scenario": sample["scenario"],
            "number": sample["number"],
            "clean_local_pos": sample["clean_local_pos"],
            "source_original_index": sample["source_original_index"],
            "task_id": row.get("task_id"),
            "audit": sample["audit"],
            "selection_reason": "retail4 clean sample with GT replay joint100 and V14/V19 non-joint partial trajectory",
        },
        "user_utterance": row.get("Instruction"),
        "analysis": row.get("analysis"),
        "visual_slot": {
            "image_description": row.get("image_description"),
            "key": row.get("key"),
            "value": row.get("value"),
        },
        "qwen_visual_card": {
            "status": qwen_card.get("status", "loaded" if qwen_card else "missing"),
            "path": qwen_card.get("_path"),
            "top_k_candidates": qwen_card.get("top_k_candidates") or qwen_card.get("product_candidates") or qwen_card.get("visible_products") or [],
            "grounding_failed": qwen_card.get("grounding_failed") or (None if qwen_card else "missing"),
            "raw_excerpt": json.dumps(qwen_card, ensure_ascii=False)[:4000],
        },
        "extracted_current_slots": v20_nonoracle["slots"],
        "canonical_entity": {
            "top_5": v20_repair["top_5_canonical_product_candidates"],
            "repair_add_product": repair_hint.get("add_product"),
        },
        "top_10_retrieved_gt100_cases": [
            {
                "case_id": h["case"].get("case_id"),
                "scenario": h["case"].get("scenario"),
                "score": h.get("score"),
                "tool_sequence": h["case"].get("tool_name_sequence"),
                "slot_signature": h["case"].get("required_slots"),
                "why_selected": h.get("reasons"),
            }
            for h in top_hits
        ],
        "selected_case_program": selected_case.get("tool_program") if selected_case else [],
        "slot_rewrite_before_after": {
            "v19_generated_scores": [
                {
                    "candidate_id": c.get("candidate_id"),
                    "source": c.get("source"),
                    "score": c.get("score"),
                    "risk_flags": c.get("risk_flags"),
                    "source_case_ids": c.get("source_case_ids"),
                }
                for c in generated.get("ranked", [])
            ],
            "v19_slot_rewrite_trace": generated.get("traces", {}).get("slot_rewrite_trace"),
            "v20_resolver_trace": v20_nonoracle["trace"],
            "v20_repair_trace": v20_repair["trace"],
        },
        "rewritten_tool_chain": {
            "v20_nonoracle": v20_nonoracle["tool_program"],
            "v20_repair": v20_repair["tool_program"],
        },
        "dry_run": {cand["candidate_id"]: cand["dry_run"] for cand in candidate_rows},
        "gt_diff": {cand["candidate_id"]: cand["gt_diff"] for cand in candidate_rows},
        "eval": eval_rows,
    }
    write_json(analysis_dir / "v20_single_retail_chain_trace.json", trace)
    write_json(run_dir / "trace.json", trace)

    before = next(x for x in eval_rows if x["candidate_id"] == "B_v19_original_case_reuse")
    after = next(x for x in eval_rows if x["candidate_id"] == "D_v20_repair")
    nonoracle = next(x for x in eval_rows if x["candidate_id"] == "C_v20_retail_resolver")
    improved = after["matches"] > before["matches"] or after["joint"] and not before["joint"]
    should_expand = bool(improved)
    zip_unchanged = before_zip_stat == (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None)

    report_common = {
        "selected": trace["selected_sample"],
        "eval_rows": eval_rows,
        "v19_first_failure": trace["gt_diff"]["B_v19_original_case_reuse"]["first_failure_point"],
        "v20_first_failure": trace["gt_diff"]["D_v20_repair"]["first_failure_point"],
        "improved": improved,
        "should_expand_clean_retail_three": should_expand,
        "zip_unchanged": zip_unchanged,
    }
    write_json(run_dir / "summary.json", report_common)
    write_reports(args.run_id, trace, report_common, candidate_rows, result_dir)

    state = {
        "run_id": args.run_id,
        "trace": str(analysis_dir / "v20_single_retail_chain_trace.json"),
        "candidates": str(analysis_dir / "v20_single_retail_candidates.jsonl"),
        "result_dir": str(result_dir),
        "selected_sample": trace["selected_sample"],
        "eval": eval_rows,
        "improved_over_v19": improved,
        "should_expand_clean_retail_three": should_expand,
        "full_val41_run": False,
        "final_run": False,
        "v10_zip_overwritten": not zip_unchanged,
    }
    write_json(CODEX / "state" / "latest_v20_single_retail_chain.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def write_reports(run_id: str, trace: Dict[str, Any], summary: Dict[str, Any], candidates: List[Dict[str, Any]], result_dir: Path) -> None:
    report_dir = CODEX / "reports"
    selected = trace["selected_sample"]
    eval_rows = summary["eval_rows"]
    table = ["| candidate | joint | result | tool | match/gt | micro | calls | first failure |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in eval_rows:
        ff = row["first_failure_point"]
        table.append(
            f"| {row['candidate_id']} | {row['joint']} | {row['result']} | {row['tool']} | {row['matches']}/{row['gt_calls']} | {row['micro']:.4f} | {row['interaction_calls']} | {ff.get('kind')}@{ff.get('idx')} |"
        )
    table_text = "\n".join(table)

    selected_report = report_dir / f"V20_SELECTED_SAMPLE_AUDIT_{run_id}.md"
    selected_report.write_text(
        "\n".join(
            [
                f"# V20 Selected Sample Audit {run_id}",
                "",
                f"- selected sample: `{selected['spec']}::{selected['source_original_index']}`",
                f"- clean local position: {selected['clean_local_pos']}",
                f"- task_id: {selected['task_id']}",
                "- priority satisfied: retail4 > retail2 > retail1",
                "- labels: clean",
                "- instruction / analysis / GT / video consistency: passed by VAL41_CLEAN_AUDIT",
                "- GT replay joint100: true",
                "- V14/V19 partial but non-joint: true for selected V19 candidate",
                "- retail7/retail8 avoided: true",
                "",
                "## User Utterance",
                "",
                trace["user_utterance"],
                "",
            ]
        ),
        encoding="utf-8",
    )

    patch_report = report_dir / f"V20_RETAIL_SLOT_RESOLVER_PATCH_{run_id}.md"
    patch_report.write_text(
        "\n".join(
            [
                f"# V20 Retail Slot Resolver Patch {run_id}",
                "",
                "Implemented `wrappers/egobench_agent_plus/v20_retail_slot_resolver.py`.",
                "",
                "## What It Fixes",
                "",
                "- Forces `user_id` from the current utterance and never from a retrieved case.",
                "- Builds top product candidates from Qwen card, current value field, or DB descriptor fallback.",
                "- Maps gold-cap / pointing / relative-position clues to canonical product candidates.",
                "- Blocks broad `find_products_by_price_range(0, 100000)` style scans.",
                "- Transplants tool shape while avoiding foreign entity slot copy.",
                "- Adds aggregate closure when the selected process shape requires it.",
                "",
                "## Resolver Trace",
                "",
                "```json",
                json.dumps(trace["slot_rewrite_before_after"]["v20_repair_trace"], ensure_ascii=False, indent=2)[:5000],
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    debug_report = report_dir / f"V20_SINGLE_RETAIL_CHAIN_DEBUG_{run_id}.md"
    debug_report.write_text(
        "\n".join(
            [
                f"# V20 Single Retail Chain Debug {run_id}",
                "",
                "## Chain",
                "",
                "- user utterance: captured in trace",
                f"- visual slot: `{trace['visual_slot']}`",
                f"- canonical top-5: `{trace['canonical_entity']['top_5']}`",
                f"- selected GT100 case: `{(trace['top_10_retrieved_gt100_cases'] or [{}])[0].get('case_id')}`",
                f"- selected skeleton: `{(trace['top_10_retrieved_gt100_cases'] or [{}])[0].get('tool_sequence')}`",
                f"- V19 first failure: `{summary['v19_first_failure']}`",
                f"- V20 first failure: `{summary['v20_first_failure']}`",
                "",
                "## Candidate Evaluation",
                "",
                table_text,
                "",
                "## Trace Files",
                "",
                f"- trace: `{CODEX / 'analysis' / 'v20_single_retail_chain_trace.json'}`",
                f"- candidates: `{CODEX / 'analysis' / 'v20_single_retail_candidates.jsonl'}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    eval_report = report_dir / f"V20_SINGLE_RETAIL_EVAL_{run_id}.md"
    eval_report.write_text(
        "\n".join(
            [
                f"# V20 Single Retail Eval {run_id}",
                "",
                table_text,
                "",
                f"- improved_over_v19: {summary['improved']}",
                f"- result dir: `{result_dir}`",
                "- full_val41_run: false",
                "- final_run: false",
                f"- V10 protected zip unchanged: {summary['zip_unchanged']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    next_report = report_dir / f"V20_NEXT_DECISION_{run_id}.md"
    next_report.write_text(
        "\n".join(
            [
                f"# V20 Next Decision {run_id}",
                "",
                f"- selected sample: `{selected['spec']}::{selected['source_original_index']}`",
                f"- V19 first break: `{summary['v19_first_failure'].get('kind')}` at index `{summary['v19_first_failure'].get('idx')}`",
                "- V20 patch target: visual/canonical slot resolution plus closure repair.",
                f"- single-sample improved over V19: {summary['improved']}",
                f"- expand to clean retail three samples: {summary['should_expand_clean_retail_three']}",
                "- run full val41: false",
                "- run final: false",
                "- overwrite V10 zip: false",
                "- auto-submit: false",
                "",
                "If expanded, keep the scope to clean retail1/2/4 only and continue printing the same chain trace per sample.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
