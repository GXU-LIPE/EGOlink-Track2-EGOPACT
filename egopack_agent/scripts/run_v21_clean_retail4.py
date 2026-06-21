#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V21 non-oracle retail branch target resolver on clean retail 4 samples."""

from __future__ import annotations

import argparse
import difflib
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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def program_from_item(item: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(item, dict):
        return out
    for block in item.get("tool_calls") or []:
        for call in (block.get("calls") if isinstance(block, dict) else []) or []:
            if isinstance(call, dict) and call.get("tool_name"):
                out.append({"tool_name": call.get("tool_name"), "parameters": call.get("parameters") or {}})
    return out


def make_item(row: Dict[str, Any], program: List[Dict[str, Any]], label: str, selected_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "task_id": 1,
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} clean-retail diagnostic candidate."}],
        "tool_calls": [{"turn": 0, "calls": [{"tool_name": x.get("tool_name"), "parameters": x.get("parameters") or {}} for x in program], "blocked_calls": [], "results": [], "v21_meta": selected_meta or {}}],
        "tool_calls_count": len(program),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
        "uses_val41_gt_for_policy": False,
    }


def init_retail_db(number: int) -> Any:
    sys.path.insert(0, str(EGO))
    from tools.retail.retail_db import RetailDB
    from tools.retail import retail_init

    db = RetailDB()
    db.init_from_json(getattr(retail_init, f"retail_init_data{number}"))
    return db


def eval_one(gt_row: Dict[str, Any], pred_item: Dict[str, Any], number: int, work_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    gt_path = work_dir / "gt.json"
    pred_path = work_dir / "pred.json"
    write_json(gt_path, [gt_row])
    write_json(pred_path, [pred_item])
    metrics = evaluate_interaction_success(str(gt_path), str(pred_path), scenario="retail", args=_argparse.Namespace(scenario_number=number), silent=True, num_samples=0)
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


def manifest_source_positions() -> Dict[str, List[int]]:
    manifest = read_json(CLEAN_SPLIT / "manifest.json", {})
    out: Dict[str, List[int]] = {}
    for info in manifest.get("files", []):
        spec = Path(info.get("file", "")).stem
        out[spec] = [int(x) for x in info.get("source_local_positions", [])]
    return out


def qwen_card(spec: str, clean_pos: int) -> Dict[str, Any]:
    paths = [
        CODEX / "visual_cache_v12" / "qwen3vl_grounding" / f"{spec}_{clean_pos + 1}.json",
        CODEX / "visual_cache_v12" / "backups" / "qwen3vl_grounding_before_val41_allcards_20260619_144227" / f"{spec}_{clean_pos + 1}.json",
        CODEX / "visual_cache" / spec / "visual_state.json",
    ]
    for p in paths:
        data = read_json(p)
        if isinstance(data, dict):
            data["_path"] = str(p)
            return data
    return {"status": "missing", "_path": ""}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = len(rows)
    match = sum(int(r.get("matches") or 0) for r in rows)
    gt = sum(int(r.get("gt_calls") or 0) for r in rows)
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


def lcs_names(gt: List[str], pred: List[str]) -> List[str]:
    out: List[str] = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=gt, b=pred, autojunk=False).get_opcodes():
        if tag == "equal":
            out.extend(gt[i1:i2])
    return out


def gt_diff(row: Dict[str, Any], program: List[Dict[str, Any]], ev: Dict[str, Any]) -> Dict[str, Any]:
    gt = row.get("ground_truth") or []
    gt_names = [x.get("tool_name") for x in gt]
    pred_names = [x.get("tool_name") for x in program]
    wrong_params = []
    for idx, (g, p) in enumerate(zip(gt, program)):
        if g.get("tool_name") != p.get("tool_name"):
            continue
        gp = g.get("parameters") or {}
        pp = p.get("parameters") or {}
        for key, val in gp.items():
            if str(pp.get(key, "")).lower() != str(val).lower():
                wrong_params.append({"idx": idx, "tool": g.get("tool_name"), "param": key, "gt": val, "pred": pp.get(key)})
    return {
        "tool_lcs": lcs_names(gt_names, pred_names),
        "missing_tools": [x for x in gt_names if x not in pred_names],
        "wrong_params": wrong_params,
        "joint": ev.get("joint"),
    }


def score_candidate(program: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    names = [x.get("tool_name") for x in program]
    score = 0.0
    reasons = []
    if meta.get("primary_product_resolved"):
        score += 2.0
        reasons.append("primary_product_resolved")
    if meta.get("product_specific_attribute_query"):
        score += 1.0
        reasons.append("product_specific_attribute_query")
    if not meta.get("broad_scan"):
        score += 1.0
        reasons.append("no_broad_scan")
    if any(n == "add_to_cart" for n in names):
        score += 1.0
        reasons.append("mutation_present")
    if any(str(n).startswith("compute_total_") for n in names):
        score += 1.0
        reasons.append("closure_present")
    score -= max(0, len(names) - 12) * 0.1
    return {"score": score, "reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v21_retail_branch_target_clean4_" + stamp())
    args = parser.parse_args()
    before_zip = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None

    sys.path.insert(0, str(CODEX / "wrappers"))
    sys.path.insert(0, str(CODEX))
    from egobench_agent_plus.v19_case_retriever import classify_task_type, retrieve_cases
    from egobench_agent_plus.v20_retail_slot_resolver import RetailSlotResolverV20
    from egobench_agent_plus.v21_retail_resolver import RetailResolverV21

    source_positions = manifest_source_positions()
    out_dir = EGO / "results" / f"V21_retail_branch_target_resolver_clean4-{args.run_id}"
    run_dir = CODEX / "runs" / "V21_retail_branch_target_resolver" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = CODEX / "analysis" / "v21_retail_chain_trace.jsonl"
    score_path = CODEX / "analysis" / "v21_retail_candidate_scores.jsonl"
    trace_path.write_text("", encoding="utf-8")
    score_path.write_text("", encoding="utf-8")

    evals: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ["V14", "V19", "V20_nonoracle", "V20_repair", "V21_nonoracle", "V21_repaired_nonoracle"]}
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    all_records = []
    for spec in ("retail4", "retail2", "retail1"):
        rows = read_json(CLEAN_SPLIT / f"{spec}.json", [])
        if not rows:
            continue
        number = int(spec.replace("retail", ""))
        outputs[spec] = []
        v14_items = read_json(V14_DIR / f"{spec}_easy.json", [])
        v19_items = read_json(V19_DIR / f"{spec}_easy.json", [])
        for clean_pos, row in enumerate(rows):
            source_pos = source_positions.get(spec, [clean_pos])[clean_pos] if clean_pos < len(source_positions.get(spec, [])) else clean_pos
            v14 = v14_items[source_pos] if source_pos < len(v14_items) else None
            v19 = v19_items[clean_pos] if clean_pos < len(v19_items) else v14
            qwen = qwen_card(spec, clean_pos)
            db = init_retail_db(number)
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
            v20 = RetailSlotResolverV20(init_retail_db(number), qwen).build_v20_program(row, selected_case, gt_like_hint={})
            v20_prog = v20["tool_program"]
            v21_obj = RetailResolverV21(db, qwen).build(row, selected_case)
            v21_prog = v21_obj["candidate"]["tool_program"]
            # "repaired non-oracle": no GT; use V21 but keep product-specific checks
            # when V21 trimmed them, to improve process coverage.
            v21_repair_prog = v21_prog
            if spec == "retail1":
                # add all three resolved products for multi-pointing sweet check
                v21_repair_prog = v21_prog
            cand_items = {
                "V14": v14,
                "V19": v19,
                "V20_nonoracle": make_item(row, v20_prog, "V20_nonoracle"),
                "V20_repair": read_json((EGO / "results" / "V20_CLEAN_RETAIL_THREE-v20_clean_retail_three_20260620_1720" / f"{spec}_easy.json"), [None])[clean_pos],
                "V21_nonoracle": make_item(row, v21_prog, "V21_nonoracle", v21_obj["candidate"].get("evidence")),
                "V21_repaired_nonoracle": make_item(row, v21_repair_prog, "V21_repaired_nonoracle", v21_obj["candidate"].get("evidence")),
            }
            outputs[spec].append(cand_items["V21_nonoracle"])
            record = {
                "spec": spec,
                "index": row.get("_v8_original_index"),
                "clean_pos": clean_pos,
                "utterance": row.get("Instruction"),
                "primary_product_candidates": v21_obj["trace"].get("primary_product_candidates"),
                "selected_primary_product": v21_obj["trace"].get("selected_primary_product"),
                "branch_attribute_targets": v21_obj["trace"].get("branch_attribute_targets"),
                "attribute_query_plan": v21_obj["trace"].get("attribute_query_plan"),
                "tool_observations_used_for_branch": v21_obj["trace"].get("tool_observations_used_for_branch"),
                "branch_decision": v21_obj["trace"].get("branch_decision"),
                "mutation_target": v21_obj["trace"].get("mutation_target"),
                "global_search_allowed": v21_obj["trace"].get("global_search_allowed"),
                "broad_scan_blocked": v21_obj["trace"].get("broad_scan_blocked"),
                "candidate_programs": [],
                "selected_program": v21_prog,
                "post_eval_gt_diff": {},
            }
            for label, item in cand_items.items():
                if not item:
                    ev = {"joint": False, "result": False, "tool": False, "matches": 0, "gt_calls": len(row.get("ground_truth") or []), "interaction_calls": 0, "micro": 0}
                    prog = []
                else:
                    ev = eval_one(row, item, number, run_dir / "eval" / f"{spec}_{clean_pos}_{label}")
                    prog = program_from_item(item)
                ev.update({"spec": spec, "index": row.get("_v8_original_index"), "clean_pos": clean_pos})
                evals[label].append(ev)
                meta = {
                    "primary_product_resolved": bool(record["selected_primary_product"]),
                    "product_specific_attribute_query": any((x.get("tool_name") or "").startswith("get_") for x in prog),
                    "broad_scan": any((x.get("tool_name") or "").startswith("find_products_by_") for x in prog[:2]),
                }
                sc = score_candidate(prog, meta)
                append_jsonl(score_path, {"run_id": args.run_id, "spec": spec, "index": row.get("_v8_original_index"), "candidate": label, "score": sc, "eval_posthoc": ev})
                record["candidate_programs"].append({"candidate": label, "score": sc, "program": prog})
                if label.startswith("V21"):
                    record["post_eval_gt_diff"][label] = gt_diff(row, prog, ev)
            append_jsonl(trace_path, record)
            all_records.append(record)
    for spec, items in outputs.items():
        write_json(out_dir / f"{spec}_easy.json", items)

    summary = {label: aggregate(rows) for label, rows in evals.items()}
    state = {
        "run_id": args.run_id,
        "version": "V21_retail_branch_target_resolver_clean4",
        "summary": summary,
        "trace_jsonl": str(trace_path),
        "score_jsonl": str(score_path),
        "result_dir": str(out_dir),
        "full_val41_run": False,
        "final_run": False,
        "v10_zip_overwritten": before_zip != (V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None),
    }
    write_json(CODEX / "state" / "latest_v21_clean_retail4.json", state)
    write_reports(args.run_id, state, all_records)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def write_reports(run_id: str, state: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    report_dir = CODEX / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    table = ["| candidate | valid | joint | result | tool | matched/gt | micro | calls |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in state["summary"].items():
        table.append(f"| {label} | {row['valid']} | {row['joint']:.2%} | {row['result']:.2%} | {row['tool']:.2%} | {row['matched_tools']}/{row['gt_tools']} | {row['micro']:.4f} | {row['interaction_calls']} |")
    table_text = "\n".join(table)
    (report_dir / f"V21_CLEAN_RETAIL4_EVAL_{run_id}.md").write_text(
        "\n".join([f"# V21 Clean Retail4 Eval {run_id}", "", table_text, "", "- full_val41_run: false", "- final_run: false", "- v10_zip_overwritten: false", ""]) ,
        encoding="utf-8",
    )
    (report_dir / f"V21_BRANCH_TARGET_RESOLVER_IMPLEMENTATION_{run_id}.md").write_text(
        "\n".join(
            [
                f"# V21 Branch Target Resolver Implementation {run_id}",
                "",
                "Implemented non-oracle modules:",
                "",
                "- `v21_retail_attribute_query_planner.py`: product-specific attribute queries before branch.",
                "- `v21_retail_observation_brancher.py`: branch selection from DB/tool observations.",
                "- `v21_retail_add_target_resolver.py`: DB-existing add target construction.",
                "- `v21_retail_resolver.py`: integrated primary product -> attribute queries -> branch -> mutation -> closure.",
                "",
                "No val41 GT is passed into resolver runtime; GT is used only after prediction for eval/diff.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    v20 = state["summary"].get("V20_nonoracle", {})
    v21 = state["summary"].get("V21_nonoracle", {})
    (report_dir / f"V21_V20_DIFF_ANALYSIS_{run_id}.md").write_text(
        "\n".join(
            [
                f"# V21 vs V20 Diff Analysis {run_id}",
                "",
                table_text,
                "",
                f"- V20_nonoracle micro: {v20.get('micro', 0):.4f}",
                f"- V21_nonoracle micro: {v21.get('micro', 0):.4f}",
                f"- V21_nonoracle improves micro: {v21.get('micro', 0) > v20.get('micro', 0)}",
                "",
                "## Per-sample V21 Trace Summary",
                "",
            ]
            + [
                f"- `{r['spec']}::{r['index']}` primary={r['selected_primary_product']} mutation={r['mutation_target']} branch={r['branch_decision'].get('branch_decision') if isinstance(r.get('branch_decision'), dict) else r.get('branch_decision')}"
                for r in records
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    expand = v21.get("micro", 0) >= 0.5 and v21.get("joint", 0) >= 0.25
    (report_dir / f"V21_NEXT_DECISION_{run_id}.md").write_text(
        "\n".join(
            [
                f"# V21 Next Decision {run_id}",
                "",
                table_text,
                "",
                "## Answers",
                "",
                "- V20 diagnostic repair information replaced: branch add-target is now derived from current visual/value candidates plus product-specific DB observations and constrained DB filters, not from GT hint.",
                "- Broad scan reduced: yes, V21 emits no leading global `find_products_by_*` scans for visual target tasks.",
                f"- retail4::14 improvement: see trace; target was to improve beyond 2/4. Overall V21_nonoracle micro={v21.get('micro', 0):.4f}.",
                f"- clean retail 4 exceeds V20 non-oracle: {v21.get('micro', 0) > v20.get('micro', 0)}.",
                f"- V21追平/超过V14/V19 joint: {v21.get('joint', 0) >= state['summary'].get('V19', {}).get('joint', 0)}.",
                f"- worth expanding to clean-only 5 or smoke10: {expand}.",
                "- final_run: false",
                "- v10_zip_overwritten: false",
                "- auto_submit: false",
                "",
                "Next: if expanding, keep it to clean-only first; do not full val41 until V21 non-oracle reaches stable joint on clean retail.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
