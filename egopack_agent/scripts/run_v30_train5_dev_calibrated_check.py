#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-set 5-scenario diagnostic for V30 dev-calibrated prior agent.

This is deliberately a small dev/oracle-assisted check. It uses only
train_data/sft_track2_tooluse_train.jsonl entries tagged no_final_gt=true, plus
V30 prior bank and post-eval calibration. It does not run final and does not
read scenarios/final.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
V10_ZIP = CODEX / "submissions" / "V10_full_memory_final_candidate_draft_track2.zip"
EXPECTED_V10_SHA = "5f2b8aacf986e4b3107b560e76fe8dfc52b5b3da6e407c9e0e6d0433db36849d"

sys.path.insert(0, str(CODEX / "wrappers"))
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(EGO))
sys.path.insert(0, str(EGO / "analysis_scripts"))

from egobench_agent_plus.v24_candidate_dryrun_and_selector import dryrun_program  # noqa: E402
from egobench_agent_plus.v30_prior_bank import ExperiencePriorBank  # noqa: E402
from egobench_agent_plus.v30_prior_retrieval_agent import PriorRetrievalServiceAgentV30  # noqa: E402
from egobench_agent_plus.v30_selector import ProtectedPriorSelectorV30  # noqa: E402


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def parse_spec(spec: str) -> Tuple[str, int]:
    m = re.fullmatch(r"([a-z]+)(\d+)", spec or "")
    if not m:
        raise ValueError(f"Bad scenario spec: {spec}")
    return m.group(1), int(m.group(2))


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


def make_eval_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    messages = sample.get("messages") or []
    user = ""
    for msg in messages:
        if msg.get("role") == "user":
            user = msg.get("content") or ""
            break
    target = json.loads(sample.get("target") or "[]")
    return {
        "task_id": int(str(sample.get("task_id") or "0") or 0),
        "Instruction": user,
        "image_description": sample.get("visual_context") or "",
        "analysis": sample.get("planner_state") or "",
        "ground_truth": target,
        "key": sample.get("scenario"),
        "value": sample.get("id"),
        "_source_sft_id": sample.get("id"),
        "_train_source": sample.get("source"),
        "_compliance_tags": sample.get("compliance_tags") or {},
    }


def make_item(row: Dict[str, Any], program: List[Dict[str, Any]], label: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "task_id": row.get("task_id", 1),
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": f"{label} train5 diagnostic candidate."}],
        "tool_calls": [
            {
                "turn": 0,
                "calls": [
                    {"tool_name": c.get("tool_name") or c.get("name"), "parameters": c.get("parameters") or {}}
                    for c in (program or [])
                ],
                "blocked_calls": [],
                "results": [],
                "v30_train5_meta": meta or {},
            }
        ],
        "tool_calls_count": len(program or []),
        "rounds_count": 1,
        "final_run": False,
        "uses_final_hidden_metadata": False,
    }


def evaluate_one(gt_item: Dict[str, Any], pred_item: Dict[str, Any], scenario: str, number: int) -> Dict[str, Any]:
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    with tempfile.TemporaryDirectory(prefix="v30_train5_eval_") as td:
        gt_path = Path(td) / "gt.json"
        pred_path = Path(td) / "pred.json"
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
        "tool_error": tb.get("error"),
        "result_error": rb.get("error"),
    }


def score_tuple(score: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        float(score.get("joint", 0) or 0),
        float(score.get("tool", 0) or 0),
        float(score.get("result", 0) or 0),
        int(score.get("matches", 0) or 0),
        -int(score.get("interaction_calls", 999999) or 999999),
    )


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    matched = sum(int(r.get("matches", 0) or 0) for r in rows)
    gt = sum(int(r.get("gt_calls", 0) or 0) for r in rows)
    return {
        "valid": n,
        "joint": sum(float(r.get("joint", 0) or 0) for r in rows) / n if n else 0.0,
        "result": sum(float(r.get("result", 0) or 0) for r in rows) / n if n else 0.0,
        "tool": sum(float(r.get("tool", 0) or 0) for r in rows) / n if n else 0.0,
        "micro": matched / gt if gt else 0.0,
        "matched_tools": matched,
        "gt_tools": gt,
        "interaction_calls": sum(int(r.get("interaction_calls", 0) or 0) for r in rows),
    }


def table_row(name: str, s: Dict[str, Any]) -> str:
    return f"| {name} | {s.get('valid',0)} | {s.get('joint',0)*100:.2f}% | {s.get('result',0)*100:.2f}% | {s.get('tool',0)*100:.2f}% | {s.get('matched_tools',0)}/{s.get('gt_tools',0)} | {s.get('micro',0):.4f} | {s.get('interaction_calls',0)} |"


def load_train_samples(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tags = r.get("compliance_tags") or {}
            if r.get("source") != "gt" or not tags.get("no_final_gt"):
                continue
            if r.get("mode") != "dev":
                continue
            try:
                target = json.loads(r.get("target") or "[]")
            except Exception:
                continue
            if not target:
                continue
            rows.append(r)
    return rows


def choose_samples(rows: List[Dict[str, Any]], preferred_specs: List[str], limit: int) -> List[Dict[str, Any]]:
    by_spec: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_spec.setdefault(r.get("scenario"), []).append(r)
    chosen: List[Dict[str, Any]] = []
    used = set()
    for spec in preferred_specs:
        if spec in by_spec and by_spec[spec]:
            chosen.append(by_spec[spec][0])
            used.add(spec)
            if len(chosen) >= limit:
                return chosen
    # Fill remaining with distinct family/spec rows, prioritising missing order/restaurant/kitchen/retail families.
    families_have = {parse_spec(r.get("scenario"))[0] for r in chosen}
    for family in ["retail", "restaurant", "kitchen", "order"]:
        if family in families_have:
            continue
        for spec in sorted(by_spec):
            if spec in used:
                continue
            if parse_spec(spec)[0] == family:
                chosen.append(by_spec[spec][0])
                used.add(spec)
                families_have.add(family)
                break
    for spec in sorted(by_spec):
        if len(chosen) >= limit:
            break
        if spec in used:
            continue
        chosen.append(by_spec[spec][0])
        used.add(spec)
    return chosen[:limit]


def run_mode(
    samples: List[Dict[str, Any]],
    bank_dir: Path,
    mode: str,
    run_id: str,
    candidate_path: Path,
    selection_path: Path,
) -> Dict[str, Any]:
    bank = ExperiencePriorBank(bank_dir)
    agent = PriorRetrievalServiceAgentV30(bank, mode=mode)
    selector = ProtectedPriorSelectorV30()
    records: List[Dict[str, Any]] = []
    selected_scores: List[Dict[str, Any]] = []
    oracle_scores: List[Dict[str, Any]] = []
    exact_gt_scores: List[Dict[str, Any]] = []

    for sample in samples:
        spec = sample["scenario"]
        scenario, number = parse_spec(spec)
        row = make_eval_row(sample)
        task_key = f"train::{spec}::{sample.get('task_id')}"
        gt_item = make_item(row, row["ground_truth"], "EXACT_GT", {"diagnostic_only": True})
        exact_gt = evaluate_one(row, gt_item, scenario, number)
        exact_gt_scores.append(exact_gt)

        # Baseline for this train diagnostic is empty/no-tool; there is no V22 train result dir.
        baseline_item = make_item(row, [], "EMPTY_BASELINE", {"baseline": "empty_no_tool"})
        baseline_score = evaluate_one(row, baseline_item, scenario, number)

        db = init_db(scenario, number)
        candidates = agent.build_candidates(task_key, scenario, row, db, top_k=8)
        for cand in candidates:
            cand["dryrun"] = dryrun_program(scenario, init_db(scenario, number), cand.get("tool_program") or [], row.get("Instruction", ""))
            cand_item = make_item(row, cand.get("tool_program") or [], cand.get("candidate_id", "cand"), {"mode": mode})
            cand["post_eval_score"] = evaluate_one(row, cand_item, scenario, number)
            append_jsonl(candidate_path, {
                "run_id": run_id,
                "mode": mode,
                "task_key": task_key,
                "candidate_id": cand.get("candidate_id"),
                "prior_id": cand.get("prior_id"),
                "prior_case_id": cand.get("prior_case_id"),
                "program_family": cand.get("program_family"),
                "retrieval_score": cand.get("retrieval_score"),
                "slot_source": cand.get("slot_source"),
                "slot_values_used": cand.get("slot_values_used"),
                "tool_program": cand.get("tool_program"),
                "dryrun": cand.get("dryrun"),
                "post_eval_score": cand.get("post_eval_score"),
                "trace": cand.get("trace"),
            })

        selection = selector.select(task_key, baseline_item, baseline_score, candidates, mode=mode)
        selected = selection.get("selected")
        if isinstance(selected, dict) and selected.get("tool_program") is not None:
            selected_item = make_item(row, selected.get("tool_program") or [], selected.get("candidate_id", "selected"), {"selection": selection, "mode": mode})
        else:
            selected_item = baseline_item
        selected_score = evaluate_one(row, selected_item, scenario, number)

        scored = [(cand, cand.get("post_eval_score") or {}) for cand in candidates]
        best_cand, best_score = max(scored + [({"candidate_id": "EMPTY_BASELINE"}, baseline_score)], key=lambda x: score_tuple(x[1]))
        selected_scores.append(selected_score)
        oracle_scores.append(best_score)

        rec = {
            "run_id": run_id,
            "mode": mode,
            "task_key": task_key,
            "sft_id": sample.get("id"),
            "spec": spec,
            "scenario_family": scenario,
            "task_id": sample.get("task_id"),
            "instruction": row.get("Instruction"),
            "visual_context": row.get("image_description"),
            "planner_state": row.get("analysis"),
            "gt_tool_names": [c.get("tool_name") or c.get("name") for c in row.get("ground_truth") or []],
            "baseline_score": baseline_score,
            "exact_gt_replay_score": exact_gt,
            "selected_score": selected_score,
            "oracle_score": best_score,
            "candidate_count": len(candidates),
            "selected_candidate_id": selection.get("selected_candidate_id"),
            "selected_source": selection.get("selected_source"),
            "selected_prior_id": selection.get("selected_prior_id"),
            "selected_program_family": selection.get("selected_program_family"),
            "selected_reason": selection.get("reason"),
            "uses_post_eval_for_selection": selection.get("uses_post_eval_for_selection"),
            "oracle_candidate_id": best_cand.get("candidate_id"),
            "oracle_prior_id": best_cand.get("prior_id"),
            "oracle_program_family": best_cand.get("program_family"),
        }
        records.append(rec)
        append_jsonl(selection_path, rec)

    return {
        "mode": mode,
        "records": records,
        "summary": aggregate(selected_scores),
        "oracle_summary": aggregate(oracle_scores),
        "exact_gt_replay_summary": aggregate(exact_gt_scores),
    }


def write_reports(run_id: str, state: Dict[str, Any]) -> List[str]:
    reports = CODEX / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    main = reports / f"V30_TRAIN5_DEV_CALIBRATED_CHECK_{run_id}.md"
    audit = reports / f"V30_TRAIN5_SAMPLE_AUDIT_{run_id}.md"
    next_decision = reports / f"V30_TRAIN5_NEXT_DECISION_{run_id}.md"

    rows = [
        "| metric set | valid | joint | result | tool | matched/gt | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("empty_baseline", state["empty_baseline"]),
        table_row("slot_only_selected", state["slot_only"]["summary"]),
        table_row("slot_only_oracle_best", state["slot_only"]["oracle_summary"]),
        table_row("dev_calibrated_selected", state["dev_calibrated"]["summary"]),
        table_row("dev_calibrated_oracle_best", state["dev_calibrated"]["oracle_summary"]),
        table_row("exact_gt_replay", state["exact_gt_replay"]),
    ]
    per_task_lines = [
        "| task | family | selected joint | selected micro | oracle joint | candidate | prior | family |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for r in state["dev_calibrated"]["records"]:
        ss = r.get("selected_score") or {}
        oscore = r.get("oracle_score") or {}
        per_task_lines.append(
            f"| {r.get('task_key')} | {r.get('scenario_family')} | {int(ss.get('joint',0))} | {float(ss.get('micro',0)):.4f} | {int(oscore.get('joint',0))} | {r.get('selected_candidate_id')} | {r.get('selected_prior_id')} | {r.get('selected_program_family')} |"
        )

    main.write_text("\n".join([
        f"# V30 Train5 Dev-Calibrated Check {run_id}",
        "",
        "Scope: 5 train/dev SFT GT samples tagged `no_final_gt=true`; uses V30 val41 GT/V29 experience prior bank and post-eval calibration as requested.",
        "This is dev-only/oracle-assisted diagnostics, not final-safe generalization evidence.",
        "",
        *rows,
        "",
        "## Per Task",
        *per_task_lines,
        "",
        f"- train_samples_total: {state['train_samples_total']}",
        f"- sampled_specs: `{state['sampled_specs']}`",
        f"- used_val41_v29_prior_bank: {state['used_val41_v29_prior_bank']}",
        f"- used_post_eval_calibration: {state['used_post_eval_calibration']}",
        f"- final_run: {state['final_run']}",
        f"- final_hidden_metadata_used: {state['uses_final_hidden_metadata']}",
        f"- V10_zip_sha256_before: `{state['V10_zip_sha256_before']}`",
        f"- V10_zip_overwritten: {state['v10_zip_overwritten']}",
    ]) + "\n", encoding="utf-8")

    audit_lines = [
        f"# V30 Train5 Sample Audit {run_id}",
        "",
        "| task | sft_id | source | no_final_gt | target_tools | user_prefix |",
        "|---|---|---|---:|---|---|",
    ]
    for r in state["dev_calibrated"]["records"]:
        prefix = (r.get("instruction") or "").replace("|", " ")[:120]
        audit_lines.append(
            f"| {r.get('task_key')} | {r.get('sft_id')} | sft_track2_tooluse_train | true | {','.join(r.get('gt_tool_names') or [])} | {prefix} |"
        )
    audit_lines.extend([
        "",
        "- final hidden metadata used: false",
        "- source file: `train_data/sft_track2_tooluse_train.jsonl`",
        "- filter: `source == gt`, `mode == dev`, `compliance_tags.no_final_gt == true`, target non-empty.",
    ])
    audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    decision = "dev_calibrated works on this small train diagnostic" if state["dev_calibrated_joint_count"] == state["sample_count"] else "dev_calibrated has train failures; inspect records"
    next_decision.write_text("\n".join([
        f"# V30 Train5 Next Decision {run_id}",
        "",
        f"- decision: {decision}",
        f"- dev_calibrated_joint_count: {state['dev_calibrated_joint_count']}/{state['sample_count']}",
        f"- slot_only_joint_count: {state['slot_only_joint_count']}/{state['sample_count']}",
        "- This check validates the dev-calibrated mechanism on train samples only; it should not be promoted as final-safe.",
        "- Next useful check, if desired: repeat on a held-out train/dev subset without task-specific post-eval calibration, then compare against V22/V30 slot-only.",
        "- no final run, no final metadata, no V10 zip overwrite, no auto-submit.",
    ]) + "\n", encoding="utf-8")
    return [str(main), str(audit), str(next_decision)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", default=str(CODEX / "memory_bank_v30_gt_experience_prior"))
    ap.add_argument("--train-jsonl", default=str(CODEX / "train_data" / "sft_track2_tooluse_train.jsonl"))
    ap.add_argument("--run-id", default="v30_train5_dev_calibrated_" + stamp())
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--preferred-specs", default="retail1,restaurant1,kitchen1,order1,retail4")
    args = ap.parse_args()

    run_id = args.run_id
    before_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    v10_sha = subprocess.check_output(["sha256sum", str(V10_ZIP)], text=True).split()[0] if V10_ZIP.exists() else ""
    if v10_sha != EXPECTED_V10_SHA:
        raise SystemExit(f"V10 protected zip sha mismatch: {v10_sha}")

    train_rows = load_train_samples(Path(args.train_jsonl))
    samples = choose_samples(train_rows, [x.strip() for x in args.preferred_specs.split(",") if x.strip()], args.limit)
    if len(samples) != args.limit:
        raise SystemExit(f"Expected {args.limit} samples, got {len(samples)}")

    candidate_path = CODEX / "analysis" / f"v30_train5_candidate_programs_{run_id}.jsonl"
    selection_path = CODEX / "analysis" / f"v30_train5_selection_trace_{run_id}.jsonl"
    candidate_path.write_text("", encoding="utf-8")
    selection_path.write_text("", encoding="utf-8")

    # Empty baseline scores are identical across modes; derive them from slot records after the run.
    slot = run_mode(samples, Path(args.bank_dir), "slot_only", run_id, candidate_path, selection_path)
    dev = run_mode(samples, Path(args.bank_dir), "dev_calibrated", run_id, candidate_path, selection_path)
    empty_scores = [r["baseline_score"] for r in dev["records"]]
    exact_gt = dev["exact_gt_replay_summary"]

    after_mtime = V10_ZIP.stat().st_mtime_ns if V10_ZIP.exists() else None
    state = {
        "run_id": run_id,
        "version": "V30_TRAIN5_DEV_CALIBRATED_CHECK",
        "bank_dir": str(args.bank_dir),
        "bank_manifest": read_json(Path(args.bank_dir) / "manifest.json", {}),
        "train_jsonl": str(args.train_jsonl),
        "train_samples_total": len(train_rows),
        "sample_count": len(samples),
        "sampled_specs": [s.get("scenario") for s in samples],
        "sampled_ids": [s.get("id") for s in samples],
        "empty_baseline": aggregate(empty_scores),
        "slot_only": {"summary": slot["summary"], "oracle_summary": slot["oracle_summary"], "records": slot["records"]},
        "dev_calibrated": {"summary": dev["summary"], "oracle_summary": dev["oracle_summary"], "records": dev["records"]},
        "exact_gt_replay": exact_gt,
        "slot_only_joint_count": round(slot["summary"]["joint"] * len(samples)),
        "dev_calibrated_joint_count": round(dev["summary"]["joint"] * len(samples)),
        "used_val41_v29_prior_bank": True,
        "used_post_eval_calibration": True,
        "final_safe": False,
        "uses_final_hidden_metadata": False,
        "final_run": False,
        "auto_submit": False,
        "V10_zip_sha256_before": v10_sha,
        "v10_zip_overwritten": before_mtime != after_mtime,
        "analysis_files": [str(candidate_path), str(selection_path)],
    }
    report_paths = write_reports(run_id, state)
    state["report_paths"] = report_paths
    state_path = CODEX / "state" / "latest_v30_train5_dev_calibrated_check.json"
    write_json(state_path, state)
    compact_path = CODEX / "analysis" / f"v30_train5_records_compact_{run_id}.jsonl"
    write_jsonl(compact_path, dev["records"])
    state["compact_records"] = str(compact_path)
    write_json(state_path, state)
    print(json.dumps({
        "run_id": run_id,
        "sampled_specs": state["sampled_specs"],
        "sampled_ids": state["sampled_ids"],
        "slot_only": state["slot_only"]["summary"],
        "dev_calibrated": state["dev_calibrated"]["summary"],
        "exact_gt_replay": state["exact_gt_replay"],
        "reports": report_paths,
        "state": str(state_path),
        "v10_zip_overwritten": state["v10_zip_overwritten"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
