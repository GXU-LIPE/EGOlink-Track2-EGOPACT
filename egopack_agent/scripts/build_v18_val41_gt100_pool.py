#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the V18 val41-oracle GT100 pool.

Boundary:
- Reads only frozen validation_A_limit30 materialized tasks.
- Uses val41 GT labels only for oracle/self-distillation diagnostics.
- Does not open official final309 scenario JSON and does not call APIs.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODEX = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
SPLIT_DIR = CODEX / "state" / "materialized_splits" / "validation_A_limit30"
OUT_DIR = CODEX / "gt_distill_v18_val41_oracle"
VERSION = "V18_VAL41_ORACLE_GT100_POOL"


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def call_name(call: Dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or "")


def build_oracle_result(row: Dict[str, Any], subset_index: int, trace_note: str) -> Dict[str, Any]:
    calls = row.get("ground_truth") or []
    return {
        "task_id": subset_index,
        "mode": "text",
        "instruction": row.get("Instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": trace_note}],
        "tool_calls": [{"turn": 0, "calls": calls, "blocked_calls": [], "results": [], "source": "val41_gt_replay"}],
        "tool_calls_count": len(calls),
        "rounds_count": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_consumed": 0,
        "final_run": False,
        "uses_val41_gt": True,
        "uses_final_hidden_metadata": False,
    }


def evaluate_spec(gt_path: Path, pred_path: Path, scenario: str, number: int) -> Dict[str, Any]:
    sys.path.insert(0, str(EGO / "analysis_scripts"))
    sys.path.insert(0, str(EGO))
    from evaluate_interaction import evaluate_interaction_success
    import argparse as _argparse

    metrics = evaluate_interaction_success(
        str(gt_path),
        str(pred_path),
        scenario=scenario,
        args=_argparse.Namespace(scenario_number=number),
        silent=True,
        num_samples=0,
    )
    micro = metrics.get("micro_tool_stats", {}) or {}
    detail = (metrics.get("detailed_results") or [{}])[0]
    return {
        "valid": metrics.get("valid_scenarios", 0),
        "joint": metrics.get("joint_success", {}).get("success_rate", 0.0),
        "result": metrics.get("result_based", {}).get("success_rate", 0.0),
        "tool": metrics.get("tool_based", {}).get("success_rate", 0.0),
        "micro": micro.get("micro_accuracy", 0.0),
        "correct_calls": micro.get("total_correct_calls", 0),
        "gt_calls": micro.get("total_ground_truth_calls", 0),
        "interaction_calls": micro.get("total_interaction_calls", 0),
        "detail": detail,
    }


def replay_one(spec: str, scenario: str, number: int, subset_index: int, row: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    replay_dir = CODEX / "runs" / VERSION / run_id / "replay"
    uid = f"{spec}_{subset_index}_{row.get('_v8_original_index', 'na')}"
    gt_path = replay_dir / "gt" / f"{uid}.json"
    pred_path = replay_dir / "pred" / f"{uid}_easy.json"
    write_json(gt_path, [row])
    write_json(pred_path, [build_oracle_result(row, subset_index, "V18 val41 GT100 replay check.")])
    try:
        metrics = evaluate_spec(gt_path, pred_path, scenario, number)
        ok = metrics.get("valid") == 1 and metrics.get("joint", 0) >= 0.999 and metrics.get("tool", 0) >= 0.999
        reason = "" if ok else f"not_joint100:{metrics}"
    except Exception as exc:
        metrics = {"error": f"{type(exc).__name__}: {exc}"}
        ok = False
        reason = metrics["error"]
    return {"ok": ok, "reason": reason, "metrics": metrics}


def load_val41_rows() -> List[Dict[str, Any]]:
    manifest = read_json(SPLIT_DIR / "manifest.json")
    out: List[Dict[str, Any]] = []
    for scenario, number, idxs in manifest.get("specs", []):
        spec = f"{scenario}{int(number)}"
        rows = read_json(SPLIT_DIR / f"{spec}.json")
        if isinstance(rows, dict):
            rows = list(rows.values())
        for pos, row in enumerate(rows):
            source_index = row.get("_v8_original_index")
            if source_index is None and pos < len(idxs):
                source_index = idxs[pos]
            out.append({
                "spec": spec,
                "scenario": str(scenario),
                "number": int(number),
                "subset_index": pos + 1,
                "source_original_index": source_index,
                "manifest_index": idxs[pos] if pos < len(idxs) else None,
                "row": row,
            })
    return out


def make_pool_row(item: Dict[str, Any], replay: Dict[str, Any]) -> Dict[str, Any]:
    row = item["row"]
    calls = row.get("ground_truth") or []
    pool_id = f"{item['spec']}::{item['source_original_index']}::{item['subset_index']}"
    return {
        "pool_id": pool_id,
        "source_kind": "val41_gt_oracle",
        "scenario": item["scenario"],
        "number": item["number"],
        "spec": item["spec"],
        "subset_index": item["subset_index"],
        "source_original_index": item["source_original_index"],
        "manifest_index": item["manifest_index"],
        "materialized_task_id": row.get("task_id"),
        "instruction": row.get("Instruction", ""),
        "analysis": row.get("analysis", ""),
        "image_name": row.get("image_name", ""),
        "image_path": row.get("image_path", ""),
        "image_description": row.get("image_description", ""),
        "key": row.get("key"),
        "value": row.get("value"),
        "tool_chain": calls,
        "tool_names": [call_name(c) for c in calls],
        "tool_count": len(calls),
        "replay_status": "joint100",
        "replay_metrics": replay.get("metrics", {}),
        "uses_val41_gt": True,
        "uses_final_hidden_metadata": False,
        "for_final_candidate": False,
    }


def write_report(run_id: str, total: int, kept: List[Dict[str, Any]], failed: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V18_VAL41_GT100_POOL_AUDIT_{run_id}.md"
    by_spec = collections.Counter(row["spec"] for row in kept)
    fail_by_reason = collections.Counter(f["reason"].split(":", 1)[0] for f in failed)
    lines = [
        f"# V18 Val41 GT100 Pool Audit {run_id}",
        "",
        "- version: V18_VAL41_ORACLE_GT100_POOL",
        "- final_run: false",
        "- oracle_self_distillation_diagnostic: true",
        "- uses_val41_gt_for_pool: true",
        "- uses_final_hidden_metadata: false",
        "- for_final_candidate: false",
        "",
        "## Summary",
        "",
        f"- val41_total_tasks: {total}",
        f"- gt_replay_attempted: {total}",
        f"- joint100_retained: {len(kept)}",
        f"- replay_failed: {len(failed)}",
        f"- pool_path: `{OUT_DIR / 'val41_gt100_pool.jsonl'}`",
        f"- manifest_path: `{OUT_DIR / 'val41_gt100_manifest.json'}`",
        "",
        "## Retained By Spec",
        "",
        "| spec | retained |",
        "|---|---:|",
    ]
    for spec, count in sorted(by_spec.items()):
        lines.append(f"| {spec} | {count} |")
    lines += ["", "## Replay Failure Reasons", ""]
    if failed:
        for reason, count in sorted(fail_by_reason.items()):
            lines.append(f"- {reason}: {count}")
        lines += ["", "| uid | reason |", "|---|---|"]
        for row in failed[:80]:
            lines.append(f"| {row.get('uid')} | {str(row.get('reason'))[:500]} |")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Manifest Digest",
        "",
        f"- status: {manifest.get('status')}",
        f"- expected_min_joint100: {manifest.get('expected_min_joint100')}",
        f"- passed_min_gate: {manifest.get('passed_min_gate')}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v18_val41_gt100_pool_{stamp()}")
    ap.add_argument("--min-joint100", type=int, default=20)
    args = ap.parse_args()

    items = load_val41_rows()
    kept: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for item in items:
        replay = replay_one(item["spec"], item["scenario"], item["number"], item["subset_index"], item["row"], args.run_id)
        uid = f"{item['spec']}::{item['source_original_index']}::{item['subset_index']}"
        if replay["ok"]:
            kept.append(make_pool_row(item, replay))
        else:
            failed.append({
                "uid": uid,
                "spec": item["spec"],
                "scenario": item["scenario"],
                "number": item["number"],
                "subset_index": item["subset_index"],
                "source_original_index": item["source_original_index"],
                "reason": replay["reason"],
                "metrics": replay.get("metrics", {}),
            })

    passed = len(kept) >= args.min_joint100
    manifest = {
        "run_id": args.run_id,
        "version": VERSION,
        "status": "PASS" if passed else "FAIL",
        "source_split": str(SPLIT_DIR),
        "val41_total_tasks": len(items),
        "gt_replay_attempted": len(items),
        "joint100_retained": len(kept),
        "replay_failed": len(failed),
        "expected_min_joint100": args.min_joint100,
        "passed_min_gate": passed,
        "uses_val41_gt": True,
        "uses_final_hidden_metadata": False,
        "for_final_candidate": False,
        "by_spec": dict(collections.Counter(row["spec"] for row in kept)),
        "failed_examples": failed[:20],
        "pool_path": str(OUT_DIR / "val41_gt100_pool.jsonl"),
    }
    write_jsonl(OUT_DIR / "val41_gt100_pool.jsonl", kept)
    write_json(OUT_DIR / "val41_gt100_manifest.json", manifest)
    report = write_report(args.run_id, len(items), kept, failed, manifest)
    state = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": args.run_id,
        "manifest": str(OUT_DIR / "val41_gt100_manifest.json"),
        "pool": str(OUT_DIR / "val41_gt100_pool.jsonl"),
        "report": str(report),
        "summary": manifest,
        "final_run": False,
    }
    write_json(CODEX / "state" / "latest_v18_val41_gt100_pool.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(f"GT100 pool too small: {len(kept)} < {args.min_joint100}")


if __name__ == "__main__":
    main()
