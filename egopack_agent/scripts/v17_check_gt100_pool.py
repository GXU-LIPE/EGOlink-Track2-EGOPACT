#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V17 fail-fast GT100 pool hard check."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
EGO = Path("/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench")
GT16 = CODEX / "gt_distill_v16"
GT17 = CODEX / "gt_distill_v17"
FINAL309_SPECS = {"retail6", "retail10", "kitchen4", "restaurant5", "order2"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def scenario_number(spec: str) -> tuple[str, int]:
    scenario = "".join(ch for ch in spec if not ch.isdigit())
    num_s = spec[len(scenario):] or "0"
    return scenario, int(num_s)


def build_oracle_result(row: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    calls = row.get("tool_calls") or row.get("tool_chain") or []
    return {
        "task_id": task_id,
        "mode": "text",
        "instruction": row.get("instruction", ""),
        "image_description": row.get("image_description", ""),
        "dialogue": [{"role": "agent", "turn": 0, "content": "V17 GT100 pool replay check."}],
        "tool_calls": [{"turn": 0, "calls": calls, "blocked_calls": [], "results": []}],
        "tool_calls_count": len(calls),
        "rounds_count": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_consumed": 0,
        "final_run": False,
        "uses_val41_gt": False,
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
    return {
        "valid": metrics.get("valid_scenarios", 0),
        "joint": metrics.get("joint_success", {}).get("success_rate", 0),
        "result": metrics.get("result_based", {}).get("success_rate", 0),
        "tool": metrics.get("tool_based", {}).get("success_rate", 0),
        "micro": micro.get("micro_accuracy", 0),
        "correct_calls": micro.get("total_correct_calls", 0),
        "gt_calls": micro.get("total_ground_truth_calls", 0),
    }


def check_required_fields(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    missing = []
    for i, row in enumerate(rows, start=1):
        calls = row.get("tool_calls") or row.get("tool_chain")
        replay_info = row.get("replay_status") or row.get("source_path")
        for key in ("scenario", "spec", "task_id"):
            if row.get(key) in (None, ""):
                missing.append({"line": i, "missing": key, "pool_id": row.get("pool_id")})
        if not isinstance(calls, list) or not calls:
            missing.append({"line": i, "missing": "tool_chain", "pool_id": row.get("pool_id")})
        if not replay_info:
            missing.append({"line": i, "missing": "replay_info", "pool_id": row.get("pool_id")})
        if len(missing) >= 20:
            break
    return {"ok": not missing, "examples": missing}


def replay_sample(rows: List[Dict[str, Any]], run_id: str, n: int) -> Dict[str, Any]:
    scenario_rows = [r for r in rows if r.get("source_kind") == "scenario_gt" and r.get("replay_status") == "joint100"]
    sample = random.Random(17).sample(scenario_rows, min(n, len(scenario_rows)))
    replay_dir = CODEX / "runs" / "V17_GT100_POOL_CHECK" / run_id
    rows_by_spec: Dict[str, List[Dict[str, Any]]] = {}
    for row in sample:
        rows_by_spec.setdefault(row["spec"], []).append(row)
    eval_rows = []
    for spec, items in rows_by_spec.items():
        scenario, number = scenario_number(spec)
        gt_items = []
        pred_items = []
        for idx, row in enumerate(items, start=1):
            gt_items.append({
                "Instruction": row.get("instruction", ""),
                "analysis": row.get("analysis", ""),
                "image_description": row.get("image_description", ""),
                "task_id": idx,
                "ground_truth": row.get("tool_calls") or [],
            })
            pred_items.append(build_oracle_result(row, idx))
        gt_path = replay_dir / "gt" / f"{spec}.json"
        pred_path = replay_dir / "pred" / f"{spec}_easy.json"
        write_json(gt_path, gt_items)
        write_json(pred_path, pred_items)
        try:
            metrics = evaluate_spec(gt_path, pred_path, scenario, number)
            ok = metrics["valid"] == len(items) and metrics["joint"] >= 0.999 and metrics["tool"] >= 0.999
        except Exception as exc:
            metrics = {"error": f"{type(exc).__name__}: {exc}"}
            ok = False
        eval_rows.append({"spec": spec, "n": len(items), "ok": ok, "metrics": metrics})
    return {
        "requested": n,
        "sampled": len(sample),
        "specs": eval_rows,
        "all_joint100": all(x["ok"] for x in eval_rows) and len(sample) == n,
    }


def write_report(run_id: str, manifest: Dict[str, Any], checks: Dict[str, Any]) -> Path:
    report = CODEX / "reports" / f"V17_GT100_POOL_CHECK_{run_id}.md"
    lines = [
        f"# V17 GT100 Pool Check {run_id}",
        "",
        f"- status: {checks['status']}",
        "- final_run: false",
        "- uses_final_hidden_metadata: false",
        "- uses_val41_label_for_training: false",
        "",
        "## Hard Checks",
        "",
    ]
    for key, val in checks["hard_checks"].items():
        lines.append(f"- {key}: {val}")
    lines += [
        "",
        "## Manifest",
        "",
        f"- total_pool_rows: {manifest.get('total_pool_rows')}",
        f"- scenario_joint100_kept: {manifest.get('scenario_joint100_kept')}",
        f"- train_data_fallback_kept: {manifest.get('train_data_fallback_kept')}",
        f"- excludes_official_final309: {manifest.get('excludes_official_final309')}",
        f"- excludes_frozen_val41: {manifest.get('excludes_frozen_val41')}",
        "",
        "## Replay Sample",
        "",
        f"- requested: {checks['replay_sample'].get('requested')}",
        f"- sampled: {checks['replay_sample'].get('sampled')}",
        f"- all_joint100: {checks['replay_sample'].get('all_joint100')}",
        "",
        "| spec | n | ok | joint | tool | micro | error |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in checks["replay_sample"].get("specs", []):
        m = row.get("metrics", {})
        lines.append(f"| {row.get('spec')} | {row.get('n')} | {row.get('ok')} | {m.get('joint', 0):.3f} | {m.get('tool', 0):.3f} | {m.get('micro', 0):.3f} | {m.get('error', '')} |")
    if checks.get("fail_reasons"):
        lines += ["", "## Fail Reasons", ""]
        for reason in checks["fail_reasons"]:
            lines.append(f"- {reason}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=f"v17_gt100_check_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    pool_path = GT16 / "gt100_pool.jsonl"
    manifest_path = GT16 / "gt100_pool_manifest.json"
    fail = []
    rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {}
    if not pool_path.exists():
        fail.append("gt100_pool.jsonl missing")
    else:
        rows = read_jsonl(pool_path)
    if not manifest_path.exists():
        fail.append("gt100_pool_manifest.json missing")
    else:
        manifest = read_json(manifest_path)

    field_check = check_required_fields(rows) if rows else {"ok": False, "examples": ["no rows"]}
    specs = {r.get("spec") for r in rows}
    final_overlap = sorted(specs & FINAL309_SPECS)
    replay = replay_sample(rows, args.run_id, args.sample) if len(rows) >= args.sample else {"all_joint100": False, "requested": args.sample, "sampled": len(rows), "specs": []}
    hard = {
        "pool_exists": pool_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "row_count_ge_600": len(rows) >= 600,
        "required_fields_present": field_check["ok"],
        "manifest_excludes_final309": manifest.get("excludes_official_final309") is True,
        "manifest_excludes_val41": manifest.get("excludes_frozen_val41") is True,
        "no_final309_spec_overlap": not final_overlap,
        "random_20_replay_joint100": replay.get("all_joint100") is True,
    }
    if not hard["row_count_ge_600"]:
        fail.append(f"row_count {len(rows)} < 600")
    if not hard["required_fields_present"]:
        fail.append(f"required field failures: {field_check['examples'][:5]}")
    if not hard["manifest_excludes_final309"]:
        fail.append("manifest does not explicitly exclude final309")
    if not hard["manifest_excludes_val41"]:
        fail.append("manifest does not explicitly exclude frozen val41")
    if final_overlap:
        fail.append(f"final309 specs present in pool: {final_overlap}")
    if not hard["random_20_replay_joint100"]:
        fail.append("random replay sample did not all reach joint100")

    checks = {
        "run_id": args.run_id,
        "status": "PASS" if not fail else "FAIL",
        "hard_checks": hard,
        "row_count": len(rows),
        "required_field_check": field_check,
        "final_overlap": final_overlap,
        "replay_sample": replay,
        "fail_reasons": fail,
    }
    GT17.mkdir(parents=True, exist_ok=True)
    out_manifest = {
        "source_manifest": str(manifest_path),
        "source_pool": str(pool_path),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": checks["status"],
        "row_count": len(rows),
        "checks": checks,
        "uses_final_hidden_metadata": False,
        "uses_val41_label_for_training": False,
    }
    report = write_report(args.run_id, manifest, checks)
    out_manifest["report"] = str(report)
    write_json(GT17 / "gt100_verified_manifest.json", out_manifest)
    print(json.dumps(out_manifest, ensure_ascii=False, indent=2))
    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
