#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect one GPT-5.5 Track2 gate run into stable reports.

This script intentionally summarizes the current TRACK2_RUN_VERSION only. The
older collector assumed a fixed four-version matrix and could overwrite the
best state with zeros when only one version had been run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


CODEX_ROOT = Path(os.environ.get("CODEX_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/codex"))
EGO_ROOT = Path(os.environ.get("EGO_ROOT", "/home/data-gxu/acm/egolink2026-main/code/track2/EgoBench"))
DEFAULT_TASKS = ["retail9", "kitchen2", "restaurant4", "order1"]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return default


def _metric(data: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _metrics_from_eval(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "joint_success": 0.0,
            "tool_success": 0.0,
            "result_success": 0.0,
            "micro_tool_accuracy": 0.0,
            "avg_rounds": 0.0,
            "avg_tool_calls": 0.0,
            "interaction_tool_calls": 0.0,
            "gt_tool_calls": 0.0,
            "tool_matches": 0.0,
            "missing": True,
        }
    detailed = data.get("detailed_results") or []
    first = detailed[0] if detailed and isinstance(detailed[0], dict) else {}
    first_tool = first.get("tool_based") if isinstance(first.get("tool_based"), dict) else {}
    return {
        "joint_success": _float(_metric(data, "joint_success.success_rate")),
        "tool_success": _float(_metric(data, "tool_based.success_rate")),
        "result_success": _float(_metric(data, "result_based.success_rate")),
        "micro_tool_accuracy": _float(_metric(data, "micro_tool_stats.micro_accuracy")),
        "avg_rounds": _float(_metric(data, "performance_metrics.avg_rounds_count")),
        "avg_tool_calls": _float(_metric(data, "performance_metrics.avg_tool_calls_count")),
        "interaction_tool_calls": _float(first_tool.get("total_interaction_calls")),
        "gt_tool_calls": _float(first_tool.get("total_gt_calls")),
        "tool_matches": _float(first_tool.get("matches")),
        "missing": False,
    }


def _api_usage_for_run(run_id: str) -> Dict[str, Any]:
    calls = images = inp = out = errors = fallbacks = 0
    for path in (CODEX_ROOT / "logs").glob("openai_gpt55_adapter_*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Older telemetry did not include run_id, so report a snapshot
                # rather than claiming exact per-run accounting.
                calls += 1
                images += int(rec.get("image_count") or 0)
                inp += int(rec.get("input_token_estimate") or 0)
                out += int(rec.get("output_token_estimate") or 0)
                if rec.get("api_error"):
                    errors += 1
                if rec.get("fallback_to_chat_completions"):
                    fallbacks += 1
        except Exception:
            continue
    return {
        "api_calls_seen_today": calls,
        "image_calls_seen_today": images,
        "input_tokens_seen_today": inp,
        "output_tokens_seen_today": out,
        "api_errors_seen_today": errors,
        "chat_fallbacks_seen_today": fallbacks,
        "note": "adapter telemetry snapshot; not exact per-run before run_id tagging",
    }


def _tasks_from_specs(specs: str) -> List[str]:
    tasks: List[str] = []
    for spec in specs.split():
        if ":" not in spec:
            continue
        scenario, num = spec.split(":", 1)
        tasks.append(f"{scenario}{num}")
    return tasks or list(DEFAULT_TASKS)


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = [_float(r.get(key)) for r in rows]
    return sum(vals) / max(1, len(vals))


def _is_better(candidate: Dict[str, Any], incumbent: Dict[str, Any]) -> bool:
    cand = (
        _float(candidate.get("joint_success")),
        _float(candidate.get("result_success")),
        _float(candidate.get("micro_tool_accuracy")),
        -_float(candidate.get("avg_tool_calls"), 999.0),
    )
    inc = (
        _float(incumbent.get("joint_success")),
        _float(incumbent.get("result_success")),
        _float(incumbent.get("micro_tool_accuracy")),
        -_float(incumbent.get("avg_tool_calls"), 999.0),
    )
    return cand > inc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=os.environ.get("TRACK2_OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--version", default=os.environ.get("TRACK2_RUN_VERSION", "V6_1_3_gpt55_guarded_endpoint"))
    parser.add_argument("--gate-specs", default=os.environ.get("TRACK2_GATE_SPECS", "retail:9 kitchen:2 restaurant:4 order:1"))
    parser.add_argument("--update-best", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_model = f"{args.model}-{args.version}-{args.run_id}"
    tasks = _tasks_from_specs(args.gate_specs)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        eval_path = EGO_ROOT / "eval_result" / output_model / f"{task}_easy_eval.json"
        metrics = _metrics_from_eval(_load_json(eval_path))
        rows.append({
            "version": args.version,
            "output_model": output_model,
            "task": task,
            "eval_path": str(eval_path),
            **metrics,
        })

    analysis_path = CODEX_ROOT / "analysis" / f"gpt55_gate_matrix_{args.run_id}.csv"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    current = {
        "version": args.version,
        "joint_success": _mean(rows, "joint_success"),
        "tool_success": _mean(rows, "tool_success"),
        "result_success": _mean(rows, "result_success"),
        "micro_tool_accuracy": _mean(rows, "micro_tool_accuracy"),
        "avg_rounds": _mean(rows, "avg_rounds"),
        "avg_tool_calls": _mean(rows, "avg_tool_calls"),
        "run_id": args.run_id,
        "model": args.model,
        "external_api_used": True,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    usage = _api_usage_for_run(args.run_id)

    state_path = CODEX_ROOT / "state" / "best_track2_api_version.json"
    previous = _load_json(state_path) if state_path.exists() else {}
    best_updated = False
    if args.update_best and (not isinstance(previous, dict) or _is_better(current, previous)):
        state_path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        best_updated = True

    latest_path = CODEX_ROOT / "state" / "latest_gpt55_endpoint_gate.json"
    latest_path.write_text(json.dumps({
        **current,
        "base_url": os.environ.get("TRACK2_OPENAI_BASE_URL") or os.environ.get("SERVICE_MODEL_API_BASE") or "",
        "key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "key_logged": False,
        "report": f"reports/02_gpt55_gate_summary_{args.run_id}.md",
        "best_updated": best_updated,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = CODEX_ROOT / "reports" / f"02_gpt55_gate_summary_{args.run_id}.md"
    lines = [
        "# GPT-5.5 Gate Summary",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- run_id: {args.run_id}",
        f"- model: {args.model}",
        f"- version: {args.version}",
        "- service_agent: GPT-5.5 commercial API",
        "- key_logged: no",
        "- final_auto_submitted: no",
        f"- matrix_csv: {analysis_path}",
        f"- current_joint_success: {current['joint_success']:.3f}",
        f"- current_result_success: {current['result_success']:.3f}",
        f"- current_tool_success: {current['tool_success']:.3f}",
        f"- current_micro_tool_accuracy: {current['micro_tool_accuracy']:.3f}",
        f"- current_avg_tool_calls: {current['avg_tool_calls']:.2f}",
        f"- best_state_updated: {'yes' if best_updated else 'no'}",
        "",
        "## Task Scores",
        "",
        "| task | joint | result | tool | micro | avg_tool_calls | tool_matches | gt_calls | interaction_calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['joint_success']:.3f} | {row['result_success']:.3f} | "
            f"{row['tool_success']:.3f} | {row['micro_tool_accuracy']:.3f} | "
            f"{row['avg_tool_calls']:.2f} | {row['tool_matches']:.0f} | "
            f"{row['gt_tool_calls']:.0f} | {row['interaction_tool_calls']:.0f} |"
        )
    lines += [
        "",
        "## API Usage Snapshot",
        "",
        "```json",
        json.dumps(usage, ensure_ascii=False, indent=2),
        "```",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    heartbeat = CODEX_ROOT / "reports" / f"gpt55_heartbeat_{time.strftime('%Y%m%d_%H%M%S')}.md"
    heartbeat.write_text(
        "\n".join([
            "# GPT-5.5 Heartbeat",
            "",
            f"- generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
            f"- run_id: {args.run_id}",
            "- current_stage: gate_collected",
            f"- current_version: {args.version}",
            f"- current_success_rate: {current['joint_success']:.3f}",
            f"- best_state_updated: {'yes' if best_updated else 'no'}",
            "- needs_human_attention: no",
        ]) + "\n",
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
