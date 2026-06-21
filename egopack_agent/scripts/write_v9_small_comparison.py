#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write V9 validation_A_small comparison/status reports.

This script is intentionally report-only: it does not submit final results and
does not update protected best.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")


RUN_REPORTS = {
    "V6_1_3_gpt55_guarded_endpoint": "reports/V8_VALIDATION_A_SMALL_v613_valA_small_20260617_quick.md",
    "V8_0_v6_stable_reval": "reports/V8_VALIDATION_A_SMALL_v8_0_valA_small_20260617_continue.md",
    "V8_2_kitchen_helper": "reports/V8_VALIDATION_A_SMALL_v8_2_valA_small_20260617_continue.md",
    "V9_2_scoring_prompt_soft_guard": "reports/V8_VALIDATION_A_SMALL_V9_2_scoring_prompt_soft_guard_validation_A_small_20260618_010142.md",
    "V9_4_memory_retrieval": "reports/V8_VALIDATION_A_SMALL_V9_4_memory_retrieval_validation_A_small_20260618_014932.md",
}


def parse_report(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8")
    out = {"path": str(path)}
    for key in ["valid", "joint", "result", "tool", "micro", "avg_task_accuracy"]:
        m = re.search(rf"- {key}: ([0-9.]+)", txt)
        out[key] = float(m.group(1)) if m else 0.0
    m = re.search(r"tool_call_match_counts: ([0-9]+)/([0-9]+) gt, interaction_calls=([0-9]+)", txt)
    if m:
        out["correct_calls"] = int(m.group(1))
        out["gt_calls"] = int(m.group(2))
        out["interaction_calls"] = int(m.group(3))
    else:
        out["correct_calls"] = out["gt_calls"] = out["interaction_calls"] = 0
    rows = []
    in_rows = False
    for line in txt.splitlines():
        if line.startswith("| scenario |"):
            in_rows = True
            continue
        if in_rows:
            if not line.startswith("|") or line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 8:
                rows.append(
                    {
                        "scenario": parts[0],
                        "n": parts[1],
                        "valid": parts[2],
                        "joint": parts[3],
                        "result": parts[4],
                        "tool": parts[5],
                        "micro": parts[6],
                        "calls": parts[7],
                    }
                )
    out["rows"] = rows
    return out


def load_event_summary(run_version: str, run_id: str) -> dict:
    import collections

    base = CODEX / "runs" / run_version / run_id / "wrapper_events"
    events = collections.Counter()
    policy = collections.Counter()
    if base.exists():
        for path in base.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                events[rec.get("event", "?")] += 1
                pol = rec.get("v9_policy") or {}
                if pol:
                    policy[pol.get("level", "?")] += 1
                    for warning in pol.get("soft_warnings") or []:
                        policy[f"warning:{warning}"] += 1
                    for signal in pol.get("rerank_signals") or []:
                        policy[f"signal:{signal}"] += 1
    return {
        "events": dict(events.most_common(20)),
        "policy": dict(policy.most_common(30)),
    }


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    metrics = {name: parse_report(CODEX / rel) for name, rel in RUN_REPORTS.items()}
    v94_run_id = "V9_4_memory_retrieval_validation_A_small_20260618_014932"
    v94_events = load_event_summary("V9_4_memory_retrieval", v94_run_id)

    report = CODEX / "reports" / f"V9_VALIDATION_A_SMALL_COMPARISON_{ts}.md"
    lines = [
        f"# V9 Validation A Small Comparison {ts}",
        "",
        "- split: `validation_A_small` frozen 20-task split",
        "- final_submission: not submitted",
        "- protected_best_unchanged: `V6_1_3_gpt55_guarded_endpoint`",
        "- current_candidate: `V9_4_memory_retrieval`",
        "- note: DeepSeek crosscheck not used in this comparison.",
        "",
        "## Summary Table",
        "",
        "| version | joint | result | tool | micro | avg_task_accuracy | matched_tools / gt_tools | interaction_calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m['joint']:.4f} | {m['result']:.4f} | {m['tool']:.4f} | "
            f"{m['micro']:.4f} | {m['avg_task_accuracy']:.4f} | {m['correct_calls']}/{m['gt_calls']} | {m['interaction_calls']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- V9_4 is the first V9 variant to clearly beat the protected V6_1_3 baseline on this frozen split.",
        "- V9_4 improves joint 0.05 -> 0.10, result 0.10 -> 0.15, tool 0.05 -> 0.10, and micro 0.3059 -> 0.3529.",
        "- Protected best is not updated because validation_A_medium and validation_B_holdout have not been run.",
        "- Next action: run `V9_4_memory_retrieval` on validation_A_medium before considering validation_B.",
        "",
        "## V9_4 Per-Scenario Metrics",
        "",
        "| scenario | n | joint | result | tool | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["V9_4_memory_retrieval"]["rows"]:
        lines.append(
            f"| {row['scenario']} | {row['n']} | {row['joint']} | {row['result']} | {row['tool']} | {row['micro']} | {row['calls']} |"
        )
    lines += [
        "",
        "## Top Failure Classes",
        "",
        "- Visual grounding gaps remain the largest failure class: retail1, restaurant1, and parts of restaurant3 still ask for item names or infer visually pointed items late.",
        "- Broad scan is reduced but not eliminated: V9_4 retail2 shrank compared with V9_2, while retail3 still made large catalog passes.",
        "- Order aggregate 0.0 loop was fixed after V9_2 by adding lowercase restaurant aliases for OrderDB and enforcing order aggregate `product_name` entries.",
        "- Order process coverage remains weak on visual pointing tasks; `order1` still scored 0/13 matched tools despite fixed aggregate execution.",
        "- Kitchen improved on result/micro but still over-adds tied recipes and uses several candidate compute calls.",
        "",
        "## V9_4 Guard/Policy Telemetry",
        "",
        "```json",
        json.dumps(v94_events, ensure_ascii=False, indent=2),
        "```",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    candidate = {
        "version": "V9_4_memory_retrieval",
        "run_id": v94_run_id,
        "split": "validation_A_small",
        "metrics": metrics["V9_4_memory_retrieval"],
        "comparison_report": str(report),
        "protected_best_updated": False,
        "reason": "beats V6_1_3 on validation_A_small but validation_A_medium/B not yet run",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (CODEX / "state" / "v9_candidate_version.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = CODEX / "README_STATUS.md"
    with readme.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## V9_VALIDATION_A_SMALL_COMPARISON {ts}\n\n"
            f"- Report: `{report}`\n"
            "- V9_4_memory_retrieval: joint 0.1000, result 0.1500, tool 0.1000, micro 0.3529, avg_task_accuracy 0.2400.\n"
            "- Compared with V6_1_3 validation_A_small: joint 0.0500, result 0.1000, tool 0.0500, micro 0.3059.\n"
            "- Candidate recorded in `state/v9_candidate_version.json`; protected best not updated.\n"
            "- Next: run validation_A_medium for V9_4 before validation_B/final.\n"
        )

    print(json.dumps({"report": str(report), "candidate_state": str(CODEX / "state" / "v9_candidate_version.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
