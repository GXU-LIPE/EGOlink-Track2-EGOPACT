#!/usr/bin/env python3
"""Finalize V19 reporting after smoke10 failed.

This script is intentionally report-only: it does not run evaluation, does not
touch final artifacts, and does not update protected best state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODEX_ROOT = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
RUN_ID = "v19_case_reuse_20260620_1546"
RUN_DIR = CODEX_ROOT / "runs" / "V19_gt100_case_reuse_val41" / RUN_ID
REPORT_DIR = CODEX_ROOT / "reports"
STATE_DIR = CODEX_ROOT / "state"
SUMMARY_PATH = RUN_DIR / "smoke_eval_summary.json"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def flatten_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    rows = block.get("rows")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    if isinstance(block, list):
        return [r for r in block if isinstance(r, dict)]
    return []


def aggregate(block: dict[str, Any]) -> dict[str, Any]:
    rows = flatten_rows(block)
    totals = {
        "valid": 0,
        "joint_successes": 0.0,
        "result_successes": 0.0,
        "tool_successes": 0.0,
        "matched_tools": 0,
        "gt_tools": 0,
        "interaction_calls": 0,
    }
    per_task: list[dict[str, Any]] = []
    for row in rows:
        valid = int(row.get("valid") or 0)
        totals["valid"] += valid
        totals["joint_successes"] += float(row.get("joint") or 0.0) * valid
        totals["result_successes"] += float(row.get("result") or 0.0) * valid
        totals["tool_successes"] += float(row.get("tool") or 0.0) * valid
        totals["matched_tools"] += int(row.get("correct_calls") or 0)
        totals["gt_tools"] += int(row.get("gt_calls") or 0)
        totals["interaction_calls"] += int(row.get("interaction_calls") or 0)
        for detail in row.get("detailed_results") or []:
            if not isinstance(detail, dict):
                continue
            per_task.append(
                {
                    "spec": row.get("spec"),
                    "task_id": detail.get("task_id"),
                    "joint": bool(detail.get("joint_success")),
                    "result": bool((detail.get("result_based") or {}).get("success")),
                    "tool": bool((detail.get("tool_based") or {}).get("success")),
                    "matches": (detail.get("tool_based") or {}).get("matches"),
                    "gt_calls": (detail.get("tool_based") or {}).get("total_gt_calls"),
                    "interaction_calls": (detail.get("tool_based") or {}).get("total_interaction_calls"),
                }
            )
    valid = totals["valid"] or 1
    gt_tools = totals["gt_tools"] or 1
    return {
        "valid": totals["valid"],
        "joint": totals["joint_successes"] / valid,
        "result": totals["result_successes"] / valid,
        "tool": totals["tool_successes"] / valid,
        "micro": totals["matched_tools"] / gt_tools,
        "matched_tools": totals["matched_tools"],
        "gt_tools": totals["gt_tools"],
        "interaction_calls": totals["interaction_calls"],
        "per_task": per_task,
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def metrics_table(v14: dict[str, Any], v19: dict[str, Any]) -> str:
    rows = [
        ("valid", v14["valid"], v19["valid"], v19["valid"] - v14["valid"]),
        ("joint", fmt_pct(v14["joint"]), fmt_pct(v19["joint"]), fmt_pct(v19["joint"] - v14["joint"])),
        ("result", fmt_pct(v14["result"]), fmt_pct(v19["result"]), fmt_pct(v19["result"] - v14["result"])),
        ("tool", fmt_pct(v14["tool"]), fmt_pct(v19["tool"]), fmt_pct(v19["tool"] - v14["tool"])),
        ("micro", fmt_pct(v14["micro"]), fmt_pct(v19["micro"]), fmt_pct(v19["micro"] - v14["micro"])),
        ("matched_tools/gt_tools", f"{v14['matched_tools']}/{v14['gt_tools']}", f"{v19['matched_tools']}/{v19['gt_tools']}", f"{v19['matched_tools'] - v14['matched_tools']}"),
        ("interaction_calls", v14["interaction_calls"], v19["interaction_calls"], v19["interaction_calls"] - v14["interaction_calls"]),
    ]
    out = ["| metric | V14 candidate smoke | V19 case reuse smoke | delta |", "|---|---:|---:|---:|"]
    for name, a, b, d in rows:
        out.append(f"| {name} | {a} | {b} | {d} |")
    return "\n".join(out)


def changed_tasks(v14: dict[str, Any], v19: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    v14_map = {(t["spec"], t["task_id"]): t for t in v14["per_task"]}
    gains = []
    losses = []
    for t in v19["per_task"]:
        key = (t["spec"], t["task_id"])
        base = v14_map.get(key)
        if not base:
            continue
        if not base["joint"] and t["joint"]:
            gains.append({"spec": t["spec"], "task_id": t["task_id"]})
        if base["joint"] and not t["joint"]:
            losses.append({"spec": t["spec"], "task_id": t["task_id"]})
    return gains, losses


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = load_json(SUMMARY_PATH, {})
    v14 = aggregate(summary.get("v14") or summary.get("V14_candidate_smoke") or {})
    v19 = aggregate(summary.get("v19") or summary.get("V19_smoke") or {})
    gains, losses = changed_tasks(v14, v19)

    val41_report = REPORT_DIR / f"V19_VAL41_CASE_REUSE_RESULT_{RUN_ID}.md"
    diff_report = REPORT_DIR / f"V19_V14_DIFF_ANALYSIS_{RUN_ID}.md"
    next_report = REPORT_DIR / f"V19_NEXT_DECISION_{RUN_ID}.md"
    state_path = STATE_DIR / "latest_v19_case_reuse.json"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    val41_report.write_text(
        "\n".join(
            [
                f"# V19 Val41 Case Reuse Result {RUN_ID}",
                "",
                "Status: SKIPPED.",
                "",
                "Full frozen val41 was not run by design because smoke10 did not pass. This preserves the V19 experiment boundary: smoke failure stops expansion, and val41 GT is not used to tune retrieval, slot rewrite, or candidate scoring.",
                "",
                "## Smoke10 Gate",
                "",
                metrics_table(v14, v19),
                "",
                "## Decision",
                "",
                "- smoke_passed: false",
                "- full_val41_run: false",
                "- final_run: false",
                "- protected_best_updated: false",
                "- V10 protected zip overwritten: false",
                "- automatic submission: false",
                "",
                "## Reason",
                "",
                "V19 tied the V14 candidate smoke exactly on joint, result, tool, and micro. The non-oracle scorer therefore did not demonstrate enough value to justify running full val41.",
                "",
                "Observed failure mode: case-reuse candidates were generally rejected or down-scored because slot rewriting produced uncertain or forbidden copied slots, so the selector fell back to V14 baseline candidates.",
                "",
                "## Artifacts",
                "",
                f"- smoke summary: `{SUMMARY_PATH}`",
                f"- smoke report: `{REPORT_DIR / ('V19_SMOKE10_CASE_REUSE_' + RUN_ID + '.md')}`",
                f"- case library audit: `{REPORT_DIR / 'V19_CASE_LIBRARY_AUDIT_v19_case_library_20260620_1545.md'}`",
                "",
                f"Generated at: {now}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    diff_report.write_text(
        "\n".join(
            [
                f"# V19 vs V14 Diff Analysis {RUN_ID}",
                "",
                "Scope: smoke10 only. Full val41 diff is unavailable because full val41 was intentionally skipped after smoke failure.",
                "",
                metrics_table(v14, v19),
                "",
                "## Joint Task Changes",
                "",
                f"- V14 -> V19 new joint successes: {len(gains)}",
                f"- V14 -> V19 joint regressions: {len(losses)}",
                "",
                "### New Joint Successes",
                "",
                json.dumps(gains, indent=2, ensure_ascii=False),
                "",
                "### Joint Regressions",
                "",
                json.dumps(losses, indent=2, ensure_ascii=False),
                "",
                "## Interpretation",
                "",
                "The smoke comparison shows no measurable improvement over V14. The case library itself is healthy, but the current non-oracle slot rewrite and scorer are too conservative: the chosen V19 trajectory is effectively the V14 fallback on this smoke set.",
                "",
                "Recommended next debugging step is a one-sample closed-loop trace before any broader run: user utterance -> visual slot -> canonical entity -> retrieved case -> selected skeleton -> rewritten tool chain -> GT diff.",
                "",
                "## Safety Notes",
                "",
                "- No final evaluation was run.",
                "- No final hidden metadata was read.",
                "- V18 val41 oracle policy was not used for V19 selection.",
                "- V10 protected zip was not modified.",
                "",
                f"Generated at: {now}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    next_report.write_text(
        "\n".join(
            [
                f"# V19 Next Decision {RUN_ID}",
                "",
                "- smoke_passed: false",
                "- run_full_val41: false",
                "- run_final: false",
                "- protected_best_updated: false",
                "- v10_zip_overwritten: false",
                "",
                "Smoke10 did not improve over V14 candidate selection, so full val41 was not run.",
                "",
                "## Evidence",
                "",
                metrics_table(v14, v19),
                "",
                "## Follow-up",
                "",
                "Do not continue broad GT100 runs until the minimal case-reuse chain works on one clean sample. The next useful debug should print: user utterance -> visual slot -> canonical entity -> selected skeleton -> compiled tool chain -> GT diff.",
                "",
                f"- skipped val41 report: `{val41_report}`",
                f"- V14 diff report: `{diff_report}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    state = load_json(state_path, {})
    state.update(
        {
            "run_id": RUN_ID,
            "smoke_passed": False,
            "full_run": False,
            "final_run": False,
            "protected_best_updated": False,
            "v10_zip_overwritten": False,
            "smoke_summary": str(SUMMARY_PATH),
            "smoke_metrics": {"v14": {k: v14[k] for k in ("valid", "joint", "result", "tool", "micro", "matched_tools", "gt_tools", "interaction_calls")}, "v19": {k: v19[k] for k in ("valid", "joint", "result", "tool", "micro", "matched_tools", "gt_tools", "interaction_calls")}},
            "smoke_report": str(REPORT_DIR / f"V19_SMOKE10_CASE_REUSE_{RUN_ID}.md"),
            "val41_result_report": str(val41_report),
            "v14_diff_report": str(diff_report),
            "decision_report": str(next_report),
            "completion_status": "stopped_after_failed_smoke10",
            "updated_at_utc": now,
        }
    )
    dump_json(state_path, state)

    print(json.dumps({"ok": True, "reports": [str(val41_report), str(diff_report), str(next_report)], "state": str(state_path)}, indent=2))


if __name__ == "__main__":
    main()
