#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write V9_4_5 medium decision report on the remote codex workspace."""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


CODEX = Path("/home/data-gxu/acm/egolink2026-main/code/track2/codex")
VERSION = "V9_4_5_visual_retrieval_fix"
RUN_ID = "V9_4_5_visual_retrieval_fix_validation_A_medium_20260618_1014"
RUN_DIR = CODEX / "runs" / VERSION / RUN_ID


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def iter_jsonl(root: Path):
    if not root.exists():
        return
    for path in sorted(root.glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path.name, json.loads(line)
                except Exception:
                    continue
        except Exception:
            continue


def count_events():
    events = Counter()
    modules = Counter()
    scenarios = Counter()
    blocked = Counter()
    for fname, rec in iter_jsonl(RUN_DIR / "v8_events"):
        event = rec.get("event") or rec.get("event_name") or ""
        module = rec.get("module") or rec.get("source") or ""
        scenario = rec.get("scenario") or ""
        events[event] += 1
        modules[module] += 1
        scenarios[scenario] += 1
        if rec.get("whether_blocked") or rec.get("blocked_count"):
            blocked[event] += 1
    return events, modules, scenarios, blocked


def log_tool_counts():
    out = []
    for path in sorted((RUN_DIR / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        calls = len(re.findall(r"\[Tool Execution\] Calling:", text))
        summaries = re.findall(r"Task\s+(\d+):.*?(\d+)\s+tool calls", text)
        max_task_calls = max([int(x[1]) for x in summaries] or [0])
        out.append((path.name, calls, max_task_calls, path.stat().st_size))
    return out


def fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    eval_result = load_json(RUN_DIR / "eval_summary.json", {"summary": {}, "rows": []})
    summary = eval_result.get("summary", {})
    rows = eval_result.get("rows", [])
    hygiene = load_json(CODEX / "runs" / "V8_tmp_scenarios" / RUN_ID / "split_hygiene.json", {})
    events, modules, scenarios, blocked = count_events()
    tool_counts = log_tool_counts()
    total_calls = sum(x[1] for x in tool_counts)
    worst_logs = sorted(tool_counts, key=lambda x: x[1], reverse=True)[:8]

    prev = {
        "version": "V9_4_memory_retrieval",
        "joint": 0.0488,
        "result": 0.0976,
        "tool": 0.0488,
        "micro": 0.1795,
    }

    lines = [
        f"# V9_4_5 Visual Retrieval Fix A_medium {ts}",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- version: `{VERSION}`",
        "- final_submission: not submitted",
        "- protected_best_updated: no",
        "- split hygiene: `retail9::48` skipped as out-of-range; 41 valid tasks evaluated",
        "",
        "## Summary",
        "",
        f"- valid: {summary.get('valid', 0)}",
        f"- joint: {fmt_pct(summary.get('joint', 0))} vs V9_4 {fmt_pct(prev['joint'])}",
        f"- result: {fmt_pct(summary.get('result', 0))} vs V9_4 {fmt_pct(prev['result'])}",
        f"- tool: {fmt_pct(summary.get('tool', 0))} vs V9_4 {fmt_pct(prev['tool'])}",
        f"- micro: {fmt_pct(summary.get('micro', 0))} vs V9_4 {fmt_pct(prev['micro'])}",
        f"- tool_call_match_counts: {summary.get('correct_calls', 0)}/{summary.get('gt_calls', 0)} gt, interaction_calls={summary.get('interaction_calls', 0)}",
        "",
        "## Interpretation",
        "",
        "- V9_4_5 is a real A_medium improvement over V9_4, especially joint/tool success.",
        "- It still misses the requested micro >30% target, so it is not final-ready and must not replace the protected V6 best.",
        "- Main remaining failures are order process coverage and retail broad-scan over-expansion.",
        "- Kitchen2 in this split is solved, but broader kitchen still has high interaction count and partial coverage only.",
        "",
        "## Per File Metrics",
        "",
        "| scenario | valid | joint | result | tool | micro | calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('scenario')}{r.get('number')} | {r.get('valid', 0)} | "
            f"{fmt_pct(r.get('joint', 0))} | {fmt_pct(r.get('result', 0))} | "
            f"{fmt_pct(r.get('tool', 0))} | {fmt_pct(r.get('micro', 0))} | "
            f"{r.get('correct_calls', 0)}/{r.get('gt_calls', 0)} |"
        )

    lines += [
        "",
        "## Telemetry",
        "",
        f"- total V9 event records: {sum(events.values())}",
        f"- telemetry lines matching V9 visual/retail/rerank family: {sum(v for k, v in events.items() if 'visual' in k or 'retail' in k or 'multicandidate' in k or 'deepseek' in k)}",
        "",
        "Top events:",
    ]
    for name, count in events.most_common(15):
        lines.append(f"- `{name or 'unknown'}`: {count}")
    lines += ["", "Top modules:"]
    for name, count in modules.most_common(15):
        lines.append(f"- `{name or 'unknown'}`: {count}")
    lines += ["", "Blocked/warning events:"]
    if blocked:
        for name, count in blocked.most_common(15):
            lines.append(f"- `{name or 'unknown'}`: {count}")
    else:
        lines.append("- none detected")

    lines += [
        "",
        "## Tool-Call Cost Hotspots",
        "",
        f"- total tool-call log count estimate: {total_calls}",
        "",
        "| log | tool calls | max per-task calls | bytes |",
        "|---|---:|---:|---:|",
    ]
    for name, calls, max_task, size in worst_logs:
        lines.append(f"| {name} | {calls} | {max_task} | {size} |")

    skipped = hygiene.get("skipped_invalid_indices") or []
    lines += [
        "",
        "## Split Hygiene",
        "",
        f"- planned_task_count_after_filter: {hygiene.get('planned_task_count', 0)}",
        f"- skipped_invalid_indices: {len(skipped)}",
    ]
    for item in skipped[:10]:
        lines.append(
            f"- skipped `{item.get('uid')}` idx={item.get('idx')} available_count={item.get('available_count')} reason={item.get('reason')}"
        )

    lines += [
        "",
        "## Decision",
        "",
        "- Continue to V9_5 with targeted rerank/crosscheck.",
        "- Do not run final.",
        "- Do not update `state/best_track2_api_version.json`.",
        "- V9_5 should specifically reduce retail broad scans and repair order process coverage.",
    ]

    out = CODEX / "reports" / f"V9_4_5_VISUAL_RETRIEVAL_FIX_{ts}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "summary": summary, "events": events.most_common(10)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
